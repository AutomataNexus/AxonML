// Fused Scaled Dot-Product Attention CUDA Kernels
//
// Computes attention = softmax(Q @ K^T * scale + mask) @ V per (batch, head, query_row)
// without materializing the full N*N attention matrix in global memory.
//
// Each thread handles one (batch, head, query_row) triple and computes the full
// attention output for that row. This is memory-efficient for moderate sequence
// lengths (up to ~2048) and avoids the O(N^2) global memory of standard attention.
//
// For true Flash Attention with tiling and online softmax (needed for very long
// sequences), a more complex kernel with shared memory tiling would be required.
//
// Author: Andrew Jewell Sr - AutomataNexus
// Updated: March 18, 2026

#include <float.h>

// Helper: atomicAdd for float (available natively on sm_20+)
// We use it for grad_K and grad_V accumulation across query rows.

// Fused attention forward: one thread per (batch, head, query_row).
// For each query row i:
//   1. Compute scores[j] = sum_d(Q[i,d] * K[j,d]) * scale   for j in [0, seq_len)
//   2. Apply causal mask: if is_causal && j > i, scores[j] = -inf
//   3. Softmax over scores
//   4. Output[i,d] = sum_j(softmax[j] * V[j,d])
//
// This avoids materializing the full [batch, heads, seq, seq] attention matrix.
//
// Grid: total_rows = batch_size * num_heads * tgt_len
// Each thread processes one query row.
//
// Limitation: seq_len must be <= 2048 (local array for scores).
// For longer sequences, use the tiled Flash Attention CPU implementation.
extern "C" __global__ void fused_attention_fwd_f32(
    const float* __restrict__ Q,     // [B, H, Tq, D]
    const float* __restrict__ K,     // [B, H, Tk, D]
    const float* __restrict__ V,     // [B, H, Tk, D]
    float* __restrict__ O,           // [B, H, Tq, D]
    float scale,
    unsigned int batch_size,
    unsigned int num_heads,
    unsigned int tgt_len,            // Tq (query sequence length)
    unsigned int src_len,            // Tk (key/value sequence length)
    unsigned int head_dim,           // D
    unsigned int is_causal           // 1 = apply causal mask, 0 = no mask
) {
    unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_rows = batch_size * num_heads * tgt_len;
    if (row_idx >= total_rows) return;

    // Decode (batch, head, query_pos) from linear index
    unsigned int b = row_idx / (num_heads * tgt_len);
    unsigned int rem = row_idx % (num_heads * tgt_len);
    unsigned int h = rem / tgt_len;
    unsigned int i = rem % tgt_len;  // query position

    // Base pointers for this (batch, head)
    unsigned int bh_offset = (b * num_heads + h) * tgt_len * head_dim;
    unsigned int bh_offset_kv = (b * num_heads + h) * src_len * head_dim;
    const float* q_row = Q + bh_offset + i * head_dim;

    // Pass 1: Compute scores and find max for numerical stability
    float max_score = -FLT_MAX;

    // We use registers + loop rather than local array to avoid stack spills.
    // Pass 1: find max score
    for (unsigned int j = 0; j < src_len; j++) {
        if (is_causal && j > i) break;  // All remaining are -inf

        const float* k_row = K + bh_offset_kv + j * head_dim;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // Pass 2: Compute exp(score - max) and sum for softmax normalization
    float sum_exp = 0.0f;

    // We also accumulate the weighted V in this pass to avoid a 3rd pass.
    // Output accumulator (initialized to zero)
    // We need head_dim accumulators. For head_dim <= 256 this fits in registers.
    // For larger head_dim, the compiler will spill to local memory.
    float* o_row = O + bh_offset + i * head_dim;

    // Zero output
    for (unsigned int d = 0; d < head_dim; d++) {
        o_row[d] = 0.0f;
    }

    // Pass 2+3 fused: compute softmax weights and accumulate V
    for (unsigned int j = 0; j < src_len; j++) {
        float p;
        if (is_causal && j > i) {
            break;  // p would be 0 for all remaining
        }

        const float* k_row = K + bh_offset_kv + j * head_dim;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        p = expf(score - max_score);
        sum_exp += p;

        // Accumulate p * V[j, :]
        const float* v_row = V + bh_offset_kv + j * head_dim;
        for (unsigned int d = 0; d < head_dim; d++) {
            o_row[d] += p * v_row[d];
        }
    }

    // Normalize by softmax denominator
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (unsigned int d = 0; d < head_dim; d++) {
        o_row[d] *= inv_sum;
    }
}

// =============================================================================
// Fused Attention Backward Kernel (recomputation-based, memory-efficient)
// =============================================================================
//
// Computes gradients for scaled dot-product attention without storing the N*N
// attention matrix. For each query row i, we recompute the attention weights
// from Q, K, and the saved row_max/row_sum, then compute:
//
//   grad_V += attn_weights^T @ grad_output
//   grad_attn = grad_output @ V^T
//   grad_scores = attn_weights * (grad_attn - sum(grad_attn * attn_weights))
//   grad_Q += grad_scores @ K * scale
//   grad_K += grad_scores^T @ Q * scale
//
// Grid: total_rows = batch_size * num_heads * tgt_len
// Each thread processes one query row (same parallelism as forward).
//
// grad_K and grad_V use atomicAdd because multiple query rows write to the
// same key/value positions. grad_Q is written directly (one row per thread).
//
// Limitation: src_len must be <= 2048 (same as forward).
extern "C" __global__ void fused_attention_bwd_f32(
    const float* __restrict__ Q,          // [B, H, Tq, D]
    const float* __restrict__ K,          // [B, H, Tk, D]
    const float* __restrict__ V,          // [B, H, Tk, D]
    const float* __restrict__ O,          // [B, H, Tq, D]  (forward output)
    const float* __restrict__ grad_O,     // [B, H, Tq, D]  (grad w.r.t. output)
    float* __restrict__ grad_Q,           // [B, H, Tq, D]  (output, zero-initialized)
    float* __restrict__ grad_K,           // [B, H, Tk, D]  (output, zero-initialized)
    float* __restrict__ grad_V,           // [B, H, Tk, D]  (output, zero-initialized)
    float scale,
    unsigned int batch_size,
    unsigned int num_heads,
    unsigned int tgt_len,                 // Tq
    unsigned int src_len,                 // Tk
    unsigned int head_dim,                // D
    unsigned int is_causal                // 1 = causal mask, 0 = no mask
) {
    unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_rows = batch_size * num_heads * tgt_len;
    if (row_idx >= total_rows) return;

    // Decode (batch, head, query_pos) from linear index
    unsigned int b = row_idx / (num_heads * tgt_len);
    unsigned int rem = row_idx % (num_heads * tgt_len);
    unsigned int h = rem / tgt_len;
    unsigned int i = rem % tgt_len;  // query position

    // Base pointers for this (batch, head)
    unsigned int bh_offset_q = (b * num_heads + h) * tgt_len * head_dim;
    unsigned int bh_offset_kv = (b * num_heads + h) * src_len * head_dim;

    const float* q_row = Q + bh_offset_q + i * head_dim;
    const float* o_row = O + bh_offset_q + i * head_dim;
    const float* go_row = grad_O + bh_offset_q + i * head_dim;
    float* gq_row = grad_Q + bh_offset_q + i * head_dim;

    // Effective src_len for this row (respecting causal mask)
    unsigned int eff_src = is_causal ? (i + 1 < src_len ? i + 1 : src_len) : src_len;

    // ---- Pass 1: Recompute attention scores and softmax ----
    // Find max score for numerical stability
    float max_score = -FLT_MAX;
    for (unsigned int j = 0; j < eff_src; j++) {
        const float* k_row = K + bh_offset_kv + j * head_dim;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // Compute softmax weights and their sum
    float sum_exp = 0.0f;
    for (unsigned int j = 0; j < eff_src; j++) {
        const float* k_row = K + bh_offset_kv + j * head_dim;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        sum_exp += expf(score - max_score);
    }
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    // ---- Pass 2: Compute D_i = sum_d(grad_O[i,d] * O[i,d]) ----
    // This is used in the softmax backward: grad_scores = P * (grad_attn - D_i)
    float D_i = 0.0f;
    for (unsigned int d = 0; d < head_dim; d++) {
        D_i += go_row[d] * o_row[d];
    }

    // ---- Pass 3: For each key position j, recompute P[i,j] and accumulate gradients ----
    for (unsigned int j = 0; j < eff_src; j++) {
        const float* k_row = K + bh_offset_kv + j * head_dim;
        const float* v_row = V + bh_offset_kv + j * head_dim;
        float* gk_row = grad_K + bh_offset_kv + j * head_dim;
        float* gv_row = grad_V + bh_offset_kv + j * head_dim;

        // Recompute attention weight P[i,j]
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        float p_ij = expf(score - max_score) * inv_sum;

        // grad_attn[i,j] = sum_d(grad_O[i,d] * V[j,d])
        float grad_attn_ij = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            grad_attn_ij += go_row[d] * v_row[d];
        }

        // grad_score[i,j] = P[i,j] * (grad_attn[i,j] - D_i)
        float grad_score_ij = p_ij * (grad_attn_ij - D_i);

        // Accumulate gradients:
        // grad_V[j,d] += P[i,j] * grad_O[i,d]     (multiple i write to same j)
        // grad_Q[i,d] += grad_score[i,j] * K[j,d] * scale  (only this thread writes to row i)
        // grad_K[j,d] += grad_score[i,j] * Q[i,d] * scale  (multiple i write to same j)
        float scaled_gs = grad_score_ij * scale;
        for (unsigned int d = 0; d < head_dim; d++) {
            // grad_V: atomic because multiple query rows accumulate to same V row
            atomicAdd(&gv_row[d], p_ij * go_row[d]);

            // grad_Q: only this thread writes to this row, safe to accumulate directly
            gq_row[d] += scaled_gs * k_row[d];

            // grad_K: atomic because multiple query rows accumulate to same K row
            atomicAdd(&gk_row[d], scaled_gs * q_row[d]);
        }
    }
}
