// Transformer per-layer ops as CUDA kernels.
//
// These four kernels collectively replace the CPU-side ops that currently
// force a D2H boundary after every matmul in the nexus-serve decode loop:
//
//   * rms_norm_f32        — RMSNorm with per-element weight scale
//   * rope_split_halves_f32 — Rotary position embedding (LLaMA / Qwen layout
//                            where rotated pairs are (d, d + head_dim/2))
//   * swiglu_f32          — Fused SiLU(gate) * up for SwiGLU FFN
//   * relu2_gate_f32      — Fused ReLU²(gate) * up for BitNet b1.58 FFN
//
// All four are decode-step kernels (single token at a time): hidden state
// is `[hidden]`, q/k vectors are `[n_heads * head_dim]`, FFN intermediate
// is `[intermediate_size]`. Each kernel launches one or a few CTAs and
// returns immediately — the goal is to stay on the device, not to
// maximize per-kernel throughput.
//
// Compile:
//   nvcc -arch=sm_80 -ptx transformer_ops.cu -o transformer_ops.ptx
//
// Author: Andrew Jewell Sr. — AutomataNexus LLC
// ORCID: 0009-0005-2158-7060

#include <float.h>
#include <math_constants.h>
#include <stdint.h>

// ============================================================================
// RMSNorm (single token)
//
// Computes y[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
//
// Single CTA, blockDim.x threads cooperate to compute the variance via a
// per-warp shuffle reduction + cross-warp reduction in shared memory.
// ============================================================================

extern "C" __global__ void rms_norm_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    uint32_t n,
    float eps
) {
    extern __shared__ float warp_sums[];
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = (blockDim.x + 31) >> 5;

    // 1. Each thread accumulates partial sum-of-squares over its assigned
    //    indices (grid-stride within the single CTA).
    float local = 0.0f;
    for (int i = tid; i < (int)n; i += blockDim.x) {
        float v = x[i];
        local += v * v;
    }

    // 2. Warp-level reduction via __shfl_xor_sync.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_xor_sync(0xFFFFFFFF, local, offset);
    }

    // 3. Cross-warp reduction through shared memory.
    if (lane == 0) warp_sums[warp_id] = local;
    __syncthreads();

    // First warp reduces the per-warp partials.
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(0xFFFFFFFF, total, offset);
        }
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();

    // 4. Broadcast normalization scale.
    float mean_sq = warp_sums[0] / (float)n;
    float scale = rsqrtf(mean_sq + eps);

    // 5. Each thread writes its assigned outputs.
    for (int i = tid; i < (int)n; i += blockDim.x) {
        out[i] = (x[i] * scale) * weight[i];
    }
}

// ============================================================================
// Rotary position embedding (split-halves layout)
//
// LLaMA / Qwen / Mistral / Phi all use the "split halves" RoPE convention:
// for each head, dimensions [0, head_dim/2) and [head_dim/2, head_dim) are
// the two halves of a complex pair. Rotating by angle θ_d at position `pos`:
//
//   x[d]              = cos(θ) * x[d]              - sin(θ) * x[d + head_dim/2]
//   x[d + head_dim/2] = sin(θ) * x[d]              + cos(θ) * x[d + head_dim/2]
//
// where θ_d = pos * theta^(-2d / head_dim) for d ∈ [0, head_dim/2).
//
// One thread per (head, pair_index). Grid = (n_heads, head_dim/2, 1).
// ============================================================================

// `src` and `out` MAY alias for in-place rotation; each thread reads both
// x[base] and x[base+half] from src BEFORE any write to out, so aliasing
// is safe. Separate-pointer signature lets callers skip the broadcast_copy
// prep pass (single kernel launch instead of copy+rotate).
extern "C" __global__ void rope_split_halves_f32(
    const float* __restrict__ src,
    float* __restrict__ out,
    uint32_t n_heads,
    uint32_t head_dim,
    float theta,
    uint32_t pos
) {
    const uint32_t head = blockIdx.x;
    const uint32_t pair = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t half = head_dim >> 1;
    if (head >= n_heads || pair >= half) return;

    // θ_d = pos * theta^(-2*pair/head_dim)
    const float exponent = -(float)(2u * pair) / (float)head_dim;
    const float angle = (float)pos * powf(theta, exponent);
    float c, s;
    sincosf(angle, &s, &c);

    const uint32_t base = head * head_dim + pair;
    const float a = src[base];
    const float b = src[base + half];
    out[base]        = c * a - s * b;
    out[base + half] = s * a + c * b;
}

// ============================================================================
// Fused SwiGLU: out = SiLU(gate) * up
//
// SiLU(x) = x / (1 + e^{-x}) = x * σ(x). One thread per output element.
// Eliminates the gate.silu() ⊕ silu_buf.mul(up) chain (two kernel launches +
// one intermediate buffer) of the unfused path.
// ============================================================================

extern "C" __global__ void swiglu_f32(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    uint32_t n
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = gate[idx];
    const float silu_g = g / (1.0f + __expf(-g));
    out[idx] = silu_g * up[idx];
}

// ============================================================================
// SwiGLU backward: given saved gate/up and grad_out, produce (grad_gate, grad_up)
//
// y = silu(gate) * up
// dL/d(gate) = dL/d(y) * up * silu'(gate)
//            = dL/d(y) * up * σ(gate) * (1 + gate * (1 - σ(gate)))
// dL/d(up)   = dL/d(y) * silu(gate)
//
// One kernel produces both gradients, replacing the separate SiluBackward +
// MulBackward chain in the MLP backward.
// ============================================================================
extern "C" __global__ void swiglu_bwd_f32(
    float* __restrict__ grad_gate,
    float* __restrict__ grad_up,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ grad_out,
    uint32_t n
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g   = gate[idx];
    const float u   = up[idx];
    const float go  = grad_out[idx];
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float silu_g = g * sig;
    const float silu_deriv = sig * (1.0f + g * (1.0f - sig));
    grad_gate[idx] = go * u * silu_deriv;
    grad_up[idx]   = go * silu_g;
}

// ============================================================================
// BitNet b1.58 fused gate: out = ReLU(gate)² * up
//
// BitNet replaces SwiGLU's smooth gate with a hard ReLU² = max(0, x)². One
// thread per output element. Decoupled from swiglu so we can dispatch by
// model architecture without a kernel-internal branch.
// ============================================================================

extern "C" __global__ void relu2_gate_f32(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    uint32_t n
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = gate[idx];
    const float r = fmaxf(0.0f, g);
    out[idx] = (r * r) * up[idx];
}

// ============================================================================
// LayerNorm (true LayerNorm, not RMSNorm) — subtract mean, divide by stddev,
// scale by gamma, shift by beta. Used by legacy Falcon's attn_norm and the
// final `output_norm` on Falcon-7B / 40B.
//
//   mean = (1/n) Σ x[i]
//   var  = (1/n) Σ (x[i] - mean)²
//   out[i] = (x[i] - mean) * rsqrt(var + eps) * gamma[i] + beta[i]
//
// Same two-pass reduction layout as rms_norm_f32. One CTA per token —
// Falcon's decode path calls this with n == 1 and n == hidden_size.
// ============================================================================

extern "C" __global__ void layer_norm_tokenwise_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    uint32_t n,
    float eps
) {
    // warp_sums holds `mean_partial` in index 0, `var_partial` in index 1.
    extern __shared__ float warp_sums[];
    float* warp_mean = warp_sums;
    float* warp_var  = warp_sums + (blockDim.x + 31) / 32;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = (blockDim.x + 31) >> 5;

    // Pass 1 — mean.
    float sum = 0.0f;
    for (int i = tid; i < (int)n; i += blockDim.x) {
        sum += x[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane == 0) warp_mean[warp_id] = sum;
    __syncthreads();
    float mean = 0.0f;
    if (warp_id == 0) {
        mean = (lane < n_warps) ? warp_mean[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            mean += __shfl_xor_sync(0xFFFFFFFF, mean, offset);
        }
        if (lane == 0) warp_mean[0] = mean;
    }
    __syncthreads();
    mean = warp_mean[0] / (float)n;

    // Pass 2 — variance around that mean.
    float sq = 0.0f;
    for (int i = tid; i < (int)n; i += blockDim.x) {
        float d = x[i] - mean;
        sq += d * d;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_xor_sync(0xFFFFFFFF, sq, offset);
    }
    if (lane == 0) warp_var[warp_id] = sq;
    __syncthreads();
    float var = 0.0f;
    if (warp_id == 0) {
        var = (lane < n_warps) ? warp_var[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            var += __shfl_xor_sync(0xFFFFFFFF, var, offset);
        }
        if (lane == 0) warp_var[0] = var;
    }
    __syncthreads();
    var = warp_var[0] / (float)n;
    const float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < (int)n; i += blockDim.x) {
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ============================================================================
// GELU (tanh approximation) — `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))`.
// Falcon MLP uses this (`gelu_pytorch_tanh`); most other HF archs use exact
// GELU via erf — they can layer over this if needed. One thread per element.
// ============================================================================

extern "C" __global__ void gelu_tanh_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    uint32_t n
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = x[idx];
    const float k = 0.7978845608028654f; // sqrt(2 / pi)
    const float y = k * (v + 0.044715f * v * v * v);
    out[idx] = 0.5f * v * (1.0f + tanhf(y));
}

// ============================================================================
// Parallel-residual add — `x = x + attn + ffn` for Falcon's parallel
// attention+FFN block. Fuses two element-wise adds into one kernel launch
// so the decode hot path doesn't round-trip through two `tensor.add()` calls.
// ============================================================================

extern "C" __global__ void parallel_residual_add_f32(
    float* __restrict__ x,
    const float* __restrict__ attn,
    const float* __restrict__ ffn,
    uint32_t n
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] = x[idx] + attn[idx] + ffn[idx];
}

// Element-wise `dst += src * scalar`. Used by the MoE expert-accumulate
// hot path: replaces a `mul_scalar(w_e)` + `add(&...)` kernel pair with
// one launch. Eight experts × 16 layers × 2 launches saved per step =
// 256 fewer kernel launches per token on OLMoE decode.
extern "C" __global__ void scaled_add_inplace_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    uint32_t n,
    float scalar
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = dst[idx] + src[idx] * scalar;
}

// ============================================================================
// Per-head RMS_norm (Qwen3 QK-norm)
//
// Applies RMS_norm(x[h, :], weight, eps) independently for every head h.
// The SAME [head_dim] weight vector is broadcast across every head (this
// is how Qwen3 stores its attn_q_norm / attn_k_norm).
//
// Layout:
//   x       : [n_heads * head_dim]  in-place
//   weight  : [head_dim]
//
// Launch: grid = (n_heads, 1, 1), block = (32, 1, 1). One warp per head.
// head_dim must be a multiple of 32 (Qwen3: head_dim = 128, so 4 elems/lane).
// ============================================================================
// `src` and `out` MAY alias (in-place normalize); the sum-of-squares
// reduction completes before any write to out, so aliasing is safe.
extern "C" __global__ void rms_norm_heads_f32(
    const float* __restrict__ src,
    float* __restrict__ out,
    const float* __restrict__ weight,
    unsigned int head_dim,
    float eps
) {
    const unsigned int h    = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const float* src_head = src + (size_t)h * (size_t)head_dim;
    float*       out_head = out + (size_t)h * (size_t)head_dim;

    // Sum-of-squares, lane-strided over this head's head_dim elements.
    float local = 0.0f;
    for (unsigned int i = lane; i < head_dim; i += 32u) {
        float v = src_head[i];
        local += v * v;
    }

    // Warp-level reduction (head_dim <= ~8 * 32, fits in a single warp).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_xor_sync(0xFFFFFFFFu, local, off);
    }

    float inv_rms = rsqrtf(local / (float)head_dim + eps);

    for (unsigned int i = lane; i < head_dim; i += 32u) {
        out_head[i] = src_head[i] * inv_rms * weight[i];
    }
}

// ============================================================================
// Batched (prefill) kernels — process m tokens in a single launch
//
// These are the multi-token counterparts of the single-token kernels above.
// They exist so prefill (prompt encoding) can stay fully GPU-resident at m>1
// and skip the m×launch-overhead path of looping forward_one_gpu_resident.
//
// Shared convention: the outer grid dim varies over tokens (blockIdx.y or .z),
// the inner grid/block dims handle per-token work identical to the m=1 case.
// ============================================================================

// One CTA per (token), same warp-reduction structure as rms_norm_f32.
// Launch: grid = (m, 1, 1), block = (blockDim.x, 1, 1), shmem = n_warps*4.
// x, out shape: [m, n] (contiguous row-major).
extern "C" __global__ void rms_norm_batched_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    uint32_t n,
    float eps
) {
    extern __shared__ float warp_sums[];
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = (blockDim.x + 31) >> 5;
    const uint32_t row = blockIdx.x;

    const float* __restrict__ x_row   = x   + (size_t)row * (size_t)n;
    float* __restrict__       out_row = out + (size_t)row * (size_t)n;

    float local = 0.0f;
    for (int i = tid; i < (int)n; i += blockDim.x) {
        float v = x_row[i];
        local += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_xor_sync(0xFFFFFFFF, local, offset);
    }

    if (lane == 0) warp_sums[warp_id] = local;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(0xFFFFFFFF, total, offset);
        }
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();

    float mean_sq = warp_sums[0] / (float)n;
    float scale = rsqrtf(mean_sq + eps);

    for (int i = tid; i < (int)n; i += blockDim.x) {
        out_row[i] = (x_row[i] * scale) * weight[i];
    }
}

// One warp per (head, token). Same structure as rms_norm_heads_f32 but the
// (head, token) pair selects the slice inside a [m, n_heads, head_dim] tensor.
// `src` and `out` MAY alias. Launch: grid = (n_heads, m, 1), block = (32, 1, 1).
extern "C" __global__ void rms_norm_heads_batched_f32(
    const float* __restrict__ src,
    float* __restrict__ out,
    const float* __restrict__ weight,
    unsigned int n_heads,
    unsigned int head_dim,
    float eps
) {
    const unsigned int h    = blockIdx.x;
    const unsigned int t    = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    const size_t row_off = (size_t)t * (size_t)n_heads * (size_t)head_dim
                         + (size_t)h * (size_t)head_dim;
    const float* src_head = src + row_off;
    float*       out_head = out + row_off;

    float local = 0.0f;
    for (unsigned int i = lane; i < head_dim; i += 32u) {
        float v = src_head[i];
        local += v * v;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_xor_sync(0xFFFFFFFFu, local, off);
    }

    float inv_rms = rsqrtf(local / (float)head_dim + eps);

    for (unsigned int i = lane; i < head_dim; i += 32u) {
        out_head[i] = src_head[i] * inv_rms * weight[i];
    }
}

// Batched split-halves RoPE. Rotates x[t, h, :] at position (pos_start + t).
// Layout: x is [m, n_heads, head_dim] row-major, flat shape [m*n_heads*head_dim].
// Launch: grid = (n_heads, head_dim/2, m), block = (blockDim.x, 1, 1). Must
// satisfy blockDim.x >= head_dim/2 at the user-level launcher — or set
// gridDim.y = ceil_div(head_dim/2, blockDim.x) and compute the pair index
// from both dims. The launcher uses the latter (mirrors rope_split_halves_f32).
// Batched split-halves RoPE. `src`/`out` may alias.
extern "C" __global__ void rope_split_halves_batched_f32(
    const float* __restrict__ src,
    float* __restrict__ out,
    uint32_t n_heads,
    uint32_t head_dim,
    float theta,
    uint32_t pos_start
) {
    const uint32_t head = blockIdx.x;
    const uint32_t pair = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t tok  = blockIdx.z;
    const uint32_t half = head_dim >> 1;
    if (head >= n_heads || pair >= half) return;

    const uint32_t pos = pos_start + tok;
    const float exponent = -(float)(2u * pair) / (float)head_dim;
    const float angle = (float)pos * powf(theta, exponent);
    float c, s;
    sincosf(angle, &s, &c);

    const size_t row_stride = (size_t)n_heads * (size_t)head_dim;
    const size_t base = (size_t)tok * row_stride
                      + (size_t)head * (size_t)head_dim
                      + (size_t)pair;
    const float a = src[base];
    const float b = src[base + half];
    out[base]        = c * a - s * b;
    out[base + half] = s * a + c * b;
}

// Broadcast per-column bias across m rows of a [m, n] matrix.
// Launch: grid = ceil_div(m*n, 256), block = 256.
extern "C" __global__ void add_bias_batched_f32(
    float* __restrict__ out,
    const float* __restrict__ bias,
    uint32_t m,
    uint32_t n
) {
    const uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)m * (uint64_t)n;
    if (idx >= total) return;
    const uint32_t col = (uint32_t)(idx % (uint64_t)n);
    out[idx] += bias[col];
}

// ============================================================================
// Fused residual-add + RMSNorm (forward)
//
// Replaces Qwen3's `residual.add(&attn_out)` (broadcast_add kernel) followed
// immediately by `post_attention_layernorm.forward(sum)` (rms_norm_batched
// kernel) with a single CTA-per-row launch. The sum `x = a + b` is computed
// on the fly, its rms is reduced across threads, and the scaled output is
// written — all without materializing the intermediate sum tensor.
//
// out[i] = (a[i] + b[i]) * weight[i] / sqrt(mean((a[i]+b[i])²) + eps)
// sum_out[i] = a[i] + b[i]   (saved separately so backward can reconstruct rms)
//
// Launch: grid = (m, 1, 1), block = (blockDim.x, 1, 1), shmem = n_warps*4.
// ============================================================================
extern "C" __global__ void add_rmsnorm_batched_f32(
    float* __restrict__ out,
    float* __restrict__ sum_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ weight,
    uint32_t n,
    float eps
) {
    extern __shared__ float warp_sums[];
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = (blockDim.x + 31) >> 5;
    const uint32_t row = blockIdx.x;

    const float* __restrict__ a_row = a + (size_t)row * (size_t)n;
    const float* __restrict__ b_row = b + (size_t)row * (size_t)n;
    float* __restrict__ s_row       = sum_out + (size_t)row * (size_t)n;
    float* __restrict__ out_row     = out + (size_t)row * (size_t)n;

    // Pass 1: parallel sum-of-squares over (a + b). Also write the sum.
    float local_sq = 0.0f;
    for (int i = tid; i < (int)n; i += blockDim.x) {
        float s = a_row[i] + b_row[i];
        s_row[i] = s;
        local_sq += s * s;
    }
    #pragma unroll
    for (int offs = 16; offs > 0; offs >>= 1) {
        local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offs);
    }
    if (lane == 0) warp_sums[warp_id] = local_sq;
    __syncthreads();
    float total_sq = 0.0f;
    if (warp_id == 0) {
        total_sq = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offs = 16; offs > 0; offs >>= 1) {
            total_sq += __shfl_xor_sync(0xFFFFFFFF, total_sq, offs);
        }
        if (lane == 0) warp_sums[0] = total_sq;
    }
    __syncthreads();

    float mean_sq = warp_sums[0] / (float)n;
    float scale = rsqrtf(mean_sq + eps);

    // Pass 2: each thread writes its normalized + weighted output.
    for (int i = tid; i < (int)n; i += blockDim.x) {
        out_row[i] = s_row[i] * scale * weight[i];
    }
}

// ============================================================================
// Fused causal-scaled softmax (forward)
//
// Replaces the qwen3/llama sequence `scores.mul_scalar(scale).add(causal_mask)
// .softmax(-1)` with one kernel launch. Saves the mask-alloc + H2D per
// forward call, the mul_scalar kernel, and the broadcast-add kernel.
//
// scores shape: [num_rows, tk] where num_rows = B*H*Tq.
// q_pos within a (B, H) batch = row_idx % tq.
// For each (row, j): valid if j <= offset + q_pos, else mask to -inf.
// Effective input: (j > offset + q_pos) ? -INF : (scores[r,j] * scale).
// Output: softmax along last dim.
//
// Launch: grid = (num_rows, 1, 1), block = (blockDim.x, 1, 1),
// shmem = n_warps * 2 * 4 bytes (running max + exp sum).
// ============================================================================
extern "C" __global__ void softmax_causal_scaled_f32(
    float* __restrict__ out,
    const float* __restrict__ scores,
    uint32_t tq,
    uint32_t tk,
    uint32_t offset,
    float scale
) {
    extern __shared__ float smem[];
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = (blockDim.x + 31) >> 5;
    const uint32_t row = blockIdx.x;
    const uint32_t q_pos = row % tq;
    const uint32_t max_k = offset + q_pos; // inclusive upper bound on valid j

    float* warp_buf = smem; // reused for max then sum

    const float* __restrict__ row_in  = scores + (size_t)row * (size_t)tk;
    float* __restrict__       row_out = out    + (size_t)row * (size_t)tk;

    // Pass 1: parallel max over valid positions.
    float local_max = -CUDART_INF_F;
    for (int j = tid; j < (int)tk; j += blockDim.x) {
        float v = ((uint32_t)j > max_k) ? -CUDART_INF_F : (row_in[j] * scale);
        local_max = fmaxf(local_max, v);
    }
    #pragma unroll
    for (int offs = 16; offs > 0; offs >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offs));
    }
    if (lane == 0) warp_buf[warp_id] = local_max;
    __syncthreads();
    float row_max = -CUDART_INF_F;
    if (warp_id == 0) {
        row_max = (lane < n_warps) ? warp_buf[lane] : -CUDART_INF_F;
        #pragma unroll
        for (int offs = 16; offs > 0; offs >>= 1) {
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offs));
        }
        if (lane == 0) warp_buf[0] = row_max;
    }
    __syncthreads();
    row_max = warp_buf[0];
    __syncthreads();

    // Pass 2: parallel sum of exp(v - row_max) over valid positions.
    float local_sum = 0.0f;
    for (int j = tid; j < (int)tk; j += blockDim.x) {
        if ((uint32_t)j > max_k) continue;
        float v = row_in[j] * scale;
        local_sum += expf(v - row_max);
    }
    #pragma unroll
    for (int offs = 16; offs > 0; offs >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offs);
    }
    if (lane == 0) warp_buf[warp_id] = local_sum;
    __syncthreads();
    float row_sum = 0.0f;
    if (warp_id == 0) {
        row_sum = (lane < n_warps) ? warp_buf[lane] : 0.0f;
        #pragma unroll
        for (int offs = 16; offs > 0; offs >>= 1) {
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offs);
        }
        if (lane == 0) warp_buf[0] = row_sum;
    }
    __syncthreads();
    float inv_sum = (warp_buf[0] > 0.0f) ? (1.0f / warp_buf[0]) : 0.0f;

    // Pass 3: write normalized outputs; masked positions are exactly 0.
    for (int j = tid; j < (int)tk; j += blockDim.x) {
        float out_val;
        if ((uint32_t)j > max_k) {
            out_val = 0.0f;
        } else {
            float v = row_in[j] * scale;
            out_val = expf(v - row_max) * inv_sum;
        }
        row_out[j] = out_val;
    }
}

// ============================================================================
// Fused causal-scaled softmax (backward, wrt raw scores)
//
// Given forward output `p` (masked positions already 0) and upstream gradient
// `grad_out`, produces `grad_scores` such that
//   grad_scores[r, j] = scale * p[r, j] * (grad_out[r, j] - Σ_k p[r, k] * grad_out[r, k])
// No mask check needed in the write because p[r, j] is 0 for masked positions,
// which makes the whole expression 0 automatically.
//
// Launch: grid = (num_rows, 1, 1), block = (blockDim.x, 1, 1),
// shmem = n_warps * 4 bytes.
// ============================================================================
extern "C" __global__ void softmax_causal_scaled_bwd_f32(
    float* __restrict__ grad_scores,
    const float* __restrict__ p,
    const float* __restrict__ grad_out,
    uint32_t tk,
    float scale
) {
    extern __shared__ float smem[];
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = (blockDim.x + 31) >> 5;
    const uint32_t row = blockIdx.x;

    const float* __restrict__ p_row   = p        + (size_t)row * (size_t)tk;
    const float* __restrict__ g_row   = grad_out + (size_t)row * (size_t)tk;
    float* __restrict__       gs_row  = grad_scores + (size_t)row * (size_t)tk;

    // Parallel dot(p, grad_out). Masked positions contribute 0 because p=0.
    float local_dot = 0.0f;
    for (int j = tid; j < (int)tk; j += blockDim.x) {
        local_dot += p_row[j] * g_row[j];
    }
    #pragma unroll
    for (int offs = 16; offs > 0; offs >>= 1) {
        local_dot += __shfl_xor_sync(0xFFFFFFFF, local_dot, offs);
    }
    if (lane == 0) smem[warp_id] = local_dot;
    __syncthreads();
    float dot = 0.0f;
    if (warp_id == 0) {
        dot = (lane < n_warps) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offs = 16; offs > 0; offs >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, offs);
        }
        if (lane == 0) smem[0] = dot;
    }
    __syncthreads();
    dot = smem[0];

    // Write grad_scores. Chain rule: scale * p * (grad_out - dot).
    for (int j = tid; j < (int)tk; j += blockDim.x) {
        float pj = p_row[j];
        float gj = g_row[j];
        gs_row[j] = scale * pj * (gj - dot);
    }
}

// ============================================================================
// RMSNorm backward (batched, grad_input only)
//
// Forward: y_i = (x_i / rms) * w_i  with rms = sqrt(mean(x²) + eps)
// Backward wrt x:
//   grad_x_i = (w_i / rms) * grad_y_i - (x_i / (rms^3 * N)) * Σ_j(x_j * w_j * grad_y_j)
//
// Launch: grid = (m, 1, 1), block = (blockDim.x, 1, 1),
// shmem   = 2 * n_warps * 4 bytes (two parallel reductions: sum(x²) and dot(x,w,g)).
// Replaces RMSNormBackward::apply's CPU-only path (full D2H of x, w, g per
// call + O(m*n) CPU compute + H2D of grad_input). The prior CPU path was
// ~61 ms/call on Qwen3-0.6B backward; this kernel is a single launch per row.
// ============================================================================
extern "C" __global__ void rms_norm_bwd_batched_f32(
    float* __restrict__ grad_input,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    uint32_t n,
    float eps
) {
    extern __shared__ float smem[];
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = (blockDim.x + 31) >> 5;
    const uint32_t row = blockIdx.x;

    // Shared memory partitions: [warp_sum_sq | warp_dot]
    float* warp_sum_sq = smem;
    float* warp_dot    = smem + n_warps;

    const float* __restrict__ x_row  = x        + (size_t)row * (size_t)n;
    const float* __restrict__ g_row  = grad_out + (size_t)row * (size_t)n;
    float* __restrict__       gi_row = grad_input + (size_t)row * (size_t)n;

    // 1. Parallel reductions: sum_sq = Σ x² and dot = Σ x·w·g, across threads.
    float local_sum_sq = 0.0f;
    float local_dot    = 0.0f;
    for (int i = tid; i < (int)n; i += blockDim.x) {
        float xi = x_row[i];
        float wi = weight[i];
        float gi = g_row[i];
        local_sum_sq += xi * xi;
        local_dot    += xi * wi * gi;
    }

    // 2. Warp-level reduction for both.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_xor_sync(0xFFFFFFFF, local_sum_sq, offset);
        local_dot    += __shfl_xor_sync(0xFFFFFFFF, local_dot,    offset);
    }
    if (lane == 0) {
        warp_sum_sq[warp_id] = local_sum_sq;
        warp_dot[warp_id]    = local_dot;
    }
    __syncthreads();

    // 3. Cross-warp reduction in warp 0.
    float total_sum_sq = 0.0f;
    float total_dot    = 0.0f;
    if (warp_id == 0) {
        total_sum_sq = (lane < n_warps) ? warp_sum_sq[lane] : 0.0f;
        total_dot    = (lane < n_warps) ? warp_dot[lane]    : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sum_sq += __shfl_xor_sync(0xFFFFFFFF, total_sum_sq, offset);
            total_dot    += __shfl_xor_sync(0xFFFFFFFF, total_dot,    offset);
        }
        if (lane == 0) {
            warp_sum_sq[0] = total_sum_sq;
            warp_dot[0]    = total_dot;
        }
    }
    __syncthreads();

    // 4. Compute per-row constants and broadcast.
    float mean_sq  = warp_sum_sq[0] / (float)n;
    float rms_inv  = rsqrtf(mean_sq + eps);
    float rms3_inv = rms_inv * rms_inv * rms_inv;
    float dot      = warp_dot[0];
    float dot_scaled = dot * rms3_inv / (float)n;

    // 5. Each thread writes its grad_input elements.
    for (int i = tid; i < (int)n; i += blockDim.x) {
        float term1 = weight[i] * g_row[i] * rms_inv;
        float term2 = x_row[i] * dot_scaled;
        gi_row[i] = term1 - term2;
    }
}
