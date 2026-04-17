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

extern "C" __global__ void rope_split_halves_f32(
    float* __restrict__ x,
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
    const float a = x[base];
    const float b = x[base + half];
    x[base]        = c * a - s * b;
    x[base + half] = s * a + c * b;
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
