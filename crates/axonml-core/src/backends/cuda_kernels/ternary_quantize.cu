// AxonML CUDA TernaryLinear shadow-weight quantizer.
//
// Two-stage absmean quantization, GPU-resident:
//
//   Stage 1: f32_abssum_reduce
//     Reads f32 shadow weights, sums their absolute values into one
//     scalar (host atomicAdd). Per-CTA warp reduce → single atomicAdd.
//     Caller divides by n to get the absmean and clamps with 1e-8.
//
//   Stage 2: f32_quantize_ternary
//     Per-element threshold:  sign(w) * round(|w| / scale), clamped to
//     {-1, 0, +1}. Output is a flat i8 buffer (passed as u8 because
//     cudarc's `DeviceRepr` doesn't admit i8; reinterpret on entry).
//
// Eliminates the per-step 4 GB GPU→CPU `to_vec()` that would otherwise
// fire inside `TernaryLinear::quantize_weights` for the 1B Trident run.
//
// Compile: nvcc -ptx -arch=sm_89 --use_fast_math ternary_quantize.cu \
//                  -o ternary_quantize.ptx

#include <cuda_fp16.h>

// =============================================================================
// Stage 1 — sum of absolute values (single-scalar reduce).
// =============================================================================

extern "C" __global__ void f32_abssum_reduce(
    const float* __restrict__ x,
    float* __restrict__ out_sum, // caller zeroes before launch
    unsigned int n
) {
    extern __shared__ float smem_abssum[];

    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local  = threadIdx.x;
    unsigned int lane   = local & 31u;
    unsigned int warp   = local >> 5;

    float v = (tid < n) ? fabsf(x[tid]) : 0.0f;

    // Intra-warp reduce.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0u) {
        smem_abssum[warp] = v;
    }
    __syncthreads();

    // First warp reduces the per-warp partials. We assume blockDim.x is
    // a multiple of 32 and ≤ 1024, so at most 32 warp partials.
    if (warp == 0u) {
        unsigned int n_warps = blockDim.x >> 5;
        v = (local < n_warps) ? smem_abssum[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0u) {
            atomicAdd(out_sum, v);
        }
    }
}

// =============================================================================
// Stage 2 — per-element absmean quantize to {-1, 0, +1} as i8.
// =============================================================================

extern "C" __global__ void f32_quantize_ternary(
    const float* __restrict__ x,
    unsigned char* __restrict__ out_i8, // signed char reinterpret
    unsigned int n,
    float scale
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float v = x[tid];
    // sign(v) * round(|v|/scale) clamped to {-1, 0, +1}.
    float abs_v = fabsf(v);
    float normalized = fminf(abs_v / scale, 1.0f);
    int rounded = __float2int_rn(normalized); // 0 or 1 after clamp
    int code;
    if (v > 0.0f) {
        code = rounded;
    } else if (v < 0.0f) {
        code = -rounded;
    } else {
        code = 0;
    }
    // Reinterpret-friendly store: take the i8 bit pattern of `code` into
    // the u8 output slot.
    out_i8[tid] = (unsigned char)(signed char)code;
}
