// AxonML CUDA TernaryLinear kernels — raw i8 ternary × f32.
//
// For axonml-nn::layers::ternary::TernaryLinear, which stores ternary weights
// as a flat `Vec<i8>` (NOT the BitNet I2_S group-strided 2-bit packing).
// Each weight is one byte in `{-1, 0, +1}` plus a tensor-wide f32 scale that
// gets multiplied at the end of the dot product.
//
// Three kernels:
//
//   1. ternary_gemv_f32 — GEMV (m=1 decode). Two warps per output row, lane
//      stride 1 over k for coalesced f32 activation reads. Each lane reads
//      one i8 weight + one f32 act per inner step, branches on sign, and
//      adds/subtracts the act into a partial accumulator. Warp reduce →
//      shared-memory partial → scale × sum.
//
//   2. ternary_gemm_f32 — GEMM (m>1, also training prefill). One thread
//      per output element, plain serial inner loop. Mirrors i2s_gemm_f32.
//
//   3. ternary_grad_input_f32 — backward for grad_input:
//        grad_input[b,j] = scale * sum_o( ternary[o*in+j] * grad_output[b,o] )
//      Same shape contract as gemv but reads ternary in column-major
//      (transposed) order. One CTA per (batch, in_features) tile.
//
// The `signed char* w` pointer carries Rust's `i8` values; CUDA reads them
// as plain bytes (-128..127). Any value outside `{-1, 0, +1}` would only
// produce wrong arithmetic, not undefined behavior.
//
// Compile: nvcc -ptx -arch=sm_89 --use_fast_math ternary_matmul.cu \
//                  -o ternary_matmul.ptx

#include <cuda_fp16.h>

// =============================================================================
// GEMV — m=1 decode. Two warps per output row.
// =============================================================================

extern "C" __global__ void ternary_gemv_f32(
    const signed char* __restrict__ w,  // [n, k] ternary weights row-major
    const float* __restrict__ a,        // [k] activations
    float* __restrict__ c,              // [n] output
    float scale,                        // tensor-wide f32 scale
    unsigned int n,                     // out_features
    unsigned int k                      // in_features
) {
    extern __shared__ float s_partial_t[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    float sum = 0.0f;
    if (j < n) {
        const signed char* row = w + (size_t)j * k;
        // Each warp owns half the k range; lane stride 32 walks contiguous
        // f32 acts → coalesced loads. Inner loop is sequential over the
        // half-range with stride 32.
        const unsigned int half = k >> 1;
        const unsigned int k_start = warp_in_row ? half : 0u;
        const unsigned int k_end   = warp_in_row ? k    : half;

        for (unsigned int kk = k_start + lane; kk < k_end; kk += 32u) {
            int wt = (int)row[kk]; // sign-extend
            float av = a[kk];
            if (wt > 0) {
                sum += av;
            } else if (wt < 0) {
                sum -= av;
            }
            // wt == 0: skip
        }
    }

    // Warp reduce.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial_t[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < n) {
        float combined = s_partial_t[row_in_cta * 2u]
                       + s_partial_t[row_in_cta * 2u + 1u];
        c[j] = scale * combined;
    }
}

// =============================================================================
// GEMM — m>1 (training prefill). One thread per output element.
// =============================================================================

extern "C" __global__ void ternary_gemm_f32(
    const signed char* __restrict__ w,  // [n, k]
    const float* __restrict__ a,        // [m, k]
    float* __restrict__ c,              // [m, n]
    float scale,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = m * n;
    if (tid >= total) return;

    unsigned int mi = tid / n;
    unsigned int j  = tid % n;

    const signed char* row = w + (size_t)j * k;
    const float* a_row = a + (size_t)mi * k;

    float sum = 0.0f;
    for (unsigned int kk = 0; kk < k; ++kk) {
        int wt = (int)row[kk];
        float av = a_row[kk];
        if (wt > 0) {
            sum += av;
        } else if (wt < 0) {
            sum -= av;
        }
    }
    c[tid] = scale * sum;
}

// =============================================================================
// Backward grad_input — same shape as forward but ternary access transposed.
// grad_input[b, j] = scale * sum_o( ternary[o, j] * grad_output[b, o] )
// =============================================================================
//
// Launch shape: one thread per (batch, in_features) output element. The
// inner loop walks `out_features` and reads ternary in column-major
// (`w[o * in_features + j]`).

extern "C" __global__ void ternary_grad_input_f32(
    const signed char* __restrict__ w,    // [out, in] ternary weights
    const float* __restrict__ grad_out,   // [batch, out]
    float* __restrict__ grad_in,          // [batch, in]
    float scale,
    unsigned int batch_size,
    unsigned int in_features,
    unsigned int out_features
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch_size * in_features;
    if (tid >= total) return;

    unsigned int b = tid / in_features;
    unsigned int j = tid % in_features;

    const float* g_row = grad_out + (size_t)b * out_features;

    float sum = 0.0f;
    for (unsigned int o = 0; o < out_features; ++o) {
        int wt = (int)w[(size_t)o * in_features + j];
        float gv = g_row[o];
        if (wt > 0) {
            sum += gv;
        } else if (wt < 0) {
            sum -= gv;
        }
    }
    grad_in[tid] = scale * sum;
}

// =============================================================================
// Backward grad_bias — sum over the batch axis.
// =============================================================================

extern "C" __global__ void ternary_grad_bias_f32(
    const float* __restrict__ grad_out, // [batch, out]
    float* __restrict__ grad_bias,      // [out]
    unsigned int batch_size,
    unsigned int out_features
) {
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;

    float sum = 0.0f;
    for (unsigned int b = 0; b < batch_size; ++b) {
        sum += grad_out[(size_t)b * out_features + o];
    }
    grad_bias[o] = sum;
}
