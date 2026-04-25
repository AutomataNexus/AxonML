// AxonML CUDA Q1_0 (1-bit) matmul — DP4A int8 path.
//
// Companion to `q1_0_matmul.cu`. Two kernels:
//
//   1. `q1_0_quantize_acts_q8`  — per-32-element absmax int8 quant of an
//      f32 activation row. Lane-cooperative, one warp per 32-element chunk.
//      Output: int8[k] + fp16 scales[k/32].
//
//   2. `q1_0_gemv_dp4a_f32`     — sign-bit weights × int8 activations via
//      `__dp4a` (4× int8 MAC per PTX instruction on sm_61+, native on the
//      integer pipeline). Mirrors PrismML's `vec_dot_q1_0_q8_1` math with
//      Q8_0-style scale-only activations (no `s` sum field — pure binary
//      weights have zero DC term, so the asymmetric `s` correction is
//      unneeded for our case).
//
// Lane layout (matches q1_0_matmul.cu v2): lane `l` covers 4 contiguous
// elements `[l*4 .. l*4+3]` of one Q1_0 block. Sign bits live in
// `qs[l/2]` low/high nibble. Activation int8s read as one int32 (4 packed)
// at byte offset `l*4` of the int8 row. The Q8 chunk index for those 4
// elements is `(l*4)/32 = l/8`; lanes 0..7 share chunk 4b+0, 8..15 share
// 4b+1, etc., so each warp covers the 4 chunks of one Q1_0 block in
// parallel.
//
// dp4a accumulates int8 × int8 → int32 inside the lane. Each lane then
// scales by its chunk's fp16 d_act × the block's fp16 d_w → float, and a
// 32-wide warp reduce sums the contributions of one full block. Two warps
// per row (matching v2's launch shape) split the n_blocks range in half.
//
// Compile: nvcc -ptx -arch=sm_89 --use_fast_math q1_0_matmul_dp4a.cu \
//                  -o q1_0_matmul_dp4a.ptx

#include <cuda_fp16.h>

#define Q1_0_BLOCK_SIZE      128u
#define Q1_0_BYTES_PER_BLOCK 18u
#define Q8_CHUNK             32u

// =============================================================================
// Activation quantization kernel — f32 row → int8 + fp16 per-chunk scales
// =============================================================================
//
// One warp per 32-element chunk. Lane stride 1 within the chunk → fully
// coalesced f32 load + int8 store.

extern "C" __global__ void q1_0_quantize_acts_q8(
    const float* __restrict__ a,   // [k] f32 activations (one row, m=1)
    unsigned char* __restrict__ a_q_u, // [k] int8 quantized output
    __half* __restrict__ a_d,      // [k/32] fp16 per-chunk scale
    unsigned int k
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int chunk = tid >> 5;
    const unsigned int n_chunks = k / Q8_CHUNK;
    if (chunk >= n_chunks) return;

    const unsigned int idx = chunk * Q8_CHUNK + lane;
    float v = a[idx];

    // Warp absmax reduce.
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }

    // Scale: amax / 127. If the chunk is all zeros, force scale=0 to keep
    // the dot product exact.
    float d = amax / 127.0f;
    float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(v * inv_d);
    q = max(-127, min(127, q));

    a_q_u[idx] = (unsigned char)(signed char)q;
    if (lane == 0u) {
        a_d[chunk] = __float2half(d);
    }
}

// =============================================================================
// DP4A matmul kernel — Q1_0 × Q8 → f32
// =============================================================================

extern "C" __global__ void q1_0_gemv_dp4a_f32(
    const unsigned char* __restrict__ w,  // [n, n_blocks * 18] Q1_0 bytes
    const unsigned char* __restrict__ a_q_u,  // [k] int8 activations
    const __half* __restrict__ a_d,       // [k/32] fp16 act scales
    float* __restrict__ c,                // [n] output
    unsigned int n,
    unsigned int k
) {
    extern __shared__ float s_partial_q10dp[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int n_blocks  = k / Q1_0_BLOCK_SIZE;
    const unsigned int row_bytes = n_blocks * Q1_0_BYTES_PER_BLOCK;

    // Lane → element-quad mapping. Lane `l` covers [l*4..l*4+3].
    const unsigned int byte_idx  = lane >> 1;        // qs byte index = l/2
    const unsigned int nibble_sh = (lane & 1u) << 2; // 0 (low) or 4 (high) nibble
    // Q8 chunk index for [l*4..l*4+3] within the current Q1_0 block:
    //   chunk_offset = (l*4)/32 = l/8 ∈ {0,1,2,3}
    const unsigned int chunk_off = lane >> 3;

    float sum = 0.0f;
    if (j < n) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;

        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * Q1_0_BYTES_PER_BLOCK;
            // Per-block fp16 weight scale.
            __half d_w_h = *reinterpret_cast<const __half*>(block);
            float d_w = __half2float(d_w_h);
            const unsigned char* qs = block + 2;

            // Pull 4 sign bits for [l*4..l*4+3].
            unsigned int nibble = ((unsigned int)qs[byte_idx] >> nibble_sh) & 0xFu;

            // Expand each bit to ±1 int8, packed into int32 (lo→hi byte order).
            int b0 = (nibble & 0x1u) ? 1 : -1;
            int b1 = (nibble & 0x2u) ? 1 : -1;
            int b2 = (nibble & 0x4u) ? 1 : -1;
            int b3 = (nibble & 0x8u) ? 1 : -1;
            int w_int = (b0 & 0xFF)
                      | ((b1 & 0xFF) << 8)
                      | ((b2 & 0xFF) << 16)
                      | ((b3 & 0xFF) << 24);

            // 4 packed int8 activations for [l*4..l*4+3].
            const signed char* a_q = reinterpret_cast<const signed char*>(a_q_u);
            int a_int = *reinterpret_cast<const int*>(a_q + b * Q1_0_BLOCK_SIZE + lane * 4u);

            // dp4a: sumi += sum of 4 int8 × int8 products as int32.
            int sumi = __dp4a(w_int, a_int, 0);

            // Activation scale for the chunk this lane lives in.
            __half d_a_h = a_d[b * 4u + chunk_off];
            float d_a = __half2float(d_a_h);

            sum = fmaf((float)sumi * d_a, d_w, sum);
        }
    }

    // Warp reduction (32 lanes per warp, summing across all 4 chunks of the
    // block * the half-block range this warp owns).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial_q10dp[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < n) {
        float combined = s_partial_q10dp[row_in_cta * 2u]
                       + s_partial_q10dp[row_in_cta * 2u + 1u];
        c[j] = combined;
    }
}
