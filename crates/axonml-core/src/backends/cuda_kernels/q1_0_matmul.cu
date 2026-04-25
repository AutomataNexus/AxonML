// AxonML CUDA PrismML Q1_0 (1-bit) matmul kernel.
//
// Q1_0 weight format (matches PrismML llama.cpp fork's `block_q1_0`):
//   - Block: 128 weights packed into 18 bytes
//       bytes  0..1   : fp16 scale `d` (per-block, NOT tensor-wide)
//       bytes  2..17  : 16 sign bytes (`qs`); element j → qs[j/8] bit j%8
//   - Code map per bit: 1 → +d,  0 → -d.  No zero state.
//
// Decode per weight: w = bit ? d : -d
//
// v2 layout (2026-04-24 fine-tune): lane-l processes 4 contiguous elements
// `[l*4 .. l*4+3]` per Q1_0 block. Activation reads collapse to ONE 16-byte
// `float4` load per lane per block (vs four separate f32 loads in v1). Sign
// bits for those 4 contiguous elements live in nibble (l*4)/8 = l/2 of the
// `qs` byte at intra-byte offset (l*4)%8 ∈ {0, 4} — i.e. low or high nibble
// of byte qs[l/2]. Even lanes take the low nibble, odd lanes the high.
// Halving the global-load count is the main bandwidth win on the lane side;
// the sign-extraction math is one byte load + four predicated selects.
//
// Bandwidth analysis for single-query decode (m=1) on Bonsai-8B (k=4096):
//   Per row: (k/128)=32 blocks × 18 bytes = 576 bytes weight
//   Activations are read once per CTA (not per row) — amortized to ~free
//   on the n=4096 axis. Q1_0 is 1.8× narrower than I2_S and 4× narrower
//   than Q4_K, so this kernel sits squarely on the DRAM ceiling once
//   launch overhead is amortized.
//
// Compile: nvcc -ptx -arch=sm_89 --use_fast_math q1_0_matmul.cu -o q1_0_matmul.ptx

#include <cuda_fp16.h>

#define Q1_0_BLOCK_SIZE      128u
#define Q1_0_BYTES_PER_BLOCK 18u

extern "C" __global__ void q1_0_gemv_f32(
    const unsigned char* __restrict__ w,  // [n, n_blocks * 18]  raw GGUF Q1_0 bytes
    const float* __restrict__ a,          // [k] activations
    float* __restrict__ c,                // [n] output
    unsigned int n,                       // output dim
    unsigned int k                        // input dim
) {
    extern __shared__ float s_partial_q10[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int n_blocks  = k / Q1_0_BLOCK_SIZE;
    const unsigned int row_bytes = n_blocks * Q1_0_BYTES_PER_BLOCK;

    // Lane→nibble mapping. Lane l covers 4 contiguous elements [l*4..l*4+3].
    // Those 4 sign bits sit in qs[byte_idx] at low (lo_nibble=true) or high
    // nibble depending on parity.
    const unsigned int byte_idx  = lane >> 1;      // (l*4)/8 = l/2
    const unsigned int nibble_sh = (lane & 1u) << 2; // 0 if even lane, 4 if odd

    float sum = 0.0f;
    if (j < n) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;

        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * Q1_0_BYTES_PER_BLOCK;
            // Per-block fp16 scale.
            __half d_h = *reinterpret_cast<const __half*>(block);
            float d = __half2float(d_h);
            float neg_d = -d;
            const unsigned char* qs = block + 2;

            // 4 sign bits for [l*4 .. l*4+3]: nibble at (qs[l/2] >> nibble_sh) & 0xF
            unsigned int nibble = ((unsigned int)qs[byte_idx] >> nibble_sh) & 0xFu;

            // Vectorized 16-byte activation read for [l*4 .. l*4+3].
            const float4 a4 = reinterpret_cast<const float4*>(a + b * Q1_0_BLOCK_SIZE)[lane];

            float s0 = (nibble & 0x1u) ? d : neg_d;
            float s1 = (nibble & 0x2u) ? d : neg_d;
            float s2 = (nibble & 0x4u) ? d : neg_d;
            float s3 = (nibble & 0x8u) ? d : neg_d;

            sum = fmaf(a4.x, s0, sum);
            sum = fmaf(a4.y, s1, sum);
            sum = fmaf(a4.z, s2, sum);
            sum = fmaf(a4.w, s3, sum);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial_q10[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < n) {
        float combined = s_partial_q10[row_in_cta * 2u]
                       + s_partial_q10[row_in_cta * 2u + 1u];
        c[j] = combined;
    }
}

// GEMM (m > 1, prefill). One thread per output element.
extern "C" __global__ void q1_0_gemm_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = m * n;
    if (tid >= total) return;

    unsigned int mi = tid / n;
    unsigned int j  = tid % n;

    const unsigned int n_blocks  = k / Q1_0_BLOCK_SIZE;
    const unsigned int row_bytes = n_blocks * Q1_0_BYTES_PER_BLOCK;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* acts = a + (size_t)mi * k;

    float sum = 0.0f;
    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + (size_t)b * Q1_0_BYTES_PER_BLOCK;
        __half d_h = *reinterpret_cast<const __half*>(block);
        float d = __half2float(d_h);
        float neg_d = -d;
        const unsigned char* qs = block + 2;

        const unsigned int k_base = b * Q1_0_BLOCK_SIZE;
        #pragma unroll
        for (unsigned int byte_off = 0; byte_off < 16; ++byte_off) {
            unsigned int byte = (unsigned int)qs[byte_off];
            #pragma unroll
            for (unsigned int bit = 0; bit < 8; ++bit) {
                float s = ((byte >> bit) & 1u) ? d : neg_d;
                sum = fmaf(acts[k_base + byte_off * 8u + bit], s, sum);
            }
        }
    }
    c[tid] = sum;
}
