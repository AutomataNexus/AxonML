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
// Kernel: two warps per output row (matches i2s_matmul.cu / Q5_0 / Q8_0).
// Inner work: each lane handles 4 sign bits at strides of 32 within the
// block. Bit `b` lives at qs[b/8] bit `b%8`. For lane `l` and stride-set
// {l, l+32, l+64, l+96}: all four bits sit at the SAME bit position `l%8`
// in four different bytes (l/8, l/8+4, l/8+8, l/8+12). 8 lanes share each
// byte read → L1 dedupes cleanly. Activation reads at lane,lane+32/64/96
// are fully coalesced f32.
//
// Bandwidth analysis for single-query decode (m=1) on Bonsai-8B (k=4096):
//   Per row: (k/128)=32 blocks × 18 bytes = 576 bytes
//   Q1_0 is 1.8× smaller than I2_S (640 B/row at the same k=2560 scale)
//   and 4.0× smaller than Q4_K (~2304 B/row at k=4096).
//   Memory-bandwidth-limited decode should outrun every other quant on
//   weight-bandwidth alone.
//
// Compile: nvcc -ptx -arch=sm_80 --use_fast_math q1_0_matmul.cu -o q1_0_matmul.ptx

#include <cuda_fp16.h>

#define Q1_0_BLOCK_SIZE   128u
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

    float sum = 0.0f;
    if (j < n) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;

        const unsigned int byte_idx = lane >> 3;          // l/8
        const unsigned int bit_idx  = lane & 7u;          // l%8

        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * Q1_0_BYTES_PER_BLOCK;
            // Per-block fp16 scale.
            __half d_h = *reinterpret_cast<const __half*>(block);
            float d = __half2float(d_h);
            float neg_d = -d;
            const unsigned char* qs = block + 2;

            // Pull bit at position byte_idx + {0,4,8,12} for this lane.
            unsigned int byte0 = (unsigned int)qs[byte_idx + 0];
            unsigned int byte1 = (unsigned int)qs[byte_idx + 4];
            unsigned int byte2 = (unsigned int)qs[byte_idx + 8];
            unsigned int byte3 = (unsigned int)qs[byte_idx + 12];

            float s0 = ((byte0 >> bit_idx) & 1u) ? d : neg_d;
            float s1 = ((byte1 >> bit_idx) & 1u) ? d : neg_d;
            float s2 = ((byte2 >> bit_idx) & 1u) ? d : neg_d;
            float s3 = ((byte3 >> bit_idx) & 1u) ? d : neg_d;

            const unsigned int k_base = b * Q1_0_BLOCK_SIZE;
            // Lane-stride-1 activation reads → fully coalesced.
            sum += a[k_base + lane]         * s0;
            sum += a[k_base + lane + 32u]   * s1;
            sum += a[k_base + lane + 64u]   * s2;
            sum += a[k_base + lane + 96u]   * s3;
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
                sum += acts[k_base + byte_off * 8u + bit] * s;
            }
        }
    }
    c[tid] = sum;
}
