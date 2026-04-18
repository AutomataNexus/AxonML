// AxonML CUDA BitNet I2_S matmul kernel.
//
// I2_S (1.58-bit ternary) weight format:
//   - Block: 128 weights packed into 32 bytes (2 bits/weight)
//   - Code map: 0 → -1, 1 → 0, 2 → +1  (3 → unused)
//   - Intra-block group-strided layout (matches Microsoft BitNet AVX2):
//       byte k bits 6..7 → weight k       (group 0, shift 6)
//       byte k bits 4..5 → weight k + 32  (group 1, shift 4)
//       byte k bits 2..3 → weight k + 64  (group 2, shift 2)
//       byte k bits 0..1 → weight k + 96  (group 3, shift 0)
//   - ONE tensor-wide f32 scale, passed separately (not in weight bytes)
//
// Decode per weight: w = scale × trit[i] where trit = code - 1 ∈ {-1, 0, +1}.
//
// Kernel: two warps per output row (v2 layout matching Q5_0 / Q5_K / Q8_0).
// Each warp walks half the block range; shared-memory combine at the end.
// Inner work: each of 32 lanes reads ONE byte of the current block (all 32
// bytes of the block consumed cooperatively in one step), decodes 4 trits
// (groups 0..3), and accumulates 4 FMAs against coalesced f32 activation
// loads at strides of 32.
//
// Bandwidth analysis for single-query decode (m=1):
//   Per block: 32 bytes weight + 512 bytes activation + 1 float write
//   Per row:   (k/128) × 32 bytes weight
//   For BitNet-2B-4T (k=2560, n=2560): 20 blocks × 32 = 640 bytes per row
//   Weights are 3.6× smaller than Q4_K (0.25 bytes/weight vs 0.56) —
//   memory-bandwidth-limited decode should be ~2× the rate of Q4_K decode.
//
// Compile: nvcc -ptx -arch=sm_80 --use_fast_math i2s_matmul.cu -o i2s_matmul.ptx

#define I2S_BLOCK_SIZE 128u
#define I2S_BYTES_PER_BLOCK 32u

extern "C" __global__ void i2s_gemv_f32(
    const unsigned char* __restrict__ w,  // [n, k/128 * 32] packed bytes (no scale)
    const float* __restrict__ a,          // [k] activations
    float* __restrict__ c,                // [n] output
    float scale,                          // tensor-wide f32 scale
    unsigned int n,                       // output dim
    unsigned int k                        // input dim
) {
    extern __shared__ float s_partial_i2s[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int n_blocks  = k / I2S_BLOCK_SIZE;
    const unsigned int row_bytes = n_blocks * I2S_BYTES_PER_BLOCK;

    float sum = 0.0f;
    if (j < n) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;

        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * I2S_BYTES_PER_BLOCK;
            // 32 lanes × 1 byte = 32 bytes = one full block in one coalesced load.
            unsigned int byte = (unsigned int)block[lane];

            unsigned int code0 = (byte >> 6) & 0x03u;
            unsigned int code1 = (byte >> 4) & 0x03u;
            unsigned int code2 = (byte >> 2) & 0x03u;
            unsigned int code3 =  byte       & 0x03u;

            // trit = code - 1 ∈ {-1, 0, +1}
            float t0 = (float)((int)code0 - 1);
            float t1 = (float)((int)code1 - 1);
            float t2 = (float)((int)code2 - 1);
            float t3 = (float)((int)code3 - 1);

            unsigned int k_base = b * I2S_BLOCK_SIZE;
            // All four activation reads are lane-stride-1 → fully coalesced.
            sum += a[k_base + lane]         * t0;
            sum += a[k_base + lane + 32u]   * t1;
            sum += a[k_base + lane + 64u]   * t2;
            sum += a[k_base + lane + 96u]   * t3;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial_i2s[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < n) {
        float combined = s_partial_i2s[row_in_cta * 2u]
                       + s_partial_i2s[row_in_cta * 2u + 1u];
        c[j] = scale * combined;
    }
}

// GEMM (m > 1, prefill). One thread per output element.
extern "C" __global__ void i2s_gemm_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
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

    unsigned int n_blocks  = k / I2S_BLOCK_SIZE;
    unsigned int row_bytes = n_blocks * I2S_BYTES_PER_BLOCK;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* a_row = a + (size_t)mi * k;

    float sum = 0.0f;
    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + (size_t)b * I2S_BYTES_PER_BLOCK;
        unsigned int k_base = b * I2S_BLOCK_SIZE;
        // Each thread walks all 32 bytes of the block serially.
        #pragma unroll 8
        for (unsigned int byte_idx = 0; byte_idx < 32; ++byte_idx) {
            unsigned int byte = (unsigned int)block[byte_idx];
            int t0 = (int)((byte >> 6) & 0x03u) - 1;
            int t1 = (int)((byte >> 4) & 0x03u) - 1;
            int t2 = (int)((byte >> 2) & 0x03u) - 1;
            int t3 = (int)( byte       & 0x03u) - 1;
            sum += a_row[k_base + byte_idx]       * (float)t0;
            sum += a_row[k_base + byte_idx + 32]  * (float)t1;
            sum += a_row[k_base + byte_idx + 64]  * (float)t2;
            sum += a_row[k_base + byte_idx + 96]  * (float)t3;
        }
    }
    c[tid] = scale * sum;
}
