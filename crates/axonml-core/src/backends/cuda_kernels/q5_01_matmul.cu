// AxonML CUDA Q5_0 / Q5_1 dequant-in-shader matmul kernels.
//
// Both variants: 32 elements per block, signed (Q5_0) or unsigned (Q5_1)
// 5-bit quants packed as 4-bit low nibble (qs) + 1-bit high (qh).
//
// Q5_0 block (22 bytes):
//   [0..2)   d  (f16)    — super-block scale
//   [2..6)   qh (u32)    — 5th bit per element (packed 32 bits)
//   [6..22)  qs (16 B)   — low nibble per element, 2 per byte
//   value_i = ((lo | (hi_bit << 4)) - 16) * d        signed [-16, 15]
//
// Q5_1 block (24 bytes):
//   [0..2)   d  (f16)
//   [2..4)   m  (f16)    — super-block min (added after scale)
//   [4..8)   qh (u32)
//   [8..24)  qs (16 B)
//   value_i = (lo | (hi_bit << 4)) * d + m           unsigned [0, 31]
//
// Element layout inside qs/qh (matches llama.cpp reference + the Rust
// dequant in nexus-serve/src/model/gguf.rs):
//   lanes 0..15  → qs[i] & 0x0F, qh bit i
//   lanes 16..31 → qs[i - 16] >> 4, qh bit i
//
// Kernel parallelism: one warp per output row. Each lane processes one
// element per block across the row's `in_dim / 32` blocks, then warp-
// reduces into the final dot product. Matches the Q6_K GEMV layout.
//
// Compile: nvcc -ptx -arch=sm_80 --use_fast_math q5_01_matmul.cu -o q5_01_matmul.ptx

// Manual f16 → f32. Matches nexus-serve/src/model/gguf.rs:f16_to_f32.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    unsigned int sign = (unsigned int)(bits >> 15) & 1u;
    int exp = (int)((bits >> 10) & 0x1F);
    unsigned int frac = (unsigned int)(bits & 0x3FF);

    unsigned int result;
    if (exp == 0) {
        if (frac == 0) {
            result = sign << 31;
        } else {
            int e = -14;
            unsigned int f = frac;
            while ((f & 0x400u) == 0) {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FFu;
            unsigned int exp32 = (unsigned int)(127 + e);
            result = (sign << 31) | (exp32 << 23) | (f << 13);
        }
    } else if (exp == 31) {
        if (frac == 0) {
            result = (sign << 31) | (0xFFu << 23);
        } else {
            result = 0x7FC00000u;
        }
    } else {
        unsigned int exp32 = (unsigned int)(exp + 112);
        result = (sign << 31) | (exp32 << 23) | (frac << 13);
    }
    return __int_as_float((int)result);
}

// Decode one lane's weight from a Q5_0 block. Returns d * (signed 5-bit).
__device__ __forceinline__ float q5_0_weight(
    const unsigned char* __restrict__ block, unsigned int lane
) {
    unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
    float d = f16_bits_to_f32(d_bits);
    unsigned int qh = (unsigned int)block[2]
                    | ((unsigned int)block[3] << 8)
                    | ((unsigned int)block[4] << 16)
                    | ((unsigned int)block[5] << 24);
    const unsigned char* qs = block + 6;
    unsigned int nibble = (lane < 16u)
        ? (qs[lane] & 0x0Fu)
        : (qs[lane - 16u] >> 4);
    unsigned int hi = (qh >> lane) & 1u;
    int q = (int)(nibble | (hi << 4)) - 16;
    return (float)q * d;
}

// Decode one lane's weight from a Q5_1 block.
__device__ __forceinline__ float q5_1_weight(
    const unsigned char* __restrict__ block, unsigned int lane
) {
    unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
    unsigned short m_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
    float d = f16_bits_to_f32(d_bits);
    float m = f16_bits_to_f32(m_bits);
    unsigned int qh = (unsigned int)block[4]
                    | ((unsigned int)block[5] << 8)
                    | ((unsigned int)block[6] << 16)
                    | ((unsigned int)block[7] << 24);
    const unsigned char* qs = block + 8;
    unsigned int nibble = (lane < 16u)
        ? (qs[lane] & 0x0Fu)
        : (qs[lane - 16u] >> 4);
    unsigned int hi = (qh >> lane) & 1u;
    unsigned int q = nibble | (hi << 4);
    return (float)q * d + m;
}

// ============================================================================
// Q5_0 GEMV — one warp per output row j. 32 lanes cooperate per block of
// 32 elements; each lane contributes one multiply-add, warp-reduce at end.
// ============================================================================

#define Q5_0_BYTES 22u
#define Q5_1_BYTES 24u
#define Q5_BLOCK   32u

// Q5_0 GEMV v2: two warps per output row split the block range. Each
// warp walks its half of blocks, lanes cooperate one-element-each per
// block, then the two warps combine their partial sums via shared
// memory. Same launch geometry as Q4_K / Q5_K v2: rows_per_cta × 2
// warps/row × 32 threads, shared_mem = rows_per_cta * 2 * f32.
extern "C" __global__ void q5_0_gemv_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int out_dim,
    unsigned int in_dim
) {
    extern __shared__ float s_partial_q5_0[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int n_blocks  = in_dim / Q5_BLOCK;
    const unsigned int row_bytes = n_blocks * Q5_0_BYTES;

    float sum = 0.0f;
    if (j < out_dim) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * Q5_0_BYTES;
            float wv = q5_0_weight(block, lane);
            float av = a[b * Q5_BLOCK + lane];
            sum += wv * av;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial_q5_0[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < out_dim) {
        c[j] = s_partial_q5_0[row_in_cta * 2u]
             + s_partial_q5_0[row_in_cta * 2u + 1u];
    }
}

extern "C" __global__ void q5_0_gemm_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int m_dim,
    unsigned int out_dim,
    unsigned int in_dim
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = m_dim * out_dim;
    if (tid >= total) return;

    unsigned int mi = tid / out_dim;
    unsigned int j  = tid % out_dim;

    unsigned int n_blocks = in_dim / Q5_BLOCK;
    unsigned int row_bytes = n_blocks * Q5_0_BYTES;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* a_row = a + (size_t)mi * in_dim;

    float sum = 0.0f;
    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + (size_t)b * Q5_0_BYTES;
        unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        float d = f16_bits_to_f32(d_bits);
        unsigned int qh = (unsigned int)block[2]
                        | ((unsigned int)block[3] << 8)
                        | ((unsigned int)block[4] << 16)
                        | ((unsigned int)block[5] << 24);
        const unsigned char* qs = block + 6;
        unsigned int off = b * Q5_BLOCK;
        #pragma unroll 8
        for (int i = 0; i < 16; ++i) {
            unsigned int lo1 = qs[i] & 0x0Fu;
            unsigned int lo2 = (qs[i] >> 4) & 0x0Fu;
            unsigned int hi1 = (qh >> i) & 1u;
            unsigned int hi2 = (qh >> (i + 16)) & 1u;
            int q1 = (int)(lo1 | (hi1 << 4)) - 16;
            int q2 = (int)(lo2 | (hi2 << 4)) - 16;
            sum += a_row[off + i]      * ((float)q1 * d);
            sum += a_row[off + i + 16] * ((float)q2 * d);
        }
    }
    c[tid] = sum;
}

// Q5_1 GEMV v2 — same two-warp-per-row structure as q5_0_gemv_f32.
extern "C" __global__ void q5_1_gemv_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int out_dim,
    unsigned int in_dim
) {
    extern __shared__ float s_partial_q5_1[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int n_blocks  = in_dim / Q5_BLOCK;
    const unsigned int row_bytes = n_blocks * Q5_1_BYTES;

    float sum = 0.0f;
    if (j < out_dim) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * Q5_1_BYTES;
            float wv = q5_1_weight(block, lane);
            float av = a[b * Q5_BLOCK + lane];
            sum += wv * av;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial_q5_1[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < out_dim) {
        c[j] = s_partial_q5_1[row_in_cta * 2u]
             + s_partial_q5_1[row_in_cta * 2u + 1u];
    }
}

extern "C" __global__ void q5_1_gemm_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int m_dim,
    unsigned int out_dim,
    unsigned int in_dim
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = m_dim * out_dim;
    if (tid >= total) return;

    unsigned int mi = tid / out_dim;
    unsigned int j  = tid % out_dim;

    unsigned int n_blocks = in_dim / Q5_BLOCK;
    unsigned int row_bytes = n_blocks * Q5_1_BYTES;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* a_row = a + (size_t)mi * in_dim;

    float sum = 0.0f;
    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + (size_t)b * Q5_1_BYTES;
        unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short m_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d = f16_bits_to_f32(d_bits);
        float m = f16_bits_to_f32(m_bits);
        unsigned int qh = (unsigned int)block[4]
                        | ((unsigned int)block[5] << 8)
                        | ((unsigned int)block[6] << 16)
                        | ((unsigned int)block[7] << 24);
        const unsigned char* qs = block + 8;
        unsigned int off = b * Q5_BLOCK;
        #pragma unroll 8
        for (int i = 0; i < 16; ++i) {
            unsigned int lo1 = qs[i] & 0x0Fu;
            unsigned int lo2 = (qs[i] >> 4) & 0x0Fu;
            unsigned int hi1 = (qh >> i) & 1u;
            unsigned int hi2 = (qh >> (i + 16)) & 1u;
            unsigned int q1 = lo1 | (hi1 << 4);
            unsigned int q2 = lo2 | (hi2 << 4);
            sum += a_row[off + i]      * ((float)q1 * d + m);
            sum += a_row[off + i + 16] * ((float)q2 * d + m);
        }
    }
    c[tid] = sum;
}
