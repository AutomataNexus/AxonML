// AxonML CUDA Q8_0 dequant-in-shader matmul kernels.
//
// Q8_0 block (34 bytes):
//   [0..2)   d  (f16)       — super-block scale
//   [2..34)  qs (32 i8)     — signed int8 quants
//   value_i = d * (float)(int8_t)qs[i]
//
// Kernel parallelism: two warps per output row (v2 layout matching Q5_0 /
// Q5_K). Each warp walks half the blocks; lanes cooperate one-element-each
// per block, then the two warps combine partial sums via shared memory.
//
// Compile: nvcc -ptx -arch=sm_80 --use_fast_math q8_0_matmul.cu -o q8_0_matmul.ptx

__device__ __forceinline__ float q8_f16_bits_to_f32(unsigned short bits) {
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

#define Q8_0_BYTES 34u
#define Q8_0_BLOCK 32u

// Q8_0 GEMV v2 — two warps per output row split the block range.
extern "C" __global__ void q8_0_gemv_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int out_dim,
    unsigned int in_dim
) {
    extern __shared__ float s_partial_q8_0[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int n_blocks  = in_dim / Q8_0_BLOCK;
    const unsigned int row_bytes = n_blocks * Q8_0_BYTES;

    float sum = 0.0f;
    if (j < out_dim) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * Q8_0_BYTES;
            unsigned short d_bits = (unsigned short)block[0]
                                  | ((unsigned short)block[1] << 8);
            float d = q8_f16_bits_to_f32(d_bits);
            // lane-th int8 quant; cast through signed char so high bit sign-extends.
            int q = (int)(signed char)block[2 + lane];
            float wv = (float)q * d;
            float av = a[b * Q8_0_BLOCK + lane];
            sum += wv * av;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial_q8_0[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < out_dim) {
        c[j] = s_partial_q8_0[row_in_cta * 2u]
             + s_partial_q8_0[row_in_cta * 2u + 1u];
    }
}

// Q8_0 GEMM — one thread per (mi, j) output. Fallback for prefill.
extern "C" __global__ void q8_0_gemm_f32(
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

    unsigned int n_blocks  = in_dim / Q8_0_BLOCK;
    unsigned int row_bytes = n_blocks * Q8_0_BYTES;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* a_row = a + (size_t)mi * in_dim;

    float sum = 0.0f;
    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + (size_t)b * Q8_0_BYTES;
        unsigned short d_bits = (unsigned short)block[0]
                              | ((unsigned short)block[1] << 8);
        float d = q8_f16_bits_to_f32(d_bits);
        unsigned int off = b * Q8_0_BLOCK;
        #pragma unroll 8
        for (int i = 0; i < 32; ++i) {
            int q = (int)(signed char)block[2 + i];
            sum += a_row[off + i] * ((float)q * d);
        }
    }
    c[tid] = sum;
}
