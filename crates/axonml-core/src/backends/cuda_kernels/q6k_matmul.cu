// AxonML CUDA Q6_K dequant-in-shader matmul kernels
//
// Q6_K super-block layout (210 bytes for 256 elements):
//   [  0..128) : ql  — 128 bytes of low 4 bits (2 values per byte)
//   [128..192) : qh  — 64 bytes of high 2 bits (4 values per byte)
//   [192..208) : sc  — 16 bytes of int8 scales (one per 16-element group)
//   [208..210) : d   — f16 super-block scale
//
// Per super-block: 2 chunks of 128 elements. Each chunk iterates l in 0..32
// and emits outputs at positions (y_off+l, y_off+l+32, y_off+l+64, y_off+l+96).
// After a chunk, y_off += 128, ql_off += 64, qh_off += 32, sc_off += 8.
// `is = l / 16`, and the four weights use scales sc[is, is+2, is+4, is+6].
// Each 6-bit weight = (ql_nibble | (qh_2bits << 4)) as int8 − 32, giving
// a signed value in [-32, 31].
//
// Physical weight layout (from GGUF): row-major [out, in], where each row of
// B is (in/256) contiguous 210-byte blocks. Byte offset to block b of row j
// is (j * (in/256) + b) * 210.
//
// Compile with: nvcc -ptx -arch=sm_80 --use_fast_math q6k_matmul.cu -o q6k_matmul.ptx

// Manual f16 → f32 (no <cuda_fp16.h> to avoid __half type-punning pitfalls).
// Exact port of the Rust `f16_to_f32` in nexus-serve/src/model/gguf.rs.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    unsigned int sign = (unsigned int)(bits >> 15) & 1u;
    int exp = (int)((bits >> 10) & 0x1F);
    unsigned int frac = (unsigned int)(bits & 0x3FF);

    unsigned int result;
    if (exp == 0) {
        if (frac == 0) {
            result = sign << 31;
        } else {
            // Subnormal f16: normalize left until bit 10 is set; start e = -14.
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
        unsigned int exp32 = (unsigned int)(exp + 112); // 127 - 15
        result = (sign << 31) | (exp32 << 23) | (frac << 13);
    }
    return __int_as_float((int)result);
}

// Core per-row accumulator for Q6_K. Given a pointer to row j of the weight
// tensor and a pointer to the corresponding activation row, walks all blocks
// of the row and returns the dot product.
__device__ __forceinline__ float q6k_row_dot(
    const unsigned char* __restrict__ row,
    const float* __restrict__ a_row,
    unsigned int n_blocks
) {
    float sum = 0.0f;
    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + b * 210;
        const unsigned char* ql_arr = block;            // 128 bytes
        const unsigned char* qh_arr = block + 128;      // 64 bytes
        const signed char*   sc_arr = (const signed char*)(block + 192); // i8[16]

        unsigned short d_bits = (unsigned short)block[208]
                              | ((unsigned short)block[209] << 8);
        float d = f16_bits_to_f32(d_bits);

        unsigned int ql_off = 0;
        unsigned int qh_off = 0;
        unsigned int sc_off = 0;
        unsigned int y_off  = 0;
        unsigned int a_base = b * 256;

        #pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            #pragma unroll
            for (unsigned int l = 0; l < 32; ++l) {
                unsigned int is = l >> 4;

                unsigned int ql0 = ql_arr[ql_off + l];
                unsigned int ql1 = ql_arr[ql_off + l + 32];
                unsigned int qhv = qh_arr[qh_off + l];

                int q1 = (int)((ql0 & 0x0Fu) | ((qhv & 0x03u) << 4)) - 32;
                int q2 = (int)((ql1 & 0x0Fu) | (((qhv >> 2) & 0x03u) << 4)) - 32;
                int q3 = (int)((ql0 >> 4)    | (((qhv >> 4) & 0x03u) << 4)) - 32;
                int q4 = (int)((ql1 >> 4)    | (((qhv >> 6) & 0x03u) << 4)) - 32;

                float s1 = d * (float)sc_arr[sc_off + is];
                float s2 = d * (float)sc_arr[sc_off + is + 2];
                float s3 = d * (float)sc_arr[sc_off + is + 4];
                float s4 = d * (float)sc_arr[sc_off + is + 6];

                sum += a_row[a_base + y_off + l]      * (s1 * (float)q1);
                sum += a_row[a_base + y_off + l + 32] * (s2 * (float)q2);
                sum += a_row[a_base + y_off + l + 64] * (s3 * (float)q3);
                sum += a_row[a_base + y_off + l + 96] * (s4 * (float)q4);
            }
            y_off  += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    return sum;
}

// Q6_K GEMV (cooperative warp reduction): c = a @ B^T.
//
// One WARP per output element, 32 threads cooperate per block:
//   - lane l, chunk c (0..2) of 128 elements each:
//     * reads ql[c*64 + l]   and ql[c*64 + l + 32] (8 nibbles → 4 weights)
//     * reads qh[c*32 + l]   (8 high-2-bit codes → 4 weights)
//     * reads sc[c*8 + is], sc[c*8 + is+2], sc[c*8 + is+4], sc[c*8 + is+6]
//       where is = l / 16
//     * accumulates 4 FMAs: a[c*128 + l], a[c*128 + l+32], a[c*128 + l+64], a[c*128 + l+96]
//
// 32 lanes × 4 outputs/chunk × 2 chunks = 256 elements per block, all work
// shared across the warp. Every lane-local read is part of a coalesced
// 32-byte (ql/qh) or 128-byte (a) transaction. Scales / d are broadcast
// by the compiler since they don't depend on lane.
//
// Block: 4 warps × 32 threads = 128 threads → 4 output rows per CTA.
// Launch: grid = ((out + 3) / 4), block = 128.
extern "C" __global__ void q6k_gemv_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int out_dim,
    unsigned int in_dim
) {
    const unsigned int tid     = threadIdx.x;
    const unsigned int lane    = tid & 31u;
    const unsigned int warp_id = tid >> 5;
    const unsigned int j       = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (j >= out_dim) return;

    const unsigned int n_blocks = in_dim / 256;
    const unsigned int row_bytes = n_blocks * 210;
    const unsigned char* row = w + (size_t)j * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + b * 210;
        const unsigned char* ql_arr = block;
        const unsigned char* qh_arr = block + 128;
        const signed char*   sc_arr = (const signed char*)(block + 192);

        unsigned short d_bits = (unsigned short)block[208]
                              | ((unsigned short)block[209] << 8);
        float d = f16_bits_to_f32(d_bits);

        const unsigned int is = lane >> 4;  // 0 for lanes 0-15, 1 for lanes 16-31

        #pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            unsigned int ql_off = (unsigned int)chunk * 64u;
            unsigned int qh_off = (unsigned int)chunk * 32u;
            unsigned int sc_off = (unsigned int)chunk * 8u;
            unsigned int a_base = b * 256u + (unsigned int)chunk * 128u;

            unsigned int ql0 = ql_arr[ql_off + lane];
            unsigned int ql1 = ql_arr[ql_off + lane + 32];
            unsigned int qhv = qh_arr[qh_off + lane];

            int q1 = (int)((ql0 & 0x0Fu) | ((qhv & 0x03u) << 4)) - 32;
            int q2 = (int)((ql1 & 0x0Fu) | (((qhv >> 2) & 0x03u) << 4)) - 32;
            int q3 = (int)((ql0 >> 4)    | (((qhv >> 4) & 0x03u) << 4)) - 32;
            int q4 = (int)((ql1 >> 4)    | (((qhv >> 6) & 0x03u) << 4)) - 32;

            float s1 = d * (float)sc_arr[sc_off + is];
            float s2 = d * (float)sc_arr[sc_off + is + 2];
            float s3 = d * (float)sc_arr[sc_off + is + 4];
            float s4 = d * (float)sc_arr[sc_off + is + 6];

            sum += a[a_base + lane]      * (s1 * (float)q1);
            sum += a[a_base + lane + 32] * (s2 * (float)q2);
            sum += a[a_base + lane + 64] * (s3 * (float)q3);
            sum += a[a_base + lane + 96] * (s4 * (float)q4);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0) c[j] = sum;
}

// Q6_K GEMM: c = a @ B^T, where a is [m, in] and B is Q6_K [out, in].
//
// Shapes:
//   a : [m, in]          float32 row-major
//   w : [out, in] Q6_K   raw bytes (out * in / 256 * 210 total)
//   c : [m, out]         float32
//
// Parallelism: one thread per output element c[mi, j]. Total = m * out.
extern "C" __global__ void q6k_gemm_f32(
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

    unsigned int n_blocks = in_dim / 256;
    unsigned int row_bytes = n_blocks * 210;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* a_row = a + (size_t)mi * in_dim;

    c[tid] = q6k_row_dot(row, a_row, n_blocks);
}
