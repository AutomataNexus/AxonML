// AxonML CUDA Q5_K dequant-in-shader matmul kernels
//
// Q5_K super-block layout (176 bytes for 256 elements):
//   [  0..  2) : d     (f16)   — super-block scale
//   [  2..  4) : dmin  (f16)   — super-block min
//   [  4.. 16) : 12 bytes of packed 6-bit scales + 6-bit mins (8 sub-blocks),
//                same encoding as Q4_K
//   [ 16.. 48) : 32 bytes of qh — 1 high bit per weight, 8 weights per byte
//   [ 48..176) : 128 bytes of qs — 4-bit low nibbles, 2 weights per byte
//                same layout as Q4_K's qs
//
// Dequantization per element:
//   weight = d_sub * ((qs_nibble | (qh_bit ? 16 : 0))) - min_sub
//
// The 256 elements are grouped into 4 chunks of 64:
//   chunk c ∈ 0..4 covers elements c*64..c*64+63
//   within chunk: positions [c*64 + 0 .. c*64 + 31]   use qs low nibbles
//                 positions [c*64 + 32 .. c*64 + 63]  use qs high nibbles
//   qs offset for this chunk: c * 32  (32 bytes per chunk)
//   qh byte index: same `l` as the element offset within the section
//   qh bit mask:   u1 = 1u << (c*2)     for low-nibble section
//                  u2 = 1u << (c*2 + 1) for high-nibble section
//   sub-block scales: (sc1, m1) at index c*2, (sc2, m2) at index c*2 + 1
//
// Physical weight layout (GGUF): row-major [out, in], each row is
// (in / 256) contiguous 176-byte super-blocks.
//
// Compile with: nvcc -ptx -arch=sm_80 --use_fast_math q5k_matmul.cu -o q5k_matmul.ptx

// Manual f16 → f32 — exact port of nexus-serve's Rust f16_to_f32. Avoids
// <cuda_fp16.h> __half type-punning pitfalls. Handles f16 subnormals.
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
        unsigned int exp32 = (unsigned int)(exp + 112); // 127 - 15
        result = (sign << 31) | (exp32 << 23) | (frac << 13);
    }
    return __int_as_float((int)result);
}

// Decode a Q4_K-style 6-bit scale-min pair at sub-block index j (0..8).
// Identical to Q4_K's get_scale_min_k4 — Q5_K reuses the scale packing.
__device__ __forceinline__ void get_scale_min_k4(
    unsigned int j,
    const unsigned char* __restrict__ q,
    unsigned char* sc,
    unsigned char* m
) {
    if (j < 4) {
        *sc = q[j]     & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// Core per-row dot product. Walks all blocks of a row and returns sum.
// One thread does all the work — used by q5k_gemm_f32. Unrolled over the
// 4 chunks and 2 (lo/hi) halves.
__device__ __forceinline__ float q5k_row_dot(
    const unsigned char* __restrict__ row,
    const float* __restrict__ a_row,
    unsigned int n_blocks
) {
    float sum = 0.0f;
    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + (size_t)b * 176u;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;
        const unsigned char* qh     = block + 16;
        const unsigned char* qs     = block + 48;

        unsigned int a_base = b * 256u;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned char sc1, m1, sc2, m2;
            get_scale_min_k4((unsigned int)(c * 2),     scales, &sc1, &m1);
            get_scale_min_k4((unsigned int)(c * 2 + 1), scales, &sc2, &m2);
            float d1   = d    * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2   = d    * (float)sc2;
            float min2 = dmin * (float)m2;

            unsigned int u1 = 1u << (unsigned int)(c * 2);
            unsigned int u2 = u1 << 1;
            unsigned int ql_off = (unsigned int)c * 32u;
            unsigned int a_off  = a_base + (unsigned int)c * 64u;

            #pragma unroll
            for (unsigned int l = 0; l < 32; ++l) {
                unsigned int qs_byte = qs[ql_off + l];
                unsigned int qh_byte = qh[l];
                unsigned int lo_nib = qs_byte & 0x0Fu;
                unsigned int hi_nib = qs_byte >> 4;
                unsigned int lo_hi  = (qh_byte & u1) ? 16u : 0u;
                unsigned int hi_hi  = (qh_byte & u2) ? 16u : 0u;
                float w_lo = d1 * (float)(lo_nib | lo_hi) - min1;
                float w_hi = d2 * (float)(hi_nib | hi_hi) - min2;
                sum += a_row[a_off + l]      * w_lo;
                sum += a_row[a_off + l + 32] * w_hi;
            }
        }
    }
    return sum;
}

// Q5_K GEMV (warp-cooperative, one warp per output): c = a @ B^T.
//
// Parallelism: one warp (32 lanes) per output element j. Each lane
// processes 8 weights per block — 2 chunks × (1 lo + 1 hi) per block.
// Wait: actually each lane covers exactly 2 weights per chunk × 4 chunks
// = 8 weights per block, matching Q4_K's layout.
//
// Lane l in chunk c touches:
//   - qs[c*32 + l]     → produces lo (low-nibble) and hi (high-nibble) weights
//   - qh[l]            → bits u1 = 1<<(c*2) and u2 = 1<<(c*2+1)
//   - a[block*256 + c*64 + l]      (lo activation)
//   - a[block*256 + c*64 + 32 + l] (hi activation)
//
// Launch: block = 128 threads (4 warps → 4 output rows/CTA),
//         grid  = ((out + 3) / 4).
// Requires in_dim % 256 == 0.
extern "C" __global__ void q5k_gemv_f32(
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

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 176u;
    const unsigned char* row     = w + (size_t)j * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + (size_t)b * 176u;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;
        const unsigned char* qh     = block + 16;
        const unsigned char* qs     = block + 48;

        unsigned int a_base = b * 256u;
        unsigned int qh_byte = qh[lane]; // one byte per lane — reused across 4 chunks

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned char sc1, m1, sc2, m2;
            get_scale_min_k4((unsigned int)(c * 2),     scales, &sc1, &m1);
            get_scale_min_k4((unsigned int)(c * 2 + 1), scales, &sc2, &m2);
            float d1   = d    * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2   = d    * (float)sc2;
            float min2 = dmin * (float)m2;

            unsigned int u1 = 1u << (unsigned int)(c * 2);
            unsigned int u2 = u1 << 1;

            unsigned int qs_byte = qs[(unsigned int)c * 32u + lane];
            unsigned int lo_nib = qs_byte & 0x0Fu;
            unsigned int hi_nib = qs_byte >> 4;
            unsigned int lo_hi  = (qh_byte & u1) ? 16u : 0u;
            unsigned int hi_hi  = (qh_byte & u2) ? 16u : 0u;
            float w_lo = d1 * (float)(lo_nib | lo_hi) - min1;
            float w_hi = d2 * (float)(hi_nib | hi_hi) - min2;

            unsigned int a_off = a_base + (unsigned int)c * 64u;
            sum += a[a_off + lane]      * w_lo;
            sum += a[a_off + lane + 32] * w_hi;
        }
    }

    // Warp reduce.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0) c[j] = sum;
}

// Q5_K GEMM: c = a @ B^T, where a is [m, in] and B is Q5_K [out, in].
//
// Shapes:
//   a : [m, in]          float32 row-major
//   w : [out, in] Q5_K   raw bytes (out * (in/256) * 176 total)
//   c : [m, out]         float32
//
// Parallelism: one thread per output element c[mi, j]. Total = m * out.
// Each thread walks all blocks of row j and reads row mi of a.
// Simple naive extension; the GEMV path (m=1) is the hot decode case
// and uses the warp-cooperative kernel above.
extern "C" __global__ void q5k_gemm_f32(
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

    unsigned int n_blocks = in_dim / 256u;
    unsigned int row_bytes = n_blocks * 176u;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* a_row = a + (size_t)mi * in_dim;

    c[tid] = q5k_row_dot(row, a_row, n_blocks);
}
