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

// ----------------------------------------------------------------------------
// Q5_K GEMV v2 — vectorized loads + two-warp cooperative reduction.
//
// Structural upgrade from the v1 kernel (one-warp-per-row, byte-at-a-time
// reads): mirrors the Q4_K GEMV v2 layout so qs reads are coalesced
// uint32 loads, activation reads are float4, and two warps cooperate on
// each output row through shared memory.
//
// Per block (256 weights):
//   - 4 chunks × 64 elements (same as v1)
//   - Lane layout: `chunk = lane / 8`, `lane_in_ch = lane & 7`
//   - Each lane reads 4 qs bytes as uint32 (128-byte coalesced warp
//     transaction on qs), 4 qh bytes as uint32 (128-byte coalesced on
//     qh), and 4 lo + 4 hi activations via float4.
//   - 8 FMAs per lane per block (4 lo nibbles + 4 hi nibbles).
//
// Block: `rows_per_cta` output rows × 2 warps/row × 32 threads =
// 64 * rows_per_cta threads/CTA. Launcher uses rows_per_cta = 4
// → 256 threads/CTA, 4 rows/CTA. Shared memory:
// `rows_per_cta * 2 * sizeof(float)`.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float q5k_gemv_partial(
    const unsigned char* __restrict__ row,
    const float*         __restrict__ a,
    unsigned int b_start,
    unsigned int b_end,
    unsigned int chunk,
    unsigned int chunk_byte_off,   // chunk * 32 + lane_in_ch * 4  (for qs)
    unsigned int chunk_qh_byte_off,// lane_in_ch * 4               (for qh)
    unsigned int chunk_a_lo,
    unsigned int chunk_a_hi
) {
    float sum = 0.0f;
    for (unsigned int b = b_start; b < b_end; ++b) {
        const unsigned char* block = row + (size_t)b * 176u;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;

        // This lane's chunk scale/min pair (sc1/m1 for lo, sc2/m2 for hi).
        unsigned int is = chunk * 2u;
        unsigned char sc1, m1, sc2, m2;
        get_scale_min_k4(is,     scales, &sc1, &m1);
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        float d1   = d    * (float)sc1;
        float min1 = dmin * (float)m1;
        float d2   = d    * (float)sc2;
        float min2 = dmin * (float)m2;

        // Bit masks for this chunk's qh — u1 for lo half, u2 for hi half.
        const unsigned int u1 = 1u << (chunk * 2u);
        const unsigned int u2 = u1 << 1;

        // Coalesced uint32 loads — 4 bytes per lane = 128 bytes per warp.
        const unsigned int* qs_u32 =
            (const unsigned int*)(block + 48u + chunk_byte_off);
        const unsigned int* qh_u32 =
            (const unsigned int*)(block + 16u + chunk_qh_byte_off);
        unsigned int qs_packed = __ldg(qs_u32);
        unsigned int qh_packed = __ldg(qh_u32);

        // float4 loads for the 4 lo + 4 hi activations this lane needs.
        const float4* a_lo_vec = (const float4*)(a + (size_t)b * 256u + chunk_a_lo);
        const float4* a_hi_vec = (const float4*)(a + (size_t)b * 256u + chunk_a_hi);
        float4 a_lo = __ldg(a_lo_vec);
        float4 a_hi = __ldg(a_hi_vec);

        unsigned int qs0 =  qs_packed        & 0xFFu;
        unsigned int qs1 = (qs_packed >>  8) & 0xFFu;
        unsigned int qs2 = (qs_packed >> 16) & 0xFFu;
        unsigned int qs3 = (qs_packed >> 24) & 0xFFu;

        unsigned int qh0 =  qh_packed        & 0xFFu;
        unsigned int qh1 = (qh_packed >>  8) & 0xFFu;
        unsigned int qh2 = (qh_packed >> 16) & 0xFFu;
        unsigned int qh3 = (qh_packed >> 24) & 0xFFu;

        // Lo-nibble weights: d1 * ((qs & 0x0F) | (qh_bit ? 16)) - min1
        float w_lo0 = d1 * (float)((qs0 & 0x0Fu) | ((qh0 & u1) ? 16u : 0u)) - min1;
        float w_lo1 = d1 * (float)((qs1 & 0x0Fu) | ((qh1 & u1) ? 16u : 0u)) - min1;
        float w_lo2 = d1 * (float)((qs2 & 0x0Fu) | ((qh2 & u1) ? 16u : 0u)) - min1;
        float w_lo3 = d1 * (float)((qs3 & 0x0Fu) | ((qh3 & u1) ? 16u : 0u)) - min1;

        // Hi-nibble weights.
        float w_hi0 = d2 * (float)((qs0 >> 4) | ((qh0 & u2) ? 16u : 0u)) - min2;
        float w_hi1 = d2 * (float)((qs1 >> 4) | ((qh1 & u2) ? 16u : 0u)) - min2;
        float w_hi2 = d2 * (float)((qs2 >> 4) | ((qh2 & u2) ? 16u : 0u)) - min2;
        float w_hi3 = d2 * (float)((qs3 >> 4) | ((qh3 & u2) ? 16u : 0u)) - min2;

        sum += a_lo.x * w_lo0 + a_lo.y * w_lo1 + a_lo.z * w_lo2 + a_lo.w * w_lo3;
        sum += a_hi.x * w_hi0 + a_hi.y * w_hi1 + a_hi.z * w_hi2 + a_hi.w * w_hi3;
    }
    return sum;
}

extern "C" __global__ void q5k_gemv_f32(
    const unsigned char* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    unsigned int out_dim,
    unsigned int in_dim
) {
    extern __shared__ float s_partial[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int chunk            = lane >> 3;
    const unsigned int lane_in_ch       = lane & 7u;
    const unsigned int chunk_byte_off   = chunk * 32u + lane_in_ch * 4u;
    const unsigned int chunk_qh_byte_off = lane_in_ch * 4u;
    const unsigned int chunk_a_lo       = chunk * 64u + lane_in_ch * 4u;
    const unsigned int chunk_a_hi       = chunk_a_lo + 32u;

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 176u;

    float sum = 0.0f;
    if (j < out_dim) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        sum = q5k_gemv_partial(
            row, a, b_start, b_end,
            chunk, chunk_byte_off, chunk_qh_byte_off,
            chunk_a_lo, chunk_a_hi
        );
    }
    // Intra-warp reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        s_partial[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < out_dim) {
        c[j] = s_partial[row_in_cta * 2u] + s_partial[row_in_cta * 2u + 1u];
    }
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
