// AxonML CUDA Q4_K dequant-in-shader matmul kernels
//
// Q4_K super-block layout (144 bytes for 256 elements):
//   [  0..  2) : d     (f16) — super-block scale
//   [  2..  4) : dmin  (f16) — super-block min
//   [  4.. 16) : 12 bytes of packed 6-bit scales + 6-bit mins (8 sub-blocks)
//   [ 16..144) : 128 bytes of 4-bit quantized values (packed 2 per byte)
//
// The weight matrix B has physical shape [out, in] row-major. GEMV computes
// c[j] = sum_k a[k] * B[j, k] for each j in [0, out). Each row of B is laid
// out as (in/256) contiguous 144-byte blocks. The byte offset to block b of
// row j is (j * (in/256) + b) * 144.
//
// Compile with: nvcc -ptx -arch=sm_80 --use_fast_math q4k_matmul.cu -o q4k_matmul.ptx

// Manual f16 → f32 (no <cuda_fp16.h> to avoid any __half type-punning pitfalls).
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
            // Infinity
            result = (sign << 31) | (0xFFu << 23);
        } else {
            // NaN
            result = 0x7FC00000u;
        }
    } else {
        unsigned int exp32 = (unsigned int)(exp + 112); // 127 - 15
        result = (sign << 31) | (exp32 << 23) | (frac << 13);
    }
    return __int_as_float((int)result);
}

// Decode a Q4_K 6-bit scale-min pair at sub-block index j (0..8).
// Exact port of the Rust `get_scale_min_k4`.
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

// Q4_K GEMM: c = a @ B^T, where a is [m, in] and B is Q4_K-quantized weight
// with physical shape [out, in] row-major.
//
// Shapes:
//   a : [m, in]          float32, row-major contiguous
//   w : [out, in] Q4_K   raw quantized bytes (out * in / 256 * 144 bytes)
//   c : [m, out]         float32
//
// Parallelism: one thread per output element c[mi, j]. Total threads = m * out.
// Each thread dequants the (in / 256) blocks of row j of w in registers and
// accumulates against the mi-th row of a.
//
// This is the naive-but-correct extension of q4k_gemv_f32 to m > 1. Perf
// optimizations (shared-memory dequant cache, cooperative threads per
// output) are a future session.
extern "C" __global__ void q4k_gemm_f32(
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
    unsigned int row_bytes = n_blocks * 144;
    const unsigned char* row = w + (size_t)j * row_bytes;
    const float* a_row = a + (size_t)mi * in_dim;

    float sum = 0.0f;

    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + b * 144;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;
        const unsigned char* qs     = block + 16;

        unsigned int is = 0;
        unsigned int q_offset = 0;
        unsigned int a_offset = b * 256;

        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            unsigned char sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);

            float d1   = d    * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2   = d    * (float)sc2;
            float min2 = dmin * (float)m2;

            #pragma unroll
            for (unsigned int l = 0; l < 32; ++l) {
                unsigned int q = qs[q_offset + l] & 0x0F;
                float wv = d1 * (float)q - min1;
                sum += a_row[a_offset + l] * wv;
            }
            #pragma unroll
            for (unsigned int l = 0; l < 32; ++l) {
                unsigned int q = (qs[q_offset + l] >> 4) & 0x0F;
                float wv = d2 * (float)q - min2;
                sum += a_row[a_offset + 32 + l] * wv;
            }

            q_offset += 32;
            a_offset += 64;
            is += 2;
        }
    }

    c[tid] = sum;
}

// Q4_K GEMV (cooperative warp reduction): c = a @ B^T.
//
// One WARP (32 threads) owns one output element. Within the warp, all 32
// threads cooperate on every super-block: each lane handles exactly 8 of
// the 256 weights in the block (2 per chunk × 4 chunks × 1 lane slot).
//
//   lane l, chunk c (0..4):
//     - reads qs[chunk*32 + l]   → low nibble (weight at offset chunk*64 + l)
//                                 and high nibble (offset chunk*64 + l + 32)
//     - reads a[block*256 + chunk*64 + l] and a[block*256 + chunk*64 + l + 32]
//     - accumulates two FMAs
//
// Memory pattern: across one warp and one chunk, the 32 lanes read 32
// consecutive bytes of qs (coalesced 128-byte transaction), and the two
// reads of `a` are 32-element coalesced slices. The per-block header
// (d, dmin, scales) is identical across lanes — broadcast.
//
// After walking all (in/256) blocks, threads hold partial sums; a 5-step
// __shfl_xor_sync reduction collapses to lane 0, which writes c[row].
//
// Block: 4 warps × 32 threads = 128 threads per CTA → 4 output rows per CTA.
// Launch: grid = ((out + 3) / 4), block = 128.
extern "C" __global__ void q4k_gemv_f32(
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
    const unsigned int row_bytes = n_blocks * 144;
    const unsigned char* row = w + (size_t)j * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + b * 144;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;
        const unsigned char* qs     = block + 16;

        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            unsigned int is = (unsigned int)(chunk * 2);

            unsigned char sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1   = d    * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2   = d    * (float)sc2;
            float min2 = dmin * (float)m2;

            unsigned int q_base = (unsigned int)chunk * 32u + lane;
            unsigned int a_base = b * 256u + (unsigned int)chunk * 64u + lane;

            unsigned int byte = qs[q_base];
            float w_lo = d1 * (float)(byte & 0x0Fu) - min1;
            float w_hi = d2 * (float)(byte >> 4)    - min2;

            sum += a[a_base]      * w_lo;
            sum += a[a_base + 32] * w_hi;
        }
    }

    // Warp reduction: 32 → 16 → 8 → 4 → 2 → 1.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0) c[j] = sum;
}

// ============================================================================
// Q4_K GEMV (fused QKV): one kernel launch produces Q, K, V projections.
//
// Q, K, V all share the same activation input `a` ([1, in_dim]) but each has
// its own Q4_K weight blob and output dimension. In the unfused path we
// launch three separate `q4k_gemv_f32` kernels back-to-back. For GQA where
// k_out = v_out << q_out (e.g. Qwen 7B has q_out=3584, k/v_out=512) the
// smaller kernels still pay a full launch + sync overhead and actually
// dominate their own runtime. Fusing collapses all three into a single grid.
//
// Layout in the global warp index space:
//   warp_id ∈ [0, q_out)                            → Q output row
//   warp_id ∈ [q_out, q_out + k_out)                → K output row (row = idx - q_out)
//   warp_id ∈ [q_out + k_out, q_out + k_out + v_out) → V output row
//
// Block: 4 warps × 32 threads = 128 threads → 4 output rows per CTA.
// Grid: ceil((q_out + k_out + v_out) / 4).
extern "C" __global__ void q4k_gemv_fused_qkv_f32(
    const unsigned char* __restrict__ q_w,
    const unsigned char* __restrict__ k_w,
    const unsigned char* __restrict__ v_w,
    const float* __restrict__ a,
    float* __restrict__ q_c,
    float* __restrict__ k_c,
    float* __restrict__ v_c,
    unsigned int q_out,
    unsigned int k_out,
    unsigned int v_out,
    unsigned int in_dim
) {
    const unsigned int tid     = threadIdx.x;
    const unsigned int lane    = tid & 31u;
    const unsigned int warp_id = tid >> 5;
    const unsigned int global_warp = blockIdx.x * (blockDim.x >> 5) + warp_id;
    const unsigned int total_out = q_out + k_out + v_out;
    if (global_warp >= total_out) return;

    // Dispatch to Q / K / V lane based on the global warp index.
    const unsigned char* w;
    float* c;
    unsigned int j;
    if (global_warp < q_out) {
        w = q_w;
        c = q_c;
        j = global_warp;
    } else if (global_warp < q_out + k_out) {
        w = k_w;
        c = k_c;
        j = global_warp - q_out;
    } else {
        w = v_w;
        c = v_c;
        j = global_warp - q_out - k_out;
    }

    const unsigned int n_blocks = in_dim / 256;
    const unsigned int row_bytes = n_blocks * 144;
    const unsigned char* row = w + (size_t)j * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + b * 144;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;
        const unsigned char* qs     = block + 16;

        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            unsigned int is = (unsigned int)(chunk * 2);

            unsigned char sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1   = d    * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2   = d    * (float)sc2;
            float min2 = dmin * (float)m2;

            unsigned int q_base = (unsigned int)chunk * 32u + lane;
            unsigned int a_base = b * 256u + (unsigned int)chunk * 64u + lane;

            unsigned int byte = qs[q_base];
            float w_lo = d1 * (float)(byte & 0x0Fu) - min1;
            float w_hi = d2 * (float)(byte >> 4)    - min2;

            sum += a[a_base]      * w_lo;
            sum += a[a_base + 32] * w_hi;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0) c[j] = sum;
}

// ============================================================================
// Q4_K GEMV (fused gate/up): one kernel launch produces gate and up
// projections of the SwiGLU / ReLU² FFN. Same input, two different
// weight matrices, same output dim (intermediate_size). Collapses the
// gate and up kernel launches into a single grid.
//
// Layout in global warp index space:
//   warp_id ∈ [0, inter)        → gate output row
//   warp_id ∈ [inter, 2*inter)  → up   output row (row = idx - inter)
// ============================================================================
extern "C" __global__ void q4k_gemv_fused_gate_up_f32(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const float* __restrict__ a,
    float* __restrict__ gate_c,
    float* __restrict__ up_c,
    unsigned int inter,
    unsigned int in_dim
) {
    const unsigned int tid     = threadIdx.x;
    const unsigned int lane    = tid & 31u;
    const unsigned int warp_id = tid >> 5;
    const unsigned int global_warp = blockIdx.x * (blockDim.x >> 5) + warp_id;
    const unsigned int total_out = inter + inter;
    if (global_warp >= total_out) return;

    const unsigned char* w;
    float* c;
    unsigned int j;
    if (global_warp < inter) {
        w = gate_w;
        c = gate_c;
        j = global_warp;
    } else {
        w = up_w;
        c = up_c;
        j = global_warp - inter;
    }

    const unsigned int n_blocks = in_dim / 256;
    const unsigned int row_bytes = n_blocks * 144;
    const unsigned char* row = w + (size_t)j * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = 0; b < n_blocks; ++b) {
        const unsigned char* block = row + b * 144;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;
        const unsigned char* qs     = block + 16;

        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            unsigned int is = (unsigned int)(chunk * 2);

            unsigned char sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1   = d    * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2   = d    * (float)sc2;
            float min2 = dmin * (float)m2;

            unsigned int q_base = (unsigned int)chunk * 32u + lane;
            unsigned int a_base = b * 256u + (unsigned int)chunk * 64u + lane;

            unsigned int byte = qs[q_base];
            float w_lo = d1 * (float)(byte & 0x0Fu) - min1;
            float w_hi = d2 * (float)(byte >> 4)    - min2;

            sum += a[a_base]      * w_lo;
            sum += a[a_base + 32] * w_hi;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0) c[j] = sum;
}
