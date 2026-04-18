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

// ----------------------------------------------------------------------------
// Warp-cooperative Q4_K GEMV partial sum.
//
// The warp is split into 4 groups of 8 lanes, one group per super-block chunk:
//   chunk       = lane / 8     (0..3)
//   lane_in_ch  = lane & 7     (0..7)
//
// Each lane processes its chunk's 8 weights (4 low nibbles + 4 high nibbles)
// per super-block using a single uint32 load of 4 consecutive qs bytes and
// two float4 loads for the 4 lo activations and 4 hi activations. Eight FMAs
// per lane per block, coalesced 128-byte warp transactions on both qs and a.
//
// Caller passes the block range [b_start, b_end); this lets two warps split
// a row's block range (for cross-warp cooperative reduction).
// ----------------------------------------------------------------------------
__device__ __forceinline__ float q4k_gemv_partial(
    const unsigned char* __restrict__ row,
    const float*         __restrict__ a,
    unsigned int b_start,
    unsigned int b_end,
    unsigned int chunk,
    unsigned int chunk_byte_off,
    unsigned int chunk_a_lo,
    unsigned int chunk_a_hi
) {
    float sum = 0.0f;
    for (unsigned int b = b_start; b < b_end; ++b) {
        const unsigned char* block = row + (size_t)b * 144u;

        // Broadcast header (4 bytes d+dmin, 12 bytes packed scales).
        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;

        // This lane's chunk-specific scale-min pair (sc1/m1 for lo, sc2/m2 for hi).
        unsigned int is = chunk * 2u;
        unsigned char sc1, m1, sc2, m2;
        get_scale_min_k4(is,     scales, &sc1, &m1);
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        float d1   = d    * (float)sc1;
        float min1 = dmin * (float)m1;
        float d2   = d    * (float)sc2;
        float min2 = dmin * (float)m2;

        // 32 lanes × 4 bytes = 128 bytes of qs in one coalesced transaction.
        const unsigned int* qs_u32 =
            (const unsigned int*)(block + 16 + chunk_byte_off);
        unsigned int packed = __ldg(qs_u32);

        // float4 loads: 4 lo activations + 4 hi activations for this lane.
        // chunk_a_lo = chunk*64 + lane_in_ch*4   (always 16-byte aligned)
        // chunk_a_hi = chunk_a_lo + 32           (also 16-byte aligned)
        const float4* a_lo_vec = (const float4*)(a + (size_t)b * 256u + chunk_a_lo);
        const float4* a_hi_vec = (const float4*)(a + (size_t)b * 256u + chunk_a_hi);
        float4 a_lo = __ldg(a_lo_vec);
        float4 a_hi = __ldg(a_hi_vec);

        unsigned int b0 =  packed        & 0xFFu;
        unsigned int b1 = (packed >>  8) & 0xFFu;
        unsigned int b2 = (packed >> 16) & 0xFFu;
        unsigned int b3 = (packed >> 24) & 0xFFu;

        float w_lo0 = d1 * (float)(b0 & 0x0Fu) - min1;
        float w_lo1 = d1 * (float)(b1 & 0x0Fu) - min1;
        float w_lo2 = d1 * (float)(b2 & 0x0Fu) - min1;
        float w_lo3 = d1 * (float)(b3 & 0x0Fu) - min1;

        float w_hi0 = d2 * (float)(b0 >> 4) - min2;
        float w_hi1 = d2 * (float)(b1 >> 4) - min2;
        float w_hi2 = d2 * (float)(b2 >> 4) - min2;
        float w_hi3 = d2 * (float)(b3 >> 4) - min2;

        sum += a_lo.x * w_lo0 + a_lo.y * w_lo1 + a_lo.z * w_lo2 + a_lo.w * w_lo3;
        sum += a_hi.x * w_hi0 + a_hi.y * w_hi1 + a_hi.z * w_hi2 + a_hi.w * w_hi3;
    }
    return sum;
}

// Standard 32-lane reduction: lane 0 ends with sum(warp).
__device__ __forceinline__ float warp_reduce_sum_f32(float sum) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    return sum;
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

// Q4_K GEMV v2 (vectorized + two-warp cooperative): c = a @ B^T.
//
// Two warps cooperate on each output row. Each warp owns half of the
// super-blocks along the in-dim, uses uint32 qs loads (4 bytes per thread
// per block → 8 weights) and float4 activation loads. Intra-warp reduction
// via __shfl_xor_sync, then cross-warp combine through shared memory.
//
// Block: `rows_per_cta` output rows × 2 warps/row × 32 threads = 64 *
// rows_per_cta threads per CTA. The launcher sets rows_per_cta = 4 by
// default → 256 threads/CTA, 4 rows/CTA. Shared memory is
// `rows_per_cta * 2 * sizeof(float)`.
//
// Launch: grid = ((out + rows_per_cta - 1) / rows_per_cta).
//
// Requires in_dim % 256 == 0 (Q4_K block size).
extern "C" __global__ void q4k_gemv_f32(
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

    const unsigned int chunk          = lane >> 3;
    const unsigned int lane_in_ch     = lane & 7u;
    const unsigned int chunk_byte_off = chunk * 32u + lane_in_ch * 4u;
    const unsigned int chunk_a_lo     = chunk * 64u + lane_in_ch * 4u;
    const unsigned int chunk_a_hi     = chunk_a_lo + 32u;

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 144u;

    float sum = 0.0f;
    if (j < out_dim) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        sum = q4k_gemv_partial(
            row, a, b_start, b_end,
            chunk, chunk_byte_off, chunk_a_lo, chunk_a_hi
        );
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0u) {
        s_partial[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < out_dim) {
        c[j] = s_partial[row_in_cta * 2u] + s_partial[row_in_cta * 2u + 1u];
    }
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
// v2 layout: `rows_per_cta` rows × 2 warps/row × 32 threads = 64 *
// rows_per_cta threads/CTA. Each row dispatches to Q / K / V based on its
// global row index. Grid: ceil((q_out+k_out+v_out) / rows_per_cta).
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
    extern __shared__ float s_partial[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int global_row   = blockIdx.x * rows_per_cta + row_in_cta;
    const unsigned int total_out    = q_out + k_out + v_out;

    const unsigned int chunk          = lane >> 3;
    const unsigned int lane_in_ch     = lane & 7u;
    const unsigned int chunk_byte_off = chunk * 32u + lane_in_ch * 4u;
    const unsigned int chunk_a_lo     = chunk * 64u + lane_in_ch * 4u;
    const unsigned int chunk_a_hi     = chunk_a_lo + 32u;

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 144u;

    const unsigned char* w = (const unsigned char*)0;
    float* c = (float*)0;
    unsigned int j = 0u;
    bool have_work = global_row < total_out;
    if (have_work) {
        if (global_row < q_out) {
            w = q_w; c = q_c; j = global_row;
        } else if (global_row < q_out + k_out) {
            w = k_w; c = k_c; j = global_row - q_out;
        } else {
            w = v_w; c = v_c; j = global_row - q_out - k_out;
        }
    }

    float sum = 0.0f;
    if (have_work) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        sum = q4k_gemv_partial(
            row, a, b_start, b_end,
            chunk, chunk_byte_off, chunk_a_lo, chunk_a_hi
        );
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0u) {
        s_partial[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (have_work && warp_in_row == 0u && lane == 0u) {
        c[j] = s_partial[row_in_cta * 2u] + s_partial[row_in_cta * 2u + 1u];
    }
}

// ============================================================================
// Q4_K GEMV (fused QKV + bias): q4k_gemv_fused_qkv_f32 with an extra
// per-section bias add at the output write. Absorbs the three separate
// bias_add kernel launches per layer that Qwen2/DeepSeek require into
// the matmul itself — saves three host→GPU launch cycles per layer.
//
// Bias buffers are mandatory. Callers without biases for a section
// should pass a zero buffer or route through `q4k_gemv_fused_qkv_f32`
// instead; this kernel does NOT test a flag.
//
// Launch geometry identical to q4k_gemv_fused_qkv_f32.
extern "C" __global__ void q4k_gemv_fused_qkv_bias_f32(
    const unsigned char* __restrict__ q_w,
    const unsigned char* __restrict__ k_w,
    const unsigned char* __restrict__ v_w,
    const float*         __restrict__ a,
    const float*         __restrict__ q_bias,
    const float*         __restrict__ k_bias,
    const float*         __restrict__ v_bias,
    float*               __restrict__ q_c,
    float*               __restrict__ k_c,
    float*               __restrict__ v_c,
    unsigned int q_out,
    unsigned int k_out,
    unsigned int v_out,
    unsigned int in_dim
) {
    extern __shared__ float s_partial[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int global_row   = blockIdx.x * rows_per_cta + row_in_cta;
    const unsigned int total_out    = q_out + k_out + v_out;

    const unsigned int chunk          = lane >> 3;
    const unsigned int lane_in_ch     = lane & 7u;
    const unsigned int chunk_byte_off = chunk * 32u + lane_in_ch * 4u;
    const unsigned int chunk_a_lo     = chunk * 64u + lane_in_ch * 4u;
    const unsigned int chunk_a_hi     = chunk_a_lo + 32u;

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 144u;

    // Dispatch: which weight matrix + output buffer + bias + local row j.
    const unsigned char* w      = (const unsigned char*)0;
    float*               c      = (float*)0;
    const float*         bias   = (const float*)0;
    unsigned int         j      = 0u;
    bool have_work = global_row < total_out;
    if (have_work) {
        if (global_row < q_out) {
            w = q_w; c = q_c; bias = q_bias; j = global_row;
        } else if (global_row < q_out + k_out) {
            w = k_w; c = k_c; bias = k_bias; j = global_row - q_out;
        } else {
            w = v_w; c = v_c; bias = v_bias; j = global_row - q_out - k_out;
        }
    }

    float sum = 0.0f;
    if (have_work) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        sum = q4k_gemv_partial(
            row, a, b_start, b_end,
            chunk, chunk_byte_off, chunk_a_lo, chunk_a_hi
        );
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0u) {
        s_partial[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (have_work && warp_in_row == 0u && lane == 0u) {
        float result = s_partial[row_in_cta * 2u] + s_partial[row_in_cta * 2u + 1u];
        result += bias[j];
        c[j] = result;
    }
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
    extern __shared__ float s_partial[];

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int global_row   = blockIdx.x * rows_per_cta + row_in_cta;
    const unsigned int total_out    = inter + inter;

    const unsigned int chunk          = lane >> 3;
    const unsigned int lane_in_ch     = lane & 7u;
    const unsigned int chunk_byte_off = chunk * 32u + lane_in_ch * 4u;
    const unsigned int chunk_a_lo     = chunk * 64u + lane_in_ch * 4u;
    const unsigned int chunk_a_hi     = chunk_a_lo + 32u;

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 144u;

    const unsigned char* w = (const unsigned char*)0;
    float* c = (float*)0;
    unsigned int j = 0u;
    bool have_work = global_row < total_out;
    if (have_work) {
        if (global_row < inter) {
            w = gate_w; c = gate_c; j = global_row;
        } else {
            w = up_w;   c = up_c;   j = global_row - inter;
        }
    }

    float sum = 0.0f;
    if (have_work) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        sum = q4k_gemv_partial(
            row, a, b_start, b_end,
            chunk, chunk_byte_off, chunk_a_lo, chunk_a_hi
        );
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0u) {
        s_partial[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (have_work && warp_in_row == 0u && lane == 0u) {
        c[j] = s_partial[row_in_cta * 2u] + s_partial[row_in_cta * 2u + 1u];
    }
}

// ============================================================================
// Q4_K GEMV (fused residual): x_out[j] = x_in[j] + matmul(a, w_j). Collapses
// the matmul + residual_add kernel pair into a single launch and eliminates
// the temporary output buffer round trip. `x_in` and `x_out` may be the same
// pointer (in-place residual) — the read happens before the accumulator write.
//
// Same warp geometry as q4k_gemv_f32: 2 warps per output row, four chunks per
// warp, vectorized qs (uint32) + activation (float4) loads.
// ============================================================================
extern "C" __global__ void q4k_gemv_residual_f32(
    const unsigned char* __restrict__ w,
    const float*         __restrict__ a,
    const float*         __restrict__ x_in,
    float*               __restrict__ x_out,
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

    const unsigned int chunk          = lane >> 3;
    const unsigned int lane_in_ch     = lane & 7u;
    const unsigned int chunk_byte_off = chunk * 32u + lane_in_ch * 4u;
    const unsigned int chunk_a_lo     = chunk * 64u + lane_in_ch * 4u;
    const unsigned int chunk_a_hi     = chunk_a_lo + 32u;

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 144u;

    float sum = 0.0f;
    if (j < out_dim) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;
        sum = q4k_gemv_partial(
            row, a, b_start, b_end,
            chunk, chunk_byte_off, chunk_a_lo, chunk_a_hi
        );
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0u) {
        s_partial[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < out_dim) {
        float proj = s_partial[row_in_cta * 2u] + s_partial[row_in_cta * 2u + 1u];
        x_out[j] = x_in[j] + proj;
    }
}

// ============================================================================
// Q4_K GEMV (fused gate/up + SwiGLU): produces ffn[j] = silu(gate_row[j] · a)
// * (up_row[j] · a) directly, eliminating the gate_c/up_c intermediate
// buffers and the SwiGLU kernel launch.
//
// Layout: each CTA handles `rows_per_cta` output rows (j values). For each j,
// two warps compute gate[j] (each handles half the super-blocks) and two
// more warps compute up[j] the same way. Partial sums land in shared memory;
// warp 0, lane 0 in each row combines them and writes silu(gate)*up to ffn[j].
//
// Launch geometry: 4 warps per output row × 32 threads × rows_per_cta =
// 128 * rows_per_cta threads/CTA. Shared mem: rows_per_cta * 4 floats.
// Grid: ceil(inter / rows_per_cta).
// ============================================================================
extern "C" __global__ void q4k_gemv_fused_gate_up_swiglu_f32(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const float*         __restrict__ a,
    float*               __restrict__ ffn,
    unsigned int inter,
    unsigned int in_dim
) {
    extern __shared__ float s_partial[];

    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & 31u;
    const unsigned int warp_id       = tid >> 5;
    const unsigned int warps_per_row = 4u;  // 2 for gate, 2 for up
    const unsigned int row_in_cta    = warp_id / warps_per_row;
    const unsigned int warp_in_row   = warp_id & 3u;         // 0..3
    const unsigned int is_up         = warp_in_row >> 1;     // 0 = gate, 1 = up
    const unsigned int half_sel      = warp_in_row & 1u;     // 0 = first half, 1 = second half
    const unsigned int rows_per_cta  = blockDim.x / (warps_per_row * 32u);
    const unsigned int j             = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int chunk          = lane >> 3;
    const unsigned int lane_in_ch     = lane & 7u;
    const unsigned int chunk_byte_off = chunk * 32u + lane_in_ch * 4u;
    const unsigned int chunk_a_lo     = chunk * 64u + lane_in_ch * 4u;
    const unsigned int chunk_a_hi     = chunk_a_lo + 32u;

    const unsigned int n_blocks  = in_dim / 256u;
    const unsigned int row_bytes = n_blocks * 144u;

    float sum = 0.0f;
    if (j < inter) {
        const unsigned char* w = is_up ? up_w : gate_w;
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = half_sel ? half : 0u;
        const unsigned int b_end   = half_sel ? n_blocks : half;
        sum = q4k_gemv_partial(
            row, a, b_start, b_end,
            chunk, chunk_byte_off, chunk_a_lo, chunk_a_hi
        );
    }
    sum = warp_reduce_sum_f32(sum);

    // Each row has 4 partial-sum slots: [gate_lo, gate_hi, up_lo, up_hi].
    if (lane == 0u) {
        s_partial[row_in_cta * 4u + warp_in_row] = sum;
    }
    __syncthreads();

    if (j < inter && warp_in_row == 0u && lane == 0u) {
        float gate_val = s_partial[row_in_cta * 4u + 0u] + s_partial[row_in_cta * 4u + 1u];
        float up_val   = s_partial[row_in_cta * 4u + 2u] + s_partial[row_in_cta * 4u + 3u];
        float silu_g = gate_val / (1.0f + __expf(-gate_val));
        ffn[j] = silu_g * up_val;
    }
}
