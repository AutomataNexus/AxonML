// AxonML CUDA Q1_0 (1-bit) FUSED matmul — single-launch DP4A path.
//
// Combines the activation-quant + gemv steps of `q1_0_matmul_dp4a.cu`
// into ONE kernel. Each CTA cooperatively quantizes the full f32
// activation row into shared memory (int8 + fp16 per-32-chunk scales)
// once, then all warps in the CTA do the dp4a gemv against the
// smem-resident acts. Eliminates the second kernel launch + the
// global int8/fp16 scratch buffers that made the standalone DP4A path
// lose ~12 % to v2 on launch-overhead-bound decode.
//
// Why smem-resident acts work despite n_CTAs reading the same f32
// activation row from global: L2 absorbs it. CTA 0 reads `a[0..k)`
// from DRAM into L2, every subsequent CTA hits L2. With k=4096 the
// activation row is 16 KB — far below the laptop GPU's L2 capacity.
//
// Smem footprint: k bytes int8 + (k/32) * 2 bytes fp16 = k + k/16 B.
// Bonsai-8B worst case is the FFN down projection: k=14336 → 15.25 KB
// per CTA. RTX 5070 Ti Laptop default smem is 48 KB → 3 CTAs/SM cap
// from smem alone, plenty of room.
//
// Compile: nvcc -ptx -arch=sm_89 --use_fast_math q1_0_matmul_fused.cu \
//                  -o q1_0_matmul_fused.ptx

#include <cuda_fp16.h>

#define Q1_0_BLOCK_SIZE      128u
#define Q1_0_BYTES_PER_BLOCK 18u
#define Q8_CHUNK             32u

extern "C" __global__ void q1_0_gemv_fused_dp4a_f32(
    const unsigned char* __restrict__ w,  // [n, n_blocks * 18] Q1_0 bytes
    const float* __restrict__ a,          // [k] f32 activations
    float* __restrict__ c,                // [n] output
    unsigned int n,
    unsigned int k
) {
    extern __shared__ unsigned char smem[];
    // Layout in smem:
    //   [0 .. k)              int8 quantized acts (as unsigned char; reinterpret)
    //   [k .. k + (k/32)*2)   fp16 per-chunk scales
    //   [tail]                 partials buffer for the warp reduce
    signed char* a_q = reinterpret_cast<signed char*>(smem);
    __half* a_d      = reinterpret_cast<__half*>(smem + k);

    const unsigned int tid          = threadIdx.x;
    const unsigned int lane         = tid & 31u;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int row_in_cta   = warp_id >> 1;
    const unsigned int warp_in_row  = warp_id & 1u;
    const unsigned int rows_per_cta = blockDim.x >> 6;
    const unsigned int warps_per_cta= blockDim.x >> 5;
    const unsigned int j            = blockIdx.x * rows_per_cta + row_in_cta;

    const unsigned int n_blocks  = k / Q1_0_BLOCK_SIZE;
    const unsigned int n_chunks  = k / Q8_CHUNK;
    const unsigned int row_bytes = n_blocks * Q1_0_BYTES_PER_BLOCK;

    // ===== Step 1: cooperative activation quantization into smem =====
    //
    // Each warp processes one Q8 chunk per pass. 8 warps per CTA × n_chunks/8
    // passes covers all chunks. For k=4096, n_chunks=128, 16 passes per warp.

    for (unsigned int chunk = warp_id; chunk < n_chunks; chunk += warps_per_cta) {
        const unsigned int idx = chunk * Q8_CHUNK + lane;
        float v = a[idx];
        float amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        float d = amax / 127.0f;
        float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = __float2int_rn(v * inv_d);
        q = max(-127, min(127, q));
        a_q[idx] = (signed char)q;
        if (lane == 0u) {
            a_d[chunk] = __float2half(d);
        }
    }
    __syncthreads();

    // Partial buffer (sum-per-warp-per-row) starts after the activation
    // tile, naturally aligned because k is a multiple of 128 → 4-byte
    // aligned scale region → fp16 is 2-byte aligned, partials follow.
    float* partials = reinterpret_cast<float*>(smem + k + n_chunks * sizeof(__half));

    // ===== Step 2: dp4a gemv against smem-resident acts =====

    const unsigned int byte_idx  = lane >> 1;        // qs byte index = l/2
    const unsigned int nibble_sh = (lane & 1u) << 2; // 0 (lo) or 4 (hi) nibble
    const unsigned int chunk_off = lane >> 3;        // which Q8 chunk in the block

    float sum = 0.0f;
    if (j < n) {
        const unsigned char* row = w + (size_t)j * row_bytes;
        const unsigned int half = n_blocks >> 1;
        const unsigned int b_start = warp_in_row ? half : 0u;
        const unsigned int b_end   = warp_in_row ? n_blocks : half;

        for (unsigned int b = b_start; b < b_end; ++b) {
            const unsigned char* block = row + (size_t)b * Q1_0_BYTES_PER_BLOCK;
            __half d_w_h = *reinterpret_cast<const __half*>(block);
            float d_w = __half2float(d_w_h);
            const unsigned char* qs = block + 2;

            unsigned int nibble = ((unsigned int)qs[byte_idx] >> nibble_sh) & 0xFu;

            int b0 = (nibble & 0x1u) ? 1 : -1;
            int b1 = (nibble & 0x2u) ? 1 : -1;
            int b2 = (nibble & 0x4u) ? 1 : -1;
            int b3 = (nibble & 0x8u) ? 1 : -1;
            int w_int = (b0 & 0xFF)
                      | ((b1 & 0xFF) << 8)
                      | ((b2 & 0xFF) << 16)
                      | ((b3 & 0xFF) << 24);

            int a_int = *reinterpret_cast<const int*>(a_q + b * Q1_0_BLOCK_SIZE + lane * 4u);

            int sumi = __dp4a(w_int, a_int, 0);

            __half d_a_h = a_d[b * 4u + chunk_off];
            float d_a = __half2float(d_a_h);

            sum = fmaf((float)sumi * d_a, d_w, sum);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }

    if (lane == 0u) {
        partials[row_in_cta * 2u + warp_in_row] = sum;
    }
    __syncthreads();

    if (warp_in_row == 0u && lane == 0u && j < n) {
        float combined = partials[row_in_cta * 2u]
                       + partials[row_in_cta * 2u + 1u];
        c[j] = combined;
    }
}
