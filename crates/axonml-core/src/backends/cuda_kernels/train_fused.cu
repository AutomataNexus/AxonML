// train_fused.cu — multi-tensor training kernels: one launch covers every parameter.
// A 250M-param step issues ~150 tensors x (clip + quantize) launches; at ~40 us of host overhead each
// that is most of the step. Pointer tables move the loop onto the device.
extern "C" __global__ void mt_sumsq_f32(
    const float* const* __restrict__ ptrs,
    const unsigned int* __restrict__ lens,
    const unsigned int* __restrict__ block_tensor,   // tensor index for each block
    const unsigned int* __restrict__ block_offset,   // element offset of each block inside its tensor
    float* __restrict__ out                          // caller zeroes; single accumulator
) {
    extern __shared__ float smem[];
    unsigned int t = block_tensor[blockIdx.x];
    unsigned int base = block_offset[blockIdx.x];
    unsigned int n = lens[t];
    const float* x = ptrs[t];
    unsigned int i = base + threadIdx.x;
    float v = 0.0f;
    if (i < n) { float a = x[i]; v = a * a; }
    if (i + blockDim.x < n && i + blockDim.x < base + 2u * blockDim.x) { float a = x[i + blockDim.x]; v += a * a; }
    unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0u) smem[warp] = v;
    __syncthreads();
    if (warp == 0u) {
        unsigned int nw = blockDim.x >> 5;
        v = (threadIdx.x < nw) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0u) atomicAdd(out, v);
    }
}

extern "C" __global__ void mt_abssum_f32(
    const float* const* __restrict__ ptrs,
    const unsigned int* __restrict__ lens,
    const unsigned int* __restrict__ block_tensor,
    const unsigned int* __restrict__ block_offset,
    float* __restrict__ sums                         // one per tensor, caller zeroes
) {
    extern __shared__ float smem[];
    unsigned int t = block_tensor[blockIdx.x];
    unsigned int base = block_offset[blockIdx.x];
    unsigned int n = lens[t];
    const float* x = ptrs[t];
    unsigned int i = base + threadIdx.x;
    float v = 0.0f;
    if (i < n) v = fabsf(x[i]);
    if (i + blockDim.x < n && i + blockDim.x < base + 2u * blockDim.x) v += fabsf(x[i + blockDim.x]);
    unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0u) smem[warp] = v;
    __syncthreads();
    if (warp == 0u) {
        unsigned int nw = blockDim.x >> 5;
        v = (threadIdx.x < nw) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0u) atomicAdd(&sums[t], v);
    }
}

// ternary STE forward for every projection at once: q = clamp(round(w / absmean), -1, 1) * absmean
extern "C" __global__ void mt_ternarize_f32(
    const float* const* __restrict__ src,
    float* const* __restrict__ dst,
    const unsigned int* __restrict__ lens,
    const unsigned int* __restrict__ block_tensor,
    const unsigned int* __restrict__ block_offset,
    const float* __restrict__ abssums
) {
    unsigned int t = block_tensor[blockIdx.x];
    unsigned int base = block_offset[blockIdx.x];
    unsigned int n = lens[t];
    float a = abssums[t] / (float)n + 1e-8f;
    const float* x = src[t];
    float* y = dst[t];
    for (unsigned int k = 0; k < 2u; ++k) {
        unsigned int i = base + threadIdx.x + k * blockDim.x;
        if (i >= n || i >= base + 2u * blockDim.x) break;
        float v = x[i] / a;
        float r = rintf(v);
        r = fminf(fmaxf(r, -1.0f), 1.0f);
        y[i] = r * a;
    }
}

// scale every gradient in place by one factor (grad clipping), one launch
extern "C" __global__ void mt_scale_f32(
    float* const* __restrict__ ptrs,
    const unsigned int* __restrict__ lens,
    const unsigned int* __restrict__ block_tensor,
    const unsigned int* __restrict__ block_offset,
    const float* __restrict__ factor                 // device scalar
) {
    unsigned int t = block_tensor[blockIdx.x];
    unsigned int base = block_offset[blockIdx.x];
    unsigned int n = lens[t];
    float c = factor[0];
    float* x = ptrs[t];
    for (unsigned int k = 0; k < 2u; ++k) {
        unsigned int i = base + threadIdx.x + k * blockDim.x;
        if (i >= n || i >= base + 2u * blockDim.x) break;
        x[i] *= c;
    }
}

// ── trainable RMSNorm scale: dL/dw_j = sum_i g[i,j] * x[i,j] / rms_i ──
// per-row 1/rms, one block per row; the same reduction and rsqrtf as rms_norm_batched_f32 so x_hat matches the forward
extern "C" __global__ void rms_inv_rows_f32(
    float* __restrict__ inv_rms,
    const float* __restrict__ x,
    unsigned int n,
    float eps
) {
    extern __shared__ float smem[];
    const float* row = x + (size_t)blockIdx.x * (size_t)n;
    float v = 0.0f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) { float a = row[i]; v += a * a; }
    unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0u) smem[warp] = v;
    __syncthreads();
    if (warp == 0u) {
        unsigned int nw = (blockDim.x + 31u) >> 5;
        v = (lane < nw) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0u) inv_rms[blockIdx.x] = rsqrtf(v / (float)n + eps);
    }
}

// column partial sums over a row slice: grid (ceil(n/block), splits); partial is [splits, n], reduced by the caller
extern "C" __global__ void rms_norm_bwd_weight_partial_f32(
    float* __restrict__ partial,
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ inv_rms,
    unsigned int m,
    unsigned int n,
    unsigned int rows_per_split
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int r0 = blockIdx.y * rows_per_split;
    unsigned int r1 = r0 + rows_per_split; if (r1 > m) r1 = m;
    float acc = 0.0f;
    for (unsigned int r = r0; r < r1; ++r) {
        size_t idx = (size_t)r * (size_t)n + j;
        acc += (x[idx] * inv_rms[r]) * grad_out[idx];
    }
    partial[(size_t)blockIdx.y * (size_t)n + j] = acc;
}
