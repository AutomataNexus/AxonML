// Batched Conv2d CUDA Kernels - whole-batch im2col / col2im + grad-weight batch reduction
//
// File: crates/axonml-core/src/backends/cuda_kernels/conv_batched.cu
// Author: Andrew Jewell Sr - AutomataNexus
//
// The original im2col_f32 / col2im_f32 (CONV_PTX in mod.rs) operate on ONE batch element, so
// conv2d_forward_cuda / conv2d_backward_cuda drove them from a `for b in 0..batch_size` loop:
// a batch-64 conv issued ~384 kernel launches instead of ~3, which dominated step time on small
// models (launch-bound, GPU ~25% utilised). These variants carry the batch dimension inside the
// kernel so the whole batch is one launch, and the per-batch GEMMs collapse into one
// cublasSgemmStridedBatched call.
//
// Column layout is UNCHANGED per batch element - [col_h, col_w] contiguous, batch stride col_n -
// so the batched GEMM addresses it with stride_a = col_n and the numerics are identical.
//
// params: u32[11] = { H, W, kH, kW, pH, pW, sH, sW, oH, oW, C_in }
//   (the single-image kernels take u32[10]; C_in is appended so the batch stride C_in*H*W and the
//    per-batch column block col_n = C_in*kH*kW*oH*oW can both be derived inside the kernel)

extern "C" {

// =============================================================================
// im2col_batched_f32
// =============================================================================
// input:  [batch, C_in, H, W]
// col:    [batch, C_in*kH*kW, oH*oW]
// n:      batch * C_in*kH*kW * oH*oW
__global__ void im2col_batched_f32(
    const float* __restrict__ input,
    float* __restrict__ col,
    const unsigned int* __restrict__ params,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned int H   = params[0];
    const unsigned int W   = params[1];
    const unsigned int kH  = params[2];
    const unsigned int kW  = params[3];
    const unsigned int pH  = params[4];
    const unsigned int pW  = params[5];
    const unsigned int sH  = params[6];
    const unsigned int sW  = params[7];
    const unsigned int oH  = params[8];
    const unsigned int oW  = params[9];
    const unsigned int Cin = params[10];

    const unsigned int col_n = Cin * kH * kW * oH * oW;
    const unsigned int b     = idx / col_n;
    const unsigned int r     = idx - b * col_n;

    const unsigned int w_col  = r % oW;
    const unsigned int h_col  = (r / oW) % oH;
    const unsigned int c_col  = r / (oW * oH);
    const unsigned int kw_off = c_col % kW;
    const unsigned int kh_off = (c_col / kW) % kH;
    const unsigned int c_in   = c_col / (kW * kH);

    const int h_in = (int)(h_col * sH) + (int)kh_off - (int)pH;
    const int w_in = (int)(w_col * sW) + (int)kw_off - (int)pW;

    float v = 0.0f;
    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
        const size_t in_off = (size_t)b * Cin * H * W
                            + (size_t)c_in * H * W
                            + (size_t)h_in * W
                            + (size_t)w_in;
        v = input[in_off];
    }
    col[idx] = v;
}

// =============================================================================
// col2im_batched_f32
// =============================================================================
// col:    [batch, C_in*kH*kW, oH*oW]
// output: [batch, C_in, H, W]  -- MUST be zero-initialised by the caller
// n:      batch * C_in*kH*kW * oH*oW
// Overlapping receptive fields scatter to the same input pixel, hence atomicAdd.
__global__ void col2im_batched_f32(
    const float* __restrict__ col,
    float* __restrict__ output,
    const unsigned int* __restrict__ params,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned int H   = params[0];
    const unsigned int W   = params[1];
    const unsigned int kH  = params[2];
    const unsigned int kW  = params[3];
    const unsigned int pH  = params[4];
    const unsigned int pW  = params[5];
    const unsigned int sH  = params[6];
    const unsigned int sW  = params[7];
    const unsigned int oH  = params[8];
    const unsigned int oW  = params[9];
    const unsigned int Cin = params[10];

    const unsigned int col_n = Cin * kH * kW * oH * oW;
    const unsigned int b     = idx / col_n;
    const unsigned int r     = idx - b * col_n;

    const unsigned int w_col  = r % oW;
    const unsigned int h_col  = (r / oW) % oH;
    const unsigned int c_col  = r / (oW * oH);
    const unsigned int kw_off = c_col % kW;
    const unsigned int kh_off = (c_col / kW) % kH;
    const unsigned int c_in   = c_col / (kW * kH);

    const int h_in = (int)(h_col * sH) + (int)kh_off - (int)pH;
    const int w_in = (int)(w_col * sW) + (int)kw_off - (int)pW;

    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
        const size_t out_off = (size_t)b * Cin * H * W
                             + (size_t)c_in * H * W
                             + (size_t)h_in * W
                             + (size_t)w_in;
        atomicAdd(&output[out_off], col[idx]);
    }
}

// =============================================================================
// sum_batch_f32
// =============================================================================
// Reduce per-batch partials into one accumulator: out[i] += sum_b partial[b*len + i].
// Used for grad_weight, whose per-batch GEMMs previously accumulated serially with beta=1.0 --
// a strided-batched GEMM must write disjoint C blocks, so it emits partials and reduces here.
// `out` is ADDED to, matching the original beta=1.0 accumulation semantics.
__global__ void sum_batch_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    unsigned int len,
    unsigned int batch)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    float acc = 0.0f;
    for (unsigned int b = 0; b < batch; ++b) {
        acc += partial[(size_t)b * len + i];
    }
    out[i] += acc;
}

// =============================================================================
// bias_add_channels_batched_f32
// =============================================================================
// data: [batch, C_out, spatial] (in-place), bias: [C_out]
// The single-image bias_add_channels_f32 computes `channel = i / spatial` with NO wrap, so feeding
// it a whole batch would index bias past C_out. This wraps per batch element.
__global__ void bias_add_channels_batched_f32(
    float* __restrict__ data,
    const float* __restrict__ bias,
    unsigned int spatial,
    unsigned int out_channels,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int ch = (i / spatial) % out_channels;
    data[i] += bias[ch];
}

// =============================================================================
// sum_bias_f32
// =============================================================================
// grad_out: [batch, C_out, spatial] -> grad_bias[C_out], ACCUMULATED (+=).
// grad_bias was previously summed on the HOST from grad_out.to_vec(), an unsynchronised D2H that
// could read grad_out before the kernel producing it had landed -> intermittently wrong bias
// gradients. Reducing on the stream removes both the race and a full grad_out download per conv.
__global__ void sum_bias_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    unsigned int spatial,
    unsigned int out_channels,
    unsigned int batch)
{
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= out_channels) return;
    float acc = 0.0f;
    for (unsigned int b = 0; b < batch; ++b) {
        const float* p = grad_out + ((size_t)b * out_channels + c) * spatial;
        for (unsigned int s = 0; s < spatial; ++s) acc += p[s];
    }
    grad_bias[c] += acc;
}

// =============================================================================
// im2col_group_batched_f32
// =============================================================================
// Grouped/depthwise im2col for the WHOLE batch AND all groups in one launch.
// input: [batch, C_in_total, H, W]
// col:   [groups][batch][icg*kH*kW][oH*oW]   (group-major, so each group's block is contiguous
//                                             and its batch stride is col_n_g)
// params: u32[13] = { H, W, kH, kW, pH, pW, sH, sW, oH, oW, icg, C_in_total, batch }
// n: groups * batch * icg*kH*kW*oH*oW
//
// conv2d_grouped_cuda looped `for b { for g { narrow+contiguous+im2col+gemm+bias+memcpy } }` --
// for a DEPTHWISE conv (groups == channels) that is batch*channels iterations, e.g. 8*64 = 512
// iterations of ~8 launches each on a single layer.
__global__ void im2col_group_batched_f32(
    const float* __restrict__ input,
    float* __restrict__ col,
    const unsigned int* __restrict__ params,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned int H     = params[0];
    const unsigned int W     = params[1];
    const unsigned int kH    = params[2];
    const unsigned int kW    = params[3];
    const unsigned int pH    = params[4];
    const unsigned int pW    = params[5];
    const unsigned int sH    = params[6];
    const unsigned int sW    = params[7];
    const unsigned int oH    = params[8];
    const unsigned int oW    = params[9];
    const unsigned int icg   = params[10];
    const unsigned int Ctot  = params[11];
    const unsigned int batch = params[12];

    const unsigned int col_n_g = icg * kH * kW * oH * oW;
    const unsigned int gb      = idx / col_n_g;
    const unsigned int r       = idx - gb * col_n_g;
    const unsigned int g       = gb / batch;
    const unsigned int b       = gb - g * batch;

    const unsigned int w_col  = r % oW;
    const unsigned int h_col  = (r / oW) % oH;
    const unsigned int c_col  = r / (oW * oH);
    const unsigned int kw_off = c_col % kW;
    const unsigned int kh_off = (c_col / kW) % kH;
    const unsigned int c_loc  = c_col / (kW * kH);
    const unsigned int c_in   = g * icg + c_loc;

    const int h_in = (int)(h_col * sH) + (int)kh_off - (int)pH;
    const int w_in = (int)(w_col * sW) + (int)kw_off - (int)pW;

    float v = 0.0f;
    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
        const size_t in_off = (size_t)b * Ctot * H * W
                            + (size_t)c_in * H * W
                            + (size_t)h_in * W
                            + (size_t)w_in;
        v = input[in_off];
    }
    col[idx] = v;
}

// =============================================================================
// col2im_group_batched_f32
// =============================================================================
// Reverse of im2col_group_batched_f32.
// col:    [groups][batch][icg*kH*kW][oH*oW]
// output: [batch, C_in_total, H, W]  -- MUST be zero-initialised by the caller
// params: u32[13] = { H, W, kH, kW, pH, pW, sH, sW, oH, oW, icg, C_in_total, batch }
// n:      groups * batch * icg*kH*kW*oH*oW
__global__ void col2im_group_batched_f32(
    const float* __restrict__ col,
    float* __restrict__ output,
    const unsigned int* __restrict__ params,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned int H     = params[0];
    const unsigned int W     = params[1];
    const unsigned int kH    = params[2];
    const unsigned int kW    = params[3];
    const unsigned int pH    = params[4];
    const unsigned int pW    = params[5];
    const unsigned int sH    = params[6];
    const unsigned int sW    = params[7];
    const unsigned int oH    = params[8];
    const unsigned int oW    = params[9];
    const unsigned int icg   = params[10];
    const unsigned int Ctot  = params[11];
    const unsigned int batch = params[12];

    const unsigned int col_n_g = icg * kH * kW * oH * oW;
    const unsigned int gb      = idx / col_n_g;
    const unsigned int r       = idx - gb * col_n_g;
    const unsigned int g       = gb / batch;
    const unsigned int b       = gb - g * batch;

    const unsigned int w_col  = r % oW;
    const unsigned int h_col  = (r / oW) % oH;
    const unsigned int c_col  = r / (oW * oH);
    const unsigned int kw_off = c_col % kW;
    const unsigned int kh_off = (c_col / kW) % kH;
    const unsigned int c_loc  = c_col / (kW * kH);
    const unsigned int c_in   = g * icg + c_loc;

    const int h_in = (int)(h_col * sH) + (int)kh_off - (int)pH;
    const int w_in = (int)(w_col * sW) + (int)kw_off - (int)pW;

    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
        const size_t out_off = (size_t)b * Ctot * H * W
                             + (size_t)c_in * H * W
                             + (size_t)h_in * W
                             + (size_t)w_in;
        atomicAdd(&output[out_off], col[idx]);
    }
}

// =============================================================================
// sum_batch_at_f32
// =============================================================================
// sum_batch_f32 with a destination offset: out[out_offset + i] += sum_b partial[b*len + i].
// Grouped grad_weight reduces each group's per-batch partials into that group's slice of the
// full weight-gradient buffer.
__global__ void sum_batch_at_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    unsigned int out_offset,
    unsigned int len,
    unsigned int batch)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    float acc = 0.0f;
    for (unsigned int b = 0; b < batch; ++b) {
        acc += partial[(size_t)b * len + i];
    }
    out[(size_t)out_offset + i] += acc;
}

// =============================================================================
// strided_block_copy_f32
// =============================================================================
// dst[o*block_dst + dst_offset + i] = src[o*block_src + i]   for o < outer, i < block_src
//
// `narrow_backward_cuda` scattered a slice back into a zeroed full-shape tensor with ONE
// memcpy_dtod PER OUTER BLOCK. For narrow(dim=1) on [batch, 4h] the outer size is the BATCH, so a
// single narrow cost `batch` launches — an 8-step LSTM with 4 gate narrows per step issued ~2048
// tiny copies per backward, and the cost scaled linearly with batch size. This does it in one
// launch, and on the compute stream (the old path used the null-stream memcpy_dtod_sync).
__global__ void strided_block_copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int block_src,
    unsigned int block_dst,
    unsigned int dst_offset,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned int o = idx / block_src;
    const unsigned int i = idx - o * block_src;
    dst[(size_t)o * block_dst + dst_offset + i] = src[idx];
}

// =============================================================================
// scatter_add_u32_f32
// =============================================================================
// grad_in[idx[o]] += grad_out[o]   for o < n_out.   grad_in MUST be pre-zeroed.
// Used by pooling backward (MaxPool/AdaptiveAvgPool): the forward saved, per output element, the
// input index it came from; the backward scatters the gradient back to those positions. This was a
// host loop after a full grad_out D2H + a grad_in H2D per call. atomicAdd handles the (rare, at
// pooling strides) case of two outputs mapping to the same input.
__global__ void scatter_add_u32_f32(
    const float* __restrict__ grad_out,
    const unsigned int* __restrict__ idx,
    float* __restrict__ grad_in,
    unsigned int in_numel,
    unsigned int n_out)
{
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    unsigned int t = idx[o];
    if (t < in_numel) atomicAdd(&grad_in[t], grad_out[o]);
}

// =============================================================================
// adaptive_avgpool2d_bwd_f32
// =============================================================================
// grad_in[b,c,ih,iw] = grad_out[b,c,oh,ow] / count(oh,ow), where (oh,ow) is the UNIQUE adaptive
// window that contains (ih,iw). One thread per INPUT element -> a gather, no atomics, no pre-zero.
// params: u32[6] = { batch, channels, in_h, in_w, out_h, out_w }
__global__ void adaptive_avgpool2d_bwd_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    const unsigned int* __restrict__ params,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned int C   = params[1];
    const unsigned int H   = params[2];
    const unsigned int W   = params[3];
    const unsigned int oH  = params[4];
    const unsigned int oW  = params[5];

    const unsigned int iw = idx % W;
    const unsigned int ih = (idx / W) % H;
    const unsigned int c  = (idx / (W * H)) % C;
    const unsigned int b  =  idx / (W * H * C);

    // Owning output window (unique). The closed-form inverse (ih*oH)/H can be off by one at
    // boundaries when H is not divisible by oH, so correct it against the actual [start,end) tiling.
    int oh = (int)((ih * oH) / H);
    while (oh > 0        && (unsigned)((oh * H) / oH) > ih)           oh--;
    while ((oh + 1) < (int)oH && (unsigned)(((oh + 1) * H) / oH) <= ih) oh++;
    int ow = (int)((iw * oW) / W);
    while (ow > 0        && (unsigned)((ow * W) / oW) > iw)           ow--;
    while ((ow + 1) < (int)oW && (unsigned)(((ow + 1) * W) / oW) <= iw) ow++;
    const unsigned int ih_start = (oh * H) / oH, ih_end = ((oh + 1) * H) / oH;
    const unsigned int iw_start = (ow * W) / oW, iw_end = ((ow + 1) * W) / oW;
    const unsigned int count = (ih_end - ih_start) * (iw_end - iw_start);

    const size_t go = ((size_t)b * C + c) * oH * oW + (size_t)oh * oW + ow;
    grad_in[idx] = count > 0 ? grad_out[go] / (float)count : 0.0f;
}

// =============================================================================
// groupnorm_bwd_stats_f32 / groupnorm_bwd_apply_f32
// =============================================================================
// GroupNorm backward. params: u32[4] = { batch, channels, spatial, num_groups }.
// Pass 1: one block per (batch, group) reduces mean, std_inv, sum_dy, sum_dy_xhat into stats
//   [batch*num_groups][4].  Pass 2: one thread per element writes d_input and atomicAdds
//   d_weight/d_bias.  weight is [channels]; d_weight/d_bias MUST be pre-zeroed.
__global__ void groupnorm_bwd_stats_f32(
    const float* __restrict__ input,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    const unsigned int* __restrict__ params,
    float eps,
    float* __restrict__ stats)
{
    const unsigned int C   = params[1];
    const unsigned int S   = params[2];
    const unsigned int G   = params[3];
    const unsigned int cpg = C / G;
    const unsigned int gsz = cpg * S;

    const unsigned int bg = blockIdx.x;           // one block per (batch, group)
    const unsigned int b  = bg / G;
    const unsigned int g  = bg - b * G;
    const size_t base = ((size_t)b * C + (size_t)g * cpg) * S;

    __shared__ float sh[4];
    if (threadIdx.x < 4) sh[threadIdx.x] = 0.0f;
    __syncthreads();
    float s_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < gsz; i += blockDim.x) s_sum += input[base + i];
    for (unsigned int o = 16; o > 0; o >>= 1) s_sum += __shfl_down_sync(0xffffffff, s_sum, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&sh[0], s_sum);
    __syncthreads();
    const float mean = sh[0] / (float)gsz;

    float s_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < gsz; i += blockDim.x) {
        float d = input[base + i] - mean; s_var += d * d;
    }
    for (unsigned int o = 16; o > 0; o >>= 1) s_var += __shfl_down_sync(0xffffffff, s_var, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&sh[1], s_var);
    __syncthreads();
    const float std_inv = rsqrtf(sh[1] / (float)gsz + eps);

    float s_dy = 0.0f, s_dyx = 0.0f;
    for (unsigned int i = threadIdx.x; i < gsz; i += blockDim.x) {
        const unsigned int ch = g * cpg + i / S;
        const float xhat = (input[base + i] - mean) * std_inv;
        const float dy = grad_out[base + i] * weight[ch];
        s_dy += dy; s_dyx += dy * xhat;
    }
    for (unsigned int o = 16; o > 0; o >>= 1) { s_dy += __shfl_down_sync(0xffffffff, s_dy, o); s_dyx += __shfl_down_sync(0xffffffff, s_dyx, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&sh[2], s_dy); atomicAdd(&sh[3], s_dyx); }
    __syncthreads();

    if (threadIdx.x == 0) {
        stats[bg * 4 + 0] = mean;
        stats[bg * 4 + 1] = std_inv;
        stats[bg * 4 + 2] = sh[2];
        stats[bg * 4 + 3] = sh[3];
    }
}

__global__ void groupnorm_bwd_apply_f32(
    const float* __restrict__ input,
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    const float* __restrict__ stats,
    const unsigned int* __restrict__ params,
    float* __restrict__ d_input,
    float* __restrict__ d_weight,
    float* __restrict__ d_bias,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned int C = params[1];
    const unsigned int S = params[2];
    const unsigned int G = params[3];
    const unsigned int cpg = C / G;
    const unsigned int gsz = cpg * S;

    const unsigned int s  = idx % S;
    const unsigned int ch = (idx / S) % C;
    const unsigned int b  =  idx / (S * C);
    const unsigned int g  = ch / cpg;
    const unsigned int bg = b * G + g;

    const float mean = stats[bg * 4 + 0], std_inv = stats[bg * 4 + 1];
    const float sum_dy = stats[bg * 4 + 2], sum_dyx = stats[bg * 4 + 3];
    const float ncnt = (float)gsz;
    (void)s;

    const float xhat = (input[idx] - mean) * std_inv;
    const float dy   = grad_out[idx] * weight[ch];
    d_input[idx] = std_inv * (dy - sum_dy / ncnt - xhat * sum_dyx / ncnt);
    atomicAdd(&d_weight[ch], grad_out[idx] * xhat);
    atomicAdd(&d_bias[ch],   grad_out[idx]);
}

// =============================================================================
// convtranspose2d backward — direct, mirrors the CPU loops exactly
// =============================================================================
// params: u32[13] = { batch, in_ch, out_ch, in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw }
// weight layout [in_ch, out_ch, kh, kw]; oh = ih*sh + ki - ph, ow = iw*sw + kj - pw.
__global__ void convtranspose2d_bwd_input_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_in,
    const unsigned int* __restrict__ p,
    unsigned int n)  // n = batch*in_ch*in_h*in_w
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned int IC=p[1], OC=p[2], IH=p[3], IW=p[4], OH=p[5], OW=p[6];
    const unsigned int KH=p[7], KW=p[8], SH=p[9], SW=p[10], PH=p[11], PW=p[12];
    const unsigned int iw = idx % IW;
    const unsigned int ih = (idx / IW) % IH;
    const unsigned int ic = (idx / (IW*IH)) % IC;
    const unsigned int b  =  idx / (IW*IH*IC);
    const int oh_base = (int)(ih*SH) - (int)PH;
    const int ow_base = (int)(iw*SW) - (int)PW;
    float sum = 0.0f;
    for (unsigned int ki=0; ki<KH; ++ki) {
        int oh = oh_base + (int)ki; if (oh<0 || oh>=(int)OH) continue;
        for (unsigned int kj=0; kj<KW; ++kj) {
            int ow = ow_base + (int)kj; if (ow<0 || ow>=(int)OW) continue;
            const size_t go_sp = ((size_t)b*OC)*OH*OW + (size_t)oh*OW + ow;
            const size_t w_kij = ((size_t)ic*OC)*KH*KW + (size_t)ki*KW + kj;
            for (unsigned int oc=0; oc<OC; ++oc)
                sum += grad_out[go_sp + (size_t)oc*OH*OW] * weight[w_kij + (size_t)oc*KH*KW];
        }
    }
    grad_in[idx] = sum;
}

__global__ void convtranspose2d_bwd_weight_f32(
    const float* __restrict__ input,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_w,
    const unsigned int* __restrict__ p,
    unsigned int n)  // n = in_ch*out_ch*kh*kw
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned int IC=p[1], OC=p[2], IH=p[3], IW=p[4], OH=p[5], OW=p[6];
    const unsigned int KH=p[7], KW=p[8], SH=p[9], SW=p[10], PH=p[11], PW=p[12], B=p[0];
    const unsigned int kj = idx % KW;
    const unsigned int ki = (idx / KW) % KH;
    const unsigned int oc = (idx / (KW*KH)) % OC;
    const unsigned int ic =  idx / (KW*KH*OC);
    float acc = 0.0f;
    for (unsigned int b=0; b<B; ++b) {
        for (unsigned int ih=0; ih<IH; ++ih) {
            int oh = (int)(ih*SH) + (int)ki - (int)PH; if (oh<0 || oh>=(int)OH) continue;
            for (unsigned int iw=0; iw<IW; ++iw) {
                int ow = (int)(iw*SW) + (int)kj - (int)PW; if (ow<0 || ow>=(int)OW) continue;
                const float in_v = input[(((size_t)b*IC+ic)*IH+ih)*IW+iw];
                acc += in_v * grad_out[(((size_t)b*OC+oc)*OH+oh)*OW+ow];
            }
        }
    }
    grad_w[idx] = acc;
}

// =============================================================================
// mul_backward_f32
// =============================================================================
// grad_lhs[i] = grad_out[i] * rhs[i];  grad_rhs[i] = grad_out[i] * lhs[i].
// One launch reading grad_out/lhs/rhs once and writing both grads, replacing two separate
// grad_out.mul() launches. Same-shape (non-broadcast) case only. Used by every elementwise product
// backward: gated activations (SwiGLU/GLU gate*up), attention score scaling, etc.
__global__ void mul_backward_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ grad_lhs,
    float* __restrict__ grad_rhs,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = grad_out[i];
    grad_lhs[i] = g * rhs[i];
    grad_rhs[i] = g * lhs[i];
}

} // extern "C"
