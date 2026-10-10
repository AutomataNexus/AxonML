// Direct depthwise Conv2d kernels - forward, grad_input, grad_weight
//
// File: crates/axonml-core/src/backends/cuda_kernels/depthwise.cu
// Author: Andrew Jewell Sr - AutomataNexus
//
// For a TRUE depthwise conv (groups == in_channels == out_channels, i.e. icg == ocg == 1) the
// im2col+GEMM formulation degenerates: every per-group GEMM is a rank-1 update, so cuBLAS launch
// overhead dominates and a 128-channel layer issues 128 (forward) or 256 (backward) GEMM launches
// to do almost no arithmetic. Measured b8 c128 40x40 k3 fwd+bwd at 27.6 ms on GPU vs 60.1 ms on
// CPU — only 2.2x, for a kernel that is pure elementwise work.
//
// These compute the convolution directly: one thread per output (or input, or weight) element,
// looping the k*k window. No column buffer is materialised at all, which also removes the
// groups*batch*icg*kH*kW*oH*oW scratch allocation the GEMM path needs.
//
// params: u32[12] = { H, W, kH, kW, pH, pW, sH, sW, oH, oW, C, batch }
// weight is [C, 1, kH, kW]; bias is applied separately by bias_add_channels_batched_f32.

extern "C" {

// =============================================================================
// depthwise_fwd_f32
// =============================================================================
// input:  [batch, C, H, W]      output: [batch, C, oH, oW]
// n: batch * C * oH * oW  (one thread per output element)
__global__ void depthwise_fwd_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const unsigned int* __restrict__ params,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned int H  = params[0];
    const unsigned int W  = params[1];
    const unsigned int kH = params[2];
    const unsigned int kW = params[3];
    const unsigned int pH = params[4];
    const unsigned int pW = params[5];
    const unsigned int sH = params[6];
    const unsigned int sW = params[7];
    const unsigned int oH = params[8];
    const unsigned int oW = params[9];
    const unsigned int C  = params[10];

    const unsigned int ow = idx % oW;
    const unsigned int oh = (idx / oW) % oH;
    const unsigned int c  = (idx / (oW * oH)) % C;
    const unsigned int b  = idx / (oW * oH * C);

    const float* in_plane = input + ((size_t)b * C + c) * H * W;
    const float* w_plane  = weight + (size_t)c * kH * kW;

    float acc = 0.0f;
    for (unsigned int i = 0; i < kH; ++i) {
        const int ih = (int)(oh * sH) + (int)i - (int)pH;
        if (ih < 0 || ih >= (int)H) continue;
        for (unsigned int j = 0; j < kW; ++j) {
            const int iw = (int)(ow * sW) + (int)j - (int)pW;
            if (iw < 0 || iw >= (int)W) continue;
            acc += in_plane[(size_t)ih * W + iw] * w_plane[i * kW + j];
        }
    }
    output[idx] = acc;
}

// =============================================================================
// depthwise_grad_input_f32
// =============================================================================
// grad_out: [batch, C, oH, oW]   grad_in: [batch, C, H, W] (fully written, no pre-zero needed)
// n: batch * C * H * W  (one thread per INPUT element -- a gather, so no atomics)
__global__ void depthwise_grad_input_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_in,
    const unsigned int* __restrict__ params,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned int H  = params[0];
    const unsigned int W  = params[1];
    const unsigned int kH = params[2];
    const unsigned int kW = params[3];
    const unsigned int pH = params[4];
    const unsigned int pW = params[5];
    const unsigned int sH = params[6];
    const unsigned int sW = params[7];
    const unsigned int oH = params[8];
    const unsigned int oW = params[9];
    const unsigned int C  = params[10];

    const unsigned int iw = idx % W;
    const unsigned int ih = (idx / W) % H;
    const unsigned int c  = (idx / (W * H)) % C;
    const unsigned int b  = idx / (W * H * C);

    const float* go_plane = grad_out + ((size_t)b * C + c) * oH * oW;
    const float* w_plane  = weight + (size_t)c * kH * kW;

    float acc = 0.0f;
    // output (oh, ow) reads this input iff ih = oh*sH + i - pH, i.e. oh = (ih + pH - i)/sH exactly
    for (unsigned int i = 0; i < kH; ++i) {
        const int oh_num = (int)ih + (int)pH - (int)i;
        if (oh_num < 0 || (unsigned int)oh_num % sH != 0) continue;
        const unsigned int oh = (unsigned int)oh_num / sH;
        if (oh >= oH) continue;
        for (unsigned int j = 0; j < kW; ++j) {
            const int ow_num = (int)iw + (int)pW - (int)j;
            if (ow_num < 0 || (unsigned int)ow_num % sW != 0) continue;
            const unsigned int ow = (unsigned int)ow_num / sW;
            if (ow >= oW) continue;
            acc += go_plane[(size_t)oh * oW + ow] * w_plane[i * kW + j];
        }
    }
    grad_in[idx] = acc;
}

// =============================================================================
// depthwise_grad_weight_f32
// =============================================================================
// grad_w: [C, 1, kH, kW], ACCUMULATED (+=) to match the conv backward's beta=1.0 semantics.
// n: C * kH * kW  (one thread per weight element, reducing over batch and output positions)
__global__ void depthwise_grad_weight_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_w,
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
    const unsigned int C     = params[10];
    const unsigned int batch = params[11];

    const unsigned int j = idx % kW;
    const unsigned int i = (idx / kW) % kH;
    const unsigned int c = idx / (kW * kH);

    float acc = 0.0f;
    for (unsigned int b = 0; b < batch; ++b) {
        const float* in_plane = input + ((size_t)b * C + c) * H * W;
        const float* go_plane = grad_out + ((size_t)b * C + c) * oH * oW;
        for (unsigned int oh = 0; oh < oH; ++oh) {
            const int ih = (int)(oh * sH) + (int)i - (int)pH;
            if (ih < 0 || ih >= (int)H) continue;
            for (unsigned int ow = 0; ow < oW; ++ow) {
                const int iw = (int)(ow * sW) + (int)j - (int)pW;
                if (iw < 0 || iw >= (int)W) continue;
                acc += go_plane[(size_t)oh * oW + ow] * in_plane[(size_t)ih * W + iw];
            }
        }
    }
    grad_w[idx] += acc;
}

} // extern "C"
