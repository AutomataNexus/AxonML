// Pooling CUDA Kernels - MaxPool2d and AvgPool2d (forward + backward)
//
// File: crates/axonml-core/src/backends/cuda_kernels/pooling.cu
// Author: Andrew Jewell Sr - AutomataNexus

extern "C" {

// =============================================================================
// maxpool2d_fwd_f32
// =============================================================================
// Input:  [N, C, H, W]
// Output: [N, C, H_out, W_out]
// Indices:[N, C, H_out, W_out] (int32, flat index into input for backward)
//
// params: u32[8] = {H, W, kH, kW, sH, sW, pH, pW}
// Each thread computes one output element.
__global__ void maxpool2d_fwd_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ indices,
    const unsigned int* __restrict__ params,
    unsigned int C,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int H   = params[0];
    unsigned int W   = params[1];
    unsigned int kH  = params[2];
    unsigned int kW  = params[3];
    unsigned int sH  = params[4];
    unsigned int sW  = params[5];
    unsigned int pH  = params[6];
    unsigned int pW  = params[7];

    // Decompose idx into (n, c, oh, ow)
    unsigned int ow = idx % out_w;
    unsigned int oh = (idx / out_w) % out_h;
    unsigned int c  = (idx / (out_w * out_h)) % C;
    unsigned int n  = idx / (out_w * out_h * C);

    float max_val = -3.402823e+38f; // -FLT_MAX
    int max_idx = -1;

    unsigned int input_offset = n * C * H * W + c * H * W;

    for (unsigned int ki = 0; ki < kH; ki++) {
        int ih = (int)(oh * sH + ki) - (int)pH;
        if (ih < 0 || ih >= (int)H) continue;
        for (unsigned int kj = 0; kj < kW; kj++) {
            int iw = (int)(ow * sW + kj) - (int)pW;
            if (iw < 0 || iw >= (int)W) continue;

            unsigned int in_idx = input_offset + (unsigned int)ih * W + (unsigned int)iw;
            float val = input[in_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = (int)in_idx;
            }
        }
    }

    output[idx] = max_val;
    indices[idx] = max_idx;
}

// =============================================================================
// maxpool2d_bwd_f32
// =============================================================================
// grad_output: [N, C, H_out, W_out]
// indices:     [N, C, H_out, W_out] (flat index into grad_input)
// grad_input:  [N, C, H, W] (must be zero-initialized)
// Each thread handles one output element and scatters grad to max index.
__global__ void maxpool2d_bwd_f32(
    const float* __restrict__ grad_output,
    const int* __restrict__ indices,
    float* __restrict__ grad_input,
    unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int max_idx = indices[idx];
    if (max_idx >= 0) {
        atomicAdd(&grad_input[max_idx], grad_output[idx]);
    }
}

// =============================================================================
// avgpool2d_fwd_f32
// =============================================================================
// Input:  [N, C, H, W]
// Output: [N, C, H_out, W_out]
// params: u32[9] = {H, W, kH, kW, sH, sW, pH, pW, count_include_pad}
// Each thread computes one output element as average of kernel window.
__global__ void avgpool2d_fwd_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int* __restrict__ params,
    unsigned int C,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int H   = params[0];
    unsigned int W   = params[1];
    unsigned int kH  = params[2];
    unsigned int kW  = params[3];
    unsigned int sH  = params[4];
    unsigned int sW  = params[5];
    unsigned int pH  = params[6];
    unsigned int pW  = params[7];
    unsigned int count_include_pad = params[8];

    // Decompose idx into (n, c, oh, ow)
    unsigned int ow = idx % out_w;
    unsigned int oh = (idx / out_w) % out_h;
    unsigned int c  = (idx / (out_w * out_h)) % C;
    unsigned int n  = idx / (out_w * out_h * C);

    float sum = 0.0f;
    unsigned int count = 0;

    unsigned int input_offset = n * C * H * W + c * H * W;

    for (unsigned int ki = 0; ki < kH; ki++) {
        int ih = (int)(oh * sH + ki) - (int)pH;
        if (ih < 0 || ih >= (int)H) continue;
        for (unsigned int kj = 0; kj < kW; kj++) {
            int iw = (int)(ow * sW + kj) - (int)pW;
            if (iw < 0 || iw >= (int)W) continue;

            unsigned int in_idx = input_offset + (unsigned int)ih * W + (unsigned int)iw;
            sum += input[in_idx];
            count++;
        }
    }

    float divisor;
    if (count_include_pad) {
        divisor = (float)(kH * kW);
    } else {
        divisor = (count > 0) ? (float)count : 1.0f;
    }

    output[idx] = sum / divisor;
}

// =============================================================================
// avgpool2d_bwd_f32
// =============================================================================
// grad_output: [N, C, H_out, W_out]
// grad_input:  [N, C, H, W] (must be zero-initialized)
// params: u32[9] = {H, W, kH, kW, sH, sW, pH, pW, count_include_pad}
// Each thread handles one output element and distributes grad to input window.
__global__ void avgpool2d_bwd_f32(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const unsigned int* __restrict__ params,
    unsigned int C,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int H   = params[0];
    unsigned int W   = params[1];
    unsigned int kH  = params[2];
    unsigned int kW  = params[3];
    unsigned int sH  = params[4];
    unsigned int sW  = params[5];
    unsigned int pH  = params[6];
    unsigned int pW  = params[7];
    unsigned int count_include_pad = params[8];

    // Decompose idx into (n, c, oh, ow)
    unsigned int ow = idx % out_w;
    unsigned int oh = (idx / out_w) % out_h;
    unsigned int c  = (idx / (out_w * out_h)) % C;
    unsigned int n  = idx / (out_w * out_h * C);

    // First count valid positions to compute divisor
    unsigned int count = 0;
    for (unsigned int ki = 0; ki < kH; ki++) {
        int ih = (int)(oh * sH + ki) - (int)pH;
        if (ih < 0 || ih >= (int)H) continue;
        for (unsigned int kj = 0; kj < kW; kj++) {
            int iw = (int)(ow * sW + kj) - (int)pW;
            if (iw < 0 || iw >= (int)W) continue;
            count++;
        }
    }

    float divisor;
    if (count_include_pad) {
        divisor = (float)(kH * kW);
    } else {
        divisor = (count > 0) ? (float)count : 1.0f;
    }

    float grad_val = grad_output[idx] / divisor;

    unsigned int input_offset = n * C * H * W + c * H * W;

    for (unsigned int ki = 0; ki < kH; ki++) {
        int ih = (int)(oh * sH + ki) - (int)pH;
        if (ih < 0 || ih >= (int)H) continue;
        for (unsigned int kj = 0; kj < kW; kj++) {
            int iw = (int)(ow * sW + kj) - (int)pW;
            if (iw < 0 || iw >= (int)W) continue;

            unsigned int in_idx = input_offset + (unsigned int)ih * W + (unsigned int)iw;
            atomicAdd(&grad_input[in_idx], grad_val);
        }
    }
}

} // extern "C"
