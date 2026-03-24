// AxonML CUDA Activation Kernels
// Compile with: nvcc -ptx -arch=sm_50 --use_fast_math activations.cu -o activations.ptx

extern "C" __global__ void relu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

extern "C" __global__ void relu_backward_f32(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

extern "C" __global__ void leaky_relu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : val * negative_slope;
    }
}

extern "C" __global__ void sigmoid_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

extern "C" __global__ void sigmoid_backward_f32(
    const float* __restrict__ grad_output,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = output[idx];
        grad_input[idx] = grad_output[idx] * s * (1.0f - s);
    }
}

extern "C" __global__ void tanh_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

extern "C" __global__ void tanh_backward_f32(
    const float* __restrict__ grad_output,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - t * t);
    }
}

extern "C" __global__ void exp_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }
}

extern "C" __global__ void log_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = logf(input[idx]);
    }
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__ void gelu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        // sqrt(2/pi) ≈ 0.7978845608
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// SiLU (Swish): x * sigmoid(x)
extern "C" __global__ void silu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

// ELU: x if x > 0, alpha * (exp(x) - 1) otherwise
extern "C" __global__ void elu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    float alpha,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}

// Softmax (per-row for 2D tensor, assumes contiguous row-major storage)
// Note: This is a simplified version. Production softmax needs numerical stability.
extern "C" __global__ void softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (unsigned int c = 0; c < cols; c++) {
        float val = input[row * cols + c];
        if (val > max_val) max_val = val;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (unsigned int c = 0; c < cols; c++) {
        float exp_val = expf(input[row * cols + c] - max_val);
        output[row * cols + c] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (unsigned int c = 0; c < cols; c++) {
        output[row * cols + c] /= sum;
    }
}
