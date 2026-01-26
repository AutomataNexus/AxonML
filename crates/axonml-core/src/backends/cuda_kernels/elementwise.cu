// AxonML CUDA Element-wise Kernels
// Compile with: nvcc -ptx -arch=sm_50 --use_fast_math elementwise.cu -o elementwise.ptx

extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void sub_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

extern "C" __global__ void mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__ void div_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

extern "C" __global__ void scale_f32(
    float* __restrict__ data,
    float alpha,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= alpha;
    }
}

extern "C" __global__ void add_scalar_f32(
    float* __restrict__ data,
    float scalar,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += scalar;
    }
}

extern "C" __global__ void neg_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -input[idx];
    }
}

extern "C" __global__ void abs_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fabsf(input[idx]);
    }
}

extern "C" __global__ void sqrt_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}

extern "C" __global__ void pow_f32(
    const float* __restrict__ base,
    const float* __restrict__ exp,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = powf(base[idx], exp[idx]);
    }
}

extern "C" __global__ void pow_scalar_f32(
    const float* __restrict__ base,
    float exp,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = powf(base[idx], exp);
    }
}

extern "C" __global__ void clamp_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    float min_val,
    float max_val,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = fminf(fmaxf(val, min_val), max_val);
    }
}
