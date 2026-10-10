// Correct elementwise pow - C99 semantics, sign preserved
//
// File: crates/axonml-core/src/backends/cuda_kernels/pow_fixed.cu
// Author: Andrew Jewell Sr - AutomataNexus
//
// The inline ELEMENTWISE PTX in mod.rs implements pow as
//     abs.f32 ; lg2.approx.f32 ; mul.f32 ; ex2.approx.f32
// i.e. exp2(n * log2(|x|)) == |x|^n. The abs is there so lg2 does not return NaN on a negative
// base, but it DISCARDS THE SIGN, so:
//   pow(x, 2) forward is accidentally right (|x|^2 == x^2) but its BACKWARD factor n*x^(n-1)
//             evaluates to 2*|x| instead of 2*x -- magnitude exact, sign destroyed;
//   pow(x, 3) and pow(x, -1) are wrong in the FORWARD for x < 0.
// `nn::MSELoss` is `diff.pow(2.0).mean()`, so every MSE-trained model on GPU received
// wrong-signed gradients for negative residuals while the printed loss looked perfectly sane.
// The CIoU box loss used across the detector suite has `d2 = (pcx-tcx).pow(2.0) + ...` on a SIGNED
// centre offset, so its box-regression gradient was affected the same way.
//
// elementwise.cu already carries the correct `powf` form for both kernels but has no generated
// .ptx, so it never ran. These are those kernels, compiled, under distinct names so the rest of the
// inline elementwise PTX is left untouched. powf follows C99: negative base with an integral
// exponent is well defined (powf(-2,3) == -8), and NaN otherwise, which is the correct real-valued
// answer rather than a silently sign-stripped one.

extern "C" {

__global__ void pow_f32_c99(
    const float* __restrict__ base,
    const float* __restrict__ exp,
    float* __restrict__ output,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = powf(base[idx], exp[idx]);
    }
}

__global__ void pow_scalar_f32_c99(
    const float* __restrict__ base,
    float exp,
    float* __restrict__ output,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = powf(base[idx], exp);
    }
}

} // extern "C"
