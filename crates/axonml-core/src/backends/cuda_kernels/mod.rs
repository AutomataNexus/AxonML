//! CUDA Kernel Registry
//!
//! Manages loading and caching of CUDA kernels for element-wise operations.
//! Kernels are compiled from PTX at runtime using cudarc.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use super::cuda::CudaError;

/// Block size for kernel launches (256 threads per block is typical optimal)
pub const BLOCK_SIZE: u32 = 256;

/// Embedded PTX for element-wise operations
#[cfg(feature = "cuda")]
pub const ELEMENTWISE_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// add_f32 kernel: out[i] = a[i] + b[i]
.visible .entry add_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__add_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__add_exit:
    ret;
}

// sub_f32 kernel: out[i] = a[i] - b[i]
.visible .entry sub_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__sub_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    sub.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__sub_exit:
    ret;
}

// mul_f32 kernel: out[i] = a[i] * b[i]
.visible .entry mul_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__mul_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__mul_exit:
    ret;
}

// div_f32 kernel: out[i] = a[i] / b[i]
.visible .entry div_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__div_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    div.approx.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__div_exit:
    ret;
}

// scale_f32 kernel: data[i] *= alpha (in-place)
.visible .entry scale_f32(
    .param .u64 data,
    .param .f32 alpha,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [data];
    ld.param.f32 %f1, [alpha];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__scale_exit;

    cvt.u64.u32 %rd2, %r2;
    shl.b64 %rd3, %rd2, 2;
    add.s64 %rd4, %rd1, %rd3;

    ld.global.f32 %f2, [%rd4];
    mul.f32 %f2, %f2, %f1;
    st.global.f32 [%rd4], %f2;

$L__scale_exit:
    ret;
}

// add_scalar_f32 kernel: out[i] = src[i] + scalar
.visible .entry add_scalar_f32(
    .param .u64 src,
    .param .f32 scalar,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [src];
    ld.param.f32 %f1, [scalar];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__addsc_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;

    ld.global.f32 %f2, [%rd5];
    add.f32 %f2, %f2, %f1;
    st.global.f32 [%rd6], %f2;

$L__addsc_exit:
    ret;
}

// neg_f32 kernel: out[i] = -src[i]
.visible .entry neg_f32(
    .param .u64 src,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__neg_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    neg.f32 %f1, %f1;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f1;

$L__neg_exit:
    ret;
}

// sqrt_f32 kernel: out[i] = sqrt(src[i])
.visible .entry sqrt_f32(
    .param .u64 src,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__sqrt_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    sqrt.approx.f32 %f1, %f1;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f1;

$L__sqrt_exit:
    ret;
}

// pow_f32 kernel: out[i] = a[i] ^ b[i]  (using lg2/ex2)
.visible .entry pow_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__pow_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    // pow(a, b) = exp2(b * log2(a))
    abs.f32 %f3, %f1;
    lg2.approx.f32 %f3, %f3;
    mul.f32 %f3, %f2, %f3;
    ex2.approx.f32 %f4, %f3;
    st.global.f32 [%rd8], %f4;

$L__pow_exit:
    ret;
}

// pow_scalar_f32 kernel: out[i] = src[i] ^ exp
.visible .entry pow_scalar_f32(
    .param .u64 src,
    .param .f32 exp,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [src];
    ld.param.f32 %f1, [exp];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__pows_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;

    ld.global.f32 %f2, [%rd5];
    abs.f32 %f3, %f2;
    lg2.approx.f32 %f3, %f3;
    mul.f32 %f3, %f1, %f3;
    ex2.approx.f32 %f4, %f3;
    st.global.f32 [%rd6], %f4;

$L__pows_exit:
    ret;
}
"#;

/// Embedded PTX for activation functions
#[cfg(feature = "cuda")]
pub const ACTIVATIONS_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// relu_f32 kernel: out[i] = max(0, src[i])
.visible .entry relu_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__relu_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    max.f32 %f1, %f1, 0f00000000;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f1;

$L__relu_exit:
    ret;
}

// relu_backward_f32: out[i] = grad[i] * (input[i] > 0 ? 1.0 : 0.0)
.visible .entry relu_backward_f32(
    .param .u64 grad_output,
    .param .u64 input,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<3>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [input];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__relub_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    setp.gt.f32 %p2, %f2, 0f00000000;
    selp.f32 %f2, %f1, 0f00000000, %p2;
    st.global.f32 [%rd8], %f2;

$L__relub_exit:
    ret;
}

// sigmoid_f32 kernel
.visible .entry sigmoid_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__sig_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    neg.f32 %f1, %f1;
    mul.f32 %f1, %f1, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f1;
    add.f32 %f3, %f2, 0f3F800000;
    rcp.approx.f32 %f4, %f3;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f4;

$L__sig_exit:
    ret;
}

// sigmoid_backward_f32: out[i] = grad[i] * output[i] * (1 - output[i])
.visible .entry sigmoid_backward_f32(
    .param .u64 grad_output,
    .param .u64 sig_output,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [sig_output];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__sigb_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    // grad * sig * (1 - sig)
    mov.f32 %f3, 0f3F800000;
    sub.f32 %f3, %f3, %f2;
    mul.f32 %f4, %f2, %f3;
    mul.f32 %f4, %f1, %f4;
    st.global.f32 [%rd8], %f4;

$L__sigb_exit:
    ret;
}

// tanh_f32 kernel
.visible .entry tanh_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<8>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__tanh_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    mul.f32 %f2, %f1, 0f40000000;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f2;
    add.f32 %f4, %f3, 0fBF800000;
    add.f32 %f5, %f3, 0f3F800000;
    div.approx.f32 %f6, %f4, %f5;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f6;

$L__tanh_exit:
    ret;
}

// tanh_backward_f32: out[i] = grad[i] * (1 - output[i]^2)
.visible .entry tanh_backward_f32(
    .param .u64 grad_output,
    .param .u64 tanh_output,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [tanh_output];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__tanhb_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    // grad * (1 - tanh^2)
    mul.f32 %f3, %f2, %f2;
    mov.f32 %f4, 0f3F800000;
    sub.f32 %f4, %f4, %f3;
    mul.f32 %f4, %f1, %f4;
    st.global.f32 [%rd8], %f4;

$L__tanhb_exit:
    ret;
}

// exp_f32 kernel: out[i] = exp(src[i])
.visible .entry exp_f32(
    .param .u64 src,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__exp_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    // exp(x) = exp2(x / ln(2)) = exp2(x * 1.4426950408889634)
    mul.f32 %f1, %f1, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f1;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f2;

$L__exp_exit:
    ret;
}

// log_f32 kernel: out[i] = ln(src[i])
.visible .entry log_f32(
    .param .u64 src,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__log_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    // ln(x) = log2(x) * ln(2) = log2(x) * 0.6931471805599453
    lg2.approx.f32 %f1, %f1;
    mul.f32 %f2, %f1, 0f3F317218;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f2;

$L__log_exit:
    ret;
}

// gelu_f32 kernel: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
.visible .entry gelu_f32(
    .param .u64 src,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<12>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__gelu_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    // x^3
    mul.f32 %f2, %f1, %f1;
    mul.f32 %f2, %f2, %f1;
    // 0.044715 * x^3
    mul.f32 %f3, %f2, 0f3D372713;
    // x + 0.044715 * x^3
    add.f32 %f4, %f1, %f3;
    // sqrt(2/pi) = 0.7978845608
    mul.f32 %f5, %f4, 0f3F4C422A;
    // tanh(f5) via (exp(2x)-1)/(exp(2x)+1)
    mul.f32 %f6, %f5, 0f40000000;
    mul.f32 %f6, %f6, 0f3FB8AA3B;
    ex2.approx.f32 %f7, %f6;
    add.f32 %f8, %f7, 0fBF800000;
    add.f32 %f9, %f7, 0f3F800000;
    div.approx.f32 %f10, %f8, %f9;
    // 1 + tanh(...)
    add.f32 %f10, %f10, 0f3F800000;
    // 0.5 * x * (1 + tanh(...))
    mul.f32 %f11, %f1, %f10;
    mul.f32 %f11, %f11, 0f3F000000;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f11;

$L__gelu_exit:
    ret;
}

// silu_f32 kernel: out[i] = x * sigmoid(x)
.visible .entry silu_f32(
    .param .u64 src,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<6>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__silu_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    // sigmoid(x)
    neg.f32 %f2, %f1;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f2;
    add.f32 %f4, %f3, 0f3F800000;
    rcp.approx.f32 %f5, %f4;
    // x * sigmoid(x)
    mul.f32 %f5, %f1, %f5;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f5;

$L__silu_exit:
    ret;
}
"#;

/// CUDA Kernel registry for managing loaded kernels
#[cfg(feature = "cuda")]
pub struct CudaKernels {
    device: Arc<CudaDevice>,
    functions: HashMap<String, CudaFunction>,
}

#[cfg(feature = "cuda")]
impl CudaKernels {
    /// Load kernels from embedded PTX
    pub fn load(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        let mut kernels = Self {
            device,
            functions: HashMap::new(),
        };

        // Load element-wise kernels
        kernels.load_module(
            "elementwise",
            ELEMENTWISE_PTX,
            &[
                "add_f32", "sub_f32", "mul_f32", "div_f32",
                "scale_f32", "add_scalar_f32",
                "neg_f32", "sqrt_f32",
                "pow_f32", "pow_scalar_f32",
            ],
        )?;

        // Load activation kernels
        kernels.load_module(
            "activations",
            ACTIVATIONS_PTX,
            &[
                "relu_f32", "relu_backward_f32",
                "sigmoid_f32", "sigmoid_backward_f32",
                "tanh_f32", "tanh_backward_f32",
                "exp_f32", "log_f32",
                "gelu_f32", "silu_f32",
            ],
        )?;

        Ok(kernels)
    }

    fn load_module(
        &mut self,
        name: &'static str,
        ptx: &'static str,
        functions: &'static [&'static str],
    ) -> Result<(), CudaError> {
        self.device
            .load_ptx(ptx.into(), name, functions)
            .map_err(|e| CudaError::ModuleLoadFailed(e.to_string()))?;

        for func_name in functions {
            let func = self
                .device
                .get_func(name, func_name)
                .ok_or_else(|| CudaError::KernelNotFound(func_name.to_string()))?;
            self.functions.insert(func_name.to_string(), func);
        }

        Ok(())
    }

    /// Get a kernel function by name
    pub fn get(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }

    /// Check if a kernel is available
    pub fn has(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

/// Compute optimal launch configuration for a given number of elements
#[cfg(feature = "cuda")]
pub fn launch_config(n: usize) -> LaunchConfig {
    let num_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config() {
        let cfg = launch_config(1000);
        assert_eq!(cfg.block_dim, (256, 1, 1));
        assert_eq!(cfg.grid_dim, (4, 1, 1)); // ceil(1000/256) = 4
    }

    #[test]
    fn test_launch_config_large() {
        let cfg = launch_config(1_000_000);
        assert_eq!(cfg.grid_dim, (3907, 1, 1)); // ceil(1000000/256) = 3907
    }
}
