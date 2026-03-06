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

/// Embedded PTX for broadcast element-wise operations.
///
/// These kernels support broadcasting by using modular indexing.
/// For `a` of shape [M, N] and `b` of shape [N]:
///   - Flatten both to 1D
///   - out[i] = a[i] OP b[i % b_numel]
///
/// This handles the most common patterns: bias addition, residual scaling, etc.
/// The `_rev` variants broadcast `a` instead of `b`.
#[cfg(feature = "cuda")]
pub const BROADCAST_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// broadcast_add_f32: out[i] = a[i] + b[i % b_len]
// a has n elements (the larger tensor), b has b_len elements (smaller, broadcast)
.visible .entry broadcast_add_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 b_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [b_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__ba_exit;

    // a index: i
    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    // b index: i % b_len
    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd7, %r6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd7, %rd2, %rd7;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__ba_exit:
    ret;
}

// broadcast_sub_f32: out[i] = a[i] - b[i % b_len]
.visible .entry broadcast_sub_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 b_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [b_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__bs_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd7, %r6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd7, %rd2, %rd7;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    sub.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bs_exit:
    ret;
}

// broadcast_mul_f32: out[i] = a[i] * b[i % b_len]
.visible .entry broadcast_mul_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 b_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [b_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__bm_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd7, %r6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd7, %rd2, %rd7;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bm_exit:
    ret;
}

// broadcast_div_f32: out[i] = a[i] / b[i % b_len]
.visible .entry broadcast_div_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 b_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [b_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__bd_exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd7, %r6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd7, %rd2, %rd7;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    div.approx.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bd_exit:
    ret;
}

// broadcast_add_rev_f32: out[i] = a[i % a_len] + b[i]
// When a is the smaller tensor (e.g., [M,1] + [M,N])
.visible .entry broadcast_add_rev_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 a_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [a_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__bar_exit;

    // a index: i % a_len
    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;

    // b index: i
    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bar_exit:
    ret;
}

// broadcast_sub_rev_f32: out[i] = a[i % a_len] - b[i]
.visible .entry broadcast_sub_rev_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 a_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [a_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__bsr_exit;

    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    sub.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bsr_exit:
    ret;
}

// broadcast_mul_rev_f32: out[i] = a[i % a_len] * b[i]
.visible .entry broadcast_mul_rev_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 a_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [a_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__bmr_exit;

    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bmr_exit:
    ret;
}

// broadcast_div_rev_f32: out[i] = a[i % a_len] / b[i]
.visible .entry broadcast_div_rev_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .u32 a_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r5, [a_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__bdr_exit;

    rem.u32 %r6, %r2, %r5;
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    div.approx.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bdr_exit:
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

/// Embedded PTX for reduction operations (softmax, etc.)
///
/// softmax_row_f32: Computes softmax along the last dimension.
/// Each block handles one row of `row_size` elements.
/// Grid: (num_rows), Block: (256).
/// Algorithm: max → subtract max → exp → sum → divide.
/// Handles row_size > blockDim.x via sequential loops + shared reduction.
#[cfg(feature = "cuda")]
pub const REDUCTION_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// softmax_row_f32: in-place softmax per row
// params: data (in/out), num_rows, row_size
.visible .entry softmax_row_f32(
    .param .u64 data,
    .param .u32 num_rows,
    .param .u32 row_size
) {
    .reg .pred %p<4>;
    .reg .f32 %f<8>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;

    // shared memory for reductions (256 floats)
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [data];
    ld.param.u32 %r1, [num_rows];
    ld.param.u32 %r2, [row_size];

    // row index = blockIdx.x
    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra $L__sm_exit;

    // tid = threadIdx.x
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    // row_base = data + row_idx * row_size * 4
    cvt.u64.u32 %rd2, %r3;
    cvt.u64.u32 %rd3, %r2;
    mul.lo.u64 %rd4, %rd2, %rd3;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd5, %rd1, %rd4;   // rd5 = &data[row_idx * row_size]

    // === Phase 1: Find max value in row ===
    // Each thread finds max of its elements
    mov.f32 %f1, 0fFF800000;     // -inf
    mov.u32 %r6, %r4;            // i = tid
$L__sm_max_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__sm_max_done;
    cvt.u64.u32 %rd6, %r6;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.f32 %f2, [%rd7];
    max.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, %r5;      // i += blockDim.x
    bra $L__sm_max_loop;
$L__sm_max_done:

    // Store to shared memory
    cvt.u64.u32 %rd6, %r4;
    shl.b64 %rd6, %rd6, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd7, %rd7, %rd6;
    st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Reduction in shared memory (max)
    mov.u32 %r6, 128;
$L__sm_max_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__sm_max_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__sm_max_red_skip;
    // sdata[tid] = max(sdata[tid], sdata[tid + stride])
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd6, %r7;
    shl.b64 %rd6, %rd6, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd6, %rd7, %rd6;
    ld.shared.f32 %f2, [%rd6];
    cvt.u64.u32 %rd6, %r4;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd6, %rd7, %rd6;
    ld.shared.f32 %f3, [%rd6];
    max.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd6], %f3;
$L__sm_max_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__sm_max_red;
$L__sm_max_red_done:

    // Broadcast max value: f4 = sdata[0]
    mov.u64 %rd7, sdata;
    ld.shared.f32 %f4, [%rd7];    // f4 = row max
    bar.sync 0;

    // === Phase 2: exp(x - max) and sum ===
    mov.f32 %f1, 0f00000000;      // sum = 0
    mov.u32 %r6, %r4;             // i = tid
$L__sm_exp_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__sm_exp_done;
    cvt.u64.u32 %rd6, %r6;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.f32 %f2, [%rd7];
    sub.f32 %f2, %f2, %f4;       // x - max
    // exp approximation using ex2 (2^(x * log2(e)))
    mul.f32 %f2, %f2, 0f3FB8AA3B;  // x * log2(e) = x * 1.4426950408889634
    ex2.approx.f32 %f2, %f2;
    st.global.f32 [%rd7], %f2;    // store exp(x-max) in-place
    add.f32 %f1, %f1, %f2;       // sum += exp(x-max)
    add.u32 %r6, %r6, %r5;       // i += blockDim.x
    bra $L__sm_exp_loop;
$L__sm_exp_done:

    // Store partial sum to shared
    cvt.u64.u32 %rd6, %r4;
    shl.b64 %rd6, %rd6, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd7, %rd7, %rd6;
    st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    // Reduction (sum)
    mov.u32 %r6, 128;
$L__sm_sum_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__sm_sum_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__sm_sum_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd6, %r7;
    shl.b64 %rd6, %rd6, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd6, %rd7, %rd6;
    ld.shared.f32 %f2, [%rd6];
    cvt.u64.u32 %rd6, %r4;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd6, %rd7, %rd6;
    ld.shared.f32 %f3, [%rd6];
    add.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd6], %f3;
$L__sm_sum_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__sm_sum_red;
$L__sm_sum_red_done:

    // Broadcast sum value: f5 = sdata[0]
    mov.u64 %rd7, sdata;
    ld.shared.f32 %f5, [%rd7];    // f5 = sum
    bar.sync 0;

    // === Phase 3: Divide by sum ===
    mov.u32 %r6, %r4;
$L__sm_div_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__sm_div_done;
    cvt.u64.u32 %rd6, %r6;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.f32 %f2, [%rd7];
    div.approx.f32 %f2, %f2, %f5;
    st.global.f32 [%rd7], %f2;
    add.u32 %r6, %r6, %r5;
    bra $L__sm_div_loop;
$L__sm_div_done:

$L__sm_exit:
    ret;
}

// softmax_backward_row_f32: per-row softmax backward
// For each row: dot = sum(s[i] * g[i]), result[i] = s[i] * (g[i] - dot)
// params: softmax_output, grad_output, result, num_rows, row_size
.visible .entry softmax_backward_row_f32(
    .param .u64 softmax_out,
    .param .u64 grad_out,
    .param .u64 result,
    .param .u32 num_rows,
    .param .u32 row_size
) {
    .reg .pred %p<4>;
    .reg .f32 %f<8>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;

    // shared memory for reductions (256 floats)
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [softmax_out];
    ld.param.u64 %rd2, [grad_out];
    ld.param.u64 %rd3, [result];
    ld.param.u32 %r1, [num_rows];
    ld.param.u32 %r2, [row_size];

    // row index = blockIdx.x
    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra $L__smb_exit;

    // tid = threadIdx.x, blockDim = ntid.x
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    // row_base offsets
    cvt.u64.u32 %rd4, %r3;
    cvt.u64.u32 %rd5, %r2;
    mul.lo.u64 %rd6, %rd4, %rd5;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd1, %rd6;     // rd7 = &softmax_out[row * row_size]
    add.s64 %rd8, %rd2, %rd6;     // rd8 = &grad_out[row * row_size]
    add.s64 %rd9, %rd3, %rd6;     // rd9 = &result[row * row_size]

    // === Phase 1: Compute dot = sum(s[i] * g[i]) ===
    mov.f32 %f1, 0f00000000;      // partial_dot = 0
    mov.u32 %r6, %r4;             // i = tid
$L__smb_dot_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__smb_dot_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd7, %rd10;
    ld.global.f32 %f2, [%rd11];   // s[i]
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];   // g[i]
    mul.f32 %f4, %f2, %f3;        // s[i] * g[i]
    add.f32 %f1, %f1, %f4;        // partial_dot += s[i]*g[i]
    add.u32 %r6, %r6, %r5;        // i += blockDim.x
    bra $L__smb_dot_loop;
$L__smb_dot_done:

    // Store partial dot to shared memory
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd11, %rd11, %rd10;
    st.shared.f32 [%rd11], %f1;
    bar.sync 0;

    // Reduction in shared memory (sum for dot product)
    mov.u32 %r6, 128;
$L__smb_dot_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__smb_dot_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__smb_dot_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd10, %r7;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd10, %rd11, %rd10;
    ld.shared.f32 %f2, [%rd10];
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd10, %rd11, %rd10;
    ld.shared.f32 %f3, [%rd10];
    add.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd10], %f3;
$L__smb_dot_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__smb_dot_red;
$L__smb_dot_red_done:

    // Broadcast dot: f5 = sdata[0]
    mov.u64 %rd11, sdata;
    ld.shared.f32 %f5, [%rd11];   // f5 = dot = sum(s*g)
    bar.sync 0;

    // === Phase 2: result[i] = s[i] * (g[i] - dot) ===
    mov.u32 %r6, %r4;
$L__smb_apply_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__smb_apply_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd7, %rd10;
    ld.global.f32 %f2, [%rd11];   // s[i]
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];   // g[i]
    sub.f32 %f4, %f3, %f5;        // g[i] - dot
    mul.f32 %f4, %f2, %f4;        // s[i] * (g[i] - dot)
    add.s64 %rd11, %rd9, %rd10;
    st.global.f32 [%rd11], %f4;
    add.u32 %r6, %r6, %r5;
    bra $L__smb_apply_loop;
$L__smb_apply_done:

$L__smb_exit:
    ret;
}

// broadcast_copy_f32: out[i] = src[i % src_len]
// Used for broadcast_to when a tensor needs to be expanded
.visible .entry broadcast_copy_f32(
    .param .u64 src,
    .param .u64 out,
    .param .u32 n,
    .param .u32 src_len
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [src_len];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r3, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra $L__bcopy_exit;

    // src_idx = i % src_len
    rem.u32 %r6, %r3, %r2;

    cvt.u64.u32 %rd3, %r6;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f1;

$L__bcopy_exit:
    ret;
}

// gather_contiguous_f32: out[i] = src[indices[i]]
// Makes a non-contiguous GPU tensor contiguous by gathering elements
// indices is computed on CPU and uploaded: for each output element i,
// indices[i] = offset + linear_index(unravel(i, shape), strides)
.visible .entry gather_contiguous_f32(
    .param .u64 src,
    .param .u64 indices,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [indices];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__gather_exit;

    // Load index
    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.u32 %r5, [%rd6];

    // Load src[index]
    cvt.u64.u32 %rd7, %r5;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.f32 %f1, [%rd8];

    // Store to out[i]
    add.s64 %rd9, %rd3, %rd5;
    st.global.f32 [%rd9], %f1;

$L__gather_exit:
    ret;
}
"#;

/// Embedded PTX for reduction along a dimension (sum_dim).
///
/// sum_dim_f32: Reduces a tensor along one dimension by summing.
/// The tensor is viewed as [outer_size, dim_size, inner_size].
/// Each thread computes one output element: out[outer * inner + inner_idx] = sum over dim.
/// Grid: ceil(outer_size * inner_size / 256), Block: 256.
#[cfg(feature = "cuda")]
pub const SUM_DIM_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// sum_dim_f32: out[i] = sum of input along the reduced dimension
// Tensor is logically [outer_size, dim_size, inner_size]
// Output has outer_size * inner_size elements
// Each thread handles one (outer, inner) pair
// params: input, output, outer_size, dim_size, inner_size
.visible .entry sum_dim_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size
) {
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [outer_size];
    ld.param.u32 %r2, [dim_size];
    ld.param.u32 %r3, [inner_size];

    // Global thread index
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    // Total output elements = outer_size * inner_size
    mul.lo.s32 %r7, %r1, %r3;
    setp.ge.u32 %p1, %r4, %r7;
    @%p1 bra $L__sum_dim_exit;

    // Decompose thread index into (outer, inner)
    // outer = thread_idx / inner_size
    // inner = thread_idx % inner_size
    div.u32 %r8, %r4, %r3;      // outer
    rem.u32 %r9, %r4, %r3;      // inner

    // Base index in input: outer * dim_size * inner_size + inner
    mul.lo.s32 %r10, %r8, %r2;  // outer * dim_size
    mul.lo.s32 %r10, %r10, %r3; // outer * dim_size * inner_size
    add.s32 %r10, %r10, %r9;    // + inner = base_idx

    // Stride between elements along dim: inner_size
    // Accumulate sum using 4x unrolled loop for better ILP
    mov.f32 %f1, 0f00000000;    // sum = 0.0
    mov.u32 %r11, 0;            // d = 0

    // Compute dim_size rounded down to multiple of 4
    and.b32 %r7, %r2, 0xFFFFFFFC; // dim_size & ~3

$L__sum_dim_loop4:
    setp.ge.u32 %p1, %r11, %r7;
    @%p1 bra $L__sum_dim_tail;

    // Load 4 elements with stride inner_size
    mul.lo.s32 %r8, %r11, %r3;
    add.s32 %r8, %r8, %r10;
    cvt.u64.u32 %rd3, %r8;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.f32 %f1, %f1, %f2;

    add.u32 %r8, %r11, 1;
    mul.lo.s32 %r8, %r8, %r3;
    add.s32 %r8, %r8, %r10;
    cvt.u64.u32 %rd3, %r8;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.f32 %f1, %f1, %f2;

    add.u32 %r8, %r11, 2;
    mul.lo.s32 %r8, %r8, %r3;
    add.s32 %r8, %r8, %r10;
    cvt.u64.u32 %rd3, %r8;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.f32 %f1, %f1, %f2;

    add.u32 %r8, %r11, 3;
    mul.lo.s32 %r8, %r8, %r3;
    add.s32 %r8, %r8, %r10;
    cvt.u64.u32 %rd3, %r8;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.f32 %f1, %f1, %f2;

    add.u32 %r11, %r11, 4;
    bra $L__sum_dim_loop4;

$L__sum_dim_tail:
    setp.ge.u32 %p1, %r11, %r2;
    @%p1 bra $L__sum_dim_done;

    mul.lo.s32 %r8, %r11, %r3;
    add.s32 %r8, %r8, %r10;
    cvt.u64.u32 %rd3, %r8;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    add.f32 %f1, %f1, %f2;

    add.u32 %r11, %r11, 1;
    bra $L__sum_dim_tail;

$L__sum_dim_done:
    // Store result
    cvt.u64.u32 %rd3, %r4;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f1;

$L__sum_dim_exit:
    ret;
}
"#;

/// Embedded PTX for LayerNorm kernel.
///
/// layer_norm_f32: One thread block per row. Each block computes
/// mean and variance via shared-memory parallel reduction, then
/// normalizes and applies affine: out[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
///
/// Grid: (num_rows, 1, 1), Block: (256, 1, 1), Shared: 256*4 bytes
#[cfg(feature = "cuda")]
pub const LAYERNORM_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// layer_norm_f32: per-row layer normalization with affine transform
// params: input, gamma, beta, output, norm_size, eps, num_rows
.visible .entry layer_norm_f32(
    .param .u64 input,
    .param .u64 gamma,
    .param .u64 beta,
    .param .u64 output,
    .param .u32 norm_size,
    .param .f32 eps,
    .param .u32 num_rows
) {
    .reg .pred %p<4>;
    .reg .f32 %f<10>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;

    // shared memory for reductions (256 floats)
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [gamma];
    ld.param.u64 %rd3, [beta];
    ld.param.u64 %rd4, [output];
    ld.param.u32 %r1, [norm_size];
    ld.param.f32 %f1, [eps];
    ld.param.u32 %r2, [num_rows];

    // row index = blockIdx.x
    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra $L__ln_exit;

    // tid = threadIdx.x, blockDim = ntid.x
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    // row_base = input + row_idx * norm_size * 4
    cvt.u64.u32 %rd5, %r3;
    cvt.u64.u32 %rd6, %r1;
    mul.lo.u64 %rd7, %rd5, %rd6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd8, %rd1, %rd7;     // rd8 = &input[row * norm_size]
    add.s64 %rd9, %rd4, %rd7;     // rd9 = &output[row * norm_size]

    // === Phase 1: Compute mean via parallel sum ===
    mov.f32 %f2, 0f00000000;      // partial_sum = 0
    mov.u32 %r6, %r4;             // i = tid
$L__ln_sum_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__ln_sum_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];
    add.f32 %f2, %f2, %f3;
    add.u32 %r6, %r6, %r5;        // i += blockDim.x
    bra $L__ln_sum_loop;
$L__ln_sum_done:

    // Store to shared memory
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd11, %rd11, %rd10;
    st.shared.f32 [%rd11], %f2;
    bar.sync 0;

    // Reduction in shared memory (sum for mean)
    mov.u32 %r6, 128;
$L__ln_mean_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__ln_mean_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__ln_mean_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd10, %r7;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd10, %rd11, %rd10;
    ld.shared.f32 %f3, [%rd10];
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd10, %rd11, %rd10;
    ld.shared.f32 %f4, [%rd10];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd10], %f4;
$L__ln_mean_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__ln_mean_red;
$L__ln_mean_red_done:

    // Broadcast mean: f5 = sdata[0] / norm_size
    mov.u64 %rd11, sdata;
    ld.shared.f32 %f5, [%rd11];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f5, %f5, %f6;  // f5 = mean
    bar.sync 0;

    // === Phase 2: Compute variance via parallel sum of (x - mean)^2 ===
    mov.f32 %f2, 0f00000000;       // partial_var = 0
    mov.u32 %r6, %r4;
$L__ln_var_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__ln_var_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];
    sub.f32 %f4, %f3, %f5;         // x - mean
    mul.f32 %f4, %f4, %f4;         // (x - mean)^2
    add.f32 %f2, %f2, %f4;
    add.u32 %r6, %r6, %r5;
    bra $L__ln_var_loop;
$L__ln_var_done:

    // Store to shared
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd11, %rd11, %rd10;
    st.shared.f32 [%rd11], %f2;
    bar.sync 0;

    // Reduction (sum for variance)
    mov.u32 %r6, 128;
$L__ln_var_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__ln_var_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__ln_var_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd10, %r7;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd10, %rd11, %rd10;
    ld.shared.f32 %f3, [%rd10];
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd10, %rd11, %rd10;
    ld.shared.f32 %f4, [%rd10];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd10], %f4;
$L__ln_var_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__ln_var_red;
$L__ln_var_red_done:

    // Broadcast variance: f7 = sdata[0] / norm_size
    mov.u64 %rd11, sdata;
    ld.shared.f32 %f7, [%rd11];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f7, %f7, %f6;  // f7 = variance
    bar.sync 0;

    // === Phase 3: Normalize and apply affine ===
    // inv_std = 1 / sqrt(var + eps)
    add.f32 %f8, %f7, %f1;         // var + eps
    sqrt.approx.f32 %f8, %f8;      // sqrt(var + eps)
    rcp.approx.f32 %f8, %f8;       // f8 = 1 / sqrt(var + eps)

    mov.u32 %r6, %r4;
$L__ln_norm_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__ln_norm_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;

    // Load input[row * norm_size + i]
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];

    // Load gamma[i] and beta[i]
    add.s64 %rd11, %rd2, %rd10;
    ld.global.f32 %f4, [%rd11];    // gamma
    add.s64 %rd11, %rd3, %rd10;
    ld.global.f32 %f6, [%rd11];    // beta

    // normalized = (x - mean) * inv_std
    sub.f32 %f9, %f3, %f5;
    mul.f32 %f9, %f9, %f8;
    // out = gamma * normalized + beta
    mul.f32 %f9, %f4, %f9;
    add.f32 %f9, %f9, %f6;

    // Store output[row * norm_size + i]
    add.s64 %rd11, %rd9, %rd10;
    st.global.f32 [%rd11], %f9;

    add.u32 %r6, %r6, %r5;
    bra $L__ln_norm_loop;
$L__ln_norm_done:

$L__ln_exit:
    ret;
}

// layer_norm_backward_dinput_f32: per-row d_input computation
// For each row:
//   mean = sum(x) / N
//   var = sum((x-mean)^2) / N
//   std_inv = 1/sqrt(var + eps)
//   x_hat = (x - mean) * std_inv
//   dy = grad_output * gamma
//   sum_dy = sum(dy)
//   sum_dy_xhat = sum(dy * x_hat)
//   d_input[i] = std_inv * (dy[i] - sum_dy/N - x_hat[i] * sum_dy_xhat/N)
//
// params: grad_output, input, gamma, d_input, norm_size, eps, num_rows
// Grid: (num_rows, 1, 1), Block: (256, 1, 1), Shared: 256*4*2 bytes
.visible .entry layer_norm_backward_dinput_f32(
    .param .u64 grad_output,
    .param .u64 input,
    .param .u64 gamma,
    .param .u64 d_input,
    .param .u32 norm_size,
    .param .f32 eps,
    .param .u32 num_rows
) {
    .reg .pred %p<4>;
    .reg .f32 %f<16>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;

    // Two shared memory arrays: sdata_a for first reduction, sdata_b for second
    .shared .align 4 .f32 sdata_a[256];
    .shared .align 4 .f32 sdata_b[256];

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [input];
    ld.param.u64 %rd3, [gamma];
    ld.param.u64 %rd4, [d_input];
    ld.param.u32 %r1, [norm_size];
    ld.param.f32 %f1, [eps];
    ld.param.u32 %r2, [num_rows];

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra $L__lnb_exit;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    // Row base pointers
    cvt.u64.u32 %rd5, %r3;
    cvt.u64.u32 %rd6, %r1;
    mul.lo.u64 %rd7, %rd5, %rd6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd8, %rd2, %rd7;     // &input[row * norm_size]
    add.s64 %rd9, %rd1, %rd7;     // &grad_output[row * norm_size]
    add.s64 %rd10, %rd4, %rd7;    // &d_input[row * norm_size]

    // === Phase 1: Compute mean ===
    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, %r4;
$L__lnb_mean_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__lnb_mean_done;
    cvt.u64.u32 %rd11, %r6;
    shl.b64 %rd11, %rd11, 2;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.f32 %f3, [%rd12];
    add.f32 %f2, %f2, %f3;
    add.u32 %r6, %r6, %r5;
    bra $L__lnb_mean_loop;
$L__lnb_mean_done:

    // Reduce mean in shared mem
    cvt.u64.u32 %rd11, %r4;
    shl.b64 %rd11, %rd11, 2;
    mov.u64 %rd12, sdata_a;
    add.s64 %rd12, %rd12, %rd11;
    st.shared.f32 [%rd12], %f2;
    bar.sync 0;

    mov.u32 %r6, 128;
$L__lnb_mean_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__lnb_mean_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__lnb_mean_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd11, %r7;
    shl.b64 %rd11, %rd11, 2;
    mov.u64 %rd12, sdata_a;
    add.s64 %rd11, %rd12, %rd11;
    ld.shared.f32 %f3, [%rd11];
    cvt.u64.u32 %rd11, %r4;
    shl.b64 %rd11, %rd11, 2;
    add.s64 %rd11, %rd12, %rd11;
    ld.shared.f32 %f4, [%rd11];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd11], %f4;
$L__lnb_mean_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__lnb_mean_red;
$L__lnb_mean_red_done:

    mov.u64 %rd12, sdata_a;
    ld.shared.f32 %f5, [%rd12];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f5, %f5, %f6;  // f5 = mean
    bar.sync 0;

    // === Phase 2: Compute variance ===
    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, %r4;
$L__lnb_var_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__lnb_var_done;
    cvt.u64.u32 %rd11, %r6;
    shl.b64 %rd11, %rd11, 2;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.f32 %f3, [%rd12];
    sub.f32 %f4, %f3, %f5;
    mul.f32 %f4, %f4, %f4;
    add.f32 %f2, %f2, %f4;
    add.u32 %r6, %r6, %r5;
    bra $L__lnb_var_loop;
$L__lnb_var_done:

    cvt.u64.u32 %rd11, %r4;
    shl.b64 %rd11, %rd11, 2;
    mov.u64 %rd12, sdata_a;
    add.s64 %rd12, %rd12, %rd11;
    st.shared.f32 [%rd12], %f2;
    bar.sync 0;

    mov.u32 %r6, 128;
$L__lnb_var_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__lnb_var_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__lnb_var_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd11, %r7;
    shl.b64 %rd11, %rd11, 2;
    mov.u64 %rd12, sdata_a;
    add.s64 %rd11, %rd12, %rd11;
    ld.shared.f32 %f3, [%rd11];
    cvt.u64.u32 %rd11, %r4;
    shl.b64 %rd11, %rd11, 2;
    add.s64 %rd11, %rd12, %rd11;
    ld.shared.f32 %f4, [%rd11];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd11], %f4;
$L__lnb_var_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__lnb_var_red;
$L__lnb_var_red_done:

    mov.u64 %rd12, sdata_a;
    ld.shared.f32 %f7, [%rd12];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f7, %f7, %f6;  // f7 = variance
    add.f32 %f8, %f7, %f1;         // var + eps
    sqrt.approx.f32 %f8, %f8;
    rcp.approx.f32 %f8, %f8;       // f8 = std_inv = 1/sqrt(var+eps)
    bar.sync 0;

    // === Phase 3: Compute sum_dy and sum_dy_xhat simultaneously ===
    // dy[i] = grad_output[i] * gamma[i]
    // x_hat[i] = (input[i] - mean) * std_inv
    // sum_dy = sum(dy[i])
    // sum_dy_xhat = sum(dy[i] * x_hat[i])
    mov.f32 %f2, 0f00000000;      // partial sum_dy
    mov.f32 %f3, 0f00000000;      // partial sum_dy_xhat
    mov.u32 %r6, %r4;
$L__lnb_sumdyx_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__lnb_sumdyx_done;
    cvt.u64.u32 %rd11, %r6;
    shl.b64 %rd11, %rd11, 2;

    // Load grad_output[i]
    add.s64 %rd12, %rd9, %rd11;
    ld.global.f32 %f9, [%rd12];
    // Load gamma[i]
    add.s64 %rd12, %rd3, %rd11;
    ld.global.f32 %f10, [%rd12];
    // dy = grad_output * gamma
    mul.f32 %f11, %f9, %f10;

    // Load input[i], compute x_hat
    add.s64 %rd12, %rd8, %rd11;
    ld.global.f32 %f12, [%rd12];
    sub.f32 %f13, %f12, %f5;      // x - mean
    mul.f32 %f13, %f13, %f8;      // x_hat = (x-mean)*std_inv

    // Accumulate
    add.f32 %f2, %f2, %f11;       // sum_dy += dy
    mul.f32 %f14, %f11, %f13;     // dy * x_hat
    add.f32 %f3, %f3, %f14;       // sum_dy_xhat += dy*x_hat

    add.u32 %r6, %r6, %r5;
    bra $L__lnb_sumdyx_loop;
$L__lnb_sumdyx_done:

    // Store both partials to shared memory
    cvt.u64.u32 %rd11, %r4;
    shl.b64 %rd11, %rd11, 2;
    mov.u64 %rd12, sdata_a;
    add.s64 %rd13, %rd12, %rd11;
    st.shared.f32 [%rd13], %f2;    // sdata_a[tid] = partial sum_dy
    mov.u64 %rd12, sdata_b;
    add.s64 %rd13, %rd12, %rd11;
    st.shared.f32 [%rd13], %f3;    // sdata_b[tid] = partial sum_dy_xhat
    bar.sync 0;

    // Reduce both simultaneously
    mov.u32 %r6, 128;
$L__lnb_sumdyx_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__lnb_sumdyx_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__lnb_sumdyx_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd11, %r7;
    shl.b64 %rd11, %rd11, 2;

    // Reduce sdata_a (sum_dy)
    mov.u64 %rd12, sdata_a;
    add.s64 %rd13, %rd12, %rd11;
    ld.shared.f32 %f9, [%rd13];
    cvt.u64.u32 %rd14, %r4;
    shl.b64 %rd14, %rd14, 2;
    add.s64 %rd13, %rd12, %rd14;
    ld.shared.f32 %f10, [%rd13];
    add.f32 %f10, %f10, %f9;
    st.shared.f32 [%rd13], %f10;

    // Reduce sdata_b (sum_dy_xhat)
    mov.u64 %rd12, sdata_b;
    add.s64 %rd13, %rd12, %rd11;
    ld.shared.f32 %f9, [%rd13];
    add.s64 %rd13, %rd12, %rd14;
    ld.shared.f32 %f10, [%rd13];
    add.f32 %f10, %f10, %f9;
    st.shared.f32 [%rd13], %f10;

$L__lnb_sumdyx_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra $L__lnb_sumdyx_red;
$L__lnb_sumdyx_red_done:

    // Broadcast: f9 = sum_dy, f10 = sum_dy_xhat
    mov.u64 %rd12, sdata_a;
    ld.shared.f32 %f9, [%rd12];    // sum_dy
    mov.u64 %rd12, sdata_b;
    ld.shared.f32 %f10, [%rd12];   // sum_dy_xhat
    bar.sync 0;

    // Precompute: sum_dy / N and sum_dy_xhat / N
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f11, %f9, %f6;   // f11 = sum_dy / N
    div.approx.f32 %f12, %f10, %f6;  // f12 = sum_dy_xhat / N

    // === Phase 4: Compute d_input[i] = std_inv * (dy[i] - sum_dy/N - x_hat[i] * sum_dy_xhat/N) ===
    mov.u32 %r6, %r4;
$L__lnb_dinput_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__lnb_dinput_done;
    cvt.u64.u32 %rd11, %r6;
    shl.b64 %rd11, %rd11, 2;

    // dy = grad_output[i] * gamma[i]
    add.s64 %rd12, %rd9, %rd11;
    ld.global.f32 %f2, [%rd12];    // grad_output[i]
    add.s64 %rd12, %rd3, %rd11;
    ld.global.f32 %f3, [%rd12];    // gamma[i]
    mul.f32 %f4, %f2, %f3;        // dy = grad_output * gamma

    // x_hat = (input[i] - mean) * std_inv
    add.s64 %rd12, %rd8, %rd11;
    ld.global.f32 %f2, [%rd12];    // input[i]
    sub.f32 %f3, %f2, %f5;        // x - mean
    mul.f32 %f3, %f3, %f8;        // x_hat

    // d_input = std_inv * (dy - sum_dy/N - x_hat * sum_dy_xhat/N)
    sub.f32 %f2, %f4, %f11;       // dy - sum_dy/N
    mul.f32 %f14, %f3, %f12;      // x_hat * sum_dy_xhat/N
    sub.f32 %f2, %f2, %f14;       // dy - sum_dy/N - x_hat*sum_dy_xhat/N
    mul.f32 %f2, %f8, %f2;        // std_inv * (...)

    // Store d_input[i]
    add.s64 %rd12, %rd10, %rd11;
    st.global.f32 [%rd12], %f2;

    add.u32 %r6, %r6, %r5;
    bra $L__lnb_dinput_loop;
$L__lnb_dinput_done:

$L__lnb_exit:
    ret;
}

// layer_norm_backward_dweight_dbias_f32: accumulate d_weight and d_bias
// For each element i in [0, norm_size):
//   d_bias[i] = sum over rows of grad_output[row * norm_size + i]
//   d_weight[i] = sum over rows of grad_output[row * norm_size + i] * x_hat[row, i]
// x_hat is computed from input, mean, var (recomputed per row).
// This kernel uses one thread per element (grid-stride loop over elements).
// Each thread loops over all rows and accumulates.
//
// params: grad_output, input, d_weight, d_bias, norm_size, eps, num_rows
// Grid: (ceil(norm_size/256), 1, 1), Block: (256, 1, 1)
.visible .entry layer_norm_backward_dweight_dbias_f32(
    .param .u64 grad_output,
    .param .u64 input,
    .param .u64 d_weight,
    .param .u64 d_bias,
    .param .u32 norm_size,
    .param .f32 eps,
    .param .u32 num_rows
) {
    .reg .pred %p<3>;
    .reg .f32 %f<12>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [grad_output];
    ld.param.u64 %rd2, [input];
    ld.param.u64 %rd3, [d_weight];
    ld.param.u64 %rd4, [d_bias];
    ld.param.u32 %r1, [norm_size];
    ld.param.f32 %f1, [eps];
    ld.param.u32 %r2, [num_rows];

    // Global thread index = element index
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r3, %r3, %r4, %r5;  // elem_idx

    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra $L__lnbwb_exit;

    // Loop over all rows, accumulate d_weight and d_bias for this element
    mov.f32 %f2, 0f00000000;      // acc_d_bias = 0
    mov.f32 %f3, 0f00000000;      // acc_d_weight = 0
    mov.u32 %r6, 0;               // row = 0

$L__lnbwb_row_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__lnbwb_row_done;

    // Compute row base offset = row * norm_size * 4
    cvt.u64.u32 %rd5, %r6;
    cvt.u64.u32 %rd6, %r1;
    mul.lo.u64 %rd7, %rd5, %rd6;
    shl.b64 %rd7, %rd7, 2;

    // Compute mean for this row
    // For correctness we need mean/var per row. We compute inline.
    // mean = sum(input[row*N .. row*N+N]) / N
    add.s64 %rd8, %rd2, %rd7;     // &input[row * norm_size]
    mov.f32 %f4, 0f00000000;      // sum for mean
    mov.u32 %r7, 0;
$L__lnbwb_mean_loop:
    setp.ge.u32 %p2, %r7, %r1;
    @%p2 bra $L__lnbwb_mean_done;
    cvt.u64.u32 %rd9, %r7;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.f32 %f5, [%rd10];
    add.f32 %f4, %f4, %f5;
    add.u32 %r7, %r7, 1;
    bra $L__lnbwb_mean_loop;
$L__lnbwb_mean_done:
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f4, %f4, %f6;  // f4 = mean

    // var = sum((x-mean)^2) / N
    mov.f32 %f5, 0f00000000;
    mov.u32 %r7, 0;
$L__lnbwb_var_loop:
    setp.ge.u32 %p2, %r7, %r1;
    @%p2 bra $L__lnbwb_var_done;
    cvt.u64.u32 %rd9, %r7;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.f32 %f7, [%rd10];
    sub.f32 %f7, %f7, %f4;
    mul.f32 %f7, %f7, %f7;
    add.f32 %f5, %f5, %f7;
    add.u32 %r7, %r7, 1;
    bra $L__lnbwb_var_loop;
$L__lnbwb_var_done:
    div.approx.f32 %f5, %f5, %f6;  // f5 = var
    add.f32 %f7, %f5, %f1;         // var + eps
    sqrt.approx.f32 %f7, %f7;
    rcp.approx.f32 %f7, %f7;       // f7 = std_inv

    // x_hat = (input[row*N + elem_idx] - mean) * std_inv
    cvt.u64.u32 %rd9, %r3;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.f32 %f8, [%rd10];    // input[row*N + elem]
    sub.f32 %f8, %f8, %f4;
    mul.f32 %f8, %f8, %f7;        // f8 = x_hat

    // grad_output[row*N + elem_idx]
    add.s64 %rd10, %rd1, %rd7;
    add.s64 %rd10, %rd10, %rd9;
    ld.global.f32 %f9, [%rd10];    // grad_output value

    // Accumulate
    add.f32 %f2, %f2, %f9;        // d_bias += grad_output
    mul.f32 %f10, %f9, %f8;       // grad_output * x_hat
    add.f32 %f3, %f3, %f10;       // d_weight += grad_output * x_hat

    add.u32 %r6, %r6, 1;
    bra $L__lnbwb_row_loop;
$L__lnbwb_row_done:

    // Store results
    cvt.u64.u32 %rd9, %r3;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd10, %rd3, %rd9;
    st.global.f32 [%rd10], %f3;    // d_weight[elem_idx]
    add.s64 %rd10, %rd4, %rd9;
    st.global.f32 [%rd10], %f2;    // d_bias[elem_idx]

$L__lnbwb_exit:
    ret;
}
"#;

/// CrossEntropy forward+backward kernels.
/// Forward: fused softmax + NLL loss (one block per batch row).
/// Backward: softmax_probs - one_hot(target), scaled by grad_output.
#[cfg(feature = "cuda")]
pub const CROSS_ENTROPY_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// cross_entropy_fwd_f32: One block per batch item. Each block does:
//   1. Find max of logits[b, :] (shared mem reduction)
//   2. Compute sum_exp = sum(exp(logits - max)) (shared mem reduction)
//   3. softmax_out[b,c] = exp(logits[b,c] - max) / sum_exp
//   4. losses[b] = -log(softmax_out[b, target[b]])
// params: logits[N*C], targets[N] (as f32), losses[N], softmax_out[N*C], C
// grid=(N,1,1), block=(256,1,1), shared_mem=256*4
.visible .entry cross_entropy_fwd_f32(
    .param .u64 logits,
    .param .u64 targets,
    .param .u64 losses,
    .param .u64 softmax_out,
    .param .u32 num_classes
) {
    .reg .pred %p<2>;
    .reg .f32 %f<8>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;

    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [logits];
    ld.param.u64 %rd2, [targets];
    ld.param.u64 %rd3, [losses];
    ld.param.u64 %rd4, [softmax_out];
    ld.param.u32 %r1, [num_classes];

    mov.u32 %r2, %ctaid.x;        // batch index b
    mov.u32 %r3, %tid.x;          // thread id within block
    mov.u32 %r4, %ntid.x;         // block size (256)

    // Base offset for this batch item: b * C
    mul.lo.s32 %r5, %r2, %r1;     // r5 = b * C

    // ===== Phase 1: Find max =====
    mov.f32 %f1, 0fFF800000;      // -inf
    mov.u32 %r6, %r3;             // c = tid
$L__ce_max_loop:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra $L__ce_max_done;
    add.s32 %r7, %r5, %r6;        // idx = b*C + c
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    max.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, %r4;        // c += blockDim
    bra $L__ce_max_loop;
$L__ce_max_done:

    // Store partial max in shared mem
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd6, %rd7, %rd5;
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;

    // Reduction for max
    mov.u32 %r8, 128;
$L__ce_max_reduce:
    setp.lt.u32 %p1, %r8, 1;
    @%p1 bra $L__ce_max_reduce_done;
    setp.ge.u32 %p1, %r3, %r8;
    @%p1 bra $L__ce_max_reduce_skip;
    add.u32 %r9, %r3, %r8;
    cvt.u64.u32 %rd5, %r9;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd8, %rd7, %rd5;
    ld.shared.f32 %f2, [%rd8];
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd8, %rd7, %rd5;
    ld.shared.f32 %f3, [%rd8];
    max.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd8], %f3;
$L__ce_max_reduce_skip:
    bar.sync 0;
    shr.u32 %r8, %r8, 1;
    bra $L__ce_max_reduce;
$L__ce_max_reduce_done:

    // Broadcast max to all threads
    ld.shared.f32 %f4, [sdata];    // f4 = max_val
    bar.sync 0;

    // ===== Phase 2: sum_exp = sum(exp(x - max)) =====
    mov.f32 %f1, 0f00000000;      // sum = 0
    mov.u32 %r6, %r3;
$L__ce_exp_loop:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra $L__ce_exp_done;
    add.s32 %r7, %r5, %r6;
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    sub.f32 %f2, %f2, %f4;
    // exp approx: use ex2 (base-2 exp) with log2(e) scaling
    mul.f32 %f2, %f2, 0f3FB8AA3B;  // x * log2(e) = x * 1.4426950408889634
    ex2.approx.f32 %f2, %f2;
    add.f32 %f1, %f1, %f2;
    // Store exp value temporarily in softmax_out
    add.s64 %rd8, %rd4, %rd5;
    st.global.f32 [%rd8], %f2;
    add.u32 %r6, %r6, %r4;
    bra $L__ce_exp_loop;
$L__ce_exp_done:

    // Reduce sum_exp in shared mem
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd7, %rd5;
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;

    mov.u32 %r8, 128;
$L__ce_sum_reduce:
    setp.lt.u32 %p1, %r8, 1;
    @%p1 bra $L__ce_sum_reduce_done;
    setp.ge.u32 %p1, %r3, %r8;
    @%p1 bra $L__ce_sum_reduce_skip;
    add.u32 %r9, %r3, %r8;
    cvt.u64.u32 %rd5, %r9;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd8, %rd7, %rd5;
    ld.shared.f32 %f2, [%rd8];
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd8, %rd7, %rd5;
    ld.shared.f32 %f3, [%rd8];
    add.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd8], %f3;
$L__ce_sum_reduce_skip:
    bar.sync 0;
    shr.u32 %r8, %r8, 1;
    bra $L__ce_sum_reduce;
$L__ce_sum_reduce_done:

    ld.shared.f32 %f5, [sdata];    // f5 = sum_exp
    bar.sync 0;

    // ===== Phase 3: Normalize softmax_out /= sum_exp =====
    rcp.approx.f32 %f6, %f5;      // f6 = 1/sum_exp
    mov.u32 %r6, %r3;
$L__ce_norm_loop:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra $L__ce_norm_done;
    add.s32 %r7, %r5, %r6;
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f2, %f2, %f6;        // softmax = exp_val / sum_exp
    st.global.f32 [%rd6], %f2;
    add.u32 %r6, %r6, %r4;
    bra $L__ce_norm_loop;
$L__ce_norm_done:
    bar.sync 0;

    // ===== Phase 4: Compute loss (thread 0 only) =====
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra $L__ce_exit;

    // Load target class for this batch item
    cvt.u64.u32 %rd5, %r2;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];     // target as f32
    cvt.rzi.s32.f32 %r10, %f2;    // convert to int

    // loss = log(sum_exp) + max - logits[b, target]
    // = -log_softmax[b, target] = -(logits[b,target] - max - log(sum_exp))
    // More directly: loss = -log(softmax[b, target])
    // softmax[b, target] is already stored in softmax_out
    add.s32 %r10, %r5, %r10;      // idx = b*C + target
    cvt.u64.u32 %rd5, %r10;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f2, [%rd6];     // softmax[b, target]

    // loss = -log(softmax_prob)
    // log via lg2 * ln(2): log2(x) * 0.693147 = ln(x)
    lg2.approx.f32 %f2, %f2;
    mul.f32 %f2, %f2, 0f3F317218;  // * ln(2)
    neg.f32 %f2, %f2;              // negate

    // Store loss
    cvt.u64.u32 %rd5, %r2;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd3, %rd5;
    st.global.f32 [%rd6], %f2;

$L__ce_exit:
    ret;
}

// cross_entropy_bwd_f32: grad_input[b,c] = (softmax[b,c] - (c==target[b])) * grad_output[b]
// Grid = ceil(N*C / 256), Block = 256
// params: softmax_probs[N*C], targets[N](f32), grad_output[N], grad_input[N*C], N, C
.visible .entry cross_entropy_bwd_f32(
    .param .u64 softmax_probs,
    .param .u64 targets,
    .param .u64 grad_output,
    .param .u64 grad_input,
    .param .u32 batch_size,
    .param .u32 num_classes
) {
    .reg .pred %p<3>;
    .reg .f32 %f<6>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [softmax_probs];
    ld.param.u64 %rd2, [targets];
    ld.param.u64 %rd3, [grad_output];
    ld.param.u64 %rd4, [grad_input];
    ld.param.u32 %r1, [batch_size];
    ld.param.u32 %r2, [num_classes];

    // Global thread index
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r3, %r3, %r4, %r5;

    // Total elements = N * C
    mul.lo.s32 %r6, %r1, %r2;
    setp.ge.u32 %p1, %r3, %r6;
    @%p1 bra $L__cebwd_exit;

    // b = idx / C, c = idx % C
    div.u32 %r7, %r3, %r2;        // b
    rem.u32 %r8, %r3, %r2;        // c

    // Load softmax[b, c]
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f1, [%rd6];     // softmax_prob

    // Load target[b]
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];
    cvt.rzi.s32.f32 %r9, %f2;     // target class as int

    // grad = softmax - (c == target ? 1 : 0)
    setp.eq.s32 %p2, %r8, %r9;
    @%p2 sub.f32 %f1, %f1, 0f3F800000;  // subtract 1.0 if c == target

    // Scale by grad_output[b]
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f3, [%rd6];
    mul.f32 %f1, %f1, %f3;

    // Store grad_input[b, c]
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    st.global.f32 [%rd6], %f1;

$L__cebwd_exit:
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
                "gelu_f32",
                "silu_f32",
            ],
        )?;

        // Load broadcast element-wise kernels
        kernels.load_module(
            "broadcast",
            BROADCAST_PTX,
            &[
                "broadcast_add_f32", "broadcast_sub_f32",
                "broadcast_mul_f32", "broadcast_div_f32",
                "broadcast_add_rev_f32", "broadcast_sub_rev_f32",
                "broadcast_mul_rev_f32", "broadcast_div_rev_f32",
            ],
        )?;

        // Load reduction kernels (softmax, etc.)
        kernels.load_module(
            "reduction",
            REDUCTION_PTX,
            &[
                "softmax_row_f32",
                "softmax_backward_row_f32",
                "broadcast_copy_f32",
                "gather_contiguous_f32",
            ],
        )?;

        // Load sum_dim reduction kernel
        kernels.load_module(
            "sum_dim",
            SUM_DIM_PTX,
            &["sum_dim_f32"],
        )?;

        // Load LayerNorm kernels (forward + backward)
        kernels.load_module(
            "layernorm",
            LAYERNORM_PTX,
            &[
                "layer_norm_f32",
                "layer_norm_backward_dinput_f32",
                "layer_norm_backward_dweight_dbias_f32",
            ],
        )?;

        // Load CrossEntropy kernels (forward + backward)
        kernels.load_module(
            "cross_entropy",
            CROSS_ENTROPY_PTX,
            &["cross_entropy_fwd_f32", "cross_entropy_bwd_f32"],
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
            .map_err(|e| {
                eprintln!("[AxonML CUDA] Failed to load module '{}': {}", name, e);
                CudaError::ModuleLoadFailed(e.to_string())
            })?;

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
