//! CUDA kernel registry — 17 PTX modules with 60+ kernel entry points.
//!
//! 3828 lines. Each PTX module is compiled from a `.cu` source file via
//! `nvcc -ptx -arch=sm_80 --use_fast_math` and embedded at compile time via
//! `include_str!`. `CudaKernels` loads all modules at backend init and stores
//! each kernel function handle in a HashMap for O(1) dispatch. Modules cover
//! elementwise ops (add, mul, scalar, neg, abs, sign, pow), activations
//! (relu, sigmoid, tanh, gelu, silu, elu, leaky_relu), softmax, layernorm,
//! RMSNorm, transpose, embedding gather, dropout, fused attention (forward +
//! backward + flash-decode + flash-prefill), and quantized matmul (Q4_K +
//! Q6_K dequant-in-shader GEMV/GEMM with cooperative warp reduction).
//! `launch_config(n)` provides standard grid/block dims.
//!
//! # File
//! `crates/axonml-core/src/backends/cuda_kernels/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#[cfg(feature = "cuda")]
use alloc::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use super::cuda::CudaError;

/// Block size for kernel launches (256 threads per block is typical optimal)
pub const BLOCK_SIZE: u32 = 256;

/// Embedded PTX for element-wise operations
#[cfg(feature = "cuda")]
pub const ELEMENTWISE_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

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
    abs.f32 %f3, %f1;
    lg2.approx.f32 %f3, %f3;
    mul.f32 %f3, %f2, %f3;
    ex2.approx.f32 %f4, %f3;
    st.global.f32 [%rd8], %f4;

$L__pow_exit:
    ret;
}

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
";

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
pub const BROADCAST_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

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
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__ba_exit:
    ret;
}

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
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__bar_exit:
    ret;
}

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
";

/// Embedded PTX for activation functions
#[cfg(feature = "cuda")]
pub const ACTIVATIONS_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

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
    mov.f32 %f3, 0f3F800000;
    sub.f32 %f3, %f3, %f2;
    mul.f32 %f4, %f2, %f3;
    mul.f32 %f4, %f1, %f4;
    st.global.f32 [%rd8], %f4;

$L__sigb_exit:
    ret;
}

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
    mul.f32 %f3, %f2, %f2;
    mov.f32 %f4, 0f3F800000;
    sub.f32 %f4, %f4, %f3;
    mul.f32 %f4, %f1, %f4;
    st.global.f32 [%rd8], %f4;

$L__tanhb_exit:
    ret;
}

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
    mul.f32 %f1, %f1, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f1;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f2;

$L__exp_exit:
    ret;
}

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
    lg2.approx.f32 %f1, %f1;
    mul.f32 %f2, %f1, 0f3F317218;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f2;

$L__log_exit:
    ret;
}

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
    mul.f32 %f2, %f1, %f1;
    mul.f32 %f2, %f2, %f1;
    mul.f32 %f3, %f2, 0f3D372713;
    add.f32 %f4, %f1, %f3;
    mul.f32 %f5, %f4, 0f3F4C422A;
    mul.f32 %f6, %f5, 0f40000000;
    mul.f32 %f6, %f6, 0f3FB8AA3B;
    ex2.approx.f32 %f7, %f6;
    add.f32 %f8, %f7, 0fBF800000;
    add.f32 %f9, %f7, 0f3F800000;
    div.approx.f32 %f10, %f8, %f9;
    add.f32 %f10, %f10, 0f3F800000;
    mul.f32 %f11, %f1, %f10;
    mul.f32 %f11, %f11, 0f3F000000;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f11;

$L__gelu_exit:
    ret;
}

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
    neg.f32 %f2, %f1;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f2;
    add.f32 %f4, %f3, 0f3F800000;
    rcp.approx.f32 %f5, %f4;
    mul.f32 %f5, %f1, %f5;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f5;

$L__silu_exit:
    ret;
}

.visible .entry silu_backward_f32(
    .param .u64 silu_backward_f32_param_0,
    .param .u64 silu_backward_f32_param_1,
    .param .u64 silu_backward_f32_param_2,
    .param .u32 silu_backward_f32_param_3
)
{
    .reg .pred     %p<2>;
    .reg .f32     %f<12>;
    .reg .b32     %r<6>;
    .reg .b64     %rd<11>;

    ld.param.u64     %rd1, [silu_backward_f32_param_0];
    ld.param.u64     %rd2, [silu_backward_f32_param_1];
    ld.param.u64     %rd3, [silu_backward_f32_param_2];
    ld.param.u32     %r2, [silu_backward_f32_param_3];
    mov.u32     %r3, %ctaid.x;
    mov.u32     %r4, %ntid.x;
    mov.u32     %r5, %tid.x;
    mad.lo.s32     %r1, %r3, %r4, %r5;
    setp.ge.u32     %p1, %r1, %r2;
    @%p1 bra     $L__silu_bwd_exit;

    cvta.to.global.u64     %rd4, %rd1;
    mul.wide.u32     %rd5, %r1, 4;
    add.s64     %rd6, %rd4, %rd5;
    ld.global.nc.f32     %f1, [%rd6];
    mul.ftz.f32     %f2, %f1, 0fBFB8AA3B;
    ex2.approx.ftz.f32     %f3, %f2;
    add.ftz.f32     %f4, %f3, 0f3F800000;
    mov.f32     %f5, 0f3F800000;
    rcp.approx.ftz.f32     %f6, %f4;
    sub.ftz.f32     %f7, %f5, %f6;
    fma.rn.ftz.f32     %f8, %f1, %f7, 0f3F800000;
    mul.ftz.f32     %f9, %f6, %f8;
    cvta.to.global.u64     %rd7, %rd2;
    add.s64     %rd8, %rd7, %rd5;
    ld.global.nc.f32     %f10, [%rd8];
    mul.ftz.f32     %f11, %f10, %f9;
    cvta.to.global.u64     %rd9, %rd3;
    add.s64     %rd10, %rd9, %rd5;
    st.global.f32     [%rd10], %f11;

$L__silu_bwd_exit:
    ret;
}

";

/// Embedded PTX for reduction operations (softmax, etc.)
///
/// softmax_row_f32: Computes softmax along the last dimension.
/// Each block handles one row of `row_size` elements.
/// Grid: (num_rows), Block: (256).
/// Algorithm: max → subtract max → exp → sum → divide.
/// Handles row_size > blockDim.x via sequential loops + shared reduction.
#[cfg(feature = "cuda")]
pub const REDUCTION_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

.visible .entry softmax_row_f32(
    .param .u64 data,
    .param .u32 num_rows,
    .param .u32 row_size
) {
    .reg .pred %p<4>;
    .reg .f32 %f<8>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;

    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [data];
    ld.param.u32 %r1, [num_rows];
    ld.param.u32 %r2, [row_size];

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra $L__sm_exit;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    cvt.u64.u32 %rd2, %r3;
    cvt.u64.u32 %rd3, %r2;
    mul.lo.u64 %rd4, %rd2, %rd3;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd5, %rd1, %rd4;

    // === Phase 1: Find max value in row ===
    mov.f32 %f1, 0fFF800000;
    mov.u32 %r6, %r4;
$L__sm_max_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__sm_max_done;
    cvt.u64.u32 %rd6, %r6;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.f32 %f2, [%rd7];
    max.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, %r5;
    bra $L__sm_max_loop;
$L__sm_max_done:

    cvt.u64.u32 %rd6, %r4;
    shl.b64 %rd6, %rd6, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd7, %rd7, %rd6;
    st.shared.f32 [%rd7], %f1;
    bar.sync 0;

    mov.u32 %r6, 128;
$L__sm_max_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__sm_max_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__sm_max_red_skip;
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

    mov.u64 %rd7, sdata;
    ld.shared.f32 %f4, [%rd7];
    bar.sync 0;

    // === Phase 2: exp(x - max) and sum ===
    mov.f32 %f1, 0f00000000;
    mov.u32 %r6, %r4;
$L__sm_exp_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__sm_exp_done;
    cvt.u64.u32 %rd6, %r6;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.f32 %f2, [%rd7];
    sub.f32 %f2, %f2, %f4;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f2;
    st.global.f32 [%rd7], %f2;
    add.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, %r5;
    bra $L__sm_exp_loop;
$L__sm_exp_done:

    cvt.u64.u32 %rd6, %r4;
    shl.b64 %rd6, %rd6, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd7, %rd7, %rd6;
    st.shared.f32 [%rd7], %f1;
    bar.sync 0;

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

    mov.u64 %rd7, sdata;
    ld.shared.f32 %f5, [%rd7];
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

    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [softmax_out];
    ld.param.u64 %rd2, [grad_out];
    ld.param.u64 %rd3, [result];
    ld.param.u32 %r1, [num_rows];
    ld.param.u32 %r2, [row_size];

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra $L__smb_exit;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    cvt.u64.u32 %rd4, %r3;
    cvt.u64.u32 %rd5, %r2;
    mul.lo.u64 %rd6, %rd4, %rd5;
    shl.b64 %rd6, %rd6, 2;
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;

    // === Phase 1: Compute dot = sum(s[i] * g[i]) ===
    mov.f32 %f1, 0f00000000;
    mov.u32 %r6, %r4;
$L__smb_dot_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__smb_dot_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd7, %rd10;
    ld.global.f32 %f2, [%rd11];
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r6, %r6, %r5;
    bra $L__smb_dot_loop;
$L__smb_dot_done:

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd11, %rd11, %rd10;
    st.shared.f32 [%rd11], %f1;
    bar.sync 0;

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

    mov.u64 %rd11, sdata;
    ld.shared.f32 %f5, [%rd11];
    bar.sync 0;

    // === Phase 2: result[i] = s[i] * (g[i] - dot) ===
    mov.u32 %r6, %r4;
$L__smb_apply_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__smb_apply_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd7, %rd10;
    ld.global.f32 %f2, [%rd11];
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];
    sub.f32 %f4, %f3, %f5;
    mul.f32 %f4, %f2, %f4;
    add.s64 %rd11, %rd9, %rd10;
    st.global.f32 [%rd11], %f4;
    add.u32 %r6, %r6, %r5;
    bra $L__smb_apply_loop;
$L__smb_apply_done:

$L__smb_exit:
    ret;
}

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

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.u32 %r5, [%rd6];

    cvt.u64.u32 %rd7, %r5;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.f32 %f1, [%rd8];

    add.s64 %rd9, %rd3, %rd5;
    st.global.f32 [%rd9], %f1;

$L__gather_exit:
    ret;
}
";

/// Embedded PTX for argmax/argmin along a dimension (argdim).
///
/// argmax_dim_f32 / argmin_dim_f32: reduce a tensor along one dimension,
/// returning the INDEX (cast to f32) of the extreme element. The tensor is
/// viewed as [outer_size, dim_size, inner_size] exactly like sum_dim_f32.
/// Each thread computes one output element:
///   out[outer*inner_size + inner] = argextreme over d of
///       input[outer*dim_size*inner_size + d*inner_size + inner]
/// Ties resolve to the lowest index (strict gt/lt), matching NumPy and
/// CpuBackend::argmax/argmin. Grid: ceil(outer_size*inner_size/256), Block: 256.
#[cfg(feature = "cuda")]
pub const ARGDIM_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

.visible .entry argmax_dim_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size
) {
    .reg .pred %p<3>;
    .reg .f32 %f<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [outer_size];
    ld.param.u32 %r2, [dim_size];
    ld.param.u32 %r3, [inner_size];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    mul.lo.s32 %r7, %r1, %r3;
    setp.ge.u32 %p1, %r4, %r7;
    @%p1 bra $L__argmax_exit;

    div.u32 %r8, %r4, %r3;
    rem.u32 %r9, %r4, %r3;

    mul.lo.s32 %r10, %r8, %r2;
    mul.lo.s32 %r10, %r10, %r3;
    add.s32 %r10, %r10, %r9;

    cvt.u64.u32 %rd3, %r10;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    mov.u32 %r12, 0;
    mov.u32 %r11, 1;

$L__argmax_loop:
    setp.ge.u32 %p1, %r11, %r2;
    @%p1 bra $L__argmax_done;

    mul.lo.s32 %r13, %r11, %r3;
    add.s32 %r13, %r13, %r10;
    cvt.u64.u32 %rd3, %r13;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    setp.gt.f32 %p2, %f2, %f1;
    @%p2 mov.f32 %f1, %f2;
    @%p2 mov.u32 %r12, %r11;

    add.u32 %r11, %r11, 1;
    bra $L__argmax_loop;

$L__argmax_done:
    cvt.rn.f32.u32 %f2, %r12;
    cvt.u64.u32 %rd3, %r4;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f2;

$L__argmax_exit:
    ret;
}

.visible .entry argmin_dim_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size
) {
    .reg .pred %p<3>;
    .reg .f32 %f<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [outer_size];
    ld.param.u32 %r2, [dim_size];
    ld.param.u32 %r3, [inner_size];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    mul.lo.s32 %r7, %r1, %r3;
    setp.ge.u32 %p1, %r4, %r7;
    @%p1 bra $L__argmin_exit;

    div.u32 %r8, %r4, %r3;
    rem.u32 %r9, %r4, %r3;

    mul.lo.s32 %r10, %r8, %r2;
    mul.lo.s32 %r10, %r10, %r3;
    add.s32 %r10, %r10, %r9;

    cvt.u64.u32 %rd3, %r10;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    mov.u32 %r12, 0;
    mov.u32 %r11, 1;

$L__argmin_loop:
    setp.ge.u32 %p1, %r11, %r2;
    @%p1 bra $L__argmin_done;

    mul.lo.s32 %r13, %r11, %r3;
    add.s32 %r13, %r13, %r10;
    cvt.u64.u32 %rd3, %r13;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    setp.lt.f32 %p2, %f2, %f1;
    @%p2 mov.f32 %f1, %f2;
    @%p2 mov.u32 %r12, %r11;

    add.u32 %r11, %r11, 1;
    bra $L__argmin_loop;

$L__argmin_done:
    cvt.rn.f32.u32 %f2, %r12;
    cvt.u64.u32 %rd3, %r4;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f2;

$L__argmin_exit:
    ret;
}
";

/// Embedded PTX for reduction along a dimension (sum_dim).
///
/// sum_dim_f32: Reduces a tensor along one dimension by summing.
/// The tensor is viewed as [outer_size, dim_size, inner_size].
/// Each thread computes one output element: out[outer * inner + inner_idx] = sum over dim.
/// Grid: ceil(outer_size * inner_size / 256), Block: 256.
#[cfg(feature = "cuda")]
pub const SUM_DIM_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

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

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    mul.lo.s32 %r7, %r1, %r3;
    setp.ge.u32 %p1, %r4, %r7;
    @%p1 bra $L__sum_dim_exit;

    div.u32 %r8, %r4, %r3;
    rem.u32 %r9, %r4, %r3;

    mul.lo.s32 %r10, %r8, %r2;
    mul.lo.s32 %r10, %r10, %r3;
    add.s32 %r10, %r10, %r9;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;

    and.b32 %r7, %r2, 0xFFFFFFFC;

$L__sum_dim_loop4:
    setp.ge.u32 %p1, %r11, %r7;
    @%p1 bra $L__sum_dim_tail;

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
    cvt.u64.u32 %rd3, %r4;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f1;

$L__sum_dim_exit:
    ret;
}
";

/// Embedded PTX for LayerNorm kernel.
///
/// layer_norm_f32: One thread block per row. Each block computes
/// mean and variance via shared-memory parallel reduction, then
/// normalizes and applies affine: out[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
///
/// Grid: (num_rows, 1, 1), Block: (256, 1, 1), Shared: 256*4 bytes
#[cfg(feature = "cuda")]
pub const LAYERNORM_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

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

    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [gamma];
    ld.param.u64 %rd3, [beta];
    ld.param.u64 %rd4, [output];
    ld.param.u32 %r1, [norm_size];
    ld.param.f32 %f1, [eps];
    ld.param.u32 %r2, [num_rows];

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra $L__ln_exit;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    cvt.u64.u32 %rd5, %r3;
    cvt.u64.u32 %rd6, %r1;
    mul.lo.u64 %rd7, %rd5, %rd6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd8, %rd1, %rd7;
    add.s64 %rd9, %rd4, %rd7;

    // === Phase 1: Compute mean via parallel sum ===
    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, %r4;
$L__ln_sum_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__ln_sum_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];
    add.f32 %f2, %f2, %f3;
    add.u32 %r6, %r6, %r5;
    bra $L__ln_sum_loop;
$L__ln_sum_done:

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd11, %rd11, %rd10;
    st.shared.f32 [%rd11], %f2;
    bar.sync 0;

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

    mov.u64 %rd11, sdata;
    ld.shared.f32 %f5, [%rd11];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f5, %f5, %f6;
    bar.sync 0;

    // === Phase 2: Compute variance via parallel sum of (x - mean)^2 ===
    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, %r4;
$L__ln_var_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__ln_var_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];
    sub.f32 %f4, %f3, %f5;
    mul.f32 %f4, %f4, %f4;
    add.f32 %f2, %f2, %f4;
    add.u32 %r6, %r6, %r5;
    bra $L__ln_var_loop;
$L__ln_var_done:

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    mov.u64 %rd11, sdata;
    add.s64 %rd11, %rd11, %rd10;
    st.shared.f32 [%rd11], %f2;
    bar.sync 0;

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

    mov.u64 %rd11, sdata;
    ld.shared.f32 %f7, [%rd11];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f7, %f7, %f6;
    bar.sync 0;

    // === Phase 3: Normalize and apply affine ===
    add.f32 %f8, %f7, %f1;
    sqrt.approx.f32 %f8, %f8;
    rcp.approx.f32 %f8, %f8;

    mov.u32 %r6, %r4;
$L__ln_norm_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__ln_norm_done;
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;

    add.s64 %rd11, %rd8, %rd10;
    ld.global.f32 %f3, [%rd11];

    add.s64 %rd11, %rd2, %rd10;
    ld.global.f32 %f4, [%rd11];
    add.s64 %rd11, %rd3, %rd10;
    ld.global.f32 %f6, [%rd11];

    sub.f32 %f9, %f3, %f5;
    mul.f32 %f9, %f9, %f8;
    mul.f32 %f9, %f4, %f9;
    add.f32 %f9, %f9, %f6;

    add.s64 %rd11, %rd9, %rd10;
    st.global.f32 [%rd11], %f9;

    add.u32 %r6, %r6, %r5;
    bra $L__ln_norm_loop;
$L__ln_norm_done:

$L__ln_exit:
    ret;
}

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

    cvt.u64.u32 %rd5, %r3;
    cvt.u64.u32 %rd6, %r1;
    mul.lo.u64 %rd7, %rd5, %rd6;
    shl.b64 %rd7, %rd7, 2;
    add.s64 %rd8, %rd2, %rd7;
    add.s64 %rd9, %rd1, %rd7;
    add.s64 %rd10, %rd4, %rd7;

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
    div.approx.f32 %f5, %f5, %f6;
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
    div.approx.f32 %f7, %f7, %f6;
    add.f32 %f8, %f7, %f1;
    sqrt.approx.f32 %f8, %f8;
    rcp.approx.f32 %f8, %f8;
    bar.sync 0;

    // === Phase 3: Compute sum_dy and sum_dy_xhat simultaneously ===
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r6, %r4;
$L__lnb_sumdyx_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__lnb_sumdyx_done;
    cvt.u64.u32 %rd11, %r6;
    shl.b64 %rd11, %rd11, 2;

    add.s64 %rd12, %rd9, %rd11;
    ld.global.f32 %f9, [%rd12];
    add.s64 %rd12, %rd3, %rd11;
    ld.global.f32 %f10, [%rd12];
    mul.f32 %f11, %f9, %f10;

    add.s64 %rd12, %rd8, %rd11;
    ld.global.f32 %f12, [%rd12];
    sub.f32 %f13, %f12, %f5;
    mul.f32 %f13, %f13, %f8;

    add.f32 %f2, %f2, %f11;
    mul.f32 %f14, %f11, %f13;
    add.f32 %f3, %f3, %f14;

    add.u32 %r6, %r6, %r5;
    bra $L__lnb_sumdyx_loop;
$L__lnb_sumdyx_done:

    cvt.u64.u32 %rd11, %r4;
    shl.b64 %rd11, %rd11, 2;
    mov.u64 %rd12, sdata_a;
    add.s64 %rd13, %rd12, %rd11;
    st.shared.f32 [%rd13], %f2;
    mov.u64 %rd12, sdata_b;
    add.s64 %rd13, %rd12, %rd11;
    st.shared.f32 [%rd13], %f3;
    bar.sync 0;

    mov.u32 %r6, 128;
$L__lnb_sumdyx_red:
    setp.lt.u32 %p2, %r6, 1;
    @%p2 bra $L__lnb_sumdyx_red_done;
    setp.ge.u32 %p3, %r4, %r6;
    @%p3 bra $L__lnb_sumdyx_red_skip;
    add.u32 %r7, %r4, %r6;
    cvt.u64.u32 %rd11, %r7;
    shl.b64 %rd11, %rd11, 2;

    mov.u64 %rd12, sdata_a;
    add.s64 %rd13, %rd12, %rd11;
    ld.shared.f32 %f9, [%rd13];
    cvt.u64.u32 %rd14, %r4;
    shl.b64 %rd14, %rd14, 2;
    add.s64 %rd13, %rd12, %rd14;
    ld.shared.f32 %f10, [%rd13];
    add.f32 %f10, %f10, %f9;
    st.shared.f32 [%rd13], %f10;

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

    mov.u64 %rd12, sdata_a;
    ld.shared.f32 %f9, [%rd12];
    mov.u64 %rd12, sdata_b;
    ld.shared.f32 %f10, [%rd12];
    bar.sync 0;

    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f11, %f9, %f6;
    div.approx.f32 %f12, %f10, %f6;

    // === Phase 4: Compute d_input[i] = std_inv * (dy[i] - sum_dy/N - x_hat[i] * sum_dy_xhat/N) ===
    mov.u32 %r6, %r4;
$L__lnb_dinput_loop:
    setp.ge.u32 %p2, %r6, %r1;
    @%p2 bra $L__lnb_dinput_done;
    cvt.u64.u32 %rd11, %r6;
    shl.b64 %rd11, %rd11, 2;

    add.s64 %rd12, %rd9, %rd11;
    ld.global.f32 %f2, [%rd12];
    add.s64 %rd12, %rd3, %rd11;
    ld.global.f32 %f3, [%rd12];
    mul.f32 %f4, %f2, %f3;

    add.s64 %rd12, %rd8, %rd11;
    ld.global.f32 %f2, [%rd12];
    sub.f32 %f3, %f2, %f5;
    mul.f32 %f3, %f3, %f8;

    sub.f32 %f2, %f4, %f11;
    mul.f32 %f14, %f3, %f12;
    sub.f32 %f2, %f2, %f14;
    mul.f32 %f2, %f8, %f2;

    add.s64 %rd12, %rd10, %rd11;
    st.global.f32 [%rd12], %f2;

    add.u32 %r6, %r6, %r5;
    bra $L__lnb_dinput_loop;
$L__lnb_dinput_done:

$L__lnb_exit:
    ret;
}

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

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r3, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra $L__lnbwb_exit;

    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r6, 0;

$L__lnbwb_row_loop:
    setp.ge.u32 %p2, %r6, %r2;
    @%p2 bra $L__lnbwb_row_done;

    cvt.u64.u32 %rd5, %r6;
    cvt.u64.u32 %rd6, %r1;
    mul.lo.u64 %rd7, %rd5, %rd6;
    shl.b64 %rd7, %rd7, 2;

    add.s64 %rd8, %rd2, %rd7;
    mov.f32 %f4, 0f00000000;
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
    div.approx.f32 %f4, %f4, %f6;

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
    div.approx.f32 %f5, %f5, %f6;
    add.f32 %f7, %f5, %f1;
    sqrt.approx.f32 %f7, %f7;
    rcp.approx.f32 %f7, %f7;

    cvt.u64.u32 %rd9, %r3;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.f32 %f8, [%rd10];
    sub.f32 %f8, %f8, %f4;
    mul.f32 %f8, %f8, %f7;

    add.s64 %rd10, %rd1, %rd7;
    add.s64 %rd10, %rd10, %rd9;
    ld.global.f32 %f9, [%rd10];

    add.f32 %f2, %f2, %f9;
    mul.f32 %f10, %f9, %f8;
    add.f32 %f3, %f3, %f10;

    add.u32 %r6, %r6, 1;
    bra $L__lnbwb_row_loop;
$L__lnbwb_row_done:

    cvt.u64.u32 %rd9, %r3;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd10, %rd3, %rd9;
    st.global.f32 [%rd10], %f3;
    add.s64 %rd10, %rd4, %rd9;
    st.global.f32 [%rd10], %f2;

$L__lnbwb_exit:
    ret;
}
";

/// CrossEntropy forward+backward kernels.
/// Forward: fused softmax + NLL loss (one block per batch row).
/// Backward: softmax_probs - one_hot(target), scaled by grad_output.
#[cfg(feature = "cuda")]
pub const CROSS_ENTROPY_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

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

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ntid.x;

    mul.lo.s32 %r5, %r2, %r1;

    // ===== Phase 1: Find max =====
    mov.f32 %f1, 0fFF800000;
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    mov.u64 %rd9, sdata;
    add.s64 %rd8, %rd9, %rd5;
    st.shared.f32 [%rd8], %f1;
    bar.sync 0;

    mov.u32 %r6, %r3;
$L__ce_max_loop:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra $L__ce_max_done;
    add.s32 %r7, %r5, %r6;
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    max.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, %r4;
    bra $L__ce_max_loop;
$L__ce_max_done:

    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    mov.u64 %rd7, sdata;
    add.s64 %rd6, %rd7, %rd5;
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;

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

    ld.shared.f32 %f4, [sdata];
    bar.sync 0;

    // ===== Phase 2: sum_exp = sum(exp(x - max)) =====
    mov.f32 %f1, 0f00000000;
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
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f2;
    add.f32 %f1, %f1, %f2;
    add.s64 %rd8, %rd4, %rd5;
    st.global.f32 [%rd8], %f2;
    add.u32 %r6, %r6, %r4;
    bra $L__ce_exp_loop;
$L__ce_exp_done:

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

    ld.shared.f32 %f5, [sdata];
    bar.sync 0;

    // ===== Phase 3: Normalize softmax_out /= sum_exp =====
    rcp.approx.f32 %f6, %f5;
    mov.u32 %r6, %r3;
$L__ce_norm_loop:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra $L__ce_norm_done;
    add.s32 %r7, %r5, %r6;
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f2, %f2, %f6;
    st.global.f32 [%rd6], %f2;
    add.u32 %r6, %r6, %r4;
    bra $L__ce_norm_loop;
$L__ce_norm_done:
    bar.sync 0;

    // ===== Phase 4: Compute loss (thread 0 only) =====
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra $L__ce_exit;

    cvt.u64.u32 %rd5, %r2;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];
    cvt.rzi.s32.f32 %r10, %f2;

    add.s32 %r10, %r5, %r10;
    cvt.u64.u32 %rd5, %r10;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f2, [%rd6];

    lg2.approx.f32 %f2, %f2;
    mul.f32 %f2, %f2, 0f3F317218;
    neg.f32 %f2, %f2;

    cvt.u64.u32 %rd5, %r2;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd3, %rd5;
    st.global.f32 [%rd6], %f2;

$L__ce_exit:
    ret;
}

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

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r3, %r3, %r4, %r5;

    mul.lo.s32 %r6, %r1, %r2;
    setp.ge.u32 %p1, %r3, %r6;
    @%p1 bra $L__cebwd_exit;

    div.u32 %r7, %r3, %r2;
    rem.u32 %r8, %r3, %r2;

    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f1, [%rd6];

    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];
    cvt.rzi.s32.f32 %r9, %f2;

    setp.eq.s32 %p2, %r8, %r9;
    @%p2 sub.f32 %f1, %f1, 0f3F800000;

    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f3, [%rd6];
    mul.f32 %f1, %f1, %f3;

    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    st.global.f32 [%rd6], %f1;

$L__cebwd_exit:
    ret;
}
";

/// PTX for embedding scatter-add backward (GPU-native gradient accumulation)
/// Each thread handles one element: given (token_index, dim_offset),
/// atomically adds grad_output[token_index * emb_dim + dim_offset] to
/// weight_grad[indices[token_index] * emb_dim + dim_offset].
#[cfg(feature = "cuda")]
pub const EMBEDDING_SCATTER_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

.visible .entry embedding_scatter_add_f32(
    .param .u64 p_grad_src,
    .param .u64 p_indices,
    .param .u64 p_weight_grad,
    .param .u32 p_total_n,
    .param .u32 p_emb_dim
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<12>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    ld.param.u32 %r4, [p_total_n];
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $L__scatter_exit;

    ld.param.u32 %r5, [p_emb_dim];
    div.u32 %r6, %r1, %r5;
    rem.u32 %r7, %r1, %r5;

    ld.param.u64 %rd1, [p_indices];
    cvt.u64.u32 %rd2, %r6;
    shl.b64 %rd2, %rd2, 2;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r8, [%rd3];

    ld.param.u64 %rd4, [p_grad_src];
    cvt.u64.u32 %rd5, %r1;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];

    mad.lo.u32 %r9, %r8, %r5, %r7;
    ld.param.u64 %rd7, [p_weight_grad];
    cvt.u64.u32 %rd8, %r9;
    shl.b64 %rd8, %rd8, 2;
    add.s64 %rd9, %rd7, %rd8;

    atom.global.add.f32 %f1, [%rd9], %f1;

$L__scatter_exit:
    ret;
}
";

/// PTX for fused Adam optimizer step (GPU-native parameter update)
/// Each thread handles one element: updates param, exp_avg, exp_avg_sq in-place.
/// Eliminates the GPU->CPU->GPU copy that standard Adam does per step.
#[cfg(feature = "cuda")]
pub const ADAM_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

.visible .entry adam_step_f32(
    .param .u64 p_param,
    .param .u64 p_grad,
    .param .u64 p_exp_avg,
    .param .u64 p_exp_avg_sq,
    .param .u32 p_n,
    .param .f32 p_lr,
    .param .f32 p_beta1,
    .param .f32 p_beta2,
    .param .f32 p_eps,
    .param .f32 p_weight_decay,
    .param .f32 p_bias_correction1,
    .param .f32 p_bias_correction2
) {
    .reg .pred %p<2>;
    .reg .f32 %f<20>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<12>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $L__adam_exit;

    cvt.u64.u32 %rd1, %r1;
    shl.b64 %rd1, %rd1, 2;

    ld.param.u64 %rd2, [p_param];
    ld.param.u64 %rd3, [p_grad];
    ld.param.u64 %rd4, [p_exp_avg];
    ld.param.u64 %rd5, [p_exp_avg_sq];

    add.s64 %rd6, %rd2, %rd1;
    ld.global.f32 %f1, [%rd6];
    add.s64 %rd7, %rd3, %rd1;
    ld.global.f32 %f2, [%rd7];
    add.s64 %rd8, %rd4, %rd1;
    ld.global.f32 %f3, [%rd8];
    add.s64 %rd9, %rd5, %rd1;
    ld.global.f32 %f4, [%rd9];

    ld.param.f32 %f5, [p_lr];
    ld.param.f32 %f6, [p_beta1];
    ld.param.f32 %f7, [p_beta2];
    ld.param.f32 %f8, [p_eps];
    ld.param.f32 %f9, [p_weight_decay];
    ld.param.f32 %f10, [p_bias_correction1];
    ld.param.f32 %f11, [p_bias_correction2];

    mul.f32 %f12, %f9, %f1;
    add.f32 %f2, %f2, %f12;

    mov.f32 %f13, 0f3F800000;
    sub.f32 %f13, %f13, %f6;
    mul.f32 %f14, %f6, %f3;
    mul.f32 %f15, %f13, %f2;
    add.f32 %f3, %f14, %f15;

    mov.f32 %f13, 0f3F800000;
    sub.f32 %f13, %f13, %f7;
    mul.f32 %f14, %f7, %f4;
    mul.f32 %f15, %f2, %f2;
    mul.f32 %f15, %f13, %f15;
    add.f32 %f4, %f14, %f15;

    div.approx.f32 %f16, %f5, %f10;

    div.approx.f32 %f17, %f4, %f11;
    sqrt.approx.f32 %f17, %f17;
    add.f32 %f17, %f17, %f8;

    div.approx.f32 %f18, %f3, %f17;
    mul.f32 %f18, %f16, %f18;
    sub.f32 %f1, %f1, %f18;

    st.global.f32 [%rd6], %f1;
    st.global.f32 [%rd8], %f3;
    st.global.f32 [%rd9], %f4;

$L__adam_exit:
    ret;
}

.visible .entry grad_norm_sq_f32(
    .param .u64 p_data,
    .param .u64 p_output,
    .param .u32 p_n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    .shared .f32 sdata[256];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;

    ld.param.u32 %r5, [p_n];

    setp.lt.u32 %p1, %r4, %r5;
    mov.f32 %f1, 0f00000000;
    @!%p1 bra $L__norm_store;

    ld.param.u64 %rd1, [p_data];
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd2, %rd2, 2;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f2, [%rd3];
    mul.f32 %f1, %f2, %f2;

$L__norm_store:
    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd4, %rd4, 2;
    mov.u64 %rd5, sdata;
    add.s64 %rd5, %rd5, %rd4;
    st.shared.f32 [%rd5], %f1;
    bar.sync 0;

    mov.u32 %r6, 128;
$L__norm_reduce:
    setp.lt.u32 %p1, %r3, %r6;
    @!%p1 bra $L__norm_reduce_done;

    mov.u64 %rd5, sdata;
    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd3, %rd5, %rd4;
    ld.shared.f32 %f1, [%rd3];

    add.u32 %r7, %r3, %r6;
    cvt.u64.u32 %rd4, %r7;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd5, %rd4;
    ld.shared.f32 %f2, [%rd4];

    add.f32 %f1, %f1, %f2;
    st.shared.f32 [%rd3], %f1;

$L__norm_reduce_done:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    setp.ge.u32 %p1, %r6, 1;
    @%p1 bra $L__norm_reduce;

    setp.eq.u32 %p1, %r3, 0;
    @!%p1 bra $L__norm_exit;

    mov.u64 %rd5, sdata;
    ld.shared.f32 %f1, [%rd5];
    ld.param.u64 %rd1, [p_output];
    atom.global.add.f32 %f1, [%rd1], %f1;

$L__norm_exit:
    ret;
}

.visible .entry grad_scale_f32(
    .param .u64 p_data,
    .param .u32 p_n,
    .param .f32 p_scale
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $L__scale_exit;

    ld.param.u64 %rd1, [p_data];
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd2, %rd2, 2;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f1, [%rd3];

    ld.param.f32 %f2, [p_scale];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd3], %f1;

$L__scale_exit:
    ret;
}
";

/// PTX for strided gather (making non-contiguous tensors contiguous on GPU)
/// Replaces the CPU index computation in contiguous_gpu()
#[cfg(feature = "cuda")]
pub const STRIDED_COPY_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

.visible .entry strided_gather_f32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_strides,
    .param .u64 p_shape,
    .param .u32 p_ndim,
    .param .u32 p_offset,
    .param .u32 p_total_n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd2, [p_dst];
    ld.param.u64 %rd3, [p_strides];
    ld.param.u64 %rd4, [p_shape];
    ld.param.u32 %r1, [p_ndim];
    ld.param.u32 %r2, [p_offset];
    ld.param.u32 %r3, [p_total_n];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    setp.ge.u32 %p1, %r4, %r3;
    @%p1 bra $L__sg_exit;

    mov.u32 %r7, %r4;
    mov.u32 %r8, %r2;

    mov.u32 %r9, %r1;
$L__sg_loop:
    setp.eq.u32 %p1, %r9, 0;
    @%p1 bra $L__sg_done;
    sub.u32 %r9, %r9, 1;

    cvt.u64.u32 %rd5, %r9;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.u32 %r10, [%rd6];

    cvt.u64.u32 %rd5, %r9;
    shl.b64 %rd5, %rd5, 3;
    add.s64 %rd7, %rd3, %rd5;
    ld.global.s32 %r11, [%rd7];

    rem.u32 %r12, %r7, %r10;
    div.u32 %r7, %r7, %r10;
    mad.lo.s32 %r8, %r12, %r11, %r8;

    bra $L__sg_loop;

$L__sg_done:
    cvt.s64.s32 %rd8, %r8;
    shl.b64 %rd8, %rd8, 2;
    add.s64 %rd9, %rd1, %rd8;
    ld.global.f32 %f1, [%rd9];

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd2, %rd10;
    st.global.f32 [%rd11], %f1;

$L__sg_exit:
    ret;
}
";

/// Embedded PTX for im2col (conv2d unfolding) and bias_add_channels
#[cfg(feature = "cuda")]
/// PTX for attention mask expansion kernels
pub const MASK_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

.visible .entry mask_expand_causal_f32(
    .param .u64 p_mask_in,
    .param .u64 p_output,
    .param .u32 p_total_n,
    .param .u32 p_tgt_len,
    .param .u32 p_src_len
) {
    .reg .pred %p<3>;
    .reg .f32 %f<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [p_mask_in];
    ld.param.u64 %rd2, [p_output];
    ld.param.u32 %r1, [p_total_n];
    ld.param.u32 %r2, [p_tgt_len];
    ld.param.u32 %r3, [p_src_len];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    setp.ge.u32 %p1, %r4, %r1;
    @%p1 bra $L__mask_causal_exit;

    rem.u32 %r7, %r4, %r3;
    div.u32 %r8, %r4, %r3;
    rem.u32 %r9, %r8, %r2;

    mad.lo.s32 %r10, %r9, %r3, %r7;

    cvt.u64.u32 %rd3, %r10;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];

    mov.f32 %f2, 0fCEE6B280;
    setp.eq.f32 %p2, %f1, 0f00000000;
    selp.f32 %f1, %f2, 0f00000000, %p2;

    cvt.u64.u32 %rd4, %r4;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f1;

$L__mask_causal_exit:
    ret;
}

.visible .entry mask_expand_padding_f32(
    .param .u64 p_mask_in,
    .param .u64 p_output,
    .param .u32 p_total_n,
    .param .u32 p_num_heads,
    .param .u32 p_tgt_len,
    .param .u32 p_src_len
) {
    .reg .pred %p<3>;
    .reg .f32 %f<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [p_mask_in];
    ld.param.u64 %rd2, [p_output];
    ld.param.u32 %r1, [p_total_n];
    ld.param.u32 %r2, [p_num_heads];
    ld.param.u32 %r3, [p_tgt_len];
    ld.param.u32 %r4, [p_src_len];

    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %tid.x;
    mad.lo.s32 %r5, %r5, %r6, %r7;

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra $L__mask_padding_exit;

    rem.u32 %r8, %r5, %r4;

    mul.lo.s32 %r9, %r2, %r3;
    mul.lo.s32 %r9, %r9, %r4;
    div.u32 %r10, %r5, %r9;

    mad.lo.s32 %r11, %r10, %r4, %r8;

    cvt.u64.u32 %rd3, %r11;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];

    mov.f32 %f2, 0fCEE6B280;
    setp.eq.f32 %p2, %f1, 0f00000000;
    selp.f32 %f1, %f2, 0f00000000, %p2;

    cvt.u64.u32 %rd4, %r5;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f1;

$L__mask_padding_exit:
    ret;
}
";

/// PTX for the sign-correct elementwise pow kernels (the inline ELEMENTWISE PTX computes |x|^n).
pub const POW_FIXED_PTX: &str = include_str!("pow_fixed.ptx");
/// Multi-tensor training kernels (train_fused.cu): mt_sumsq/abssum/ternarize/scale, rms_inv_rows.
pub const TRAIN_FUSED_PTX: &str = include_str!("train_fused.ptx");

/// PTX for the DIRECT depthwise Conv2d kernels (forward / grad_input / grad_weight).
pub const DEPTHWISE_PTX: &str = include_str!("depthwise.ptx");

/// PTX for the BATCHED Conv2d kernels (whole-batch im2col/col2im + grad-weight batch reduction).
/// Compiled from `conv_batched.cu` via NVRTC (the installed gcc/glibc headers break offline nvcc).
pub const CONV_BATCHED_PTX: &str = include_str!("conv_batched.ptx");

/// PTX assembly for Conv2d CUDA kernels (im2col, bias_add, col2im).
pub const CONV_PTX: &str = r"
.version 7.0
.target sm_50
.address_size 64

.visible .entry im2col_f32(
    .param .u64 p_input,
    .param .u64 p_col,
    .param .u64 p_params,
    .param .u32 p_n
) {
    .reg .pred %p<4>;
    .reg .f32 %f<2>;
    .reg .b32 %r<30>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_col];
    ld.param.u64 %rd7, [p_params];
    ld.param.u32 %r20, [p_n];

    ld.global.u32 %r10, [%rd7 + 0];
    ld.global.u32 %r11, [%rd7 + 4];
    ld.global.u32 %r12, [%rd7 + 8];
    ld.global.u32 %r13, [%rd7 + 12];
    ld.global.u32 %r14, [%rd7 + 16];
    ld.global.u32 %r15, [%rd7 + 20];
    ld.global.u32 %r16, [%rd7 + 24];
    ld.global.u32 %r17, [%rd7 + 28];
    ld.global.u32 %r18, [%rd7 + 32];
    ld.global.u32 %r19, [%rd7 + 36];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r1, %r20;
    @%p1 bra $L__im2col_exit;

    rem.u32 %r4, %r1, %r19;
    div.u32 %r5, %r1, %r19;
    rem.u32 %r6, %r5, %r18;
    div.u32 %r7, %r5, %r18;

    rem.u32 %r8, %r7, %r13;
    div.u32 %r9, %r7, %r13;
    rem.u32 %r21, %r9, %r12;
    div.u32 %r22, %r9, %r12;

    mad.lo.s32 %r23, %r6, %r16, %r21;
    sub.s32 %r23, %r23, %r14;
    mad.lo.s32 %r24, %r4, %r17, %r8;
    sub.s32 %r24, %r24, %r15;

    setp.lt.s32 %p2, %r23, 0;
    @%p2 bra $L__im2col_zero;
    setp.ge.s32 %p2, %r23, %r10;
    @%p2 bra $L__im2col_zero;
    setp.lt.s32 %p3, %r24, 0;
    @%p3 bra $L__im2col_zero;
    setp.ge.s32 %p3, %r24, %r11;
    @%p3 bra $L__im2col_zero;

    mul.lo.s32 %r25, %r10, %r11;
    mul.lo.s32 %r26, %r22, %r25;
    mad.lo.s32 %r26, %r23, %r11, %r26;
    add.s32 %r26, %r26, %r24;

    cvt.u64.u32 %rd3, %r26;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];

    cvt.u64.u32 %rd4, %r1;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f1;
    bra $L__im2col_exit;

$L__im2col_zero:
    mov.f32 %f1, 0f00000000;
    cvt.u64.u32 %rd4, %r1;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f1;

$L__im2col_exit:
    ret;
}

.visible .entry bias_add_channels_f32(
    .param .u64 p_data,
    .param .u64 p_bias,
    .param .u32 p_spatial,
    .param .u32 p_n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [p_data];
    ld.param.u64 %rd2, [p_bias];
    ld.param.u32 %r1, [p_spatial];
    ld.param.u32 %r2, [p_n];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r3, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r3, %r2;
    @%p1 bra $L__bias_exit;

    div.u32 %r6, %r3, %r1;

    cvt.u64.u32 %rd3, %r6;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd2, %rd3;
    ld.global.f32 %f1, [%rd3];

    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    add.f32 %f2, %f2, %f1;
    st.global.f32 [%rd5], %f2;

$L__bias_exit:
    ret;
}

.visible .entry col2im_f32(
    .param .u64 p_col,
    .param .u64 p_output,
    .param .u64 p_params,
    .param .u32 p_n
) {
    .reg .pred %p<4>;
    .reg .f32 %f<3>;
    .reg .b32 %r<30>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [p_col];
    ld.param.u64 %rd2, [p_output];
    ld.param.u64 %rd7, [p_params];
    ld.param.u32 %r20, [p_n];

    ld.global.u32 %r10, [%rd7 + 0];
    ld.global.u32 %r11, [%rd7 + 4];
    ld.global.u32 %r12, [%rd7 + 8];
    ld.global.u32 %r13, [%rd7 + 12];
    ld.global.u32 %r14, [%rd7 + 16];
    ld.global.u32 %r15, [%rd7 + 20];
    ld.global.u32 %r16, [%rd7 + 24];
    ld.global.u32 %r17, [%rd7 + 28];
    ld.global.u32 %r18, [%rd7 + 32];
    ld.global.u32 %r19, [%rd7 + 36];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r1, %r20;
    @%p1 bra $L__col2im_exit;

    rem.u32 %r4, %r1, %r19;
    div.u32 %r5, %r1, %r19;
    rem.u32 %r6, %r5, %r18;
    div.u32 %r7, %r5, %r18;

    rem.u32 %r8, %r7, %r13;
    div.u32 %r9, %r7, %r13;
    rem.u32 %r21, %r9, %r12;
    div.u32 %r22, %r9, %r12;

    mad.lo.s32 %r23, %r6, %r16, %r21;
    sub.s32 %r23, %r23, %r14;
    mad.lo.s32 %r24, %r4, %r17, %r8;
    sub.s32 %r24, %r24, %r15;

    setp.lt.s32 %p2, %r23, 0;
    @%p2 bra $L__col2im_exit;
    setp.ge.s32 %p2, %r23, %r10;
    @%p2 bra $L__col2im_exit;
    setp.lt.s32 %p3, %r24, 0;
    @%p3 bra $L__col2im_exit;
    setp.ge.s32 %p3, %r24, %r11;
    @%p3 bra $L__col2im_exit;

    cvt.u64.u32 %rd3, %r1;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];

    mul.lo.s32 %r25, %r10, %r11;
    mul.lo.s32 %r26, %r22, %r25;
    mad.lo.s32 %r26, %r23, %r11, %r26;
    add.s32 %r26, %r26, %r24;

    cvt.u64.u32 %rd4, %r26;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd2, %rd4;
    atom.global.add.f32 %f2, [%rd4], %f1;

$L__col2im_exit:
    ret;
}
";

/// LSTM/GRU/BatchNorm fused kernels (compiled from lstm.cu)
pub const LSTM_PTX: &str = include_str!("lstm.ptx");

/// Pooling kernels: MaxPool2d + AvgPool2d forward/backward (compiled from pooling.cu)
pub const POOLING_PTX: &str = include_str!("pooling.ptx");

/// Fused attention kernel: scaled dot-product attention without materializing N*N matrix
pub const ATTENTION_PTX: &str = include_str!("attention.ptx");

/// Q4_K quantized matmul (dequant-in-shader).
///
/// Compiled from `q4k_matmul.cu`. Exposes `q4k_gemv_f32` which computes
/// `c[j] = sum_k a[k] * B[j, k]` where B is laid out as Q4_K super-blocks
/// in physical GGUF layout `[out, in]`. Used for large quantized models that
/// can't fit f32 weights in 12 GB VRAM.
pub const Q4K_MATMUL_PTX: &str = include_str!("q4k_matmul.ptx");

/// Q5_K quantized matmul (dequant-in-shader).
///
/// Compiled from `q5k_matmul.cu`. Same shape contract as Q4_K but with
/// the 176-byte Q5_K super-block layout (adds 32 bytes of qh — one high
/// bit per 5-bit weight — between Q4_K's packed scales and qs nibbles).
/// Unlocks Phi-3-mini Q4_K_M at GPU-native decode speed by removing the
/// eager Q5_K → F32 dequant fallback on `attn_qkv`.
pub const Q5K_MATMUL_PTX: &str = include_str!("q5k_matmul.ptx");

/// Q5_0 / Q5_1 quantized matmul (dequant-in-shader).
///
/// Compiled from `q5_01_matmul.cu`. Block size 32 (vs 256 for Q5_K), one
/// warp per output row, lanes cooperate one-element-each per block.
/// Q5_0 is signed `((lo | hi*16) - 16) * d`; Q5_1 is unsigned
/// `(lo | hi*16) * d + m`. Lands legacy-Falcon on GPU — its attn_qkv is
/// Q5_1 and the rest (attn_output, ffn_up, token_embd) is Q5_0, both of
/// which fell through to `cpu_dequant_matmul` before this kernel existed.
pub const Q5_01_MATMUL_PTX: &str = include_str!("q5_01_matmul.ptx");

/// Q8_0 quantized matmul (dequant-in-shader).
///
/// Compiled from `q8_0_matmul.cu`. 34-byte block: f16 scale + 32 signed
/// int8 quants. Two warps per output row, same v2 layout as Q5_0. Primary
/// consumer: Falcon-7B's Q8_0 LM head (4544 × 65024), which otherwise
/// falls through to `cpu_dequant_matmul` on every decode token.
pub const Q8_0_MATMUL_PTX: &str = include_str!("q8_0_matmul.ptx");

/// BitNet I2_S (1.58-bit ternary) matmul kernel.
///
/// Compiled from `i2s_matmul.cu`. 32-byte block holding 128 ternary
/// weights (2 bits each, group-strided layout matching Microsoft's
/// bitnet.cpp AVX2 reference). Two warps per output row. Tensor-wide
/// f32 scale passed as a kernel argument (not stored per-block). This
/// is what lets BitNet move off the CPU-only `matmul_i2s` path that
/// caps BitNet-2B decode at ~7 tok/s on WSL.
pub const I2S_MATMUL_PTX: &str = include_str!("i2s_matmul.ptx");

/// Raw-i8 ternary matmul kernels for a ternary quantized linear layer.
///
/// Compiled from `ternary_matmul.cu`. Distinct from the BitNet I2_S
/// kernels — the ternary linear layer stores ternary weights as a flat `Vec<i8>`
/// in `{-1, 0, +1}` plus a tensor-wide f32 scale. The kernels here
/// branch on each i8 byte and add/subtract/skip the activation, then
/// scale at the end. Forward (gemv/gemm), backward grad_input, and
/// backward grad_bias all live in this module.
pub const TERNARY_MATMUL_PTX: &str = include_str!("ternary_matmul.ptx");

/// GPU shadow-weight ternary quantizer for a ternary linear layer.
///
/// Compiled from `ternary_quantize.cu`. Stage 1 reduces the f32 shadow
/// weight tensor to a single absolute-sum scalar (host divides by N to
/// get absmean); stage 2 thresholds each element to a `{-1, 0, +1}` i8
/// byte using that scale. Eliminates the per-step 4 GB GPU→CPU copy
/// that would otherwise gate `quantize_weights` on the 1B run.
pub const TERNARY_QUANTIZE_PTX: &str = include_str!("ternary_quantize.ptx");

/// PrismML Q1_0 (1-bit) matmul kernel.
///
/// Compiled from `q1_0_matmul.cu`. 18-byte block holding 128 weights
/// (1 bit each, linear bit-order) plus a per-block fp16 scale. Two warps
/// per output row, same v2 layout as I2_S. No tensor-wide scale — each
/// block carries its own. Primary consumer: PrismML Bonsai-8B family
/// (Qwen3-8B QAT'd to Q1_0). Effective bits/weight: 1.125.
pub const Q1_0_MATMUL_PTX: &str = include_str!("q1_0_matmul.ptx");

/// Q1_0 DP4A path — int8 activations + `__dp4a` accumulate.
///
/// Compiled from `q1_0_matmul_dp4a.cu`. Companion to Q1_0_MATMUL_PTX:
/// adds an online `q1_0_quantize_acts_q8` step that converts the f32
/// activation row to int8 + per-32-chunk fp16 scales, and a
/// `q1_0_gemv_dp4a_f32` matmul that uses `__dp4a` for the inner loop
/// (4× int8 MAC per PTX instruction on the integer pipeline). Mirrors
/// PrismML's `vec_dot_q1_0_q8_1` math with Q8_0-style scale-only
/// activations (binary weights have zero DC term so no `s` correction).
pub const Q1_0_MATMUL_DP4A_PTX: &str = include_str!("q1_0_matmul_dp4a.ptx");

/// Q1_0 fused single-launch DP4A path.
///
/// Compiled from `q1_0_matmul_fused.cu`. Combines activation quant +
/// dp4a gemv into one kernel: each CTA cooperatively quantizes the
/// f32 activation row into shared memory once, then all warps in the
/// CTA do the dp4a matmul against smem-resident acts. Eliminates the
/// extra launch + global scratch allocation that made the standalone
/// DP4A path lose to v2 on launch-overhead-bound decode.
pub const Q1_0_MATMUL_FUSED_PTX: &str = include_str!("q1_0_matmul_fused.ptx");

/// Q6_K quantized matmul (dequant-in-shader).
///
/// Compiled from `q6k_matmul.cu`. Same shape contract as Q4_K but with
/// the 210-byte Q6_K super-block layout. Primary consumer: LM head matmul,
/// which is Q6_K in most GGUF exports and fires every decode token — moving
/// it off the CPU dequant path is the biggest single-matmul decode win.
pub const Q6K_MATMUL_PTX: &str = include_str!("q6k_matmul.ptx");

/// Transformer per-layer ops (rms_norm, RoPE split-halves, SwiGLU, ReLU²).
///
/// Compiled from `transformer_ops.cu`. These kernels collectively let the
/// axonml-serve decode loop keep activations on GPU through the entire
/// layer instead of round-tripping to CPU after every matmul. Used by
/// `Tensor::rms_norm`, `Tensor::apply_rope_split_halves`, `Tensor::swiglu`,
/// and `Tensor::relu2_gate`.
pub const TRANSFORMER_OPS_PTX: &str = include_str!("transformer_ops.ptx");

/// Per-output-channel LSQ fake-quant kernels (QAT). Folded from the vendored
/// AxonML core so sub-bit models train on the canonical private framework.
pub const FAKE_QUANT_PTX: &str = include_str!("fake_quant.ptx");

/// CUDA Kernel registry for managing loaded kernels
#[cfg(feature = "cuda")]
pub struct CudaKernels {
    ctx: Arc<CudaContext>,
    functions: HashMap<String, CudaFunction>,
}

#[cfg(feature = "cuda")]
impl CudaKernels {
    /// Load kernels from embedded PTX
    pub fn load(ctx: Arc<CudaContext>) -> Result<Self, CudaError> {
        let mut kernels = Self {
            ctx,
            functions: HashMap::new(),
        };

        kernels.load_module(
            "elementwise",
            ELEMENTWISE_PTX,
            &[
                "add_f32",
                "sub_f32",
                "mul_f32",
                "div_f32",
                "scale_f32",
                "add_scalar_f32",
                "neg_f32",
                "sqrt_f32",
                "pow_f32",
                "pow_scalar_f32",
            ],
        )?;

        kernels.load_module(
            "fake_quant",
            FAKE_QUANT_PTX,
            &["fake_quant_pc_fwd_f32", "fake_quant_pc_bwd_f32"],
        )?;

        kernels.load_module(
            "activations",
            ACTIVATIONS_PTX,
            &[
                "relu_f32",
                "relu_backward_f32",
                "sigmoid_f32",
                "sigmoid_backward_f32",
                "tanh_f32",
                "tanh_backward_f32",
                "exp_f32",
                "log_f32",
                "gelu_f32",
                "silu_f32",
                "silu_backward_f32",
            ],
        )?;

        kernels.load_module(
            "broadcast",
            BROADCAST_PTX,
            &[
                "broadcast_add_f32",
                "broadcast_sub_f32",
                "broadcast_mul_f32",
                "broadcast_div_f32",
                "broadcast_add_rev_f32",
                "broadcast_sub_rev_f32",
                "broadcast_mul_rev_f32",
                "broadcast_div_rev_f32",
            ],
        )?;

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

        kernels.load_module("sum_dim", SUM_DIM_PTX, &["sum_dim_f32"])?;
        kernels.load_module("argdim", ARGDIM_PTX, &["argmax_dim_f32", "argmin_dim_f32"])?;

        kernels.load_module(
            "layernorm",
            LAYERNORM_PTX,
            &[
                "layer_norm_f32",
                "layer_norm_backward_dinput_f32",
                "layer_norm_backward_dweight_dbias_f32",
            ],
        )?;

        kernels.load_module(
            "cross_entropy",
            CROSS_ENTROPY_PTX,
            &["cross_entropy_fwd_f32", "cross_entropy_bwd_f32"],
        )?;

        kernels.load_module(
            "conv",
            CONV_PTX,
            &["im2col_f32", "col2im_f32", "bias_add_channels_f32"],
        )?;

        kernels.load_module(
            "pow_fixed",
            POW_FIXED_PTX,
            &["pow_f32_c99", "pow_scalar_f32_c99"],
        )?;

        kernels.load_module(
            "train_fused",
            TRAIN_FUSED_PTX,
            &[
                "mt_sumsq_f32",
                "mt_abssum_f32",
                "mt_ternarize_f32",
                "mt_scale_f32",
                "rms_inv_rows_f32",
                "rms_norm_bwd_weight_partial_f32",
            ],
        )?;

        kernels.load_module(
            "depthwise",
            DEPTHWISE_PTX,
            &[
                "depthwise_fwd_f32",
                "depthwise_grad_input_f32",
                "depthwise_grad_weight_f32",
            ],
        )?;

        kernels.load_module(
            "conv_batched",
            CONV_BATCHED_PTX,
            &[
                "im2col_batched_f32",
                "col2im_batched_f32",
                "sum_batch_f32",
                "bias_add_channels_batched_f32",
                "im2col_group_batched_f32",
                "col2im_group_batched_f32",
                "sum_batch_at_f32",
                "strided_block_copy_f32",
                "scatter_add_u32_f32",
                "adaptive_avgpool2d_bwd_f32",
                "groupnorm_bwd_stats_f32",
                "groupnorm_bwd_apply_f32",
                "convtranspose2d_bwd_input_f32",
                "convtranspose2d_bwd_weight_f32",
                "mul_backward_f32",
                "sum_bias_f32",
            ],
        )?;

        kernels.load_module(
            "mask",
            MASK_PTX,
            &["mask_expand_causal_f32", "mask_expand_padding_f32"],
        )?;

        kernels.load_module("strided_copy", STRIDED_COPY_PTX, &["strided_gather_f32"])?;

        kernels.load_module(
            "embedding",
            EMBEDDING_SCATTER_PTX,
            &["embedding_scatter_add_f32"],
        )?;

        kernels.load_module(
            "adam",
            ADAM_PTX,
            &["adam_step_f32", "grad_norm_sq_f32", "grad_scale_f32"],
        )?;

        kernels.load_module(
            "lstm",
            LSTM_PTX,
            &[
                "lstm_gates_f32",
                "lstm_gates_backward_f32",
                "gru_gates_f32",
                "gru_gates_backward_f32",
                "batchnorm_stats_f32",
                "batchnorm_norm_f32",
                "batchnorm_bwd_reduce_f32",
                "batchnorm_bwd_input_f32",
            ],
        )?;

        kernels.load_module(
            "pooling",
            POOLING_PTX,
            &[
                "maxpool2d_fwd_f32",
                "maxpool2d_bwd_f32",
                "avgpool2d_fwd_f32",
                "avgpool2d_bwd_f32",
            ],
        )?;

        kernels.load_module(
            "attention",
            ATTENTION_PTX,
            &[
                "fused_attention_fwd_f32",
                "fused_attention_bwd_f32",
                "fused_attn_decode_f32",
                "fused_attn_decode_q8_f32",
                "fused_attn_prefill_f32",
                "quantize_kv_row_q8_f32",
            ],
        )?;

        kernels.load_module(
            "q4k_matmul",
            Q4K_MATMUL_PTX,
            &[
                "q4k_gemv_f32",
                "q4k_gemm_f32",
                "q4k_gemm_matched_f32",
                "q4k_gemv_fused_qkv_f32",
                "q4k_gemv_fused_qkv_bias_f32",
                "q4k_gemv_fused_gate_up_f32",
                "q4k_gemv_residual_f32",
                "q4k_gemv_fused_gate_up_swiglu_f32",
            ],
        )?;

        kernels.load_module(
            "q5k_matmul",
            Q5K_MATMUL_PTX,
            &[
                "q5k_gemv_f32",
                "q5k_gemm_f32",
                "q5k_gemv_fused_qkv_f32",
                "q5k_gemm_matched_f32",
            ],
        )?;

        kernels.load_module(
            "q5_01_matmul",
            Q5_01_MATMUL_PTX,
            &[
                "q5_0_gemv_f32",
                "q5_0_gemm_f32",
                "q5_1_gemv_f32",
                "q5_1_gemm_f32",
                "q5_1_gemv_fused_qkv_f32",
            ],
        )?;

        kernels.load_module(
            "q8_0_matmul",
            Q8_0_MATMUL_PTX,
            &["q8_0_gemv_f32", "q8_0_gemm_f32"],
        )?;

        kernels.load_module(
            "i2s_matmul",
            I2S_MATMUL_PTX,
            &["i2s_gemv_f32", "i2s_gemm_f32"],
        )?;

        kernels.load_module(
            "ternary_matmul",
            TERNARY_MATMUL_PTX,
            &[
                "ternary_gemv_f32",
                "ternary_gemm_f32",
                "ternary_grad_input_f32",
                "ternary_grad_bias_f32",
            ],
        )?;

        kernels.load_module(
            "ternary_quantize",
            TERNARY_QUANTIZE_PTX,
            &["f32_abssum_reduce", "f32_quantize_ternary"],
        )?;

        kernels.load_module(
            "q1_0_matmul",
            Q1_0_MATMUL_PTX,
            &["q1_0_gemv_f32", "q1_0_gemm_f32"],
        )?;

        kernels.load_module(
            "q1_0_matmul_dp4a",
            Q1_0_MATMUL_DP4A_PTX,
            &["q1_0_quantize_acts_q8", "q1_0_gemv_dp4a_f32"],
        )?;

        kernels.load_module(
            "q1_0_matmul_fused",
            Q1_0_MATMUL_FUSED_PTX,
            &["q1_0_gemv_fused_dp4a_f32"],
        )?;

        kernels.load_module(
            "q6k_matmul",
            Q6K_MATMUL_PTX,
            &["q6k_gemv_f32", "q6k_gemm_f32", "q6k_gemm_matched_f32"],
        )?;

        kernels.load_module(
            "transformer_ops",
            TRANSFORMER_OPS_PTX,
            &[
                "rms_norm_f32",
                "layer_norm_tokenwise_f32",
                "gelu_tanh_f32",
                "parallel_residual_add_f32",
                "scaled_add_inplace_f32",
                "rms_norm_heads_f32",
                "rope_split_halves_f32",
                "swiglu_f32",
                "relu2_gate_f32",
                "rms_norm_batched_f32",
                "rms_norm_bwd_batched_f32",
                "rms_norm_heads_batched_f32",
                "rope_split_halves_batched_f32",
                "add_bias_batched_f32",
                "softmax_causal_scaled_f32",
                "softmax_causal_scaled_bwd_f32",
                "swiglu_bwd_f32",
                "add_rmsnorm_batched_f32",
                "rope_split_halves_bhsd_f32",
                "rope_split_halves_bhsd_bwd_f32",
                "repeat_kv_f32",
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
        let ptx_obj = Ptx::from_src(ptx);
        let module: Arc<CudaModule> = self.ctx.load_module(ptx_obj).map_err(|e| {
            eprintln!("[AxonML CUDA] Failed to load module '{}': {}", name, e);
            CudaError::ModuleLoadFailed(e.to_string())
        })?;

        for func_name in functions {
            let func = module.load_function(func_name).map_err(|e| {
                eprintln!(
                    "[AxonML CUDA] Failed to load function '{}' from '{}': {}",
                    func_name, name, e
                );
                CudaError::KernelNotFound(func_name.to_string())
            })?;
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

    /// Runtime-JIT a CUDA C source string, load `entry`, and cache it under `key`. Returns the
    /// cached function on subsequent calls with the same key (no recompile). Used by the elementwise
    /// chain fuser: each unique chain signature compiles once, then dispatches for free.
    pub fn get_or_compile(
        &mut self,
        key: &str,
        entry: &str,
        src: &str,
    ) -> Result<&CudaFunction, CudaError> {
        if !self.functions.contains_key(key) {
            let ptx = cudarc::nvrtc::compile_ptx(src)
                .map_err(|e| CudaError::ModuleLoadFailed(format!("nvrtc: {e}")))?;
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| CudaError::ModuleLoadFailed(e.to_string()))?;
            let func = module
                .load_function(entry)
                .map_err(|e| CudaError::KernelNotFound(format!("{entry}: {e}")))?;
            self.functions.insert(key.to_string(), func);
        }
        Ok(self.functions.get(key).unwrap())
    }
}

/// Compute optimal launch configuration for a given number of elements
#[cfg(feature = "cuda")]
pub fn launch_config(n: usize) -> LaunchConfig {
    let num_blocks = (n as u32).div_ceil(BLOCK_SIZE);
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
        assert_eq!(cfg.grid_dim, (4, 1, 1));
    }

    #[test]
    fn test_launch_config_large() {
        let cfg = launch_config(1_000_000);
        assert_eq!(cfg.grid_dim, (3907, 1, 1));
    }
}
