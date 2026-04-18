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
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
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
    // Initialize all shared memory slots to -inf so unused threads don't
    // contribute garbage values during the reduction.
    mov.f32 %f1, 0fFF800000;      // -inf
    cvt.u64.u32 %rd5, %r3;
    shl.b64 %rd5, %rd5, 2;
    mov.u64 %rd9, sdata;
    add.s64 %rd8, %rd9, %rd5;
    st.shared.f32 [%rd8], %f1;
    bar.sync 0;

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

/// PTX for embedding scatter-add backward (GPU-native gradient accumulation)
/// Each thread handles one element: given (token_index, dim_offset),
/// atomically adds grad_output[token_index * emb_dim + dim_offset] to
/// weight_grad[indices[token_index] * emb_dim + dim_offset].
#[cfg(feature = "cuda")]
pub const EMBEDDING_SCATTER_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// embedding_scatter_add_f32: Scatter-add gradients for embedding backward
// Thread i handles one element in grad_output (total = num_indices * emb_dim)
// Params: grad_src, indices, weight_grad, num_indices, emb_dim
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

    // Global thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    // Bounds check: r1 = thread_idx, must be < total_n
    ld.param.u32 %r4, [p_total_n];
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $L__scatter_exit;

    // Compute token_index = thread_idx / emb_dim
    // Compute dim_offset  = thread_idx % emb_dim
    ld.param.u32 %r5, [p_emb_dim];
    div.u32 %r6, %r1, %r5;    // r6 = token_index
    rem.u32 %r7, %r1, %r5;    // r7 = dim_offset

    // Load indices[token_index] -> r8 = embedding row index
    ld.param.u64 %rd1, [p_indices];
    cvt.u64.u32 %rd2, %r6;
    shl.b64 %rd2, %rd2, 2;       // *4 bytes per u32
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r8, [%rd3];   // r8 = indices[token_index]

    // Load grad_src[thread_idx]
    ld.param.u64 %rd4, [p_grad_src];
    cvt.u64.u32 %rd5, %r1;
    shl.b64 %rd5, %rd5, 2;       // *4 bytes per f32
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];

    // Compute dest offset: r8 * emb_dim + dim_offset
    mad.lo.u32 %r9, %r8, %r5, %r7;
    ld.param.u64 %rd7, [p_weight_grad];
    cvt.u64.u32 %rd8, %r9;
    shl.b64 %rd8, %rd8, 2;       // *4 bytes per f32
    add.s64 %rd9, %rd7, %rd8;

    // Atomic add: weight_grad[r8*emb_dim + dim_offset] += grad_src[thread_idx]
    atom.global.add.f32 %f1, [%rd9], %f1;

$L__scatter_exit:
    ret;
}
"#;

/// PTX for fused Adam optimizer step (GPU-native parameter update)
/// Each thread handles one element: updates param, exp_avg, exp_avg_sq in-place.
/// Eliminates the GPU->CPU->GPU copy that standard Adam does per step.
#[cfg(feature = "cuda")]
pub const ADAM_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// adam_step_f32: Fused Adam parameter update
// Thread i updates one element of param, exp_avg, exp_avg_sq
// Params: param, grad, exp_avg, exp_avg_sq, n,
//         lr, beta1, beta2, eps, weight_decay,
//         bias_correction1, bias_correction2
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

    // Global thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    // Bounds check
    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $L__adam_exit;

    // Compute byte offset for this element
    cvt.u64.u32 %rd1, %r1;
    shl.b64 %rd1, %rd1, 2;    // *4 bytes

    // Load all pointers
    ld.param.u64 %rd2, [p_param];
    ld.param.u64 %rd3, [p_grad];
    ld.param.u64 %rd4, [p_exp_avg];
    ld.param.u64 %rd5, [p_exp_avg_sq];

    // Load values: param[i], grad[i], m[i], v[i]
    add.s64 %rd6, %rd2, %rd1;
    ld.global.f32 %f1, [%rd6];      // f1 = param[i]
    add.s64 %rd7, %rd3, %rd1;
    ld.global.f32 %f2, [%rd7];      // f2 = grad[i]
    add.s64 %rd8, %rd4, %rd1;
    ld.global.f32 %f3, [%rd8];      // f3 = exp_avg[i]
    add.s64 %rd9, %rd5, %rd1;
    ld.global.f32 %f4, [%rd9];      // f4 = exp_avg_sq[i]

    // Load hyperparams
    ld.param.f32 %f5, [p_lr];
    ld.param.f32 %f6, [p_beta1];
    ld.param.f32 %f7, [p_beta2];
    ld.param.f32 %f8, [p_eps];
    ld.param.f32 %f9, [p_weight_decay];
    ld.param.f32 %f10, [p_bias_correction1];
    ld.param.f32 %f11, [p_bias_correction2];

    // Apply weight decay to grad: grad = grad + weight_decay * param
    // f12 = weight_decay * param[i]
    mul.f32 %f12, %f9, %f1;
    add.f32 %f2, %f2, %f12;         // f2 = grad + wd*param

    // Update exp_avg: m = beta1 * m + (1-beta1) * grad
    // f13 = 1.0 - beta1
    mov.f32 %f13, 0f3F800000;       // 1.0
    sub.f32 %f13, %f13, %f6;        // 1 - beta1
    mul.f32 %f14, %f6, %f3;         // beta1 * m
    mul.f32 %f15, %f13, %f2;        // (1-beta1) * grad
    add.f32 %f3, %f14, %f15;        // new m

    // Update exp_avg_sq: v = beta2 * v + (1-beta2) * grad^2
    mov.f32 %f13, 0f3F800000;       // 1.0
    sub.f32 %f13, %f13, %f7;        // 1 - beta2
    mul.f32 %f14, %f7, %f4;         // beta2 * v
    mul.f32 %f15, %f2, %f2;         // grad^2
    mul.f32 %f15, %f13, %f15;       // (1-beta2) * grad^2
    add.f32 %f4, %f14, %f15;        // new v

    // Bias-corrected step: step_size = lr / bias_correction1
    div.approx.f32 %f16, %f5, %f10; // step_size

    // denom = sqrt(v / bias_correction2) + eps
    div.approx.f32 %f17, %f4, %f11; // v / bc2
    sqrt.approx.f32 %f17, %f17;     // sqrt(v/bc2)
    add.f32 %f17, %f17, %f8;        // + eps

    // param = param - step_size * m / denom
    div.approx.f32 %f18, %f3, %f17; // m / denom
    mul.f32 %f18, %f16, %f18;       // step_size * m / denom
    sub.f32 %f1, %f1, %f18;         // param -= update

    // Store updated values
    st.global.f32 [%rd6], %f1;      // param[i]
    st.global.f32 [%rd8], %f3;      // exp_avg[i]
    st.global.f32 [%rd9], %f4;      // exp_avg_sq[i]

$L__adam_exit:
    ret;
}

// grad_norm_sq_f32: Compute sum of squares of a vector (partial reduction)
// Each block reduces its portion, atomically adds to output[0]
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

    // Global thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;   // global_idx

    ld.param.u32 %r5, [p_n];

    // Load and square, or zero if out of bounds
    setp.lt.u32 %p1, %r4, %r5;
    mov.f32 %f1, 0f00000000;          // 0.0
    @!%p1 bra $L__norm_store;

    ld.param.u64 %rd1, [p_data];
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd2, %rd2, 2;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f2, [%rd3];
    mul.f32 %f1, %f2, %f2;            // f1 = data[i]^2

$L__norm_store:
    // Store to shared memory
    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd4, %rd4, 2;
    mov.u64 %rd5, sdata;
    add.s64 %rd5, %rd5, %rd4;
    st.shared.f32 [%rd5], %f1;
    bar.sync 0;

    // Tree reduction in shared memory
    mov.u32 %r6, 128;
$L__norm_reduce:
    setp.lt.u32 %p1, %r3, %r6;
    @!%p1 bra $L__norm_reduce_done;

    // Load sdata[tid] and sdata[tid + stride]
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

    // Thread 0 atomically adds block result to global output
    setp.eq.u32 %p1, %r3, 0;
    @!%p1 bra $L__norm_exit;

    mov.u64 %rd5, sdata;
    ld.shared.f32 %f1, [%rd5];
    ld.param.u64 %rd1, [p_output];
    atom.global.add.f32 %f1, [%rd1], %f1;

$L__norm_exit:
    ret;
}

// grad_scale_f32: Multiply all elements by a scalar
// data[i] *= scale
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
"#;

/// PTX for strided gather (making non-contiguous tensors contiguous on GPU)
/// Replaces the CPU index computation in contiguous_gpu()
#[cfg(feature = "cuda")]
pub const STRIDED_COPY_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// strided_gather_f32: Copy elements from strided layout to contiguous output
// src:     source data pointer (device)
// dst:     destination data pointer (device, contiguous)
// strides: [ndim] array of strides (device, i64)
// shape:   [ndim] array of shape dims (device, u32)
// ndim:    number of dimensions
// offset:  storage offset
// total_n: total number of elements to copy
//
// Each thread computes its multi-dim coordinate from its linear index,
// then computes the source offset using strides.
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

    // Global thread index
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    setp.ge.u32 %p1, %r4, %r3;
    @%p1 bra $L__sg_exit;

    // Compute multi-dim coordinate from linear index, then source offset
    // remaining = idx
    // src_offset = storage_offset
    // For each dim d from 0 to ndim-1:
    //   coord_d = remaining / product(shape[d+1..ndim])
    //   remaining = remaining % product(shape[d+1..ndim])
    //   src_offset += coord_d * strides[d]
    //
    // We precompute by iterating right-to-left (innermost dim first)
    // using: coord_d = remaining % shape[d], remaining /= shape[d]

    mov.u32 %r7, %r4;           // remaining = idx
    mov.u32 %r8, %r2;           // src_offset = storage_offset (as i32 for now)

    // Loop from dim = ndim-1 down to 0
    mov.u32 %r9, %r1;           // d = ndim
$L__sg_loop:
    setp.eq.u32 %p1, %r9, 0;
    @%p1 bra $L__sg_done;
    sub.u32 %r9, %r9, 1;        // d--

    // Load shape[d]
    cvt.u64.u32 %rd5, %r9;
    shl.b64 %rd5, %rd5, 2;      // * sizeof(u32)
    add.s64 %rd6, %rd4, %rd5;
    ld.global.u32 %r10, [%rd6]; // shape[d]

    // Load strides[d] (stored as i64 / isize)
    cvt.u64.u32 %rd5, %r9;
    shl.b64 %rd5, %rd5, 3;      // * sizeof(i64)
    add.s64 %rd7, %rd3, %rd5;
    ld.global.s32 %r11, [%rd7]; // strides[d] (lower 32 bits, sufficient for most tensors)

    // coord = remaining % shape[d]
    rem.u32 %r12, %r7, %r10;
    // remaining = remaining / shape[d]
    div.u32 %r7, %r7, %r10;
    // src_offset += coord * stride[d]
    mad.lo.s32 %r8, %r12, %r11, %r8;

    bra $L__sg_loop;

$L__sg_done:
    // Load src[src_offset]
    cvt.s64.s32 %rd8, %r8;
    shl.b64 %rd8, %rd8, 2;      // * sizeof(f32)
    add.s64 %rd9, %rd1, %rd8;
    ld.global.f32 %f1, [%rd9];

    // Store to dst[idx]
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd11, %rd2, %rd10;
    st.global.f32 [%rd11], %f1;

$L__sg_exit:
    ret;
}
"#;

/// Embedded PTX for im2col (conv2d unfolding) and bias_add_channels
#[cfg(feature = "cuda")]
/// PTX for attention mask expansion kernels
pub const MASK_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// mask_expand_causal_f32: Expand [T, S] causal mask to [B, H, T, S] with 0 to -1e9
// mask_in: [T * S] input mask (device pointer)
// output:  [B * H * T * S] expanded mask (device pointer)
// total_n: B * H * T * S
// tgt_len: T
// src_len: S
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

    // Global thread index
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r4, %r4, %r5, %r6;

    setp.ge.u32 %p1, %r4, %r1;
    @%p1 bra $L__mask_causal_exit;

    // idx to j = idx % S, i = (idx / S) % T
    rem.u32 %r7, %r4, %r3;          // j = idx % src_len
    div.u32 %r8, %r4, %r3;          // tmp = idx / src_len
    rem.u32 %r9, %r8, %r2;          // i = tmp % tgt_len

    // mask_in index = i * S + j
    mad.lo.s32 %r10, %r9, %r3, %r7;

    // Load mask_in[i * S + j]
    cvt.u64.u32 %rd3, %r10;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];

    // Convert: 0.0 to -1e9, nonzero to 0.0
    mov.f32 %f2, 0fCEE6B280;        // -1e9 in IEEE 754
    setp.eq.f32 %p2, %f1, 0f00000000;
    selp.f32 %f1, %f2, 0f00000000, %p2;

    // Store to output[idx]
    cvt.u64.u32 %rd4, %r4;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f1;

$L__mask_causal_exit:
    ret;
}

// mask_expand_padding_f32: Expand [B, S] padding mask to [B, H, T, S] with 0 to -1e9
// mask_in: [B * S] input mask (device pointer)
// output:  [B * H * T * S] expanded mask (device pointer)
// total_n: B * H * T * S
// num_heads: H
// tgt_len: T
// src_len: S
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

    // Global thread index
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %tid.x;
    mad.lo.s32 %r5, %r5, %r6, %r7;

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra $L__mask_padding_exit;

    // idx to j = idx % S, stride = H * T * S, b = idx / stride
    rem.u32 %r8, %r5, %r4;          // j = idx % src_len

    // stride = num_heads * tgt_len * src_len
    mul.lo.s32 %r9, %r2, %r3;
    mul.lo.s32 %r9, %r9, %r4;       // stride = H * T * S
    div.u32 %r10, %r5, %r9;         // b = idx / stride

    // mask_in index = b * S + j
    mad.lo.s32 %r11, %r10, %r4, %r8;

    // Load mask_in[b * S + j]
    cvt.u64.u32 %rd3, %r11;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];

    // Convert: 0.0 to -1e9, nonzero to 0.0
    mov.f32 %f2, 0fCEE6B280;        // -1e9 in IEEE 754
    setp.eq.f32 %p2, %f1, 0f00000000;
    selp.f32 %f1, %f2, 0f00000000, %p2;

    // Store to output[idx]
    cvt.u64.u32 %rd4, %r5;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f1;

$L__mask_padding_exit:
    ret;
}
"#;

/// PTX assembly for Conv2d CUDA kernels (im2col, bias_add, col2im).
pub const CONV_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// im2col_f32: Unfold input patches into column matrix on GPU
// input:  [C_in, H, W] (one batch element, device pointer)
// col:    [C_in*kH*kW, out_H*out_W] (output column matrix, device pointer)
// params: u32[10] = {H, W, kH, kW, pH, pW, sH, sW, oH, oW}
// n:      total elements = C_in * kH * kW * oH * oW
//
// Decomposition for thread idx:
//   w_col  = idx % out_w
//   h_col  = (idx / out_w) % out_h
//   c_col  = idx / (out_w * out_h)     (0 .. C_in*kH*kW)
//   kw_off = c_col % kW
//   kh_off = (c_col / kW) % kH
//   c_in   = c_col / (kW * kH)
//   h_in   = h_col * stride_h + kh_off - pad_h   (signed)
//   w_in   = w_col * stride_w + kw_off - pad_w   (signed)
//
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

    // Load conv params from global memory buffer
    ld.global.u32 %r10, [%rd7 + 0];   // height
    ld.global.u32 %r11, [%rd7 + 4];   // width
    ld.global.u32 %r12, [%rd7 + 8];   // kernel_h
    ld.global.u32 %r13, [%rd7 + 12];  // kernel_w
    ld.global.u32 %r14, [%rd7 + 16];  // pad_h
    ld.global.u32 %r15, [%rd7 + 20];  // pad_w
    ld.global.u32 %r16, [%rd7 + 24];  // stride_h
    ld.global.u32 %r17, [%rd7 + 28];  // stride_w
    ld.global.u32 %r18, [%rd7 + 32];  // out_h
    ld.global.u32 %r19, [%rd7 + 36];  // out_w

    // Global thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r1, %r20;
    @%p1 bra $L__im2col_exit;

    // w_col = idx % out_w
    rem.u32 %r4, %r1, %r19;
    // tmp = idx / out_w
    div.u32 %r5, %r1, %r19;
    // h_col = tmp % out_h
    rem.u32 %r6, %r5, %r18;
    // c_col = tmp / out_h
    div.u32 %r7, %r5, %r18;

    // kw_off = c_col % kW
    rem.u32 %r8, %r7, %r13;
    // tmp2 = c_col / kW
    div.u32 %r9, %r7, %r13;
    // kh_off = tmp2 % kH
    rem.u32 %r21, %r9, %r12;
    // c_in = tmp2 / kH
    div.u32 %r22, %r9, %r12;

    // h_in = h_col * stride_h + kh_off - pad_h  (signed)
    mad.lo.s32 %r23, %r6, %r16, %r21;
    sub.s32 %r23, %r23, %r14;
    // w_in = w_col * stride_w + kw_off - pad_w  (signed)
    mad.lo.s32 %r24, %r4, %r17, %r8;
    sub.s32 %r24, %r24, %r15;

    // Bounds: h_in in [0, height) and w_in in [0, width)
    setp.lt.s32 %p2, %r23, 0;
    @%p2 bra $L__im2col_zero;
    setp.ge.s32 %p2, %r23, %r10;
    @%p2 bra $L__im2col_zero;
    setp.lt.s32 %p3, %r24, 0;
    @%p3 bra $L__im2col_zero;
    setp.ge.s32 %p3, %r24, %r11;
    @%p3 bra $L__im2col_zero;

    // In bounds: input[c_in * H * W + h_in * W + w_in]
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

// bias_add_channels_f32: data[i] += bias[i / spatial_size]
// data:    [C_out * spatial] (in-place, device pointer)
// bias:    [C_out] (device pointer)
// spatial: out_h * out_w
// n:       C_out * spatial (total elements)
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

    // channel = i / spatial_size
    div.u32 %r6, %r3, %r1;

    // Load bias[channel]
    cvt.u64.u32 %rd3, %r6;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd2, %rd3;
    ld.global.f32 %f1, [%rd3];

    // Load data[i]
    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    // data[i] += bias[channel]
    add.f32 %f2, %f2, %f1;
    st.global.f32 [%rd5], %f2;

$L__bias_exit:
    ret;
}

// col2im_f32: Scatter column matrix back to input spatial positions (reverse of im2col).
// Iterates over each output position in col and atomicAdds to the corresponding input position.
// col:     [C_in*kH*kW, out_H*out_W] (input column matrix, device pointer)
// output:  [C_in, H, W] (output image, device pointer - MUST be zero-initialized)
// params:  u32[10] = {H, W, kH, kW, pH, pW, sH, sW, oH, oW}
// n:       total col elements = C_in * kH * kW * oH * oW
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

    // Load conv params
    ld.global.u32 %r10, [%rd7 + 0];   // height
    ld.global.u32 %r11, [%rd7 + 4];   // width
    ld.global.u32 %r12, [%rd7 + 8];   // kernel_h
    ld.global.u32 %r13, [%rd7 + 12];  // kernel_w
    ld.global.u32 %r14, [%rd7 + 16];  // pad_h
    ld.global.u32 %r15, [%rd7 + 20];  // pad_w
    ld.global.u32 %r16, [%rd7 + 24];  // stride_h
    ld.global.u32 %r17, [%rd7 + 28];  // stride_w
    ld.global.u32 %r18, [%rd7 + 32];  // out_h
    ld.global.u32 %r19, [%rd7 + 36];  // out_w

    // Global thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r1, %r20;
    @%p1 bra $L__col2im_exit;

    // Same decomposition as im2col
    // w_col = idx % out_w
    rem.u32 %r4, %r1, %r19;
    // tmp = idx / out_w
    div.u32 %r5, %r1, %r19;
    // h_col = tmp % out_h
    rem.u32 %r6, %r5, %r18;
    // c_col = tmp / out_h
    div.u32 %r7, %r5, %r18;

    // kw_off = c_col % kW
    rem.u32 %r8, %r7, %r13;
    // tmp2 = c_col / kW
    div.u32 %r9, %r7, %r13;
    // kh_off = tmp2 % kH
    rem.u32 %r21, %r9, %r12;
    // c_in = tmp2 / kH
    div.u32 %r22, %r9, %r12;

    // h_in = h_col * stride_h + kh_off - pad_h
    mad.lo.s32 %r23, %r6, %r16, %r21;
    sub.s32 %r23, %r23, %r14;
    // w_in = w_col * stride_w + kw_off - pad_w
    mad.lo.s32 %r24, %r4, %r17, %r8;
    sub.s32 %r24, %r24, %r15;

    // Bounds check
    setp.lt.s32 %p2, %r23, 0;
    @%p2 bra $L__col2im_exit;
    setp.ge.s32 %p2, %r23, %r10;
    @%p2 bra $L__col2im_exit;
    setp.lt.s32 %p3, %r24, 0;
    @%p3 bra $L__col2im_exit;
    setp.ge.s32 %p3, %r24, %r11;
    @%p3 bra $L__col2im_exit;

    // Read col[idx]
    cvt.u64.u32 %rd3, %r1;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];

    // output[c_in * H * W + h_in * W + w_in] += col[idx]
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
"#;

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
/// in physical GGUF layout `[out, in]`. Used for Oracle-class models that
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
/// nexus-serve decode loop keep activations on GPU through the entire
/// layer instead of round-tripping to CPU after every matmul. Used by
/// `Tensor::rms_norm`, `Tensor::apply_rope_split_halves`, `Tensor::swiglu`,
/// and `Tensor::relu2_gate`.
pub const TRANSFORMER_OPS_PTX: &str = include_str!("transformer_ops.ptx");

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

        // Load element-wise kernels
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

        // Load activation kernels
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
            ],
        )?;

        // Load broadcast element-wise kernels
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
        kernels.load_module("sum_dim", SUM_DIM_PTX, &["sum_dim_f32"])?;

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

        // Load conv kernels (im2col + col2im + bias_add)
        kernels.load_module(
            "conv",
            CONV_PTX,
            &["im2col_f32", "col2im_f32", "bias_add_channels_f32"],
        )?;

        // Load attention mask expansion kernels
        kernels.load_module(
            "mask",
            MASK_PTX,
            &["mask_expand_causal_f32", "mask_expand_padding_f32"],
        )?;

        // Load strided gather kernel (for GPU-native contiguous())
        kernels.load_module("strided_copy", STRIDED_COPY_PTX, &["strided_gather_f32"])?;

        // Load embedding scatter-add kernel (for GPU-native embedding backward)
        kernels.load_module(
            "embedding",
            EMBEDDING_SCATTER_PTX,
            &["embedding_scatter_add_f32"],
        )?;

        // Load fused Adam optimizer + gradient utility kernels
        kernels.load_module(
            "adam",
            ADAM_PTX,
            &["adam_step_f32", "grad_norm_sq_f32", "grad_scale_f32"],
        )?;

        // Load fused LSTM/GRU/BatchNorm kernels
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
            ],
        )?;

        // Load pooling kernels (MaxPool2d + AvgPool2d forward/backward)
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

        // Load fused attention kernels (scaled dot-product attention forward + backward)
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

        // Load Q4_K dequant-in-shader matmul (LLM inference on quantized weights).
        // GEMV (m=1) for decode + GEMM (m>1) for prefill.
        kernels.load_module(
            "q4k_matmul",
            Q4K_MATMUL_PTX,
            &[
                "q4k_gemv_f32",
                "q4k_gemm_f32",
                "q4k_gemv_fused_qkv_f32",
                "q4k_gemv_fused_qkv_bias_f32",
                "q4k_gemv_fused_gate_up_f32",
                // Fused residual-add: x_out = x_in + matmul(a, w). Used
                // by forward_one_gpu_resident for O-proj (post-attention
                // residual) and down-proj (post-FFN residual).
                "q4k_gemv_residual_f32",
                // Fused gate/up + SwiGLU: writes ffn = silu(gate·a)*up·a
                // directly, skipping the gate_c/up_c intermediates and
                // the separate swiglu kernel launch.
                "q4k_gemv_fused_gate_up_swiglu_f32",
            ],
        )?;

        // Q5_K dequant-in-shader matmul — Phi-3's attn_qkv, Mistral
        // Q5_K_M bodies, and any model that mixes Q5_K into otherwise
        // Q4_K_M weights.
        kernels.load_module(
            "q5k_matmul",
            Q5K_MATMUL_PTX,
            &["q5k_gemv_f32", "q5k_gemm_f32"],
        )?;

        // Q5_0 / Q5_1 dequant-in-shader matmul — legacy Falcon bodies.
        kernels.load_module(
            "q5_01_matmul",
            Q5_01_MATMUL_PTX,
            &["q5_0_gemv_f32", "q5_0_gemm_f32", "q5_1_gemv_f32", "q5_1_gemm_f32"],
        )?;

        // Q6_K dequant-in-shader matmul — LM head + higher-precision weights.
        kernels.load_module(
            "q6k_matmul",
            Q6K_MATMUL_PTX,
            &["q6k_gemv_f32", "q6k_gemm_f32"],
        )?;

        // Transformer per-layer ops — RMSNorm / RoPE / SwiGLU / ReLU² gate.
        // Lets nexus-serve keep activations on GPU through the whole layer.
        kernels.load_module(
            "transformer_ops",
            TRANSFORMER_OPS_PTX,
            &[
                "rms_norm_f32",
                // Single-token LayerNorm (mean-subtracting, gamma+beta) —
                // Falcon arch decode. Distinct from `layer_norm_f32` in the
                // `layernorm` module which is the batched multi-row variant
                // used by training.
                "layer_norm_tokenwise_f32",
                // tanh-approximation GELU — Falcon MLP activation.
                "gelu_tanh_f32",
                // Parallel residual `x = x + attn + ffn` — Falcon's
                // parallel-attn+FFN block fuses two adds into one launch.
                "parallel_residual_add_f32",
                // `dst += src * scalar` — MoE expert-accumulate hot path.
                "scaled_add_inplace_f32",
                "rms_norm_heads_f32",
                "rope_split_halves_f32",
                "swiglu_f32",
                "relu2_gate_f32",
                // Batched (prefill, m>1) counterparts — used by
                // forward_batch_gpu_resident. Same math as their m=1
                // siblings; grid dims vary over tokens.
                "rms_norm_batched_f32",
                "rms_norm_heads_batched_f32",
                "rope_split_halves_batched_f32",
                "add_bias_batched_f32",
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
