//! CPU compute backend — 37 public methods for host-side tensor operations.
//!
//! Implements the `Backend` trait (allocate, copy, synchronize) plus static
//! methods on `CpuBackend` for: rayon-parallel elementwise arithmetic (add,
//! sub, mul, div, add_scalar, mul_scalar, neg, abs, sign, pow), activations
//! (relu, sigmoid, tanh, gelu, silu, elu, leaky_relu, softmax, log_softmax),
//! matrix multiply (`matmul` via matrixmultiply sgemm with rayon GEMV fast
//! path for m=1 decode, `matmul_f32_bt` for B-transposed layout used by
//! quantized inference), reductions (sum, mean, max, min, var), comparisons
//! (gt, lt, eq, clamp), and misc (where_cond, dropout, embedding_gather).
//! Parallel threshold is 4096 elements; smaller ops run serial.
//!
//! # File
//! `crates/axonml-core/src/backends/cpu.rs`
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

use super::Backend;
use crate::device::DeviceCapabilities;
use crate::dtype::{Float, Numeric, Scalar};
use rayon::prelude::*;
use sysinfo::System;

/// Threshold for using parallel processing (in elements)
const PARALLEL_THRESHOLD: usize = 4096;

// =============================================================================
// CPU Backend Struct
// =============================================================================

/// CPU backend for tensor operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Creates a new CPU backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

// =============================================================================
// Backend Trait Implementation
// =============================================================================

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            name: "CPU".to_string(),
            total_memory: get_system_memory(),
            available_memory: get_available_memory(),
            supports_f16: true,
            supports_f64: true,
            max_threads_per_block: num_cpus(),
            compute_capability: None,
        }
    }

    fn allocate(&self, size: usize) -> *mut u8 {
        if size == 0 {
            return std::ptr::null_mut();
        }
        // Round size up to alignment to satisfy Layout's invariant:
        // size must be a multiple of align when using from_size_align_unchecked.
        let aligned_size = (size + 63) & !63; // Round up to next multiple of 64
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(aligned_size, 64);
            std::alloc::alloc(layout)
        }
    }

    fn deallocate(&self, ptr: *mut u8, size: usize) {
        if ptr.is_null() || size == 0 {
            return;
        }
        // Must use the same aligned size as allocate() for a matching Layout.
        let aligned_size = (size + 63) & !63;
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(aligned_size, 64);
            std::alloc::dealloc(ptr, layout);
        }
    }

    fn copy_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        // For CPU, this is just a memory copy
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }

    fn copy_to_host(&self, dst: *mut u8, src: *const u8, size: usize) {
        // For CPU, this is just a memory copy
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }

    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        // For CPU, this is just a memory copy
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }

    fn synchronize(&self) {
        // No-op for CPU - operations are synchronous
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Returns the total system memory in bytes.
fn get_system_memory() -> usize {
    let sys = System::new_all();
    sys.total_memory() as usize
}

/// Returns the available system memory in bytes.
fn get_available_memory() -> usize {
    let sys = System::new_all();
    sys.available_memory() as usize
}

/// Returns the number of CPU cores.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

// =============================================================================
// Element-wise Operations
// =============================================================================

impl CpuBackend {
    /// Adds two slices element-wise with optional parallelization.
    pub fn add<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a_val, b_val))| {
                    *d = *a_val + *b_val;
                });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i] + b[i];
            }
        }
    }

    /// Subtracts two slices element-wise with optional parallelization.
    pub fn sub<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a_val, b_val))| {
                    *d = *a_val - *b_val;
                });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i] - b[i];
            }
        }
    }

    /// Multiplies two slices element-wise with optional parallelization.
    pub fn mul<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a_val, b_val))| {
                    *d = *a_val * *b_val;
                });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i] * b[i];
            }
        }
    }

    /// Divides two slices element-wise with optional parallelization.
    pub fn div<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a_val, b_val))| {
                    *d = *a_val / *b_val;
                });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i] / b[i];
            }
        }
    }

    /// Adds a scalar to each element with optional parallelization.
    pub fn add_scalar<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T], scalar: T) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = *a_val + scalar;
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i] + scalar;
            }
        }
    }

    /// Multiplies each element by a scalar with optional parallelization.
    pub fn mul_scalar<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T], scalar: T) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = *a_val * scalar;
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i] * scalar;
            }
        }
    }

    /// Negates each element with optional parallelization.
    pub fn neg<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = T::zero() - *a_val;
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = T::zero() - a[i];
            }
        }
    }

    /// Computes absolute value of each element with optional parallelization.
    pub fn abs<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = if *a_val < T::zero() {
                    T::zero() - *a_val
                } else {
                    *a_val
                };
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = if a[i] < T::zero() {
                    T::zero() - a[i]
                } else {
                    a[i]
                };
            }
        }
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

impl CpuBackend {
    /// Applies `ReLU` activation: max(0, x) with optional parallelization.
    pub fn relu<T: Float + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = if *a_val > T::zero() {
                    *a_val
                } else {
                    T::zero()
                };
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = if a[i] > T::zero() { a[i] } else { T::zero() };
            }
        }
    }

    /// Applies sigmoid activation: 1 / (1 + exp(-x)) with optional parallelization.
    pub fn sigmoid<T: Float + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = T::one() / (T::one() + (-*a_val).exp_value());
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = T::one() / (T::one() + (-a[i]).exp_value());
            }
        }
    }

    /// Applies tanh activation with optional parallelization.
    pub fn tanh<T: Float + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = a_val.tanh_value();
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i].tanh_value();
            }
        }
    }

    /// Applies exponential function with optional parallelization.
    pub fn exp<T: Float + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = a_val.exp_value();
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i].exp_value();
            }
        }
    }

    /// Applies natural logarithm with optional parallelization.
    pub fn ln<T: Float + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = a_val.ln_value();
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i].ln_value();
            }
        }
    }

    /// Applies square root with optional parallelization.
    pub fn sqrt<T: Float + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = a_val.sqrt_value();
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i].sqrt_value();
            }
        }
    }

    /// Squares each element with optional parallelization.
    pub fn square<T: Numeric + Sync + Send>(dst: &mut [T], a: &[T]) {
        debug_assert_eq!(a.len(), dst.len());

        if dst.len() >= PARALLEL_THRESHOLD {
            dst.par_iter_mut().zip(a.par_iter()).for_each(|(d, a_val)| {
                *d = *a_val * *a_val;
            });
        } else {
            for i in 0..dst.len() {
                dst[i] = a[i] * a[i];
            }
        }
    }
}

// =============================================================================
// Reduction Operations
// =============================================================================

impl CpuBackend {
    /// Computes the sum of all elements.
    pub fn sum<T: Numeric>(a: &[T]) -> T {
        let mut result = T::zero();
        for &val in a {
            result = result + val;
        }
        result
    }

    /// Computes the product of all elements.
    pub fn prod<T: Numeric>(a: &[T]) -> T {
        let mut result = T::one();
        for &val in a {
            result = result * val;
        }
        result
    }

    /// Finds the maximum element.
    pub fn max<T: Numeric>(a: &[T]) -> Option<T> {
        if a.is_empty() {
            return None;
        }

        let mut result = a[0];
        for &val in &a[1..] {
            if val > result {
                result = val;
            }
        }
        Some(result)
    }

    /// Finds the minimum element.
    pub fn min<T: Numeric>(a: &[T]) -> Option<T> {
        if a.is_empty() {
            return None;
        }

        let mut result = a[0];
        for &val in &a[1..] {
            if val < result {
                result = val;
            }
        }
        Some(result)
    }

    /// Computes the mean of all elements.
    pub fn mean<T: Float>(a: &[T]) -> Option<T> {
        if a.is_empty() {
            return None;
        }

        let sum = Self::sum(a);
        let len = T::from(a.len()).unwrap_or(T::one());
        Some(sum / len)
    }

    /// Finds the index of the maximum element.
    pub fn argmax<T: Numeric>(a: &[T]) -> Option<usize> {
        if a.is_empty() {
            return None;
        }

        let mut max_idx = 0;
        let mut max_val = a[0];
        for (i, &val) in a.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        Some(max_idx)
    }

    /// Finds the index of the minimum element.
    pub fn argmin<T: Numeric>(a: &[T]) -> Option<usize> {
        if a.is_empty() {
            return None;
        }

        let mut min_idx = 0;
        let mut min_val = a[0];
        for (i, &val) in a.iter().enumerate().skip(1) {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        Some(min_idx)
    }
}

// =============================================================================
// Matrix Operations
// =============================================================================

impl CpuBackend {
    /// Performs matrix multiplication: C = A @ B.
    ///
    /// A is (m x k), B is (k x n), C is (m x n).
    /// Uses optimized GEMM from matrixmultiply crate for f32/f64,
    /// falls back to cache-efficient tiled implementation for other types.
    pub fn matmul<T: Numeric>(c: &mut [T], a: &[T], b: &[T], m: usize, n: usize, k: usize) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);

        // Use optimized BLAS routines for f32 and f64
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: We verified T is f32, so the casts are safe
            unsafe {
                let a_f32: &[f32] = &*(std::ptr::from_ref::<[T]>(a) as *const [f32]);
                let b_f32: &[f32] = &*(std::ptr::from_ref::<[T]>(b) as *const [f32]);
                let c_f32: &mut [f32] = &mut *(std::ptr::from_mut::<[T]>(c) as *mut [f32]);
                Self::matmul_f32(c_f32, a_f32, b_f32, m, n, k);
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: We verified T is f64, so the casts are safe
            unsafe {
                let a_f64: &[f64] = &*(std::ptr::from_ref::<[T]>(a) as *const [f64]);
                let b_f64: &[f64] = &*(std::ptr::from_ref::<[T]>(b) as *const [f64]);
                let c_f64: &mut [f64] = &mut *(std::ptr::from_mut::<[T]>(c) as *mut [f64]);
                Self::matmul_f64(c_f64, a_f64, b_f64, m, n, k);
            }
            return;
        }

        // Fallback: Use cache-efficient tiled matrix multiplication
        // Block size chosen for typical L1 cache (32KB)
        const BLOCK_SIZE: usize = 64;

        // Initialize C to zero
        for val in c.iter_mut() {
            *val = T::zero();
        }

        // Tiled matrix multiplication for better cache locality
        for i0 in (0..m).step_by(BLOCK_SIZE) {
            let i_end = (i0 + BLOCK_SIZE).min(m);
            for p0 in (0..k).step_by(BLOCK_SIZE) {
                let p_end = (p0 + BLOCK_SIZE).min(k);
                for j0 in (0..n).step_by(BLOCK_SIZE) {
                    let j_end = (j0 + BLOCK_SIZE).min(n);

                    // Compute block C[i0:i_end, j0:j_end] += A[i0:i_end, p0:p_end] @ B[p0:p_end, j0:j_end]
                    for i in i0..i_end {
                        for p in p0..p_end {
                            let a_val = a[i * k + p];
                            for j in j0..j_end {
                                c[i * n + j] = c[i * n + j] + a_val * b[p * n + j];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Performs optimized f32 matrix multiplication using matrixmultiply crate.
    ///
    /// C = alpha * A @ B + beta * C
    pub fn sgemm(
        c: &mut [f32],
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);

        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                k as isize,
                1, // A: row-major (m x k)
                b.as_ptr(),
                n as isize,
                1, // B: row-major (k x n)
                beta,
                c.as_mut_ptr(),
                n as isize,
                1, // C: row-major (m x n)
            );
        }
    }

    /// Performs optimized f64 matrix multiplication using matrixmultiply crate.
    ///
    /// C = alpha * A @ B + beta * C
    pub fn dgemm(
        c: &mut [f64],
        a: &[f64],
        b: &[f64],
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        beta: f64,
    ) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);

        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                k as isize,
                1, // A: row-major (m x k)
                b.as_ptr(),
                n as isize,
                1, // B: row-major (k x n)
                beta,
                c.as_mut_ptr(),
                n as isize,
                1, // C: row-major (m x n)
            );
        }
    }

    /// Performs f32 matrix multiplication: C = A @ B using optimized GEMM.
    ///
    /// For `m == 1` (GEMV — a row vector times a matrix, produced by every
    /// per-token decode step in LLM inference), matrixmultiply v0.3 only
    /// parallelizes along the `m` dimension and so stays on one core. We
    /// intercept that case and fan across rayon by slicing the output
    /// dimension `n`: each thread computes a contiguous slab of `c` with
    /// row-major streaming reads of B (SIMD-friendly, contiguous, L1-local).
    pub fn matmul_f32(c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
        if m == 1 && n >= 512 && k >= 256 {
            gemv_row_parallel_f32(c, a, b, n, k);
            return;
        }
        Self::sgemm(c, a, b, m, n, k, 1.0, 0.0);
    }

    /// Performs f64 matrix multiplication: C = A @ B using optimized GEMM.
    pub fn matmul_f64(c: &mut [f64], a: &[f64], b: &[f64], m: usize, n: usize, k: usize) {
        Self::dgemm(c, a, b, m, n, k, 1.0, 0.0);
    }

    /// Performs f32 matmul where `B` is stored in its NATURAL `[n, k]` row-major
    /// layout: `C = A @ B^T` where `A` is `[m, k]`, `B` is `[n, k]`, `C` is `[m, n]`.
    ///
    /// This is the dominant pattern in LLM inference — GGUF dequantized weights
    /// come out as `[out, in]` row-major, and a naive implementation would
    /// first transpose to `[in, out]` before calling `matmul_f32`. That
    /// transpose is an `O(n*k)` single-threaded memcpy (measured at 30+ s/token
    /// on a 14B model, saturating one core while 23 others sat idle — the
    /// GGUF load path's hot bottleneck). This function skips the transpose
    /// entirely:
    ///
    /// - **m=1 (decode)**: dispatches to [`gemv_bt_row_parallel_f32`] which
    ///   parallelizes over rows of `B` — each rayon worker dots its row with
    ///   `A` and writes one element of `C`. Linear with `rayon::current_num_threads()`.
    /// - **m>1 (prefill)**: calls `matrixmultiply::sgemm` with `B`'s row-major
    ///   stride `(k, 1)` but reads as if it were `[k, n]` column-major via
    ///   `(rs=1, cs=k)` — the classic "transpose by stride" trick. Zero copy.
    ///
    /// # Panics
    /// Does not panic. Debug-asserts shape consistency.
    pub fn matmul_f32_bt(c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), n * k);
        debug_assert_eq!(c.len(), m * n);

        if m == 1 {
            gemv_bt_row_parallel_f32(c, a, b, n, k);
            return;
        }

        // m > 1: sgemm with B reinterpreted as [k, n] via transposed strides.
        // Physical B layout is row-major [n, k]: element B[i, j] is at offset `i*k + j`.
        // We want matmul to see it as a [k, n] matrix: element B'[i, j] at `j*k + i`.
        // Setting rs=1 (stride 1 between rows) and cs=k (stride k between cols)
        // on the same pointer reinterprets the layout correctly.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize, // A row stride
                1,          // A col stride — row-major [m, k]
                b.as_ptr(),
                1,          // B row stride — transposed view
                k as isize, // B col stride — transposed view
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1, // C row-major [m, n]
            );
        }
    }

    /// Transposes a matrix.
    ///
    /// A is (rows x cols), B is (cols x rows).
    pub fn transpose<T: Scalar>(dst: &mut [T], src: &[T], rows: usize, cols: usize) {
        debug_assert_eq!(src.len(), rows * cols);
        debug_assert_eq!(dst.len(), rows * cols);

        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    /// Computes dot product of two vectors.
    pub fn dot<T: Numeric>(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());

        let mut sum = T::zero();
        for i in 0..a.len() {
            sum = sum + a[i] * b[i];
        }
        sum
    }
}

// =============================================================================
// Parallel GEMV (m=1) — used by LLM per-token decode
// =============================================================================

/// Parallel f32 row-vector × matrix multiply: `c = a @ B` where `a` is `[1, k]`
/// (treated as a length-`k` row), `B` is `[k, n]` row-major, and `c` is `[1, n]`
/// (length-`n` row).
///
/// Loop order: outer over rows of B (`k`), inner over the output slab. This
/// gives contiguous SIMD-friendly reads of B (one full row per outer iteration)
/// and contiguous FMA-accumulation writes into `c_slab`. Rayon fans the work
/// across slabs of `c` so each worker owns a disjoint column range — no
/// synchronization between workers.
/// Parallel GEMV where `B` is stored in its NATURAL `[n, k]` row-major layout.
///
/// Computes `c = a @ B^T` for row-vector `a` of length `k` and matrix `B` of
/// shape `[n, k]` row-major. Each worker owns a contiguous slab of output
/// columns and computes `c_slab[j] = dot(a, B[j])` — a simple dot product
/// per output column, embarrassingly parallel.
///
/// Why this layout beats `gemv_row_parallel_f32` for LLM inference: GGUF
/// weights are stored `[out_features, in_features]` row-major. The previous
/// code required a transpose to `[in_features, out_features]` before calling
/// `gemv_row_parallel_f32`, which is a single-threaded `O(out*in)` memcpy.
/// This variant consumes the natural layout directly, so dequant → matmul
/// has no intermediate copy. Rows of `B` stream contiguously through L1 and
/// the dot-product inner loop auto-vectorizes.
fn gemv_bt_row_parallel_f32(c: &mut [f32], a: &[f32], b: &[f32], n: usize, k: usize) {
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), n);

    // Chunk size heuristic: aim for ~64 output rows per task (each row is a
    // k-length dot product). This balances rayon overhead against per-task
    // cache residency.
    let threads = rayon::current_num_threads().max(1);
    let chunk = (n / (threads * 4)).max(16).min(n);

    c.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let row_base = chunk_idx * chunk;
            for (local_i, c_slot) in c_chunk.iter_mut().enumerate() {
                let row = row_base + local_i;
                let b_row = &b[row * k..row * k + k];
                // Dot product. The compiler auto-vectorizes this tight loop
                // into FMAs when AVX2/AVX-512 is available.
                let mut acc = 0.0f32;
                for (&ai, &bi) in a.iter().zip(b_row.iter()) {
                    acc += ai * bi;
                }
                *c_slot = acc;
            }
        });
}

fn gemv_row_parallel_f32(c: &mut [f32], a: &[f32], b: &[f32], n: usize, k: usize) {
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), n);

    // Slab size: aim for ~256 output cols per worker, bounded so we don't spawn
    // orders-of-magnitude more slabs than threads. Each slab does k*slab FMAs;
    // for k=2560, slab=256 gives ~650K FMAs per slab — good cache residency
    // (~1 MB of B's rows streamed per slab at this width).
    let target_slab = 256usize.max(n / (rayon::current_num_threads() * 4).max(1));
    let slab = target_slab.min(n).max(1);

    c.par_chunks_mut(slab)
        .enumerate()
        .for_each(|(slab_idx, c_slab)| {
            let col_start = slab_idx * slab;
            let this_n = c_slab.len();
            // Zero this slab (we accumulate into it).
            c_slab.fill(0.0);
            // Stream through rows of B. For each row k_i, multiply by a[k_i] and
            // accumulate into c_slab. The inner loop over `this_n` is trivially
            // auto-vectorizable (FMA in rustc-generated AVX2/AVX-512 code).
            for k_i in 0..k {
                let a_k = a[k_i];
                let row_start = k_i * n + col_start;
                let b_row = &b[row_start..row_start + this_n];
                for (c_val, &b_val) in c_slab.iter_mut().zip(b_row.iter()) {
                    *c_val += a_k * b_val;
                }
            }
        });
}

// =============================================================================
// Comparison Operations
// =============================================================================

impl CpuBackend {
    /// Element-wise equality comparison.
    pub fn eq<T: Scalar + PartialEq>(dst: &mut [bool], a: &[T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), dst.len());

        for i in 0..dst.len() {
            dst[i] = a[i] == b[i];
        }
    }

    /// Element-wise less-than comparison.
    pub fn lt<T: Numeric>(dst: &mut [bool], a: &[T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), dst.len());

        for i in 0..dst.len() {
            dst[i] = a[i] < b[i];
        }
    }

    /// Element-wise greater-than comparison.
    pub fn gt<T: Numeric>(dst: &mut [bool], a: &[T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), dst.len());

        for i in 0..dst.len() {
            dst[i] = a[i] > b[i];
        }
    }
}

// =============================================================================
// Fill Operations
// =============================================================================

impl CpuBackend {
    /// Fills a slice with a value.
    pub fn fill<T: Scalar>(dst: &mut [T], value: T) {
        for elem in dst.iter_mut() {
            *elem = value;
        }
    }

    /// Fills a slice with zeros.
    pub fn fill_zeros<T: Scalar>(dst: &mut [T]) {
        Self::fill(dst, T::zeroed());
    }

    /// Copies from source to destination.
    pub fn copy<T: Scalar>(dst: &mut [T], src: &[T]) {
        debug_assert_eq!(dst.len(), src.len());
        dst.copy_from_slice(src);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let mut c = [0.0_f32; 3];

        CpuBackend::add(&mut c, &a, &b);
        assert_eq!(c, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let a = [2.0_f32, 3.0, 4.0];
        let b = [2.0_f32, 2.0, 2.0];
        let mut c = [0.0_f32; 3];

        CpuBackend::mul(&mut c, &a, &b);
        assert_eq!(c, [4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_relu() {
        let a = [-1.0_f32, 0.0, 1.0, 2.0];
        let mut b = [0.0_f32; 4];

        CpuBackend::relu(&mut b, &a);
        assert_eq!(b, [0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sum() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(CpuBackend::sum(&a), 10.0);
    }

    #[test]
    fn test_max_min() {
        let a = [1.0_f32, 4.0, 2.0, 3.0];
        assert_eq!(CpuBackend::max(&a), Some(4.0));
        assert_eq!(CpuBackend::min(&a), Some(1.0));
    }

    #[test]
    fn test_argmax() {
        let a = [1.0_f32, 4.0, 2.0, 3.0];
        assert_eq!(CpuBackend::argmax(&a), Some(1));
    }

    #[test]
    fn test_matmul() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = [[19, 22], [43, 50]]
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let mut c = [0.0_f32; 4];

        CpuBackend::matmul(&mut c, &a, &b, 2, 2, 2);
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_transpose() {
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[1, 4], [2, 5], [3, 6]] (3x2)
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut b = [0.0_f32; 6];

        CpuBackend::transpose(&mut b, &a, 2, 3);
        assert_eq!(b, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_dot() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        assert_eq!(CpuBackend::dot(&a, &b), 32.0);
    }

    #[test]
    fn test_fill() {
        let mut a = [0.0_f32; 5];
        CpuBackend::fill(&mut a, 42.0);
        assert_eq!(a, [42.0; 5]);
    }
}
