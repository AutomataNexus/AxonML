//! Backward functions for basic arithmetic and shape operations.
//!
//! 1067 lines. `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`,
//! `NegBackward`, `AddScalarBackward`, `MulScalarBackward`, `PowBackward`,
//! `ExpBackward`, `LogBackward`, `SqrtBackward`, `ClampBackward`,
//! `SumBackward`, `MeanBackward`, `SumDimBackward`, `MeanDimBackward`,
//! `VarDimBackward`, `ReshapeBackward`, `TransposeBackward`,
//! `NarrowBackward`, `SelectBackward`, `UnsqueezeBackward`,
//! `ExpandBackward`, `CatBackward`. Each stores the inputs/shapes needed
//! for the backward pass and implements `GradientFunction::backward`.
//!
//! # File
//! `crates/axonml-autograd/src/functions/basic.rs`
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

use std::any::Any;

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// Add Backward
// =============================================================================

/// Gradient function for addition.
///
/// d/dx(x + y) = 1, d/dy(x + y) = 1
#[derive(Debug)]
pub struct AddBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shapes: (Vec<usize>, Vec<usize>),
}

impl AddBackward {
    /// Creates a new `AddBackward`.
    #[must_use]
    pub fn new(
        lhs_grad_fn: Option<GradFn>,
        rhs_grad_fn: Option<GradFn>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    ) -> Self {
        Self {
            next_fns: vec![lhs_grad_fn, rhs_grad_fn],
            input_shapes: (lhs_shape, rhs_shape),
        }
    }
}

impl GradientFunction for AddBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Gradient flows through unchanged, but may need to reduce for broadcasting
        let grad_lhs = reduce_grad_for_broadcast(grad_output, &self.input_shapes.0);
        let grad_rhs = reduce_grad_for_broadcast(grad_output, &self.input_shapes.1);
        vec![Some(grad_lhs), Some(grad_rhs)]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Sub Backward
// =============================================================================

/// Gradient function for subtraction.
///
/// d/dx(x - y) = 1, d/dy(x - y) = -1
#[derive(Debug)]
pub struct SubBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shapes: (Vec<usize>, Vec<usize>),
}

impl SubBackward {
    /// Creates a new `SubBackward`.
    #[must_use]
    pub fn new(
        lhs_grad_fn: Option<GradFn>,
        rhs_grad_fn: Option<GradFn>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    ) -> Self {
        Self {
            next_fns: vec![lhs_grad_fn, rhs_grad_fn],
            input_shapes: (lhs_shape, rhs_shape),
        }
    }
}

impl GradientFunction for SubBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_lhs = reduce_grad_for_broadcast(grad_output, &self.input_shapes.0);
        let grad_rhs = reduce_grad_for_broadcast(&grad_output.neg(), &self.input_shapes.1);
        vec![Some(grad_lhs), Some(grad_rhs)]
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Mul Backward
// =============================================================================

/// Gradient function for multiplication.
///
/// d/dx(x * y) = y, d/dy(x * y) = x
#[derive(Debug)]
pub struct MulBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_lhs: Tensor<f32>,
    saved_rhs: Tensor<f32>,
}

impl MulBackward {
    /// Creates a new `MulBackward`.
    #[must_use]
    pub fn new(
        lhs_grad_fn: Option<GradFn>,
        rhs_grad_fn: Option<GradFn>,
        lhs: Tensor<f32>,
        rhs: Tensor<f32>,
    ) -> Self {
        Self {
            next_fns: vec![lhs_grad_fn, rhs_grad_fn],
            saved_lhs: lhs,
            saved_rhs: rhs,
        }
    }
}

impl GradientFunction for MulBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad_lhs = grad_output * rhs
        let grad_lhs = grad_output
            .mul(&self.saved_rhs)
            .expect("backward: tensor mul failed");
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs, self.saved_lhs.shape());

        // grad_rhs = grad_output * lhs
        let grad_rhs = grad_output
            .mul(&self.saved_lhs)
            .expect("backward: tensor mul failed");
        let grad_rhs = reduce_grad_for_broadcast(&grad_rhs, self.saved_rhs.shape());

        vec![Some(grad_lhs), Some(grad_rhs)]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Div Backward
// =============================================================================

/// Gradient function for division.
///
/// d/dx(x / y) = 1/y, d/dy(x / y) = -x/y^2
#[derive(Debug)]
pub struct DivBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_lhs: Tensor<f32>,
    saved_rhs: Tensor<f32>,
}

impl DivBackward {
    /// Creates a new `DivBackward`.
    #[must_use]
    pub fn new(
        lhs_grad_fn: Option<GradFn>,
        rhs_grad_fn: Option<GradFn>,
        lhs: Tensor<f32>,
        rhs: Tensor<f32>,
    ) -> Self {
        Self {
            next_fns: vec![lhs_grad_fn, rhs_grad_fn],
            saved_lhs: lhs,
            saved_rhs: rhs,
        }
    }
}

impl GradientFunction for DivBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad_lhs = grad_output / rhs
        let grad_lhs = grad_output.div(&self.saved_rhs).unwrap();
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs, self.saved_lhs.shape());

        // grad_rhs = -grad_output * lhs / rhs^2
        let rhs_sq = self
            .saved_rhs
            .mul(&self.saved_rhs)
            .expect("backward: tensor mul failed");
        let grad_rhs = grad_output
            .mul(&self.saved_lhs)
            .expect("backward: tensor mul failed");
        let grad_rhs = grad_rhs.div(&rhs_sq).unwrap().neg();
        let grad_rhs = reduce_grad_for_broadcast(&grad_rhs, self.saved_rhs.shape());

        vec![Some(grad_lhs), Some(grad_rhs)]
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Neg Backward
// =============================================================================

/// Gradient function for negation.
///
/// d/dx(-x) = -1
#[derive(Debug)]
pub struct NegBackward {
    next_fns: Vec<Option<GradFn>>,
}

impl NegBackward {
    /// Creates a new `NegBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
        }
    }
}

impl GradientFunction for NegBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        vec![Some(grad_output.neg())]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Pow Backward
// =============================================================================

/// Gradient function for power.
///
/// d/dx(x^n) = n * x^(n-1)
#[derive(Debug)]
pub struct PowBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    exponent: f32,
}

impl PowBackward {
    /// Creates a new `PowBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>, exponent: f32) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
            exponent,
        }
    }
}

impl GradientFunction for PowBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad = grad_output * exponent * input^(exponent - 1)
        let grad = self.saved_input.pow(self.exponent - 1.0);
        let grad = grad.mul_scalar(self.exponent);
        let grad = grad_output.mul(&grad).expect("backward: tensor mul failed");
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Sum Backward
// =============================================================================

/// Gradient function for sum reduction.
///
/// d/dx(sum(x)) = `ones_like(x)`
#[derive(Debug)]
pub struct SumBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
}

impl SumBackward {
    /// Creates a new `SumBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
        }
    }
}

impl GradientFunction for SumBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Broadcast the scalar gradient to the input shape (stays on GPU)
        let grad = grad_output.broadcast_to(&self.input_shape);
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Mean Backward
// =============================================================================

/// Gradient function for mean reduction.
///
/// d/dx(mean(x)) = `ones_like(x)` / numel(x)
#[derive(Debug)]
pub struct MeanBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
}

impl MeanBackward {
    /// Creates a new `MeanBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
        }
    }
}

impl GradientFunction for MeanBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let numel: usize = self.input_shape.iter().product();
        // Scale by 1/numel, then broadcast to input shape (stays on GPU)
        let scaled = grad_output.mul_scalar(1.0 / numel as f32);
        let grad = scaled.broadcast_to(&self.input_shape);
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// MeanDim Backward
// =============================================================================

/// Gradient function for mean along a dimension.
///
/// d/dx(mean(x, dim, keepdim)) = grad_output / dim_size, expanded to input shape
#[derive(Debug)]
pub struct MeanDimBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    dim: usize,
    keepdim: bool,
}

impl MeanDimBackward {
    /// Creates a new `MeanDimBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        dim: usize,
        keepdim: bool,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            dim,
            keepdim,
        }
    }
}

impl GradientFunction for MeanDimBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let dim_size = self.input_shape[self.dim];
        let scale = 1.0 / dim_size as f32;

        // GPU fast path: scale grad_output, then broadcast to input shape
        // This uses tensor ops which dispatch to GPU natively
        #[cfg(feature = "cuda")]
        if grad_output.device().is_gpu() {
            let scaled = grad_output.mul_scalar(scale);
            // Ensure keepdim shape for broadcasting
            let expanded = if self.keepdim {
                // Already has dim=1, can broadcast directly
                scaled.broadcast_to(&self.input_shape)
            } else {
                // Need to unsqueeze the reduced dim first
                let mut expanded_shape = grad_output.shape().to_vec();
                expanded_shape.insert(self.dim, 1);
                let reshaped_dims: Vec<isize> =
                    expanded_shape.iter().map(|&x| x as isize).collect();
                let reshaped = scaled
                    .reshape(&reshaped_dims)
                    .expect("backward: reshape failed");
                reshaped.broadcast_to(&self.input_shape)
            };
            return vec![Some(expanded.contiguous())];
        }

        // CPU path
        let grad_vec = grad_output.to_vec();
        let numel: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; numel];

        let ndim = self.input_shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.input_shape[i + 1];
        }

        let out_shape: Vec<usize> = if self.keepdim {
            let mut s = self.input_shape.clone();
            s[self.dim] = 1;
            s
        } else {
            let mut s = Vec::with_capacity(ndim - 1);
            for (i, &sz) in self.input_shape.iter().enumerate() {
                if i != self.dim {
                    s.push(sz);
                }
            }
            s
        };
        let out_ndim = out_shape.len();
        let mut out_strides = vec![1usize; out_ndim];
        if out_ndim > 1 {
            for i in (0..out_ndim - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }

        if numel >= 4096 {
            use rayon::prelude::*;
            grad_input.par_iter_mut().enumerate().for_each(|(flat_idx, gi)| {
                let mut remaining = flat_idx;
                let mut out_flat = 0usize;
                let mut out_d = 0;
                for d in 0..ndim {
                    let coord = remaining / strides[d];
                    remaining %= strides[d];
                    if d == self.dim {
                        if self.keepdim {
                            out_d += 1;
                        }
                    } else {
                        out_flat += coord * out_strides[out_d];
                        out_d += 1;
                    }
                }
                *gi = grad_vec[out_flat] * scale;
            });
        } else {
            for flat_idx in 0..numel {
                let mut remaining = flat_idx;
                let mut out_flat = 0usize;
                let mut out_d = 0;
                for d in 0..ndim {
                    let coord = remaining / strides[d];
                    remaining %= strides[d];
                    if d == self.dim {
                        if self.keepdim {
                            out_d += 1;
                        }
                    } else {
                        out_flat += coord * out_strides[out_d];
                        out_d += 1;
                    }
                }
                grad_input[flat_idx] = grad_vec[out_flat] * scale;
            }
        }

        let grad = Tensor::from_vec(grad_input, &self.input_shape)
            .expect("backward: tensor creation failed");
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "MeanDimBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// VarDim Backward
// =============================================================================

/// Gradient function for variance along a dimension.
///
/// d/dx(var(x, dim)) = 2 * (x - mean(x, dim)) / N
#[derive(Debug)]
pub struct VarDimBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    dim: usize,
    keepdim: bool,
}

impl VarDimBackward {
    /// Creates a new `VarDimBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        saved_input: Tensor<f32>,
        dim: usize,
        keepdim: bool,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input,
            dim,
            keepdim,
        }
    }
}

impl GradientFunction for VarDimBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: use tensor ops (mean_dim, sub, mul_scalar, mul, broadcast_to)
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let dim = self.dim as i32;
            let dim_size = self.saved_input.shape()[self.dim];
            // mean along dim (keepdim=true for broadcasting)
            let mean = self.saved_input.mean_dim(dim, true);
            // x - mean (broadcasts)
            let diff = self
                .saved_input
                .sub(&mean)
                .expect("backward: tensor sub failed");
            // 2 * (x - mean) / N
            let scale = 2.0 / dim_size as f32;
            let scaled_diff = diff.mul_scalar(scale);
            // Multiply by upstream gradient (broadcast grad_output to input shape)
            let grad_expanded = if self.keepdim {
                grad_output.broadcast_to(self.saved_input.shape())
            } else {
                // Insert dim=1 at the reduced dimension, then broadcast
                let mut expanded_shape = grad_output.shape().to_vec();
                expanded_shape.insert(self.dim, 1);
                let reshaped_dims: Vec<isize> =
                    expanded_shape.iter().map(|&x| x as isize).collect();
                let reshaped = grad_output
                    .reshape(&reshaped_dims)
                    .expect("backward: reshape failed");
                reshaped.broadcast_to(self.saved_input.shape())
            };
            let result = scaled_diff
                .mul(&grad_expanded)
                .expect("backward: tensor mul failed");
            return vec![Some(result)];
        }

        let input_shape = self.saved_input.shape();
        let input_vec = self.saved_input.to_vec();
        let grad_vec = grad_output.to_vec();
        let dim = self.dim;
        let dim_size = input_shape[dim];
        let ndim = input_shape.len();
        let numel: usize = input_shape.iter().product();

        // Compute strides
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * input_shape[i + 1];
        }

        // Compute output strides
        let out_shape: Vec<usize> = if self.keepdim {
            let mut s = input_shape.to_vec();
            s[dim] = 1;
            s
        } else {
            let mut s = Vec::new();
            for (i, &sz) in input_shape.iter().enumerate() {
                if i != dim {
                    s.push(sz);
                }
            }
            s
        };
        let out_ndim = out_shape.len();
        let mut out_strides = vec![1usize; out_ndim];
        if out_ndim > 1 {
            for i in (0..out_ndim - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }

        // Helper: map input flat index to output flat index (skipping dim)
        let map_to_out = |flat_idx: usize| -> usize {
            let mut remaining = flat_idx;
            let mut out_flat = 0usize;
            let mut out_d = 0;
            for d in 0..ndim {
                let coord = remaining / strides[d];
                remaining %= strides[d];
                if d == dim {
                    if self.keepdim {
                        out_d += 1;
                    }
                } else {
                    out_flat += coord * out_strides[out_d];
                    out_d += 1;
                }
            }
            out_flat
        };

        // First pass: compute means along dim
        // Parallelized with rayon fold+reduce for accumulation (local per-thread vecs, then merge).
        // Complements the parallel second pass for full CPU parallel VarDimBackward.
        let out_numel: usize = out_shape.iter().product();
        let (means, counts) = if numel >= 4096 {
            use rayon::prelude::*;
            (0..numel).into_par_iter().fold(
                || (vec![0.0f32; out_numel], vec![0usize; out_numel]),
                |mut acc, flat_idx| {
                    let out_idx = map_to_out(flat_idx);
                    acc.0[out_idx] += input_vec[flat_idx];
                    acc.1[out_idx] += 1;
                    acc
                }
            ).reduce(
                || (vec![0.0f32; out_numel], vec![0usize; out_numel]),
                |mut a, b| {
                    for i in 0..out_numel {
                        a.0[i] += b.0[i];
                        a.1[i] += b.1[i];
                    }
                    a
                }
            )
        } else {
            let mut means = vec![0.0f32; out_numel];
            let mut counts = vec![0usize; out_numel];
            for flat_idx in 0..numel {
                let out_idx = map_to_out(flat_idx);
                means[out_idx] += input_vec[flat_idx];
                counts[out_idx] += 1;
            }
            (means, counts)
        };
        let mut means = means; // to mut for divide
        for i in 0..out_numel {
            if counts[i] > 0 {
                means[i] /= counts[i] as f32;
            }
        }

        // Second pass: compute gradients = 2 * (x - mean) / N * grad_output
        // Parallelized with rayon (independent per flat_idx). Big win for CPU
        // variance/mean-dim backward (used in RMS/LayerNorm etc. for single-GPU/CPU training).
        let mut grad_input = vec![0.0f32; numel];
        let n = dim_size as f32;

        if numel >= 4096 {
            use rayon::prelude::*;
            grad_input.par_iter_mut().enumerate().for_each(|(flat_idx, gi)| {
                let out_idx = map_to_out(flat_idx);
                *gi = 2.0 * (input_vec[flat_idx] - means[out_idx]) / n * grad_vec[out_idx];
            });
        } else {
            for flat_idx in 0..numel {
                let out_idx = map_to_out(flat_idx);
                grad_input[flat_idx] =
                    2.0 * (input_vec[flat_idx] - means[out_idx]) / n * grad_vec[out_idx];
            }
        }

        let mut grad =
            Tensor::from_vec(grad_input, input_shape).expect("backward: tensor creation failed");
        // Preserve device
        if self.saved_input.device().is_gpu() {
            grad = grad.to_device(self.saved_input.device()).unwrap();
        }
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "VarDimBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Narrow Backward
// =============================================================================

/// Gradient function for narrow (slice along a dimension).
///
/// Forward: output = input.narrow(dim, start, length)
/// Backward: grad_input = zeros_like(input); grad_input.narrow(dim, start, length) = grad_output
#[derive(Debug)]
pub struct NarrowBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    dim: usize,
    start: usize,
}

impl NarrowBackward {
    /// Creates a new `NarrowBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        dim: usize,
        start: usize,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            dim,
            start,
        }
    }
}

impl GradientFunction for NarrowBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: use tensor-level narrow_backward_cuda
        #[cfg(feature = "cuda")]
        if grad_output.device().is_gpu() {
            let grad = grad_output.narrow_backward_cuda(&self.input_shape, self.dim, self.start);
            return vec![Some(grad)];
        }

        // CPU path — parallel over output elements (writes are to distinct input positions for a narrow slice).
        let numel: usize = self.input_shape.iter().product();
        let mut grad_data = vec![0.0f32; numel];
        let grad_out_data = grad_output.to_vec();

        let mut strides = vec![1usize; self.input_shape.len()];
        for i in (0..self.input_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.input_shape[i + 1];
        }

        let output_shape = grad_output.shape();
        let out_numel: usize = output_shape.iter().product();

        if out_numel >= 4096 {
            use rayon::prelude::*;
            let grad_ptr = grad_data.as_mut_ptr() as usize;
            (0..out_numel).into_par_iter().for_each(|out_idx| {
                let grad_ptr = grad_ptr as *mut f32;
                let mut indices = vec![0usize; output_shape.len()];
                let mut remaining = out_idx;
                for d in (0..output_shape.len()).rev() {
                    indices[d] = remaining % output_shape[d];
                    remaining /= output_shape[d];
                }

                indices[self.dim] += self.start;

                let in_idx: usize = indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                unsafe {
                    *grad_ptr.add(in_idx) = grad_out_data[out_idx];
                }
            });
        } else {
            for out_idx in 0..out_numel {
                let mut indices = vec![0usize; output_shape.len()];
                let mut remaining = out_idx;
                for d in (0..output_shape.len()).rev() {
                    indices[d] = remaining % output_shape[d];
                    remaining /= output_shape[d];
                }

                indices[self.dim] += self.start;

                let in_idx: usize = indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                grad_data[in_idx] = grad_out_data[out_idx];
            }
        }

        let grad = Tensor::from_vec(grad_data, &self.input_shape)
            .expect("backward: tensor creation failed");
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "NarrowBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Reduces gradient to match the original input shape after broadcasting.
fn reduce_grad_for_broadcast(grad: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone();
    }

    // Handle scalar target
    if target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == 1) {
        return grad.sum();
    }

    let grad_numel: usize = grad_shape.iter().product();
    let target_numel: usize = target_shape.iter().product();

    if grad_numel == target_numel {
        let target_isize: Vec<isize> = target_shape.iter().map(|&x| x as isize).collect();
        return grad.reshape(&target_isize).unwrap_or_else(|_| grad.clone());
    }

    // Fast path: sum over leading dimensions only (common bias backward case)
    // e.g., grad [4064, 4000] → target [4000]: sum dim 0
    // e.g., grad [4064, 128] → target [128]: sum dim 0
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();
    if target_ndim < grad_ndim {
        // Check if target matches trailing dims of grad
        let trailing_match = target_shape
            .iter()
            .rev()
            .zip(grad_shape.iter().rev())
            .all(|(t, g)| t == g);
        if trailing_match {
            let dims_to_reduce = grad_ndim - target_ndim;
            // For 2D grad summing along dim 0: use matmul for optimal GPU utilization
            // ones(1, M) @ grad(M, N) = result(1, N) — cuBLAS GEMM is highly optimized
            #[cfg(feature = "cuda")]
            if dims_to_reduce == 1 && grad_ndim == 2 && grad.device().is_gpu() {
                let m = grad_shape[0];
                let ones_data = vec![1.0f32; m];
                let ones = Tensor::from_vec(ones_data, &[1, m])
                    .unwrap()
                    .to_device(grad.device())
                    .unwrap();
                let result_2d = ones.matmul(grad).expect("backward: matmul failed");
                let target_isize: Vec<isize> = target_shape.iter().map(|&x| x as isize).collect();
                return result_2d
                    .reshape(&target_isize)
                    .expect("backward: reshape failed");
            }
            // General case: iteratively sum_dim(0)
            // On CPU, use direct parallel reduction for the common leading-dim bias case (heavy in training elementwise bwd).
            if !grad.device().is_gpu() {
                let leading_size: usize = grad_shape[..dims_to_reduce].iter().product();
                let out_size = target_numel;
                let mut out = vec![0.0f32; out_size];
                let g_data: Vec<f32> = grad.to_vec();
                if grad_numel >= 4096 {
                    use rayon::prelude::*;
                    out.par_iter_mut().enumerate().for_each(|(i, o)| {
                        let mut s = 0.0f32;
                        for k in 0..leading_size {
                            s += g_data[k * out_size + i];
                        }
                        *o = s;
                    });
                } else {
                    for i in 0..out_size {
                        let mut s = 0.0f32;
                        for k in 0..leading_size {
                            s += g_data[k * out_size + i];
                        }
                        out[i] = s;
                    }
                }
                return Tensor::from_vec(out, target_shape)
                    .expect("reduce_grad_for_broadcast: from_vec for leading sum");
            }
            let mut result = grad.clone();
            for _ in 0..dims_to_reduce {
                result = result.sum_dim(0, false);
            }
            return result;
        }
    }

    // General case: pad target_shape, sum over broadcast dims
    let pad = grad_ndim.saturating_sub(target_ndim);
    let mut padded_target = vec![1usize; pad];
    padded_target.extend_from_slice(target_shape);

    if !grad.device().is_gpu() {
        // Direct single-pass parallel reduction for CPU general case (complete the broadcast bwd cleanup; matches the leading hot-path style).
        // Par over grad elements; each maps to its collapsed target bin (reduce dims are ignored in the target flat calc) and accumulates.
        // Uses thread-local vecs + reduce (lock-free, like VarDim means pass).
        let g_data: Vec<f32> = grad.to_vec();
        let out_numel = target_numel;
        if grad_numel >= 4096 {
            use rayon::prelude::*;
            // Precompute target strides for the kept dims.
            let mut t_strides = vec![1usize; target_ndim];
            if target_ndim > 0 {
                for i in (0..target_ndim-1).rev() {
                    t_strides[i] = t_strides[i+1] * target_shape[i+1];
                }
            }
            let merged = (0..grad_numel).into_par_iter().fold(
                || vec![0.0f32; out_numel],
                |mut local, g_flat| {
                    // Decompose g_flat -> coords
                    let mut coords = vec![0usize; grad_ndim];
                    let mut rem = g_flat;
                    for d in (0..grad_ndim).rev() {
                        coords[d] = rem % grad_shape[d];
                        rem /= grad_shape[d];
                    }
                    // t_flat only from kept dims (padded_target[d]==1 means this is a reduce dim -> do not add its coord*stride)
                    let mut t_f = 0usize;
                    let mut t_d = 0;
                    for d in 0..grad_ndim {
                        if padded_target[d] != 1 {
                            t_f += coords[d] * t_strides[t_d];
                            t_d += 1;
                        }
                    }
                    local[t_f] += g_data[g_flat];
                    local
                }
            ).reduce(
                || vec![0.0f32; out_numel],
                |mut a, b| {
                    for i in 0..out_numel { a[i] += b[i]; }
                    a
                }
            );
            return Tensor::from_vec(merged, target_shape).expect("reduce_grad general cpu direct par");
        } else {
            let mut res = grad.clone();
            for d in 0..grad_ndim {
                if padded_target[d] == 1 && res.shape()[d] > 1 {
                    res = res.sum_dim(d as i32, true);
                }
            }
            return res;
        }
    }

    let mut result = grad.clone();
    for d in 0..grad_ndim {
        if padded_target[d] == 1 && result.shape()[d] > 1 {
            result = result.sum_dim(d as i32, true);
        }
    }

    // Reshape to target shape
    if result.shape() != target_shape {
        let target_isize: Vec<isize> = target_shape.iter().map(|&x| x as isize).collect();
        result = result.reshape(&target_isize).unwrap_or(result);
    }
    result
}

// =============================================================================
// MulScalar Backward
// =============================================================================

/// Gradient function for scalar multiplication.
///
/// d/dx(x * scalar) = scalar
#[derive(Debug)]
pub struct MulScalarBackward {
    next_fns: Vec<Option<GradFn>>,
    scalar: f32,
}

impl MulScalarBackward {
    /// Creates a new `MulScalarBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, scalar: f32) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            scalar,
        }
    }
}

impl GradientFunction for MulScalarBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // d/dx(x * scalar) = scalar → scale gradient by scalar (GPU-native via Tensor::mul_scalar)
        vec![Some(grad_output.mul_scalar(self.scalar))]
    }

    fn name(&self) -> &'static str {
        "MulScalarBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// AddScalar Backward
// =============================================================================

/// Gradient function for scalar addition.
///
/// d/dx(x + scalar) = 1
#[derive(Debug)]
pub struct AddScalarBackward {
    next_fns: Vec<Option<GradFn>>,
}

impl AddScalarBackward {
    /// Creates a new `AddScalarBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
        }
    }
}

impl GradientFunction for AddScalarBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // d/dx(x + scalar) = 1 → pass gradient through unchanged
        vec![Some(grad_output.clone())]
    }

    fn name(&self) -> &'static str {
        "AddScalarBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward() {
        let grad_fn = AddBackward::new(None, None, vec![2, 3], vec![2, 3]);
        assert_eq!(grad_fn.name(), "AddBackward");

        let grad_output =
            Tensor::from_vec(vec![1.0; 6], &[2, 3]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);
        assert_eq!(grads.len(), 2);
        assert!(grads[0].is_some());
        assert!(grads[1].is_some());
    }

    #[test]
    fn test_mul_backward() {
        let lhs =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("backward: tensor creation failed");
        let rhs =
            Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).expect("backward: tensor creation failed");
        let grad_fn = MulBackward::new(None, None, lhs, rhs);

        let grad_output =
            Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        // grad_lhs should be rhs: [4, 5, 6]
        assert_eq!(grads[0].as_ref().unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
        // grad_rhs should be lhs: [1, 2, 3]
        assert_eq!(grads[1].as_ref().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pow_backward() {
        let input =
            Tensor::from_vec(vec![2.0, 3.0], &[2]).expect("backward: tensor creation failed");
        let grad_fn = PowBackward::new(None, input, 2.0);

        let grad_output =
            Tensor::from_vec(vec![1.0, 1.0], &[2]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        // d/dx(x^2) = 2x, so [4.0, 6.0]
        assert_eq!(grads[0].as_ref().unwrap().to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_sum_backward() {
        let grad_fn = SumBackward::new(None, vec![2, 3]);

        let grad_output = Tensor::scalar(2.0);
        let grads = grad_fn.apply(&grad_output);

        // All elements get the same gradient
        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
        assert_eq!(grad.to_vec(), vec![2.0; 6]);
    }

    #[test]
    fn test_mean_backward() {
        let grad_fn = MeanBackward::new(None, vec![2, 3]);

        let grad_output = Tensor::scalar(1.0);
        let grads = grad_fn.apply(&grad_output);

        // Each element gets 1/6 of the gradient
        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
        for &v in &grad.to_vec() {
            assert!((v - 1.0 / 6.0).abs() < 1e-6);
        }
    }
}
