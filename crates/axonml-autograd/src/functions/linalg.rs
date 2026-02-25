//! Linear Algebra Gradient Functions
//!
//! Gradient functions for linear algebra operations: matmul, transpose, etc.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::any::Any;

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// MatMul Backward
// =============================================================================

/// Gradient function for matrix multiplication.
///
/// For C = A @ B:
/// dL/dA = dL/dC @ B^T
/// dL/dB = A^T @ dL/dC
#[derive(Debug)]
pub struct MatMulBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_lhs: Tensor<f32>,
    saved_rhs: Tensor<f32>,
}

impl MatMulBackward {
    /// Creates a new `MatMulBackward`.
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

impl GradientFunction for MatMulBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad_lhs = grad_output @ rhs^T
        let rhs_t = self.saved_rhs.t().unwrap();
        let grad_lhs = grad_output.matmul(&rhs_t).unwrap();

        // grad_rhs = lhs^T @ grad_output
        let lhs_t = self.saved_lhs.t().unwrap();
        let grad_rhs = lhs_t.matmul(grad_output).unwrap();

        vec![Some(grad_lhs), Some(grad_rhs)]
    }

    fn name(&self) -> &'static str {
        "MatMulBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Transpose Backward
// =============================================================================

/// Gradient function for transpose.
///
/// d/dx(x^T) = (`grad_output)^T`
#[derive(Debug)]
pub struct TransposeBackward {
    next_fns: Vec<Option<GradFn>>,
    dim0: usize,
    dim1: usize,
}

impl TransposeBackward {
    /// Creates a new `TransposeBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, dim0: usize, dim1: usize) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            dim0,
            dim1,
        }
    }
}

impl GradientFunction for TransposeBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Transpose the gradient back
        let grad = grad_output
            .transpose(self.dim0 as i64, self.dim1 as i64)
            .unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Reshape Backward
// =============================================================================

/// Gradient function for reshape.
///
/// d/dx(reshape(x)) = `reshape(grad_output`, `original_shape`)
#[derive(Debug)]
pub struct ReshapeBackward {
    next_fns: Vec<Option<GradFn>>,
    original_shape: Vec<usize>,
}

impl ReshapeBackward {
    /// Creates a new `ReshapeBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, original_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            original_shape,
        }
    }
}

impl GradientFunction for ReshapeBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape_isize: Vec<isize> = self.original_shape.iter().map(|&x| x as isize).collect();
        let grad = grad_output.reshape(&shape_isize).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Squeeze Backward
// =============================================================================

/// Gradient function for squeeze.
#[derive(Debug)]
pub struct SqueezeBackward {
    next_fns: Vec<Option<GradFn>>,
    original_shape: Vec<usize>,
}

impl SqueezeBackward {
    /// Creates a new `SqueezeBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, original_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            original_shape,
        }
    }
}

impl GradientFunction for SqueezeBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape_isize: Vec<isize> = self.original_shape.iter().map(|&x| x as isize).collect();
        let grad = grad_output.reshape(&shape_isize).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "SqueezeBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Unsqueeze Backward
// =============================================================================

/// Gradient function for unsqueeze.
#[derive(Debug)]
pub struct UnsqueezeBackward {
    next_fns: Vec<Option<GradFn>>,
    dim: usize,
}

impl UnsqueezeBackward {
    /// Creates a new `UnsqueezeBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, dim: usize) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            dim,
        }
    }
}

impl GradientFunction for UnsqueezeBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad = grad_output.squeeze(Some(self.dim as i64)).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// View/Contiguous Backward
// =============================================================================

/// Gradient function for view operations.
#[derive(Debug)]
pub struct ViewBackward {
    next_fns: Vec<Option<GradFn>>,
    original_shape: Vec<usize>,
}

impl ViewBackward {
    /// Creates a new `ViewBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, original_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            original_shape,
        }
    }
}

impl GradientFunction for ViewBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape_isize: Vec<isize> = self.original_shape.iter().map(|&x| x as isize).collect();
        let grad = grad_output.reshape(&shape_isize).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "ViewBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Expand Backward
// =============================================================================

/// Gradient function for expand (broadcast).
///
/// Forward: output = input.broadcast_to(new_shape)
/// Backward: grad_input = sum grad_output over expanded dimensions
#[derive(Debug)]
pub struct ExpandBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
}

impl ExpandBackward {
    /// Creates a new `ExpandBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
        }
    }
}

impl GradientFunction for ExpandBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let out_shape = grad_output.shape();
        let in_shape = &self.input_shape;

        // Sum over dimensions that were broadcast (size 1 -> size N)
        let grad_data = grad_output.to_vec();
        let in_numel: usize = in_shape.iter().product();
        let out_numel: usize = out_shape.iter().product();

        if in_numel == out_numel {
            // No broadcast happened, just reshape
            let grad = Tensor::from_vec(grad_data, in_shape).unwrap();
            return vec![Some(grad)];
        }

        // Compute strides for output shape
        let ndim = out_shape.len();
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        // Pad input shape with leading 1s to match output ndim
        let in_ndim = in_shape.len();
        let pad = ndim - in_ndim;
        let mut padded_in_shape = vec![1usize; pad];
        padded_in_shape.extend_from_slice(in_shape);

        // Compute strides for padded input shape
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            in_strides[i] = in_strides[i + 1] * padded_in_shape[i + 1];
        }

        // Accumulate gradients
        let mut grad_input = vec![0.0f32; in_numel];
        for out_idx in 0..out_numel {
            // Decompose out_idx into multi-dim indices
            let mut remaining = out_idx;
            let mut in_linear = 0usize;
            for d in 0..ndim {
                let coord = remaining / out_strides[d];
                remaining %= out_strides[d];
                // If input dim was 1 (broadcast), map to 0
                let in_coord = if padded_in_shape[d] == 1 { 0 } else { coord };
                in_linear += in_coord * in_strides[d];
            }
            grad_input[in_linear] += grad_data[out_idx];
        }

        let grad = Tensor::from_vec(grad_input, in_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Select Backward
// =============================================================================

/// Gradient function for select (index into a dimension, reducing rank by 1).
///
/// Forward: output = input.select(dim, index)  — shape removes dim
/// Backward: grad_input = zeros; grad_input[..., index, ...] = grad_output
#[derive(Debug)]
pub struct SelectBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    dim: usize,
    index: usize,
}

impl SelectBackward {
    /// Creates a new `SelectBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        dim: usize,
        index: usize,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            dim,
            index,
        }
    }
}

impl GradientFunction for SelectBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let in_numel: usize = self.input_shape.iter().product();
        let mut grad_data = vec![0.0f32; in_numel];
        let grad_out_data = grad_output.to_vec();
        let out_shape = grad_output.shape();

        // Compute input strides
        let ndim = self.input_shape.len();
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            in_strides[i] = in_strides[i + 1] * self.input_shape[i + 1];
        }

        // Compute output strides (output has dim removed)
        let out_ndim = out_shape.len();
        let mut out_strides = vec![1usize; out_ndim];
        if out_ndim > 0 {
            for i in (0..out_ndim - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }

        let out_numel: usize = out_shape.iter().product();
        for out_idx in 0..out_numel {
            // Decompose out_idx into multi-dim output indices
            let mut remaining = out_idx;
            let mut in_linear = 0usize;
            let mut out_d = 0;
            for d in 0..ndim {
                if d == self.dim {
                    // Insert the selected index
                    in_linear += self.index * in_strides[d];
                } else {
                    let coord = remaining / out_strides[out_d];
                    remaining %= out_strides[out_d];
                    in_linear += coord * in_strides[d];
                    out_d += 1;
                }
            }
            grad_data[in_linear] = grad_out_data[out_idx];
        }

        let grad = Tensor::from_vec(grad_data, &self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "SelectBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Cat Backward
// =============================================================================

/// Gradient function for concatenation along a dimension.
///
/// Splits grad_output back to each input along the concatenation dimension.
#[derive(Debug)]
pub struct CatBackward {
    next_fns: Vec<Option<GradFn>>,
    /// Size of each input along the cat dimension.
    sizes: Vec<usize>,
    dim: usize,
}

impl CatBackward {
    /// Creates a new `CatBackward`.
    #[must_use]
    pub fn new(next_fns: Vec<Option<GradFn>>, sizes: Vec<usize>, dim: usize) -> Self {
        Self {
            next_fns,
            sizes,
            dim,
        }
    }
}

impl GradientFunction for CatBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Split grad_output along dim according to sizes
        let mut offset = 0;
        let mut grads = Vec::with_capacity(self.sizes.len());
        for &size in &self.sizes {
            let grad = grad_output.narrow(self.dim, offset, size)
                .unwrap_or_else(|_| grad_output.clone());
            // narrow returns a view; make it contiguous
            let grad_data = grad.to_vec();
            let mut grad_shape: Vec<usize> = grad_output.shape().to_vec();
            grad_shape[self.dim] = size;
            grads.push(Some(Tensor::from_vec(grad_data, &grad_shape).unwrap()));
            offset += size;
        }
        grads
    }

    fn name(&self) -> &'static str {
        "CatBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// SumDim Backward
// =============================================================================

/// Gradient function for sum along a dimension.
///
/// d/dx(sum(x, dim)) = expand grad_output back to input shape along dim.
#[derive(Debug)]
pub struct SumDimBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    dim: usize,
}

impl SumDimBackward {
    /// Creates a new `SumDimBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input_shape: Vec<usize>, dim: usize) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            dim,
        }
    }
}

impl GradientFunction for SumDimBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad_output has input_shape with dim removed (or size 1 if keepdim).
        // We need to broadcast it back to input_shape by repeating along dim.
        let dim_size = self.input_shape[self.dim];
        let grad_data = grad_output.to_vec();

        let outer_size: usize = self.input_shape[..self.dim].iter().product();
        let inner_size: usize = self.input_shape[self.dim + 1..].iter().product();

        let in_numel: usize = self.input_shape.iter().product();
        let mut result = vec![0.0f32; in_numel];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let grad_idx = outer * inner_size + inner;
                let grad_val = grad_data[grad_idx];
                for d in 0..dim_size {
                    let in_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    result[in_idx] = grad_val;
                }
            }
        }

        let grad = Tensor::from_vec(result, &self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "SumDimBackward"
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
    fn test_matmul_backward() {
        // A: 2x3, B: 3x4, C: 2x4
        let a = Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
        let grad_fn = MatMulBackward::new(None, None, a, b);

        let grad_output = Tensor::from_vec(vec![1.0; 8], &[2, 4]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        // grad_lhs should be 2x3
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[2, 3]);
        // grad_rhs should be 3x4
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[3, 4]);
    }

    #[test]
    fn test_transpose_backward() {
        let grad_fn = TransposeBackward::new(None, 0, 1);

        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        // Transposing back should give 2x3
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_backward() {
        let grad_fn = ReshapeBackward::new(None, vec![2, 3]);

        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        assert_eq!(grads[0].as_ref().unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn test_expand_backward() {
        // Input (2, 1, 3) expanded to (2, 4, 3) — dim 1 broadcast from 1 to 4
        let grad_fn = ExpandBackward::new(None, vec![2, 1, 3]);
        let grad_output = Tensor::from_vec(vec![1.0; 24], &[2, 4, 3]).unwrap();
        let grads = grad_fn.apply(&grad_output);
        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 1, 3]);
        // Each value should be summed over the 4 repeats = 4.0
        assert_eq!(grad.to_vec(), vec![4.0; 6]);
    }

    #[test]
    fn test_select_backward() {
        // Input (3, 4), select dim=0, index=1 → output (4,)
        let grad_fn = SelectBackward::new(None, vec![3, 4], 0, 1);
        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let grads = grad_fn.apply(&grad_output);
        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[3, 4]);
        let data = grad.to_vec();
        // Row 0 should be zeros
        assert_eq!(&data[0..4], &[0.0, 0.0, 0.0, 0.0]);
        // Row 1 should be the grad_output
        assert_eq!(&data[4..8], &[1.0, 2.0, 3.0, 4.0]);
        // Row 2 should be zeros
        assert_eq!(&data[8..12], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unsqueeze_backward() {
        // unsqueeze(dim=1): (2, 3) → (2, 1, 3), backward squeezes dim 1
        let grad_fn = UnsqueezeBackward::new(None, 1);
        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap();
        let grads = grad_fn.apply(&grad_output);
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[2, 3]);
    }
}
