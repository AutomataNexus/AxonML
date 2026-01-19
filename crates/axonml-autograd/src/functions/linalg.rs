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
    #[must_use] pub fn new(
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, dim0: usize, dim1: usize) -> Self {
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, original_shape: Vec<usize>) -> Self {
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, original_shape: Vec<usize>) -> Self {
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, dim: usize) -> Self {
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, original_shape: Vec<usize>) -> Self {
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
}
