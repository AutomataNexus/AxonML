//! Basic Gradient Functions - Arithmetic Operations
//!
//! Gradient functions for basic arithmetic operations: add, sub, mul, div, neg, pow.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

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
    #[must_use] pub fn new(
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
    #[must_use] pub fn new(
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

impl GradientFunction for MulBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad_lhs = grad_output * rhs
        let grad_lhs = grad_output.mul(&self.saved_rhs).unwrap();
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs, self.saved_lhs.shape());

        // grad_rhs = grad_output * lhs
        let grad_rhs = grad_output.mul(&self.saved_lhs).unwrap();
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

impl GradientFunction for DivBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad_lhs = grad_output / rhs
        let grad_lhs = grad_output.div(&self.saved_rhs).unwrap();
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs, self.saved_lhs.shape());

        // grad_rhs = -grad_output * lhs / rhs^2
        let rhs_sq = self.saved_rhs.mul(&self.saved_rhs).unwrap();
        let grad_rhs = grad_output.mul(&self.saved_lhs).unwrap();
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>) -> Self {
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>, exponent: f32) -> Self {
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
        let grad = grad_output.mul(&grad).unwrap();
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, input_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
        }
    }
}

impl GradientFunction for SumBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Broadcast the scalar gradient to the input shape
        let grad_value = grad_output.to_vec()[0];
        let numel: usize = self.input_shape.iter().product();
        let grad = Tensor::from_vec(vec![grad_value; numel], &self.input_shape).unwrap();
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
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, input_shape: Vec<usize>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
        }
    }
}

impl GradientFunction for MeanBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let numel: usize = self.input_shape.iter().product();
        let grad_value = grad_output.to_vec()[0] / numel as f32;
        let grad = Tensor::from_vec(vec![grad_value; numel], &self.input_shape).unwrap();
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
// Helper Functions
// =============================================================================

/// Reduces gradient to match the original input shape after broadcasting.
fn reduce_grad_for_broadcast(grad: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    if grad.shape() == target_shape {
        return grad.clone();
    }

    // Handle scalar target
    if target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == 1) {
        return Tensor::scalar(grad.to_vec().iter().sum::<f32>());
    }

    // For now, handle the simple case where shapes match in numel
    // A full implementation would handle arbitrary broadcasting
    let grad_numel: usize = grad.shape().iter().product();
    let target_numel: usize = target_shape.iter().product();

    if grad_numel == target_numel {
        // Same number of elements, just reshape
        let target_isize: Vec<isize> = target_shape.iter().map(|&x| x as isize).collect();
        return grad.reshape(&target_isize).unwrap_or_else(|_| grad.clone());
    }

    // Sum over extra elements
    let grad_data = grad.to_vec();
    let mut result_data = vec![0.0f32; target_numel];

    for (i, &val) in grad_data.iter().enumerate() {
        result_data[i % target_numel] += val;
    }

    Tensor::from_vec(result_data, target_shape).unwrap()
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

        let grad_output = Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap();
        let grads = grad_fn.apply(&grad_output);
        assert_eq!(grads.len(), 2);
        assert!(grads[0].is_some());
        assert!(grads[1].is_some());
    }

    #[test]
    fn test_mul_backward() {
        let lhs = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let rhs = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let grad_fn = MulBackward::new(None, None, lhs, rhs);

        let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        // grad_lhs should be rhs: [4, 5, 6]
        assert_eq!(grads[0].as_ref().unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
        // grad_rhs should be lhs: [1, 2, 3]
        assert_eq!(grads[1].as_ref().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pow_backward() {
        let input = Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap();
        let grad_fn = PowBackward::new(None, input, 2.0);

        let grad_output = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
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
