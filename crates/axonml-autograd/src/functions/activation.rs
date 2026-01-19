//! Activation Gradient Functions
//!
//! Gradient functions for activation operations: `ReLU`, Sigmoid, Tanh, Softmax, etc.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::any::Any;

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// ReLU Backward
// =============================================================================

/// Gradient function for `ReLU`.
///
/// d/dx(relu(x)) = 1 if x > 0, else 0
#[derive(Debug)]
pub struct ReluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
}

impl ReluBackward {
    /// Creates a new `ReluBackward`.
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
        }
    }
}

impl GradientFunction for ReluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Gradient is grad_output where input > 0, else 0
        let input_data = self.saved_input.to_vec();
        let grad_data = grad_output.to_vec();

        let result: Vec<f32> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect();

        vec![Some(
            Tensor::from_vec(result, self.saved_input.shape()).unwrap(),
        )]
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Sigmoid Backward
// =============================================================================

/// Gradient function for Sigmoid.
///
/// d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
#[derive(Debug)]
pub struct SigmoidBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
}

impl SigmoidBackward {
    /// Creates a new `SigmoidBackward`.
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
        }
    }
}

impl GradientFunction for SigmoidBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad = grad_output * output * (1 - output)
        let output_data = self.saved_output.to_vec();
        let grad_data = grad_output.to_vec();

        let result: Vec<f32> = output_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&o, &g)| g * o * (1.0 - o))
            .collect();

        vec![Some(
            Tensor::from_vec(result, self.saved_output.shape()).unwrap(),
        )]
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tanh Backward
// =============================================================================

/// Gradient function for Tanh.
///
/// d/dx(tanh(x)) = 1 - tanh(x)^2
#[derive(Debug)]
pub struct TanhBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
}

impl TanhBackward {
    /// Creates a new `TanhBackward`.
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
        }
    }
}

impl GradientFunction for TanhBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad = grad_output * (1 - output^2)
        let output_data = self.saved_output.to_vec();
        let grad_data = grad_output.to_vec();

        let result: Vec<f32> = output_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&o, &g)| g * (1.0 - o * o))
            .collect();

        vec![Some(
            Tensor::from_vec(result, self.saved_output.shape()).unwrap(),
        )]
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Softmax Backward
// =============================================================================

/// Gradient function for Softmax.
///
/// The Jacobian of softmax is: diag(s) - s * s^T
/// For element i: `ds_i/dx_j` = `s_i` * (`delta_ij` - `s_j`)
#[derive(Debug)]
pub struct SoftmaxBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
    dim: i64,
}

impl SoftmaxBackward {
    /// Creates a new `SoftmaxBackward`.
    ///
    /// # Arguments
    /// * `input_grad_fn` - The gradient function from the input
    /// * `output` - The softmax output (saved for backward computation)
    /// * `dim` - The dimension along which softmax was applied
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>, dim: i64) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
            dim,
        }
    }
}

impl GradientFunction for SoftmaxBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape = self.saved_output.shape();
        let ndim = shape.len();

        // Normalize dim to positive index
        let dim = if self.dim < 0 {
            (ndim as i64 + self.dim) as usize
        } else {
            self.dim as usize
        };

        let s = self.saved_output.to_vec();
        let g = grad_output.to_vec();
        let mut result = vec![0.0f32; s.len()];

        if ndim == 1 {
            // 1D case: simple dot product
            let dot: f32 = s.iter().zip(g.iter()).map(|(&si, &gi)| si * gi).sum();
            for i in 0..s.len() {
                result[i] = s[i] * (g[i] - dot);
            }
        } else if ndim == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            if dim == 0 {
                // Softmax along rows (each column is independent)
                for col in 0..cols {
                    let mut dot = 0.0f32;
                    for row in 0..rows {
                        let idx = row * cols + col;
                        dot += s[idx] * g[idx];
                    }
                    for row in 0..rows {
                        let idx = row * cols + col;
                        result[idx] = s[idx] * (g[idx] - dot);
                    }
                }
            } else {
                // Softmax along columns (each row is independent) - most common case
                for row in 0..rows {
                    let start = row * cols;
                    let mut dot = 0.0f32;
                    for col in 0..cols {
                        let idx = start + col;
                        dot += s[idx] * g[idx];
                    }
                    for col in 0..cols {
                        let idx = start + col;
                        result[idx] = s[idx] * (g[idx] - dot);
                    }
                }
            }
        } else {
            // General N-D case: compute strides and iterate
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            let dim_size = shape[dim];
            let dim_stride = strides[dim];
            let total = s.len();
            let outer_size = total / dim_size;

            for outer in 0..outer_size {
                // Calculate base index for this slice
                let mut base_idx = 0;
                let mut temp = outer;
                for d in 0..ndim {
                    if d != dim {
                        let divisor = outer_size / strides[d] * if d > dim { dim_size } else { 1 };
                        let coord = temp / divisor;
                        temp %= divisor;
                        base_idx += coord * strides[d];
                    }
                }

                // Compute dot product along this slice
                let mut dot = 0.0f32;
                for i in 0..dim_size {
                    let idx = base_idx + i * dim_stride;
                    if idx < total {
                        dot += s[idx] * g[idx];
                    }
                }

                // Compute gradient for this slice
                for i in 0..dim_size {
                    let idx = base_idx + i * dim_stride;
                    if idx < total {
                        result[idx] = s[idx] * (g[idx] - dot);
                    }
                }
            }
        }

        vec![Some(Tensor::from_vec(result, shape).unwrap())]
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// LeakyReLU Backward
// =============================================================================

/// Gradient function for `LeakyReLU`.
///
/// `d/dx(leaky_relu(x))` = 1 if x > 0, else `negative_slope`
#[derive(Debug)]
pub struct LeakyReluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    negative_slope: f32,
}

impl LeakyReluBackward {
    /// Creates a new `LeakyReluBackward`.
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>, negative_slope: f32) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
            negative_slope,
        }
    }
}

impl GradientFunction for LeakyReluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let input_data = self.saved_input.to_vec();
        let grad_data = grad_output.to_vec();

        let result: Vec<f32> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { g * self.negative_slope })
            .collect();

        vec![Some(
            Tensor::from_vec(result, self.saved_input.shape()).unwrap(),
        )]
    }

    fn name(&self) -> &'static str {
        "LeakyReluBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// GELU Backward
// =============================================================================

/// Gradient function for GELU (Gaussian Error Linear Unit).
///
/// GELU(x) = x * Phi(x), where Phi is the CDF of standard normal.
/// Approximate: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[derive(Debug)]
pub struct GeluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
}

impl GeluBackward {
    /// Creates a new `GeluBackward`.
    #[must_use] pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
        }
    }
}

impl GradientFunction for GeluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let input_data = self.saved_input.to_vec();
        let grad_data = grad_output.to_vec();

        let sqrt_2_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let c = 0.044715_f32;

        let result: Vec<f32> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let x3 = x * x * x;
                let inner = sqrt_2_pi * (x + c * x3);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;

                // GELU = 0.5 * x * (1 + tanh(inner))
                // d/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d(inner)/dx
                // d(inner)/dx = sqrt(2/pi) * (1 + 3 * c * x^2)
                let d_inner = sqrt_2_pi * (1.0 + 3.0 * c * x * x);
                let grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;

                g * grad
            })
            .collect();

        vec![Some(
            Tensor::from_vec(result, self.saved_input.shape()).unwrap(),
        )]
    }

    fn name(&self) -> &'static str {
        "GeluBackward"
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
    fn test_relu_backward() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let grad_fn = ReluBackward::new(None, input);

        let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        // Gradient is 0 where input <= 0, 1 where input > 0
        assert_eq!(
            grads[0].as_ref().unwrap().to_vec(),
            vec![0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn test_sigmoid_backward() {
        // sigmoid(0) = 0.5, derivative at 0 is 0.5 * 0.5 = 0.25
        let output = Tensor::from_vec(vec![0.5], &[1]).unwrap();
        let grad_fn = SigmoidBackward::new(None, output);

        let grad_output = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        assert!((grads[0].as_ref().unwrap().to_vec()[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_backward() {
        // tanh(0) = 0, derivative at 0 is 1 - 0^2 = 1
        let output = Tensor::from_vec(vec![0.0], &[1]).unwrap();
        let grad_fn = TanhBackward::new(None, output);

        let grad_output = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        assert!((grads[0].as_ref().unwrap().to_vec()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_backward() {
        let input = Tensor::from_vec(vec![-1.0, 1.0], &[2]).unwrap();
        let grad_fn = LeakyReluBackward::new(None, input, 0.01);

        let grad_output = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let grads = grad_fn.apply(&grad_output);

        let result = grads[0].as_ref().unwrap().to_vec();
        assert!((result[0] - 0.01).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }
}
