//! Fused Linear Operations
//!
//! Fused matrix multiplication with bias and activation.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_tensor::Tensor;
use rayon::prelude::*;

use crate::error::{FusionError, FusionResult};
use crate::FusedOp;

// =============================================================================
// Activation Type
// =============================================================================

/// Activation function for fused linear operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// No activation (identity).
    None,
    /// ReLU activation.
    Relu,
    /// GELU activation.
    Gelu,
    /// Sigmoid activation.
    Sigmoid,
    /// Tanh activation.
    Tanh,
    /// SiLU (Swish) activation.
    Silu,
}

impl Activation {
    /// Applies the activation to a value.
    #[inline(always)]
    fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::None => x,
            Activation::Relu => x.max(0.0),
            Activation::Gelu => {
                let sqrt_2_over_pi = 0.7978845608028654f32;
                let coeff = 0.044715f32;
                let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                x * 0.5 * (1.0 + inner.tanh())
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Silu => x * (1.0 / (1.0 + (-x).exp())), // x * sigmoid(x)
        }
    }
}

// =============================================================================
// Fused Linear Operation
// =============================================================================

/// Fused linear operation: output = activation(input @ weight^T + bias)
#[derive(Debug, Clone)]
pub struct FusedLinear {
    /// Weight matrix (out_features x in_features).
    weight: Tensor<f32>,
    /// Optional bias vector (out_features).
    bias: Option<Tensor<f32>>,
    /// Activation function.
    activation: Activation,
    /// Input features.
    in_features: usize,
    /// Output features.
    out_features: usize,
}

impl FusedLinear {
    /// Creates a new fused linear operation.
    pub fn new(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
        activation: Activation,
    ) -> FusionResult<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(FusionError::InvalidConfig(format!(
                "Weight must be 2D, got shape {:?}",
                shape
            )));
        }

        let out_features = shape[0];
        let in_features = shape[1];

        if let Some(ref b) = bias {
            if b.numel() != out_features {
                return Err(FusionError::ShapeMismatch {
                    expected: vec![out_features],
                    actual: b.shape().to_vec(),
                });
            }
        }

        Ok(Self {
            weight,
            bias,
            activation,
            in_features,
            out_features,
        })
    }

    /// Executes the fused linear operation.
    pub fn forward(&self, input: &Tensor<f32>) -> FusionResult<Tensor<f32>> {
        let input_shape = input.shape();

        // Handle batch dimension
        let (batch_size, input_features) = if input_shape.len() == 1 {
            (1, input_shape[0])
        } else if input_shape.len() == 2 {
            (input_shape[0], input_shape[1])
        } else {
            return Err(FusionError::InvalidConfig(format!(
                "Input must be 1D or 2D, got shape {:?}",
                input_shape
            )));
        };

        if input_features != self.in_features {
            return Err(FusionError::ShapeMismatch {
                expected: vec![self.in_features],
                actual: vec![input_features],
            });
        }

        let input_data = input.to_vec();
        let weight_data = self.weight.to_vec();
        let bias_data = self.bias.as_ref().map(|b| b.to_vec());

        // Fused matrix multiply + bias + activation
        let result: Vec<f32> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                let input_row = &input_data[b * self.in_features..(b + 1) * self.in_features];

                (0..self.out_features)
                    .map(|o| {
                        // Dot product
                        let weight_row =
                            &weight_data[o * self.in_features..(o + 1) * self.in_features];
                        let mut sum: f32 = input_row
                            .iter()
                            .zip(weight_row.iter())
                            .map(|(&a, &b)| a * b)
                            .sum();

                        // Add bias
                        if let Some(ref bias) = bias_data {
                            sum += bias[o];
                        }

                        // Apply activation
                        self.activation.apply(sum)
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        let output_shape = if input_shape.len() == 1 {
            vec![self.out_features]
        } else {
            vec![batch_size, self.out_features]
        };

        Tensor::from_vec(result, &output_shape)
            .map_err(|e| FusionError::TensorError(format!("{:?}", e)))
    }
}

impl FusedOp for FusedLinear {
    fn execute(&self, inputs: &[&Tensor<f32>]) -> FusionResult<Tensor<f32>> {
        let input = inputs
            .first()
            .ok_or_else(|| FusionError::InvalidConfig("No input provided".to_string()))?;
        self.forward(input)
    }

    fn name(&self) -> &str {
        "FusedLinear"
    }

    fn num_ops(&self) -> usize {
        let mut n = 1; // MatMul
        if self.bias.is_some() {
            n += 1; // Bias
        }
        if self.activation != Activation::None {
            n += 1; // Activation
        }
        n
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Creates a fused MatMul + Bias + ReLU operation.
pub fn fuse_matmul_bias_relu(
    weight: &Tensor<f32>,
    bias: &Tensor<f32>,
) -> FusionResult<FusedLinear> {
    FusedLinear::new(weight.clone(), Some(bias.clone()), Activation::Relu)
}

/// Creates a fused MatMul + Bias + GELU operation.
pub fn fuse_matmul_bias_gelu(
    weight: &Tensor<f32>,
    bias: &Tensor<f32>,
) -> FusionResult<FusedLinear> {
    FusedLinear::new(weight.clone(), Some(bias.clone()), Activation::Gelu)
}

/// Creates a fused MatMul + Bias operation (no activation).
pub fn fuse_matmul_bias(weight: &Tensor<f32>, bias: &Tensor<f32>) -> FusionResult<FusedLinear> {
    FusedLinear::new(weight.clone(), Some(bias.clone()), Activation::None)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_linear_no_activation() {
        // 2x3 weight matrix
        let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Bias
        let bias = Tensor::from_vec(vec![0.5, 0.5], &[2]).unwrap();

        let fused = FusedLinear::new(weight, Some(bias), Activation::None).unwrap();

        // Input: [1, 1, 1]
        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
        let output = fused.forward(&input).unwrap();

        // Expected: [1+2+3+0.5, 4+5+6+0.5] = [6.5, 15.5]
        let result = output.to_vec();
        assert!((result[0] - 6.5).abs() < 0.001);
        assert!((result[1] - 15.5).abs() < 0.001);
    }

    #[test]
    fn test_fused_linear_with_relu() {
        let weight = Tensor::from_vec(vec![1.0, -1.0, -1.0, 1.0], &[2, 2]).unwrap();

        let bias = Tensor::from_vec(vec![-0.5, -0.5], &[2]).unwrap();

        let fused = FusedLinear::new(weight, Some(bias), Activation::Relu).unwrap();

        let input = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let output = fused.forward(&input).unwrap();

        // [1-2-0.5, -1+2-0.5] = [-1.5, 0.5] -> relu -> [0, 0.5]
        let result = output.to_vec();
        assert!((result[0] - 0.0).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_fused_linear_batch() {
        let weight = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        let bias = Tensor::from_vec(vec![1.0], &[1]).unwrap();

        let fused = FusedLinear::new(weight, Some(bias), Activation::None).unwrap();

        // Batch of 2
        let input = Tensor::from_vec(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
        let output = fused.forward(&input).unwrap();

        // [1+2+1, 2+4+1] = [4, 7]
        let result = output.to_vec();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 4.0).abs() < 0.001);
        assert!((result[1] - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_fused_op_trait() {
        let weight = Tensor::from_vec(vec![1.0], &[1, 1]).unwrap();
        let bias = Tensor::from_vec(vec![0.0], &[1]).unwrap();

        let fused = FusedLinear::new(weight, Some(bias), Activation::Relu).unwrap();

        assert_eq!(fused.num_ops(), 3); // MatMul + Bias + ReLU
        assert_eq!(fused.name(), "FusedLinear");
    }

    #[test]
    fn test_activation_gelu() {
        // GELU(0) should be 0
        assert!((Activation::Gelu.apply(0.0) - 0.0).abs() < 0.001);
        // GELU(x) ≈ x for large positive x
        assert!((Activation::Gelu.apply(3.0) - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_convenience_functions() {
        let weight = Tensor::from_vec(vec![1.0], &[1, 1]).unwrap();
        let bias = Tensor::from_vec(vec![0.0], &[1]).unwrap();

        let _ = fuse_matmul_bias_relu(&weight, &bias).unwrap();
        let _ = fuse_matmul_bias_gelu(&weight, &bias).unwrap();
        let _ = fuse_matmul_bias(&weight, &bias).unwrap();
    }
}
