//! Linear Layer - Fully Connected Layer
//!
//! Applies a linear transformation: y = xW^T + b
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::init::{kaiming_uniform, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// Linear
// =============================================================================

/// Applies a linear transformation to the input.
///
/// y = xW^T + b
///
/// # Arguments
/// * `in_features` - Size of each input sample
/// * `out_features` - Size of each output sample
/// * `bias` - If true, adds a learnable bias (default: true)
///
/// # Shape
/// - Input: (*, in_features) where * means any number of dimensions
/// - Output: (*, out_features)
///
/// # Example
/// ```ignore
/// let linear = Linear::new(20, 30);
/// let input = Variable::new(randn(&[128, 20]), true);
/// let output = linear.forward(&input);  // Shape: [128, 30]
/// ```
pub struct Linear {
    /// Weight matrix of shape (out_features, in_features).
    pub weight: Parameter,
    /// Bias vector of shape (out_features).
    pub bias: Option<Parameter>,
    /// Input features.
    in_features: usize,
    /// Output features.
    out_features: usize,
}

impl Linear {
    /// Creates a new Linear layer with bias.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, true)
    }

    /// Creates a new Linear layer with optional bias.
    pub fn with_bias(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Initialize weights using Kaiming uniform
        let weight_data = kaiming_uniform(out_features, in_features);
        let weight = Parameter::named("weight", weight_data, true);

        let bias_param = if bias {
            // Initialize bias to zeros
            let bias_data = zeros(&[out_features]);
            Some(Parameter::named("bias", bias_data, true))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
        }
    }

    /// Creates a Linear layer from existing weight and bias tensors.
    pub fn from_weights(weight: Tensor<f32>, bias: Option<Tensor<f32>>) -> Self {
        let out_features = weight.shape()[0];
        let in_features = weight.shape()[1];

        Self {
            weight: Parameter::named("weight", weight, true),
            bias: bias.map(|b| Parameter::named("bias", b, true)),
            in_features,
            out_features,
        }
    }

    /// Returns the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Variable {
        // Get input shape
        let input_shape = input.shape();
        let batch_dims: Vec<usize> = input_shape[..input_shape.len() - 1].to_vec();

        // Reshape to 2D: (batch, in_features)
        let total_batch: usize = batch_dims.iter().product();
        let input_2d = if input_shape.len() > 2 {
            // Use autograd-tracked reshape to maintain gradient flow
            input.reshape(&[total_batch, self.in_features])
        } else {
            input.clone()
        };

        // y = xW^T
        // x: (batch, in_features), W: (out_features, in_features)
        // We need x @ W^T = (batch, out_features)
        let weight_var = self.weight.variable();
        // Use autograd-tracked transpose to maintain gradient flow
        let weight_t = weight_var.transpose(0, 1);
        let mut output = input_2d.matmul(&weight_t);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_var = bias.variable();
            output = output.add_var(&bias_var);
        }

        // Reshape back to original batch dimensions
        if batch_dims.len() > 1 || (batch_dims.len() == 1 && input_shape.len() > 2) {
            let mut output_shape: Vec<usize> = batch_dims.clone();
            output_shape.push(self.out_features);
            output.reshape(&output_shape)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "Linear"
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(10, 5);
        assert_eq!(linear.in_features(), 10);
        assert_eq!(linear.out_features(), 5);
        assert!(linear.bias.is_some());
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::with_bias(10, 5, false);
        assert!(linear.bias.is_none());
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(3, 2);

        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(),
            false,
        );
        let output = linear.forward(&input);

        assert_eq!(output.shape(), vec![1, 2]);
    }

    #[test]
    fn test_linear_batch_forward() {
        let linear = Linear::new(4, 2);

        let input = Variable::new(Tensor::from_vec(vec![1.0; 12], &[3, 4]).unwrap(), false);
        let output = linear.forward(&input);

        assert_eq!(output.shape(), vec![3, 2]);
    }

    #[test]
    fn test_linear_parameters() {
        let linear = Linear::new(10, 5);
        let params = linear.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let linear_no_bias = Linear::with_bias(10, 5, false);
        let params_no_bias = linear_no_bias.parameters();
        assert_eq!(params_no_bias.len(), 1); // weight only
    }

    #[test]
    fn test_linear_num_parameters() {
        let linear = Linear::new(10, 5);
        // weight: 10*5 = 50, bias: 5, total: 55
        assert_eq!(linear.num_parameters(), 55);
    }
}
