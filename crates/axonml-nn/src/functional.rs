//! Functional API - Stateless Neural Network Operations
//!
//! # File
//! `crates/axonml-nn/src/functional.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;

// =============================================================================
// Activation Functions
// =============================================================================

/// ReLU activation function.
pub fn relu(input: &Variable) -> Variable {
    input.relu()
}

/// Leaky ReLU activation function.
pub fn leaky_relu(input: &Variable, negative_slope: f32) -> Variable {
    input.leaky_relu(negative_slope)
}

/// Sigmoid activation function.
pub fn sigmoid(input: &Variable) -> Variable {
    input.sigmoid()
}

/// Tanh activation function.
pub fn tanh(input: &Variable) -> Variable {
    input.tanh()
}

/// GELU activation function.
pub fn gelu(input: &Variable) -> Variable {
    input.gelu()
}

/// SiLU (Swish) activation function.
pub fn silu(input: &Variable) -> Variable {
    let sigmoid = input.sigmoid();
    input.mul_var(&sigmoid)
}

/// ELU activation function.
pub fn elu(input: &Variable, alpha: f32) -> Variable {
    input.elu(alpha)
}

/// Softmax along a dimension.
pub fn softmax(input: &Variable, dim: i64) -> Variable {
    input.softmax(dim as i32)
}

/// Log softmax along a dimension.
pub fn log_softmax(input: &Variable, dim: i64) -> Variable {
    input.log_softmax(dim as i32)
}

// =============================================================================
// Linear Operations
// =============================================================================

/// Linear transformation: y = xA^T + b
pub fn linear(input: &Variable, weight: &Variable, bias: Option<&Variable>) -> Variable {
    let weight_t = weight.transpose(0, 1);
    let mut output = input.matmul(&weight_t);
    if let Some(b) = bias {
        output = output.add_var(b);
    }
    output
}

// =============================================================================
// Normalization
// =============================================================================

/// Layer normalization.
pub fn layer_norm(
    input: &Variable,
    normalized_shape: &[usize],
    _weight: Option<&Variable>,
    _bias: Option<&Variable>,
    eps: f32,
) -> Variable {
    // Delegate to LayerNorm module which has proper backward pass
    use crate::layers::LayerNorm;
    use crate::module::Module;
    let ln = LayerNorm::with_eps(normalized_shape.to_vec(), eps);
    ln.forward(input)
}

// =============================================================================
// Dropout
// =============================================================================

/// Dropout during training.
pub fn dropout(input: &Variable, p: f32, training: bool) -> Variable {
    if !training || p == 0.0 {
        return input.clone();
    }

    // Delegate to Dropout module which has proper backward pass
    use crate::layers::Dropout;
    use crate::module::Module;
    let d = Dropout::new(p);
    d.forward(input)
}

// =============================================================================
// Loss Functions
// =============================================================================

/// Mean squared error loss.
pub fn mse_loss(input: &Variable, target: &Variable) -> Variable {
    input.mse_loss(target)
}

/// Cross entropy loss.
///
/// Combines log_softmax and negative log-likelihood loss.
pub fn cross_entropy(input: &Variable, target: &Variable) -> Variable {
    use crate::loss::CrossEntropyLoss;
    CrossEntropyLoss::new().compute(input, target)
}

/// Binary cross entropy loss.
pub fn binary_cross_entropy(input: &Variable, target: &Variable) -> Variable {
    input.binary_cross_entropy(target)
}

// =============================================================================
// Pooling
// =============================================================================

/// Adaptive average pooling to output size.
pub fn adaptive_avg_pool2d(input: &Variable, output_size: (usize, usize)) -> Variable {
    use crate::layers::AdaptiveAvgPool2d;
    use crate::module::Module;
    let pool = AdaptiveAvgPool2d::new(output_size);
    pool.forward(input)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_relu_functional() {
        let input = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap(), false);
        let output = relu(&input);
        assert_eq!(output.data().to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_functional() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(),
            false,
        );
        let output = softmax(&input, -1);
        let sum: f32 = output.data().to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dropout_functional() {
        let input = Variable::new(Tensor::from_vec(vec![1.0; 100], &[100]).unwrap(), false);
        let output = dropout(&input, 0.5, true);
        let output_vec = output.data().to_vec();
        let num_zeros = output_vec.iter().filter(|&&x| x == 0.0).count();
        assert!(num_zeros > 30 && num_zeros < 70);
    }

    #[test]
    fn test_mse_loss_functional() {
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let loss = mse_loss(&input, &target);
        assert!((loss.data().to_vec()[0] - 0.0).abs() < 1e-6);
    }
}
