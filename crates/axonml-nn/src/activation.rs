//! Activation Modules - Non-linear Activation Functions
//!
//! Provides activation functions as modules for use in Sequential and other containers.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::module::Module;

// =============================================================================
// ReLU
// =============================================================================

/// Applies the rectified linear unit function element-wise.
///
/// ReLU(x) = max(0, x)
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;

impl ReLU {
    /// Creates a new ReLU activation.
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Variable {
        input.relu()
    }

    fn name(&self) -> &'static str {
        "ReLU"
    }
}

// =============================================================================
// LeakyReLU
// =============================================================================

/// Applies the leaky ReLU function element-wise.
///
/// LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    negative_slope: f32,
}

impl LeakyReLU {
    /// Creates a new LeakyReLU with default negative slope (0.01).
    pub fn new() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }

    /// Creates a LeakyReLU with custom negative slope.
    pub fn with_slope(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Variable) -> Variable {
        let data = input.data();
        let result: Vec<f32> = data
            .to_vec()
            .iter()
            .map(|&x| if x > 0.0 { x } else { x * self.negative_slope })
            .collect();
        Variable::new(
            Tensor::from_vec(result, data.shape()).unwrap(),
            input.requires_grad(),
        )
    }

    fn name(&self) -> &'static str {
        "LeakyReLU"
    }
}

// =============================================================================
// Sigmoid
// =============================================================================

/// Applies the sigmoid function element-wise.
///
/// Sigmoid(x) = 1 / (1 + exp(-x))
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;

impl Sigmoid {
    /// Creates a new Sigmoid activation.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Variable {
        input.sigmoid()
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

// =============================================================================
// Tanh
// =============================================================================

/// Applies the hyperbolic tangent function element-wise.
///
/// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

impl Tanh {
    /// Creates a new Tanh activation.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Variable {
        input.tanh()
    }

    fn name(&self) -> &'static str {
        "Tanh"
    }
}

// =============================================================================
// Softmax
// =============================================================================

/// Applies the softmax function along a dimension.
///
/// Softmax(x_i) = exp(x_i) / sum(exp(x_j))
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    dim: i64,
}

impl Softmax {
    /// Creates a new Softmax along the specified dimension.
    pub fn new(dim: i64) -> Self {
        Self { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Variable) -> Variable {
        // Simple implementation for last dimension
        let data = input.data();
        let shape = data.shape().to_vec();
        let data_vec = data.to_vec();

        let ndim = shape.len();
        let dim = if self.dim < 0 {
            (ndim as i64 + self.dim) as usize
        } else {
            self.dim as usize
        };

        let outer_size: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();

        let mut result = vec![0.0f32; data_vec.len()];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    max_val = max_val.max(data_vec[idx]);
                }

                // Compute exp and sum
                let mut sum = 0.0f32;
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    let exp_val = (data_vec[idx] - max_val).exp();
                    result[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    result[idx] /= sum;
                }
            }
        }

        Variable::new(
            Tensor::from_vec(result, &shape).unwrap(),
            input.requires_grad(),
        )
    }

    fn name(&self) -> &'static str {
        "Softmax"
    }
}

// =============================================================================
// LogSoftmax
// =============================================================================

/// Applies log(softmax(x)) along a dimension.
#[derive(Debug, Clone, Copy)]
pub struct LogSoftmax {
    dim: i64,
}

impl LogSoftmax {
    /// Creates a new LogSoftmax along the specified dimension.
    pub fn new(dim: i64) -> Self {
        Self { dim }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Variable) -> Variable {
        let softmax = Softmax::new(self.dim);
        let sm = softmax.forward(input);
        let sm_vec = sm.data().to_vec();
        let result: Vec<f32> = sm_vec.iter().map(|&x| x.ln()).collect();
        Variable::new(
            Tensor::from_vec(result, sm.data().shape()).unwrap(),
            input.requires_grad(),
        )
    }

    fn name(&self) -> &'static str {
        "LogSoftmax"
    }
}

// =============================================================================
// GELU
// =============================================================================

/// Applies the Gaussian Error Linear Unit function.
///
/// GELU(x) = x * Phi(x) where Phi is the CDF of standard normal distribution.
#[derive(Debug, Clone, Copy, Default)]
pub struct GELU;

impl GELU {
    /// Creates a new GELU activation.
    pub fn new() -> Self {
        Self
    }
}

impl Module for GELU {
    fn forward(&self, input: &Variable) -> Variable {
        let data = input.data();
        let data_vec = data.to_vec();
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let result: Vec<f32> = data_vec
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Variable::new(
            Tensor::from_vec(result, data.shape()).unwrap(),
            input.requires_grad(),
        )
    }

    fn name(&self) -> &'static str {
        "GELU"
    }
}

// =============================================================================
// SiLU / Swish
// =============================================================================

/// Applies the SiLU (Swish) function element-wise.
///
/// SiLU(x) = x * sigmoid(x)
#[derive(Debug, Clone, Copy, Default)]
pub struct SiLU;

impl SiLU {
    /// Creates a new SiLU activation.
    pub fn new() -> Self {
        Self
    }
}

impl Module for SiLU {
    fn forward(&self, input: &Variable) -> Variable {
        let sigmoid = input.sigmoid();
        input.mul_var(&sigmoid)
    }

    fn name(&self) -> &'static str {
        "SiLU"
    }
}

// =============================================================================
// ELU
// =============================================================================

/// Applies the Exponential Linear Unit function.
///
/// ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
#[derive(Debug, Clone, Copy)]
pub struct ELU {
    alpha: f32,
}

impl ELU {
    /// Creates a new ELU with default alpha (1.0).
    pub fn new() -> Self {
        Self { alpha: 1.0 }
    }

    /// Creates an ELU with custom alpha.
    pub fn with_alpha(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ELU {
    fn forward(&self, input: &Variable) -> Variable {
        let data = input.data();
        let result: Vec<f32> = data
            .to_vec()
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    x
                } else {
                    self.alpha * (x.exp() - 1.0)
                }
            })
            .collect();
        Variable::new(
            Tensor::from_vec(result, data.shape()).unwrap(),
            input.requires_grad(),
        )
    }

    fn name(&self) -> &'static str {
        "ELU"
    }
}

// =============================================================================
// Identity
// =============================================================================

/// Identity activation (no-op).
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

impl Identity {
    /// Creates a new Identity activation.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, input: &Variable) -> Variable {
        input.clone()
    }

    fn name(&self) -> &'static str {
        "Identity"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let relu = ReLU::new();
        let input = Variable::new(
            Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap(),
            false,
        );
        let output = relu.forward(&input);
        assert_eq!(output.data().to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid::new();
        let input = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        let output = sigmoid.forward(&input);
        assert!((output.data().to_vec()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let softmax = Softmax::new(-1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(),
            false,
        );
        let output = softmax.forward(&input);
        let sum: f32 = output.data().to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_leaky_relu() {
        let leaky = LeakyReLU::with_slope(0.1);
        let input = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap(), false);
        let output = leaky.forward(&input);
        assert_eq!(output.data().to_vec(), vec![-0.1, 0.0, 1.0]);
    }

    #[test]
    fn test_identity() {
        let id = Identity::new();
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let output = id.forward(&input);
        assert_eq!(output.data().to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
