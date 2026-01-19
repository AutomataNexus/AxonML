//! Loss Functions - Training Objectives
//!
//! Provides loss functions for training neural networks.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::module::Module;

// =============================================================================
// Reduction Enum
// =============================================================================

/// Specifies how to reduce the loss over elements.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Reduction {
    /// No reduction - return loss per element.
    None,
    /// Mean of all losses.
    #[default]
    Mean,
    /// Sum of all losses.
    Sum,
}

// =============================================================================
// MSELoss
// =============================================================================

/// Mean Squared Error loss.
///
/// loss = mean((input - target)^2)
#[derive(Debug, Clone, Copy)]
pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    /// Creates a new MSELoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates MSELoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let diff = input.sub_var(target);
        let squared = diff.pow(2.0);

        match self.reduction {
            Reduction::None => squared,
            Reduction::Mean => squared.mean(),
            Reduction::Sum => squared.sum(),
        }
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MSELoss {
    fn forward(&self, input: &Variable) -> Variable {
        // For Module interface, we can't easily pass two inputs
        // This is primarily used via compute() method
        input.clone()
    }

    fn name(&self) -> &'static str {
        "MSELoss"
    }
}

// =============================================================================
// L1Loss
// =============================================================================

/// Mean Absolute Error loss.
///
/// loss = mean(|input - target|)
#[derive(Debug, Clone, Copy)]
pub struct L1Loss {
    reduction: Reduction,
}

impl L1Loss {
    /// Creates a new L1Loss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates L1Loss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let diff = input.sub_var(target);
        let diff_data = diff.data();
        let abs_data: Vec<f32> = diff_data.to_vec().iter().map(|x| x.abs()).collect();
        let abs_tensor = Tensor::from_vec(abs_data, diff_data.shape()).unwrap();
        let abs_var = Variable::new(abs_tensor, diff.requires_grad());

        match self.reduction {
            Reduction::None => abs_var,
            Reduction::Mean => abs_var.mean(),
            Reduction::Sum => abs_var.sum(),
        }
    }
}

impl Default for L1Loss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CrossEntropyLoss
// =============================================================================

/// Cross entropy loss with log softmax.
///
/// This combines LogSoftmax and NLLLoss in a single class.
///
/// # Shape
/// - Input: (N, C) where C = number of classes
/// - Target: (N,) with class indices
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
}

impl CrossEntropyLoss {
    /// Creates a new CrossEntropyLoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates CrossEntropyLoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    ///
    /// # Arguments
    /// * `input` - Logits of shape (N, C)
    /// * `target` - Class indices of shape (N,) as f32 (will be cast to usize)
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let num_classes = shape[1];

        let input_vec = input_data.to_vec();
        let target_vec = target_data.to_vec();

        let mut losses = vec![0.0f32; batch_size];

        for b in 0..batch_size {
            // Log softmax
            let offset = b * num_classes;
            let max_val = (0..num_classes)
                .map(|c| input_vec[offset + c])
                .fold(f32::NEG_INFINITY, f32::max);

            let mut log_sum_exp = 0.0f32;
            for c in 0..num_classes {
                log_sum_exp += (input_vec[offset + c] - max_val).exp();
            }
            log_sum_exp = max_val + log_sum_exp.ln();

            // NLL loss
            let target_class = target_vec[b] as usize;
            losses[b] = log_sum_exp - input_vec[offset + target_class];
        }

        let loss_tensor = Tensor::from_vec(losses.clone(), &[batch_size]).unwrap();
        let loss_var = Variable::new(loss_tensor, input.requires_grad());

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// NLLLoss
// =============================================================================

/// Negative Log Likelihood loss.
///
/// Expects input to be log-probabilities.
#[derive(Debug, Clone, Copy)]
pub struct NLLLoss {
    reduction: Reduction,
}

impl NLLLoss {
    /// Creates a new NLLLoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates NLLLoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let num_classes = shape[1];

        let input_vec = input_data.to_vec();
        let target_vec = target_data.to_vec();

        let mut losses = vec![0.0f32; batch_size];

        for b in 0..batch_size {
            let target_class = target_vec[b] as usize;
            losses[b] = -input_vec[b * num_classes + target_class];
        }

        let loss_tensor = Tensor::from_vec(losses, &[batch_size]).unwrap();
        let loss_var = Variable::new(loss_tensor, input.requires_grad());

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for NLLLoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BCELoss
// =============================================================================

/// Binary Cross Entropy loss.
///
/// Expects input to be probabilities in [0, 1].
#[derive(Debug, Clone, Copy)]
pub struct BCELoss {
    reduction: Reduction,
}

impl BCELoss {
    /// Creates a new BCELoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates BCELoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let eps = 1e-7f32;
        let input_data = input.data();
        let target_data = target.data();

        let input_vec = input_data.to_vec();
        let target_vec = target_data.to_vec();

        let losses: Vec<f32> = input_vec
            .iter()
            .zip(target_vec.iter())
            .map(|(&p, &t)| {
                let p_clamped = p.max(eps).min(1.0 - eps);
                -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
            })
            .collect();

        let loss_tensor = Tensor::from_vec(losses, input_data.shape()).unwrap();
        let loss_var = Variable::new(loss_tensor, input.requires_grad());

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for BCELoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BCEWithLogitsLoss
// =============================================================================

/// Binary Cross Entropy with Logits.
///
/// Combines sigmoid and BCE in a numerically stable way.
#[derive(Debug, Clone, Copy)]
pub struct BCEWithLogitsLoss {
    reduction: Reduction,
}

impl BCEWithLogitsLoss {
    /// Creates a new BCEWithLogitsLoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates BCEWithLogitsLoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();

        let input_vec = input_data.to_vec();
        let target_vec = target_data.to_vec();

        // Numerically stable: max(x, 0) - x*t + log(1 + exp(-|x|))
        let losses: Vec<f32> = input_vec
            .iter()
            .zip(target_vec.iter())
            .map(|(&x, &t)| {
                let max_val = x.max(0.0);
                max_val - x * t + (1.0 + (-x.abs()).exp()).ln()
            })
            .collect();

        let loss_tensor = Tensor::from_vec(losses, input_data.shape()).unwrap();
        let loss_var = Variable::new(loss_tensor, input.requires_grad());

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SmoothL1Loss
// =============================================================================

/// Smooth L1 loss (Huber loss).
///
/// Uses L2 loss when |x| < beta, L1 loss otherwise.
#[derive(Debug, Clone, Copy)]
pub struct SmoothL1Loss {
    reduction: Reduction,
    beta: f32,
}

impl SmoothL1Loss {
    /// Creates a new SmoothL1Loss with default beta (1.0).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            beta: 1.0,
        }
    }

    /// Creates SmoothL1Loss with specified beta.
    pub fn with_beta(beta: f32) -> Self {
        Self {
            reduction: Reduction::Mean,
            beta,
        }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let diff = input.sub_var(target);
        let diff_data = diff.data();
        let diff_vec = diff_data.to_vec();

        let losses: Vec<f32> = diff_vec
            .iter()
            .map(|&d| {
                let abs_d = d.abs();
                if abs_d < self.beta {
                    0.5 * d * d / self.beta
                } else {
                    abs_d - 0.5 * self.beta
                }
            })
            .collect();

        let loss_tensor = Tensor::from_vec(losses, diff_data.shape()).unwrap();
        let loss_var = Variable::new(loss_tensor, diff.requires_grad());

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss::new();
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        assert!((loss.data().to_vec()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let loss_fn = MSELoss::new();
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        // Each diff is 1.0, squared is 1.0, mean is 1.0
        assert!((loss.data().to_vec()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap(),
            false,
        );
        let target = Variable::new(Tensor::from_vec(vec![2.0, 0.0], &[2]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        assert!(loss.data().to_vec()[0] > 0.0);
    }

    #[test]
    fn test_bce_loss() {
        let loss_fn = BCELoss::new();
        let input = Variable::new(Tensor::from_vec(vec![0.5, 0.5], &[2]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        // -[1*ln(0.5) + 0*ln(0.5)] - [0*ln(0.5) + 1*ln(0.5)] = -2*ln(0.5) / 2 = -ln(0.5) = 0.693
        assert!((loss.data().to_vec()[0] - 0.693).abs() < 0.01);
    }
}
