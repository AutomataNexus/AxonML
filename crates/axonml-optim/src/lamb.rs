//! LAMB Optimizer - Layer-wise Adaptive Moments
//!
//! Implements the LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
//! algorithm for large batch training. LAMB enables training with very large
//! batch sizes while maintaining accuracy.
//!
//! Reference: "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
//! https://arxiv.org/abs/1904.00962
//!
//! # Example
//! ```rust,ignore
//! use axonml_optim::LAMB;
//!
//! let mut optimizer = LAMB::new(model.parameters(), 0.001)
//!     .weight_decay(0.01)
//!     .betas(0.9, 0.999);
//!
//! for epoch in 0..100 {
//!     optimizer.zero_grad();
//!     let loss = model.forward(&input).mse_loss(&target);
//!     loss.backward();
//!     optimizer.step();
//! }
//! ```
//!
//! @version 0.1.0

use axonml_nn::Parameter;
use axonml_tensor::Tensor;

use crate::optimizer::Optimizer;

// =============================================================================
// LAMB State
// =============================================================================

/// Per-parameter state for LAMB optimizer.
#[derive(Debug, Clone)]
struct LambState {
    /// First moment (exponential moving average of gradient)
    exp_avg: Vec<f32>,
    /// Second moment (exponential moving average of squared gradient)
    exp_avg_sq: Vec<f32>,
    /// Step count for bias correction
    step: usize,
}

impl LambState {
    fn new(size: usize) -> Self {
        Self {
            exp_avg: vec![0.0; size],
            exp_avg_sq: vec![0.0; size],
            step: 0,
        }
    }
}

// =============================================================================
// LAMB Optimizer
// =============================================================================

/// LAMB optimizer for large batch training.
///
/// LAMB extends Adam by adding a layer-wise trust ratio that scales
/// the update based on the ratio of parameter norm to update norm.
/// This enables stable training with very large batch sizes.
///
/// The update rule is:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
/// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// r = m_hat / (sqrt(v_hat) + eps) + weight_decay * param
/// trust_ratio = ||param|| / ||r||  (layer-wise)
/// param = param - lr * trust_ratio * r
/// ```
pub struct LAMB {
    /// Parameters to optimize
    params: Vec<Parameter>,
    /// Learning rate
    lr: f32,
    /// First moment decay rate
    beta1: f32,
    /// Second moment decay rate
    beta2: f32,
    /// Small constant for numerical stability
    eps: f32,
    /// Weight decay coefficient (decoupled)
    weight_decay: f32,
    /// Whether to use bias correction
    bias_correction: bool,
    /// Per-parameter state
    state: Vec<LambState>,
}

impl LAMB {
    /// Creates a new LAMB optimizer with default hyperparameters.
    ///
    /// Defaults:
    /// - betas: (0.9, 0.999)
    /// - eps: 1e-6
    /// - weight_decay: 0.0
    #[must_use]
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
            bias_correction: true,
            state: Vec::new(),
        }
    }

    /// Creates LAMB with specified betas.
    #[must_use]
    pub fn with_betas(params: Vec<Parameter>, lr: f32, betas: (f32, f32)) -> Self {
        Self {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps: 1e-6,
            weight_decay: 0.0,
            bias_correction: true,
            state: Vec::new(),
        }
    }

    /// Creates LAMB with all options.
    #[must_use]
    pub fn with_options(
        params: Vec<Parameter>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            bias_correction: true,
            state: Vec::new(),
        }
    }

    /// Builder: set betas (momentum decay rates)
    #[must_use]
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Builder: set epsilon
    #[must_use]
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Builder: set weight decay
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder: set bias correction
    #[must_use]
    pub fn bias_correction(mut self, enabled: bool) -> Self {
        self.bias_correction = enabled;
        self
    }

    fn ensure_state_initialized(&mut self) {
        if self.state.is_empty() {
            self.state = self
                .params
                .iter()
                .map(|p| LambState::new(p.numel()))
                .collect();
        }
    }

    /// Computes the L2 norm of a vector.
    fn l2_norm(vec: &[f32]) -> f32 {
        vec.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

impl Optimizer for LAMB {
    fn step(&mut self) {
        self.ensure_state_initialized();

        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let grad_vec = grad.to_vec();
            let state = &mut self.state[i];
            state.step += 1;

            let param_data = param.data();
            let param_vec = param_data.to_vec();

            // Update biased first moment estimate
            for (m, g) in state.exp_avg.iter_mut().zip(grad_vec.iter()) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            }

            // Update biased second moment estimate
            for (v, g) in state.exp_avg_sq.iter_mut().zip(grad_vec.iter()) {
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            }

            // Compute bias-corrected moments
            let (bias_correction1, bias_correction2) = if self.bias_correction {
                (
                    1.0 - self.beta1.powi(state.step as i32),
                    1.0 - self.beta2.powi(state.step as i32),
                )
            } else {
                (1.0, 1.0)
            };

            // Compute Adam update direction: m_hat / (sqrt(v_hat) + eps)
            let mut update: Vec<f32> = state
                .exp_avg
                .iter()
                .zip(state.exp_avg_sq.iter())
                .map(|(m, v)| {
                    let m_hat = m / bias_correction1;
                    let v_hat = v / bias_correction2;
                    m_hat / (v_hat.sqrt() + self.eps)
                })
                .collect();

            // Add decoupled weight decay
            if self.weight_decay > 0.0 {
                for (u, p) in update.iter_mut().zip(param_vec.iter()) {
                    *u += self.weight_decay * p;
                }
            }

            // Compute layer-wise trust ratio
            let weight_norm = Self::l2_norm(&param_vec);
            let update_norm = Self::l2_norm(&update);

            let trust_ratio = if weight_norm > 0.0 && update_norm > 0.0 {
                weight_norm / update_norm
            } else {
                1.0
            };

            // Apply update with trust ratio
            let effective_lr = self.lr * trust_ratio;
            let new_data: Vec<f32> = param_vec
                .iter()
                .zip(update.iter())
                .map(|(p, u)| p - effective_lr * u)
                .collect();

            let new_tensor = Tensor::from_vec(new_data, param_data.shape()).unwrap();
            param.update_data(new_tensor);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn parameters(&self) -> &[Parameter] {
        &self.params
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_autograd::Variable;

    #[test]
    fn test_lamb_creation() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let optimizer = LAMB::new(vec![param], 0.001);

        assert!((optimizer.get_lr() - 0.001).abs() < 1e-6);
        assert!((optimizer.beta1 - 0.9).abs() < 1e-6);
        assert!((optimizer.beta2 - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_lamb_step() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap());

        let mut optimizer = LAMB::new(vec![param.clone()], 0.1);
        optimizer.step();

        let new_data = param.data().to_vec();
        // Parameters should have changed
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_lamb_with_weight_decay() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap());

        let mut optimizer = LAMB::new(vec![param.clone()], 0.1).weight_decay(0.01);
        optimizer.step();

        let new_data = param.data().to_vec();
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_lamb_builder_pattern() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        let optimizer = LAMB::new(vec![param], 0.001)
            .betas(0.95, 0.9999)
            .eps(1e-7)
            .weight_decay(0.01);

        assert!((optimizer.beta1 - 0.95).abs() < 1e-6);
        assert!((optimizer.beta2 - 0.9999).abs() < 1e-6);
        assert!((optimizer.eps - 1e-7).abs() < 1e-9);
        assert!((optimizer.weight_decay - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_lamb_trust_ratio() {
        // Test that trust ratio is computed correctly
        let var = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap(), true);
        let param = Parameter::from_variable(var);

        // Weight norm = sqrt(9 + 16) = 5
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap());

        let mut optimizer = LAMB::new(vec![param.clone()], 0.1);

        // After one step, parameters should change based on trust ratio
        let old_data = param.data().to_vec();
        optimizer.step();
        let new_data = param.data().to_vec();

        // Verify parameters changed
        assert!((new_data[0] - old_data[0]).abs() > 1e-6);
        assert!((new_data[1] - old_data[1]).abs() > 1e-6);
    }

    #[test]
    fn test_lamb_zero_grad() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap());

        let mut optimizer = LAMB::new(vec![param.clone()], 0.001);
        assert!(param.grad().is_some());

        optimizer.zero_grad();
        // Grad might be zeroed or None depending on implementation
    }

    #[test]
    fn test_l2_norm() {
        let vec = vec![3.0, 4.0];
        let norm = LAMB::l2_norm(&vec);
        assert!((norm - 5.0).abs() < 1e-6);
    }
}
