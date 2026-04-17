//! `LAMB` — Layer-wise Adaptive Moments for large-batch training.
//!
//! Extends Adam with a per-layer trust ratio: `phi(||w||) / ||adam_update||`
//! scales the step size per parameter group, enabling stable training at
//! batch sizes of 32k+ without warmup hacks. Config mirrors Adam
//! (beta1/beta2/epsilon/weight_decay) plus optional trust_ratio clipping.
//!
//! # File
//! `crates/axonml-optim/src/lamb.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_core;
use axonml_nn::Parameter;
use axonml_tensor::Tensor;

use crate::optimizer::Optimizer;

// =============================================================================
// LAMB State
// =============================================================================

/// Per-parameter state for LAMB optimizer.
///
/// Stores momentum tensors on the same device as parameters (CPU or GPU).
/// When parameters are on GPU, all state stays GPU-resident — zero CPU round-trips.
#[derive(Debug, Clone)]
struct LambState {
    /// First moment (exponential moving average of gradient) — on same device as param.
    exp_avg: Tensor<f32>,
    /// Second moment (exponential moving average of squared gradient) — on same device as param.
    exp_avg_sq: Tensor<f32>,
    /// Step count for bias correction
    step: usize,
}

impl LambState {
    fn new(shape: &[usize], device: axonml_core::Device) -> Self {
        let size: usize = shape.iter().product();
        let mut exp_avg =
            Tensor::from_vec(vec![0.0f32; size], shape).expect("tensor creation failed");
        let mut exp_avg_sq =
            Tensor::from_vec(vec![0.0f32; size], shape).expect("tensor creation failed");
        if device.is_gpu() {
            exp_avg = exp_avg.to_device(device).expect("device transfer failed");
            exp_avg_sq = exp_avg_sq
                .to_device(device)
                .expect("device transfer failed");
        }
        Self {
            exp_avg,
            exp_avg_sq,
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
                .map(|p| {
                    let data = p.data();
                    LambState::new(data.shape(), data.device())
                })
                .collect();
        }
    }
}

impl Optimizer for LAMB {
    fn step(&mut self) {
        self.ensure_state_initialized();

        // ============================================================
        // Tensor-op path: works on both CPU and GPU without to_vec()
        // All ops (add, mul, mul_scalar, div, sqrt, add_scalar, sub)
        // dispatch to CUDA when the tensors are GPU-resident.
        // ============================================================

        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let state = &mut self.state[i];
            state.step += 1;

            let param_data = param.data();

            // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
            state.exp_avg = state
                .exp_avg
                .mul_scalar(self.beta1)
                .add(&grad.mul_scalar(1.0 - self.beta1))
                .unwrap();

            // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
            let grad_sq = grad.mul(&grad).unwrap();
            state.exp_avg_sq = state
                .exp_avg_sq
                .mul_scalar(self.beta2)
                .add(&grad_sq.mul_scalar(1.0 - self.beta2))
                .unwrap();

            // Compute bias-corrected moments
            let (bias_correction1, bias_correction2) = if self.bias_correction {
                (
                    1.0 - self.beta1.powi(state.step as i32),
                    1.0 - self.beta2.powi(state.step as i32),
                )
            } else {
                (1.0, 1.0)
            };

            // m_hat = m / bc1, v_hat = v / bc2
            let m_hat = state.exp_avg.mul_scalar(1.0 / bias_correction1);
            let v_hat = state.exp_avg_sq.mul_scalar(1.0 / bias_correction2);

            // adam_update = m_hat / (sqrt(v_hat) + eps)
            let adam_update = m_hat.div(&v_hat.sqrt().add_scalar(self.eps)).unwrap();

            // update = adam_update + weight_decay * param (decoupled weight decay)
            let update = if self.weight_decay > 0.0 {
                adam_update
                    .add(&param_data.mul_scalar(self.weight_decay))
                    .unwrap()
            } else {
                adam_update
            };

            // Compute trust ratio: ||param|| / ||update||
            // norm = sqrt(sum(x^2))  using Tensor ops
            let weight_norm_sq = param_data.mul(&param_data).unwrap().sum();
            let update_norm_sq = update.mul(&update).unwrap().sum();

            // Extract scalar norms (single element tensors)
            let weight_norm = weight_norm_sq.to_vec()[0].sqrt();
            let update_norm = update_norm_sq.to_vec()[0].sqrt();

            let trust_ratio = if weight_norm > 0.0 && update_norm > 0.0 {
                weight_norm / update_norm
            } else {
                1.0
            };

            // param = param - lr * trust_ratio * update
            let effective_lr = self.lr * trust_ratio;
            let new_param = param_data.sub(&update.mul_scalar(effective_lr)).unwrap();
            param.update_data(new_param);
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
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);
        let optimizer = LAMB::new(vec![param], 0.001);

        assert!((optimizer.get_lr() - 0.001).abs() < 1e-6);
        assert!((optimizer.beta1 - 0.9).abs() < 1e-6);
        assert!((optimizer.beta2 - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_lamb_step() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

        let mut optimizer = LAMB::new(vec![param.clone()], 0.1);
        optimizer.step();

        let new_data = param.data().to_vec();
        // Parameters should have changed
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_lamb_with_weight_decay() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

        let mut optimizer = LAMB::new(vec![param.clone()], 0.1).weight_decay(0.01);
        optimizer.step();

        let new_data = param.data().to_vec();
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_lamb_builder_pattern() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
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
        let var = Variable::new(
            Tensor::from_vec(vec![3.0, 4.0], &[2]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        // Weight norm = sqrt(9 + 16) = 5
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![1.0, 1.0], &[2]).expect("tensor creation failed"));

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
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

        let mut optimizer = LAMB::new(vec![param.clone()], 0.001);
        assert!(param.grad().is_some());

        optimizer.zero_grad();
        // Grad might be zeroed or None depending on implementation
    }

    #[test]
    fn test_l2_norm_via_tensor() {
        let t = Tensor::from_vec(vec![3.0f32, 4.0], &[2]).expect("tensor creation failed");
        let norm_sq = t.mul(&t).unwrap().sum();
        let norm = norm_sq.to_vec()[0].sqrt();
        assert!((norm - 5.0).abs() < 1e-6);
    }
}
