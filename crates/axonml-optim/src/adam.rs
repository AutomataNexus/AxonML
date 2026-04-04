//! Adam Optimizer - Adaptive Moment Estimation
//!
//! # File
//! `crates/axonml-optim/src/adam.rs`
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

use axonml_nn::Parameter;
use axonml_tensor::Tensor;

use crate::optimizer::Optimizer;

// =============================================================================
// Adam
// =============================================================================

/// Adam optimizer.
///
/// Maintains per-parameter adaptive learning rates using first and
/// second moment estimates of gradients.
///
/// Update rule:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
/// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// param = param - lr * m_hat / (sqrt(v_hat) + eps)
/// ```
pub struct Adam {
    /// Parameters to optimize.
    params: Vec<Parameter>,
    /// Learning rate.
    lr: f32,
    /// First moment decay rate.
    beta1: f32,
    /// Second moment decay rate.
    beta2: f32,
    /// Small constant for numerical stability.
    eps: f32,
    /// Weight decay (L2 regularization for standard Adam).
    weight_decay: f32,
    /// Whether to use `AMSGrad` variant.
    amsgrad: bool,
    /// Per-parameter state.
    state: Vec<AdamState>,
}

/// State for Adam optimizer.
///
/// Stores momentum tensors on the same device as parameters (CPU or GPU).
/// When parameters are on GPU, all state stays on GPU — zero CPU round-trips.
#[derive(Debug, Clone)]
struct AdamState {
    /// First moment (mean of gradients) — on same device as param.
    exp_avg: Tensor<f32>,
    /// Second moment (variance of gradients) — on same device as param.
    exp_avg_sq: Tensor<f32>,
    /// Maximum of all past exp_avg_sq values (for AMSGrad).
    max_exp_avg_sq: Option<Tensor<f32>>,
    /// Step count for bias correction.
    step: usize,
}

impl AdamState {
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
            max_exp_avg_sq: None, // Initialized on first use if amsgrad=true
            step: 0,
        }
    }
}

impl Adam {
    /// Creates a new Adam optimizer with default hyperparameters.
    #[must_use]
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        Self::with_betas(params, lr, (0.9, 0.999))
    }

    /// Creates Adam with specified betas.
    #[must_use]
    pub fn with_betas(params: Vec<Parameter>, lr: f32, betas: (f32, f32)) -> Self {
        Self {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            state: Vec::new(),
        }
    }

    /// Creates Adam with all options.
    #[must_use]
    pub fn with_options(
        params: Vec<Parameter>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        Self {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            amsgrad,
            state: Vec::new(),
        }
    }

    /// Builder method to set betas.
    #[must_use]
    pub fn betas(mut self, betas: (f32, f32)) -> Self {
        self.beta1 = betas.0;
        self.beta2 = betas.1;
        self
    }

    /// Builder method to set epsilon.
    #[must_use]
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Builder method to set weight decay.
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to enable `AMSGrad`.
    #[must_use]
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    fn ensure_state_initialized(&mut self) {
        if self.state.is_empty() {
            self.state = self
                .params
                .iter()
                .map(|p| {
                    let data = p.data();
                    AdamState::new(data.shape(), data.device())
                })
                .collect();
        }
    }
}

impl Optimizer for Adam {
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

            let state = &mut self.state[i];
            state.step += 1;

            let param_data = param.data();

            // GPU path: fused CUDA kernel — single launch per parameter, zero CPU copies
            #[cfg(feature = "cuda")]
            if param_data.device().is_gpu() {
                // Auto-migrate gradient to GPU if backward produced CPU gradients
                // (happens when backward functions use CPU fallback computation)
                let grad = if !grad.device().is_gpu() {
                    grad.to_device(param_data.device())
                        .expect("Adam: failed to migrate CPU gradient to GPU")
                } else {
                    grad
                };
                let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);

                // In-place fused Adam update on GPU
                param_data.adam_step_inplace(
                    &grad,
                    &state.exp_avg,
                    &state.exp_avg_sq,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    bias_correction1,
                    bias_correction2,
                );
                // No need for update_data — the kernel modified the GPU buffer in-place
                continue;
            }

            // CPU fallback — fused single-loop update for cache locality
            let grad_vec = grad.to_vec();
            let mut param_vec = param_data.to_vec();
            let mut exp_avg_vec = state.exp_avg.to_vec();
            let mut exp_avg_sq_vec = state.exp_avg_sq.to_vec();

            let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);
            let step_size = self.lr / bias_correction1;
            let beta1 = self.beta1;
            let beta2 = self.beta2;
            let one_minus_beta1 = 1.0 - beta1;
            let one_minus_beta2 = 1.0 - beta2;
            let eps = self.eps;
            let wd = self.weight_decay;

            // AMSGrad: track max of all past exp_avg_sq values
            let mut max_sq_vec = if self.amsgrad {
                state
                    .max_exp_avg_sq
                    .as_ref()
                    .map_or_else(|| vec![0.0f32; param_vec.len()], |t| t.to_vec())
            } else {
                Vec::new()
            };

            for i in 0..param_vec.len() {
                let g = if wd == 0.0 {
                    grad_vec[i]
                } else {
                    grad_vec[i] + wd * param_vec[i]
                };
                exp_avg_vec[i] = beta1 * exp_avg_vec[i] + one_minus_beta1 * g;
                exp_avg_sq_vec[i] = beta2 * exp_avg_sq_vec[i] + one_minus_beta2 * g * g;

                let v_hat = if self.amsgrad {
                    max_sq_vec[i] = max_sq_vec[i].max(exp_avg_sq_vec[i]);
                    max_sq_vec[i] / bias_correction2
                } else {
                    exp_avg_sq_vec[i] / bias_correction2
                };

                let denom = v_hat.sqrt() + eps;
                param_vec[i] -= step_size * exp_avg_vec[i] / denom;
            }

            state.exp_avg =
                Tensor::from_vec(exp_avg_vec, param_data.shape()).expect("tensor creation failed");
            state.exp_avg_sq = Tensor::from_vec(exp_avg_sq_vec, param_data.shape())
                .expect("tensor creation failed");
            if self.amsgrad {
                state.max_exp_avg_sq = Some(
                    Tensor::from_vec(max_sq_vec, param_data.shape())
                        .expect("tensor creation failed"),
                );
            }
            param.update_data(
                Tensor::from_vec(param_vec, param_data.shape()).expect("tensor creation failed"),
            );
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
// AdamW
// =============================================================================

/// `AdamW` optimizer (Adam with decoupled weight decay).
///
/// Unlike standard Adam which applies L2 regularization to the gradient,
/// `AdamW` applies weight decay directly to the parameters.
///
/// Update rule:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
/// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
/// ```
pub struct AdamW {
    /// Parameters to optimize.
    params: Vec<Parameter>,
    /// Learning rate.
    lr: f32,
    /// First moment decay rate.
    beta1: f32,
    /// Second moment decay rate.
    beta2: f32,
    /// Small constant for numerical stability.
    eps: f32,
    /// Decoupled weight decay coefficient.
    weight_decay: f32,
    /// Whether to use `AMSGrad` variant.
    amsgrad: bool,
    /// Per-parameter state.
    state: Vec<AdamState>,
}

impl AdamW {
    /// Creates a new `AdamW` optimizer with default hyperparameters.
    #[must_use]
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        Self::with_betas(params, lr, (0.9, 0.999))
    }

    /// Creates `AdamW` with specified betas.
    #[must_use]
    pub fn with_betas(params: Vec<Parameter>, lr: f32, betas: (f32, f32)) -> Self {
        Self {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps: 1e-8,
            weight_decay: 0.01, // Default weight decay for AdamW
            amsgrad: false,
            state: Vec::new(),
        }
    }

    /// Creates `AdamW` with all options.
    #[must_use]
    pub fn with_options(
        params: Vec<Parameter>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        Self {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            amsgrad,
            state: Vec::new(),
        }
    }

    /// Builder method to set betas.
    #[must_use]
    pub fn betas(mut self, betas: (f32, f32)) -> Self {
        self.beta1 = betas.0;
        self.beta2 = betas.1;
        self
    }

    /// Builder method to set epsilon.
    #[must_use]
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Builder method to set weight decay.
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to enable `AMSGrad`.
    #[must_use]
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    fn ensure_state_initialized(&mut self) {
        if self.state.is_empty() {
            self.state = self
                .params
                .iter()
                .map(|p| {
                    let data = p.data();
                    AdamState::new(data.shape(), data.device())
                })
                .collect();
        }
    }
}

impl Optimizer for AdamW {
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

            let state = &mut self.state[i];
            state.step += 1;

            let param_data = param.data();

            // GPU path: decoupled weight decay + fused Adam step
            #[cfg(feature = "cuda")]
            if param_data.device().is_gpu() {
                // Auto-migrate gradient to GPU if backward produced CPU gradients
                let grad = if !grad.device().is_gpu() {
                    grad.to_device(param_data.device())
                        .expect("AdamW: failed to migrate CPU gradient to GPU")
                } else {
                    grad
                };

                // DECOUPLED weight decay: param *= (1 - lr * wd)
                // This is the key difference from Adam's L2 regularization.
                // Applied BEFORE the Adam update, directly to parameters.
                if self.weight_decay > 0.0 {
                    let decay_factor = 1.0 - self.lr * self.weight_decay;
                    let decayed = param_data.mul_scalar(decay_factor);
                    param.update_data(decayed);
                }

                // Re-read param_data after potential decay update
                let param_data = param.data();

                let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);

                // Adam step with wd=0 (decay already applied above)
                param_data.adam_step_inplace(
                    &grad,
                    &state.exp_avg,
                    &state.exp_avg_sq,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    0.0, // wd=0: decoupled decay already applied
                    bias_correction1,
                    bias_correction2,
                );
                continue;
            }

            // CPU fallback — fused single-loop update for cache locality
            let grad_vec = grad.to_vec();
            let mut param_vec = param_data.to_vec();
            let mut exp_avg_vec = state.exp_avg.to_vec();
            let mut exp_avg_sq_vec = state.exp_avg_sq.to_vec();

            let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);
            let step_size = self.lr / bias_correction1;
            let beta1 = self.beta1;
            let beta2 = self.beta2;
            let one_minus_beta1 = 1.0 - beta1;
            let one_minus_beta2 = 1.0 - beta2;
            let eps = self.eps;
            let wd_factor = 1.0 - self.lr * self.weight_decay;
            let has_wd = self.weight_decay != 0.0;

            for i in 0..param_vec.len() {
                // Decoupled weight decay: apply directly to param
                if has_wd {
                    param_vec[i] *= wd_factor;
                }
                let g = grad_vec[i];
                exp_avg_vec[i] = beta1 * exp_avg_vec[i] + one_minus_beta1 * g;
                exp_avg_sq_vec[i] = beta2 * exp_avg_sq_vec[i] + one_minus_beta2 * g * g;
                let denom = (exp_avg_sq_vec[i] / bias_correction2).sqrt() + eps;
                param_vec[i] -= step_size * exp_avg_vec[i] / denom;
            }

            state.exp_avg = Tensor::from_vec(exp_avg_vec, param_data.shape()).unwrap();
            state.exp_avg_sq = Tensor::from_vec(exp_avg_sq_vec, param_data.shape()).unwrap();
            param.update_data(Tensor::from_vec(param_vec, param_data.shape()).unwrap());
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
    fn test_adam_creation() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);
        let optimizer = Adam::new(vec![param], 0.001);

        assert!((optimizer.get_lr() - 0.001).abs() < 1e-6);
        assert!((optimizer.beta1 - 0.9).abs() < 1e-6);
        assert!((optimizer.beta2 - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_adam_step() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

        let mut optimizer = Adam::new(vec![param.clone()], 0.1);
        optimizer.step();

        let new_data = param.data().to_vec();
        // Parameters should have changed
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_adamw_creation() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);
        let optimizer = AdamW::new(vec![param], 0.001);

        assert!((optimizer.weight_decay - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_adam_builder_pattern() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        let optimizer = Adam::new(vec![param], 0.001)
            .betas((0.95, 0.9999))
            .eps(1e-7)
            .weight_decay(0.01)
            .amsgrad(true);

        assert!((optimizer.beta1 - 0.95).abs() < 1e-6);
        assert!((optimizer.beta2 - 0.9999).abs() < 1e-6);
        assert!((optimizer.eps - 1e-7).abs() < 1e-9);
        assert!(optimizer.amsgrad);
    }
}
