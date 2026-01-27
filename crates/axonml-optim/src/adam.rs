//! Adam Optimizer - Adaptive Moment Estimation
//!
//! Implements Adam and `AdamW` (Adam with decoupled weight decay).
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

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
#[derive(Debug, Clone)]
struct AdamState {
    /// First moment (mean of gradients).
    exp_avg: Vec<f32>,
    /// Second moment (variance of gradients).
    exp_avg_sq: Vec<f32>,
    /// Max second moment for `AMSGrad`.
    max_exp_avg_sq: Option<Vec<f32>>,
    /// Step count for bias correction.
    step: usize,
}

impl AdamState {
    fn new(size: usize, amsgrad: bool) -> Self {
        Self {
            exp_avg: vec![0.0; size],
            exp_avg_sq: vec![0.0; size],
            max_exp_avg_sq: if amsgrad { Some(vec![0.0; size]) } else { None },
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
                .map(|p| AdamState::new(p.numel(), self.amsgrad))
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

            let grad_vec = grad.to_vec();
            let state = &mut self.state[i];
            state.step += 1;

            let param_data = param.data();
            let mut param_vec = param_data.to_vec();

            // Apply L2 regularization to gradient (standard Adam weight decay)
            let grad_vec: Vec<f32> = if self.weight_decay == 0.0 {
                grad_vec
            } else {
                grad_vec
                    .iter()
                    .zip(param_vec.iter())
                    .map(|(g, p)| g + self.weight_decay * p)
                    .collect()
            };

            // Update biased first moment estimate
            for (m, g) in state.exp_avg.iter_mut().zip(grad_vec.iter()) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            }

            // Update biased second moment estimate
            for (v, g) in state.exp_avg_sq.iter_mut().zip(grad_vec.iter()) {
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            }

            // Bias correction
            let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);

            // Compute step size
            let step_size = self.lr / bias_correction1;

            // Update parameters
            if self.amsgrad {
                // AMSGrad variant
                let max_exp_avg_sq = state.max_exp_avg_sq.as_mut().unwrap();
                for (max_v, v) in max_exp_avg_sq.iter_mut().zip(state.exp_avg_sq.iter()) {
                    *max_v = max_v.max(*v);
                }
                for (p, (m, max_v)) in param_vec
                    .iter_mut()
                    .zip(state.exp_avg.iter().zip(max_exp_avg_sq.iter()))
                {
                    let denom = (max_v / bias_correction2).sqrt() + self.eps;
                    *p -= step_size * m / denom;
                }
            } else {
                // Standard Adam
                for (p, (m, v)) in param_vec
                    .iter_mut()
                    .zip(state.exp_avg.iter().zip(state.exp_avg_sq.iter()))
                {
                    let denom = (v / bias_correction2).sqrt() + self.eps;
                    *p -= step_size * m / denom;
                }
            }

            let update = Tensor::from_vec(param_vec, param_data.shape()).unwrap();
            param.update_data(update);
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
                .map(|p| AdamState::new(p.numel(), self.amsgrad))
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

            let grad_vec = grad.to_vec();
            let state = &mut self.state[i];
            state.step += 1;

            let param_data = param.data();
            let mut param_vec = param_data.to_vec();

            // Decoupled weight decay (applied directly to parameters)
            if self.weight_decay != 0.0 {
                for p in &mut param_vec {
                    *p *= 1.0 - self.lr * self.weight_decay;
                }
            }

            // Update biased first moment estimate
            for (m, g) in state.exp_avg.iter_mut().zip(grad_vec.iter()) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            }

            // Update biased second moment estimate
            for (v, g) in state.exp_avg_sq.iter_mut().zip(grad_vec.iter()) {
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            }

            // Bias correction
            let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);

            // Compute step size
            let step_size = self.lr / bias_correction1;

            // Update parameters
            if self.amsgrad {
                let max_exp_avg_sq = state.max_exp_avg_sq.as_mut().unwrap();
                for (max_v, v) in max_exp_avg_sq.iter_mut().zip(state.exp_avg_sq.iter()) {
                    *max_v = max_v.max(*v);
                }
                for (p, (m, max_v)) in param_vec
                    .iter_mut()
                    .zip(state.exp_avg.iter().zip(max_exp_avg_sq.iter()))
                {
                    let denom = (max_v / bias_correction2).sqrt() + self.eps;
                    *p -= step_size * m / denom;
                }
            } else {
                for (p, (m, v)) in param_vec
                    .iter_mut()
                    .zip(state.exp_avg.iter().zip(state.exp_avg_sq.iter()))
                {
                    let denom = (v / bias_correction2).sqrt() + self.eps;
                    *p -= step_size * m / denom;
                }
            }

            let update = Tensor::from_vec(param_vec, param_data.shape()).unwrap();
            param.update_data(update);
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
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let optimizer = Adam::new(vec![param], 0.001);

        assert!((optimizer.get_lr() - 0.001).abs() < 1e-6);
        assert!((optimizer.beta1 - 0.9).abs() < 1e-6);
        assert!((optimizer.beta2 - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_adam_step() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap());

        let mut optimizer = Adam::new(vec![param.clone()], 0.1);
        optimizer.step();

        let new_data = param.data().to_vec();
        // Parameters should have changed
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_adamw_creation() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let optimizer = AdamW::new(vec![param], 0.001);

        assert!((optimizer.weight_decay - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_adam_builder_pattern() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
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
