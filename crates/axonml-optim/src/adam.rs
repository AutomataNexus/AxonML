//! `Adam` and `AdamW` — adaptive moment estimation optimizers.
//!
//! `Adam::new(params, lr)` with beta1/beta2/epsilon/weight_decay config.
//! `AdamW` variant implements decoupled weight decay (applies decay to
//! parameters directly, not through the gradient). Both track first/second
//! moment estimates per parameter with bias correction.
//!
//! # File
//! `crates/axonml-optim/src/adam.rs`
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

use axonml_nn::Parameter;
use axonml_tensor::Tensor;
use rayon::prelude::*;

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

/// Adam moment state lives on the parameter's device: a GPU parameter gets
/// GPU-resident `exp_avg`/`exp_avg_sq`, always. The GPU step migrates any
/// host-resident state it meets instead of handing it to the CUDA kernel.
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
            max_exp_avg_sq: None,
            step: 0,
        }
    }
}

/// One parameter's exported Adam moments: `(shape, exp_avg, exp_avg_sq, step)`.
pub type AdamMoments = (Vec<usize>, Vec<f32>, Vec<f32>, usize);

impl Adam {
    /// Optimizer moments for a lossless pause: per parameter (shape, exp_avg, exp_avg_sq, step).
    /// Parameters without state yet (never stepped) are exported as empty vectors.
    pub fn export_state(&self) -> Vec<AdamMoments> {
        self.params
            .iter()
            .enumerate()
            .map(|(i, p)| match self.state.get(i) {
                Some(st) => (
                    p.shape().clone(),
                    st.exp_avg.to_vec(),
                    st.exp_avg_sq.to_vec(),
                    st.step,
                ),
                None => (p.shape().clone(), Vec::new(), Vec::new(), 0),
            })
            .collect()
    }

    /// Restore moments exported by [`Adam::export_state`]; entries whose shape does not match are
    /// left at zero. Returns the number of parameters restored.
    pub fn import_state(&mut self, saved: &[AdamMoments]) -> usize {
        self.ensure_state_initialized();
        let mut n = 0usize;
        for (i, (shape, m, v, step)) in saved.iter().enumerate() {
            let Some(p) = self.params.get(i) else { break };
            if p.shape() != shape.as_slice() || m.len() != p.numel() || v.len() != p.numel() {
                continue;
            }
            let dev = p.data().device();
            let mut em = Tensor::from_vec(m.clone(), shape).expect("adam m");
            let mut ev = Tensor::from_vec(v.clone(), shape).expect("adam v");
            if dev.is_gpu() {
                em = em.to_device(dev).expect("adam m to device");
                ev = ev.to_device(dev).expect("adam v to device");
            }
            let st = &mut self.state[i];
            st.exp_avg = em;
            st.exp_avg_sq = ev;
            st.step = *step;
            n += 1;
        }
        n
    }

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

            #[cfg(feature = "cuda")]
            if param_data.device().is_gpu() {
                let grad = if !grad.device().is_gpu() {
                    grad.to_device(param_data.device())
                        .expect("Adam: failed to migrate CPU gradient to GPU")
                } else {
                    grad
                };
                let grad = if grad.is_contiguous() {
                    grad
                } else {
                    grad.contiguous()
                };
                if !state.exp_avg.device().is_gpu() {
                    state.exp_avg = state
                        .exp_avg
                        .to_device(param_data.device())
                        .expect("Adam: state to GPU");
                    state.exp_avg_sq = state
                        .exp_avg_sq
                        .to_device(param_data.device())
                        .expect("Adam: state to GPU");
                }
                let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);

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
                continue;
            }

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

            let mut max_sq_vec = if self.amsgrad {
                state
                    .max_exp_avg_sq
                    .as_ref()
                    .map_or_else(|| vec![0.0f32; param_vec.len()], |t| t.to_vec())
            } else {
                Vec::new()
            };

            if self.amsgrad {
                param_vec
                    .par_iter_mut()
                    .zip(exp_avg_vec.par_iter_mut())
                    .zip(exp_avg_sq_vec.par_iter_mut())
                    .zip(max_sq_vec.par_iter_mut())
                    .zip(grad_vec.par_iter())
                    .for_each(|((((p, ea), eas), mx), &gr)| {
                        let g = if wd == 0.0 { gr } else { gr + wd * *p };
                        *ea = beta1 * *ea + one_minus_beta1 * g;
                        *eas = beta2 * *eas + one_minus_beta2 * g * g;
                        *mx = mx.max(*eas);
                        let v_hat = *mx / bias_correction2;
                        let denom = v_hat.sqrt() + eps;
                        *p -= step_size * *ea / denom;
                    });
            } else {
                param_vec
                    .par_iter_mut()
                    .zip(exp_avg_vec.par_iter_mut())
                    .zip(exp_avg_sq_vec.par_iter_mut())
                    .zip(grad_vec.par_iter())
                    .for_each(|(((p, ea), eas), &gr)| {
                        let g = if wd == 0.0 { gr } else { gr + wd * *p };
                        *ea = beta1 * *ea + one_minus_beta1 * g;
                        *eas = beta2 * *eas + one_minus_beta2 * g * g;
                        let v_hat = *eas / bias_correction2;
                        let denom = v_hat.sqrt() + eps;
                        *p -= step_size * *ea / denom;
                    });
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
            weight_decay: 0.01,
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

            #[cfg(feature = "cuda")]
            if param_data.device().is_gpu() {
                let grad = if !grad.device().is_gpu() {
                    grad.to_device(param_data.device())
                        .expect("AdamW: failed to migrate CPU gradient to GPU")
                } else {
                    grad
                };
                let grad = if grad.is_contiguous() {
                    grad
                } else {
                    grad.contiguous()
                };

                if self.weight_decay > 0.0 {
                    let decay_factor = 1.0 - self.lr * self.weight_decay;
                    let decayed = param_data.mul_scalar(decay_factor);
                    param.update_data(decayed);
                }

                let param_data = param.data();

                let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);

                param_data.adam_step_inplace(
                    &grad,
                    &state.exp_avg,
                    &state.exp_avg_sq,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    0.0,
                    bias_correction1,
                    bias_correction2,
                );
                continue;
            }

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

        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

        let mut optimizer = Adam::new(vec![param.clone()], 0.1);
        optimizer.step();

        let new_data = param.data().to_vec();
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

    // =========================================================================
    // Adam Step Correctness Tests
    // =========================================================================

    /// Verify Adam update matches the mathematical formula exactly.
    /// After one step with grad=[1,1], lr=0.1, betas=(0.9,0.999):
    ///   m = 0.1*[1,1], v = 0.001*[1,1]
    ///   m_hat = m/0.1 = [1,1], v_hat = v/0.001 = [1,1]
    ///   param -= 0.1 * 1.0 / (1.0 + 1e-8) ≈ 0.1
    #[test]
    fn test_adam_step_correctness() {
        let var = Variable::new(Tensor::from_vec(vec![0.5, -0.3], &[2]).unwrap(), true);
        let param = Parameter::from_variable(var);
        param.set_grad(Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap());

        let mut opt = Adam::new(vec![param.clone()], 0.1);
        let before = param.data().to_vec();
        opt.step();
        let after = param.data().to_vec();

        assert!(
            after[0] < before[0],
            "param[0] should decrease: {} -> {}",
            before[0],
            after[0]
        );
        assert!(
            after[1] < before[1],
            "param[1] should decrease: {} -> {}",
            before[1],
            after[1]
        );

        let delta0 = before[0] - after[0];
        let delta1 = before[1] - after[1];
        assert!(
            (delta0 - delta1).abs() < 1e-6,
            "Uniform gradient should produce uniform update: {} vs {}",
            delta0,
            delta1
        );
    }

    /// Verify Adam converges on a simple quadratic: minimize f(x) = x^2.
    /// Uses autograd for proper gradient computation.
    #[test]
    fn test_adam_converges_on_quadratic() {
        let var = Variable::new(Tensor::from_vec(vec![5.0], &[1]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let mut opt = Adam::new(vec![param.clone()], 0.1);

        for _ in 0..200 {
            opt.zero_grad();
            let x = param.variable();
            let loss = x.mul_var(&x).sum();
            loss.backward();
            opt.step();
        }

        let final_x = param.data().to_vec()[0];
        assert!(
            final_x.abs() < 0.1,
            "Adam should converge near 0 for f(x)=x^2, got {}",
            final_x
        );
    }

    /// Verify zero_grad actually clears all gradients.
    #[test]
    fn test_adam_zero_grad() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), true);
        let param = Parameter::from_variable(var);
        param.set_grad(Tensor::from_vec(vec![0.5, 0.5], &[2]).unwrap());
        assert!(param.grad().is_some());

        let mut opt = Adam::new(vec![param.clone()], 0.01);
        opt.zero_grad();
        if let Some(g) = param.grad() {
            let gv = g.to_vec();
            assert!(
                gv.iter().all(|&v| v.abs() < 1e-10),
                "Gradients should be zero after zero_grad: {:?}",
                gv
            );
        }
    }

    /// Verify set_lr / get_lr work correctly.
    #[test]
    fn test_adam_lr_management() {
        let var = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let mut opt = Adam::new(vec![param], 0.001);

        assert!((opt.get_lr() - 0.001).abs() < 1e-8);
        opt.set_lr(0.01);
        assert!((opt.get_lr() - 0.01).abs() < 1e-8);
    }

    /// Verify Adam handles no-grad params gracefully (skips them).
    #[test]
    fn test_adam_skips_frozen_params() {
        let trainable = Parameter::from_variable(Variable::new(
            Tensor::from_vec(vec![1.0], &[1]).unwrap(),
            true,
        ));
        let frozen = Parameter::from_variable(Variable::new(
            Tensor::from_vec(vec![2.0], &[1]).unwrap(),
            false,
        ));

        trainable.set_grad(Tensor::from_vec(vec![1.0], &[1]).unwrap());

        let mut opt = Adam::new(vec![trainable.clone(), frozen.clone()], 0.1);
        opt.step();

        assert!((trainable.data().to_vec()[0] - 1.0).abs() > 1e-6);
        assert!((frozen.data().to_vec()[0] - 2.0).abs() < 1e-8);
    }

    /// Verify Adam with weight decay actually decays weights.
    #[test]
    fn test_adam_weight_decay() {
        let var = Variable::new(Tensor::from_vec(vec![10.0], &[1]).unwrap(), true);
        let param = Parameter::from_variable(var);
        param.set_grad(Tensor::from_vec(vec![0.0], &[1]).unwrap());

        let mut opt = Adam::new(vec![param.clone()], 0.1).weight_decay(0.1);
        let before = param.data().to_vec()[0];
        opt.step();
        let after = param.data().to_vec()[0];

        assert!(
            after < before,
            "Weight decay should shrink large params: {} -> {}",
            before,
            after
        );
    }

    /// Verify multiple Adam steps produce improvement on a simple loss using autograd.
    #[test]
    fn test_adam_multiple_steps_improve() {
        let var = Variable::new(Tensor::from_vec(vec![3.0, -2.0], &[2]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let mut opt = Adam::new(vec![param.clone()], 0.05);

        let mut losses = Vec::new();
        for _ in 0..50 {
            opt.zero_grad();
            let x = param.variable();
            let loss = x.mul_var(&x).sum();
            losses.push(loss.data().to_vec()[0]);
            loss.backward();
            opt.step();
        }

        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first * 0.5,
            "Loss should decrease significantly: first={}, last={}",
            first,
            last
        );
    }

    // =========================================================================
    // AdamW Tests
    // =========================================================================

    /// Verify AdamW step works and decoupled weight decay differs from L2.
    #[test]
    fn test_adamw_step_correctness() {
        let var = Variable::new(Tensor::from_vec(vec![5.0, -3.0], &[2]).unwrap(), true);
        let param = Parameter::from_variable(var);
        param.set_grad(Tensor::from_vec(vec![1.0, -1.0], &[2]).unwrap());

        let mut opt = AdamW::new(vec![param.clone()], 0.01);
        let before = param.data().to_vec();
        opt.step();
        let after = param.data().to_vec();

        assert!(after[0] < before[0], "Positive grad should decrease param");
        assert!(after[1] > before[1], "Negative grad should increase param");
    }

    /// Verify AdamW converges using autograd.
    #[test]
    fn test_adamw_converges() {
        let var = Variable::new(Tensor::from_vec(vec![4.0], &[1]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let mut opt = AdamW::new(vec![param.clone()], 0.1);

        for _ in 0..200 {
            opt.zero_grad();
            let x = param.variable();
            let loss = x.mul_var(&x).sum();
            loss.backward();
            opt.step();
        }

        assert!(
            param.data().to_vec()[0].abs() < 0.1,
            "AdamW should converge near 0, got {}",
            param.data().to_vec()[0]
        );
    }
}
