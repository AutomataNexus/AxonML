//! `RMSprop` — root mean square propagation optimizer.
//!
//! Exponential moving average of squared gradients for per-parameter
//! adaptive learning rates. Config: alpha (decay), epsilon, optional
//! momentum, optional centered mode (subtract mean of squared grads).
//!
//! # File
//! `crates/axonml-optim/src/rmsprop.rs`
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

use crate::optimizer::Optimizer;

// Re-import Device for state initialization
use axonml_core;

// =============================================================================
// RMSprop
// =============================================================================

/// `RMSprop` optimizer.
///
/// Maintains a moving average of squared gradients to normalize updates.
///
/// Update rule:
/// ```text
/// v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
/// param = param - lr * grad / (sqrt(v_t) + eps)
/// ```
///
/// With momentum:
/// ```text
/// v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
/// buf_t = momentum * buf_{t-1} + grad / (sqrt(v_t) + eps)
/// param = param - lr * buf_t
/// ```
pub struct RMSprop {
    /// Parameters to optimize.
    params: Vec<Parameter>,
    /// Learning rate.
    lr: f32,
    /// Smoothing constant (decay rate for moving average).
    alpha: f32,
    /// Small constant for numerical stability.
    eps: f32,
    /// Weight decay (L2 regularization).
    weight_decay: f32,
    /// Momentum factor.
    momentum: f32,
    /// Whether to center the gradient (subtract mean).
    centered: bool,
    /// Per-parameter state.
    state: Vec<RMSpropState>,
}

/// Tensor-based state for `RMSprop` optimizer.
///
/// All buffers are stored as `Tensor<f32>` so they stay GPU-resident when
/// parameters are on GPU, avoiding round-trip copies through `to_vec()`.
#[derive(Debug, Clone)]
struct RMSpropState {
    /// Square average of gradients.
    square_avg: Tensor<f32>,
    /// Momentum buffer.
    momentum_buffer: Option<Tensor<f32>>,
    /// Gradient average (for centered `RMSprop`).
    grad_avg: Option<Tensor<f32>>,
}

impl RMSpropState {
    fn new(shape: &[usize], device: axonml_core::Device, momentum: bool, centered: bool) -> Self {
        let square_avg = {
            let t = Tensor::zeros(shape);
            if device.is_gpu() {
                t.to_device(device).expect("device transfer failed")
            } else {
                t
            }
        };
        let momentum_buffer = if momentum {
            let t = Tensor::zeros(shape);
            Some(if device.is_gpu() {
                t.to_device(device).expect("device transfer failed")
            } else {
                t
            })
        } else {
            None
        };
        let grad_avg = if centered {
            let t = Tensor::zeros(shape);
            Some(if device.is_gpu() {
                t.to_device(device).expect("device transfer failed")
            } else {
                t
            })
        } else {
            None
        };
        Self {
            square_avg,
            momentum_buffer,
            grad_avg,
        }
    }
}

impl RMSprop {
    /// Creates a new `RMSprop` optimizer with default settings.
    #[must_use]
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        Self {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            state: Vec::new(),
        }
    }

    /// Creates `RMSprop` with specified alpha (smoothing constant).
    #[must_use]
    pub fn with_alpha(params: Vec<Parameter>, lr: f32, alpha: f32) -> Self {
        Self {
            params,
            lr,
            alpha,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            state: Vec::new(),
        }
    }

    /// Creates `RMSprop` with all options.
    #[must_use]
    pub fn with_options(
        params: Vec<Parameter>,
        lr: f32,
        alpha: f32,
        eps: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
    ) -> Self {
        Self {
            params,
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            state: Vec::new(),
        }
    }

    /// Builder method to set alpha.
    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
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

    /// Builder method to set momentum.
    #[must_use]
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Builder method to enable centered `RMSprop`.
    #[must_use]
    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    fn ensure_state_initialized(&mut self) {
        if self.state.is_empty() {
            self.state = self
                .params
                .iter()
                .map(|p| {
                    let data = p.data();
                    RMSpropState::new(
                        data.shape(),
                        data.device(),
                        self.momentum != 0.0,
                        self.centered,
                    )
                })
                .collect();
        }
    }
}

impl Optimizer for RMSprop {
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

            let param_data = param.data();
            let state = &mut self.state[i];

            // Apply weight decay: d = grad + weight_decay * param
            let d = if self.weight_decay == 0.0 {
                grad.clone()
            } else {
                grad.add(&param_data.mul_scalar(self.weight_decay)).unwrap()
            };

            // Update square average: sq_avg = alpha * sq_avg + (1 - alpha) * d^2
            let d_sq = d.mul(&d).unwrap();
            state.square_avg = state
                .square_avg
                .mul_scalar(self.alpha)
                .add(&d_sq.mul_scalar(1.0 - self.alpha))
                .unwrap();

            // Compute denominator
            let denom = if self.centered {
                // Update gradient average: grad_avg = alpha * grad_avg + (1 - alpha) * d
                let grad_avg = state.grad_avg.as_mut().unwrap();
                *grad_avg = grad_avg
                    .mul_scalar(self.alpha)
                    .add(&d.mul_scalar(1.0 - self.alpha))
                    .unwrap();

                // denom = sqrt(sq_avg - grad_avg^2) + eps
                let ga_sq = grad_avg.mul(grad_avg).unwrap();
                state
                    .square_avg
                    .sub(&ga_sq)
                    .unwrap()
                    .sqrt()
                    .add_scalar(self.eps)
            } else {
                // denom = sqrt(sq_avg) + eps
                state.square_avg.sqrt().add_scalar(self.eps)
            };

            // Apply update with or without momentum
            let update = if self.momentum == 0.0 {
                // update = d / denom
                d.div(&denom).unwrap()
            } else {
                // buf = momentum * buf + d / denom
                let normalized = d.div(&denom).unwrap();
                let buf = state.momentum_buffer.as_mut().unwrap();
                *buf = buf.mul_scalar(self.momentum).add(&normalized).unwrap();
                buf.clone()
            };

            // param = param - lr * update
            let new_param = param_data.sub(&update.mul_scalar(self.lr)).unwrap();
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
    fn test_rmsprop_creation() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);
        let optimizer = RMSprop::new(vec![param], 0.01);

        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);
        assert!((optimizer.alpha - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_step() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

        let mut optimizer = RMSprop::new(vec![param.clone()], 0.01);
        optimizer.step();

        let new_data = param.data().to_vec();
        // Parameters should have changed
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        let optimizer = RMSprop::new(vec![param], 0.01).momentum(0.9);

        assert!((optimizer.momentum - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_centered() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        let optimizer = RMSprop::new(vec![param], 0.01).centered(true);

        assert!(optimizer.centered);
    }

    #[test]
    fn test_rmsprop_builder_pattern() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        let optimizer = RMSprop::new(vec![param], 0.01)
            .alpha(0.95)
            .eps(1e-6)
            .weight_decay(0.0001)
            .momentum(0.9)
            .centered(true);

        assert!((optimizer.alpha - 0.95).abs() < 1e-6);
        assert!((optimizer.eps - 1e-6).abs() < 1e-9);
        assert!((optimizer.weight_decay - 0.0001).abs() < 1e-6);
        assert!((optimizer.momentum - 0.9).abs() < 1e-6);
        assert!(optimizer.centered);
    }
}
