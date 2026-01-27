//! `RMSprop` Optimizer
//!
//! Implements `RMSprop` (Root Mean Square Propagation) optimizer.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_nn::Parameter;
use axonml_tensor::Tensor;

use crate::optimizer::Optimizer;

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

/// State for `RMSprop` optimizer.
#[derive(Debug, Clone)]
struct RMSpropState {
    /// Square average of gradients.
    square_avg: Vec<f32>,
    /// Momentum buffer.
    momentum_buffer: Option<Vec<f32>>,
    /// Gradient average (for centered `RMSprop`).
    grad_avg: Option<Vec<f32>>,
}

impl RMSpropState {
    fn new(size: usize, momentum: bool, centered: bool) -> Self {
        Self {
            square_avg: vec![0.0; size],
            momentum_buffer: if momentum {
                Some(vec![0.0; size])
            } else {
                None
            },
            grad_avg: if centered {
                Some(vec![0.0; size])
            } else {
                None
            },
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
                .map(|p| RMSpropState::new(p.numel(), self.momentum != 0.0, self.centered))
                .collect();
        }
    }
}

impl Optimizer for RMSprop {
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

            let mut grad_vec = grad.to_vec();
            let state = &mut self.state[i];

            let param_data = param.data();
            let mut param_vec = param_data.to_vec();

            // Apply weight decay
            if self.weight_decay != 0.0 {
                for (g, p) in grad_vec.iter_mut().zip(param_vec.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update square average
            for (sq, g) in state.square_avg.iter_mut().zip(grad_vec.iter()) {
                *sq = self.alpha * *sq + (1.0 - self.alpha) * g * g;
            }

            // Compute denominator
            let avg: Vec<f32> = if self.centered {
                // Update gradient average for centered RMSprop
                let grad_avg = state.grad_avg.as_mut().unwrap();
                for (ga, g) in grad_avg.iter_mut().zip(grad_vec.iter()) {
                    *ga = self.alpha * *ga + (1.0 - self.alpha) * g;
                }
                // avg = sqrt(square_avg - grad_avg^2)
                state
                    .square_avg
                    .iter()
                    .zip(grad_avg.iter())
                    .map(|(sq, ga)| (sq - ga * ga).sqrt() + self.eps)
                    .collect()
            } else {
                state
                    .square_avg
                    .iter()
                    .map(|sq| sq.sqrt() + self.eps)
                    .collect()
            };

            // Update parameters
            if self.momentum == 0.0 {
                // Without momentum
                for ((p, g), a) in param_vec.iter_mut().zip(grad_vec.iter()).zip(avg.iter()) {
                    *p -= self.lr * g / a;
                }
            } else {
                // With momentum
                let buf = state.momentum_buffer.as_mut().unwrap();
                for ((b, g), a) in buf.iter_mut().zip(grad_vec.iter()).zip(avg.iter()) {
                    *b = self.momentum * *b + g / a;
                }
                for (p, b) in param_vec.iter_mut().zip(buf.iter()) {
                    *p -= self.lr * b;
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
    fn test_rmsprop_creation() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let optimizer = RMSprop::new(vec![param], 0.01);

        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);
        assert!((optimizer.alpha - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_step() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap());

        let mut optimizer = RMSprop::new(vec![param.clone()], 0.01);
        optimizer.step();

        let new_data = param.data().to_vec();
        // Parameters should have changed
        assert!((new_data[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        let optimizer = RMSprop::new(vec![param], 0.01).momentum(0.9);

        assert!((optimizer.momentum - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_centered() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        let optimizer = RMSprop::new(vec![param], 0.01).centered(true);

        assert!(optimizer.centered);
    }

    #[test]
    fn test_rmsprop_builder_pattern() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
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
