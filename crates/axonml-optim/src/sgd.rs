//! SGD Optimizer - Stochastic Gradient Descent
//!
//! Implements SGD with optional momentum and Nesterov acceleration.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_nn::Parameter;
use axonml_tensor::Tensor;

use crate::optimizer::{Optimizer, ParamState};

// =============================================================================
// SGD
// =============================================================================

/// Stochastic Gradient Descent optimizer.
///
/// Supports momentum and Nesterov acceleration.
///
/// Update rule (with momentum):
/// ```text
/// v_t = momentum * v_{t-1} + grad
/// param = param - lr * v_t
/// ```
///
/// Update rule (with Nesterov):
/// ```text
/// v_t = momentum * v_{t-1} + grad
/// param = param - lr * (momentum * v_t + grad)
/// ```
pub struct SGD {
    /// Parameters to optimize.
    params: Vec<Parameter>,
    /// Learning rate.
    lr: f32,
    /// Momentum factor.
    momentum: f32,
    /// Weight decay (L2 regularization).
    weight_decay: f32,
    /// Whether to use Nesterov momentum.
    nesterov: bool,
    /// Dampening factor for momentum.
    dampening: f32,
    /// Per-parameter state (momentum buffers).
    state: Vec<ParamState>,
}

impl SGD {
    /// Creates a new SGD optimizer with default settings.
    #[must_use] pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            dampening: 0.0,
            state: vec![ParamState::new(); num_params],
        }
    }

    /// Creates SGD with momentum.
    #[must_use] pub fn with_momentum(params: Vec<Parameter>, lr: f32, momentum: f32) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
            dampening: 0.0,
            state: vec![ParamState::new(); num_params],
        }
    }

    /// Creates SGD with all options.
    #[must_use] pub fn with_options(
        params: Vec<Parameter>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        dampening: f32,
        nesterov: bool,
    ) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum,
            weight_decay,
            nesterov,
            dampening,
            state: vec![ParamState::new(); num_params],
        }
    }

    /// Builder method to set momentum.
    #[must_use] pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Builder method to set weight decay.
    #[must_use] pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to enable Nesterov momentum.
    #[must_use] pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Builder method to set dampening.
    #[must_use] pub fn dampening(mut self, dampening: f32) -> Self {
        self.dampening = dampening;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let mut grad_vec = grad.to_vec();

            // Apply weight decay
            if self.weight_decay != 0.0 {
                let param_vec = param.data().to_vec();
                for (g, p) in grad_vec.iter_mut().zip(param_vec.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Apply momentum
            if self.momentum != 0.0 {
                let state = &mut self.state[i];

                if state.momentum_buffer.is_none() {
                    // First iteration: initialize momentum buffer
                    state.init_momentum(grad_vec.len());
                    let buf = state.momentum_buffer.as_mut().unwrap();
                    buf.copy_from_slice(&grad_vec);
                } else {
                    // Subsequent iterations: update momentum buffer
                    let buf = state.momentum_buffer.as_mut().unwrap();
                    for (b, g) in buf.iter_mut().zip(grad_vec.iter()) {
                        *b = self.momentum * *b + (1.0 - self.dampening) * *g;
                    }
                }

                let buf = state.momentum_buffer.as_ref().unwrap();

                if self.nesterov {
                    // Nesterov: use momentum * buf + grad
                    let nesterov_grad: Vec<f32> = buf
                        .iter()
                        .zip(grad_vec.iter())
                        .map(|(b, g)| self.momentum * *b + *g)
                        .collect();
                    grad_vec = nesterov_grad;
                } else {
                    // Standard momentum: use buf directly
                    grad_vec = buf.clone();
                }
            }

            // Update parameters: param = param - lr * grad
            let param_data = param.data();
            let param_vec = param_data.to_vec();
            let new_data: Vec<f32> = param_vec
                .iter()
                .zip(grad_vec.iter())
                .map(|(p, g)| p - self.lr * g)
                .collect();

            let update = Tensor::from_vec(new_data, param_data.shape()).unwrap();
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
    fn test_sgd_creation() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let optimizer = SGD::new(vec![param], 0.01);

        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);
        assert_eq!(optimizer.num_parameters(), 1);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);
        let optimizer = SGD::with_momentum(vec![param], 0.01, 0.9);

        assert!((optimizer.momentum - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        // Manually set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap());

        let mut optimizer = SGD::new(vec![param.clone()], 0.1);
        optimizer.step();

        let new_data = param.data().to_vec();
        // param = param - lr * grad = [1, 2, 3] - 0.1 * [0.1, 0.2, 0.3]
        assert!((new_data[0] - 0.99).abs() < 1e-5);
        assert!((new_data[1] - 1.98).abs() < 1e-5);
        assert!((new_data[2] - 2.97).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_zero_grad() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap());

        let mut optimizer = SGD::new(vec![param.clone()], 0.1);

        // Verify gradient exists
        assert!(param.grad().is_some());

        optimizer.zero_grad();

        // Gradient should be zeroed
        let grad = param.grad();
        if let Some(g) = grad {
            assert!(g.to_vec().iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_sgd_builder_pattern() {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);

        let optimizer = SGD::new(vec![param], 0.01)
            .momentum(0.9)
            .weight_decay(0.0001)
            .nesterov(true);

        assert!((optimizer.momentum - 0.9).abs() < 1e-6);
        assert!((optimizer.weight_decay - 0.0001).abs() < 1e-6);
        assert!(optimizer.nesterov);
    }
}
