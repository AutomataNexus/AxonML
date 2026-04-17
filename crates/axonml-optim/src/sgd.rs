//! `SGD` — Stochastic Gradient Descent with optional momentum and Nesterov.
//!
//! `SGD::new(params, lr)`, `.momentum(m)`, `.nesterov(true)`,
//! `.weight_decay(wd)`, `.dampening(d)`. Standard PyTorch-equivalent
//! update rule with velocity buffer for momentum variants.
//!
//! # File
//! `crates/axonml-optim/src/sgd.rs`
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
    /// Per-parameter Tensor-based momentum buffers (GPU or CPU).
    /// Lazily initialized on first step when momentum != 0.
    momentum_buffers: Vec<Option<Tensor<f32>>>,
}

impl SGD {
    /// Creates a new SGD optimizer with default settings.
    #[must_use]
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            dampening: 0.0,
            momentum_buffers: vec![None; num_params],
        }
    }

    /// Creates SGD with momentum.
    #[must_use]
    pub fn with_momentum(params: Vec<Parameter>, lr: f32, momentum: f32) -> Self {
        let num_params = params.len();
        Self {
            params,
            lr,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
            dampening: 0.0,
            momentum_buffers: vec![None; num_params],
        }
    }

    /// Creates SGD with all options.
    #[must_use]
    pub fn with_options(
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
            momentum_buffers: vec![None; num_params],
        }
    }

    /// Builder method to set momentum.
    #[must_use]
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Builder method to set weight decay.
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to enable Nesterov momentum.
    #[must_use]
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Builder method to set dampening.
    #[must_use]
    pub fn dampening(mut self, dampening: f32) -> Self {
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

            let param_data = param.data();

            // ============================================================
            // Tensor-op path: works on both CPU and GPU without to_vec()
            // All ops (add, mul, mul_scalar, sub) dispatch to CUDA when
            // the tensors are GPU-resident.
            // ============================================================

            // Apply weight decay: d = grad + weight_decay * param
            let d = if self.weight_decay == 0.0 {
                grad.clone()
            } else {
                grad.add(&param_data.mul_scalar(self.weight_decay)).unwrap()
            };

            // Apply momentum
            let update_dir = if self.momentum == 0.0 {
                d
            } else {
                let buf = &mut self.momentum_buffers[i];

                if buf.is_none() {
                    // First iteration: momentum buffer = d
                    *buf = Some(d.clone());
                } else {
                    // buf = momentum * buf + (1 - dampening) * d
                    let old = buf.as_ref().unwrap();
                    let new_buf = old
                        .mul_scalar(self.momentum)
                        .add(&d.mul_scalar(1.0 - self.dampening))
                        .unwrap();
                    *buf = Some(new_buf);
                }

                let buf_ref = buf.as_ref().unwrap();

                if self.nesterov {
                    // effective = d + momentum * buf
                    d.add(&buf_ref.mul_scalar(self.momentum)).unwrap()
                } else {
                    buf_ref.clone()
                }
            };

            // param = param - lr * update_dir
            let new_param = param_data.sub(&update_dir.mul_scalar(self.lr)).unwrap();
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
    fn test_sgd_creation() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);
        let optimizer = SGD::new(vec![param], 0.01);

        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);
        assert_eq!(optimizer.num_parameters(), 1);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);
        let optimizer = SGD::with_momentum(vec![param], 0.01, 0.9);

        assert!((optimizer.momentum - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        // Manually set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

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
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let param = Parameter::from_variable(var);

        // Set gradient
        param
            .variable()
            .set_grad(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).expect("tensor creation failed"));

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
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
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
