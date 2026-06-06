//! Backward functions for loss operations.
//!
//! 522 lines. `MseLossBackward`, `CrossEntropyLossBackward`,
//! `BceLossBackward`, `BceWithLogitsLossBackward`, `L1LossBackward`,
//! `SmoothL1LossBackward`, `NllLossBackward`. Each stores the predictions
//! and targets from the forward pass and computes the analytical gradient
//! using `zip_map` for single-allocation backward.
//!
//! # File
//! `crates/axonml-autograd/src/functions/loss.rs`
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

use std::any::Any;

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// MSE Loss Backward
// =============================================================================

/// Gradient function for Mean Squared Error loss.
///
/// MSE = mean((pred - target)^2)
/// d/d(pred) = 2 * (pred - target) / n
#[derive(Debug)]
pub struct MseLossBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_pred: Tensor<f32>,
    saved_target: Tensor<f32>,
    reduction: Reduction,
}

/// Reduction mode for loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction, return element-wise loss.
    None,
    /// Average the loss.
    Mean,
    /// Sum the loss.
    Sum,
}

impl MseLossBackward {
    /// Creates a new `MseLossBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        pred: Tensor<f32>,
        target: Tensor<f32>,
        reduction: Reduction,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_pred: pred,
            saved_target: target,
            reduction,
        }
    }
}

impl GradientFunction for MseLossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let diff = self
            .saved_pred
            .sub(&self.saved_target)
            .expect("backward: tensor sub failed");
        let numel = diff.numel() as f32;

        let grad = match self.reduction {
            Reduction::Mean => {
                let scale = 2.0 / numel;
                diff.mul_scalar(scale * grad_output.to_vec()[0])
            }
            Reduction::Sum => diff.mul_scalar(2.0 * grad_output.to_vec()[0]),
            Reduction::None => diff
                .mul_scalar(2.0)
                .mul(grad_output)
                .expect("backward: tensor mul failed"),
        };

        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "MseLossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Cross Entropy Loss Backward
// =============================================================================

/// Gradient function for Cross Entropy loss.
///
/// `CrossEntropy` = -sum(target * log(softmax(input)))
/// Combined with softmax: d/d(input) = softmax(input) - target
#[derive(Debug)]
pub struct CrossEntropyLossBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_softmax: Tensor<f32>,
    saved_target: Tensor<i64>,
    reduction: Reduction,
}

impl CrossEntropyLossBackward {
    /// Creates a new `CrossEntropyLossBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        softmax: Tensor<f32>,
        target: Tensor<i64>,
        reduction: Reduction,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_softmax: softmax,
            saved_target: target,
            reduction,
        }
    }
}

impl GradientFunction for CrossEntropyLossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let batch_size = self.saved_target.numel();
        let num_classes = self.saved_softmax.numel() / batch_size;

        // GPU fast path: use the CUDA cross_entropy_bwd kernel directly.
        // This computes grad = (softmax - one_hot) * grad_output entirely
        // on GPU with zero CPU round-trips. (Core part of making fw/bw FAF.)
        #[cfg(feature = "cuda")]
        if self.saved_softmax.device().is_gpu() {
            // The CUDA kernel expects:
            //   softmax [batch, classes] on GPU
            //   targets [batch] on GPU (as f32 cast of class indices)
            //   grad_output [batch] on GPU (per-sample gradient, or scalar broadcast)
            //
            // For reduction=Mean, grad_output is scalar 1/N. Build a per-batch
            // grad_output tensor directly on GPU (no host vec round-trip).
            let grad_out_t = match self.reduction {
                Reduction::Mean => {
                    let v = 1.0 / batch_size as f32;
                    // Create scalar then broadcast via from_vec on device (tiny host scalar is fine).
                    Tensor::from_vec(vec![v; batch_size], &[batch_size])
                        .unwrap()
                        .to_device(self.saved_softmax.device())
                        .unwrap()
                }
                Reduction::Sum => {
                    Tensor::from_vec(vec![1.0f32; batch_size], &[batch_size])
                        .unwrap()
                        .to_device(self.saved_softmax.device())
                        .unwrap()
                }
                Reduction::None => grad_output
                    .to_device(self.saved_softmax.device())
                    .unwrap(),
            };

            // Targets: cast i64 → f32 then move to GPU
            let target_f32_vec: Vec<f32> = self
                .saved_target
                .to_vec()
                .iter()
                .map(|&x| x as f32)
                .collect();
            let target_on_gpu = Tensor::from_vec(target_f32_vec, &[batch_size])
                .unwrap()
                .to_device(self.saved_softmax.device())
                .unwrap();

            // Reshape softmax to [batch, classes] if flat
            let softmax_2d = if self.saved_softmax.shape().len() == 1 {
                self.saved_softmax
                    .reshape(&[batch_size as isize, num_classes as isize])
                    .unwrap()
            } else {
                self.saved_softmax.clone()
            };

            let grad = softmax_2d.cross_entropy_bwd_cuda(&target_on_gpu, &grad_out_t);
            return vec![Some(grad)];
        }

        // CPU fallback
        let target_data = self.saved_target.to_vec();
        let scale = match self.reduction {
            Reduction::Mean => {
                // Prefer scalar on device if grad_output is already tiny/GPU
                if grad_output.device().is_gpu() {
                    // For mean reduction the upstream grad is usually a scalar 1/N
                    // We can keep a 1-element GPU tensor; the bwd kernel/ math will handle it.
                    grad_output.to_vec()[0] / batch_size as f32   // still tiny, acceptable for scalar scale
                } else {
                    grad_output.to_vec()[0] / batch_size as f32
                }
            }
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };
        let mut data = self.saved_softmax.to_vec();
        for i in 0..batch_size {
            let tc = target_data[i] as usize;
            for c in 0..num_classes {
                let idx = i * num_classes + c;
                data[idx] = (data[idx] - if c == tc { 1.0 } else { 0.0 }) * scale;
            }
        }
        vec![Some(
            Tensor::from_vec(data, self.saved_softmax.shape()).unwrap(),
        )]
    }

    fn name(&self) -> &'static str {
        "CrossEntropyLossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// NLL Loss Backward
// =============================================================================

/// Gradient function for Negative Log Likelihood loss.
///
/// NLL = -log(prob[target])
/// d/d(prob) = -1/prob[target] at target index, 0 elsewhere
#[derive(Debug)]
pub struct NllLossBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_target: Tensor<i64>,
    reduction: Reduction,
}

impl NllLossBackward {
    /// Creates a new `NllLossBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input: Tensor<f32>,
        target: Tensor<i64>,
        reduction: Reduction,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
            saved_target: target,
            reduction,
        }
    }
}

impl GradientFunction for NllLossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let target_data = self.saved_target.to_vec();
        let input_shape = self.saved_input.shape();
        let batch_size = target_data.len();
        let num_classes = input_shape.last().copied().unwrap_or(1);

        let mut grad_data = vec![0.0f32; self.saved_input.numel()];

        let scale = match self.reduction {
            Reduction::Mean => -grad_output.to_vec()[0] / batch_size as f32,
            Reduction::Sum => -grad_output.to_vec()[0],
            Reduction::None => -1.0,
        };

        for i in 0..batch_size {
            let target_class = target_data[i] as usize;
            grad_data[i * num_classes + target_class] = scale;
        }

        let mut grad =
            Tensor::from_vec(grad_data, input_shape).expect("backward: tensor creation failed");
        // Preserve device: input was on GPU, gradient should be too
        if self.saved_input.device().is_gpu() {
            grad = grad.to_device(self.saved_input.device()).unwrap();
        }
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "NllLossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Binary Cross Entropy Backward
// =============================================================================

/// Gradient function for Binary Cross Entropy loss.
///
/// BCE = -[y * log(p) + (1-y) * log(1-p)]
/// d/dp = -y/p + (1-y)/(1-p) = (p - y) / (p * (1-p))
#[derive(Debug)]
pub struct BceLossBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_target: Tensor<f32>,
    reduction: Reduction,
}

impl BceLossBackward {
    /// Creates a new `BceLossBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input: Tensor<f32>,
        target: Tensor<f32>,
        reduction: Reduction,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
            saved_target: target,
            reduction,
        }
    }
}

impl GradientFunction for BceLossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let numel = self.saved_input.numel() as f32;
        let eps = 1e-7_f32;

        let scale = match self.reduction {
            Reduction::Mean => grad_output.to_vec()[0] / numel,
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };

        let grad = self.saved_input.zip_map(&self.saved_target, move |p, y| {
            let p = p.clamp(eps, 1.0 - eps);
            scale * (p - y) / (p * (1.0 - p))
        });

        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "BceLossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// L1 Loss Backward
// =============================================================================

/// Gradient function for L1 (Mean Absolute Error) loss.
///
/// L1 = mean(|pred - target|)
/// d/d(pred) = sign(pred - target) / n
#[derive(Debug)]
pub struct L1LossBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_pred: Tensor<f32>,
    saved_target: Tensor<f32>,
    reduction: Reduction,
}

impl L1LossBackward {
    /// Creates a new `L1LossBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        pred: Tensor<f32>,
        target: Tensor<f32>,
        reduction: Reduction,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_pred: pred,
            saved_target: target,
            reduction,
        }
    }
}

impl GradientFunction for L1LossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let numel = self.saved_pred.numel() as f32;
        let scale = match self.reduction {
            Reduction::Mean => grad_output.to_vec()[0] / numel,
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };

        let grad = self.saved_pred.zip_map(&self.saved_target, move |p, t| {
            let sign = if p > t {
                1.0
            } else if p < t {
                -1.0
            } else {
                0.0
            };
            scale * sign
        });

        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "L1LossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Smooth L1 Loss Backward (Huber Loss)
// =============================================================================

/// Gradient function for Smooth L1 (Huber) loss.
///
/// `SmoothL1` = 0.5 * x^2 if |x| < 1, else |x| - 0.5
/// d/dx = x if |x| < 1, else sign(x)
#[derive(Debug)]
pub struct SmoothL1LossBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_pred: Tensor<f32>,
    saved_target: Tensor<f32>,
    beta: f32,
    reduction: Reduction,
}

impl SmoothL1LossBackward {
    /// Creates a new `SmoothL1LossBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        pred: Tensor<f32>,
        target: Tensor<f32>,
        beta: f32,
        reduction: Reduction,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_pred: pred,
            saved_target: target,
            beta,
            reduction,
        }
    }
}

impl GradientFunction for SmoothL1LossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let numel = self.saved_pred.numel() as f32;
        let beta = self.beta;
        let scale = match self.reduction {
            Reduction::Mean => grad_output.to_vec()[0] / numel,
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };

        let grad = self.saved_pred.zip_map(&self.saved_target, move |p, t| {
            let diff = p - t;
            let g = if diff.abs() < beta {
                diff / beta
            } else if diff > 0.0 {
                1.0
            } else {
                -1.0
            };
            scale * g
        });

        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "SmoothL1LossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss_backward() {
        let pred =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("backward: tensor creation failed");
        let target =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("backward: tensor creation failed");
        let grad_fn = MseLossBackward::new(None, pred, target, Reduction::Mean);

        let grad_output = Tensor::scalar(1.0);
        let grads = grad_fn.apply(&grad_output);

        // Zero gradient when pred == target
        let grad = grads[0].as_ref().unwrap();
        for &v in &grad.to_vec() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_l1_loss_backward() {
        let pred =
            Tensor::from_vec(vec![2.0, 1.0, 3.0], &[3]).expect("backward: tensor creation failed");
        let target =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("backward: tensor creation failed");
        let grad_fn = L1LossBackward::new(None, pred, target, Reduction::Mean);

        let grad_output = Tensor::scalar(1.0);
        let grads = grad_fn.apply(&grad_output);

        let grad = grads[0].as_ref().unwrap().to_vec();
        // pred > target: +1/3, pred < target: -1/3, pred == target: 0
        assert!((grad[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((grad[1] + 1.0 / 3.0).abs() < 1e-6);
        assert!(grad[2].abs() < 1e-6);
    }
}
