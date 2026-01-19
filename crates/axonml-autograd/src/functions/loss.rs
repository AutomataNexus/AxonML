//! Loss Gradient Functions
//!
//! Gradient functions for loss operations: MSE, `CrossEntropy`, NLL, etc.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

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
    #[must_use] pub fn new(
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
        let diff = self.saved_pred.sub(&self.saved_target).unwrap();
        let numel = diff.numel() as f32;

        let grad = match self.reduction {
            Reduction::Mean => {
                let scale = 2.0 / numel;
                diff.mul_scalar(scale * grad_output.to_vec()[0])
            }
            Reduction::Sum => diff.mul_scalar(2.0 * grad_output.to_vec()[0]),
            Reduction::None => diff.mul_scalar(2.0).mul(grad_output).unwrap(),
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
    #[must_use] pub fn new(
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
        let softmax_data = self.saved_softmax.to_vec();
        let target_data = self.saved_target.to_vec();
        let batch_size = self.saved_target.numel();
        let num_classes = softmax_data.len() / batch_size;

        let mut grad_data = softmax_data.clone();

        // For each sample, subtract 1 from the target class probability
        for i in 0..batch_size {
            let target_class = target_data[i] as usize;
            grad_data[i * num_classes + target_class] -= 1.0;
        }

        let scale = match self.reduction {
            Reduction::Mean => grad_output.to_vec()[0] / batch_size as f32,
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };

        let grad: Vec<f32> = grad_data.iter().map(|&v| v * scale).collect();
        let grad = Tensor::from_vec(grad, self.saved_softmax.shape()).unwrap();

        vec![Some(grad)]
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
    #[must_use] pub fn new(
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

        let grad = Tensor::from_vec(grad_data, input_shape).unwrap();
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
    #[must_use] pub fn new(
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
        let input_data = self.saved_input.to_vec();
        let target_data = self.saved_target.to_vec();
        let numel = input_data.len() as f32;
        let eps = 1e-7_f32;

        let grad_data: Vec<f32> = input_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &y)| {
                let p = p.clamp(eps, 1.0 - eps);
                (p - y) / (p * (1.0 - p))
            })
            .collect();

        let scale = match self.reduction {
            Reduction::Mean => grad_output.to_vec()[0] / numel,
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };

        let grad: Vec<f32> = grad_data.iter().map(|&v| v * scale).collect();
        let grad = Tensor::from_vec(grad, self.saved_input.shape()).unwrap();

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
    #[must_use] pub fn new(
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
        let pred_data = self.saved_pred.to_vec();
        let target_data = self.saved_target.to_vec();
        let numel = pred_data.len() as f32;

        let grad_data: Vec<f32> = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &t)| {
                if p > t {
                    1.0
                } else if p < t {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();

        let scale = match self.reduction {
            Reduction::Mean => grad_output.to_vec()[0] / numel,
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };

        let grad: Vec<f32> = grad_data.iter().map(|&v| v * scale).collect();
        let grad = Tensor::from_vec(grad, self.saved_pred.shape()).unwrap();

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
    #[must_use] pub fn new(
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
        let pred_data = self.saved_pred.to_vec();
        let target_data = self.saved_target.to_vec();
        let numel = pred_data.len() as f32;

        let grad_data: Vec<f32> = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                if diff.abs() < self.beta {
                    diff / self.beta
                } else if diff > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        let scale = match self.reduction {
            Reduction::Mean => grad_output.to_vec()[0] / numel,
            Reduction::Sum => grad_output.to_vec()[0],
            Reduction::None => 1.0,
        };

        let grad: Vec<f32> = grad_data.iter().map(|&v| v * scale).collect();
        let grad = Tensor::from_vec(grad, self.saved_pred.shape()).unwrap();

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
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
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
        let pred = Tensor::from_vec(vec![2.0, 1.0, 3.0], &[3]).unwrap();
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
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
