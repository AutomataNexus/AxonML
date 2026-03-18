//! Loss Functions - Training Objectives
//!
//! # File
//! `crates/axonml-nn/src/loss.rs`
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

use std::any::Any;

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_tensor::Tensor;

use crate::module::Module;

// =============================================================================
// Reduction Enum
// =============================================================================

/// Specifies how to reduce the loss over elements.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Reduction {
    /// No reduction - return loss per element.
    None,
    /// Mean of all losses.
    #[default]
    Mean,
    /// Sum of all losses.
    Sum,
}

// =============================================================================
// MSELoss
// =============================================================================

/// Mean Squared Error loss.
///
/// loss = mean((input - target)^2)
#[derive(Debug, Clone, Copy)]
pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    /// Creates a new MSELoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates MSELoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let diff = input.sub_var(target);
        let squared = diff.pow(2.0);

        match self.reduction {
            Reduction::None => squared,
            Reduction::Mean => squared.mean(),
            Reduction::Sum => squared.sum(),
        }
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MSELoss {
    fn forward(&self, input: &Variable) -> Variable {
        // For Module interface, we can't easily pass two inputs
        // This is primarily used via compute() method
        input.clone()
    }

    fn name(&self) -> &'static str {
        "MSELoss"
    }
}

// =============================================================================
// L1Loss
// =============================================================================

/// Mean Absolute Error loss.
///
/// loss = mean(|input - target|)
#[derive(Debug, Clone, Copy)]
pub struct L1Loss {
    reduction: Reduction,
}

impl L1Loss {
    /// Creates a new L1Loss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates L1Loss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();
        // diff = input - target (Tensor op, auto-dispatches to GPU)
        let diff_tensor = input_data.sub(&target_data).unwrap();
        // |diff| = relu(diff) + relu(-diff), using Tensor ops that auto-dispatch to GPU
        let relu_diff = axonml_tensor::ops::clamp_min(&diff_tensor, 0.0);
        let relu_neg_diff = axonml_tensor::ops::clamp_min(&diff_tensor.neg(), 0.0);
        let abs_tensor = relu_diff.add(&relu_neg_diff).unwrap();

        let requires_grad = (input.requires_grad() || target.requires_grad()) && is_grad_enabled();
        let loss_var = if requires_grad {
            let grad_fn = GradFn::new(L1LossBackward {
                next_fns: vec![input.grad_fn().cloned(), target.grad_fn().cloned()],
                diff_tensor,
            });
            Variable::from_operation(abs_tensor, grad_fn, true)
        } else {
            Variable::new(abs_tensor, false)
        };

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for L1Loss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// L1LossBackward
// =============================================================================

/// Gradient function for L1Loss.
///
/// d/d(input) = sign(input - target)
/// d/d(target) = -sign(input - target)
///
/// Stores diff as Tensor<f32> so it stays on GPU when applicable.
#[derive(Debug)]
struct L1LossBackward {
    next_fns: Vec<Option<GradFn>>,
    diff_tensor: Tensor<f32>,
}

impl GradientFunction for L1LossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // sign(diff): +1 where diff > 0, -1 where diff < 0, 0 where diff == 0
        // Compute as: diff / (|diff| + eps) which gives sign and handles GPU
        let eps_tensor = Tensor::full(self.diff_tensor.shape(), 1e-12);
        let eps_on_device = if self.diff_tensor.device().is_gpu() {
            eps_tensor.to_device(self.diff_tensor.device()).unwrap()
        } else {
            eps_tensor
        };
        // |diff| approximated as sqrt(diff^2 + eps)  — but simpler: diff * diff then sqrt
        let diff_sq = self.diff_tensor.mul(&self.diff_tensor).unwrap();
        let diff_sq_eps = diff_sq.add(&eps_on_device).unwrap();
        // sqrt via exp(0.5 * ln(x))
        let abs_diff = diff_sq_eps.ln().mul_scalar(0.5).exp();
        let sign_diff = self.diff_tensor.div(&abs_diff).unwrap();

        // grad_input = sign(diff) * grad_output
        let gi = sign_diff.mul(grad_output).unwrap();
        // grad_target = -grad_input
        let gt = gi.neg();
        vec![Some(gi), Some(gt)]
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
// CrossEntropyBackward
// =============================================================================

/// Gradient function for CrossEntropyLoss.
///
/// The gradient of CE w.r.t. logits is: softmax(logits) - one_hot(target).
/// For per-sample losses, each sample's gradient is scaled by the upstream
/// gradient (from reduction).
#[derive(Debug)]
struct CrossEntropyBackward {
    next_fns: Vec<Option<GradFn>>,
    /// Softmax probabilities computed during forward pass, shape (N, C).
    /// Stays on GPU if input was on GPU.
    softmax_probs: Tensor<f32>,
    /// Target class indices as f32, shape (N,). Stays on GPU if input was on GPU.
    targets: Tensor<f32>,
    batch_size: usize,
    num_classes: usize,
}

impl GradientFunction for CrossEntropyBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: use CUDA cross_entropy_bwd kernel
        #[cfg(feature = "cuda")]
        if self.softmax_probs.device().is_gpu() {
            let grad_out_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.softmax_probs.device()).unwrap()
            };
            let grad_tensor = self
                .softmax_probs
                .cross_entropy_bwd_cuda(&self.targets, &grad_out_gpu);
            return vec![Some(grad_tensor)];
        }

        // CPU path
        let softmax_vec = self.softmax_probs.to_vec();
        let target_vec = self.targets.to_vec();
        let grad_vec = grad_output.to_vec();
        let mut grad_input = vec![0.0f32; self.batch_size * self.num_classes];

        for b in 0..self.batch_size {
            let grad_scale = grad_vec[b];
            let offset = b * self.num_classes;
            let tc = target_vec[b] as usize;
            for c in 0..self.num_classes {
                let mut g = softmax_vec[offset + c];
                if c == tc {
                    g -= 1.0;
                }
                grad_input[offset + c] = g * grad_scale;
            }
        }

        let mut grad_tensor =
            Tensor::from_vec(grad_input, &[self.batch_size, self.num_classes]).unwrap();
        if grad_output.device().is_gpu() {
            grad_tensor = grad_tensor.to_device(grad_output.device()).unwrap();
        }
        vec![Some(grad_tensor)]
    }

    fn name(&self) -> &'static str {
        "CrossEntropyBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// CrossEntropyLoss
// =============================================================================

/// Cross entropy loss with log softmax.
///
/// This combines LogSoftmax and NLLLoss in a single class.
///
/// # Shape
/// - Input: (N, C) where C = number of classes
/// - Target: (N,) with class indices
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
}

impl CrossEntropyLoss {
    /// Creates a new CrossEntropyLoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates CrossEntropyLoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    ///
    /// # Arguments
    /// * `input` - Logits of shape (N, C)
    /// * `target` - Class indices of shape (N,) as f32 (will be cast to usize)
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let num_classes = shape[1];

        // GPU fast path: fused softmax + NLL loss kernel
        #[cfg(feature = "cuda")]
        if input_data.device().is_gpu() {
            // Ensure targets are on GPU
            let targets_gpu = if target_data.device().is_gpu() {
                target_data.clone()
            } else {
                target_data.to_device(input_data.device()).unwrap()
            };

            let (loss_tensor, softmax_tensor) = input_data.cross_entropy_fwd_cuda(&targets_gpu);

            let loss_var = if input.requires_grad() {
                let grad_fn = GradFn::new(CrossEntropyBackward {
                    next_fns: vec![input.grad_fn().cloned()],
                    softmax_probs: softmax_tensor,
                    targets: targets_gpu,
                    batch_size,
                    num_classes,
                });
                Variable::from_operation(loss_tensor, grad_fn, true)
            } else {
                Variable::new(loss_tensor, false)
            };

            return match self.reduction {
                Reduction::None => loss_var,
                Reduction::Mean => loss_var.mean(),
                Reduction::Sum => loss_var.sum(),
            };
        }

        // CPU path
        let input_vec = input_data.to_vec();
        let target_vec = target_data.to_vec();

        let mut losses = vec![0.0f32; batch_size];
        let mut softmax_probs_vec = vec![0.0f32; batch_size * num_classes];
        let mut target_classes = vec![0usize; batch_size];

        for b in 0..batch_size {
            let offset = b * num_classes;

            // Numerically stable log-softmax
            let max_val = (0..num_classes)
                .map(|c| input_vec[offset + c])
                .fold(f32::NEG_INFINITY, f32::max);

            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                let exp_val = (input_vec[offset + c] - max_val).exp();
                softmax_probs_vec[offset + c] = exp_val;
                sum_exp += exp_val;
            }

            for c in 0..num_classes {
                softmax_probs_vec[offset + c] /= sum_exp;
            }

            let log_sum_exp = max_val + sum_exp.ln();

            let tc = target_vec[b] as usize;
            target_classes[b] = tc;
            losses[b] = log_sum_exp - input_vec[offset + tc];
        }

        let loss_tensor = Tensor::from_vec(losses, &[batch_size]).unwrap();
        let softmax_tensor =
            Tensor::from_vec(softmax_probs_vec, &[batch_size, num_classes]).unwrap();
        let targets_f32: Vec<f32> = target_classes.iter().map(|&tc| tc as f32).collect();
        let targets_tensor = Tensor::from_vec(targets_f32, &[batch_size]).unwrap();

        let loss_var = if input.requires_grad() {
            let grad_fn = GradFn::new(CrossEntropyBackward {
                next_fns: vec![input.grad_fn().cloned()],
                softmax_probs: softmax_tensor,
                targets: targets_tensor,
                batch_size,
                num_classes,
            });
            Variable::from_operation(loss_tensor, grad_fn, true)
        } else {
            Variable::new(loss_tensor, false)
        };

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// NLLLoss
// =============================================================================

/// Negative Log Likelihood loss.
///
/// Expects input to be log-probabilities.
#[derive(Debug, Clone, Copy)]
pub struct NLLLoss {
    reduction: Reduction,
}

impl NLLLoss {
    /// Creates a new NLLLoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates NLLLoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let num_classes = shape[1];

        // NLL forward still needs per-sample gather (index into class dimension).
        // We pull target indices to CPU for the gather but keep input on device.
        let target_vec = target_data.to_vec();
        let input_vec = input_data.to_vec();

        let mut losses = vec![0.0f32; batch_size];
        for b in 0..batch_size {
            let tc = target_vec[b] as usize;
            losses[b] = -input_vec[b * num_classes + tc];
        }

        let mut loss_tensor = Tensor::from_vec(losses, &[batch_size]).unwrap();
        if input_data.device().is_gpu() {
            loss_tensor = loss_tensor.to_device(input_data.device()).unwrap();
        }

        let requires_grad = input.requires_grad() && is_grad_enabled();
        let loss_var = if requires_grad {
            let grad_fn = GradFn::new(NLLLossBackward {
                next_fns: vec![input.grad_fn().cloned()],
                target_tensor: target_data.clone(),
                batch_size,
                num_classes,
            });
            Variable::from_operation(loss_tensor, grad_fn, true)
        } else {
            Variable::new(loss_tensor, false)
        };

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for NLLLoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// NLLLossBackward
// =============================================================================

/// Gradient function for NLLLoss.
///
/// d/d(input)[b, c] = -1 if c == target[b], else 0
///
/// Stores targets as Tensor<f32> (GPU-resident when applicable).
/// The scatter in backward still uses CPU indexing since it's a sparse write,
/// but the result is moved to GPU if needed.
#[derive(Debug)]
struct NLLLossBackward {
    next_fns: Vec<Option<GradFn>>,
    target_tensor: Tensor<f32>,
    batch_size: usize,
    num_classes: usize,
}

impl GradientFunction for NLLLossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_out_vec = grad_output.to_vec();
        let target_vec = self.target_tensor.to_vec();
        let mut grad_input = vec![0.0f32; self.batch_size * self.num_classes];

        for b in 0..self.batch_size {
            let g = if grad_out_vec.len() == 1 {
                grad_out_vec[0]
            } else {
                grad_out_vec[b]
            };
            let tc = target_vec[b] as usize;
            grad_input[b * self.num_classes + tc] = -g;
        }

        let mut gi = Tensor::from_vec(grad_input, &[self.batch_size, self.num_classes]).unwrap();
        if grad_output.device().is_gpu() {
            gi = gi.to_device(grad_output.device()).unwrap();
        }
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "NLLLossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// BCELoss
// =============================================================================

/// Binary Cross Entropy loss.
///
/// Expects input to be probabilities in [0, 1].
#[derive(Debug, Clone, Copy)]
pub struct BCELoss {
    reduction: Reduction,
}

impl BCELoss {
    /// Creates a new BCELoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates BCELoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();

        // Clamp predictions to [eps, 1-eps] using Tensor ops
        let eps = 1e-7f32;
        let p_clamped = axonml_tensor::ops::clamp(&input_data, eps, 1.0 - eps);

        // loss = -(t * ln(p) + (1 - t) * ln(1 - p))
        let ln_p = p_clamped.ln();
        let one_minus_p = p_clamped.neg().add_scalar(1.0);
        let ln_one_minus_p = one_minus_p.ln();
        let one_minus_t = target_data.neg().add_scalar(1.0);

        // t * ln(p)
        let term1 = target_data.mul(&ln_p).unwrap();
        // (1-t) * ln(1-p)
        let term2 = one_minus_t.mul(&ln_one_minus_p).unwrap();
        // -(term1 + term2)
        let loss_tensor = term1.add(&term2).unwrap().neg();

        let requires_grad = input.requires_grad() && is_grad_enabled();
        let loss_var = if requires_grad {
            let grad_fn = GradFn::new(BCELossBackward {
                next_fns: vec![input.grad_fn().cloned()],
                input_tensor: input_data,
                target_tensor: target_data,
            });
            Variable::from_operation(loss_tensor, grad_fn, true)
        } else {
            Variable::new(loss_tensor, false)
        };

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for BCELoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BCELossBackward
// =============================================================================

/// Gradient function for BCELoss.
///
/// d/dp BCE = (p - y) / (p * (1 - p))
///
/// Stores input/target as Tensor<f32> (GPU-resident when applicable).
#[derive(Debug)]
struct BCELossBackward {
    next_fns: Vec<Option<GradFn>>,
    input_tensor: Tensor<f32>,
    target_tensor: Tensor<f32>,
}

impl GradientFunction for BCELossBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let eps = 1e-7f32;
        // p_clamped = clamp(input, eps, 1-eps)
        let p_clamped = axonml_tensor::ops::clamp(&self.input_tensor, eps, 1.0 - eps);
        // (p - y)
        let p_minus_y = p_clamped.sub(&self.target_tensor).unwrap();
        // p * (1 - p)
        let one_minus_p = p_clamped.neg().add_scalar(1.0);
        let denom = p_clamped.mul(&one_minus_p).unwrap();
        // grad = grad_output * (p - y) / (p * (1 - p))
        let ratio = p_minus_y.div(&denom).unwrap();
        let grad_tensor = grad_output.mul(&ratio).unwrap();
        vec![Some(grad_tensor)]
    }

    fn name(&self) -> &'static str {
        "BCELossBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// BCEWithLogitsBackward
// =============================================================================

/// Gradient function for BCEWithLogitsLoss.
///
/// The gradient of BCE w.r.t. input logits is: sigmoid(input) - target.
///
/// Stores input/target as Tensor<f32> (GPU-resident when applicable).
#[derive(Debug)]
struct BCEWithLogitsBackward {
    next_fns: Vec<Option<GradFn>>,
    input_tensor: Tensor<f32>,
    target_tensor: Tensor<f32>,
}

impl GradientFunction for BCEWithLogitsBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // sigmoid(input) - target, all via Tensor ops (auto-dispatch to GPU)
        let sig = self.input_tensor.sigmoid();
        let sig_minus_t = sig.sub(&self.target_tensor).unwrap();
        let grad_tensor = grad_output.mul(&sig_minus_t).unwrap();
        vec![Some(grad_tensor)]
    }

    fn name(&self) -> &'static str {
        "BCEWithLogitsBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// BCEWithLogitsLoss
// =============================================================================

/// Binary Cross Entropy with Logits.
///
/// Combines sigmoid and BCE in a numerically stable way.
#[derive(Debug, Clone, Copy)]
pub struct BCEWithLogitsLoss {
    reduction: Reduction,
}

impl BCEWithLogitsLoss {
    /// Creates a new BCEWithLogitsLoss with default reduction (Mean).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Creates BCEWithLogitsLoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();

        // Numerically stable: max(x, 0) - x*t + log(1 + exp(-|x|))
        // max(x, 0) = relu(x) = clamp_min(x, 0)
        let relu_x = axonml_tensor::ops::clamp_min(&input_data, 0.0);
        // x * t
        let x_times_t = input_data.mul(&target_data).unwrap();
        // |x| via clamp trick: max(x, 0) + max(-x, 0) = relu(x) + relu(-x)
        let neg_x = input_data.neg();
        let relu_neg_x = axonml_tensor::ops::clamp_min(&neg_x, 0.0);
        let abs_x = relu_x.add(&relu_neg_x).unwrap();
        // exp(-|x|)
        let exp_neg_abs = abs_x.neg().exp();
        // log(1 + exp(-|x|))
        let log_term = exp_neg_abs.add_scalar(1.0).ln();
        // loss = relu(x) - x*t + log(1 + exp(-|x|))
        let loss_tensor = relu_x.sub(&x_times_t).unwrap().add(&log_term).unwrap();

        let loss_var = if input.requires_grad() {
            let grad_fn = GradFn::new(BCEWithLogitsBackward {
                next_fns: vec![input.grad_fn().cloned()],
                input_tensor: input_data,
                target_tensor: target_data,
            });
            Variable::from_operation(loss_tensor, grad_fn, true)
        } else {
            Variable::new(loss_tensor, false)
        };

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SmoothL1Backward
// =============================================================================

/// Gradient function for SmoothL1Loss.
///
/// The gradient is: diff/beta if |diff| < beta, else sign(diff).
/// Returns gradients for both input and target (negated for target).
///
/// Stores diff as Tensor<f32> (GPU-resident when applicable).
#[derive(Debug)]
struct SmoothL1Backward {
    next_fns: Vec<Option<GradFn>>,
    diff_tensor: Tensor<f32>,
    beta: f32,
    shape: Vec<usize>,
}

impl GradientFunction for SmoothL1Backward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Compute |diff| = sqrt(diff^2 + eps) to stay differentiable and device-agnostic
        let eps = 1e-12f32;
        let diff_sq = self.diff_tensor.mul(&self.diff_tensor).unwrap();
        let diff_sq_eps = diff_sq.add_scalar(eps);
        let abs_diff = diff_sq_eps.ln().mul_scalar(0.5).exp();

        // sign(diff) = diff / |diff|
        let sign_diff = self.diff_tensor.div(&abs_diff).unwrap();

        // For the L2 region (|diff| < beta): grad = diff / beta
        // For the L1 region (|diff| >= beta): grad = sign(diff)
        // Blend: mask = clamp(|diff| / beta, 0, 1), but we need a hard cutoff.
        // Use a smooth approximation via where: if |diff| < beta -> diff/beta, else sign
        // Since we need element-wise branching and don't have a GPU where_cond,
        // we use: grad = (1 - mask) * (diff / beta) + mask * sign(diff)
        // where mask = clamp((|diff| - beta) * large_value, 0, 1) approximates step function.
        // Actually simpler: mask_l2 = clamp(1 - |diff|/beta, 0, 1) gives 1 in L2 region, 0 in L1
        // BUT this gives a soft transition. For exact correctness, use CPU branching on the mask.
        //
        // Practical approach: compute both branches with Tensor ops, build mask on CPU, blend.
        let grad_l2 = self.diff_tensor.mul_scalar(1.0 / self.beta); // diff / beta
        let grad_l1 = sign_diff; // sign(diff)

        // Build mask tensor: 1.0 where |d| < beta, 0.0 otherwise
        let abs_vec = abs_diff.to_vec();
        let beta = self.beta;
        let mask_vec: Vec<f32> = abs_vec
            .iter()
            .map(|&a| if a < beta { 1.0 } else { 0.0 })
            .collect();
        let mut mask = Tensor::from_vec(mask_vec, &self.shape).unwrap();
        if self.diff_tensor.device().is_gpu() {
            mask = mask.to_device(self.diff_tensor.device()).unwrap();
        }
        let inv_mask = mask.neg().add_scalar(1.0);

        // grad_per_elem = mask * grad_l2 + (1 - mask) * grad_l1
        let blended = mask
            .mul(&grad_l2)
            .unwrap()
            .add(&inv_mask.mul(&grad_l1).unwrap())
            .unwrap();

        // gi = blended * grad_output
        let gi = blended.mul(grad_output).unwrap();
        let gt = gi.neg();
        vec![Some(gi), Some(gt)]
    }

    fn name(&self) -> &'static str {
        "SmoothL1Backward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// SmoothL1Loss
// =============================================================================

/// Smooth L1 loss (Huber loss).
///
/// Uses L2 loss when |x| < beta, L1 loss otherwise.
#[derive(Debug, Clone, Copy)]
pub struct SmoothL1Loss {
    reduction: Reduction,
    beta: f32,
}

impl SmoothL1Loss {
    /// Creates a new SmoothL1Loss with default beta (1.0).
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            beta: 1.0,
        }
    }

    /// Creates SmoothL1Loss with specified beta.
    pub fn with_beta(beta: f32) -> Self {
        Self {
            reduction: Reduction::Mean,
            beta,
        }
    }

    /// Computes the loss.
    pub fn compute(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.data();
        let target_data = target.data();
        let diff_tensor = input_data.sub(&target_data).unwrap();
        let shape = diff_tensor.shape().to_vec();

        // Compute |diff| via relu(diff) + relu(-diff)
        let relu_diff = axonml_tensor::ops::clamp_min(&diff_tensor, 0.0);
        let relu_neg_diff = axonml_tensor::ops::clamp_min(&diff_tensor.neg(), 0.0);
        let abs_diff = relu_diff.add(&relu_neg_diff).unwrap();

        // L2 branch: 0.5 * diff^2 / beta
        let diff_sq = diff_tensor.mul(&diff_tensor).unwrap();
        let l2_loss = diff_sq.mul_scalar(0.5 / self.beta);

        // L1 branch: |diff| - 0.5 * beta
        let l1_loss = abs_diff.add_scalar(-0.5 * self.beta);

        // Build mask: 1.0 where |diff| < beta, 0.0 otherwise
        let abs_vec = abs_diff.to_vec();
        let beta = self.beta;
        let mask_vec: Vec<f32> = abs_vec
            .iter()
            .map(|&a| if a < beta { 1.0 } else { 0.0 })
            .collect();
        let mut mask = Tensor::from_vec(mask_vec, &shape).unwrap();
        if diff_tensor.device().is_gpu() {
            mask = mask.to_device(diff_tensor.device()).unwrap();
        }
        let inv_mask = mask.neg().add_scalar(1.0);

        // loss = mask * l2_loss + (1-mask) * l1_loss
        let loss_tensor = mask
            .mul(&l2_loss)
            .unwrap()
            .add(&inv_mask.mul(&l1_loss).unwrap())
            .unwrap();

        let loss_var = if input.requires_grad() || target.requires_grad() {
            let grad_fn = GradFn::new(SmoothL1Backward {
                next_fns: vec![input.grad_fn().cloned(), target.grad_fn().cloned()],
                diff_tensor,
                beta: self.beta,
                shape,
            });
            Variable::from_operation(loss_tensor, grad_fn, true)
        } else {
            Variable::new(loss_tensor, false)
        };

        match self.reduction {
            Reduction::None => loss_var,
            Reduction::Mean => loss_var.mean(),
            Reduction::Sum => loss_var.sum(),
        }
    }
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss::new();
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        assert!((loss.data().to_vec()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let loss_fn = MSELoss::new();
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        // Each diff is 1.0, squared is 1.0, mean is 1.0
        assert!((loss.data().to_vec()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap(),
            false,
        );
        let target = Variable::new(Tensor::from_vec(vec![2.0, 0.0], &[2]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        assert!(loss.data().to_vec()[0] > 0.0);
    }

    #[test]
    fn test_bce_loss() {
        let loss_fn = BCELoss::new();
        let input = Variable::new(Tensor::from_vec(vec![0.5, 0.5], &[2]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        // -[1*ln(0.5) + 0*ln(0.5)] - [0*ln(0.5) + 1*ln(0.5)] = -2*ln(0.5) / 2 = -ln(0.5) = 0.693
        assert!((loss.data().to_vec()[0] - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_cross_entropy_gradient_flow() {
        use axonml_autograd::backward;

        // Create input logits with requires_grad=true
        let input = Variable::new(
            Tensor::from_vec(vec![2.0, 1.0, 0.1, 0.5, 2.5, 0.3], &[2, 3]).unwrap(),
            true,
        );
        let target = Variable::new(Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap(), false);

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(&input, &target);

        // Loss should be positive
        let loss_val = loss.data().to_vec()[0];
        assert!(loss_val > 0.0, "Loss should be positive, got {}", loss_val);

        // Backward pass
        let ones = Tensor::from_vec(vec![1.0], &loss.shape()).unwrap();
        backward(&loss, &ones);

        // Input should have gradient
        let grad = input
            .grad()
            .expect("Input should have gradient after backward");
        let grad_vec = grad.to_vec();

        // Gradient should be non-zero
        let grad_norm: f32 = grad_vec.iter().map(|g| g * g).sum();
        assert!(
            grad_norm > 1e-10,
            "Gradient should be non-zero, got norm {}",
            grad_norm
        );

        // Gradient shape should match input shape
        assert_eq!(grad.shape(), &[2, 3]);

        // For the correct class, gradient should be negative (softmax - 1 < 0)
        // Sample 0, class 0 (target): grad should be (softmax[0,0] - 1) / 2
        assert!(
            grad_vec[0] < 0.0,
            "Gradient for correct class should be negative"
        );
        // Sample 1, class 1 (target): grad should be (softmax[1,1] - 1) / 2
        assert!(
            grad_vec[4] < 0.0,
            "Gradient for correct class should be negative"
        );

        // For wrong classes, gradient should be positive (softmax > 0)
        assert!(
            grad_vec[1] > 0.0,
            "Gradient for wrong class should be positive"
        );
        assert!(
            grad_vec[2] > 0.0,
            "Gradient for wrong class should be positive"
        );
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        // When logits strongly favor the correct class, loss should be near zero
        let loss_fn = CrossEntropyLoss::new();
        let input = Variable::new(
            Tensor::from_vec(vec![10.0, -10.0, -10.0], &[1, 3]).unwrap(),
            false,
        );
        let target = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        assert!(
            loss.data().to_vec()[0] < 0.001,
            "Perfect prediction should have near-zero loss"
        );
    }

    #[test]
    fn test_cross_entropy_uniform_prediction() {
        // When logits are all equal, loss should be ln(num_classes)
        let loss_fn = CrossEntropyLoss::new();
        let num_classes = 16;
        let input = Variable::new(
            Tensor::from_vec(vec![0.0; num_classes], &[1, num_classes]).unwrap(),
            false,
        );
        let target = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        let loss = loss_fn.compute(&input, &target);
        let expected = (num_classes as f32).ln(); // ln(16) ≈ 2.77
        let actual = loss.data().to_vec()[0];
        assert!(
            (actual - expected).abs() < 0.01,
            "Uniform logits should give ln(C)={}, got {}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_bce_with_logits_gradient_flow() {
        use axonml_autograd::backward;

        let input = Variable::new(
            Tensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[4]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4]).unwrap(),
            false,
        );

        let loss_fn = BCEWithLogitsLoss::new();
        let loss = loss_fn.compute(&input, &target);
        assert!(loss.data().to_vec()[0] > 0.0);

        let ones = Tensor::from_vec(vec![1.0], &loss.shape()).unwrap();
        backward(&loss, &ones);

        let grad = input.grad().expect("Input should have gradient");
        let grad_vec = grad.to_vec();
        assert_eq!(grad_vec.len(), 4);

        // For target=1, grad = sigmoid(x) - 1 < 0
        assert!(grad_vec[0] < 0.0);
        // For target=0, grad = sigmoid(x) > 0
        assert!(grad_vec[1] > 0.0);
    }

    #[test]
    fn test_smooth_l1_gradient_flow() {
        use axonml_autograd::backward;

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 5.0], &[3]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![1.5, 1.5, 1.5], &[3]).unwrap(), false);

        let loss_fn = SmoothL1Loss::new();
        let loss = loss_fn.compute(&input, &target);
        assert!(loss.data().to_vec()[0] > 0.0);

        let ones = Tensor::from_vec(vec![1.0], &loss.shape()).unwrap();
        backward(&loss, &ones);

        let grad = input.grad().expect("Input should have gradient");
        let grad_vec = grad.to_vec();
        assert_eq!(grad_vec.len(), 3);
        // Gradients should be non-zero
        let grad_norm: f32 = grad_vec.iter().map(|g| g * g).sum();
        assert!(grad_norm > 1e-10);
    }
}
