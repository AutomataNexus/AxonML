//! `Optimizer` trait — the core interface for all gradient-based optimizers.
//!
//! Requires `step()` (apply one update), `zero_grad()` (clear accumulated
//! gradients), `get_lr()` / `set_lr()` (learning rate access), and
//! `parameters()` (list of tracked Parameter refs). Also `clip_grad_norm`
//! utility for gradient clipping before the step.
//!
//! # File
//! `crates/axonml-optim/src/optimizer.rs`
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

// =============================================================================
// Optimizer Trait
// =============================================================================

/// Trait for all optimizers.
///
/// Optimizers update model parameters based on gradients.
pub trait Optimizer {
    /// Performs a single optimization step.
    ///
    /// Updates all parameters based on their gradients.
    fn step(&mut self);

    /// Zeros all parameter gradients.
    fn zero_grad(&mut self);

    /// Returns the current learning rate.
    fn get_lr(&self) -> f32;

    /// Sets the learning rate.
    fn set_lr(&mut self, lr: f32);

    /// Returns the parameters being optimized.
    fn parameters(&self) -> &[Parameter];

    /// Returns the number of parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().len()
    }
}

// =============================================================================
// Parameter State
// =============================================================================

/// State associated with a parameter during optimization.
///
/// Different optimizers store different state (e.g., momentum, variance).
#[derive(Debug, Clone)]
pub struct ParamState {
    /// First moment (momentum) - used by SGD with momentum, Adam
    pub momentum_buffer: Option<Vec<f32>>,
    /// Second moment (variance) - used by Adam, `RMSprop`
    pub exp_avg_sq: Option<Vec<f32>>,
    /// Max second moment - used by `AdaMax`
    pub max_exp_avg_sq: Option<Vec<f32>>,
    /// Step count for bias correction
    pub step: usize,
}

impl ParamState {
    /// Creates a new empty parameter state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            momentum_buffer: None,
            exp_avg_sq: None,
            max_exp_avg_sq: None,
            step: 0,
        }
    }

    /// Initializes momentum buffer with zeros.
    pub fn init_momentum(&mut self, size: usize) {
        self.momentum_buffer = Some(vec![0.0; size]);
    }

    /// Initializes exponential average squared buffer with zeros.
    pub fn init_exp_avg_sq(&mut self, size: usize) {
        self.exp_avg_sq = Some(vec![0.0; size]);
    }
}

impl Default for ParamState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

// ── gradient clipping ──

/// Global-norm gradient clipping. Returns the total L2 norm BEFORE clipping, so a caller can log or
/// gate on it. Scales every gradient in place by `max_norm / total_norm` when the norm exceeds
/// `max_norm`, which is the standard formulation: it preserves the gradient DIRECTION and only
/// bounds its magnitude.
///
/// The reduction stays on device — `Tensor::sum` and `Tensor::mul_scalar` both keep GPU tensors on
/// GPU, so this costs one small D2H per parameter for the partial sum rather than downloading whole
/// gradient buffers. A non-finite partial (inf/NaN from a diverging step) zeroes that gradient
/// instead of poisoning the whole update.
fn zeroed_like(g: &axonml_tensor::Tensor<f32>) -> axonml_tensor::Tensor<f32> {
    let z = axonml_tensor::Tensor::from_vec(vec![0.0f32; g.numel()], g.shape()).expect("zero grad");
    z.to_device(g.device()).unwrap_or(z)
}

/// Scales every gradient in place so their global L2 norm is at most `max_norm`; returns the
/// norm before clipping.
pub fn clip_grad_norm(params: &[Parameter], max_norm: f32) -> f32 {
    let mut total_sq = 0.0f64;
    let mut grads = Vec::with_capacity(params.len());
    for p in params {
        let Some(g) = p.grad() else { continue };
        let sq = g.mul(&g).map(|v| v.sum()).map(|v| v.to_vec());
        let part = match sq {
            Ok(v) => v.first().copied().unwrap_or(0.0),
            Err(_) => 0.0,
        };
        if part.is_finite() {
            total_sq += f64::from(part);
            grads.push((p, g));
        } else {
            p.set_grad(zeroed_like(&g));
        }
    }

    let total = total_sq.sqrt() as f32;
    if !total.is_finite() {
        for (p, g) in grads {
            p.set_grad(zeroed_like(&g));
        }
        return total;
    }
    if total > max_norm && total > 0.0 {
        let scale = max_norm / (total + 1e-6);
        for (p, g) in grads {
            p.set_grad(g.mul_scalar(scale));
        }
    }
    total
}

#[cfg(test)]
mod clip_tests {
    use super::*;
    use axonml_tensor::Tensor;

    fn param_with_grad(g: Vec<f32>) -> Parameter {
        let n = g.len();
        let p = Parameter::new(Tensor::from_vec(vec![0.0; n], &[n]).unwrap(), true);
        p.set_grad(Tensor::from_vec(g, &[n]).unwrap());
        p
    }

    #[test]
    fn returns_true_norm_and_leaves_small_grads_alone() {
        let p = param_with_grad(vec![3.0, 4.0]);
        let n = clip_grad_norm(std::slice::from_ref(&p), 100.0);
        assert!((n - 5.0).abs() < 1e-5, "norm {n}");
        assert_eq!(p.grad().unwrap().to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn scales_to_max_norm_preserving_direction() {
        let p = param_with_grad(vec![3.0, 4.0]);
        let n = clip_grad_norm(std::slice::from_ref(&p), 1.0);
        assert!((n - 5.0).abs() < 1e-5);
        let g = p.grad().unwrap().to_vec();
        let after = (g[0] * g[0] + g[1] * g[1]).sqrt();
        assert!((after - 1.0).abs() < 1e-4, "clipped norm {after}");
        assert!(
            (g[0] / g[1] - 0.75).abs() < 1e-5,
            "direction changed: {g:?}"
        );
    }

    #[test]
    fn norm_is_global_across_parameters() {
        let a = param_with_grad(vec![3.0]);
        let b = param_with_grad(vec![4.0]);
        let n = clip_grad_norm(&[a.clone(), b.clone()], 5.0);
        assert!((n - 5.0).abs() < 1e-5, "global norm {n}");
        assert_eq!(a.grad().unwrap().to_vec(), vec![3.0]);
    }

    #[test]
    fn non_finite_gradient_is_zeroed_not_propagated() {
        let good = param_with_grad(vec![1.0]);
        let bad = param_with_grad(vec![f32::INFINITY]);
        let _ = clip_grad_norm(&[good.clone(), bad.clone()], 1.0);
        assert_eq!(bad.grad().unwrap().to_vec(), vec![0.0]);
        assert!(good.grad().unwrap().to_vec()[0].is_finite());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_state_creation() {
        let mut state = ParamState::new();
        assert!(state.momentum_buffer.is_none());
        assert!(state.exp_avg_sq.is_none());
        assert_eq!(state.step, 0);

        state.init_momentum(10);
        assert!(state.momentum_buffer.is_some());
        assert_eq!(state.momentum_buffer.as_ref().unwrap().len(), 10);
    }
}
