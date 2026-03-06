//! Dropout Layers - Regularization via Random Zeroing
//!
//! Randomly zeros elements during training to prevent overfitting.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_tensor::Tensor;
use rand::Rng;

use crate::module::Module;

// =============================================================================
// DropoutBackward
// =============================================================================

/// Gradient function for Dropout.
///
/// Applies the same mask used in the forward pass: gradient is scaled where
/// elements were kept, and zeroed where elements were dropped.
#[derive(Debug)]
struct DropoutBackward {
    next_fns: Vec<Option<GradFn>>,
    /// The mask applied during forward: 0.0 for dropped, scale for kept.
    mask: Vec<f32>,
    shape: Vec<usize>,
}

impl GradientFunction for DropoutBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU path: create mask tensor on CPU, move to GPU, multiply
        let mask_tensor = Tensor::from_vec(self.mask.clone(), &self.shape).unwrap();
        let result = if grad_output.device().is_gpu() {
            let mask_gpu = mask_tensor.to_device(grad_output.device()).unwrap();
            grad_output.mul(&mask_gpu).unwrap()
        } else {
            grad_output.mul(&mask_tensor).unwrap()
        };
        vec![Some(result)]
    }

    fn name(&self) -> &'static str {
        "DropoutBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Dropout
// =============================================================================

/// During training, randomly zeros some elements with probability p.
///
/// During evaluation, returns input unchanged.
///
/// # Arguments
/// * `p` - Probability of an element to be zeroed (default: 0.5)
pub struct Dropout {
    /// Dropout probability.
    p: f32,
    /// Whether in training mode.
    training: AtomicBool,
}

impl std::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout")
            .field("p", &self.p)
            .field("training", &self.training.load(Ordering::Relaxed))
            .finish()
    }
}

impl Dropout {
    /// Creates a new Dropout layer with the given probability.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1)"
        );
        Self {
            p,
            training: AtomicBool::new(true),
        }
    }

    /// Creates a Dropout layer with default probability (0.5).
    pub fn default_p() -> Self {
        Self::new(0.5)
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::default_p()
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Variable) -> Variable {
        if !self.training.load(Ordering::Relaxed) || self.p == 0.0 {
            return input.clone();
        }

        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        let numel = input_data.numel();
        let mut rng = rand::thread_rng();

        // Scale factor for inverted dropout
        let scale = 1.0 / (1.0 - self.p);

        // Build mask on CPU: 0.0 for dropped, scale for kept
        let mask: Vec<f32> = (0..numel)
            .map(|_| {
                if rng.gen::<f32>() < self.p {
                    0.0
                } else {
                    scale
                }
            })
            .collect();

        // Create mask tensor and move to input device — multiply on GPU, no input copy needed
        let mask_tensor = Tensor::from_vec(mask.clone(), &shape).unwrap();
        let output = if input_data.device().is_gpu() {
            let mask_gpu = mask_tensor.to_device(input_data.device()).unwrap();
            input_data.mul(&mask_gpu).unwrap()
        } else {
            input_data.mul(&mask_tensor).unwrap()
        };

        let requires_grad = input.requires_grad() && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(DropoutBackward {
                next_fns: vec![input.grad_fn().cloned()],
                mask,
                shape,
            });
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::from_tensor(output)
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    fn name(&self) -> &'static str {
        "Dropout"
    }
}

// =============================================================================
// Dropout2d
// =============================================================================

/// Randomly zeros entire channels during training.
///
/// Useful for spatial data like images.
///
/// # Shape
/// - Input: (N, C, H, W)
/// - Output: Same as input
pub struct Dropout2d {
    /// Dropout probability.
    p: f32,
    /// Whether in training mode.
    training: AtomicBool,
}

impl std::fmt::Debug for Dropout2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout2d")
            .field("p", &self.p)
            .field("training", &self.training.load(Ordering::Relaxed))
            .finish()
    }
}

impl Dropout2d {
    /// Creates a new Dropout2d layer.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1)"
        );
        Self {
            p,
            training: AtomicBool::new(true),
        }
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Variable) -> Variable {
        if !self.training.load(Ordering::Relaxed) || self.p == 0.0 {
            return input.clone();
        }

        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        let input_vec = input_data.to_vec();
        let total = input_vec.len();
        let mut mask = vec![0.0f32; total];
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);

        for b in 0..batch_size {
            for c in 0..channels {
                let keep = rng.gen::<f32>() >= self.p;
                let start = b * channels * spatial_size + c * spatial_size;
                if keep {
                    for i in 0..spatial_size {
                        mask[start + i] = scale;
                    }
                }
            }
        }

        let output_vec: Vec<f32> = input_vec
            .iter()
            .zip(mask.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        let output = Tensor::from_vec(output_vec, &shape).unwrap();
        let requires_grad = input.requires_grad() && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(DropoutBackward {
                next_fns: vec![input.grad_fn().cloned()],
                mask,
                shape,
            });
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::from_tensor(output)
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    fn name(&self) -> &'static str {
        "Dropout2d"
    }
}

// =============================================================================
// AlphaDropout
// =============================================================================

/// Alpha Dropout for Self-Normalizing Neural Networks (SNNs).
///
/// Preserves the mean and variance of inputs by using specific alpha values.
pub struct AlphaDropout {
    /// Dropout probability.
    p: f32,
    /// Whether in training mode.
    training: AtomicBool,
}

impl AlphaDropout {
    /// Creates a new AlphaDropout layer.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1)"
        );
        Self {
            p,
            training: AtomicBool::new(true),
        }
    }
}

impl Module for AlphaDropout {
    fn forward(&self, input: &Variable) -> Variable {
        if !self.training.load(Ordering::Relaxed) || self.p == 0.0 {
            return input.clone();
        }

        // SELU parameters
        const ALPHA: f32 = 1.673_263_2;
        const SCALE: f32 = 1.050_701;

        let alpha_p = -ALPHA * SCALE;
        let a = ((1.0 - self.p) * (1.0 + self.p * alpha_p.powi(2)))
            .sqrt()
            .recip();
        let b = -a * alpha_p * self.p;

        let input_data = input.data();
        let input_vec = input_data.to_vec();
        let shape = input_data.shape().to_vec();
        let mut rng = rand::thread_rng();

        // For alpha dropout, the mask stores the scale factor 'a' where kept
        // and 0.0 where dropped (the additive term b is a constant, no gradient)
        let mask: Vec<f32> = input_vec
            .iter()
            .map(|_| if rng.gen::<f32>() < self.p { 0.0 } else { a })
            .collect();

        let output_vec: Vec<f32> = input_vec
            .iter()
            .zip(mask.iter())
            .map(|(&x, &m)| {
                if m == 0.0 {
                    a * alpha_p + b
                } else {
                    m * x + b
                }
            })
            .collect();

        let output = Tensor::from_vec(output_vec, &shape).unwrap();
        let requires_grad = input.requires_grad() && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(DropoutBackward {
                next_fns: vec![input.grad_fn().cloned()],
                mask,
                shape,
            });
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::from_tensor(output)
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    fn name(&self) -> &'static str {
        "AlphaDropout"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_training() {
        let dropout = Dropout::new(0.5);
        let input = Variable::new(Tensor::from_vec(vec![1.0; 1000], &[1000]).unwrap(), false);
        let output = dropout.forward(&input);

        // Some values should be zero, some should be scaled
        let output_vec = output.data().to_vec();
        let num_zeros = output_vec.iter().filter(|&&x| x == 0.0).count();

        // With p=0.5, roughly half should be zero (with some variance)
        assert!(num_zeros > 300 && num_zeros < 700);
    }

    #[test]
    fn test_dropout_eval() {
        let mut dropout = Dropout::new(0.5);
        dropout.eval();

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let output = dropout.forward(&input);

        // In eval mode, output should equal input
        assert_eq!(output.data().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dropout_zero_probability() {
        let dropout = Dropout::new(0.0);
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let output = dropout.forward(&input);

        assert_eq!(output.data().to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
