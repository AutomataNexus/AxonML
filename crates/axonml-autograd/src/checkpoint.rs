//! Gradient Checkpointing - Memory-Efficient Training
//!
//! # File
//! `crates/axonml-autograd/src/checkpoint.rs`
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

use crate::grad_fn::{GradFn, GradientFunction};
use crate::no_grad::{enable_grad, no_grad};
use crate::Variable;
use axonml_tensor::Tensor;
use std::any::Any;
use std::sync::Arc;

// =============================================================================
// CheckpointBackward
// =============================================================================

/// Gradient function that recomputes the forward pass during backward.
///
/// Instead of storing all intermediate activations from the forward pass,
/// this stores only the function and input. During backward, it re-runs
/// the forward pass with gradients enabled to recompute activations,
/// then calls backward on the recomputed output.
struct CheckpointBackward {
    /// The function to re-run during backward
    func: Arc<dyn Fn(&Variable) -> Variable + Send + Sync>,
    /// The saved input (detached, to avoid circular references)
    saved_input: Variable,
    /// next_functions pointing to the input's grad_fn
    next_fns: Vec<Option<GradFn>>,
}

impl std::fmt::Debug for CheckpointBackward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBackward")
            .field("saved_input_shape", &self.saved_input.shape())
            .finish()
    }
}

impl GradientFunction for CheckpointBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Re-run the forward pass with gradients enabled
        let input_for_recompute =
            Variable::new(self.saved_input.data(), true);

        let recomputed_output = enable_grad(|| (self.func)(&input_for_recompute));

        // Now run backward on the recomputed output to get gradients
        recomputed_output.backward_with_grad(grad_output);

        // Extract the gradient that flowed to our recomputed input
        let input_grad = input_for_recompute.grad();

        vec![input_grad]
    }

    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Checkpoint Function
// =============================================================================

/// Checkpoints a computation to save memory during training.
///
/// During the forward pass, the function runs without saving intermediate
/// activations. During the backward pass, the forward computation is re-run
/// to recompute the necessary activations for gradient computation.
///
/// # Arguments
/// * `func` - The function to checkpoint. Should be deterministic.
/// * `input` - The input variable to the function.
///
/// # Returns
/// The output of the function, with gradient support if input requires grad.
///
/// # Notes
/// - The function must be deterministic for correct gradients
/// - RNG states should be saved/restored if using dropout, etc.
/// - There is a 2x compute overhead for checkpointed segments
pub fn checkpoint<F>(func: F, input: &Variable) -> Variable
where
    F: Fn(&Variable) -> Variable + Send + Sync + 'static,
{
    // Run forward pass without gradient tracking to avoid storing activations
    let output = no_grad(|| func(input));

    // If input doesn't require gradients, just return the output
    if !input.requires_grad() {
        return output;
    }

    // Save input data (detached) and the function for recomputation
    let func_arc: Arc<dyn Fn(&Variable) -> Variable + Send + Sync> = Arc::new(func);

    let next_fns = vec![input.grad_fn().cloned()];

    let grad_fn = GradFn::new(CheckpointBackward {
        func: func_arc,
        saved_input: Variable::new(input.data(), false), // detached copy
        next_fns,
    });

    Variable::from_operation(output.data(), grad_fn, true)
}

/// Checkpoints a sequential model by dividing it into segments.
///
/// This is useful for models with many repeated layers (like transformers)
/// where checkpointing every N layers provides good memory savings.
///
/// # Arguments
/// * `num_layers` - Total number of layers
/// * `segments` - Number of checkpoint segments (more segments = less memory, more compute)
/// * `input` - The input variable
/// * `layer_fn` - Function that runs layer i on an input
pub fn checkpoint_sequential<F>(
    num_layers: usize,
    segments: usize,
    input: &Variable,
    layer_fn: F,
) -> Variable
where
    F: Fn(usize, &Variable) -> Variable + Send + Sync + Clone + 'static,
{
    if segments == 0 || num_layers == 0 {
        return input.clone();
    }

    let segment_size = (num_layers + segments - 1) / segments;
    let mut x = input.clone();

    for seg in 0..segments {
        let start = seg * segment_size;
        let end = (start + segment_size).min(num_layers);

        if start >= num_layers {
            break;
        }

        let f = layer_fn.clone();

        x = checkpoint(
            move |inp| {
                let mut h = inp.clone();
                for i in start..end {
                    h = f(i, &h);
                }
                h
            },
            &x,
        );
    }

    x
}

// =============================================================================
// Checkpoint Utilities
// =============================================================================

/// Estimates memory savings from checkpointing.
///
/// # Arguments
/// * `num_layers` - Number of layers in the model
/// * `segments` - Number of checkpoint segments
/// * `activation_size_mb` - Approximate size of activations per layer in MB
///
/// # Returns
/// Tuple of (memory_without_checkpoint, memory_with_checkpoint) in MB
#[must_use]
pub fn estimate_memory_savings(
    num_layers: usize,
    segments: usize,
    activation_size_mb: f32,
) -> (f32, f32) {
    let without = num_layers as f32 * activation_size_mb;
    let with = if segments > 0 {
        (num_layers as f32 / segments as f32).ceil() * activation_size_mb
    } else {
        without
    };
    (without, with)
}

/// Suggests optimal number of segments based on available memory.
///
/// # Arguments
/// * `num_layers` - Number of layers in the model
/// * `activation_size_mb` - Approximate size of activations per layer in MB
/// * `available_memory_mb` - Available GPU memory in MB
///
/// # Returns
/// Suggested number of checkpoint segments
#[must_use]
pub fn suggest_segments(
    num_layers: usize,
    activation_size_mb: f32,
    available_memory_mb: f32,
) -> usize {
    let total_activation_memory = num_layers as f32 * activation_size_mb;

    if total_activation_memory <= available_memory_mb {
        // No checkpointing needed
        return 0;
    }

    // How many activations can we store?
    let storable_layers = (available_memory_mb / activation_size_mb).floor() as usize;

    if storable_layers == 0 {
        // Need to checkpoint every layer
        return num_layers;
    }

    // Number of segments = ceil(num_layers / storable_layers)
    (num_layers + storable_layers - 1) / storable_layers
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_checkpoint_basic() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            true,
        );

        let output = checkpoint(
            |x| {
                // Simple operation for testing
                x.clone()
            },
            &input,
        );

        assert_eq!(output.shape(), vec![2, 2]);
    }

    #[test]
    fn test_checkpoint_without_grad() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            false, // No gradient required
        );

        let output = checkpoint(|x| x.clone(), &input);

        assert!(!output.requires_grad());
    }

    #[test]
    fn test_checkpoint_gradient_flow() {
        // Test that gradients actually flow through checkpointed computation
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            true,
        );

        // Checkpoint a simple multiply-by-2 operation
        let output = checkpoint(
            |x| {
                let two = Variable::new(
                    Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[2, 2]).unwrap(),
                    false,
                );
                x.mul_var(&two)
            },
            &input,
        );

        // output = input * 2, so d(output)/d(input) = 2
        let grad = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();
        output.backward_with_grad(&grad);

        let input_grad = input.grad().expect("Input should have gradient");
        let grad_data = input_grad.to_vec();
        for &v in &grad_data {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "Expected gradient 2.0 but got {}",
                v
            );
        }
    }

    #[test]
    fn test_checkpoint_sequential_basic() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            true,
        );

        let output = checkpoint_sequential(4, 2, &input, |_layer_idx, x| {
            // Identity for testing
            x.clone()
        });

        assert_eq!(output.shape(), vec![2, 2]);
    }

    #[test]
    fn test_checkpoint_sequential_single_segment() {
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), true);

        let output = checkpoint_sequential(3, 1, &input, |_layer_idx, x| x.clone());

        assert_eq!(output.shape(), vec![2]);
    }

    #[test]
    fn test_checkpoint_sequential_zero_segments() {
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), true);

        let output = checkpoint_sequential(3, 0, &input, |_layer_idx, x| x.clone());

        // Should return input unchanged
        assert_eq!(output.shape(), vec![2]);
    }

    #[test]
    fn test_estimate_memory_savings() {
        let (without, with) = estimate_memory_savings(12, 4, 100.0);

        assert!((without - 1200.0).abs() < 1e-6);
        assert!((with - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_suggest_segments_no_checkpoint_needed() {
        let segments = suggest_segments(10, 100.0, 2000.0);
        assert_eq!(segments, 0);
    }

    #[test]
    fn test_suggest_segments_moderate() {
        // 12 layers * 100MB = 1200MB needed, 400MB available
        // Can store 4 layers, so need 3 segments
        let segments = suggest_segments(12, 100.0, 400.0);
        assert_eq!(segments, 3);
    }

    #[test]
    fn test_suggest_segments_extreme() {
        // Very limited memory
        let segments = suggest_segments(12, 100.0, 50.0);
        assert_eq!(segments, 12);
    }
}
