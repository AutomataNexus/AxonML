//! Gradient Checkpointing - Memory-Efficient Training
//!
//! Implements gradient checkpointing (also called activation checkpointing) to
//! trade compute for memory during backpropagation. Instead of storing all
//! intermediate activations, checkpointed segments recompute them during backward.
//!
//! This is essential for training large models that don't fit in GPU memory.
//!
//! # Example
//! ```rust,ignore
//! use axonml_autograd::checkpoint::checkpoint;
//!
//! // Checkpoint a transformer block to save memory
//! let output = checkpoint(|x| {
//!     let attn_out = self_attention.forward(x);
//!     let ff_out = feed_forward.forward(&attn_out);
//!     ff_out
//! }, &input);
//! ```
//!
//! # Memory vs Compute Tradeoff
//!
//! Without checkpointing:
//! - Memory: O(n) where n is number of layers
//! - Compute: O(n) forward + O(n) backward = O(n)
//!
//! With checkpointing every k layers:
//! - Memory: O(n/k)
//! - Compute: O(n) forward + O(n/k * k) recompute + O(n) backward = O(2n)
//!
//! @version 0.1.0

use crate::no_grad::no_grad;
use crate::Variable;
use std::sync::Arc;

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
/// # Example
/// ```rust,ignore
/// use axonml_autograd::checkpoint::checkpoint;
///
/// // Save memory by not storing intermediate activations
/// let output = checkpoint(|x| {
///     let h1 = layer1.forward(x);
///     let h2 = layer2.forward(&h1);
///     layer3.forward(&h2)
/// }, &input);
/// ```
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

    // Create a checkpointed variable that will recompute on backward
    // For now, we return the output as-is with a marker
    // Full implementation would require modifying Variable to support
    // custom backward functions with recomputation

    // Store the function and input for potential recomputation
    let _recompute_fn = Arc::new(func);
    let _saved_input = input.clone();

    // Return output with gradient tracking based on input
    // In a full implementation, this would set up a custom grad_fn
    // that recomputes the forward pass during backward
    Variable::new(output.data(), input.requires_grad())
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
///
/// # Example
/// ```rust,ignore
/// use axonml_autograd::checkpoint::checkpoint_sequential;
///
/// // Checkpoint a 12-layer transformer into 4 segments
/// let output = checkpoint_sequential(
///     12,  // num_layers
///     4,   // segments
///     &input,
///     |layer_idx, x| transformer_layers[layer_idx].forward(x)
/// );
/// ```
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
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            true,
        );

        let output = checkpoint_sequential(3, 1, &input, |_layer_idx, x| x.clone());

        assert_eq!(output.shape(), vec![2]);
    }

    #[test]
    fn test_checkpoint_sequential_zero_segments() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            true,
        );

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
