//! Gradient Function Traits - Differentiable Operation Interface
//!
//! Defines the interface for gradient functions that compute derivatives
//! during the backward pass. Each differentiable operation implements
//! `GradientFunction` to specify how gradients flow backward.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use axonml_tensor::Tensor;
use parking_lot::RwLock;

// =============================================================================
// Gradient Function Trait
// =============================================================================

/// Trait for gradient computation functions.
///
/// Each differentiable operation creates a `GradFn` that knows how to compute
/// the gradient with respect to its inputs given the gradient of its output.
pub trait GradientFunction: Debug + Send + Sync {
    /// Computes gradients with respect to inputs.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of the loss with respect to this operation's output
    ///
    /// # Returns
    /// Vector of gradients, one for each input that requires grad.
    /// None entries indicate inputs that don't require gradients.
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>>;

    /// Returns the name of this gradient function for debugging.
    fn name(&self) -> &'static str;

    /// Returns references to the next functions in the backward graph.
    fn next_functions(&self) -> &[Option<GradFn>];

    /// Allows downcasting to concrete types.
    fn as_any(&self) -> &dyn Any;
}

// =============================================================================
// GradFn - Arc Wrapper
// =============================================================================

/// Unique identifier for a `GradFn` that survives cloning.
///
/// Since `GradFn` wraps an Arc, we use the Arc's pointer as a stable ID.
pub type GradFnId = usize;

/// Reference-counted gradient function.
///
/// Wraps a `GradientFunction` in an Arc for efficient sharing in the
/// computational graph.
#[derive(Clone)]
pub struct GradFn {
    inner: Arc<dyn GradientFunction>,
}

impl GradFn {
    /// Creates a new `GradFn` from a gradient function.
    pub fn new<F: GradientFunction + 'static>(func: F) -> Self {
        Self {
            inner: Arc::new(func),
        }
    }

    /// Applies the gradient function.
    #[must_use] pub fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        self.inner.apply(grad_output)
    }

    /// Returns the name of the gradient function.
    #[must_use] pub fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Returns the next functions in the graph.
    #[must_use] pub fn next_functions(&self) -> &[Option<GradFn>] {
        self.inner.next_functions()
    }

    /// Returns a stable ID for this `GradFn` that survives cloning.
    ///
    /// This uses the Arc's pointer address, which remains the same
    /// when the `GradFn` is cloned (since `Arc::clone` just increments
    /// the reference count).
    #[must_use] pub fn id(&self) -> GradFnId {
        // For trait objects, we need to extract just the data pointer (not vtable)
        let ptr = Arc::as_ptr(&self.inner);
        ptr.cast::<()>() as GradFnId
    }
}

impl Debug for GradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradFn({})", self.name())
    }
}

// =============================================================================
// Accumulate Grad - Leaf Node Gradient Function
// =============================================================================

/// Shared gradient accumulator for leaf variables.
pub type GradAccumulator = Arc<RwLock<Option<Tensor<f32>>>>;

/// Gradient function for leaf variables (accumulates gradients).
pub struct AccumulateGrad {
    /// Shared gradient storage with the Variable.
    grad_accumulator: GradAccumulator,
}

impl AccumulateGrad {
    /// Creates a new `AccumulateGrad` with a shared gradient accumulator.
    pub fn new(grad_accumulator: GradAccumulator) -> Self {
        Self { grad_accumulator }
    }

    /// Accumulates gradient into the shared storage.
    pub fn accumulate(&self, grad: &Tensor<f32>) {
        let mut guard = self.grad_accumulator.write();
        if let Some(ref existing) = *guard {
            *guard = Some(existing.add(grad).unwrap());
        } else {
            *guard = Some(grad.clone());
        }
    }
}

impl Debug for AccumulateGrad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccumulateGrad").finish()
    }
}

impl GradientFunction for AccumulateGrad {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Accumulate gradient directly into the shared storage
        self.accumulate(grad_output);
        // No gradients to propagate further (leaf node)
        vec![]
    }

    fn name(&self) -> &'static str {
        "AccumulateGrad"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        // Leaf node has no next functions
        &[]
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
    fn test_accumulate_grad() {
        let grad_acc: GradAccumulator = Arc::new(RwLock::new(None));
        let acc = AccumulateGrad::new(Arc::clone(&grad_acc));
        assert_eq!(acc.name(), "AccumulateGrad");
        assert!(acc.next_functions().is_empty());

        // Test accumulation
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        acc.accumulate(&grad);
        assert!(grad_acc.read().is_some());
        assert_eq!(
            grad_acc.read().as_ref().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0]
        );

        // Test accumulation again
        acc.accumulate(&grad);
        assert_eq!(
            grad_acc.read().as_ref().unwrap().to_vec(),
            vec![2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn test_grad_fn_wrapper() {
        let grad_acc: GradAccumulator = Arc::new(RwLock::new(None));
        let acc = AccumulateGrad::new(grad_acc);
        let grad_fn = GradFn::new(acc);
        assert_eq!(grad_fn.name(), "AccumulateGrad");
    }
}
