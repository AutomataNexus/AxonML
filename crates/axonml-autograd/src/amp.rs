//! Automatic Mixed Precision (AMP) Support
//!
//! Provides autocast contexts and utilities for mixed precision training.
//! Autocast enables automatic dtype casting for better performance on modern GPUs.
//!
//! # Example
//! ```rust,ignore
//! use axonml_autograd::amp::{autocast, is_autocast_enabled};
//! use axonml_core::DType;
//!
//! // Run forward pass with autocast
//! let output = autocast(DType::F16, || {
//!     model.forward(&input)
//! });
//! ```
//!
//! @version 0.1.0

use std::cell::Cell;

use axonml_core::DType;

// =============================================================================
// Thread-Local Autocast State
// =============================================================================

thread_local! {
    /// Whether autocast is currently enabled
    static AUTOCAST_ENABLED: Cell<bool> = const { Cell::new(false) };
    /// The dtype to cast to when autocast is enabled
    static AUTOCAST_DTYPE: Cell<DType> = const { Cell::new(DType::F16) };
    /// Nesting depth for autocast contexts
    static AUTOCAST_DEPTH: Cell<usize> = const { Cell::new(0) };
}

// =============================================================================
// Autocast Query Functions
// =============================================================================

/// Returns whether autocast is currently enabled.
///
/// This can be used to conditionally cast tensors within operations.
#[must_use]
pub fn is_autocast_enabled() -> bool {
    AUTOCAST_ENABLED.with(Cell::get)
}

/// Returns the current autocast dtype.
///
/// Only meaningful when `is_autocast_enabled()` returns true.
#[must_use]
pub fn autocast_dtype() -> DType {
    AUTOCAST_DTYPE.with(Cell::get)
}

/// Returns the current autocast nesting depth.
#[must_use]
pub fn autocast_depth() -> usize {
    AUTOCAST_DEPTH.with(Cell::get)
}

// =============================================================================
// Autocast Guard
// =============================================================================

/// RAII guard for autocast context.
///
/// When created, enables autocast with the specified dtype.
/// When dropped, restores the previous autocast state.
pub struct AutocastGuard {
    prev_enabled: bool,
    prev_dtype: DType,
}

impl AutocastGuard {
    /// Creates a new autocast guard with the specified dtype.
    ///
    /// Enables autocast and increments the nesting depth.
    pub fn new(dtype: DType) -> Self {
        let prev_enabled = is_autocast_enabled();
        let prev_dtype = autocast_dtype();

        AUTOCAST_ENABLED.with(|e| e.set(true));
        AUTOCAST_DTYPE.with(|d| d.set(dtype));
        AUTOCAST_DEPTH.with(|d| d.set(d.get() + 1));

        Self {
            prev_enabled,
            prev_dtype,
        }
    }

    /// Creates a disabled autocast guard.
    ///
    /// Temporarily disables autocast within a nested context.
    pub fn disabled() -> Self {
        let prev_enabled = is_autocast_enabled();
        let prev_dtype = autocast_dtype();

        AUTOCAST_ENABLED.with(|e| e.set(false));

        Self {
            prev_enabled,
            prev_dtype,
        }
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST_ENABLED.with(|e| e.set(self.prev_enabled));
        AUTOCAST_DTYPE.with(|d| d.set(self.prev_dtype));
        AUTOCAST_DEPTH.with(|d| {
            let depth = d.get();
            if depth > 0 {
                d.set(depth - 1);
            }
        });
    }
}

// =============================================================================
// Autocast Function
// =============================================================================

/// Runs a function with autocast enabled.
///
/// Operations within the closure may automatically use the specified dtype
/// for better performance, particularly on GPUs with tensor cores.
///
/// # Arguments
/// * `dtype` - The dtype to use for autocasted operations (typically F16 or BF16)
/// * `f` - The function to run with autocast enabled
///
/// # Example
/// ```rust,ignore
/// use axonml_autograd::amp::autocast;
/// use axonml_core::DType;
///
/// let output = autocast(DType::F16, || {
///     // Operations here may use F16 for compute
///     model.forward(&input)
/// });
/// ```
pub fn autocast<F, R>(dtype: DType, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = AutocastGuard::new(dtype);
    f()
}

/// Runs a function with autocast disabled.
///
/// Useful for operations that should always run in full precision,
/// such as loss computation or certain reductions.
///
/// # Example
/// ```rust,ignore
/// use axonml_autograd::amp::{autocast, disable_autocast};
/// use axonml_core::DType;
///
/// autocast(DType::F16, || {
///     let features = model.forward(&input);
///
///     // Compute loss in full precision
///     disable_autocast(|| {
///         loss_fn.compute(&features, &target)
///     })
/// });
/// ```
pub fn disable_autocast<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = AutocastGuard::disabled();
    f()
}

// =============================================================================
// Autocast Policy
// =============================================================================

/// Policy for which operations should be autocasted.
///
/// Different operations benefit from different precision levels:
/// - Matrix multiplications: F16 for speed
/// - Normalization: F32 for accuracy
/// - Loss computation: F32 for stability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocastPolicy {
    /// Use lower precision (e.g., F16) - good for matmul, conv
    LowerPrecision,
    /// Keep full precision (F32) - good for normalization, softmax
    FullPrecision,
    /// Compute in lower precision but accumulate in F32
    MixedAccumulate,
}

impl AutocastPolicy {
    /// Returns the recommended policy for matrix multiplication.
    #[must_use]
    pub const fn for_matmul() -> Self {
        Self::LowerPrecision
    }

    /// Returns the recommended policy for convolution.
    #[must_use]
    pub const fn for_conv() -> Self {
        Self::LowerPrecision
    }

    /// Returns the recommended policy for batch normalization.
    #[must_use]
    pub const fn for_batchnorm() -> Self {
        Self::FullPrecision
    }

    /// Returns the recommended policy for layer normalization.
    #[must_use]
    pub const fn for_layernorm() -> Self {
        Self::FullPrecision
    }

    /// Returns the recommended policy for softmax.
    #[must_use]
    pub const fn for_softmax() -> Self {
        Self::FullPrecision
    }

    /// Returns the recommended policy for loss functions.
    #[must_use]
    pub const fn for_loss() -> Self {
        Self::FullPrecision
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocast_disabled_by_default() {
        assert!(!is_autocast_enabled());
        assert_eq!(autocast_depth(), 0);
    }

    #[test]
    fn test_autocast_guard() {
        assert!(!is_autocast_enabled());

        {
            let _guard = AutocastGuard::new(DType::F16);
            assert!(is_autocast_enabled());
            assert_eq!(autocast_dtype(), DType::F16);
            assert_eq!(autocast_depth(), 1);
        }

        assert!(!is_autocast_enabled());
        assert_eq!(autocast_depth(), 0);
    }

    #[test]
    fn test_autocast_function() {
        assert!(!is_autocast_enabled());

        let result = autocast(DType::F16, || {
            assert!(is_autocast_enabled());
            assert_eq!(autocast_dtype(), DType::F16);
            42
        });

        assert_eq!(result, 42);
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_nested_autocast() {
        autocast(DType::F16, || {
            assert_eq!(autocast_depth(), 1);
            assert_eq!(autocast_dtype(), DType::F16);

            autocast(DType::F32, || {
                assert_eq!(autocast_depth(), 2);
                assert_eq!(autocast_dtype(), DType::F32);
            });

            assert_eq!(autocast_depth(), 1);
            assert_eq!(autocast_dtype(), DType::F16);
        });

        assert_eq!(autocast_depth(), 0);
    }

    #[test]
    fn test_disable_autocast() {
        autocast(DType::F16, || {
            assert!(is_autocast_enabled());

            disable_autocast(|| {
                assert!(!is_autocast_enabled());
            });

            assert!(is_autocast_enabled());
        });
    }

    #[test]
    fn test_autocast_policy() {
        assert_eq!(AutocastPolicy::for_matmul(), AutocastPolicy::LowerPrecision);
        assert_eq!(AutocastPolicy::for_conv(), AutocastPolicy::LowerPrecision);
        assert_eq!(
            AutocastPolicy::for_batchnorm(),
            AutocastPolicy::FullPrecision
        );
        assert_eq!(AutocastPolicy::for_softmax(), AutocastPolicy::FullPrecision);
        assert_eq!(AutocastPolicy::for_loss(), AutocastPolicy::FullPrecision);
    }
}
