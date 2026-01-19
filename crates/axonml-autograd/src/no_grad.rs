//! No-Grad Context - Disable Gradient Computation
//!
//! Provides context managers for temporarily disabling gradient computation.
//! This is essential for inference, evaluation, and parts of training that
//! don't need gradients (like updating running statistics in `BatchNorm`).
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::cell::Cell;

// =============================================================================
// Thread-Local Gradient State
// =============================================================================

thread_local! {
    /// Whether gradient computation is enabled for this thread.
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };

    /// Stack depth for nested no_grad contexts.
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Returns whether gradient computation is currently enabled.
#[must_use] pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(std::cell::Cell::get)
}

/// Sets whether gradient computation is enabled.
fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|g| g.set(enabled));
}

// =============================================================================
// NoGradGuard
// =============================================================================

/// RAII guard that disables gradient computation within its scope.
///
/// When the guard is dropped, gradient computation is restored to its
/// previous state. Guards can be nested safely.
///
/// # Example
/// ```rust,ignore
/// use axonml_autograd::NoGradGuard;
///
/// {
///     let _guard = NoGradGuard::new();
///     // Gradient computation is disabled here
///     let y = x.relu();  // No gradient tracking
/// }
/// // Gradient computation is re-enabled here
/// ```
pub struct NoGradGuard {
    prev_state: bool,
}

impl NoGradGuard {
    /// Creates a new `NoGradGuard`, disabling gradient computation.
    #[must_use] pub fn new() -> Self {
        let prev_state = is_grad_enabled();
        set_grad_enabled(false);
        NO_GRAD_DEPTH.with(|d| d.set(d.get() + 1));
        Self { prev_state }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD_DEPTH.with(|d| d.set(d.get() - 1));
        // Only restore if we're at the outermost guard
        if NO_GRAD_DEPTH.with(std::cell::Cell::get) == 0 {
            set_grad_enabled(self.prev_state);
        }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EnableGradGuard
// =============================================================================

/// RAII guard that enables gradient computation within its scope.
///
/// Useful for temporarily enabling gradients inside a `no_grad` context.
pub struct EnableGradGuard {
    prev_state: bool,
}

impl EnableGradGuard {
    /// Creates a new `EnableGradGuard`, enabling gradient computation.
    #[must_use] pub fn new() -> Self {
        let prev_state = is_grad_enabled();
        set_grad_enabled(true);
        Self { prev_state }
    }
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev_state);
    }
}

impl Default for EnableGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Executes a closure with gradient computation disabled.
///
/// # Example
/// ```rust,ignore
/// use axonml_autograd::no_grad;
///
/// let output = no_grad(|| {
///     model.forward(&input)
/// });
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

/// Executes a closure with gradient computation enabled.
///
/// Useful for temporarily enabling gradients inside a `no_grad` context.
pub fn enable_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = EnableGradGuard::new();
    f()
}

/// Context manager that sets gradient computation mode.
///
/// # Arguments
/// * `mode` - Whether to enable (true) or disable (false) gradients
pub fn set_grad_enabled_context<F, R>(mode: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = is_grad_enabled();
    set_grad_enabled(mode);
    let result = f();
    set_grad_enabled(prev);
    result
}

// =============================================================================
// Inference Mode (Optimized No-Grad)
// =============================================================================

thread_local! {
    /// Whether we're in inference mode (more aggressive optimization).
    static INFERENCE_MODE: Cell<bool> = const { Cell::new(false) };
}

/// Returns whether inference mode is currently enabled.
#[must_use] pub fn is_inference_mode() -> bool {
    INFERENCE_MODE.with(std::cell::Cell::get)
}

/// RAII guard for inference mode.
///
/// Inference mode is like `no_grad` but with additional optimizations.
/// Variables created in inference mode cannot later require gradients.
pub struct InferenceModeGuard {
    prev_grad_state: bool,
    prev_inference_state: bool,
}

impl InferenceModeGuard {
    /// Creates a new `InferenceModeGuard`.
    #[must_use] pub fn new() -> Self {
        let prev_grad_state = is_grad_enabled();
        let prev_inference_state = is_inference_mode();
        set_grad_enabled(false);
        INFERENCE_MODE.with(|i| i.set(true));
        Self {
            prev_grad_state,
            prev_inference_state,
        }
    }
}

impl Drop for InferenceModeGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev_grad_state);
        INFERENCE_MODE.with(|i| i.set(self.prev_inference_state));
    }
}

impl Default for InferenceModeGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Executes a closure in inference mode.
pub fn inference_mode<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = InferenceModeGuard::new();
    f()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_enabled_default() {
        // Reset state
        set_grad_enabled(true);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_guard() {
        set_grad_enabled(true);
        assert!(is_grad_enabled());

        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_nested_no_grad() {
        set_grad_enabled(true);

        {
            let _guard1 = NoGradGuard::new();
            assert!(!is_grad_enabled());

            {
                let _guard2 = NoGradGuard::new();
                assert!(!is_grad_enabled());
            }

            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_function() {
        set_grad_enabled(true);

        let result = no_grad(|| {
            assert!(!is_grad_enabled());
            42
        });

        assert_eq!(result, 42);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_enable_grad_inside_no_grad() {
        set_grad_enabled(true);

        no_grad(|| {
            assert!(!is_grad_enabled());

            enable_grad(|| {
                assert!(is_grad_enabled());
            });

            assert!(!is_grad_enabled());
        });
    }

    #[test]
    fn test_inference_mode() {
        set_grad_enabled(true);
        assert!(!is_inference_mode());

        inference_mode(|| {
            assert!(!is_grad_enabled());
            assert!(is_inference_mode());
        });

        assert!(is_grad_enabled());
        assert!(!is_inference_mode());
    }
}
