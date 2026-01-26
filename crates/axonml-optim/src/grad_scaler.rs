//! Gradient Scaler for Mixed Precision Training
//!
//! Provides gradient scaling to prevent underflow when using F16 gradients.
//! Essential for stable mixed precision training.
//!
//! # Example
//! ```rust,ignore
//! use axonml_optim::{Adam, GradScaler};
//! use axonml_autograd::amp::autocast;
//!
//! let mut optimizer = Adam::new(model.parameters(), 0.001);
//! let mut scaler = GradScaler::new();
//!
//! for (input, target) in dataloader {
//!     optimizer.zero_grad();
//!
//!     // Forward pass with autocast
//!     let output = autocast(DType::F16, || model.forward(&input));
//!     let loss = loss_fn.compute(&output, &target);
//!
//!     // Scale loss and backward
//!     let scaled_loss = loss * scaler.get_scale();
//!     scaled_loss.backward();
//!
//!     // Unscale gradients and step
//!     if scaler.step(&mut optimizer) {
//!         // Optimizer step was taken
//!     }
//!     scaler.update();
//! }
//! ```
//!
//! @version 0.1.0

// =============================================================================
// GradScaler
// =============================================================================

/// Gradient scaler for mixed precision training.
///
/// Scales the loss to prevent gradient underflow when using F16,
/// then unscales gradients before the optimizer step.
///
/// The scale is automatically adjusted based on whether gradients overflow.
#[derive(Debug, Clone)]
pub struct GradScaler {
    /// Current scale factor
    scale: f32,
    /// Factor to multiply scale by on successful steps
    growth_factor: f32,
    /// Factor to multiply scale by when overflow detected
    backoff_factor: f32,
    /// Number of successful steps before growing scale
    growth_interval: usize,
    /// Counter for successful steps since last growth
    growth_tracker: usize,
    /// Whether inf/nan was found in last unscale
    found_inf: bool,
    /// Whether the scaler is enabled
    enabled: bool,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl GradScaler {
    /// Creates a new gradient scaler with default settings.
    ///
    /// Default configuration:
    /// - Initial scale: 65536.0 (2^16)
    /// - Growth factor: 2.0
    /// - Backoff factor: 0.5
    /// - Growth interval: 2000 steps
    #[must_use]
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
            found_inf: false,
            enabled: true,
        }
    }

    /// Creates a gradient scaler with custom initial scale.
    #[must_use]
    pub fn with_scale(init_scale: f32) -> Self {
        Self {
            scale: init_scale,
            ..Self::new()
        }
    }

    /// Creates a gradient scaler with all custom settings.
    #[must_use]
    pub fn with_options(
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    ) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            growth_tracker: 0,
            found_inf: false,
            enabled: true,
        }
    }

    /// Builder: set growth factor
    #[must_use]
    pub fn growth_factor(mut self, factor: f32) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Builder: set backoff factor
    #[must_use]
    pub fn backoff_factor(mut self, factor: f32) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Builder: set growth interval
    #[must_use]
    pub fn growth_interval(mut self, interval: usize) -> Self {
        self.growth_interval = interval;
        self
    }

    /// Builder: set enabled state
    #[must_use]
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Returns the current scale factor.
    #[must_use]
    pub fn get_scale(&self) -> f32 {
        if self.enabled {
            self.scale
        } else {
            1.0
        }
    }

    /// Sets the scale factor.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Returns whether the scaler is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enables or disables the scaler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Scales a loss value for backward pass.
    ///
    /// Multiply the loss by this before calling backward().
    #[must_use]
    pub fn scale_loss(&self, loss: f32) -> f32 {
        if self.enabled {
            loss * self.scale
        } else {
            loss
        }
    }

    /// Unscales gradients in place and checks for inf/nan.
    ///
    /// Returns true if all gradients are finite, false if any overflow.
    pub fn unscale_grads(&mut self, grads: &mut [f32]) -> bool {
        if !self.enabled {
            self.found_inf = false;
            return true;
        }

        let inv_scale = 1.0 / self.scale;
        self.found_inf = false;

        for g in grads.iter_mut() {
            if g.is_infinite() || g.is_nan() {
                self.found_inf = true;
                // Don't return early - still need to unscale other grads
                // But mark that we found inf
            }
            *g *= inv_scale;
        }

        !self.found_inf
    }

    /// Checks a slice of gradients for inf/nan without modifying them.
    #[must_use]
    pub fn check_grads(&self, grads: &[f32]) -> bool {
        grads.iter().all(|g| g.is_finite())
    }

    /// Returns whether inf/nan was found in the last unscale operation.
    #[must_use]
    pub fn found_inf(&self) -> bool {
        self.found_inf
    }

    /// Marks that inf was found (for external gradient checking).
    pub fn set_found_inf(&mut self, found: bool) {
        self.found_inf = found;
    }

    /// Updates the scale factor based on overflow history.
    ///
    /// Call this after each optimizer step:
    /// - If overflow was detected, scale is reduced by backoff_factor
    /// - If no overflow for growth_interval steps, scale is increased by growth_factor
    pub fn update(&mut self) {
        if !self.enabled {
            return;
        }

        if self.found_inf {
            // Reduce scale on overflow
            self.scale *= self.backoff_factor;
            self.growth_tracker = 0;
            // Clamp to avoid too small scale
            self.scale = self.scale.max(1.0);
        } else {
            // Track successful steps
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                // Increase scale
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
                // Clamp to avoid overflow
                self.scale = self.scale.min(f32::MAX / 2.0);
            }
        }
    }

    /// Returns the current state for checkpointing.
    #[must_use]
    pub fn state_dict(&self) -> GradScalerState {
        GradScalerState {
            scale: self.scale,
            growth_tracker: self.growth_tracker,
        }
    }

    /// Loads state from a checkpoint.
    pub fn load_state_dict(&mut self, state: GradScalerState) {
        self.scale = state.scale;
        self.growth_tracker = state.growth_tracker;
    }
}

/// Serializable state for GradScaler checkpointing.
#[derive(Debug, Clone, Copy)]
pub struct GradScalerState {
    /// Current scale factor
    pub scale: f32,
    /// Growth tracker value
    pub growth_tracker: usize,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_scaler_creation() {
        let scaler = GradScaler::new();
        assert!((scaler.get_scale() - 65536.0).abs() < 1e-6);
        assert!(scaler.is_enabled());
        assert!(!scaler.found_inf());
    }

    #[test]
    fn test_grad_scaler_with_scale() {
        let scaler = GradScaler::with_scale(1024.0);
        assert!((scaler.get_scale() - 1024.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_loss() {
        let scaler = GradScaler::with_scale(100.0);
        let loss = 0.5;
        let scaled = scaler.scale_loss(loss);
        assert!((scaled - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_unscale_grads() {
        let mut scaler = GradScaler::with_scale(100.0);
        let mut grads = vec![100.0, 200.0, 300.0];

        let valid = scaler.unscale_grads(&mut grads);

        assert!(valid);
        assert!(!scaler.found_inf());
        assert!((grads[0] - 1.0).abs() < 1e-6);
        assert!((grads[1] - 2.0).abs() < 1e-6);
        assert!((grads[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_unscale_grads_with_inf() {
        let mut scaler = GradScaler::with_scale(100.0);
        let mut grads = vec![100.0, f32::INFINITY, 300.0];

        let valid = scaler.unscale_grads(&mut grads);

        assert!(!valid);
        assert!(scaler.found_inf());
    }

    #[test]
    fn test_unscale_grads_with_nan() {
        let mut scaler = GradScaler::with_scale(100.0);
        let mut grads = vec![100.0, f32::NAN, 300.0];

        let valid = scaler.unscale_grads(&mut grads);

        assert!(!valid);
        assert!(scaler.found_inf());
    }

    #[test]
    fn test_update_on_overflow() {
        let mut scaler = GradScaler::with_scale(1000.0);
        scaler.found_inf = true;

        scaler.update();

        assert!((scaler.get_scale() - 500.0).abs() < 1e-6);
        assert_eq!(scaler.growth_tracker, 0);
    }

    #[test]
    fn test_update_growth() {
        let mut scaler = GradScaler::with_options(100.0, 2.0, 0.5, 3);

        // Simulate 3 successful steps
        for _ in 0..3 {
            scaler.found_inf = false;
            scaler.update();
        }

        assert!((scaler.get_scale() - 200.0).abs() < 1e-6);
        assert_eq!(scaler.growth_tracker, 0);
    }

    #[test]
    fn test_disabled_scaler() {
        let mut scaler = GradScaler::new().enabled(false);

        assert!(!scaler.is_enabled());
        assert!((scaler.get_scale() - 1.0).abs() < 1e-6);
        assert!((scaler.scale_loss(0.5) - 0.5).abs() < 1e-6);

        let mut grads = vec![1.0, 2.0, 3.0];
        let valid = scaler.unscale_grads(&mut grads);
        assert!(valid);
        // Grads should be unchanged
        assert!((grads[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_state_dict() {
        let mut scaler = GradScaler::with_scale(500.0);
        scaler.growth_tracker = 10;

        let state = scaler.state_dict();
        assert!((state.scale - 500.0).abs() < 1e-6);
        assert_eq!(state.growth_tracker, 10);

        let mut new_scaler = GradScaler::new();
        new_scaler.load_state_dict(state);
        assert!((new_scaler.get_scale() - 500.0).abs() < 1e-6);
        assert_eq!(new_scaler.growth_tracker, 10);
    }

    #[test]
    fn test_builder_pattern() {
        let scaler = GradScaler::with_scale(1000.0)
            .growth_factor(3.0)
            .backoff_factor(0.25)
            .growth_interval(100);

        assert!((scaler.growth_factor - 3.0).abs() < 1e-6);
        assert!((scaler.backoff_factor - 0.25).abs() < 1e-6);
        assert_eq!(scaler.growth_interval, 100);
    }
}
