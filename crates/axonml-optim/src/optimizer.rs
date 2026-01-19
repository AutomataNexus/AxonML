//! Optimizer Trait - Core Optimizer Interface
//!
//! Defines the trait that all optimizers implement.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

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
    #[must_use] pub fn new() -> Self {
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
