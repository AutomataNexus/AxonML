//! Exponential Moving Average (EMA) for Model Weights
//!
//! Maintains a shadow copy of model parameters updated as:
//!   ema_param = decay * ema_param + (1 - decay) * model_param
//!
//! Standard practice for YOLO/detection training — EMA weights typically
//! produce 1-2% mAP improvement at zero inference cost.
//!
//! @version 0.1.0

use axonml_nn::Parameter;
use axonml_tensor::Tensor;

// =============================================================================
// ModelEMA
// =============================================================================

/// Exponential Moving Average of model parameters.
///
/// # Usage
/// ```ignore
/// let model = Helios::nano(80);
/// let mut ema = ModelEMA::new(&model.parameters(), 0.9999);
///
/// for epoch in 0..epochs {
///     // ... training step ...
///     ema.update(&model.parameters());
/// }
///
/// // Apply EMA weights for evaluation
/// ema.apply_to(&model.parameters());
/// ```
pub struct ModelEMA {
    /// Shadow (EMA) parameter values.
    shadow: Vec<Vec<f32>>,
    /// Decay factor (typically 0.9999).
    decay: f32,
    /// Number of updates applied.
    num_updates: usize,
}

impl ModelEMA {
    /// Create EMA tracker from initial model parameters.
    ///
    /// - `params`: Model parameters (cloned as initial shadow).
    /// - `decay`: EMA decay factor (0.9999 recommended for detection).
    pub fn new(params: &[Parameter], decay: f32) -> Self {
        let shadow: Vec<Vec<f32>> = params
            .iter()
            .map(|p| p.variable().data().to_vec())
            .collect();

        Self {
            shadow,
            decay,
            num_updates: 0,
        }
    }

    /// Create with warmup decay (YOLOv8 style).
    ///
    /// Effective decay ramps up: `decay * (1 - exp(-updates / tau))`
    /// Default: decay=0.9999, tau=2000.
    pub fn with_warmup(params: &[Parameter]) -> Self {
        Self::new(params, 0.9999)
    }

    /// Update shadow parameters from current model parameters.
    pub fn update(&mut self, params: &[Parameter]) {
        self.num_updates += 1;

        // Warmup: effective decay increases over time
        let d = self.effective_decay();

        for (shadow, param) in self.shadow.iter_mut().zip(params.iter()) {
            let param_data = param.variable().data().to_vec();
            for (s, &p) in shadow.iter_mut().zip(param_data.iter()) {
                *s = d * *s + (1.0 - d) * p;
            }
        }
    }

    /// Apply EMA shadow weights to model parameters.
    ///
    /// Call this before evaluation/inference.
    pub fn apply_to(&self, params: &[Parameter]) {
        for (shadow, param) in self.shadow.iter().zip(params.iter()) {
            let tensor = Tensor::from_vec(shadow.clone(), param.data().shape()).unwrap();
            param.update_data(tensor);
        }
    }

    /// Store current model weights, apply EMA, run closure, restore original.
    ///
    /// Useful for evaluation within a training loop without permanently
    /// modifying model weights.
    pub fn apply_and_restore<F, R>(&self, params: &[Parameter], f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Save original weights
        let originals: Vec<Vec<f32>> = params
            .iter()
            .map(|p| p.data().to_vec())
            .collect();

        // Apply EMA
        self.apply_to(params);

        // Run closure
        let result = f();

        // Restore original weights
        for (orig, param) in originals.iter().zip(params.iter()) {
            let tensor = Tensor::from_vec(orig.clone(), param.data().shape()).unwrap();
            param.update_data(tensor);
        }

        result
    }

    /// Get current effective decay (includes warmup ramp).
    pub fn effective_decay(&self) -> f32 {
        // YOLOv8 warmup: decay * (1 - exp(-updates / 2000))
        let tau = 2000.0f32;
        self.decay * (1.0 - (-(self.num_updates as f32) / tau).exp())
    }

    /// Number of updates performed.
    pub fn num_updates(&self) -> usize {
        self.num_updates
    }

    /// Get shadow parameter values (for saving).
    pub fn shadow_params(&self) -> &[Vec<f32>] {
        &self.shadow
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    fn make_params() -> Vec<Parameter> {
        vec![
            Parameter::new(
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
                true,
            ),
            Parameter::new(
                Tensor::from_vec(vec![0.5, 0.5], &[2]).unwrap(),
                true,
            ),
        ]
    }

    #[test]
    fn test_ema_creation() {
        let params = make_params();
        let ema = ModelEMA::new(&params, 0.999);

        assert_eq!(ema.num_updates(), 0);
        assert_eq!(ema.shadow.len(), 2);
        assert_eq!(ema.shadow[0], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ema_update() {
        let params = make_params();
        let mut ema = ModelEMA::new(&params, 0.9);

        // Modify params
        params[0].update_data(
            Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap(),
        );

        ema.update(&params);
        assert_eq!(ema.num_updates(), 1);

        // Shadow should move toward new values
        // effective_decay at step 1 ≈ 0.9 * (1 - exp(-1/2000)) ≈ 0.9 * 0.0005 ≈ very small
        // So shadow ≈ mostly the new values
        for &v in &ema.shadow[0] {
            assert!(v > 1.0, "Shadow should move toward new values, got {v}");
        }
    }

    #[test]
    fn test_ema_apply_and_restore() {
        let params = make_params();
        let original_data = params[0].variable().data().to_vec();

        let mut ema = ModelEMA::new(&params, 0.5);

        // Modify params and update EMA multiple times
        for i in 0..100 {
            params[0].update_data(
                Tensor::from_vec(vec![10.0; 4], &[2, 2]).unwrap(),
            );
            ema.update(&params);
        }

        // Apply and restore
        let result = ema.apply_and_restore(&params, || {
            // During closure, params should have EMA values
            let data = params[0].variable().data().to_vec();
            assert!(data[0] > 5.0, "EMA values should be closer to 10.0");
            42
        });

        assert_eq!(result, 42);

        // After restore, params should be back to what they were before apply_and_restore
        let restored = params[0].variable().data().to_vec();
        assert_eq!(restored, vec![10.0; 4]); // last set_data value
    }

    #[test]
    fn test_effective_decay_warmup() {
        let params = make_params();
        let mut ema = ModelEMA::new(&params, 0.9999);

        // At step 0, effective decay should be ~0
        assert!(ema.effective_decay() < 0.01);

        // After many steps, should approach target decay
        ema.num_updates = 10000;
        let d = ema.effective_decay();
        assert!(d > 0.99, "After 10K steps, decay should be ~0.9999, got {d}");
    }
}
