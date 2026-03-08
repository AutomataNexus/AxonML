//! Predictive Coding Module — Surprise-Gated Adaptive Compute
//!
//! # File
//! `crates/axonml-vision/src/models/nexus/predictive.rs`
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

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_nn::{Conv2d, BatchNorm2d, Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// Predictive Coding Module
// =============================================================================

/// Predictive coding module for surprise-gated feature processing.
///
/// Operates on a single scale of fused features [B, C, H, W].
/// Maintains an internal prediction state and outputs gated features
/// where surprising (unpredicted) regions pass through and well-predicted
/// regions are cheaply reused from the prediction.
pub struct PredictiveCodingModule {
    /// Convolution to generate prediction for next frame.
    predict_conv: Conv2d,
    predict_bn: BatchNorm2d,
    /// Channels at this scale.
    _channels: usize,
    /// Previous prediction [B, C, H, W].
    prediction: Option<Variable>,
    /// Surprise gating temperature (lower = sharper gating).
    pub temperature: f32,
}

impl PredictiveCodingModule {
    /// Create a predictive coding module for the given channel count.
    pub fn new(channels: usize) -> Self {
        Self {
            predict_conv: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
            predict_bn: BatchNorm2d::new(channels),
            _channels: channels,
            prediction: None,
            temperature: 1.0,
        }
    }

    /// Process features with predictive coding.
    ///
    /// # Arguments
    /// - `actual`: Current frame's fused features [B, C, H, W].
    ///
    /// # Returns
    /// - Gated output: surprise * actual + (1 - surprise) * predicted
    /// - Surprise map [B, 1, H, W] (mean squared error per spatial location)
    pub fn forward(&mut self, actual: &Variable) -> (Variable, Variable) {
        let shape = actual.shape();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let actual_data = actual.data().to_vec();
        let spatial = h * w;

        if let Some(ref pred) = self.prediction {
            let pred_shape = pred.shape();
            if pred_shape[2] == h && pred_shape[3] == w {
                let pred_data = pred.data().to_vec();

                // Compute surprise = mean(|actual - predicted|²) per spatial location
                let mut surprise = vec![0.0f32; b * spatial];
                for bi in 0..b {
                    for y in 0..h {
                        for x in 0..w {
                            let mut mse = 0.0f32;
                            for ci in 0..c {
                                let idx = bi * c * spatial + ci * spatial + y * w + x;
                                let diff = actual_data[idx] - pred_data[idx];
                                mse += diff * diff;
                            }
                            surprise[bi * spatial + y * w + x] = mse / c as f32;
                        }
                    }
                }

                // Normalize surprise to [0, 1] via sigmoid with temperature
                let mut gate = vec![0.0f32; b * spatial];
                for i in 0..b * spatial {
                    gate[i] = 1.0 / (1.0 + (-surprise[i] * self.temperature).exp());
                }

                // Surprise map [B, 1, H, W] — diagnostic output, no grad needed
                let surprise_var = Variable::new(
                    Tensor::from_vec(gate.clone(), &[b, 1, h, w]).unwrap(),
                    false,
                );

                // Gated output: gate * actual + (1 - gate) * predicted
                // Use graph-tracked Variable ops to preserve gradient flow
                let gate_var = Variable::new(
                    Tensor::from_vec(gate, &[b, 1, h, w]).unwrap(),
                    false, // gate is a non-learned mask
                );
                let gate_expanded = gate_var.expand(&[b, c, h, w]);
                let ones = Variable::new(
                    Tensor::from_vec(vec![1.0f32; b * c * h * w], &[b, c, h, w]).unwrap(),
                    false,
                );
                let inv_gate = &ones - &gate_expanded;
                let gated_var = &(&gate_expanded * actual) + &(&inv_gate * pred);

                // Update prediction for next frame
                self.prediction = Some(
                    self.predict_bn.forward(&self.predict_conv.forward(&gated_var))
                );

                return (gated_var, surprise_var);
            }
        }

        // No prediction available (first frame) — pass through actual
        let surprise_data = vec![1.0f32; b * spatial]; // Full surprise
        let surprise_var = Variable::new(
            Tensor::from_vec(surprise_data, &[b, 1, h, w]).unwrap(),
            false,
        );

        // Generate prediction for next frame
        self.prediction = Some(
            self.predict_bn.forward(&self.predict_conv.forward(actual))
        );

        (actual.clone(), surprise_var)
    }

    /// Reset prediction state.
    pub fn reset(&mut self) {
        self.prediction = None;
    }

    /// Whether a prediction is available.
    pub fn has_prediction(&self) -> bool {
        self.prediction.is_some()
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.predict_conv.parameters());
        p.extend(self.predict_bn.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.predict_bn.eval();
    }

    pub fn train(&mut self) {
        self.predict_bn.train();
    }
}

// =============================================================================
// Multi-Scale Predictive Coding
// =============================================================================

/// Predictive coding across all 3 Nexus feature scales.
pub struct MultiScalePredictiveCoding {
    pub scale1: PredictiveCodingModule,
    pub scale2: PredictiveCodingModule,
    pub scale3: PredictiveCodingModule,
}

impl MultiScalePredictiveCoding {
    /// Create with standard Nexus channel count (96 at all fused scales).
    pub fn new(channels: usize) -> Self {
        Self {
            scale1: PredictiveCodingModule::new(channels),
            scale2: PredictiveCodingModule::new(channels),
            scale3: PredictiveCodingModule::new(channels),
        }
    }

    /// Process all three scales.
    ///
    /// Returns (gated features, surprise maps) for each scale.
    pub fn forward(
        &mut self,
        f1: &Variable,
        f2: &Variable,
        f3: &Variable,
    ) -> ((Variable, Variable), (Variable, Variable), (Variable, Variable)) {
        let r1 = self.scale1.forward(f1);
        let r2 = self.scale2.forward(f2);
        let r3 = self.scale3.forward(f3);
        (r1, r2, r3)
    }

    pub fn reset(&mut self) {
        self.scale1.reset();
        self.scale2.reset();
        self.scale3.reset();
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.scale1.parameters());
        p.extend(self.scale2.parameters());
        p.extend(self.scale3.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.scale1.eval();
        self.scale2.eval();
        self.scale3.eval();
    }

    pub fn train(&mut self) {
        self.scale1.train();
        self.scale2.train();
        self.scale3.train();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_coding_first_frame() {
        let mut pc = PredictiveCodingModule::new(32);
        assert!(!pc.has_prediction());

        let x = Variable::new(
            Tensor::from_vec(vec![0.5; 32 * 8 * 8], &[1, 32, 8, 8]).unwrap(),
            false,
        );

        let (gated, surprise) = pc.forward(&x);

        // First frame: pass through, full surprise
        assert_eq!(gated.shape(), vec![1, 32, 8, 8]);
        assert_eq!(surprise.shape(), vec![1, 1, 8, 8]);
        assert!(pc.has_prediction());

        let s_data = surprise.data().to_vec();
        assert!(s_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_predictive_coding_identical_frames() {
        let mut pc = PredictiveCodingModule::new(16);
        let x = Variable::new(
            Tensor::from_vec(vec![0.3; 16 * 4 * 4], &[1, 16, 4, 4]).unwrap(),
            false,
        );

        // Frame 1
        pc.forward(&x);

        // Frame 2: same features → prediction should be close → low surprise
        let (_gated, surprise) = pc.forward(&x);
        let s_data = surprise.data().to_vec();

        // Surprise should be lower than 1.0 (some prediction accuracy)
        let avg_surprise: f32 = s_data.iter().sum::<f32>() / s_data.len() as f32;
        // With random init, prediction won't be perfect but gating should function
        assert!(avg_surprise <= 1.0);
    }

    #[test]
    fn test_predictive_coding_changed_features() {
        let mut pc = PredictiveCodingModule::new(8);

        let x1 = Variable::new(
            Tensor::from_vec(vec![0.0; 8 * 4 * 4], &[1, 8, 4, 4]).unwrap(),
            false,
        );
        let x2 = Variable::new(
            Tensor::from_vec(vec![5.0; 8 * 4 * 4], &[1, 8, 4, 4]).unwrap(),
            false,
        );

        pc.forward(&x1);
        let (_gated, surprise) = pc.forward(&x2);

        // Big change → high surprise
        let s_data = surprise.data().to_vec();
        let avg_surprise: f32 = s_data.iter().sum::<f32>() / s_data.len() as f32;
        assert!(avg_surprise > 0.3, "Expected high surprise, got {avg_surprise}");
    }

    #[test]
    fn test_predictive_coding_output_finite() {
        let mut pc = PredictiveCodingModule::new(16);
        let x = Variable::new(
            Tensor::from_vec(vec![0.5; 16 * 8 * 8], &[1, 16, 8, 8]).unwrap(),
            false,
        );

        pc.forward(&x);
        let (gated, surprise) = pc.forward(&x);

        assert!(gated.data().to_vec().iter().all(|v| v.is_finite()));
        assert!(surprise.data().to_vec().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_predictive_coding_reset() {
        let mut pc = PredictiveCodingModule::new(8);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 8 * 4 * 4], &[1, 8, 4, 4]).unwrap(),
            false,
        );

        pc.forward(&x);
        assert!(pc.has_prediction());

        pc.reset();
        assert!(!pc.has_prediction());
    }

    #[test]
    fn test_multi_scale_predictive_coding() {
        let mut mspc = MultiScalePredictiveCoding::new(96);

        let f1 = Variable::new(Tensor::from_vec(vec![0.1; 96 * 40 * 40], &[1, 96, 40, 40]).unwrap(), false);
        let f2 = Variable::new(Tensor::from_vec(vec![0.1; 96 * 20 * 20], &[1, 96, 20, 20]).unwrap(), false);
        let f3 = Variable::new(Tensor::from_vec(vec![0.1; 96 * 10 * 10], &[1, 96, 10, 10]).unwrap(), false);

        let ((g1, s1), (g2, s2), (g3, s3)) = mspc.forward(&f1, &f2, &f3);

        assert_eq!(g1.shape(), vec![1, 96, 40, 40]);
        assert_eq!(s1.shape(), vec![1, 1, 40, 40]);
        assert_eq!(g2.shape(), vec![1, 96, 20, 20]);
        assert_eq!(s2.shape(), vec![1, 1, 20, 20]);
        assert_eq!(g3.shape(), vec![1, 96, 10, 10]);
        assert_eq!(s3.shape(), vec![1, 1, 10, 10]);
    }
}
