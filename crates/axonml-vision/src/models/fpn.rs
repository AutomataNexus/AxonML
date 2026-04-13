//! Feature Pyramid Network (FPN)
//!
//! # File
//! `crates/axonml-vision/src/models/fpn.rs`
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

use axonml_autograd::Variable;
use axonml_nn::{Conv2d, Module, Parameter};

use crate::ops::{InterpolateMode, interpolate_var};

// =============================================================================
// FPN
// =============================================================================

/// Feature Pyramid Network.
///
/// Takes multi-scale features from a backbone (e.g., ResNet C2-C5)
/// and produces refined feature maps at each level (P2-P5).
///
/// # Architecture
///
/// ```text
/// C5 ──── 1x1 ──── P5
///           │ (upsample + add)
/// C4 ──── 1x1 ──── + ──── 3x3 ──── P4
///           │ (upsample + add)
/// C3 ──── 1x1 ──── + ──── 3x3 ──── P3
///           │ (upsample + add)
/// C2 ──── 1x1 ──── + ──── 3x3 ──── P2
/// ```
pub struct FPN {
    /// 1x1 lateral connections that reduce channel dims
    lateral_convs: Vec<Conv2d>,
    /// 3x3 smoothing convolutions on each output level
    smooth_convs: Vec<Conv2d>,
    /// Number of pyramid levels
    num_levels: usize,
    /// Output channel dimension
    out_channels: usize,
}

impl FPN {
    /// Create a new FPN.
    ///
    /// # Arguments
    /// - `in_channels`: Channel dimensions for each backbone level (e.g., `[256, 512, 1024, 2048]`)
    /// - `out_channels`: Output channels for all pyramid levels (e.g., 256)
    pub fn new(in_channels: &[usize], out_channels: usize) -> Self {
        let num_levels = in_channels.len();
        let mut lateral_convs = Vec::with_capacity(num_levels);
        let mut smooth_convs = Vec::with_capacity(num_levels);

        for &in_c in in_channels {
            // 1x1 lateral: reduce to out_channels
            lateral_convs.push(Conv2d::with_options(
                in_c,
                out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            ));
            // 3x3 smoothing: reduce aliasing from upsampling
            smooth_convs.push(Conv2d::with_options(
                out_channels,
                out_channels,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
            ));
        }

        Self {
            lateral_convs,
            smooth_convs,
            num_levels,
            out_channels,
        }
    }

    /// Forward pass through the FPN.
    ///
    /// # Arguments
    /// - `features`: Multi-scale backbone features, ordered from lowest resolution
    ///   (deepest) to highest resolution. E.g., `[C2, C3, C4, C5]` where C5 is smallest.
    ///
    /// # Returns
    /// Pyramid features `[P2, P3, P4, P5]` (same ordering as input).
    pub fn forward(&self, features: &[Variable]) -> Vec<Variable> {
        assert_eq!(features.len(), self.num_levels);

        // Step 1: Apply lateral (1x1) convolutions
        let mut laterals: Vec<Variable> = features
            .iter()
            .enumerate()
            .map(|(i, feat)| self.lateral_convs[i].forward(feat))
            .collect();

        // Step 2: Top-down pathway (from deepest to shallowest)
        for i in (0..self.num_levels - 1).rev() {
            let upper = &laterals[i + 1];
            let target_h = laterals[i].shape()[2];
            let target_w = laterals[i].shape()[3];

            // Upsample the deeper level to match spatial dimensions (graph-tracked)
            let upsampled = interpolate_var(upper, target_h, target_w, InterpolateMode::Nearest);

            // Add lateral + upsampled
            laterals[i] = laterals[i].add_var(&upsampled);
        }

        // Step 3: Apply 3x3 smoothing to each level
        laterals
            .iter()
            .enumerate()
            .map(|(i, lat)| self.smooth_convs[i].forward(lat))
            .collect()
    }

    /// Returns the output channel dimension.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

impl Module for FPN {
    fn forward(&self, _x: &Variable) -> Variable {
        panic!("FPN requires multi-scale input. Use FPN::forward(&[Variable]) instead.");
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for conv in &self.lateral_convs {
            params.extend(conv.parameters());
        }
        for conv in &self.smooth_convs {
            params.extend(conv.parameters());
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_fpn_creation() {
        let fpn = FPN::new(&[64, 128, 256, 512], 256);
        assert_eq!(fpn.num_levels, 4);
        assert_eq!(fpn.out_channels, 256);
        assert!(!fpn.parameters().is_empty());
    }

    #[test]
    fn test_fpn_forward() {
        let fpn = FPN::new(&[64, 128, 256, 512], 64);

        // Simulate backbone features at 4 scales
        let c2 = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 16 * 16], &[1, 64, 16, 16]).unwrap(),
            false,
        );
        let c3 = Variable::new(
            Tensor::from_vec(vec![0.1; 128 * 8 * 8], &[1, 128, 8, 8]).unwrap(),
            false,
        );
        let c4 = Variable::new(
            Tensor::from_vec(vec![0.1; 256 * 4 * 4], &[1, 256, 4, 4]).unwrap(),
            false,
        );
        let c5 = Variable::new(
            Tensor::from_vec(vec![0.1; 512 * 2 * 2], &[1, 512, 2, 2]).unwrap(),
            false,
        );

        let pyramid = fpn.forward(&[c2, c3, c4, c5]);
        assert_eq!(pyramid.len(), 4);

        // All outputs should have out_channels=64
        assert_eq!(pyramid[0].shape(), vec![1, 64, 16, 16]);
        assert_eq!(pyramid[1].shape(), vec![1, 64, 8, 8]);
        assert_eq!(pyramid[2].shape(), vec![1, 64, 4, 4]);
        assert_eq!(pyramid[3].shape(), vec![1, 64, 2, 2]);
    }

    #[test]
    fn test_fpn_parameter_count() {
        let fpn = FPN::new(&[256, 512, 1024, 2048], 256);
        let params = fpn.parameters();
        // 4 lateral (1x1) + 4 smooth (3x3) = 8 conv layers, each with weight + bias
        assert_eq!(params.len(), 16);
    }
}
