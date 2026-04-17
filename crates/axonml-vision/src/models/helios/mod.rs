//! Helios — High-Efficiency Lightweight Inference Object Sentinel
//!
//! Top-level module for the Helios object detector. Re-exports `Helios` (full
//! detection pipeline), `HeliosLoss`, `CIoULoss`, and `TaskAlignedAssigner`.
//! Defines `HeliosSize` (Nano/Small/Medium/Large/XLarge variants with width and
//! depth multipliers), `HeliosConfig` (num_classes, input_size, DFL reg_max,
//! score/NMS thresholds, strides, and computed stage channels/depths), and
//! training output types `HeliosScaleOutput` (per-scale cls logits + bbox DFL
//! distributions) and `HeliosTrainOutput` (3-scale P3/P4/P5 outputs).
//!
//! # File
//! `crates/axonml-vision/src/models/helios/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

pub mod backbone;
pub mod detector;
pub mod head;
pub mod loss;
pub mod neck;

pub use detector::Helios;
pub use loss::{CIoULoss, HeliosLoss, TaskAlignedAssigner};

use axonml_autograd::Variable;

// =============================================================================
// Model Size
// =============================================================================

/// Model size variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeliosSize {
    /// Nano — ~3M params, edge deployment.
    Nano,
    /// Small — ~9M params, mobile/edge.
    Small,
    /// Medium — ~23M params, balanced.
    Medium,
    /// Large — ~44M params, accuracy-focused.
    Large,
    /// XLarge — ~68M params, maximum accuracy.
    XLarge,
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Helios detector.
#[derive(Debug, Clone)]
pub struct HeliosConfig {
    /// Number of object classes.
    pub num_classes: usize,
    /// Square input size (default 640).
    pub input_size: usize,
    /// Model size variant.
    pub size: HeliosSize,
    /// Channel width multiplier.
    pub width_mul: f32,
    /// Block depth multiplier.
    pub depth_mul: f32,
    /// DFL regression max bins (default 16).
    pub reg_max: usize,
    /// Minimum detection score.
    pub score_threshold: f32,
    /// NMS IoU threshold.
    pub nms_threshold: f32,
    /// Feature map strides.
    pub strides: Vec<usize>,
}

impl HeliosConfig {
    /// Create nano configuration.
    pub fn nano(num_classes: usize) -> Self {
        Self {
            num_classes,
            input_size: 640,
            size: HeliosSize::Nano,
            width_mul: 0.25,
            depth_mul: 0.33,
            reg_max: 16,
            score_threshold: 0.25,
            nms_threshold: 0.45,
            strides: vec![8, 16, 32],
        }
    }

    /// Create small configuration.
    pub fn small(num_classes: usize) -> Self {
        Self {
            size: HeliosSize::Small,
            width_mul: 0.50,
            depth_mul: 0.33,
            ..Self::nano(num_classes)
        }
    }

    /// Create medium configuration.
    pub fn medium(num_classes: usize) -> Self {
        Self {
            size: HeliosSize::Medium,
            width_mul: 0.75,
            depth_mul: 0.67,
            ..Self::nano(num_classes)
        }
    }

    /// Create large configuration.
    pub fn large(num_classes: usize) -> Self {
        Self {
            size: HeliosSize::Large,
            width_mul: 1.00,
            depth_mul: 1.00,
            ..Self::nano(num_classes)
        }
    }

    /// Create xlarge configuration.
    pub fn xlarge(num_classes: usize) -> Self {
        Self {
            size: HeliosSize::XLarge,
            width_mul: 1.25,
            depth_mul: 1.00,
            ..Self::nano(num_classes)
        }
    }

    /// Compute channel width for each backbone stage, rounded to multiples of 8.
    /// Base channels: [64, 128, 256, 512, 1024].
    pub fn stage_channels(&self) -> [usize; 5] {
        let base = [64, 128, 256, 512, 1024];
        let mut out = [0usize; 5];
        for (i, &b) in base.iter().enumerate() {
            out[i] = make_divisible((b as f32 * self.width_mul) as usize, 8).max(8);
        }
        out
    }

    /// Compute block repeat count for each C2f stage.
    /// Base repeats: [3, 6, 6, 3].
    pub fn stage_depths(&self) -> [usize; 4] {
        let base = [3, 6, 6, 3];
        let mut out = [0usize; 4];
        for (i, &b) in base.iter().enumerate() {
            out[i] = ((b as f32 * self.depth_mul).ceil() as usize).max(1);
        }
        out
    }
}

/// Round to nearest multiple of `divisor`.
fn make_divisible(v: usize, divisor: usize) -> usize {
    let d = divisor;
    ((v + d / 2) / d) * d
}

// =============================================================================
// Output Types
// =============================================================================

/// Per-scale raw detection outputs for training.
pub struct HeliosScaleOutput {
    /// Classification logits [N, num_classes, H_i, W_i].
    pub cls_logits: Variable,
    /// Bounding box DFL distributions [N, 4*reg_max, H_i, W_i].
    pub bbox_dfl: Variable,
    /// Stride for this scale level.
    pub stride: usize,
}

/// Training output from Helios (raw head outputs, no NMS/decode).
pub struct HeliosTrainOutput {
    /// Per-scale outputs (3 scales: P3/8, P4/16, P5/32).
    pub scales: Vec<HeliosScaleOutput>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_nano() {
        let cfg = HeliosConfig::nano(80);
        assert_eq!(cfg.num_classes, 80);
        let ch = cfg.stage_channels();
        assert_eq!(ch, [16, 32, 64, 128, 256]);
        let depths = cfg.stage_depths();
        assert_eq!(depths, [1, 2, 2, 1]);
    }

    #[test]
    fn test_config_small() {
        let cfg = HeliosConfig::small(80);
        let ch = cfg.stage_channels();
        assert_eq!(ch, [32, 64, 128, 256, 512]);
    }

    #[test]
    fn test_config_large() {
        let cfg = HeliosConfig::large(80);
        let ch = cfg.stage_channels();
        assert_eq!(ch, [64, 128, 256, 512, 1024]);
        let depths = cfg.stage_depths();
        assert_eq!(depths, [3, 6, 6, 3]);
    }

    #[test]
    fn test_make_divisible() {
        assert_eq!(make_divisible(15, 8), 16);
        assert_eq!(make_divisible(64, 8), 64);
        assert_eq!(make_divisible(48, 8), 48);
        assert_eq!(make_divisible(100, 8), 104);
    }
}
