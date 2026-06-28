//! Detection Training Module — FCOS Assignment and Per-Step Loss Runners
//!
//! Top-level module for detection-model training utilities in axonml-vision.
//! Re-exports submodules for target assignment (`assign`), augmentation
//! (`augment`), COCO benchmarks (`coco_bench`), EMA weight averaging (`ema`),
//! and metric helpers (AP / mAP / COCO mAP). Defines the shared `TrainConfig`
//! struct used by detection-model training.
//!
//! # File
//! `crates/axonml-vision/src/training/mod.rs`
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

pub mod assign;
pub mod augment;
pub mod benchmarks;
pub mod coco_bench;
pub mod convergence;
pub mod ema;
pub mod gpu_bench;
pub mod integration;
pub mod metrics;
pub use assign::{
    FcosTarget, assign_fcos_targets, assign_single_scale_targets, fcos_targets_to_tensors,
};
pub use augment::{
    DetAugPipeline, DetRandomAffine, DetRandomHFlip, DetSample, HSVJitter, LetterBox, MixUp, Mosaic,
};
pub use ema::ModelEMA;
pub use metrics::{DetectionResult, GroundTruth, compute_ap, compute_coco_map, compute_map};

// =============================================================================
// Training Configuration
// =============================================================================

/// Training configuration for detection models.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Dataset root directory.
    pub dataset_root: String,
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size (currently only batch_size=1 supported for detection).
    pub batch_size: usize,
    /// Learning rate.
    pub lr: f32,
    /// Weight decay.
    pub weight_decay: f32,
    /// Path to save checkpoints.
    pub save_path: Option<String>,
    /// Print loss every N steps.
    pub log_interval: usize,
    /// Image input size (height, width).
    pub input_size: (usize, usize),
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dataset_root: String::new(),
            epochs: 50,
            batch_size: 1,
            lr: 1e-3,
            weight_decay: 1e-4,
            save_path: None,
            log_interval: 100,
            input_size: (320, 320),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_config_default() {
        let config = TrainConfig::default();
        assert_eq!(config.epochs, 50);
        assert_eq!(config.batch_size, 1);
        assert!((config.lr - 1e-3).abs() < 1e-6);
    }
}
