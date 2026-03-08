//! Training Infrastructure — Detection Training Loops
//!
//! Provides training loops for Nexus (COCO) and Phantom (WIDER FACE) detectors,
//! plus target assignment and evaluation metrics.
//!
//! @version 0.1.0

pub mod assign;
pub mod augment;
pub mod benchmarks;
pub mod coco_bench;
pub mod convergence;
pub mod ema;
pub mod gpu_bench;
pub mod helios_trainer;
pub mod integration;
pub mod metrics;
pub use assign::{assign_fcos_targets, assign_phantom_targets, fcos_targets_to_tensors, FcosTarget};
pub use augment::{
    DetAugPipeline, DetRandomAffine, DetRandomHFlip, DetSample, HSVJitter, LetterBox, MixUp,
    Mosaic,
};
pub use coco_bench::evaluate_helios_coco;
pub use ema::ModelEMA;
pub use helios_trainer::{HeliosTrainConfig, HeliosTrainer};
pub use metrics::{compute_ap, compute_coco_map, compute_map, DetectionResult, GroundTruth};

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

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
// Training Step Utilities
// =============================================================================

/// Run a single Nexus training step.
///
/// Returns the total loss value for this step.
pub fn nexus_training_step(
    model: &mut crate::models::nexus::Nexus,
    frame: &Variable,
    gt_boxes: &[[f32; 4]],  // pixel coords
    gt_classes: &[usize],
    optimizer: &mut dyn axonml_optim::Optimizer,
) -> f32 {
    use crate::losses::FocalLoss;
    use axonml_nn::SmoothL1Loss;

    // Forward pass (training mode — returns raw head outputs)
    let train_out = model.forward_train(frame);

    // Target assignment
    let strides = [8.0f32, 16.0, 32.0];
    let feat_sizes: Vec<(usize, usize)> = train_out.scales.iter().map(|s| {
        let cls_shape = s.cls_logits.shape();
        (cls_shape[2], cls_shape[3])
    }).collect();
    let size_ranges = vec![(0.0, 64.0), (64.0, 128.0), (128.0, f32::MAX)];

    let targets = assign_fcos_targets(gt_boxes, gt_classes, &feat_sizes, &strides, &size_ranges);
    let target_tensors = fcos_targets_to_tensors(&targets);

    // Compute losses per scale
    let focal_loss = FocalLoss::new();
    let smooth_l1 = SmoothL1Loss::new();
    let mut total_loss = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);

    for (scale_idx, scale_out) in train_out.scales.iter().enumerate() {
        let (ref cls_target, ref bbox_target, ref _center_target) = target_tensors[scale_idx];

        let cls_shape = scale_out.cls_logits.shape();
        let fh = cls_shape[2];
        let fw = cls_shape[3];

        // Reshape cls_logits from [1, 1, H, W] → [H*W]
        let cls_pred = scale_out.cls_logits.reshape(&[fh * fw]);
        let _cls_tgt = Variable::new(cls_target.clone(), false);

        // Convert class targets: -1 (bg) → 0, >=0 → 1 (for binary focal loss)
        let binary_target_data: Vec<f32> = cls_target.to_vec().iter()
            .map(|&v| if v >= 0.0 { 1.0 } else { 0.0 })
            .collect();
        let binary_target = Variable::new(
            Tensor::from_vec(binary_target_data, &[fh * fw]).unwrap(),
            false,
        );

        let cls_loss = focal_loss.compute(&cls_pred, &binary_target);

        // Bbox regression loss (only on positive locations)
        let positive_mask: Vec<bool> = cls_target.to_vec().iter().map(|&v| v >= 0.0).collect();
        let num_pos = positive_mask.iter().filter(|&&v| v).count();

        let bbox_loss = if num_pos > 0 {
            let bbox_pred = scale_out.bbox_pred.reshape(&[fh * fw, 4]);
            let bbox_tgt = Variable::new(bbox_target.clone(), false);
            smooth_l1.compute(&bbox_pred, &bbox_tgt).mul_scalar(num_pos as f32 / (fh * fw) as f32)
        } else {
            Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false)
        };

        total_loss = total_loss.add_var(&cls_loss).add_var(&bbox_loss);
    }

    // Backward + optimizer step
    let loss_val = total_loss.data().to_vec()[0];
    if total_loss.requires_grad() {
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();
    }

    loss_val
}

/// Run a single Phantom training step.
///
/// Returns the total loss value for this step.
pub fn phantom_training_step(
    model: &mut crate::models::phantom::Phantom,
    frame: &Variable,
    gt_faces: &[[f32; 4]],  // pixel coords
    optimizer: &mut dyn axonml_optim::Optimizer,
) -> f32 {
    use crate::losses::FocalLoss;
    use axonml_nn::SmoothL1Loss;

    // Forward pass
    let train_out = model.forward_train(frame);

    let cls_shape = train_out.face_cls.shape();
    let fh = cls_shape[2];
    let fw = cls_shape[3];
    let stride = 4.0f32;

    // Target assignment
    let (cls_target, bbox_target) = assign_phantom_targets(gt_faces, fh, fw, stride);

    // Classification loss
    let cls_pred = train_out.face_cls.reshape(&[fh * fw]);
    let cls_tgt = Variable::new(
        Tensor::from_vec(cls_target.to_vec(), &[fh * fw]).unwrap(),
        false,
    );
    let focal_loss = FocalLoss::new();
    let cls_loss = focal_loss.compute(&cls_pred, &cls_tgt);

    // Bbox regression loss (only on positive cells)
    let bbox_pred = train_out.face_bbox.reshape(&[fh * fw, 4]);
    let bbox_tgt = Variable::new(
        Tensor::from_vec(bbox_target.to_vec(), &[fh * fw, 4]).unwrap(),
        false,
    );
    let smooth_l1 = SmoothL1Loss::new();
    let bbox_loss = smooth_l1.compute(&bbox_pred, &bbox_tgt);

    let total_loss = cls_loss.add_var(&bbox_loss);

    let loss_val = total_loss.data().to_vec()[0];
    if total_loss.requires_grad() {
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();
    }

    loss_val
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

    #[test]
    fn test_phantom_training_step_smoke() {
        let mut model = crate::models::phantom::Phantom::new();
        let frame = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let gt_faces = vec![[10.0, 10.0, 30.0, 30.0]];

        let params = model.parameters();
        let mut optimizer = axonml_optim::Adam::new(params, 1e-3);

        let loss = phantom_training_step(&mut model, &frame, &gt_faces, &mut optimizer);
        assert!(loss.is_finite(), "Loss should be finite, got {loss}");
    }
}
