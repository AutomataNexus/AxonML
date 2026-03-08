//! Helios Training Loop — Complete Detection Training Pipeline
//!
//! Integrates model, loss, augmentation, EMA, and LR scheduling into
//! a production-ready training loop for the Helios detector.
//!
//! @version 0.1.0

use axonml_autograd::Variable;
use axonml_nn::{Module, Parameter};
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::{Device, Tensor};

use crate::models::helios::{Helios, HeliosLoss};
use super::augment::{DetAugPipeline, DetSample};
use super::ema::ModelEMA;
use super::metrics::{compute_map, DetectionResult, GroundTruth};

// =============================================================================
// Helios Training Config
// =============================================================================

/// Configuration for Helios training.
#[derive(Debug, Clone)]
pub struct HeliosTrainConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Initial learning rate.
    pub lr: f32,
    /// Weight decay (L2 regularization).
    pub weight_decay: f32,
    /// Warmup epochs (linear LR ramp).
    pub warmup_epochs: usize,
    /// Image input size (height, width).
    pub input_size: (usize, usize),
    /// Enable mosaic augmentation.
    pub use_mosaic: bool,
    /// Enable MixUp augmentation.
    pub use_mixup: bool,
    /// Enable EMA.
    pub use_ema: bool,
    /// EMA decay factor.
    pub ema_decay: f32,
    /// Log interval (steps between prints).
    pub log_interval: usize,
    /// Evaluate every N epochs.
    pub eval_interval: usize,
    /// Close mosaic in last N epochs (switch to simple augmentation).
    pub close_mosaic_epochs: usize,
    /// Number of classes.
    pub num_classes: usize,
}

impl HeliosTrainConfig {
    /// Default config for COCO-like training.
    pub fn coco(num_classes: usize) -> Self {
        Self {
            epochs: 300,
            batch_size: 16,
            lr: 0.01,
            weight_decay: 5e-4,
            warmup_epochs: 3,
            input_size: (640, 640),
            use_mosaic: true,
            use_mixup: true,
            use_ema: true,
            ema_decay: 0.9999,
            log_interval: 100,
            eval_interval: 10,
            close_mosaic_epochs: 10,
            num_classes,
        }
    }

    /// Fast config for small datasets / debugging.
    pub fn fast(num_classes: usize) -> Self {
        Self {
            epochs: 50,
            batch_size: 8,
            lr: 0.001,
            weight_decay: 1e-4,
            warmup_epochs: 1,
            input_size: (320, 320),
            use_mosaic: false,
            use_mixup: false,
            use_ema: true,
            ema_decay: 0.999,
            log_interval: 10,
            eval_interval: 5,
            close_mosaic_epochs: 5,
            num_classes,
        }
    }
}

// =============================================================================
// Training State
// =============================================================================

/// Training state returned after each epoch.
#[derive(Debug, Clone)]
pub struct EpochResult {
    /// Epoch number.
    pub epoch: usize,
    /// Average total loss.
    pub total_loss: f32,
    /// Average classification loss.
    pub cls_loss: f32,
    /// Average box regression loss.
    pub box_loss: f32,
    /// Average DFL loss.
    pub dfl_loss: f32,
    /// Current learning rate.
    pub lr: f32,
    /// mAP@50 if evaluation was run.
    pub map50: Option<f32>,
}

// =============================================================================
// Helios Trainer
// =============================================================================

/// Complete Helios training pipeline.
pub struct HeliosTrainer {
    /// The model being trained.
    pub model: Helios,
    /// Training configuration.
    pub config: HeliosTrainConfig,
    /// Loss function.
    loss_fn: HeliosLoss,
    /// Optimizer.
    optimizer: Adam,
    /// EMA tracker (if enabled).
    ema: Option<ModelEMA>,
    /// Augmentation pipeline.
    augment: DetAugPipeline,
    /// Current epoch.
    current_epoch: usize,
    /// Global step counter.
    global_step: usize,
    /// Device (CPU or CUDA).
    device: Device,
}

impl HeliosTrainer {
    /// Create a new trainer. Auto-detects GPU and moves model to CUDA if available.
    pub fn new(model: Helios, config: HeliosTrainConfig) -> Self {
        let reg_max = model.config().reg_max;
        let loss_fn = HeliosLoss::new(config.num_classes, reg_max);

        // Auto-detect GPU
        #[cfg(feature = "cuda")]
        let device = {
            let d = Device::Cuda(0);
            let test_t = Tensor::<f32>::from_vec(vec![0.0], &[1]).unwrap();
            if test_t.to_device(d).is_ok() {
                println!("[HeliosTrainer] Using GPU (CUDA:0)");
                d
            } else {
                println!("[HeliosTrainer] GPU not available, using CPU");
                Device::Cpu
            }
        };
        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;

        // Move model parameters to target device
        model.to_device(device);

        let optimizer = Adam::new(model.parameters(), config.lr)
            .weight_decay(config.weight_decay);

        let ema = if config.use_ema {
            Some(ModelEMA::new(&model.parameters(), config.ema_decay))
        } else {
            None
        };

        let (th, tw) = config.input_size;
        let augment = if config.use_mosaic {
            DetAugPipeline::yolo(th, tw)
        } else {
            DetAugPipeline::simple(th, tw)
        };

        Self {
            model,
            config,
            loss_fn,
            optimizer,
            ema,
            augment,
            current_epoch: 0,
            global_step: 0,
            device,
        }
    }

    /// Run a single training step on one batch.
    ///
    /// - `images`: Batch of augmented images [B, 3, H, W].
    /// - `gt_boxes`: Per-image GT boxes (xyxy pixel coords).
    /// - `gt_classes`: Per-image GT class labels.
    ///
    /// Returns (total_loss, cls_loss, box_loss, dfl_loss).
    pub fn train_step(
        &mut self,
        images: &Variable,
        gt_boxes: &[Vec<[f32; 4]>],
        gt_classes: &[Vec<usize>],
    ) -> (f32, f32, f32, f32) {
        // Warmup LR
        let lr = self.warmup_lr();
        self.optimizer.set_lr(lr);

        // Move input to device (GPU if available)
        let images_dev = images.to_device(self.device);

        // Forward
        self.optimizer.zero_grad();
        let train_out = self.model.forward_train(&images_dev);
        let (total_loss, cls, bx, dfl) = self.loss_fn.compute(
            &train_out,
            gt_boxes,
            gt_classes,
            self.config.num_classes,
        );

        // Backward
        total_loss.backward();
        self.optimizer.step();

        // EMA update
        if let Some(ref mut ema) = self.ema {
            ema.update(&self.model.parameters());
        }

        self.global_step += 1;
        (total_loss.data().to_vec()[0], cls, bx, dfl)
    }

    /// Run augmentation on a batch of samples.
    pub fn augment_batch(&self, samples: &[DetSample]) -> Vec<DetSample> {
        samples.iter().map(|s| self.augment.apply_single(s)).collect()
    }

    /// Run evaluation on a set of (image, gt) pairs.
    ///
    /// Returns mAP@50.
    pub fn evaluate(
        &self,
        eval_images: &[Tensor<f32>],
        eval_gt_boxes: &[Vec<[f32; 4]>],
        eval_gt_classes: &[Vec<usize>],
    ) -> f32 {
        let params = self.model.parameters();
        let num_classes = self.config.num_classes;

        let run_eval = || {
            let mut all_dets = Vec::new();
            let mut all_gts = Vec::new();

            for (i, img) in eval_images.iter().enumerate() {
                let input = Variable::new(
                    Tensor::from_vec(img.to_vec(), img.shape()).unwrap(),
                    false,
                );

                // Add batch dimension if needed
                let input = if input.shape().len() == 3 {
                    input.reshape(&[1, input.shape()[0], input.shape()[1], input.shape()[2]])
                } else {
                    input
                };
                let input = input.to_device(self.device);

                let detections = self.model.detect(&input, 0.001, 0.65);

                let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                    bbox: d.bbox,
                    confidence: d.confidence,
                    class_id: d.class_id,
                }).collect();

                let gts: Vec<GroundTruth> = eval_gt_boxes[i].iter().zip(eval_gt_classes[i].iter())
                    .map(|(bbox, &cls)| GroundTruth { bbox: *bbox, class_id: cls })
                    .collect();

                all_dets.push(dets);
                all_gts.push(gts);
            }

            compute_map(&all_dets, &all_gts, num_classes, 0.5)
        };

        // Use EMA weights for evaluation if available
        if let Some(ref ema) = self.ema {
            ema.apply_and_restore(&params, run_eval)
        } else {
            run_eval()
        }
    }

    /// Get current learning rate (with warmup).
    fn warmup_lr(&self) -> f32 {
        let warmup_steps = self.config.warmup_epochs * 100; // approximate steps per epoch
        if warmup_steps == 0 || self.global_step >= warmup_steps {
            // Cosine decay after warmup
            let total_steps = self.config.epochs * 100;
            let progress = (self.global_step as f32 - warmup_steps as f32)
                / (total_steps as f32 - warmup_steps as f32).max(1.0);
            let progress = progress.clamp(0.0, 1.0);
            let min_lr = self.config.lr * 0.01;
            min_lr + (self.config.lr - min_lr) * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
        } else {
            // Linear warmup
            self.config.lr * (self.global_step as f32 / warmup_steps as f32)
        }
    }

    /// Get current epoch.
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Advance epoch counter. Call after processing all batches in an epoch.
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;

        // Close mosaic in final epochs
        if self.config.use_mosaic
            && self.current_epoch >= self.config.epochs - self.config.close_mosaic_epochs
        {
            let (th, tw) = self.config.input_size;
            self.augment = DetAugPipeline::simple(th, tw);
        }
    }

    /// Get model parameters (for saving).
    pub fn parameters(&self) -> Vec<Parameter> {
        self.model.parameters()
    }

    /// Get EMA shadow parameters (for saving best model).
    pub fn ema_parameters(&self) -> Option<&[Vec<f32>]> {
        self.ema.as_ref().map(|e| e.shadow_params())
    }

    /// Get the augmentation pipeline (mutable, for reconfiguration).
    pub fn augment_mut(&mut self) -> &mut DetAugPipeline {
        &mut self.augment
    }

    /// Get the device (CPU or CUDA).
    pub fn device(&self) -> Device {
        self.device
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let model = Helios::nano(2);
        let config = HeliosTrainConfig::fast(2);
        let trainer = HeliosTrainer::new(model, config);

        assert_eq!(trainer.current_epoch(), 0);
        assert!(trainer.ema.is_some());
    }

    #[test]
    fn test_trainer_train_step() {
        let model = Helios::nano(2);
        let config = HeliosTrainConfig::fast(2);
        let mut trainer = HeliosTrainer::new(model, config);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let gt_boxes = vec![vec![[10.0, 10.0, 40.0, 40.0]]];
        let gt_classes = vec![vec![0usize]];

        let (total, cls, bx, dfl) = trainer.train_step(&input, &gt_boxes, &gt_classes);
        assert!(total.is_finite(), "Total loss should be finite");
        assert!(cls >= 0.0);
        assert!(bx >= 0.0);
        assert_eq!(trainer.global_step, 1);
    }

    #[test]
    fn test_trainer_warmup_lr() {
        let model = Helios::nano(2);
        let mut config = HeliosTrainConfig::fast(2);
        config.lr = 0.01;
        config.warmup_epochs = 2;
        let trainer = HeliosTrainer::new(model, config);

        // At step 0, LR should be near 0 (warmup)
        let lr = trainer.warmup_lr();
        assert!(lr < 0.001, "LR at step 0 should be small (warmup), got {lr}");
    }

    #[test]
    fn test_trainer_epoch_advance() {
        let model = Helios::nano(2);
        let mut config = HeliosTrainConfig::fast(2);
        config.epochs = 10;
        config.close_mosaic_epochs = 3;
        config.use_mosaic = true;
        let mut trainer = HeliosTrainer::new(model, config);

        assert!(trainer.augment.use_mosaic);

        // Advance past close_mosaic threshold
        for _ in 0..8 {
            trainer.advance_epoch();
        }

        // Mosaic should be disabled in final epochs
        assert!(!trainer.augment.use_mosaic);
    }

    #[test]
    fn test_trainer_evaluate_smoke() {
        let model = Helios::nano(2);
        let config = HeliosTrainConfig::fast(2);
        let trainer = HeliosTrainer::new(model, config);

        let eval_img = Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap();
        let eval_boxes = vec![vec![[10.0, 10.0, 40.0, 40.0]]];
        let eval_classes = vec![vec![0usize]];

        let map50 = trainer.evaluate(&[eval_img], &eval_boxes, &eval_classes);
        assert!(map50.is_finite());
        assert!(map50 >= 0.0 && map50 <= 1.0);
    }

    #[test]
    fn test_training_loop_smoke() {
        let model = Helios::nano(2);
        let config = HeliosTrainConfig::fast(2);
        let mut trainer = HeliosTrainer::new(model, config);

        // 3 steps of training
        let mut losses = Vec::new();
        for step in 0..3 {
            let seed = step as f32 * 0.1;
            let pixels: Vec<f32> = (0..3 * 64 * 64)
                .map(|i| ((i as f32 * 0.001 + seed).sin() * 0.5 + 0.5))
                .collect();
            let input = Variable::new(
                Tensor::from_vec(pixels, &[1, 3, 64, 64]).unwrap(),
                false,
            );

            let (total, _, _, _) = trainer.train_step(
                &input,
                &[vec![[8.0, 8.0, 48.0, 48.0]]],
                &[vec![0]],
            );
            losses.push(total);
        }

        trainer.advance_epoch();
        assert_eq!(trainer.current_epoch(), 1);

        for &l in &losses {
            assert!(l.is_finite(), "Loss should be finite");
        }
    }
}
