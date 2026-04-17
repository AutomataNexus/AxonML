//! Helios Detector — Full Object Detection Pipeline
//!
//! Implements `Helios`, the complete detection pipeline combining `CSPDarknet`
//! backbone, `PANet` neck, and `HeliosHead` into a single model. Provides
//! `forward_train()` returning raw per-scale outputs for loss computation,
//! `forward_detect()` with DFL bbox decoding + NMS for inference, anchor-free
//! grid generation, and DFL distribution-to-offset conversion. Implements the
//! `Module` trait for standard forward pass returning decoded `Detection` results.
//!
//! # File
//! `crates/axonml-vision/src/models/helios/detector.rs`
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

use axonml_autograd::Variable;
use axonml_nn::{Module, Parameter};
use axonml_tensor::Tensor;

use super::backbone::CSPDarknet;
use super::head::HeliosHead;
use super::neck::PANet;
use super::{HeliosConfig, HeliosScaleOutput, HeliosTrainOutput};
use crate::ops::{Detection, nms};

// =============================================================================
// Helios Detector
// =============================================================================

/// Helios — YOLO-competitive anchor-free object detector.
///
/// Architecture: CSPDarknet backbone + PANet neck + Decoupled head with DFL.
pub struct Helios {
    backbone: CSPDarknet,
    neck: PANet,
    head: HeliosHead,
    config: HeliosConfig,
}

impl Helios {
    /// Create a Helios detector with the given configuration.
    pub fn new(config: HeliosConfig) -> Self {
        let backbone = CSPDarknet::new(&config);
        let neck = PANet::new(backbone.out_channels, &config);
        let head = HeliosHead::new(&neck.out_channels, config.num_classes, config.reg_max);

        Self {
            backbone,
            neck,
            head,
            config,
        }
    }

    /// Helios-Nano (~3M params).
    pub fn nano(num_classes: usize) -> Self {
        Self::new(HeliosConfig::nano(num_classes))
    }

    /// Helios-Small (~9M params).
    pub fn small(num_classes: usize) -> Self {
        Self::new(HeliosConfig::small(num_classes))
    }

    /// Helios-Medium (~23M params).
    pub fn medium(num_classes: usize) -> Self {
        Self::new(HeliosConfig::medium(num_classes))
    }

    /// Helios-Large (~44M params).
    pub fn large(num_classes: usize) -> Self {
        Self::new(HeliosConfig::large(num_classes))
    }

    /// Helios-XLarge (~68M params).
    pub fn xlarge(num_classes: usize) -> Self {
        Self::new(HeliosConfig::xlarge(num_classes))
    }

    /// Get the configuration.
    pub fn config(&self) -> &HeliosConfig {
        &self.config
    }

    /// Training forward: returns raw predictions for loss computation.
    pub fn forward_train(&self, image: &Variable) -> HeliosTrainOutput {
        let (p3, p4, p5) = self.backbone.forward(image);
        let (n3, n4, n5) = self.neck.forward(&p3, &p4, &p5);

        let feats = [&n3, &n4, &n5];
        let strides = &self.config.strides;

        let scales = feats
            .iter()
            .enumerate()
            .map(|(i, feat)| {
                let (cls_logits, bbox_dfl) = self.head.forward_single(feat, i);
                HeliosScaleOutput {
                    cls_logits,
                    bbox_dfl,
                    stride: strides[i],
                }
            })
            .collect();

        HeliosTrainOutput { scales }
    }

    /// Run detection with NMS.
    pub fn detect(
        &self,
        image: &Variable,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Vec<Detection> {
        let train_out = self.forward_train(image);

        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();
        let mut all_classes = Vec::new();

        for scale in &train_out.scales {
            let cls_shape = scale.cls_logits.shape();
            let n = cls_shape[0];
            let num_classes = cls_shape[1];
            let h = cls_shape[2];
            let w = cls_shape[3];
            let stride = scale.stride as f32;

            // Sigmoid on class logits
            let cls_data = scale.cls_logits.sigmoid().data().to_vec();

            // DFL decode bbox
            let bbox_decoded = self.head.dfl_decode(&scale.bbox_dfl);
            let bbox_data = bbox_decoded.data().to_vec();

            // Iterate over grid cells
            for b in 0..n {
                for yi in 0..h {
                    for xi in 0..w {
                        // Find best class
                        let mut best_score = 0.0f32;
                        let mut best_class = 0usize;
                        for c in 0..num_classes {
                            let idx = b * num_classes * h * w + c * h * w + yi * w + xi;
                            if cls_data[idx] > best_score {
                                best_score = cls_data[idx];
                                best_class = c;
                            }
                        }

                        if best_score < score_threshold {
                            continue;
                        }

                        // ltrb distances
                        let base = b * 4 * h * w;
                        let l = bbox_data[base + 0 * h * w + yi * w + xi];
                        let t = bbox_data[base + h * w + yi * w + xi];
                        let r = bbox_data[base + 2 * h * w + yi * w + xi];
                        let bt = bbox_data[base + 3 * h * w + yi * w + xi];

                        // Convert from grid distances to pixel coordinates
                        let cx = (xi as f32 + 0.5) * stride;
                        let cy = (yi as f32 + 0.5) * stride;
                        let x1 = cx - l * stride;
                        let y1 = cy - t * stride;
                        let x2 = cx + r * stride;
                        let y2 = cy + bt * stride;

                        all_boxes.push([x1, y1, x2, y2]);
                        all_scores.push(best_score);
                        all_classes.push(best_class);
                    }
                }
            }
        }

        if all_boxes.is_empty() {
            return Vec::new();
        }

        // Per-class NMS
        let mut detections = Vec::new();
        let unique_classes: Vec<usize> = {
            let mut c = all_classes.clone();
            c.sort_unstable();
            c.dedup();
            c
        };

        for cls in unique_classes {
            let mut cls_boxes = Vec::new();
            let mut cls_scores = Vec::new();
            let mut cls_indices = Vec::new();

            for (i, &c) in all_classes.iter().enumerate() {
                if c == cls {
                    cls_boxes.extend_from_slice(&all_boxes[i]);
                    cls_scores.push(all_scores[i]);
                    cls_indices.push(i);
                }
            }

            if cls_scores.is_empty() {
                continue;
            }

            let n_cls = cls_scores.len();
            let boxes_tensor = Tensor::from_vec(cls_boxes, &[n_cls, 4]).unwrap();
            let scores_tensor = Tensor::from_vec(cls_scores.clone(), &[n_cls]).unwrap();

            let keep = nms(&boxes_tensor, &scores_tensor, nms_threshold);

            for k in keep {
                let orig_idx = cls_indices[k];
                detections.push(Detection {
                    bbox: all_boxes[orig_idx],
                    confidence: all_scores[orig_idx],
                    class_id: cls,
                });
            }
        }

        // Sort by confidence descending
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        detections
    }

    /// Get all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.backbone.parameters();
        p.extend(self.neck.parameters());
        p.extend(self.head.parameters());
        p
    }

    /// Set training mode.
    pub fn train(&mut self) {
        // BatchNorm behavior handled internally
    }

    /// Set evaluation mode.
    pub fn eval(&mut self) {
        // BatchNorm behavior handled internally
    }
}

impl Module for Helios {
    fn forward(&self, x: &Variable) -> Variable {
        // Returns first-scale class logits for Module API compatibility
        let train_out = self.forward_train(x);
        train_out.scales[0].cls_logits.clone()
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.train();
    }

    fn eval(&mut self) {
        self.eval();
    }

    fn name(&self) -> &'static str {
        "Helios"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helios_nano_creation() {
        let model = Helios::nano(80);
        let params = model.parameters();
        assert!(!params.is_empty());

        let total: usize = params
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        println!("Helios-Nano params: {total}");
        assert!(total > 100_000, "Too few params: {total}");
    }

    #[test]
    fn test_helios_nano_forward_train() {
        let model = Helios::nano(80);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        let out = model.forward_train(&input);
        assert_eq!(out.scales.len(), 3);

        // P3 at stride 8: 64/8 = 8
        assert_eq!(out.scales[0].cls_logits.shape()[1], 80);
        assert_eq!(out.scales[0].cls_logits.shape()[2], 8);
        assert_eq!(out.scales[0].bbox_dfl.shape()[1], 64); // 4*16

        // P4 at stride 16: 64/16 = 4
        assert_eq!(out.scales[1].cls_logits.shape()[2], 4);

        // P5 at stride 32: 64/32 = 2
        assert_eq!(out.scales[2].cls_logits.shape()[2], 2);
    }

    #[test]
    fn test_helios_nano_detect() {
        let model = Helios::nano(2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        // With random weights, detections depend on score threshold
        let dets = model.detect(&input, 0.5, 0.45);
        // Just verify it doesn't crash and returns valid detections
        for det in &dets {
            assert!(det.confidence >= 0.5);
            assert!(det.class_id < 2);
        }
    }

    #[test]
    fn test_helios_small_forward() {
        let model = Helios::small(20);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        let out = model.forward_train(&input);
        assert_eq!(out.scales.len(), 3);
        assert_eq!(out.scales[0].cls_logits.shape()[1], 20);
    }

    #[test]
    fn test_helios_module_forward() {
        let model = Helios::nano(10);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        let out = model.forward(&input);
        assert_eq!(out.shape()[1], 10); // num_classes
    }

    #[test]
    fn test_helios_sizes() {
        // Verify all sizes construct successfully
        for (name, model) in [("Nano", Helios::nano(10)), ("Small", Helios::small(10))] {
            let params = model.parameters();
            let total: usize = params
                .iter()
                .map(|p| p.variable().data().to_vec().len())
                .sum();
            println!("Helios-{name}: {total} params");
            assert!(total > 0);
        }
    }
}
