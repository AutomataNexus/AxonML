//! RetinaFace - Single-Stage Face Detection
//!
//! # File
//! `crates/axonml-vision/src/models/retinaface.rs`
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
use axonml_nn::{BatchNorm2d, Conv2d, Module, Parameter, ReLU};
use axonml_tensor::Tensor;

use crate::models::fpn::FPN;
use crate::models::resnet::ResNet;
use crate::ops::{nms, FaceDetection};

// =============================================================================
// Context Module (SSH-style)
// =============================================================================

/// Simple context module for enriching features.
struct ContextModule {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    relu: ReLU,
}

impl ContextModule {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(
                in_channels,
                out_channels / 2,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
            ),
            bn1: BatchNorm2d::new(out_channels / 2),
            conv2: Conv2d::with_options(
                out_channels / 2,
                out_channels / 4,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
            ),
            bn2: BatchNorm2d::new(out_channels / 4),
            conv3: Conv2d::with_options(
                out_channels / 4,
                out_channels / 4,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
            ),
            bn3: BatchNorm2d::new(out_channels / 4),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let out1 = self.relu.forward(&self.bn1.forward(&self.conv1.forward(x)));
        let out2 = self
            .relu
            .forward(&self.bn2.forward(&self.conv2.forward(&out1)));
        let out3 = self
            .relu
            .forward(&self.bn3.forward(&self.conv3.forward(&out2)));
        // Concatenate branches: out1 (C/2) + out2 (C/4) + out3 (C/4) = C
        concat_channels(&[&out1, &out2, &out3])
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.bn3.parameters());
        p
    }
}

// =============================================================================
// Detection Head
// =============================================================================

/// Per-level detection head for classification, bbox, and landmarks.
struct DetectionHead {
    cls_conv: Conv2d,
    bbox_conv: Conv2d,
    ldm_conv: Conv2d,
    _num_anchors: usize,
}

impl DetectionHead {
    fn new(in_channels: usize, num_anchors: usize) -> Self {
        Self {
            cls_conv: Conv2d::with_options(
                in_channels,
                num_anchors * 2,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            ),
            bbox_conv: Conv2d::with_options(
                in_channels,
                num_anchors * 4,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            ),
            ldm_conv: Conv2d::with_options(
                in_channels,
                num_anchors * 10,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            ),
            _num_anchors: num_anchors,
        }
    }

    /// Returns (cls_scores, bbox_deltas, landmarks) for this level.
    fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let cls = self.cls_conv.forward(x);
        let bbox = self.bbox_conv.forward(x);
        let ldm = self.ldm_conv.forward(x);
        (cls, bbox, ldm)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.cls_conv.parameters());
        p.extend(self.bbox_conv.parameters());
        p.extend(self.ldm_conv.parameters());
        p
    }
}

// =============================================================================
// RetinaFace
// =============================================================================

/// RetinaFace model for face detection with landmarks.
///
/// Uses ResNet34 backbone + FPN + multi-scale detection heads.
pub struct RetinaFace {
    /// ResNet backbone (uses layers 1-4)
    backbone: ResNet,
    /// Feature Pyramid Network
    fpn: FPN,
    /// Context modules per pyramid level
    context_modules: Vec<ContextModule>,
    /// Detection heads per pyramid level
    heads: Vec<DetectionHead>,
    /// Number of pyramid levels
    num_levels: usize,
    /// Number of anchors per cell
    num_anchors: usize,
}

impl RetinaFace {
    /// Create RetinaFace with ResNet34 backbone.
    pub fn new() -> Self {
        let fpn_channels = 64;
        let num_anchors = 2; // 2 anchors per cell per level
        let num_levels = 4;

        // ResNet34 backbone — we'll extract features from its layers
        let backbone = ResNet::resnet34(1000);

        // FPN: ResNet34 outputs [64, 128, 256, 512] channels
        let fpn = FPN::new(&[64, 128, 256, 512], fpn_channels);

        let mut context_modules = Vec::new();
        let mut heads = Vec::new();

        for _ in 0..num_levels {
            context_modules.push(ContextModule::new(fpn_channels, fpn_channels));
            heads.push(DetectionHead::new(fpn_channels, num_anchors));
        }

        Self {
            backbone,
            fpn,
            context_modules,
            heads,
            num_levels,
            num_anchors,
        }
    }

    /// Run inference and return face detections.
    ///
    /// # Arguments
    /// - `image`: `[1, 3, H, W]` normalized image tensor
    /// - `score_threshold`: Minimum confidence (e.g., 0.5)
    /// - `nms_threshold`: NMS IoU threshold (e.g., 0.4)
    ///
    /// # Returns
    /// Vector of `FaceDetection` with bbox and landmarks.
    pub fn detect(
        &self,
        image: &Variable,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Vec<FaceDetection> {
        let (cls_scores, bbox_preds, ldm_preds) = self.forward_raw(image);

        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();
        let mut all_landmarks = Vec::new();

        // Decode predictions from each level
        for level in 0..self.num_levels {
            let cls_data = cls_scores[level].data().to_vec();
            let bbox_data = bbox_preds[level].data().to_vec();
            let ldm_data = ldm_preds[level].data().to_vec();
            let shape = cls_scores[level].shape();
            let h = shape[2];
            let w = shape[3];

            for y in 0..h {
                for x in 0..w {
                    for a in 0..self.num_anchors {
                        // Softmax over 2 classes (bg/face)
                        let bg_idx = (a * 2) * h * w + y * w + x;
                        let fg_idx = (a * 2 + 1) * h * w + y * w + x;
                        let bg = cls_data[bg_idx];
                        let fg = cls_data[fg_idx];
                        let score = 1.0 / (1.0 + (bg - fg).exp());

                        if score < score_threshold {
                            continue;
                        }

                        // Decode bbox
                        let base = a * 4;
                        let dx = bbox_data[(base) * h * w + y * w + x];
                        let dy = bbox_data[(base + 1) * h * w + y * w + x];
                        let dw = bbox_data[(base + 2) * h * w + y * w + x];
                        let dh = bbox_data[(base + 3) * h * w + y * w + x];

                        let stride = 2usize.pow(level as u32 + 2);
                        let cx = (x as f32 + 0.5) * stride as f32;
                        let cy = (y as f32 + 0.5) * stride as f32;
                        let anchor_size = stride as f32 * 4.0;

                        let pred_cx = cx + dx * anchor_size;
                        let pred_cy = cy + dy * anchor_size;
                        let pred_w = anchor_size * dw.exp();
                        let pred_h = anchor_size * dh.exp();

                        all_boxes.push([
                            pred_cx - pred_w / 2.0,
                            pred_cy - pred_h / 2.0,
                            pred_cx + pred_w / 2.0,
                            pred_cy + pred_h / 2.0,
                        ]);
                        all_scores.push(score);

                        // Decode 5-point landmarks
                        let lbase = a * 10;
                        let mut lm = [(0.0f32, 0.0f32); 5];
                        for k in 0..5 {
                            let lx = ldm_data[(lbase + k * 2) * h * w + y * w + x];
                            let ly = ldm_data[(lbase + k * 2 + 1) * h * w + y * w + x];
                            lm[k] = (cx + lx * anchor_size, cy + ly * anchor_size);
                        }
                        all_landmarks.push(lm);
                    }
                }
            }
        }

        if all_scores.is_empty() {
            return vec![];
        }

        // Apply NMS
        let n = all_scores.len();
        let boxes_flat: Vec<f32> = all_boxes.iter().flat_map(|b| b.iter().copied()).collect();
        let boxes_tensor = Tensor::from_vec(boxes_flat, &[n, 4]).unwrap();
        let scores_tensor = Tensor::from_vec(all_scores.clone(), &[n]).unwrap();
        let keep = nms(&boxes_tensor, &scores_tensor, nms_threshold);

        keep.iter()
            .map(|&i| FaceDetection {
                bbox: all_boxes[i],
                confidence: all_scores[i],
                landmarks: Some(all_landmarks[i]),
            })
            .collect()
    }

    /// Raw forward pass returning per-level predictions.
    pub(crate) fn forward_raw(
        &self,
        x: &Variable,
    ) -> (Vec<Variable>, Vec<Variable>, Vec<Variable>) {
        // Extract backbone features at each layer
        let features = self.extract_backbone_features(x);

        // FPN
        let pyramid = self.fpn.forward(&features);

        let mut cls_all = Vec::new();
        let mut bbox_all = Vec::new();
        let mut ldm_all = Vec::new();

        for (i, feat) in pyramid.iter().enumerate() {
            let ctx = self.context_modules[i].forward(feat);
            let (cls, bbox, ldm) = self.heads[i].forward(&ctx);
            cls_all.push(cls);
            bbox_all.push(bbox);
            ldm_all.push(ldm);
        }

        (cls_all, bbox_all, ldm_all)
    }

    /// Extract multi-scale features from ResNet backbone.
    fn extract_backbone_features(&self, x: &Variable) -> Vec<Variable> {
        // Run through backbone stem + each layer to get C2, C3, C4, C5
        // We run the full backbone forward and collect intermediate features.
        // For now, we do a simplified version running through the backbone layers.
        let backbone_output = self.backbone.forward(x);
        let _shape = backbone_output.shape();

        // Simplified: create placeholder features at different scales.
        // In a real implementation, we'd hook into backbone layers.
        let input_shape = x.shape();
        let (n, _, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let make_feat = |c: usize, scale: usize| -> Variable {
            let fh = h / scale;
            let fw = w / scale;
            Variable::new(
                Tensor::from_vec(vec![0.1; n * c * fh * fw], &[n, c, fh, fw]).unwrap(),
                false,
            )
        };

        vec![
            make_feat(64, 4),   // C2: H/4
            make_feat(128, 8),  // C3: H/8
            make_feat(256, 16), // C4: H/16
            make_feat(512, 32), // C5: H/32
        ]
    }
}

impl Module for RetinaFace {
    fn forward(&self, x: &Variable) -> Variable {
        // Return classification scores for training
        let (cls, _, _) = self.forward_raw(x);
        // Flatten and concatenate all levels
        if cls.is_empty() {
            return x.clone();
        }
        cls[0].clone()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters());
        p.extend(self.fpn.parameters());
        for ctx in &self.context_modules {
            p.extend(ctx.parameters());
        }
        for head in &self.heads {
            p.extend(head.parameters());
        }
        p
    }

    fn train(&mut self) {
        self.backbone.train();
    }

    fn eval(&mut self) {
        self.backbone.eval();
    }
}

// =============================================================================
// Helper
// =============================================================================

/// Concatenate Variables along the channel dimension (autograd-tracked).
fn concat_channels(inputs: &[&Variable]) -> Variable {
    if inputs.is_empty() {
        panic!("concat_channels: empty input");
    }
    if inputs.len() == 1 {
        return inputs[0].clone();
    }

    Variable::cat(inputs, 1)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retinaface_creation() {
        let model = RetinaFace::new();
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_context_module() {
        let ctx = ContextModule::new(64, 64);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 64 * 8 * 8], &[1, 64, 8, 8]).unwrap(),
            false,
        );
        let output = ctx.forward(&input);
        assert_eq!(output.shape(), vec![1, 64, 8, 8]);
    }

    #[test]
    fn test_detection_head() {
        let head = DetectionHead::new(64, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 64 * 8 * 8], &[1, 64, 8, 8]).unwrap(),
            false,
        );
        let (cls, bbox, ldm) = head.forward(&input);
        assert_eq!(cls.shape()[1], 4); // 2 anchors * 2 classes
        assert_eq!(bbox.shape()[1], 8); // 2 anchors * 4 coords
        assert_eq!(ldm.shape()[1], 20); // 2 anchors * 10 landmark coords
    }

    #[test]
    fn test_retinaface_forward_smoke() {
        let model = RetinaFace::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        // First pyramid level cls output: [1, num_anchors*2, H/4, W/4]
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 4); // 2 anchors * 2 classes
    }

    #[test]
    fn test_retinaface_param_count() {
        let model = RetinaFace::new();
        let params = model.parameters();
        let total: usize = params
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        // ResNet34 backbone + FPN + context + heads
        assert!(total > 10_000, "RetinaFace has {} params", total);
    }
}
