//! BlazeFace - Lightweight Face Detection for Edge
//!
//! Ultra-fast face detection designed for mobile/edge deployment.
//! Uses depthwise separable convolutions for ~100K parameters.
//!
//! # Reference
//!
//! "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"
//! (Bazarevsky et al., 2019) <https://arxiv.org/abs/1907.05047>

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Module, Parameter, ReLU};
use axonml_tensor::Tensor;

use crate::ops::{nms, FaceDetection};

// =============================================================================
// BlazeBlock
// =============================================================================

/// BlazeBlock — depthwise separable residual block.
///
/// Uses depthwise + pointwise convolutions for efficiency.
struct BlazeBlock {
    /// Depthwise conv (groups = in_channels)
    dw_conv: Conv2d,
    dw_bn: BatchNorm2d,
    /// Pointwise conv (1x1)
    pw_conv: Conv2d,
    pw_bn: BatchNorm2d,
    /// Optional channel projection for residual
    project: Option<(Conv2d, BatchNorm2d)>,
    relu: ReLU,
    stride: usize,
}

impl BlazeBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let project = if in_channels != out_channels || stride != 1 {
            Some((
                Conv2d::with_options(in_channels, out_channels, (1, 1), (stride, stride), (0, 0), false),
                BatchNorm2d::new(out_channels),
            ))
        } else {
            None
        };

        Self {
            dw_conv: Conv2d::with_groups(in_channels, in_channels, (3, 3), (stride, stride), (1, 1), true, in_channels),
            dw_bn: BatchNorm2d::new(in_channels),
            pw_conv: Conv2d::with_options(in_channels, out_channels, (1, 1), (1, 1), (0, 0), true),
            pw_bn: BatchNorm2d::new(out_channels),
            project,
            relu: ReLU,
            stride,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let identity = match &self.project {
            Some((conv, bn)) => bn.forward(&conv.forward(x)),
            None => x.clone(),
        };

        let out = self.dw_conv.forward(x);
        let out = self.dw_bn.forward(&out);
        let out = self.relu.forward(&out);
        let out = self.pw_conv.forward(&out);
        let out = self.pw_bn.forward(&out);

        let out = out.add_var(&identity);
        self.relu.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.dw_conv.parameters());
        p.extend(self.dw_bn.parameters());
        p.extend(self.pw_conv.parameters());
        p.extend(self.pw_bn.parameters());
        if let Some((conv, bn)) = &self.project {
            p.extend(conv.parameters());
            p.extend(bn.parameters());
        }
        p
    }
}

// =============================================================================
// BlazeFace
// =============================================================================

/// BlazeFace lightweight face detector for edge deployment.
///
/// ~100K parameters, designed for real-time inference on mobile GPUs.
pub struct BlazeFace {
    /// Initial convolution
    stem: Conv2d,
    stem_bn: BatchNorm2d,
    relu: ReLU,
    /// BlazeBlocks forming the backbone
    blocks: Vec<BlazeBlock>,
    /// Classification head (face / not-face)
    cls_head: Conv2d,
    /// Bounding box regression head
    bbox_head: Conv2d,
    /// Number of anchors per cell
    num_anchors: usize,
}

impl BlazeFace {
    /// Create a BlazeFace model.
    ///
    /// Default configuration for 128x128 input images.
    pub fn new() -> Self {
        let num_anchors = 2;

        Self {
            stem: Conv2d::with_options(3, 24, (5, 5), (2, 2), (2, 2), true),
            stem_bn: BatchNorm2d::new(24),
            relu: ReLU,
            blocks: vec![
                BlazeBlock::new(24, 24, 1),
                BlazeBlock::new(24, 28, 1),
                BlazeBlock::new(28, 32, 2),  // Downsample
                BlazeBlock::new(32, 36, 1),
                BlazeBlock::new(36, 42, 1),
                BlazeBlock::new(42, 48, 2),  // Downsample
                BlazeBlock::new(48, 56, 1),
                BlazeBlock::new(56, 64, 1),
                BlazeBlock::new(64, 72, 2),  // Downsample
                BlazeBlock::new(72, 80, 1),
                BlazeBlock::new(80, 88, 1),
            ],
            cls_head: Conv2d::with_options(88, num_anchors * 1, (1, 1), (1, 1), (0, 0), true),
            bbox_head: Conv2d::with_options(88, num_anchors * 4, (1, 1), (1, 1), (0, 0), true),
            num_anchors,
        }
    }

    /// Run detection on a 128x128 input image.
    pub fn detect(
        &self,
        image: &Variable,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Vec<FaceDetection> {
        let cls = self.forward_cls(image);
        let bbox = self.forward_bbox(image);

        let cls_data = cls.data().to_vec();
        let bbox_data = bbox.data().to_vec();
        let shape = cls.shape();
        let h = shape[2];
        let w = shape[3];
        let input_shape = image.shape();
        let img_h = input_shape[2] as f32;
        let img_w = input_shape[3] as f32;

        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();

        let stride = (img_h / h as f32) as usize;

        for y in 0..h {
            for x in 0..w {
                for a in 0..self.num_anchors {
                    let score_idx = a * h * w + y * w + x;
                    let score = 1.0 / (1.0 + (-cls_data[score_idx]).exp()); // sigmoid

                    if score < score_threshold {
                        continue;
                    }

                    let base = a * 4;
                    let dx = bbox_data[(base) * h * w + y * w + x];
                    let dy = bbox_data[(base + 1) * h * w + y * w + x];
                    let dw = bbox_data[(base + 2) * h * w + y * w + x];
                    let dh = bbox_data[(base + 3) * h * w + y * w + x];

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
                }
            }
        }

        if all_scores.is_empty() {
            return vec![];
        }

        let n = all_scores.len();
        let boxes_flat: Vec<f32> = all_boxes.iter().flat_map(|b| b.iter().copied()).collect();
        let boxes_tensor = Tensor::from_vec(boxes_flat, &[n, 4]).unwrap();
        let scores_tensor = Tensor::from_vec(all_scores.clone(), &[n]).unwrap();
        let keep = nms(&boxes_tensor, &scores_tensor, nms_threshold);

        keep.iter()
            .map(|&i| FaceDetection {
                bbox: all_boxes[i],
                confidence: all_scores[i],
                landmarks: None,
            })
            .collect()
    }

    fn forward_features(&self, x: &Variable) -> Variable {
        let mut out = self.relu.forward(&self.stem_bn.forward(&self.stem.forward(x)));
        for block in &self.blocks {
            out = block.forward(&out);
        }
        out
    }

    fn forward_cls(&self, x: &Variable) -> Variable {
        let feat = self.forward_features(x);
        self.cls_head.forward(&feat)
    }

    fn forward_bbox(&self, x: &Variable) -> Variable {
        let feat = self.forward_features(x);
        self.bbox_head.forward(&feat)
    }
}

impl Module for BlazeFace {
    fn forward(&self, x: &Variable) -> Variable {
        self.forward_cls(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        p.extend(self.stem_bn.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.cls_head.parameters());
        p.extend(self.bbox_head.parameters());
        p
    }

    fn train(&mut self) {
        self.stem_bn.train();
        for block in &mut self.blocks {
            block.dw_bn.train();
            block.pw_bn.train();
        }
    }

    fn eval(&mut self) {
        self.stem_bn.eval();
        for block in &mut self.blocks {
            block.dw_bn.eval();
            block.pw_bn.eval();
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
    fn test_blazeblock() {
        let block = BlazeBlock::new(24, 24, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 24 * 16 * 16], &[1, 24, 16, 16]).unwrap(),
            false,
        );
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![1, 24, 16, 16]);
    }

    #[test]
    fn test_blazeblock_downsample() {
        let block = BlazeBlock::new(24, 48, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 24 * 16 * 16], &[1, 24, 16, 16]).unwrap(),
            false,
        );
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![1, 48, 8, 8]);
    }

    #[test]
    fn test_blazeface_creation() {
        let model = BlazeFace::new();
        let params = model.parameters();
        assert!(!params.is_empty());

        // Should be lightweight (<500K params)
        let total: usize = params.iter().map(|p| p.variable().data().to_vec().len()).sum();
        assert!(total < 500_000);
    }

    #[test]
    fn test_blazeface_forward() {
        let model = BlazeFace::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape()[0], 1);
    }
}
