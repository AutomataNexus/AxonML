//! BlazeFace - Lightweight Face Detection for Edge
//!
//! # File
//! `crates/axonml-vision/src/models/blazeface.rs`
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

use crate::ops::{FaceDetection, nms};

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
}

impl BlazeBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let project = if in_channels != out_channels || stride != 1 {
            Some((
                Conv2d::with_options(
                    in_channels,
                    out_channels,
                    (1, 1),
                    (stride, stride),
                    (0, 0),
                    false,
                ),
                BatchNorm2d::new(out_channels),
            ))
        } else {
            None
        };

        Self {
            dw_conv: Conv2d::with_groups(
                in_channels,
                in_channels,
                (3, 3),
                (stride, stride),
                (1, 1),
                true,
                in_channels,
            ),
            dw_bn: BatchNorm2d::new(in_channels),
            pw_conv: Conv2d::with_options(in_channels, out_channels, (1, 1), (1, 1), (0, 0), true),
            pw_bn: BatchNorm2d::new(out_channels),
            project,
            relu: ReLU,
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

    fn train_mode(&mut self) {
        self.dw_bn.train();
        self.pw_bn.train();
        if let Some((_, bn)) = &mut self.project {
            bn.train();
        }
    }

    fn eval_mode(&mut self) {
        self.dw_bn.eval();
        self.pw_bn.eval();
        if let Some((_, bn)) = &mut self.project {
            bn.eval();
        }
    }
}

// =============================================================================
// DoubleBlazeBlock
// =============================================================================

/// DoubleBlazeBlock — two depthwise convolutions stacked (5x5 receptive field).
///
/// Used in the second half of the backbone for larger receptive fields.
struct DoubleBlazeBlock {
    /// First depthwise conv
    dw_conv1: Conv2d,
    dw_bn1: BatchNorm2d,
    /// First pointwise conv
    pw_conv1: Conv2d,
    pw_bn1: BatchNorm2d,
    /// Second depthwise conv
    dw_conv2: Conv2d,
    dw_bn2: BatchNorm2d,
    /// Second pointwise conv
    pw_conv2: Conv2d,
    pw_bn2: BatchNorm2d,
    /// Residual projection
    project: Option<(Conv2d, BatchNorm2d)>,
    relu: ReLU,
}

impl DoubleBlazeBlock {
    fn new(in_channels: usize, mid_channels: usize, out_channels: usize, stride: usize) -> Self {
        let project = if in_channels != out_channels || stride != 1 {
            Some((
                Conv2d::with_options(
                    in_channels,
                    out_channels,
                    (1, 1),
                    (stride, stride),
                    (0, 0),
                    false,
                ),
                BatchNorm2d::new(out_channels),
            ))
        } else {
            None
        };

        Self {
            dw_conv1: Conv2d::with_groups(
                in_channels,
                in_channels,
                (3, 3),
                (stride, stride),
                (1, 1),
                true,
                in_channels,
            ),
            dw_bn1: BatchNorm2d::new(in_channels),
            pw_conv1: Conv2d::with_options(in_channels, mid_channels, (1, 1), (1, 1), (0, 0), true),
            pw_bn1: BatchNorm2d::new(mid_channels),
            dw_conv2: Conv2d::with_groups(
                mid_channels,
                mid_channels,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
                mid_channels,
            ),
            dw_bn2: BatchNorm2d::new(mid_channels),
            pw_conv2: Conv2d::with_options(
                mid_channels,
                out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            ),
            pw_bn2: BatchNorm2d::new(out_channels),
            project,
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let identity = match &self.project {
            Some((conv, bn)) => bn.forward(&conv.forward(x)),
            None => x.clone(),
        };

        let out = self.dw_conv1.forward(x);
        let out = self.dw_bn1.forward(&out);
        let out = self.relu.forward(&out);
        let out = self.pw_conv1.forward(&out);
        let out = self.pw_bn1.forward(&out);
        let out = self.relu.forward(&out);

        let out = self.dw_conv2.forward(&out);
        let out = self.dw_bn2.forward(&out);
        let out = self.relu.forward(&out);
        let out = self.pw_conv2.forward(&out);
        let out = self.pw_bn2.forward(&out);

        let out = out.add_var(&identity);
        self.relu.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.dw_conv1.parameters());
        p.extend(self.dw_bn1.parameters());
        p.extend(self.pw_conv1.parameters());
        p.extend(self.pw_bn1.parameters());
        p.extend(self.dw_conv2.parameters());
        p.extend(self.dw_bn2.parameters());
        p.extend(self.pw_conv2.parameters());
        p.extend(self.pw_bn2.parameters());
        if let Some((conv, bn)) = &self.project {
            p.extend(conv.parameters());
            p.extend(bn.parameters());
        }
        p
    }

    fn train_mode(&mut self) {
        self.dw_bn1.train();
        self.pw_bn1.train();
        self.dw_bn2.train();
        self.pw_bn2.train();
        if let Some((_, bn)) = &mut self.project {
            bn.train();
        }
    }

    fn eval_mode(&mut self) {
        self.dw_bn1.eval();
        self.pw_bn1.eval();
        self.dw_bn2.eval();
        self.pw_bn2.eval();
        if let Some((_, bn)) = &mut self.project {
            bn.eval();
        }
    }
}

// =============================================================================
// BlazeFace
// =============================================================================

/// BlazeFace lightweight face detector for edge deployment.
///
/// Dual-scale SSD-style architecture matching the original paper:
/// - Front backbone: 5 BlazeBlocks → 16x16 feature map (2 anchors/cell)
/// - Back backbone: 6 DoubleBlazeBlocks → 8x8 feature map (6 anchors/cell)
/// - Total: 896 anchors
/// - ~100K parameters
pub struct BlazeFace {
    // Stem
    stem: Conv2d,
    stem_bn: BatchNorm2d,
    relu: ReLU,

    // Front backbone (produces 16x16 feature map)
    front_blocks: Vec<BlazeBlock>,

    // Back backbone (produces 8x8 feature map)
    back_blocks: Vec<DoubleBlazeBlock>,

    // Scale 1 heads (16x16, 2 anchors per cell)
    cls_pre1: Conv2d,
    cls_pre1_bn: BatchNorm2d,
    cls_head1: Conv2d,
    bbox_head1: Conv2d,

    // Scale 2 heads (8x8, 6 anchors per cell)
    cls_pre2: Conv2d,
    cls_pre2_bn: BatchNorm2d,
    cls_head2: Conv2d,
    bbox_head2: Conv2d,
}

/// Configuration for anchor generation at each scale.
#[allow(dead_code)]
struct AnchorConfig {
    feature_size: usize,
    num_anchors: usize,
    stride: f32,
    anchor_sizes: Vec<f32>,
}

impl Default for BlazeFace {
    fn default() -> Self {
        Self::new()
    }
}

impl BlazeFace {
    /// Create a BlazeFace model.
    ///
    /// Dual-scale architecture for 128x128 input images.
    pub fn new() -> Self {
        Self {
            // Stem: 128x128x3 → 64x64x24
            stem: Conv2d::with_options(3, 24, (5, 5), (2, 2), (2, 2), true),
            stem_bn: BatchNorm2d::new(24),
            relu: ReLU,

            // Front backbone: 64x64 → 16x16
            // Each stride-2 block halves spatial dims
            front_blocks: vec![
                BlazeBlock::new(24, 24, 1),
                BlazeBlock::new(24, 28, 1),
                BlazeBlock::new(28, 32, 2), // → 32x32
                BlazeBlock::new(32, 36, 1),
                BlazeBlock::new(36, 42, 1),
                BlazeBlock::new(42, 48, 2), // → 16x16
                BlazeBlock::new(48, 56, 1),
                BlazeBlock::new(56, 64, 1),
            ],

            // Back backbone: 16x16 → 8x8
            back_blocks: vec![
                DoubleBlazeBlock::new(64, 64, 96, 2), // → 8x8
                DoubleBlazeBlock::new(96, 96, 96, 1),
                DoubleBlazeBlock::new(96, 96, 96, 1),
            ],

            // Scale 1: 16x16 with 2 anchors/cell — deeper cls head for better confidence
            cls_pre1: Conv2d::with_options(64, 64, (3, 3), (1, 1), (1, 1), true),
            cls_pre1_bn: BatchNorm2d::new(64),
            cls_head1: Conv2d::with_options(64, 2, (3, 3), (1, 1), (1, 1), true),
            bbox_head1: Conv2d::with_options(64, 2 * 4, (3, 3), (1, 1), (1, 1), true),

            // Scale 2: 8x8 with 6 anchors/cell — deeper cls head
            cls_pre2: Conv2d::with_options(96, 96, (3, 3), (1, 1), (1, 1), true),
            cls_pre2_bn: BatchNorm2d::new(96),
            cls_head2: Conv2d::with_options(96, 6, (3, 3), (1, 1), (1, 1), true),
            bbox_head2: Conv2d::with_options(96, 6 * 4, (3, 3), (1, 1), (1, 1), true),
        }
    }

    /// Forward pass returning raw predictions for training.
    ///
    /// Returns (cls_logits, bbox_preds) concatenated across both scales.
    /// - cls_logits: [batch, num_anchors] (896 for 128x128 input)
    /// - bbox_preds: [batch, num_anchors, 4]
    pub fn forward_train(&self, x: &Variable) -> (Variable, Variable) {
        let (feat1, feat2) = self.forward_features(x);

        // Scale 1: 16x16, 2 anchors — deeper cls path for better confidence
        let cls1_feat = self
            .relu
            .forward(&self.cls_pre1_bn.forward(&self.cls_pre1.forward(&feat1)));
        let cls1 = self.cls_head1.forward(&cls1_feat); // [B, 2, 16, 16]
        let bbox1 = self.bbox_head1.forward(&feat1); // [B, 8, 16, 16]

        // Scale 2: 8x8, 6 anchors — deeper cls path
        let cls2_feat = self
            .relu
            .forward(&self.cls_pre2_bn.forward(&self.cls_pre2.forward(&feat2)));
        let cls2 = self.cls_head2.forward(&cls2_feat); // [B, 6, 8, 8]
        let bbox2 = self.bbox_head2.forward(&feat2); // [B, 24, 8, 8]

        let batch = x.shape()[0];

        // Get actual spatial dimensions from feature maps (supports any input size)
        let h1 = cls1.shape()[2];
        let w1 = cls1.shape()[3];
        let h2 = cls2.shape()[2];
        let w2 = cls2.shape()[3];

        // Reshape and concatenate: cls
        // Conv2d outputs [B, A, H, W] but anchors are ordered (H, W, A) — spatial first.
        // Permute to [B, H, W, A] before flattening to match generate_anchors() order.
        let cls1_perm = cls1.transpose(1, 2); // [B, H, A, W]
        let cls1_perm = cls1_perm.transpose(2, 3); // [B, H, W, A]
        let cls1_flat = cls1_perm.reshape(&[batch, h1 * w1 * 2]);

        let cls2_perm = cls2.transpose(1, 2); // [B, H, A, W]
        let cls2_perm = cls2_perm.transpose(2, 3); // [B, H, W, A]
        let cls2_flat = cls2_perm.reshape(&[batch, h2 * w2 * 6]);

        let cls_all = Variable::cat(&[&cls1_flat, &cls2_flat], 1);

        // Reshape and concatenate: bbox
        // [B, A*4, H, W] → permute to [B, H, W, A*4] → reshape to [B, H*W*A, 4]
        let n1 = 2 * h1 * w1;
        let n2 = 6 * h2 * w2;

        // bbox1: [B, 8, H, W] → [B, H, W, 8] → [B, H*W*2, 4]
        let bbox1_perm = bbox1.transpose(1, 2); // [B, H, 8, W]
        let bbox1_perm = bbox1_perm.transpose(2, 3); // [B, H, W, 8]
        let bbox1_flat = bbox1_perm.reshape(&[batch, n1, 4]);

        // bbox2: [B, 24, H, W] → [B, H, W, 24] → [B, H*W*6, 4]
        let bbox2_perm = bbox2.transpose(1, 2); // [B, H, 24, W]
        let bbox2_perm = bbox2_perm.transpose(2, 3); // [B, H, W, 24]
        let bbox2_flat = bbox2_perm.reshape(&[batch, n2, 4]);

        let bbox_all = Variable::cat(&[&bbox1_flat, &bbox2_flat], 1); // [B, 896, 4]

        (cls_all, bbox_all)
    }

    /// Generate anchor boxes for given input size.
    ///
    /// Returns Vec of (cx, cy, w, h) in pixel coordinates, one per anchor.
    /// Uses multi-aspect-ratio anchors for better face matching.
    pub fn generate_anchors(input_size: usize) -> Vec<[f32; 4]> {
        // Scale 1: 16x16 feature map, 2 anchors/cell (matches cls_head1 output)
        // Scale 2: 8x8 feature map, 6 anchors/cell (matches cls_head2 output)
        // Aspect ratios: (scale, w_ratio, h_ratio)
        let scale1_anchors: Vec<(f32, f32, f32)> = vec![
            (0.75, 1.0, 1.0), // 6px — tiny faces
            (1.5, 1.0, 1.0),  // 12px — small faces
        ];
        let scale2_anchors: Vec<(f32, f32, f32)> = vec![
            (1.0, 1.0, 1.0), // 16px — small-medium faces
            (1.5, 1.0, 1.0), // 24px — medium faces
            (2.5, 1.0, 1.0), // 40px — medium-large faces
            (4.0, 1.0, 1.0), // 64px — large faces
            (1.5, 1.0, 1.3), // 24x31px — tall portrait faces
            (6.0, 1.0, 1.0), // 96px — very large faces (close-up)
        ];

        let mut anchors = Vec::new();

        // Feature map sizes depend on input:
        // Stem: /2, front_blocks has 2 stride-2 blocks: /4 → total /8 for feat1
        // Back_blocks has 1 stride-2 block: /2 → total /16 for feat2
        let feat1_size = input_size / 8;
        let feat2_size = input_size / 16;

        // Scale 1
        let stride1 = input_size as f32 / feat1_size as f32;
        for y in 0..feat1_size {
            for x in 0..feat1_size {
                let cx = (x as f32 + 0.5) * stride1;
                let cy = (y as f32 + 0.5) * stride1;
                for &(scale, wr, hr) in &scale1_anchors {
                    let base = stride1 * scale;
                    anchors.push([cx, cy, base * wr, base * hr]);
                }
            }
        }

        // Scale 2
        let stride2 = input_size as f32 / feat2_size as f32;
        for y in 0..feat2_size {
            for x in 0..feat2_size {
                let cx = (x as f32 + 0.5) * stride2;
                let cy = (y as f32 + 0.5) * stride2;
                for &(scale, wr, hr) in &scale2_anchors {
                    let base = stride2 * scale;
                    anchors.push([cx, cy, base * wr, base * hr]);
                }
            }
        }
        anchors
    }

    /// Run detection on an input image.
    pub fn detect(
        &self,
        image: &Variable,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Vec<FaceDetection> {
        let input_size = image.shape()[2]; // assume square
        let (cls_logits, bbox_preds) = self.forward_train(image);

        let cls_data = cls_logits.data().to_vec();
        let bbox_data = bbox_preds.data().to_vec();
        let anchors = Self::generate_anchors(input_size);
        let num_anchors = anchors.len();

        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();

        for i in 0..num_anchors {
            let score = 1.0 / (1.0 + (-cls_data[i]).exp()); // sigmoid
            if score < score_threshold {
                continue;
            }

            let anchor = &anchors[i];
            let (acx, acy, aw, ah) = (anchor[0], anchor[1], anchor[2], anchor[3]);

            // Decode: predictions are (dx, dy, dw, dh) offsets
            let dx = bbox_data[i * 4];
            let dy = bbox_data[i * 4 + 1];
            let dw = bbox_data[i * 4 + 2];
            let dh = bbox_data[i * 4 + 3];

            let pred_cx = acx + dx * aw;
            let pred_cy = acy + dy * ah;
            let pred_w = aw * dw.exp();
            let pred_h = ah * dh.exp();

            all_boxes.push([
                pred_cx - pred_w / 2.0,
                pred_cy - pred_h / 2.0,
                pred_cx + pred_w / 2.0,
                pred_cy + pred_h / 2.0,
            ]);
            all_scores.push(score);
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

    /// Forward through backbone, returning features at two scales.
    fn forward_features(&self, x: &Variable) -> (Variable, Variable) {
        let mut out = self
            .relu
            .forward(&self.stem_bn.forward(&self.stem.forward(x)));

        // Front backbone → 16x16
        for block in &self.front_blocks {
            out = block.forward(&out);
        }
        let feat1 = out.clone(); // 16x16 features

        // Back backbone → 8x8
        let mut out = feat1.clone();
        for block in &self.back_blocks {
            out = block.forward(&out);
        }
        let feat2 = out; // 8x8 features

        (feat1, feat2)
    }

    /// Forward classification only (for backward compatibility).
    pub(crate) fn forward_cls(&self, x: &Variable) -> Variable {
        let (cls, _bbox) = self.forward_train(x);
        cls
    }

    /// Forward bbox only (for backward compatibility).
    #[allow(dead_code)]
    pub(crate) fn forward_bbox(&self, x: &Variable) -> Variable {
        let (_cls, bbox) = self.forward_train(x);
        bbox.reshape(&[bbox.shape()[0], bbox.shape()[1] * 4])
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
        for block in &self.front_blocks {
            p.extend(block.parameters());
        }
        for block in &self.back_blocks {
            p.extend(block.parameters());
        }
        p.extend(self.cls_pre1.parameters());
        p.extend(self.cls_pre1_bn.parameters());
        p.extend(self.cls_head1.parameters());
        p.extend(self.bbox_head1.parameters());
        p.extend(self.cls_pre2.parameters());
        p.extend(self.cls_pre2_bn.parameters());
        p.extend(self.cls_head2.parameters());
        p.extend(self.bbox_head2.parameters());
        p
    }

    fn train(&mut self) {
        self.stem_bn.train();
        self.cls_pre1_bn.train();
        self.cls_pre2_bn.train();
        for block in &mut self.front_blocks {
            block.train_mode();
        }
        for block in &mut self.back_blocks {
            block.train_mode();
        }
    }

    fn eval(&mut self) {
        self.stem_bn.eval();
        self.cls_pre1_bn.eval();
        self.cls_pre2_bn.eval();
        for block in &mut self.front_blocks {
            block.eval_mode();
        }
        for block in &mut self.back_blocks {
            block.eval_mode();
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
            Tensor::from_vec(vec![0.1; 24 * 16 * 16], &[1, 24, 16, 16]).unwrap(),
            false,
        );
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![1, 24, 16, 16]);
    }

    #[test]
    fn test_blazeblock_downsample() {
        let block = BlazeBlock::new(24, 48, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 24 * 16 * 16], &[1, 24, 16, 16]).unwrap(),
            false,
        );
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![1, 48, 8, 8]);
    }

    #[test]
    fn test_double_blazeblock() {
        let block = DoubleBlazeBlock::new(64, 64, 96, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 16 * 16], &[1, 64, 16, 16]).unwrap(),
            false,
        );
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![1, 96, 8, 8]);
    }

    #[test]
    fn test_blazeface_creation() {
        let model = BlazeFace::new();
        let params = model.parameters();
        assert!(!params.is_empty());

        let total: usize = params
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        println!("BlazeFace total params: {}", total);
        // Should be lightweight (<300K params)
        assert!(total < 300_000);
    }

    #[test]
    fn test_blazeface_forward_train() {
        let model = BlazeFace::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        let (cls, bbox) = model.forward_train(&input);
        // 2*16*16 + 6*8*8 = 512 + 384 = 896 anchors
        assert_eq!(cls.shape(), vec![1, 896]);
        assert_eq!(bbox.shape(), vec![1, 896, 4]);
    }

    #[test]
    fn test_blazeface_anchors() {
        let anchors = BlazeFace::generate_anchors(128);
        assert_eq!(anchors.len(), 896);

        // First anchor should be near top-left of 16x16 grid
        let a = &anchors[0];
        assert!(a[0] > 0.0 && a[0] < 128.0); // cx
        assert!(a[1] > 0.0 && a[1] < 128.0); // cy
    }

    #[test]
    fn test_blazeface_detect() {
        let model = BlazeFace::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        // Just verify it runs without panic
        let _dets = model.detect(&input, 0.5, 0.3);
    }

    #[test]
    fn test_blazeface_backward() {
        let model = BlazeFace::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            true,
        );
        let (cls, _bbox) = model.forward_train(&input);
        let loss = cls.mean();
        loss.backward();
        // Verify at least some gradients flowed (BatchNorm may sever some)
        let grads: usize = model
            .parameters()
            .iter()
            .filter(|p| p.variable().grad().is_some())
            .count();
        assert!(grads > 0, "At least some parameters should have gradients");
    }
}
