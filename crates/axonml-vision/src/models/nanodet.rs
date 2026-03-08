//! NanoDet — Lightweight Anchor-Free Object Detection for Edge
//!
//! # File
//! `crates/axonml-vision/src/models/nanodet.rs`
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

use std::any::Any;

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_nn::{BatchNorm2d, Conv2d, Module, Parameter, ReLU};
use axonml_tensor::Tensor;

use crate::ops::{nms, Detection};

// =============================================================================
// ShuffleNet V2 Backbone
// =============================================================================

/// Channel shuffle operation for ShuffleNet.
fn channel_shuffle(x: &Variable, groups: usize) -> Variable {
    let shape = x.shape();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let channels_per_group = c / groups;

    // Reshape -> transpose -> reshape
    let data = x.data().to_vec();
    let mut shuffled = vec![0.0f32; data.len()];

    for b in 0..n {
        for g in 0..groups {
            for cg in 0..channels_per_group {
                let src_c = g * channels_per_group + cg;
                let dst_c = cg * groups + g;
                for y in 0..h {
                    for x_pos in 0..w {
                        let src_idx = b * c * h * w + src_c * h * w + y * w + x_pos;
                        let dst_idx = b * c * h * w + dst_c * h * w + y * w + x_pos;
                        shuffled[dst_idx] = data[src_idx];
                    }
                }
            }
        }
    }

    let output_tensor = Tensor::from_vec(shuffled, &[n, c, h, w]).unwrap();
    if x.requires_grad() && is_grad_enabled() {
        let grad_fn = GradFn::new(ChannelShuffleBackward {
            next_fns: vec![x.grad_fn().cloned()],
            groups,
            channels_per_group,
            shape: shape.to_vec(),
        });
        Variable::from_operation(output_tensor, grad_fn, true)
    } else {
        Variable::new(output_tensor, false)
    }
}

/// Gradient function for channel shuffle (inverse permutation).
#[derive(Debug)]
struct ChannelShuffleBackward {
    next_fns: Vec<Option<GradFn>>,
    groups: usize,
    channels_per_group: usize,
    shape: Vec<usize>,
}

impl GradientFunction for ChannelShuffleBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let g_vec = grad_output.to_vec();
        let mut grad_input = vec![0.0f32; g_vec.len()];

        // Inverse shuffle: reverse the permutation
        for b in 0..n {
            for g in 0..self.groups {
                for cg in 0..self.channels_per_group {
                    let dst_c = g * self.channels_per_group + cg; // original channel
                    let src_c = cg * self.groups + g; // shuffled channel
                    for y in 0..h {
                        for x_pos in 0..w {
                            let src_idx = b * c * h * w + src_c * h * w + y * w + x_pos;
                            let dst_idx = b * c * h * w + dst_c * h * w + y * w + x_pos;
                            grad_input[dst_idx] = g_vec[src_idx];
                        }
                    }
                }
            }
        }

        let gi = Tensor::from_vec(grad_input, &self.shape).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "ChannelShuffleBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// ShuffleNet V2 inverted residual block.
struct ShuffleBlock {
    /// Branch 2: 1x1 conv -> depthwise 3x3 -> 1x1 conv
    branch2_pw1: Conv2d,
    branch2_bn1: BatchNorm2d,
    branch2_dw: Conv2d,
    branch2_bn2: BatchNorm2d,
    branch2_pw2: Conv2d,
    branch2_bn3: BatchNorm2d,
    /// Shortcut branch (only for stride=2)
    shortcut: Option<(Conv2d, BatchNorm2d, Conv2d, BatchNorm2d)>,
    relu: ReLU,
    stride: usize,
    _in_channels: usize,
    _out_channels: usize,
}

impl ShuffleBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let branch_channels = out_channels / 2;
        let inp = if stride == 2 { in_channels } else { in_channels / 2 };

        let shortcut = if stride == 2 {
            Some((
                Conv2d::with_groups(in_channels, in_channels, (3, 3), (2, 2), (1, 1), true, in_channels),
                BatchNorm2d::new(in_channels),
                Conv2d::with_options(in_channels, branch_channels, (1, 1), (1, 1), (0, 0), true),
                BatchNorm2d::new(branch_channels),
            ))
        } else {
            None
        };

        Self {
            branch2_pw1: Conv2d::with_options(inp, branch_channels, (1, 1), (1, 1), (0, 0), true),
            branch2_bn1: BatchNorm2d::new(branch_channels),
            branch2_dw: Conv2d::with_groups(branch_channels, branch_channels, (3, 3), (stride, stride), (1, 1), true, branch_channels),
            branch2_bn2: BatchNorm2d::new(branch_channels),
            branch2_pw2: Conv2d::with_options(branch_channels, branch_channels, (1, 1), (1, 1), (0, 0), true),
            branch2_bn3: BatchNorm2d::new(branch_channels),
            shortcut,
            relu: ReLU,
            stride,
            _in_channels: in_channels,
            _out_channels: out_channels,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        if self.stride == 2 {
            // Stride-2: both branches process full input
            let (sc_dw, sc_bn, sc_pw, sc_bn2) = self.shortcut.as_ref().unwrap();
            let branch1 = self.relu.forward(&sc_bn2.forward(&sc_pw.forward(
                &self.relu.forward(&sc_bn.forward(&sc_dw.forward(x))),
            )));

            let branch2 = self.relu.forward(&self.branch2_bn1.forward(&self.branch2_pw1.forward(x)));
            let branch2 = self.branch2_bn2.forward(&self.branch2_dw.forward(&branch2));
            let branch2 = self.relu.forward(&self.branch2_bn3.forward(&self.branch2_pw2.forward(&branch2)));

            // Concat channels
            concat_channels(&branch1, &branch2)
        } else {
            // Stride-1: split channels using narrow (preserves autograd graph)
            let c = x.shape()[1];
            let mid = c / 2;

            let branch1 = x.narrow(1, 0, mid);
            let inp = x.narrow(1, mid, c - mid);

            let branch2 = self.relu.forward(&self.branch2_bn1.forward(&self.branch2_pw1.forward(&inp)));
            let branch2 = self.branch2_bn2.forward(&self.branch2_dw.forward(&branch2));
            let branch2 = self.relu.forward(&self.branch2_bn3.forward(&self.branch2_pw2.forward(&branch2)));

            let out = concat_channels(&branch1, &branch2);
            channel_shuffle(&out, 2)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.branch2_pw1.parameters());
        p.extend(self.branch2_bn1.parameters());
        p.extend(self.branch2_dw.parameters());
        p.extend(self.branch2_bn2.parameters());
        p.extend(self.branch2_pw2.parameters());
        p.extend(self.branch2_bn3.parameters());
        if let Some((dw, bn, pw, bn2)) = &self.shortcut {
            p.extend(dw.parameters());
            p.extend(bn.parameters());
            p.extend(pw.parameters());
            p.extend(bn2.parameters());
        }
        p
    }
}

/// Concatenate two Variables along channel dimension.
fn concat_channels(a: &Variable, b: &Variable) -> Variable {
    Variable::cat(&[a, b], 1)
}

/// ShuffleNet V2 backbone stages.
pub(crate) struct ShuffleNetBackbone {
    stem: Conv2d,
    stem_bn: BatchNorm2d,
    relu: ReLU,
    stages: Vec<Vec<ShuffleBlock>>,
    stage_out_channels: Vec<usize>,
}

impl ShuffleNetBackbone {
    /// Create a 0.5x ShuffleNet V2 backbone (~350K params).
    fn new() -> Self {
        // 0.5x channel config: [24, 48, 96, 192]
        let stage_channels = [48, 96, 192];
        let stage_repeats = [3, 7, 3];

        let mut stages = Vec::new();
        let mut in_ch = 24;

        for (_i, (&out_ch, &repeats)) in stage_channels.iter().zip(stage_repeats.iter()).enumerate() {
            let mut blocks = Vec::new();
            // First block with stride 2
            blocks.push(ShuffleBlock::new(in_ch, out_ch, 2));
            // Remaining blocks with stride 1
            for _ in 1..repeats {
                blocks.push(ShuffleBlock::new(out_ch, out_ch, 1));
            }
            stages.push(blocks);
            in_ch = out_ch;
        }

        Self {
            stem: Conv2d::with_options(3, 24, (3, 3), (2, 2), (1, 1), true),
            stem_bn: BatchNorm2d::new(24),
            relu: ReLU,
            stages,
            stage_out_channels: stage_channels.to_vec(),
        }
    }

    /// Forward pass returning multi-scale features [P3, P4, P5].
    pub(crate) fn forward(&self, x: &Variable) -> Vec<Variable> {
        let mut out = self.relu.forward(&self.stem_bn.forward(&self.stem.forward(x)));
        let mut features = Vec::new();

        for stage in &self.stages {
            for block in stage {
                out = block.forward(&out);
            }
            features.push(out.clone());
        }

        features
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        p.extend(self.stem_bn.parameters());
        for stage in &self.stages {
            for block in stage {
                p.extend(block.parameters());
            }
        }
        p
    }
}

// =============================================================================
// Ghost PAN Neck
// =============================================================================

/// Depthwise separable convolution block.
struct DepthwiseSeparable {
    dw: Conv2d,
    dw_bn: BatchNorm2d,
    pw: Conv2d,
    pw_bn: BatchNorm2d,
    relu: ReLU,
}

impl DepthwiseSeparable {
    fn new(in_ch: usize, out_ch: usize) -> Self {
        Self {
            dw: Conv2d::with_groups(in_ch, in_ch, (3, 3), (1, 1), (1, 1), true, in_ch),
            dw_bn: BatchNorm2d::new(in_ch),
            pw: Conv2d::with_options(in_ch, out_ch, (1, 1), (1, 1), (0, 0), true),
            pw_bn: BatchNorm2d::new(out_ch),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let out = self.relu.forward(&self.dw_bn.forward(&self.dw.forward(x)));
        self.relu.forward(&self.pw_bn.forward(&self.pw.forward(&out)))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.dw.parameters());
        p.extend(self.dw_bn.parameters());
        p.extend(self.pw.parameters());
        p.extend(self.pw_bn.parameters());
        p
    }
}

/// Ghost PAN (Path Aggregation Network) neck.
///
/// Lightweight feature fusion with depthwise separable convolutions.
pub(crate) struct GhostPAN {
    /// Reduce channels from backbone to neck dimension
    reduce: Vec<(Conv2d, BatchNorm2d)>,
    /// Top-down fusion (depthwise separable)
    top_down: Vec<DepthwiseSeparable>,
    /// Bottom-up fusion (depthwise separable)
    bottom_up: Vec<DepthwiseSeparable>,
    /// Downsampling convs for bottom-up path
    downsample: Vec<(Conv2d, BatchNorm2d)>,
    relu: ReLU,
    _neck_channels: usize,
}

impl GhostPAN {
    fn new(in_channels: &[usize], neck_channels: usize) -> Self {
        let num_levels = in_channels.len();

        let reduce: Vec<_> = in_channels
            .iter()
            .map(|&c| {
                (
                    Conv2d::with_options(c, neck_channels, (1, 1), (1, 1), (0, 0), true),
                    BatchNorm2d::new(neck_channels),
                )
            })
            .collect();

        let top_down: Vec<_> = (0..num_levels - 1)
            .map(|_| DepthwiseSeparable::new(neck_channels, neck_channels))
            .collect();

        let bottom_up: Vec<_> = (0..num_levels - 1)
            .map(|_| DepthwiseSeparable::new(neck_channels, neck_channels))
            .collect();

        let downsample: Vec<_> = (0..num_levels - 1)
            .map(|_| {
                (
                    Conv2d::with_groups(neck_channels, neck_channels, (3, 3), (2, 2), (1, 1), true, neck_channels),
                    BatchNorm2d::new(neck_channels),
                )
            })
            .collect();

        Self {
            reduce,
            top_down,
            bottom_up,
            downsample,
            relu: ReLU,
            _neck_channels: neck_channels,
        }
    }

    pub(crate) fn forward(&self, features: &[Variable]) -> Vec<Variable> {
        let num = features.len();

        // 1. Reduce all feature maps to neck_channels
        let mut reduced: Vec<Variable> = features
            .iter()
            .zip(self.reduce.iter())
            .map(|(f, (conv, bn))| self.relu.forward(&bn.forward(&conv.forward(f))))
            .collect();

        // 2. Top-down pathway (coarse -> fine)
        for i in (0..num - 1).rev() {
            let coarse = &reduced[i + 1];
            let shape = reduced[i].shape();
            let (target_h, target_w) = (shape[2], shape[3]);

            let upsampled = crate::ops::interpolate(
                &coarse.data(),
                target_h,
                target_w,
                crate::ops::InterpolateMode::Nearest,
            );
            let up_var = Variable::new(upsampled, coarse.requires_grad());
            let fused = reduced[i].add_var(&up_var);
            reduced[i] = self.top_down[i].forward(&fused);
        }

        // 3. Bottom-up pathway (fine -> coarse)
        for i in 0..num - 1 {
            let fine = &reduced[i];
            let (conv, bn) = &self.downsample[i];
            let downsampled = self.relu.forward(&bn.forward(&conv.forward(fine)));
            let fused = reduced[i + 1].add_var(&downsampled);
            reduced[i + 1] = self.bottom_up[i].forward(&fused);
        }

        reduced
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for (conv, bn) in &self.reduce {
            p.extend(conv.parameters());
            p.extend(bn.parameters());
        }
        for layer in &self.top_down {
            p.extend(layer.parameters());
        }
        for layer in &self.bottom_up {
            p.extend(layer.parameters());
        }
        for (conv, bn) in &self.downsample {
            p.extend(conv.parameters());
            p.extend(bn.parameters());
        }
        p
    }
}

// =============================================================================
// Detection Head
// =============================================================================

/// Anchor-free detection head for NanoDet.
///
/// Predicts class scores and bounding boxes at each spatial location.
pub(crate) struct NanoDetHead {
    /// Shared convolution layers
    shared: Vec<(Conv2d, BatchNorm2d)>,
    /// Classification output (per level)
    cls_out: Conv2d,
    /// Bounding box regression output (per level)
    bbox_out: Conv2d,
    relu: ReLU,
    _num_classes: usize,
}

impl NanoDetHead {
    fn new(in_channels: usize, num_classes: usize) -> Self {
        // 2 shared depthwise-separable-like conv layers
        let shared = vec![
            (
                Conv2d::with_options(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true),
                BatchNorm2d::new(in_channels),
            ),
            (
                Conv2d::with_options(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true),
                BatchNorm2d::new(in_channels),
            ),
        ];

        Self {
            shared,
            cls_out: Conv2d::with_options(in_channels, num_classes, (1, 1), (1, 1), (0, 0), true),
            bbox_out: Conv2d::with_options(in_channels, 4, (1, 1), (1, 1), (0, 0), true),
            relu: ReLU,
            _num_classes: num_classes,
        }
    }

    /// Forward on a single feature level.
    pub(crate) fn forward_single(&self, x: &Variable) -> (Variable, Variable) {
        let mut out = x.clone();
        for (conv, bn) in &self.shared {
            out = self.relu.forward(&bn.forward(&conv.forward(&out)));
        }

        let cls = self.cls_out.forward(&out);
        let bbox = self.bbox_out.forward(&out);
        (cls, bbox)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for (conv, bn) in &self.shared {
            p.extend(conv.parameters());
            p.extend(bn.parameters());
        }
        p.extend(self.cls_out.parameters());
        p.extend(self.bbox_out.parameters());
        p
    }
}

// =============================================================================
// NanoDet
// =============================================================================

/// NanoDet — Lightweight anchor-free object detector for edge deployment.
///
/// Architecture: ShuffleNet V2 backbone + Ghost PAN neck + anchor-free head.
/// Designed for <1M parameters and real-time inference on ARM/mobile devices.
pub struct NanoDet {
    pub(crate) backbone: ShuffleNetBackbone,
    pub(crate) neck: GhostPAN,
    pub(crate) head: NanoDetHead,
    num_classes: usize,
    strides: Vec<usize>,
}

impl NanoDet {
    /// Create a NanoDet model.
    ///
    /// # Arguments
    /// - `num_classes`: Number of object classes (e.g., 80 for COCO)
    pub fn new(num_classes: usize) -> Self {
        let backbone = ShuffleNetBackbone::new();
        let neck_channels = 96;

        Self {
            neck: GhostPAN::new(&backbone.stage_out_channels, neck_channels),
            head: NanoDetHead::new(neck_channels, num_classes),
            backbone,
            num_classes,
            strides: vec![8, 16, 32],
        }
    }

    /// Run detection on an input image.
    ///
    /// # Arguments
    /// - `image`: `[N, 3, H, W]` input tensor
    /// - `score_threshold`: Minimum confidence score
    /// - `nms_threshold`: IoU threshold for NMS
    ///
    /// # Returns
    /// Vector of detections with class, bbox, and confidence.
    pub fn detect(
        &self,
        image: &Variable,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Vec<Detection> {
        let features = self.backbone.forward(image);
        let neck_features = self.neck.forward(&features);

        let input_shape = image.shape();
        let img_h = input_shape[2] as f32;
        let img_w = input_shape[3] as f32;

        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();
        let mut all_classes = Vec::new();

        for (level, feat) in neck_features.iter().enumerate() {
            let (cls, bbox) = self.head.forward_single(feat);
            let cls_data = cls.data().to_vec();
            let bbox_data = bbox.data().to_vec();
            let shape = cls.shape();
            let (_n, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let stride = self.strides[level] as f32;

            for y in 0..h {
                for x in 0..w {
                    // Find best class at this location
                    let mut best_cls = 0;
                    let mut best_score = f32::NEG_INFINITY;

                    for c in 0..self.num_classes {
                        let idx = c * h * w + y * w + x;
                        let score = 1.0 / (1.0 + (-cls_data[idx]).exp()); // sigmoid
                        if score > best_score {
                            best_score = score;
                            best_cls = c;
                        }
                    }

                    if best_score < score_threshold {
                        continue;
                    }

                    // Decode bbox (center offset + size)
                    let dx = bbox_data[0 * h * w + y * w + x];
                    let dy = bbox_data[1 * h * w + y * w + x];
                    let dw = bbox_data[2 * h * w + y * w + x];
                    let dh = bbox_data[3 * h * w + y * w + x];

                    let cx = (x as f32 + 0.5) * stride + dx * stride;
                    let cy = (y as f32 + 0.5) * stride + dy * stride;
                    let bw = (dw.exp() * stride).min(img_w);
                    let bh = (dh.exp() * stride).min(img_h);

                    all_boxes.push([
                        (cx - bw / 2.0).max(0.0),
                        (cy - bh / 2.0).max(0.0),
                        (cx + bw / 2.0).min(img_w),
                        (cy + bh / 2.0).min(img_h),
                    ]);
                    all_scores.push(best_score);
                    all_classes.push(best_cls);
                }
            }
        }

        if all_scores.is_empty() {
            return vec![];
        }

        // Per-class NMS
        let mut results = Vec::new();
        for cls in 0..self.num_classes {
            let indices: Vec<usize> = (0..all_scores.len())
                .filter(|&i| all_classes[i] == cls)
                .collect();

            if indices.is_empty() {
                continue;
            }

            let cls_boxes: Vec<f32> = indices
                .iter()
                .flat_map(|&i| all_boxes[i].iter().copied())
                .collect();
            let cls_scores: Vec<f32> = indices.iter().map(|&i| all_scores[i]).collect();

            let n = indices.len();
            let boxes_t = Tensor::from_vec(cls_boxes, &[n, 4]).unwrap();
            let scores_t = Tensor::from_vec(cls_scores.clone(), &[n]).unwrap();
            let keep = nms(&boxes_t, &scores_t, nms_threshold);

            for &k in &keep {
                let orig_idx = indices[k];
                results.push(Detection {
                    class_id: cls,
                    bbox: all_boxes[orig_idx],
                    confidence: all_scores[orig_idx],
                });
            }
        }

        results
    }
}

impl Module for NanoDet {
    fn forward(&self, x: &Variable) -> Variable {
        let features = self.backbone.forward(x);
        let neck_features = self.neck.forward(&features);
        let (cls, _bbox) = self.head.forward_single(&neck_features[0]);
        cls
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters());
        p.extend(self.neck.parameters());
        p.extend(self.head.parameters());
        p
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

    #[test]
    fn test_shuffle_block_stride1() {
        let block = ShuffleBlock::new(48, 48, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 48 * 8 * 8], &[1, 48, 8, 8]).unwrap(),
            false,
        );
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![1, 48, 8, 8]);
    }

    #[test]
    fn test_shuffle_block_stride2() {
        let block = ShuffleBlock::new(24, 48, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 24 * 16 * 16], &[1, 24, 16, 16]).unwrap(),
            false,
        );
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![1, 48, 8, 8]);
    }

    #[test]
    fn test_nanodet_creation() {
        let model = NanoDet::new(80);
        let params = model.parameters();
        assert!(!params.is_empty());

        // Should be lightweight (<1M params)
        let total: usize = params.iter().map(|p| p.variable().data().to_vec().len()).sum();
        assert!(total < 1_000_000, "NanoDet has {} params, expected < 1M", total);
    }

    #[test]
    fn test_nanodet_forward() {
        let model = NanoDet::new(20);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape()[0], 1);
        // Output should have num_classes channels
        assert_eq!(output.shape()[1], 20);
    }

    #[test]
    fn test_channel_shuffle() {
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 4 * 2 * 2], &[1, 4, 2, 2]).unwrap(),
            false,
        );
        let output = channel_shuffle(&input, 2);
        assert_eq!(output.shape(), vec![1, 4, 2, 2]);
    }
}
