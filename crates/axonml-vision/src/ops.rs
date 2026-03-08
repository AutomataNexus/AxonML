//! Vision Operations - Detection and Spatial Primitives
//!
//! # File
//! `crates/axonml-vision/src/ops.rs`
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
use axonml_tensor::Tensor;

// =============================================================================
// Bounding Box Utilities
// =============================================================================

/// Compute Intersection over Union (IoU) between two sets of boxes.
///
/// # Arguments
/// - `boxes1`: `[N, 4]` tensor in `(x1, y1, x2, y2)` format
/// - `boxes2`: `[M, 4]` tensor in `(x1, y1, x2, y2)` format
///
/// # Returns
/// `[N, M]` tensor of pairwise IoU values.
pub fn box_iou(boxes1: &Tensor<f32>, boxes2: &Tensor<f32>) -> Tensor<f32> {
    let n = boxes1.shape()[0];
    let m = boxes2.shape()[0];
    let b1 = boxes1.to_vec();
    let b2 = boxes2.to_vec();

    let mut iou = vec![0.0f32; n * m];

    for i in 0..n {
        let x1_a = b1[i * 4];
        let y1_a = b1[i * 4 + 1];
        let x2_a = b1[i * 4 + 2];
        let y2_a = b1[i * 4 + 3];
        let area_a = (x2_a - x1_a).max(0.0) * (y2_a - y1_a).max(0.0);

        for j in 0..m {
            let x1_b = b2[j * 4];
            let y1_b = b2[j * 4 + 1];
            let x2_b = b2[j * 4 + 2];
            let y2_b = b2[j * 4 + 3];
            let area_b = (x2_b - x1_b).max(0.0) * (y2_b - y1_b).max(0.0);

            // Intersection
            let ix1 = x1_a.max(x1_b);
            let iy1 = y1_a.max(y1_b);
            let ix2 = x2_a.min(x2_b);
            let iy2 = y2_a.min(y2_b);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);

            // Union
            let union = area_a + area_b - inter;
            iou[i * m + j] = if union > 0.0 { inter / union } else { 0.0 };
        }
    }

    Tensor::from_vec(iou, &[n, m]).unwrap()
}

/// Convert boxes from `(cx, cy, w, h)` to `(x1, y1, x2, y2)` format.
pub fn box_cxcywh_to_xyxy(boxes: &Tensor<f32>) -> Tensor<f32> {
    let data = boxes.to_vec();
    let n = boxes.shape()[0];
    let mut result = vec![0.0f32; n * 4];

    for i in 0..n {
        let cx = data[i * 4];
        let cy = data[i * 4 + 1];
        let w = data[i * 4 + 2];
        let h = data[i * 4 + 3];
        result[i * 4] = cx - w / 2.0;
        result[i * 4 + 1] = cy - h / 2.0;
        result[i * 4 + 2] = cx + w / 2.0;
        result[i * 4 + 3] = cy + h / 2.0;
    }

    Tensor::from_vec(result, &[n, 4]).unwrap()
}

/// Convert boxes from `(x1, y1, x2, y2)` to `(cx, cy, w, h)` format.
pub fn box_xyxy_to_cxcywh(boxes: &Tensor<f32>) -> Tensor<f32> {
    let data = boxes.to_vec();
    let n = boxes.shape()[0];
    let mut result = vec![0.0f32; n * 4];

    for i in 0..n {
        let x1 = data[i * 4];
        let y1 = data[i * 4 + 1];
        let x2 = data[i * 4 + 2];
        let y2 = data[i * 4 + 3];
        result[i * 4] = (x1 + x2) / 2.0;
        result[i * 4 + 1] = (y1 + y2) / 2.0;
        result[i * 4 + 2] = x2 - x1;
        result[i * 4 + 3] = y2 - y1;
    }

    Tensor::from_vec(result, &[n, 4]).unwrap()
}

// =============================================================================
// Non-Maximum Suppression
// =============================================================================

/// Non-Maximum Suppression (NMS).
///
/// Filters overlapping bounding boxes by keeping only the highest-scoring
/// non-overlapping detections.
///
/// # Arguments
/// - `boxes`: `[N, 4]` tensor in `(x1, y1, x2, y2)` format
/// - `scores`: `[N]` tensor of confidence scores
/// - `iou_threshold`: IoU threshold for suppression (e.g., 0.5)
///
/// # Returns
/// Vector of kept indices, sorted by descending score.
pub fn nms(boxes: &Tensor<f32>, scores: &Tensor<f32>, iou_threshold: f32) -> Vec<usize> {
    let n = boxes.shape()[0];
    if n == 0 {
        return vec![];
    }

    let scores_vec = scores.to_vec();
    let boxes_vec = boxes.to_vec();

    // Sort indices by descending score
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores_vec[b].partial_cmp(&scores_vec[a]).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; n];

    for pos in 0..order.len() {
        let idx = order[pos];
        if suppressed[idx] {
            continue;
        }
        keep.push(idx);

        let x1_a = boxes_vec[idx * 4];
        let y1_a = boxes_vec[idx * 4 + 1];
        let x2_a = boxes_vec[idx * 4 + 2];
        let y2_a = boxes_vec[idx * 4 + 3];
        let area_a = (x2_a - x1_a).max(0.0) * (y2_a - y1_a).max(0.0);

        // Only check boxes after current position in score-sorted order;
        // earlier boxes are either kept or already suppressed.
        for &other in &order[pos + 1..] {
            if suppressed[other] {
                continue;
            }

            let x1_b = boxes_vec[other * 4];
            let y1_b = boxes_vec[other * 4 + 1];
            let x2_b = boxes_vec[other * 4 + 2];
            let y2_b = boxes_vec[other * 4 + 3];
            let area_b = (x2_b - x1_b).max(0.0) * (y2_b - y1_b).max(0.0);

            let ix1 = x1_a.max(x1_b);
            let iy1 = y1_a.max(y1_b);
            let ix2 = x2_a.min(x2_b);
            let iy2 = y2_a.min(y2_b);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
            let union = area_a + area_b - inter;

            let iou = if union > 0.0 { inter / union } else { 0.0 };
            if iou > iou_threshold {
                suppressed[other] = true;
            }
        }
    }

    keep
}

/// Batched NMS — applies NMS per class.
///
/// # Arguments
/// - `boxes`: `[N, 4]` tensor
/// - `scores`: `[N]` tensor
/// - `class_ids`: `[N]` tensor of integer class IDs (as f32)
/// - `iou_threshold`: IoU threshold
///
/// # Returns
/// Vector of kept indices.
pub fn batched_nms(
    boxes: &Tensor<f32>,
    scores: &Tensor<f32>,
    class_ids: &Tensor<f32>,
    iou_threshold: f32,
) -> Vec<usize> {
    let n = boxes.shape()[0];
    if n == 0 {
        return vec![];
    }

    let boxes_vec = boxes.to_vec();
    let class_vec = class_ids.to_vec();

    // Offset boxes by class ID to prevent cross-class suppression
    let max_coord = boxes_vec.iter().copied().fold(0.0f32, f32::max);
    let mut offset_boxes = vec![0.0f32; n * 4];

    for i in 0..n {
        let class_offset = class_vec[i] * (max_coord + 1.0);
        offset_boxes[i * 4] = boxes_vec[i * 4] + class_offset;
        offset_boxes[i * 4 + 1] = boxes_vec[i * 4 + 1] + class_offset;
        offset_boxes[i * 4 + 2] = boxes_vec[i * 4 + 2] + class_offset;
        offset_boxes[i * 4 + 3] = boxes_vec[i * 4 + 3] + class_offset;
    }

    let offset_tensor = Tensor::from_vec(offset_boxes, &[n, 4]).unwrap();
    nms(&offset_tensor, scores, iou_threshold)
}

// =============================================================================
// Anchor Generation
// =============================================================================

/// Generate anchor boxes for a single feature map level.
///
/// # Arguments
/// - `feature_h`: Feature map height
/// - `feature_w`: Feature map width
/// - `stride`: Stride of this feature level relative to input
/// - `sizes`: Anchor sizes (absolute pixels)
/// - `ratios`: Anchor aspect ratios (width / height)
///
/// # Returns
/// `[feature_h * feature_w * len(sizes) * len(ratios), 4]` tensor in `(x1, y1, x2, y2)` format.
pub fn generate_anchors(
    feature_h: usize,
    feature_w: usize,
    stride: usize,
    sizes: &[f32],
    ratios: &[f32],
) -> Tensor<f32> {
    let num_anchors_per_cell = sizes.len() * ratios.len();
    let total = feature_h * feature_w * num_anchors_per_cell;
    let mut anchors = vec![0.0f32; total * 4];

    let mut idx = 0;
    for fy in 0..feature_h {
        for fx in 0..feature_w {
            let cx = (fx as f32 + 0.5) * stride as f32;
            let cy = (fy as f32 + 0.5) * stride as f32;

            for &size in sizes {
                for &ratio in ratios {
                    let w = size * ratio.sqrt();
                    let h = size / ratio.sqrt();
                    anchors[idx * 4] = cx - w / 2.0;
                    anchors[idx * 4 + 1] = cy - h / 2.0;
                    anchors[idx * 4 + 2] = cx + w / 2.0;
                    anchors[idx * 4 + 3] = cy + h / 2.0;
                    idx += 1;
                }
            }
        }
    }

    Tensor::from_vec(anchors, &[total, 4]).unwrap()
}

/// Generate multi-scale anchors across multiple feature pyramid levels.
///
/// # Arguments
/// - `feature_sizes`: `[(h, w)]` for each pyramid level
/// - `strides`: Stride for each level
/// - `sizes_per_level`: Anchor sizes per level
/// - `ratios`: Aspect ratios (shared across levels)
///
/// # Returns
/// `[total_anchors, 4]` tensor.
pub fn generate_multi_scale_anchors(
    feature_sizes: &[(usize, usize)],
    strides: &[usize],
    sizes_per_level: &[Vec<f32>],
    ratios: &[f32],
) -> Tensor<f32> {
    let mut all_anchors = Vec::new();
    let mut total = 0;

    for (i, &(fh, fw)) in feature_sizes.iter().enumerate() {
        let level_anchors = generate_anchors(fh, fw, strides[i], &sizes_per_level[i], ratios);
        let n = level_anchors.shape()[0];
        all_anchors.extend(level_anchors.to_vec());
        total += n;
    }

    Tensor::from_vec(all_anchors, &[total, 4]).unwrap()
}

// =============================================================================
// Interpolate (Bilinear / Nearest Upsampling)
// =============================================================================

/// Interpolation mode for resizing feature maps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolateMode {
    /// Nearest-neighbor interpolation.
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
}

/// Resize a 4D tensor `[N, C, H, W]` to target `(out_h, out_w)`.
///
/// This is a non-differentiable utility. For differentiable upsampling
/// in model architectures, use `Upsample`.
pub fn interpolate(
    input: &Tensor<f32>,
    out_h: usize,
    out_w: usize,
    mode: InterpolateMode,
) -> Tensor<f32> {
    let shape = input.shape();
    assert!(shape.len() == 4, "interpolate expects [N, C, H, W]");
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let data = input.to_vec();
    let mut output = vec![0.0f32; n * c * out_h * out_w];

    let scale_h = h as f32 / out_h as f32;
    let scale_w = w as f32 / out_w as f32;

    for batch in 0..n {
        for ch in 0..c {
            let base_in = batch * c * h * w + ch * h * w;
            let base_out = batch * c * out_h * out_w + ch * out_h * out_w;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let val = match mode {
                        InterpolateMode::Nearest => {
                            let ih = ((oh as f32 + 0.5) * scale_h) as usize;
                            let iw = ((ow as f32 + 0.5) * scale_w) as usize;
                            let ih = ih.min(h - 1);
                            let iw = iw.min(w - 1);
                            data[base_in + ih * w + iw]
                        }
                        InterpolateMode::Bilinear => {
                            let src_h = (oh as f32 + 0.5) * scale_h - 0.5;
                            let src_w = (ow as f32 + 0.5) * scale_w - 0.5;

                            let h0 = src_h.floor() as i32;
                            let w0 = src_w.floor() as i32;
                            let h1 = h0 + 1;
                            let w1 = w0 + 1;

                            let hf = src_h - h0 as f32;
                            let wf = src_w - w0 as f32;

                            let sample = |iy: i32, ix: i32| -> f32 {
                                let iy = iy.clamp(0, h as i32 - 1) as usize;
                                let ix = ix.clamp(0, w as i32 - 1) as usize;
                                data[base_in + iy * w + ix]
                            };

                            let v00 = sample(h0, w0);
                            let v01 = sample(h0, w1);
                            let v10 = sample(h1, w0);
                            let v11 = sample(h1, w1);

                            v00 * (1.0 - hf) * (1.0 - wf)
                                + v01 * (1.0 - hf) * wf
                                + v10 * hf * (1.0 - wf)
                                + v11 * hf * wf
                        }
                    };
                    output[base_out + oh * out_w + ow] = val;
                }
            }
        }
    }

    Tensor::from_vec(output, &[n, c, out_h, out_w]).unwrap()
}

// =============================================================================
// Upsample Module (Differentiable)
// =============================================================================

/// Differentiable upsampling module for use in neural networks.
///
/// Wraps `interpolate` with Variable support. Uses nearest-neighbor
/// upsampling which doesn't require gradient computation through the
/// sampling grid (gradients flow through the values, not the coordinates).
pub struct Upsample {
    scale_factor: usize,
    mode: InterpolateMode,
}

impl Upsample {
    /// Create a new Upsample with a scale factor.
    pub fn new(scale_factor: usize) -> Self {
        Self {
            scale_factor,
            mode: InterpolateMode::Nearest,
        }
    }

    /// Create with bilinear interpolation.
    pub fn bilinear(scale_factor: usize) -> Self {
        Self {
            scale_factor,
            mode: InterpolateMode::Bilinear,
        }
    }

    /// Forward pass — upsamples the input feature map.
    pub fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let out_h = shape[2] * self.scale_factor;
        let out_w = shape[3] * self.scale_factor;
        let output = interpolate(&input.data(), out_h, out_w, self.mode);

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = GradFn::new(InterpolateBackward {
                next_fns: vec![input.grad_fn().cloned()],
                input_shape: shape.to_vec(),
                mode: self.mode,
            });
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
    }
}

/// Variable-level interpolate that preserves the autograd graph.
pub fn interpolate_var(
    input: &Variable,
    out_h: usize,
    out_w: usize,
    mode: InterpolateMode,
) -> Variable {
    let output = interpolate(&input.data(), out_h, out_w, mode);
    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = GradFn::new(InterpolateBackward {
            next_fns: vec![input.grad_fn().cloned()],
            input_shape: input.shape().to_vec(),
            mode,
        });
        Variable::from_operation(output, grad_fn, true)
    } else {
        Variable::new(output, false)
    }
}

// =============================================================================
// InterpolateBackward
// =============================================================================

/// Gradient function for interpolation (nearest/bilinear).
#[derive(Debug)]
struct InterpolateBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    mode: InterpolateMode,
}

impl GradientFunction for InterpolateBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let out_shape = grad_output.shape();
        let (n, c, out_h, out_w) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
        let (h, w) = (self.input_shape[2], self.input_shape[3]);

        let g_vec = grad_output.to_vec();
        let mut grad_input = vec![0.0f32; n * c * h * w];

        let scale_h = h as f32 / out_h as f32;
        let scale_w = w as f32 / out_w as f32;

        for batch in 0..n {
            for ch in 0..c {
                let base_in = batch * c * h * w + ch * h * w;
                let base_out = batch * c * out_h * out_w + ch * out_h * out_w;

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let g = g_vec[base_out + oh * out_w + ow];
                        match self.mode {
                            InterpolateMode::Nearest => {
                                let ih = ((oh as f32 + 0.5) * scale_h) as usize;
                                let iw = ((ow as f32 + 0.5) * scale_w) as usize;
                                let ih = ih.min(h - 1);
                                let iw = iw.min(w - 1);
                                grad_input[base_in + ih * w + iw] += g;
                            }
                            InterpolateMode::Bilinear => {
                                let src_h = (oh as f32 + 0.5) * scale_h - 0.5;
                                let src_w = (ow as f32 + 0.5) * scale_w - 0.5;

                                let h0 = src_h.floor() as i32;
                                let w0 = src_w.floor() as i32;
                                let h1 = h0 + 1;
                                let w1 = w0 + 1;

                                let hf = src_h - h0 as f32;
                                let wf = src_w - w0 as f32;

                                let add_grad = |iy: i32, ix: i32, weight: f32, gi: &mut [f32]| {
                                    let iy = iy.clamp(0, h as i32 - 1) as usize;
                                    let ix = ix.clamp(0, w as i32 - 1) as usize;
                                    gi[base_in + iy * w + ix] += g * weight;
                                };

                                add_grad(h0, w0, (1.0 - hf) * (1.0 - wf), &mut grad_input);
                                add_grad(h0, w1, (1.0 - hf) * wf, &mut grad_input);
                                add_grad(h1, w0, hf * (1.0 - wf), &mut grad_input);
                                add_grad(h1, w1, hf * wf, &mut grad_input);
                            }
                        }
                    }
                }
            }
        }

        let gi = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "InterpolateBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// RoI Align
// =============================================================================

/// Region of Interest Align.
///
/// Extracts fixed-size feature maps from regions of interest using
/// bilinear interpolation (no quantization artifacts unlike RoI Pool).
///
/// # Arguments
/// - `features`: `[N, C, H, W]` feature map
/// - `rois`: `[K, 5]` tensor where each row is `(batch_idx, x1, y1, x2, y2)`
/// - `output_size`: `(out_h, out_w)` for each RoI
/// - `spatial_scale`: Scale factor from input image coords to feature map coords
///
/// # Returns
/// `[K, C, out_h, out_w]` tensor.
pub fn roi_align(
    features: &Tensor<f32>,
    rois: &Tensor<f32>,
    output_size: (usize, usize),
    spatial_scale: f32,
) -> Tensor<f32> {
    let feat_shape = features.shape();
    let (c, h, w) = (feat_shape[1], feat_shape[2], feat_shape[3]);
    let feat_data = features.to_vec();
    let roi_data = rois.to_vec();
    let k = rois.shape()[0];
    let (out_h, out_w) = output_size;

    let mut output = vec![0.0f32; k * c * out_h * out_w];

    for roi_idx in 0..k {
        let batch_idx = roi_data[roi_idx * 5] as usize;
        let x1 = roi_data[roi_idx * 5 + 1] * spatial_scale;
        let y1 = roi_data[roi_idx * 5 + 2] * spatial_scale;
        let x2 = roi_data[roi_idx * 5 + 3] * spatial_scale;
        let y2 = roi_data[roi_idx * 5 + 4] * spatial_scale;

        let roi_w = (x2 - x1).max(1e-6);
        let roi_h = (y2 - y1).max(1e-6);
        let bin_h = roi_h / out_h as f32;
        let bin_w = roi_w / out_w as f32;

        let feat_base = batch_idx * c * h * w;

        for ch in 0..c {
            let ch_base = feat_base + ch * h * w;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    // Sample center of each bin
                    let src_y = y1 + (oh as f32 + 0.5) * bin_h;
                    let src_x = x1 + (ow as f32 + 0.5) * bin_w;

                    // Bilinear interpolation
                    let iy0 = src_y.floor() as i32;
                    let ix0 = src_x.floor() as i32;
                    let iy1 = iy0 + 1;
                    let ix1 = ix0 + 1;

                    let hy = src_y - iy0 as f32;
                    let hx = src_x - ix0 as f32;

                    let sample = |iy: i32, ix: i32| -> f32 {
                        if iy < 0 || iy >= h as i32 || ix < 0 || ix >= w as i32 {
                            return 0.0;
                        }
                        feat_data[ch_base + iy as usize * w + ix as usize]
                    };

                    let val = sample(iy0, ix0) * (1.0 - hy) * (1.0 - hx)
                        + sample(iy0, ix1) * (1.0 - hy) * hx
                        + sample(iy1, ix0) * hy * (1.0 - hx)
                        + sample(iy1, ix1) * hy * hx;

                    output[roi_idx * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow] =
                        val;
                }
            }
        }
    }

    Tensor::from_vec(output, &[k, c, out_h, out_w]).unwrap()
}

// =============================================================================
// Detection Output Types
// =============================================================================

/// A single detected object.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in `(x1, y1, x2, y2)` format.
    pub bbox: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
    /// Class ID.
    pub class_id: usize,
}

/// A detected face with optional landmarks.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// Bounding box in `(x1, y1, x2, y2)` format.
    pub bbox: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
    /// 5-point facial landmarks `[(x, y); 5]` — eyes, nose, mouth corners.
    pub landmarks: Option<[(f32, f32); 5]>,
}

/// Anomaly detection result.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Global anomaly score (higher = more anomalous).
    pub score: f32,
    /// Whether the sample is considered anomalous.
    pub is_anomalous: bool,
    /// Per-pixel anomaly heatmap `[H, W]` (optional).
    pub heatmap: Option<Tensor<f32>>,
}

/// A Nexus detection with uncertainty quantification.
#[derive(Debug, Clone)]
pub struct NexusDetection {
    /// Bounding box mean in `(x1, y1, x2, y2)` format.
    pub bbox_mean: [f32; 4],
    /// Bounding box log-variance (aleatoric uncertainty) for each coordinate.
    pub bbox_log_var: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
    /// Class ID.
    pub class_id: usize,
    /// Tracking ID (persistent across frames via object memory).
    pub tracking_id: u64,
    /// Number of frames this object has been tracked.
    pub frames_tracked: u32,
}

/// A Phantom temporal face detection with tracking.
#[derive(Debug, Clone)]
pub struct PhantomFaceDetection {
    /// Bounding box in `(x1, y1, x2, y2)` format.
    pub bbox: [f32; 4],
    /// Accumulated confidence score (builds over time).
    pub confidence: f32,
    /// Tracking ID (persistent identity via temporal continuity).
    pub tracking_id: u64,
    /// Velocity vector `(vx, vy)` in pixels per frame.
    pub velocity: [f32; 2],
    /// Number of consecutive frames this face has been tracked.
    pub frames_tracked: u32,
}

/// Depth estimation output.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Dense depth map `[H, W]` with relative depth values.
    pub depth: Tensor<f32>,
    /// Minimum depth value.
    pub min_depth: f32,
    /// Maximum depth value.
    pub max_depth: f32,
}

// =============================================================================
// 2D Positional Encoding (for DETR)
// =============================================================================

/// Generate 2D sinusoidal positional encoding for spatial feature maps.
///
/// Used by DETR and other detection transformers.
///
/// # Arguments
/// - `h`: Feature map height
/// - `w`: Feature map width
/// - `d_model`: Embedding dimension (must be divisible by 4)
///
/// # Returns
/// `[1, d_model, h, w]` tensor.
pub fn positional_encoding_2d(h: usize, w: usize, d_model: usize) -> Tensor<f32> {
    assert!(d_model % 4 == 0, "d_model must be divisible by 4");
    let half_d = d_model / 2;
    let quarter_d = d_model / 4;

    let mut encoding = vec![0.0f32; d_model * h * w];

    // Precompute frequency table to avoid repeated powf in inner loops
    let freqs: Vec<f32> = (0..quarter_d)
        .map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / half_d as f32))
        .collect();

    for y in 0..h {
        for x in 0..w {
            // Height encoding (first half)
            for i in 0..quarter_d {
                let freq = freqs[i];
                let pos = y as f32;
                encoding[(2 * i) * h * w + y * w + x] = (pos * freq).sin();
                encoding[(2 * i + 1) * h * w + y * w + x] = (pos * freq).cos();
            }
            // Width encoding (second half)
            for i in 0..quarter_d {
                let freq = freqs[i];
                let pos = x as f32;
                encoding[(half_d + 2 * i) * h * w + y * w + x] = (pos * freq).sin();
                encoding[(half_d + 2 * i + 1) * h * w + y * w + x] = (pos * freq).cos();
            }
        }
    }

    Tensor::from_vec(encoding, &[1, d_model, h, w]).unwrap()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_iou_identical() {
        let boxes = Tensor::from_vec(vec![0.0, 0.0, 10.0, 10.0], &[1, 4]).unwrap();
        let iou = box_iou(&boxes, &boxes);
        assert!((iou.to_vec()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_box_iou_no_overlap() {
        let b1 = Tensor::from_vec(vec![0.0, 0.0, 5.0, 5.0], &[1, 4]).unwrap();
        let b2 = Tensor::from_vec(vec![10.0, 10.0, 20.0, 20.0], &[1, 4]).unwrap();
        let iou = box_iou(&b1, &b2);
        assert!(iou.to_vec()[0] < 1e-5);
    }

    #[test]
    fn test_box_iou_partial_overlap() {
        let b1 = Tensor::from_vec(vec![0.0, 0.0, 10.0, 10.0], &[1, 4]).unwrap();
        let b2 = Tensor::from_vec(vec![5.0, 5.0, 15.0, 15.0], &[1, 4]).unwrap();
        let iou = box_iou(&b1, &b2);
        // Intersection: 5x5=25, Union: 100+100-25=175
        let expected = 25.0 / 175.0;
        assert!((iou.to_vec()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_box_iou_batch() {
        let b1 = Tensor::from_vec(
            vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 5.0, 5.0],
            &[2, 4],
        )
        .unwrap();
        let b2 = Tensor::from_vec(vec![0.0, 0.0, 10.0, 10.0], &[1, 4]).unwrap();
        let iou = box_iou(&b1, &b2);
        assert_eq!(iou.shape(), &[2, 1]);
        assert!((iou.to_vec()[0] - 1.0).abs() < 1e-5);
        assert!((iou.to_vec()[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_box_format_conversion() {
        let xyxy = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]).unwrap();
        let cxcywh = box_xyxy_to_cxcywh(&xyxy);
        assert_eq!(cxcywh.to_vec(), vec![20.0, 30.0, 20.0, 20.0]);

        let back = box_cxcywh_to_xyxy(&cxcywh);
        let diff: f32 = back
            .to_vec()
            .iter()
            .zip(xyxy.to_vec().iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-5);
    }

    #[test]
    fn test_nms_basic() {
        let boxes = Tensor::from_vec(
            vec![
                0.0, 0.0, 10.0, 10.0,
                1.0, 1.0, 11.0, 11.0,
                50.0, 50.0, 60.0, 60.0,
            ],
            &[3, 4],
        )
        .unwrap();
        let scores = Tensor::from_vec(vec![0.9, 0.8, 0.7], &[3]).unwrap();

        let kept = nms(&boxes, &scores, 0.5);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], 0);
        assert_eq!(kept[1], 2);
    }

    #[test]
    fn test_nms_no_suppression() {
        let boxes = Tensor::from_vec(
            vec![
                0.0, 0.0, 5.0, 5.0,
                10.0, 10.0, 15.0, 15.0,
                20.0, 20.0, 25.0, 25.0,
            ],
            &[3, 4],
        )
        .unwrap();
        let scores = Tensor::from_vec(vec![0.9, 0.8, 0.7], &[3]).unwrap();

        let kept = nms(&boxes, &scores, 0.5);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn test_nms_empty() {
        let boxes = Tensor::<f32>::from_vec(Vec::<f32>::new(), &[0, 4]).unwrap();
        let scores = Tensor::<f32>::from_vec(Vec::<f32>::new(), &[0]).unwrap();
        let kept = nms(&boxes, &scores, 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_generate_anchors() {
        let anchors = generate_anchors(2, 2, 16, &[32.0, 64.0], &[0.5, 1.0, 2.0]);
        assert_eq!(anchors.shape(), &[24, 4]);

        let data = anchors.to_vec();
        for i in 0..24 {
            let w = data[i * 4 + 2] - data[i * 4];
            let h = data[i * 4 + 3] - data[i * 4 + 1];
            assert!(w > 0.0);
            assert!(h > 0.0);
        }
    }

    #[test]
    fn test_interpolate_nearest() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let output = interpolate(&input, 4, 4, InterpolateMode::Nearest);
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_interpolate_bilinear() {
        let input = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let output = interpolate(&input, 4, 4, InterpolateMode::Bilinear);
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_interpolate_identity() {
        let input = Tensor::from_vec(vec![1.0f32; 16], &[1, 1, 4, 4]).unwrap();
        let output = interpolate(&input, 4, 4, InterpolateMode::Bilinear);
        let diff: f32 = output
            .to_vec()
            .iter()
            .zip(input.to_vec().iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-5);
    }

    #[test]
    fn test_roi_align() {
        let features = Tensor::from_vec(
            (0..16).map(|i| i as f32).collect(),
            &[1, 1, 4, 4],
        )
        .unwrap();

        let rois = Tensor::from_vec(vec![0.0, 0.0, 0.0, 4.0, 4.0], &[1, 5]).unwrap();
        let output = roi_align(&features, &rois, (2, 2), 1.0);
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_positional_encoding_2d() {
        let pe = positional_encoding_2d(8, 8, 64);
        assert_eq!(pe.shape(), &[1, 64, 8, 8]);

        let data = pe.to_vec();
        for &v in &data {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_batched_nms() {
        let boxes = Tensor::from_vec(
            vec![
                0.0, 0.0, 10.0, 10.0,
                1.0, 1.0, 11.0, 11.0,
            ],
            &[2, 4],
        )
        .unwrap();
        let scores = Tensor::from_vec(vec![0.9, 0.8], &[2]).unwrap();
        let classes = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let kept = batched_nms(&boxes, &scores, &classes, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_upsample() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).unwrap(),
            false,
        );
        let upsample = Upsample::new(2);
        let output = upsample.forward(&input);
        assert_eq!(output.shape(), vec![1, 1, 4, 4]);
    }
}
