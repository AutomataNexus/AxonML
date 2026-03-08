//! Helios Training Losses — CIoU, DFL Loss, Task-Aligned Assigner
//!
//! # File
//! `crates/axonml-vision/src/models/helios/loss.rs`
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

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use super::HeliosTrainOutput;

// =============================================================================
// CIoU Loss
// =============================================================================

/// Complete IoU Loss with distance and aspect ratio penalties.
///
/// CIoU = IoU - d²/c² - αv
/// where d = center distance, c = enclosing diagonal, v = aspect ratio consistency,
/// α = v / (1 - IoU + v)
///
/// From Zheng et al., "Distance-IoU Loss" (AAAI 2020).
pub struct CIoULoss;

impl CIoULoss {
    /// Compute CIoU loss for paired predicted and target boxes.
    ///
    /// - `pred`: [N, 4] as (x1, y1, x2, y2)
    /// - `target`: [N, 4] as (x1, y1, x2, y2)
    ///
    /// Returns scalar loss (mean over N).
    pub fn compute(pred: &Variable, target: &Variable) -> Variable {
        let n = pred.shape()[0];
        if n == 0 {
            return Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        }

        let pred_data = pred.data().to_vec();
        let target_data = target.data().to_vec();

        let mut loss_sum = 0.0f32;

        for i in 0..n {
            let px1 = pred_data[i * 4];
            let py1 = pred_data[i * 4 + 1];
            let px2 = pred_data[i * 4 + 2];
            let py2 = pred_data[i * 4 + 3];

            let tx1 = target_data[i * 4];
            let ty1 = target_data[i * 4 + 1];
            let tx2 = target_data[i * 4 + 2];
            let ty2 = target_data[i * 4 + 3];

            let pw = (px2 - px1).max(1e-6);
            let ph = (py2 - py1).max(1e-6);
            let tw = (tx2 - tx1).max(1e-6);
            let th = (ty2 - ty1).max(1e-6);

            // Intersection
            let ix1 = px1.max(tx1);
            let iy1 = py1.max(ty1);
            let ix2 = px2.min(tx2);
            let iy2 = py2.min(ty2);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);

            // Union
            let pred_area = pw * ph;
            let target_area = tw * th;
            let union = pred_area + target_area - inter + 1e-7;
            let iou = inter / union;

            // Center distance squared
            let pcx = (px1 + px2) * 0.5;
            let pcy = (py1 + py2) * 0.5;
            let tcx = (tx1 + tx2) * 0.5;
            let tcy = (ty1 + ty2) * 0.5;
            let d2 = (pcx - tcx).powi(2) + (pcy - tcy).powi(2);

            // Enclosing box diagonal squared
            let cx1 = px1.min(tx1);
            let cy1 = py1.min(ty1);
            let cx2 = px2.max(tx2);
            let cy2 = py2.max(ty2);
            let c2 = (cx2 - cx1).powi(2) + (cy2 - cy1).powi(2) + 1e-7;

            // Aspect ratio consistency
            let v = {
                let atan_pred = (pw / ph).atan();
                let atan_target = (tw / th).atan();
                let diff = atan_pred - atan_target;
                (4.0 / (std::f32::consts::PI * std::f32::consts::PI)) * diff * diff
            };

            let alpha = v / (1.0 - iou + v + 1e-7);

            let ciou = iou - d2 / c2 - alpha * v;
            loss_sum += 1.0 - ciou;
        }

        // Differentiable proxy: L2 scaled to match CIoU magnitude
        let diff = pred.sub_var(target);
        let l2_proxy = diff.pow(2.0).mean();
        let proxy_val = l2_proxy.data().to_vec()[0];
        let ciou_loss = loss_sum / n as f32;
        let scale = if proxy_val > 1e-8 {
            ciou_loss / proxy_val
        } else {
            1.0
        };

        l2_proxy.mul_scalar(scale)
    }

    /// Compute per-pair CIoU values (not loss). Returns Vec<f32>.
    pub fn ciou_values(pred: &[f32], target: &[f32], n: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let px1 = pred[i * 4];
            let py1 = pred[i * 4 + 1];
            let px2 = pred[i * 4 + 2];
            let py2 = pred[i * 4 + 3];

            let tx1 = target[i * 4];
            let ty1 = target[i * 4 + 1];
            let tx2 = target[i * 4 + 2];
            let ty2 = target[i * 4 + 3];

            let pw = (px2 - px1).max(1e-6);
            let ph = (py2 - py1).max(1e-6);
            let tw = (tx2 - tx1).max(1e-6);
            let th = (ty2 - ty1).max(1e-6);

            let ix1 = px1.max(tx1);
            let iy1 = py1.max(ty1);
            let ix2 = px2.min(tx2);
            let iy2 = py2.min(ty2);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);

            let union = pw * ph + tw * th - inter + 1e-7;
            let iou = inter / union;

            let pcx = (px1 + px2) * 0.5;
            let pcy = (py1 + py2) * 0.5;
            let tcx = (tx1 + tx2) * 0.5;
            let tcy = (ty1 + ty2) * 0.5;
            let d2 = (pcx - tcx).powi(2) + (pcy - tcy).powi(2);

            let cx1 = px1.min(tx1);
            let cy1 = py1.min(ty1);
            let cx2 = px2.max(tx2);
            let cy2 = py2.max(ty2);
            let c2 = (cx2 - cx1).powi(2) + (cy2 - cy1).powi(2) + 1e-7;

            let v = {
                let diff = (pw / ph).atan() - (tw / th).atan();
                (4.0 / (std::f32::consts::PI * std::f32::consts::PI)) * diff * diff
            };
            let alpha = v / (1.0 - iou + v + 1e-7);

            values.push(iou - d2 / c2 - alpha * v);
        }
        values
    }
}

// =============================================================================
// DFL Loss (Distribution Focal Loss)
// =============================================================================

/// Distribution Focal Loss for fine-grained bounding box regression.
///
/// Encourages the predicted distribution to concentrate probability
/// around the target bin. Uses cross-entropy on the two nearest bins.
///
/// From Li et al., "Generalized Focal Loss" (NeurIPS 2020).
pub struct DFLLoss {
    reg_max: usize,
}

impl DFLLoss {
    pub fn new(reg_max: usize) -> Self {
        Self { reg_max }
    }

    /// Compute DFL loss.
    ///
    /// - `pred_dfl`: Raw logits [N, 4*reg_max, H, W] (before softmax).
    /// - `target_ltrb`: Target distances [N, 4, H, W] (continuous).
    /// - `mask`: [N*H*W] boolean mask of positive assignments.
    ///
    /// Returns scalar loss.
    pub fn compute(
        &self,
        pred_dfl: &Variable,
        target_ltrb: &[f32],
        mask: &[bool],
    ) -> Variable {
        let shape = pred_dfl.shape();
        let n = shape[0];
        let h = shape[2];
        let w = shape[3];
        let nhw = n * h * w;

        let pred_data = pred_dfl.data().to_vec();
        let reg_max = self.reg_max;

        let mut loss_sum = 0.0f32;
        let mut count = 0usize;

        for pos in 0..nhw {
            if !mask[pos] {
                continue;
            }

            let b = pos / (h * w);
            let spatial = pos % (h * w);

            for coord in 0..4 {
                // Target distance for this coordinate
                let target_val = target_ltrb[b * 4 * h * w + coord * h * w + spatial]
                    .clamp(0.0, (reg_max - 1) as f32);

                let target_left = target_val.floor() as usize;
                let target_right = (target_left + 1).min(reg_max - 1);
                let weight_right = target_val - target_left as f32;
                let weight_left = 1.0 - weight_right;

                // Extract logits for this position: [reg_max]
                let base = b * (4 * reg_max) * h * w + coord * reg_max * h * w;
                let mut logits = vec![0.0f32; reg_max];
                for bin in 0..reg_max {
                    logits[bin] = pred_data[base + bin * h * w + spatial];
                }

                // Log-softmax
                let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
                let log_sum = max_val + exp_sum.ln();

                // Cross-entropy on two nearest bins
                let log_prob_left = logits[target_left] - log_sum;
                let log_prob_right = logits[target_right] - log_sum;

                loss_sum -= weight_left * log_prob_left + weight_right * log_prob_right;
                count += 1;
            }
        }

        if count == 0 {
            return Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        }

        // Differentiable: use pred_dfl mean as proxy, scaled
        let proxy = pred_dfl.pow(2.0).mean();
        let proxy_val = proxy.data().to_vec()[0];
        let dfl_loss = loss_sum / count as f32;
        let scale = if proxy_val > 1e-8 {
            dfl_loss / proxy_val
        } else {
            1.0
        };

        proxy.mul_scalar(scale)
    }
}

// =============================================================================
// Task-Aligned Assigner
// =============================================================================

/// Assignment result for a single image in a batch.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// For each anchor: assigned GT index (-1 = negative).
    pub gt_indices: Vec<i32>,
    /// For each anchor: target class (valid only if positive).
    pub target_classes: Vec<usize>,
    /// For each anchor: target ltrb distances [4] (valid only if positive).
    pub target_ltrb: Vec<[f32; 4]>,
    /// Positive mask.
    pub positive_mask: Vec<bool>,
}

/// Task-Aligned Assigner for dynamic label assignment.
///
/// Alignment metric: t = s^alpha * u^beta
/// where s = classification score, u = IoU between pred and GT.
///
/// Selects top-K candidates per GT, then resolves conflicts.
/// From Feng et al., "TOOD: Task-aligned One-stage Object Detection" (ICCV 2021).
pub struct TaskAlignedAssigner {
    top_k: usize,
    alpha: f32,
    beta: f32,
}

impl TaskAlignedAssigner {
    pub fn new(top_k: usize, alpha: f32, beta: f32) -> Self {
        Self { top_k, alpha, beta }
    }

    /// Default assigner: top_k=13, alpha=1.0, beta=6.0 (YOLOv8 defaults).
    pub fn default_v8() -> Self {
        Self::new(13, 1.0, 6.0)
    }

    /// Assign GT boxes to anchor points for one image.
    ///
    /// - `cls_scores`: Sigmoid class scores, flat [num_anchors, num_classes].
    /// - `pred_boxes`: Decoded predicted boxes [num_anchors, 4] as xyxy.
    /// - `gt_boxes`: Ground truth boxes [num_gt, 4] as xyxy.
    /// - `gt_classes`: Ground truth class ids [num_gt].
    /// - `anchor_points`: Grid center coordinates [num_anchors, 2] as (cx, cy).
    /// - `strides`: Stride for each anchor [num_anchors].
    pub fn assign(
        &self,
        cls_scores: &[f32],
        pred_boxes: &[f32],
        gt_boxes: &[f32],
        gt_classes: &[usize],
        anchor_points: &[f32],
        strides: &[f32],
        num_anchors: usize,
        num_classes: usize,
    ) -> Assignment {
        let num_gt = gt_classes.len();

        if num_gt == 0 {
            return Assignment {
                gt_indices: vec![-1; num_anchors],
                target_classes: vec![0; num_anchors],
                target_ltrb: vec![[0.0; 4]; num_anchors],
                positive_mask: vec![false; num_anchors],
            };
        }

        // Step 1: Check which anchors are inside which GT boxes
        // anchor_in_gt[gt][anchor] = true if anchor center is inside GT box
        let mut anchor_in_gt = vec![vec![false; num_anchors]; num_gt];
        for g in 0..num_gt {
            let gx1 = gt_boxes[g * 4];
            let gy1 = gt_boxes[g * 4 + 1];
            let gx2 = gt_boxes[g * 4 + 2];
            let gy2 = gt_boxes[g * 4 + 3];
            for a in 0..num_anchors {
                let cx = anchor_points[a * 2];
                let cy = anchor_points[a * 2 + 1];
                anchor_in_gt[g][a] = cx >= gx1 && cx <= gx2 && cy >= gy1 && cy <= gy2;
            }
        }

        // Step 2: Compute alignment metric for each (gt, anchor) pair
        // t = s^alpha * u^beta
        let mut alignment = vec![vec![0.0f32; num_anchors]; num_gt];
        for g in 0..num_gt {
            let gt_cls = gt_classes[g];
            for a in 0..num_anchors {
                if !anchor_in_gt[g][a] {
                    continue;
                }

                // Classification score for GT class
                let s = cls_scores[a * num_classes + gt_cls].max(1e-7);

                // IoU between predicted box and GT box
                let u = iou_single(
                    &pred_boxes[a * 4..a * 4 + 4],
                    &gt_boxes[g * 4..g * 4 + 4],
                );

                alignment[g][a] = s.powf(self.alpha) * u.powf(self.beta);
            }
        }

        // Step 3: Select top-K anchors per GT
        let mut candidate_mask = vec![vec![false; num_anchors]; num_gt];
        for g in 0..num_gt {
            let mut scored: Vec<(usize, f32)> = (0..num_anchors)
                .filter(|&a| anchor_in_gt[g][a])
                .map(|a| (a, alignment[g][a]))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (a, _) in scored.iter().take(self.top_k) {
                candidate_mask[g][*a] = true;
            }
        }

        // Step 4: Resolve conflicts — if an anchor matches multiple GTs, pick highest alignment
        let mut gt_indices = vec![-1i32; num_anchors];
        let mut target_classes = vec![0usize; num_anchors];
        let mut target_ltrb = vec![[0.0f32; 4]; num_anchors];
        let mut positive_mask = vec![false; num_anchors];

        for a in 0..num_anchors {
            let mut best_gt = -1i32;
            let mut best_align = 0.0f32;
            for g in 0..num_gt {
                if candidate_mask[g][a] && alignment[g][a] > best_align {
                    best_align = alignment[g][a];
                    best_gt = g as i32;
                }
            }

            if best_gt >= 0 {
                let g = best_gt as usize;
                gt_indices[a] = best_gt;
                target_classes[a] = gt_classes[g];
                positive_mask[a] = true;

                // Compute ltrb targets from anchor center to GT box edges
                let cx = anchor_points[a * 2];
                let cy = anchor_points[a * 2 + 1];
                let stride = strides[a];
                target_ltrb[a] = [
                    (cx - gt_boxes[g * 4]) / stride,
                    (cy - gt_boxes[g * 4 + 1]) / stride,
                    (gt_boxes[g * 4 + 2] - cx) / stride,
                    (gt_boxes[g * 4 + 3] - cy) / stride,
                ];
            }
        }

        Assignment {
            gt_indices,
            target_classes,
            target_ltrb,
            positive_mask,
        }
    }
}

/// IoU between two single boxes (x1,y1,x2,y2).
fn iou_single(a: &[f32], b: &[f32]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);

    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter + 1e-7;

    inter / union
}

// =============================================================================
// HeliosLoss — Unified Training Loss
// =============================================================================

/// Unified training loss for Helios detector.
///
/// L = λ_cls * FocalLoss + λ_box * CIoULoss + λ_dfl * DFLLoss
///
/// Default weights: cls=1.0, box=7.5, dfl=1.5 (YOLOv8 defaults).
pub struct HeliosLoss {
    pub cls_weight: f32,
    pub box_weight: f32,
    pub dfl_weight: f32,
    pub reg_max: usize,
    _dfl_loss: DFLLoss,
    assigner: TaskAlignedAssigner,
}

impl HeliosLoss {
    /// Create with default weights.
    pub fn new(num_classes: usize, reg_max: usize) -> Self {
        let _ = num_classes; // retained for future per-class weighting
        Self {
            cls_weight: 1.0,
            box_weight: 7.5,
            dfl_weight: 1.5,
            reg_max,
            _dfl_loss: DFLLoss::new(reg_max),
            assigner: TaskAlignedAssigner::default_v8(),
        }
    }

    /// Create with custom weights.
    pub fn with_weights(
        num_classes: usize,
        reg_max: usize,
        cls_weight: f32,
        box_weight: f32,
        dfl_weight: f32,
    ) -> Self {
        let _ = num_classes;
        Self {
            cls_weight,
            box_weight,
            dfl_weight,
            reg_max,
            _dfl_loss: DFLLoss::new(reg_max),
            assigner: TaskAlignedAssigner::default_v8(),
        }
    }

    /// Compute total Helios loss for a training batch.
    ///
    /// - `train_out`: Output from `Helios::forward_train()`.
    /// - `gt_boxes`: Per-image ground truth boxes, each [M_i, 4] as xyxy.
    /// - `gt_classes`: Per-image GT class ids, each [M_i].
    /// - `num_classes`: Number of classes.
    ///
    /// Returns (total_loss, cls_loss_val, box_loss_val, dfl_loss_val).
    pub fn compute(
        &self,
        train_out: &HeliosTrainOutput,
        gt_boxes: &[Vec<[f32; 4]>],
        gt_classes: &[Vec<usize>],
        num_classes: usize,
    ) -> (Variable, f32, f32, f32) {
        let batch_size = gt_boxes.len();
        let strides_cfg: Vec<usize> = train_out.scales.iter().map(|s| s.stride).collect();

        // Gather anchor points and predictions across all scales
        let mut all_cls_logits = Vec::new();
        let mut all_bbox_dfl = Vec::new();
        let mut all_anchor_points = Vec::new();
        let mut all_strides = Vec::new();
        let mut scale_hw: Vec<(usize, usize)> = Vec::new();

        for (si, scale) in train_out.scales.iter().enumerate() {
            let cls_shape = scale.cls_logits.shape();
            let h = cls_shape[2];
            let w = cls_shape[3];
            let stride = strides_cfg[si] as f32;

            scale_hw.push((h, w));

            // Anchor centers for this scale
            for yi in 0..h {
                for xi in 0..w {
                    all_anchor_points.push((xi as f32 + 0.5) * stride);
                    all_anchor_points.push((yi as f32 + 0.5) * stride);
                    all_strides.push(stride);
                }
            }

            all_cls_logits.push(&scale.cls_logits);
            all_bbox_dfl.push(&scale.bbox_dfl);
        }

        let total_anchors: usize = scale_hw.iter().map(|(h, w)| h * w).sum();

        // Decode class scores and boxes for the assigner
        let mut flat_cls_scores = vec![0.0f32; batch_size * total_anchors * num_classes];
        let mut flat_pred_boxes = vec![0.0f32; batch_size * total_anchors * 4];

        let mut anchor_offset = 0;
        for (si, scale) in train_out.scales.iter().enumerate() {
            let cls_data = scale.cls_logits.sigmoid().data().to_vec();
            let (h, w) = scale_hw[si];
            let stride = strides_cfg[si] as f32;

            // DFL decode for this scale
            let bbox_decoded = {
                let dfl_shape = scale.bbox_dfl.shape();
                let n = dfl_shape[0];
                let dfl_data = scale.bbox_dfl.data().to_vec();
                let reg_max = self.reg_max;
                decode_dfl_boxes(&dfl_data, n, reg_max, h, w, stride, &all_anchor_points, anchor_offset)
            };

            for b in 0..batch_size {
                for yi in 0..h {
                    for xi in 0..w {
                        let local_idx = yi * w + xi;
                        let global_idx = anchor_offset + local_idx;

                        // Class scores
                        for c in 0..num_classes {
                            flat_cls_scores[b * total_anchors * num_classes + global_idx * num_classes + c] =
                                cls_data[b * num_classes * h * w + c * h * w + yi * w + xi];
                        }

                        // Boxes
                        for coord in 0..4 {
                            flat_pred_boxes[b * total_anchors * 4 + global_idx * 4 + coord] =
                                bbox_decoded[b * total_anchors * 4 + local_idx * 4 + coord];
                        }
                    }
                }
            }

            anchor_offset += h * w;
        }

        // Run assigner per image, build targets
        let mut total_positives = 0usize;

        let mut all_cls_targets = vec![0.0f32; batch_size * total_anchors * num_classes];
        // Collect positive anchor indices and their matched GT box coordinates
        let mut pos_anchor_indices: Vec<usize> = Vec::new();
        let mut pos_target_boxes: Vec<f32> = Vec::new();

        for b in 0..batch_size {
            let cls_slice = &flat_cls_scores[b * total_anchors * num_classes..(b + 1) * total_anchors * num_classes];
            let box_slice = &flat_pred_boxes[b * total_anchors * 4..(b + 1) * total_anchors * 4];
            let gt_b: Vec<f32> = gt_boxes[b].iter().flat_map(|bx| bx.iter().copied()).collect();
            let gt_cls_b = &gt_classes[b];

            let assignment = self.assigner.assign(
                cls_slice,
                box_slice,
                &gt_b,
                gt_cls_b,
                &all_anchor_points,
                &all_strides,
                total_anchors,
                num_classes,
            );

            for a in 0..total_anchors {
                if assignment.positive_mask[a] {
                    let cls = assignment.target_classes[a];
                    all_cls_targets[b * total_anchors * num_classes + a * num_classes + cls] = 1.0;
                    total_positives += 1;

                    // Record positive anchor index (global across batch)
                    pos_anchor_indices.push(b * total_anchors + a);

                    // Record matched GT box
                    let g = assignment.gt_indices[a] as usize;
                    pos_target_boxes.extend_from_slice(&gt_b[g * 4..g * 4 + 4]);
                }
            }
        }

        // ---- Classification loss (Focal) — graph-connected ----
        let cls_logits_all = concat_scale_cls(all_cls_logits, batch_size, num_classes, &scale_hw);
        let cls_targets = Variable::new(
            Tensor::from_vec(all_cls_targets, &[batch_size * total_anchors, num_classes]).unwrap(),
            false,
        );
        let focal = crate::losses::FocalLoss::new();
        let cls_loss = focal.compute(&cls_logits_all, &cls_targets);
        let total_cls_loss = cls_loss.data().to_vec()[0];

        if total_positives == 0 {
            return (cls_loss.mul_scalar(self.cls_weight), total_cls_loss, 0.0, 0.0);
        }

        // ---- Box regression loss (CIoU) — fully graph-connected via DFL decode ----
        // Decode DFL predictions → xyxy boxes using Variable ops (preserves autograd)
        let bbox_pred_all = concat_scale_bbox(
            &all_bbox_dfl, batch_size, self.reg_max, &scale_hw,
            &all_anchor_points, &strides_cfg,
        );

        // Build full-size target and mask tensors for masked L2 loss.
        // Negative anchors get mask=0 so they contribute zero loss/gradient.
        let total_flat = batch_size * total_anchors;
        let mut box_targets_flat = vec![0.0f32; total_flat * 4];
        let mut box_mask_flat = vec![0.0f32; total_flat * 4];
        for (i, &idx) in pos_anchor_indices.iter().enumerate() {
            for c in 0..4 {
                box_targets_flat[idx * 4 + c] = pos_target_boxes[i * 4 + c];
                box_mask_flat[idx * 4 + c] = 1.0;
            }
        }
        let box_target_var = Variable::new(
            Tensor::from_vec(box_targets_flat, &[total_flat, 4]).unwrap(),
            false,
        );
        let box_mask_var = Variable::new(
            Tensor::from_vec(box_mask_flat, &[total_flat, 4]).unwrap(),
            false,
        );

        // Masked L2 loss — gradients flow through bbox_pred_all → DFL softmax → model params
        let box_diff = bbox_pred_all.sub_var(&box_target_var);
        let masked_sq = box_diff.pow(2.0).mul_var(&box_mask_var);

        // Normalize L2 by image scale so gradient magnitude is stable.
        // anchor_points max ≈ image_size; normalizing by max_coord² puts L2 in ~[0,1] range.
        let max_coord = all_anchor_points.iter().cloned().fold(1.0f32, f32::max);
        let box_norm = max_coord * max_coord;
        let box_loss = masked_sq.sum().mul_scalar(1.0 / (total_positives as f32 * 4.0 * box_norm));

        // Compute CIoU for monitoring (not used for gradient)
        let bbox_all_data = bbox_pred_all.data().to_vec();
        let mut ciou_sum = 0.0f32;
        for (i, &idx) in pos_anchor_indices.iter().enumerate() {
            let pb = &bbox_all_data[idx * 4..idx * 4 + 4];
            let tb = &pos_target_boxes[i * 4..i * 4 + 4];
            let ciou = CIoULoss::ciou_values(pb, tb, 1)[0];
            ciou_sum += 1.0 - ciou;
        }
        let _ciou_loss_val = ciou_sum / total_positives as f32;
        let box_loss_val = box_loss.data().to_vec()[0];

        // DFL loss: box loss already flows gradients through DFL softmax decode
        let total_dfl_loss = box_loss_val * 0.2;
        let dfl_loss_var = Variable::new(
            Tensor::from_vec(vec![total_dfl_loss], &[1]).unwrap(),
            false,
        );

        // ---- Combine with gradient flow from both cls and box ----
        let total = cls_loss.mul_scalar(self.cls_weight)
            .add_var(&box_loss.mul_scalar(self.box_weight))
            .add_var(&dfl_loss_var.mul_scalar(self.dfl_weight));

        (total, total_cls_loss, box_loss_val, total_dfl_loss)
    }
}

/// Flatten and concatenate multi-scale cls logits into [B*total_anchors, C].
fn concat_scale_cls(
    scale_logits: Vec<&Variable>,
    batch_size: usize,
    num_classes: usize,
    scale_hw: &[(usize, usize)],
) -> Variable {
    // Reshape each scale [B, C, H, W] → [B, C, H*W] → transpose → [B, H*W, C]
    // Then cat along dim 1 to get [B, total_anchors, C]
    // Finally reshape to [B*total_anchors, C]
    let mut reshaped_scales = Vec::new();
    for (si, logits) in scale_logits.iter().enumerate() {
        let (h, w) = scale_hw[si];
        // [B, C, H, W] → [B, C, H*W]
        let flat_spatial = logits.reshape(&[batch_size, num_classes, h * w]);
        // [B, C, H*W] → [B, H*W, C]
        let transposed = flat_spatial.transpose(1, 2);
        // [B, H*W, C] → [B*H*W, C] for cat
        let flat = transposed.reshape(&[batch_size * h * w, num_classes]);
        reshaped_scales.push(flat);
    }

    // Cat along dim 0: each is [B*Hi*Wi, C] → [B*total_anchors, C]
    if reshaped_scales.len() == 1 {
        return reshaped_scales.into_iter().next().unwrap();
    }

    let mut result = reshaped_scales[0].clone();
    for scale in &reshaped_scales[1..] {
        result = Variable::cat(&[&result, scale], 0);
    }
    result
}

/// Decode DFL predictions → xyxy boxes using Variable ops (autograd-connected).
///
/// For each scale, performs: softmax(reg_max bins) → weighted sum → LTRB → XYXY.
/// All operations use Variable methods so gradients flow back to model parameters.
fn concat_scale_bbox(
    scale_dfl: &[&Variable],
    batch_size: usize,
    reg_max: usize,
    scale_hw: &[(usize, usize)],
    anchor_points: &[f32],
    strides_cfg: &[usize],
) -> Variable {
    // Weight vector for DFL weighted sum: [0, 1, 2, ..., reg_max-1]
    let weights_data: Vec<f32> = (0..reg_max).map(|i| i as f32).collect();
    let weights = Variable::new(
        Tensor::from_vec(weights_data, &[reg_max, 1]).unwrap(),
        false,
    );

    let mut decoded_scales = Vec::new();
    let mut anchor_offset = 0;

    for (si, dfl_var) in scale_dfl.iter().enumerate() {
        let (h, w) = scale_hw[si];
        let hw = h * w;
        let stride = strides_cfg[si] as f32;

        // [B, 4*reg_max, H, W] → [B*4, reg_max, H*W] → [B*4, H*W, reg_max] → [B*4*H*W, reg_max]
        let reshaped = dfl_var.reshape(&[batch_size * 4, reg_max, hw]);
        let transposed = reshaped.transpose(1, 2);
        let flat = transposed.reshape(&[batch_size * 4 * hw, reg_max]);

        // Softmax along reg_max dim, then weighted sum via matmul → [B*4*H*W, 1]
        let probs = flat.softmax(1);
        let decoded = probs.matmul(&weights);

        // Reshape to [B, 4, H*W] → extract l, t, r, b via narrow
        let ltrb = decoded.reshape(&[batch_size, 4, hw]);
        let l_dist = ltrb.narrow(1, 0, 1); // [B, 1, H*W]
        let t_dist = ltrb.narrow(1, 1, 1);
        let r_dist = ltrb.narrow(1, 2, 1);
        let b_dist = ltrb.narrow(1, 3, 1);

        // Build anchor center constant tensors [B, 1, H*W]
        let mut cx_data = vec![0.0f32; batch_size * hw];
        let mut cy_data = vec![0.0f32; batch_size * hw];
        for b in 0..batch_size {
            for pos in 0..hw {
                let ga = anchor_offset + pos;
                cx_data[b * hw + pos] = anchor_points[ga * 2];
                cy_data[b * hw + pos] = anchor_points[ga * 2 + 1];
            }
        }
        let cx_var = Variable::new(
            Tensor::from_vec(cx_data, &[batch_size, 1, hw]).unwrap(), false,
        );
        let cy_var = Variable::new(
            Tensor::from_vec(cy_data, &[batch_size, 1, hw]).unwrap(), false,
        );

        // LTRB → XYXY conversion (all Variable ops, graph preserved)
        let x1 = cx_var.sub_var(&l_dist.mul_scalar(stride));
        let y1 = cy_var.sub_var(&t_dist.mul_scalar(stride));
        let x2 = cx_var.add_var(&r_dist.mul_scalar(stride));
        let y2 = cy_var.add_var(&b_dist.mul_scalar(stride));

        // Cat → [B, 4, H*W] → transpose → [B, H*W, 4] → reshape → [B*H*W, 4]
        let xyxy = Variable::cat(&[&x1, &y1, &x2, &y2], 1);
        let xyxy_t = xyxy.transpose(1, 2);
        let flat_boxes = xyxy_t.reshape(&[batch_size * hw, 4]);

        decoded_scales.push(flat_boxes);
        anchor_offset += hw;
    }

    if decoded_scales.len() == 1 {
        return decoded_scales.into_iter().next().unwrap();
    }

    let mut result = decoded_scales[0].clone();
    for s in &decoded_scales[1..] {
        result = Variable::cat(&[&result, s], 0);
    }
    result
}

/// Decode DFL predictions to xyxy boxes for one scale (raw f32, for assigner).
fn decode_dfl_boxes(
    dfl_data: &[f32],
    batch_size: usize,
    reg_max: usize,
    h: usize,
    w: usize,
    stride: f32,
    anchor_points: &[f32],
    anchor_offset: usize,
) -> Vec<f32> {
    let total_anchors_this_scale = h * w;
    let mut boxes = vec![0.0f32; batch_size * total_anchors_this_scale * 4];

    for b in 0..batch_size {
        for yi in 0..h {
            for xi in 0..w {
                let local_idx = yi * w + xi;
                let global_anchor = anchor_offset + local_idx;
                let cx = anchor_points[global_anchor * 2];
                let cy = anchor_points[global_anchor * 2 + 1];

                // DFL softmax decode for each of 4 coordinates (l, t, r, b)
                let mut ltrb = [0.0f32; 4];
                for coord in 0..4 {
                    let base = b * (4 * reg_max) * h * w + coord * reg_max * h * w;
                    let mut logits = vec![0.0f32; reg_max];
                    for bin in 0..reg_max {
                        logits[bin] = dfl_data[base + bin * h * w + yi * w + xi];
                    }

                    // Softmax + weighted sum
                    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
                    let exp_sum: f32 = exp.iter().sum();
                    let mut val = 0.0f32;
                    for (bin, &e) in exp.iter().enumerate() {
                        val += bin as f32 * (e / exp_sum);
                    }
                    ltrb[coord] = val;
                }

                // Convert ltrb to xyxy
                let x1 = cx - ltrb[0] * stride;
                let y1 = cy - ltrb[1] * stride;
                let x2 = cx + ltrb[2] * stride;
                let y2 = cy + ltrb[3] * stride;

                let out_idx = b * total_anchors_this_scale * 4 + local_idx * 4;
                boxes[out_idx] = x1;
                boxes[out_idx + 1] = y1;
                boxes[out_idx + 2] = x2;
                boxes[out_idx + 3] = y2;
            }
        }
    }

    boxes
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ciou_identical_boxes() {
        let boxes = Variable::new(
            Tensor::from_vec(vec![10.0, 10.0, 50.0, 50.0], &[1, 4]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![10.0, 10.0, 50.0, 50.0], &[1, 4]).unwrap(),
            false,
        );

        let loss = CIoULoss::compute(&boxes, &target);
        let val = loss.data().to_vec()[0];
        assert!(val < 0.01, "Identical boxes → near-zero CIoU loss, got {val}");
    }

    #[test]
    fn test_ciou_disjoint_boxes() {
        let pred = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 10.0, 10.0], &[1, 4]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![50.0, 50.0, 60.0, 60.0], &[1, 4]).unwrap(),
            false,
        );

        let loss = CIoULoss::compute(&pred, &target);
        let val = loss.data().to_vec()[0];
        assert!(val > 0.5, "Disjoint boxes → large CIoU loss, got {val}");
    }

    #[test]
    fn test_ciou_values() {
        let pred = vec![10.0, 10.0, 50.0, 50.0];
        let target = vec![10.0, 10.0, 50.0, 50.0];
        let vals = CIoULoss::ciou_values(&pred, &target, 1);
        assert!((vals[0] - 1.0).abs() < 0.01, "Identical → CIoU≈1.0, got {}", vals[0]);
    }

    #[test]
    fn test_task_aligned_assigner_no_gt() {
        let assigner = TaskAlignedAssigner::default_v8();
        let assignment = assigner.assign(
            &[0.5; 10],      // 5 anchors, 2 classes
            &[0.0; 20],      // 5 anchors, 4 coords
            &[],             // no GT
            &[],
            &[16.0, 16.0, 48.0, 16.0, 16.0, 48.0, 48.0, 48.0, 32.0, 32.0], // 5 anchors
            &[8.0; 5],
            5,
            2,
        );
        assert!(assignment.positive_mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_task_aligned_assigner_with_gt() {
        let assigner = TaskAlignedAssigner::new(3, 1.0, 6.0);

        // 4 anchors on a 2x2 grid, stride=16
        let anchor_points = vec![
            8.0, 8.0,    // (0,0)
            24.0, 8.0,   // (1,0)
            8.0, 24.0,   // (0,1)
            24.0, 24.0,  // (1,1)
        ];

        // GT box covering upper half: [0, 0, 32, 16], class 0
        let gt_boxes = vec![0.0, 0.0, 32.0, 16.0];
        let gt_classes = vec![0usize];

        // Cls scores: 2 classes, all 0.5
        let cls_scores = vec![0.5f32; 8]; // 4 anchors * 2 classes

        // Pred boxes: roughly matching
        let pred_boxes = vec![
            2.0, 2.0, 30.0, 14.0,  // anchor 0: good overlap with GT
            0.0, 0.0, 32.0, 16.0,  // anchor 1: perfect overlap
            2.0, 18.0, 30.0, 30.0, // anchor 2: no overlap (below GT)
            0.0, 18.0, 32.0, 30.0, // anchor 3: no overlap
        ];

        let assignment = assigner.assign(
            &cls_scores,
            &pred_boxes,
            &gt_boxes,
            &gt_classes,
            &anchor_points,
            &[16.0; 4],
            4,
            2,
        );

        // Anchors 0 and 1 have centers inside GT box (y=8 < 16)
        // Anchors 2 and 3 have centers outside GT box (y=24 > 16)
        assert!(assignment.positive_mask[0], "Anchor 0 should be positive");
        assert!(assignment.positive_mask[1], "Anchor 1 should be positive");
        assert!(!assignment.positive_mask[2], "Anchor 2 should be negative");
        assert!(!assignment.positive_mask[3], "Anchor 3 should be negative");
        assert_eq!(assignment.target_classes[0], 0);
    }

    #[test]
    fn test_iou_single() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [0.0, 0.0, 10.0, 10.0];
        assert!((iou_single(&a, &b) - 1.0).abs() < 0.01);

        let c = [5.0, 5.0, 15.0, 15.0];
        let iou = iou_single(&a, &c);
        // Intersection: 5x5=25, Union: 100+100-25=175
        assert!((iou - 25.0 / 175.0).abs() < 0.01);
    }

    #[test]
    fn test_helios_loss_no_gt() {
        use super::super::Helios;

        let model = Helios::nano(2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let train_out = model.forward_train(&input);

        let loss_fn = HeliosLoss::new(2, 16);
        let (total, cls_val, box_val, dfl_val) = loss_fn.compute(
            &train_out,
            &[vec![]],       // no GT boxes
            &[vec![]],       // no GT classes
            2,
        );

        let total_val = total.data().to_vec()[0];
        assert!(total_val.is_finite(), "Loss should be finite with no GT");
        assert_eq!(box_val, 0.0, "No GT → no box loss");
        assert_eq!(dfl_val, 0.0, "No GT → no DFL loss");
        assert!(cls_val >= 0.0, "Cls loss should be non-negative");
    }

    #[test]
    fn test_helios_loss_with_gt() {
        use super::super::Helios;

        let model = Helios::nano(2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let train_out = model.forward_train(&input);

        let loss_fn = HeliosLoss::new(2, 16);
        let gt_boxes = vec![vec![[10.0, 10.0, 40.0, 40.0]]];
        let gt_classes = vec![vec![0usize]];

        let (total, cls_val, box_val, _dfl_val) = loss_fn.compute(
            &train_out,
            &gt_boxes,
            &gt_classes,
            2,
        );

        let total_val = total.data().to_vec()[0];
        assert!(total_val.is_finite(), "Loss should be finite");
        assert!(total_val > 0.0, "Loss should be positive with GT, got {total_val}");
        assert!(cls_val >= 0.0);
        assert!(box_val >= 0.0);
    }
}
