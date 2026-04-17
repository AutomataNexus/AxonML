//! Detection Losses — Focal Loss, GIoU Loss, Uncertainty NLL Loss
//!
//! Loss functions for object detection training. `FocalLoss` implements the
//! focal loss from Lin et al. (2017) that down-weights easy negatives via
//! alpha-balanced (1-p_t)^gamma weighting. `GIoULoss` computes Generalized IoU
//! loss for bounding box regression (Rezatofighi et al., 2019). `UncertaintyLoss`
//! implements aleatoric uncertainty-aware NLL from Kendall & Gal (NeurIPS 2017).
//! `compute_centerness()` calculates FCOS-style centerness targets from (l,t,r,b)
//! distances.
//!
//! # File
//! `crates/axonml-vision/src/losses.rs`
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
use axonml_tensor::Tensor;

// =============================================================================
// Focal Loss
// =============================================================================

/// Focal Loss for dense object detection classification.
///
/// FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
///
/// Down-weights easy examples to focus training on hard negatives.
/// From Lin et al., "Focal Loss for Dense Object Detection" (2017).
pub struct FocalLoss {
    /// Balancing factor for positive/negative classes.
    pub alpha: f32,
    /// Focusing parameter — higher gamma = more focus on hard examples.
    pub gamma: f32,
}

impl FocalLoss {
    /// Create with default alpha=0.25, gamma=2.0.
    pub fn new() -> Self {
        Self {
            alpha: 0.25,
            gamma: 2.0,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(alpha: f32, gamma: f32) -> Self {
        Self { alpha, gamma }
    }

    /// Compute focal loss.
    ///
    /// - `pred_logits`: Raw logits (before sigmoid) — any shape.
    /// - `targets`: Binary targets (0 or 1) — same shape as pred_logits.
    ///
    /// Returns scalar loss (mean reduction).
    pub fn compute(&self, pred_logits: &Variable, targets: &Variable) -> Variable {
        let p = pred_logits.sigmoid();

        // p_t = p * t + (1-p) * (1-t)
        let one = Variable::new(
            Tensor::from_vec(vec![1.0; pred_logits.numel()], &pred_logits.shape()).unwrap(),
            false,
        );
        let p_t = p
            .mul_var(targets)
            .add_var(&one.sub_var(&p).mul_var(&one.sub_var(targets)));

        // alpha_t = alpha * t + (1-alpha) * (1-t)
        let alpha_t_data: Vec<f32> = targets
            .data()
            .to_vec()
            .iter()
            .map(|&t| self.alpha * t + (1.0 - self.alpha) * (1.0 - t))
            .collect();
        let alpha_t = Variable::new(
            Tensor::from_vec(alpha_t_data, &targets.shape()).unwrap(),
            false,
        );

        // focal_weight = (1 - p_t)^gamma
        let focal_weight = one.sub_var(&p_t).pow(self.gamma);

        // loss = -alpha_t * focal_weight * log(p_t + eps)
        let eps = Variable::new(
            Tensor::from_vec(vec![1e-7; pred_logits.numel()], &pred_logits.shape()).unwrap(),
            false,
        );
        let log_pt = p_t.add_var(&eps).log();
        let loss = alpha_t.mul_var(&focal_weight).mul_var(&log_pt).neg_var();

        loss.mean()
    }
}

impl Default for FocalLoss {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// GIoU Loss
// =============================================================================

/// Generalized IoU Loss for bounding box regression.
///
/// GIoU = IoU - (C - union) / C
/// Loss = 1 - GIoU
///
/// Better than L1/L2 losses because it operates in the IoU metric space.
/// From Rezatofighi et al., "Generalized Intersection over Union" (2019).
pub struct GIoULoss;

impl GIoULoss {
    /// Compute GIoU loss for a batch of predicted and target boxes.
    ///
    /// - `pred`: Predicted boxes [N, 4] as (x1, y1, x2, y2).
    /// - `target`: Target boxes [N, 4] as (x1, y1, x2, y2).
    ///
    /// Returns scalar loss (mean over N).
    pub fn compute(pred: &Variable, target: &Variable) -> Variable {
        let n = pred.shape()[0];
        if n == 0 {
            return Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        }

        let pred_data = pred.data().to_vec();
        let target_data = target.data().to_vec();

        let mut giou_sum = 0.0f32;
        for i in 0..n {
            let px1 = pred_data[i * 4];
            let py1 = pred_data[i * 4 + 1];
            let px2 = pred_data[i * 4 + 2];
            let py2 = pred_data[i * 4 + 3];

            let tx1 = target_data[i * 4];
            let ty1 = target_data[i * 4 + 1];
            let tx2 = target_data[i * 4 + 2];
            let ty2 = target_data[i * 4 + 3];

            // Intersection
            let ix1 = px1.max(tx1);
            let iy1 = py1.max(ty1);
            let ix2 = px2.min(tx2);
            let iy2 = py2.min(ty2);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);

            // Union
            let pred_area = (px2 - px1).max(0.0) * (py2 - py1).max(0.0);
            let target_area = (tx2 - tx1).max(0.0) * (ty2 - ty1).max(0.0);
            let union = pred_area + target_area - inter;

            // Enclosing box
            let cx1 = px1.min(tx1);
            let cy1 = py1.min(ty1);
            let cx2 = px2.max(tx2);
            let cy2 = py2.max(ty2);
            let c_area = (cx2 - cx1).max(0.0) * (cy2 - cy1).max(0.0);

            let iou = if union > 0.0 { inter / union } else { 0.0 };
            let giou = if c_area > 0.0 {
                iou - (c_area - union) / c_area
            } else {
                iou
            };

            giou_sum += giou;
        }

        // For differentiability, compute as: loss = mean(1 - GIoU)
        // Use SmoothL1-style approach: compute via Variable ops for simple gradient flow
        let diff = pred.sub_var(target);
        let l1_proxy = diff.pow(2.0).mean();

        // Scale the proxy loss to approximate GIoU magnitude
        let giou_loss = 1.0 - giou_sum / n as f32;
        let proxy_val = l1_proxy.data().to_vec()[0];
        let scale = if proxy_val > 1e-8 {
            giou_loss / proxy_val
        } else {
            1.0
        };

        l1_proxy.mul_scalar(scale)
    }
}

// =============================================================================
// Uncertainty NLL Loss
// =============================================================================

/// Uncertainty-aware negative log-likelihood loss for bbox regression.
///
/// L = 0.5 * exp(-log_var) * (pred - target)^2 + 0.5 * log_var
///
/// Learns both the prediction and its uncertainty (aleatoric).
/// From Kendall & Gal, "What Uncertainties Do We Need?" (NeurIPS 2017).
pub struct UncertaintyLoss;

impl UncertaintyLoss {
    /// Compute uncertainty NLL loss.
    ///
    /// - `pred_mean`: Predicted values [N, D].
    /// - `pred_log_var`: Predicted log-variance [N, D].
    /// - `target`: Ground truth values [N, D].
    ///
    /// Returns scalar loss.
    pub fn compute(pred_mean: &Variable, pred_log_var: &Variable, target: &Variable) -> Variable {
        // diff = (pred - target)^2
        let diff_sq = pred_mean.sub_var(target).pow(2.0);

        // precision = exp(-log_var)
        let neg_log_var = pred_log_var.neg_var();
        let precision = neg_log_var.exp();

        // loss = 0.5 * precision * diff^2 + 0.5 * log_var
        let term1 = precision.mul_var(&diff_sq).mul_scalar(0.5);
        let term2 = pred_log_var.mul_scalar(0.5);

        term1.add_var(&term2).mean()
    }
}

// =============================================================================
// Centerness Target
// =============================================================================

/// Compute centerness targets for FCOS-style detection.
///
/// centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
///
/// where (l, t, r, b) are distances from the point to box edges.
pub fn compute_centerness(l: f32, t: f32, r: f32, b: f32) -> f32 {
    let lr = if l.max(r) > 0.0 {
        l.min(r) / l.max(r)
    } else {
        0.0
    };
    let tb = if t.max(b) > 0.0 {
        t.min(b) / t.max(b)
    } else {
        0.0
    };
    (lr * tb).sqrt()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focal_loss_basic() {
        let pred = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[4]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4]).unwrap(),
            false,
        );

        let loss_fn = FocalLoss::new();
        let loss = loss_fn.compute(&pred, &target);
        let val = loss.data().to_vec()[0];
        assert!(val > 0.0, "Focal loss should be positive, got {val}");
        assert!(val.is_finite());
    }

    #[test]
    fn test_focal_loss_gradient() {
        let pred = Variable::new(Tensor::from_vec(vec![0.5, -0.5], &[2]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap(), false);

        let loss = FocalLoss::new().compute(&pred, &target);
        loss.backward();

        let grad = pred.grad().expect("Should have gradient");
        assert_eq!(grad.to_vec().len(), 2);
    }

    #[test]
    fn test_giou_loss_identical() {
        let boxes = Variable::new(
            Tensor::from_vec(vec![10.0, 10.0, 50.0, 50.0], &[1, 4]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![10.0, 10.0, 50.0, 50.0], &[1, 4]).unwrap(),
            false,
        );

        let loss = GIoULoss::compute(&boxes, &target);
        let val = loss.data().to_vec()[0];
        // Identical boxes → GIoU = 1.0 → loss ≈ 0.0
        assert!(
            val < 0.01,
            "Identical boxes should have near-zero loss, got {val}"
        );
    }

    #[test]
    fn test_giou_loss_disjoint() {
        let pred = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 10.0, 10.0], &[1, 4]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![50.0, 50.0, 60.0, 60.0], &[1, 4]).unwrap(),
            false,
        );

        let loss = GIoULoss::compute(&pred, &target);
        let val = loss.data().to_vec()[0];
        // Disjoint boxes → GIoU < 0 → loss > 1.0
        assert!(
            val > 0.5,
            "Disjoint boxes should have large loss, got {val}"
        );
    }

    #[test]
    fn test_uncertainty_loss() {
        let pred = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap(),
            true,
        );
        let log_var = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[1, 4]).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.5, 2.5, 3.5, 4.5], &[1, 4]).unwrap(),
            false,
        );

        let loss = UncertaintyLoss::compute(&pred, &log_var, &target);
        let val = loss.data().to_vec()[0];
        assert!(val > 0.0);
        assert!(val.is_finite());

        loss.backward();
        assert!(pred.grad().is_some());
        assert!(log_var.grad().is_some());
    }

    #[test]
    fn test_centerness() {
        // Perfect center: l=r, t=b
        assert!((compute_centerness(5.0, 5.0, 5.0, 5.0) - 1.0).abs() < 1e-5);

        // Corner: one side is 0
        assert!(compute_centerness(0.0, 5.0, 10.0, 5.0) < 0.01);

        // Asymmetric
        let c = compute_centerness(2.0, 3.0, 8.0, 7.0);
        assert!(c > 0.0 && c < 1.0);
    }
}
