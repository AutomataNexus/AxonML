//! Detection Metrics — AP and mAP Computation
//!
//! # File
//! `crates/axonml-vision/src/training/metrics.rs`
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

// =============================================================================
// Detection Result
// =============================================================================

/// A single detection result for evaluation.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Bounding box [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
    /// Predicted class ID.
    pub class_id: usize,
}

/// A single ground truth box for evaluation.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// Bounding box [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Class ID.
    pub class_id: usize,
}

// =============================================================================
// IoU Computation
// =============================================================================

/// Compute IoU between two boxes [x1, y1, x2, y2].
fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;

    if union > 0.0 { inter / union } else { 0.0 }
}

// =============================================================================
// Average Precision
// =============================================================================

/// Compute Average Precision (AP) for a single class at a given IoU threshold.
///
/// Uses the 11-point interpolation method (Pascal VOC 2007).
///
/// - `detections`: All detections for this class, sorted by confidence (descending).
/// - `ground_truths`: All ground truths for this class.
/// - `iou_threshold`: IoU threshold for considering a detection as true positive.
pub fn compute_ap(
    detections: &[DetectionResult],
    ground_truths: &[GroundTruth],
    iou_threshold: f32,
) -> f32 {
    if ground_truths.is_empty() {
        return 0.0;
    }

    let n_gt = ground_truths.len();
    let mut matched = vec![false; n_gt];

    let mut tp = Vec::with_capacity(detections.len());
    let mut fp = Vec::with_capacity(detections.len());

    // Detections should already be sorted by confidence
    for det in detections {
        let mut best_iou = 0.0f32;
        let mut best_gt = None;

        for (gi, gt) in ground_truths.iter().enumerate() {
            if matched[gi] {
                continue;
            }
            let iou = compute_iou(&det.bbox, &gt.bbox);
            if iou > best_iou {
                best_iou = iou;
                best_gt = Some(gi);
            }
        }

        if best_iou >= iou_threshold {
            if let Some(gi) = best_gt {
                matched[gi] = true;
                tp.push(1.0f32);
                fp.push(0.0);
            } else {
                tp.push(0.0);
                fp.push(1.0);
            }
        } else {
            tp.push(0.0);
            fp.push(1.0);
        }
    }

    // Cumulative sums
    let mut cum_tp = Vec::with_capacity(tp.len());
    let mut cum_fp = Vec::with_capacity(fp.len());
    let mut sum_tp = 0.0f32;
    let mut sum_fp = 0.0f32;

    for i in 0..tp.len() {
        sum_tp += tp[i];
        sum_fp += fp[i];
        cum_tp.push(sum_tp);
        cum_fp.push(sum_fp);
    }

    // Precision and recall
    let precision: Vec<f32> = cum_tp
        .iter()
        .zip(cum_fp.iter())
        .map(|(&tp, &fp)| if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 })
        .collect();

    let recall: Vec<f32> = cum_tp.iter().map(|&tp| tp / n_gt as f32).collect();

    // 11-point interpolation
    let mut ap = 0.0f32;
    for t in 0..=10 {
        let r_threshold = t as f32 / 10.0;
        let mut max_prec = 0.0f32;

        for i in 0..precision.len() {
            if recall[i] >= r_threshold && precision[i] > max_prec {
                max_prec = precision[i];
            }
        }

        ap += max_prec;
    }

    ap / 11.0
}

/// Compute mean Average Precision (mAP) across all classes.
///
/// - `all_detections`: Per-image detections.
/// - `all_ground_truths`: Per-image ground truths.
/// - `num_classes`: Total number of classes.
/// - `iou_threshold`: IoU threshold for AP computation.
pub fn compute_map(
    all_detections: &[Vec<DetectionResult>],
    all_ground_truths: &[Vec<GroundTruth>],
    num_classes: usize,
    iou_threshold: f32,
) -> f32 {
    let mut aps = Vec::new();

    for class_id in 0..num_classes {
        // Collect all detections for this class
        let mut class_dets: Vec<DetectionResult> = all_detections
            .iter()
            .flat_map(|dets| dets.iter().filter(|d| d.class_id == class_id).cloned())
            .collect();

        // Sort by confidence (descending)
        class_dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Collect all GTs for this class
        let class_gts: Vec<GroundTruth> = all_ground_truths
            .iter()
            .flat_map(|gts| gts.iter().filter(|g| g.class_id == class_id).cloned())
            .collect();

        if !class_gts.is_empty() {
            aps.push(compute_ap(&class_dets, &class_gts, iou_threshold));
        }
    }

    if aps.is_empty() {
        0.0
    } else {
        aps.iter().sum::<f32>() / aps.len() as f32
    }
}

/// Compute COCO-style mAP averaged over IoU thresholds [0.5, 0.55, ..., 0.95].
pub fn compute_coco_map(
    all_detections: &[Vec<DetectionResult>],
    all_ground_truths: &[Vec<GroundTruth>],
    num_classes: usize,
) -> f32 {
    let thresholds: Vec<f32> = (0..10).map(|i| 0.5 + i as f32 * 0.05).collect();
    let maps: Vec<f32> = thresholds
        .iter()
        .map(|&t| compute_map(all_detections, all_ground_truths, num_classes, t))
        .collect();

    maps.iter().sum::<f32>() / maps.len() as f32
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_iou_identical() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [0.0, 0.0, 10.0, 10.0];
        assert!((compute_iou(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_iou_disjoint() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [20.0, 20.0, 30.0, 30.0];
        assert!(compute_iou(&a, &b) < 1e-5);
    }

    #[test]
    fn test_compute_iou_partial() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        let expected = 25.0 / 175.0; // intersection=5*5=25, union=100+100-25=175
        assert!((compute_iou(&a, &b) - expected).abs() < 1e-4);
    }

    #[test]
    fn test_ap_perfect() {
        let dets = vec![DetectionResult {
            bbox: [0.0, 0.0, 10.0, 10.0],
            confidence: 0.9,
            class_id: 0,
        }];
        let gts = vec![GroundTruth {
            bbox: [0.0, 0.0, 10.0, 10.0],
            class_id: 0,
        }];

        let ap = compute_ap(&dets, &gts, 0.5);
        assert!(
            (ap - 1.0).abs() < 1e-5,
            "Perfect detection should have AP=1.0, got {ap}"
        );
    }

    #[test]
    fn test_ap_no_detections() {
        let dets: Vec<DetectionResult> = vec![];
        let gts = vec![GroundTruth {
            bbox: [0.0, 0.0, 10.0, 10.0],
            class_id: 0,
        }];

        let ap = compute_ap(&dets, &gts, 0.5);
        assert!((ap - 0.0).abs() < 1e-5, "No detections should have AP=0.0");
    }

    #[test]
    fn test_ap_false_positive() {
        let dets = vec![DetectionResult {
            bbox: [50.0, 50.0, 60.0, 60.0],
            confidence: 0.9,
            class_id: 0,
        }];
        let gts = vec![GroundTruth {
            bbox: [0.0, 0.0, 10.0, 10.0],
            class_id: 0,
        }];

        let ap = compute_ap(&dets, &gts, 0.5);
        assert!(ap < 0.1, "False positive should have low AP, got {ap}");
    }

    #[test]
    fn test_map_single_class() {
        let all_dets = vec![vec![DetectionResult {
            bbox: [0.0, 0.0, 10.0, 10.0],
            confidence: 0.9,
            class_id: 0,
        }]];
        let all_gts = vec![vec![GroundTruth {
            bbox: [0.0, 0.0, 10.0, 10.0],
            class_id: 0,
        }]];

        let map = compute_map(&all_dets, &all_gts, 1, 0.5);
        assert!((map - 1.0).abs() < 1e-5);
    }
}
