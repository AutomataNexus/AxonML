//! Target Assignment — FCOS-style Anchor-Free Target Generation
//!
//! # File
//! `crates/axonml-vision/src/training/assign.rs`
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

use axonml_tensor::Tensor;

// =============================================================================
// FCOS Target Assignment (for Nexus)
// =============================================================================

/// Per-location target for FCOS-style detection.
#[derive(Debug, Clone)]
pub struct FcosTarget {
    /// Classification target (class_id, or -1 for background).
    pub class_id: i32,
    /// Regression target [l, t, r, b] — distances from location to box edges.
    pub ltrb: [f32; 4],
    /// Centerness target in [0, 1].
    pub centerness: f32,
}

/// FCOS target assignment for multi-scale feature maps.
///
/// For each spatial location on each scale:
/// 1. Check if location falls inside any GT box
/// 2. Assign to the GT box with smallest area (if multiple match)
/// 3. Compute regression targets (l, t, r, b)
/// 4. Compute centerness
///
/// Scale assignment by size range prevents ambiguity:
/// - Scale 0 (stride 8): objects with max(l,t,r,b) in [0, 64]
/// - Scale 1 (stride 16): objects with max(l,t,r,b) in [64, 128]
/// - Scale 2 (stride 32): objects with max(l,t,r,b) in [128, ∞]
pub fn assign_fcos_targets(
    gt_boxes: &[[f32; 4]],         // [N, 4] as (x1, y1, x2, y2) in pixel coords
    gt_classes: &[usize],          // [N] class IDs
    feat_sizes: &[(usize, usize)], // [(H, W)] per scale
    strides: &[f32],               // stride per scale
    size_ranges: &[(f32, f32)],    // (min_size, max_size) per scale
) -> Vec<Vec<FcosTarget>> {
    let num_scales = feat_sizes.len();
    let mut all_targets = Vec::with_capacity(num_scales);

    for scale_idx in 0..num_scales {
        let (fh, fw) = feat_sizes[scale_idx];
        let stride = strides[scale_idx];
        let (min_size, max_size) = size_ranges[scale_idx];

        let mut targets = Vec::with_capacity(fh * fw);

        for fy in 0..fh {
            for fx in 0..fw {
                // Location center in input image coordinates
                let cx = (fx as f32 + 0.5) * stride;
                let cy = (fy as f32 + 0.5) * stride;

                let mut best_target: Option<FcosTarget> = None;
                let mut best_area = f32::MAX;

                for (gi, gt_box) in gt_boxes.iter().enumerate() {
                    let (x1, y1, x2, y2) = (gt_box[0], gt_box[1], gt_box[2], gt_box[3]);

                    // Check if center falls inside GT box
                    if cx < x1 || cx > x2 || cy < y1 || cy > y2 {
                        continue;
                    }

                    // Compute LTRB distances
                    let l = cx - x1;
                    let t = cy - y1;
                    let r = x2 - cx;
                    let b = y2 - cy;

                    // Check size range for this scale
                    let max_dist = l.max(t).max(r).max(b);
                    if max_dist < min_size || max_dist > max_size {
                        continue;
                    }

                    // Prefer smallest GT box (by area)
                    let area = (x2 - x1) * (y2 - y1);
                    if area < best_area {
                        best_area = area;
                        let centerness = crate::losses::compute_centerness(l, t, r, b);
                        best_target = Some(FcosTarget {
                            class_id: gt_classes[gi] as i32,
                            ltrb: [l, t, r, b],
                            centerness,
                        });
                    }
                }

                targets.push(best_target.unwrap_or(FcosTarget {
                    class_id: -1, // background
                    ltrb: [0.0; 4],
                    centerness: 0.0,
                }));
            }
        }

        all_targets.push(targets);
    }

    all_targets
}

/// Convert FCOS targets to tensors for loss computation.
///
/// Returns per-scale: (cls_targets [H*W], bbox_targets [H*W, 4], centerness_targets [H*W]).
pub fn fcos_targets_to_tensors(
    targets: &[Vec<FcosTarget>],
) -> Vec<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    targets
        .iter()
        .map(|scale_targets| {
            let n = scale_targets.len();

            let cls: Vec<f32> = scale_targets.iter().map(|t| t.class_id as f32).collect();
            let bbox: Vec<f32> = scale_targets
                .iter()
                .flat_map(|t| t.ltrb.iter().copied())
                .collect();
            let centerness: Vec<f32> = scale_targets.iter().map(|t| t.centerness).collect();

            (
                Tensor::from_vec(cls, &[n]).unwrap(),
                Tensor::from_vec(bbox, &[n, 4]).unwrap(),
                Tensor::from_vec(centerness, &[n]).unwrap(),
            )
        })
        .collect()
}

// =============================================================================
// Phantom Face Target Assignment
// =============================================================================

/// Single-scale target assignment for Phantom face detection.
///
/// - `gt_faces`: Face bounding boxes [x1, y1, x2, y2] in pixel coords.
/// - `feat_h`, `feat_w`: Feature map spatial dimensions.
/// - `stride`: Spatial stride (typically 4 for P2 features).
///
/// Returns (cls_target [H, W], bbox_target [H, W, 4]):
/// - cls_target: 1.0 if GT face center falls in cell, else 0.0
/// - bbox_target: (dx, dy, dw, dh) relative to cell center, log-space for w/h
pub fn assign_phantom_targets(
    gt_faces: &[[f32; 4]],
    feat_h: usize,
    feat_w: usize,
    stride: f32,
) -> (Tensor<f32>, Tensor<f32>) {
    let mut cls = vec![0.0f32; feat_h * feat_w];
    let mut bbox = vec![0.0f32; feat_h * feat_w * 4];

    for face in gt_faces {
        let (x1, y1, x2, y2) = (face[0], face[1], face[2], face[3]);
        let face_cx = (x1 + x2) / 2.0;
        let face_cy = (y1 + y2) / 2.0;
        let face_w = (x2 - x1).max(1.0);
        let face_h = (y2 - y1).max(1.0);

        // Find which cell the center falls in
        let fx = ((face_cx / stride) as usize).min(feat_w.saturating_sub(1));
        let fy = ((face_cy / stride) as usize).min(feat_h.saturating_sub(1));

        let cell_cx = (fx as f32 + 0.5) * stride;
        let cell_cy = (fy as f32 + 0.5) * stride;

        let idx = fy * feat_w + fx;
        cls[idx] = 1.0;

        // Regression targets: offset + log-space size
        bbox[idx * 4] = (face_cx - cell_cx) / stride;
        bbox[idx * 4 + 1] = (face_cy - cell_cy) / stride;
        bbox[idx * 4 + 2] = (face_w / stride).ln();
        bbox[idx * 4 + 3] = (face_h / stride).ln();
    }

    (
        Tensor::from_vec(cls, &[feat_h, feat_w]).unwrap(),
        Tensor::from_vec(bbox, &[feat_h, feat_w, 4]).unwrap(),
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fcos_assignment_basic() {
        // One GT box covering most of a small image
        let gt_boxes = vec![[10.0, 10.0, 50.0, 50.0]];
        let gt_classes = vec![0usize];
        let feat_sizes = vec![(8, 8)];
        let strides = vec![8.0];
        let size_ranges = vec![(0.0, f32::MAX)];

        let targets =
            assign_fcos_targets(&gt_boxes, &gt_classes, &feat_sizes, &strides, &size_ranges);

        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].len(), 64); // 8*8

        // Count positive locations
        let positives = targets[0].iter().filter(|t| t.class_id >= 0).count();
        assert!(positives > 0, "Should have at least one positive location");
        assert!(positives < 64, "Not all locations should be positive");
    }

    #[test]
    fn test_fcos_centerness() {
        // Box exactly centered on a cell
        let gt_boxes = vec![[0.0, 0.0, 16.0, 16.0]];
        let gt_classes = vec![0usize];
        let feat_sizes = vec![(2, 2)];
        let strides = vec![8.0];
        let size_ranges = vec![(0.0, f32::MAX)];

        let targets =
            assign_fcos_targets(&gt_boxes, &gt_classes, &feat_sizes, &strides, &size_ranges);

        // Cell at (0,0) has center (4,4) — inside [0,0,16,16]
        // l=4, t=4, r=12, b=12 → centerness = sqrt((4/12)*(4/12)) = 4/12 = 1/3
        let t = &targets[0][0];
        assert_eq!(t.class_id, 0);
        assert!(t.centerness > 0.0 && t.centerness <= 1.0);
    }

    #[test]
    fn test_fcos_background() {
        // Box far from any feature map location
        let gt_boxes = vec![[200.0, 200.0, 210.0, 210.0]];
        let gt_classes = vec![0usize];
        let feat_sizes = vec![(2, 2)];
        let strides = vec![8.0];
        let size_ranges = vec![(0.0, f32::MAX)];

        let targets =
            assign_fcos_targets(&gt_boxes, &gt_classes, &feat_sizes, &strides, &size_ranges);

        // All cells should be background (GT box center is outside feat map)
        let positives = targets[0].iter().filter(|t| t.class_id >= 0).count();
        assert_eq!(positives, 0);
    }

    #[test]
    fn test_phantom_target_assignment() {
        let gt_faces = vec![[20.0, 20.0, 40.0, 40.0]];
        let (cls, bbox) = assign_phantom_targets(&gt_faces, 16, 16, 4.0);

        assert_eq!(cls.shape(), &[16, 16]);
        assert_eq!(bbox.shape(), &[16, 16, 4]);

        let cls_data = cls.to_vec();
        let positive_count = cls_data.iter().filter(|&&v| v > 0.5).count();
        assert_eq!(positive_count, 1, "Exactly one cell should be positive");
    }

    #[test]
    fn test_fcos_targets_to_tensors() {
        let targets = vec![vec![
            FcosTarget {
                class_id: 0,
                ltrb: [1.0, 2.0, 3.0, 4.0],
                centerness: 0.5,
            },
            FcosTarget {
                class_id: -1,
                ltrb: [0.0; 4],
                centerness: 0.0,
            },
        ]];

        let tensors = fcos_targets_to_tensors(&targets);
        assert_eq!(tensors.len(), 1);
        let (cls, bbox, center) = &tensors[0];
        assert_eq!(cls.shape(), &[2]);
        assert_eq!(bbox.shape(), &[2, 4]);
        assert_eq!(center.shape(), &[2]);
    }
}
