//! Face State Tracker — GRU-based Persistent Face Identity
//!
//! Maintains per-face hidden state across frames using GRU cells.
//! Implicit identity persistence: tracking ID emerges from temporal
//! continuity without explicit re-identification networks.
//!
//! @version 0.1.0

use axonml_autograd::Variable;
use axonml_nn::{GRUCell, Linear, Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// Face State
// =============================================================================

/// Internal state for a single tracked face.
#[derive(Clone)]
pub struct FaceState {
    /// Unique tracking ID.
    pub id: u64,
    /// Current bounding box [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Accumulated confidence (builds over time).
    pub confidence: f32,
    /// Velocity [vx, vy] in normalized coordinates per frame.
    pub velocity: [f32; 2],
    /// GRU hidden state [1, hidden_size].
    pub hidden: Variable,
    /// Number of consecutive frames tracked.
    pub frames_tracked: u32,
    /// Frames since last match (for decay).
    pub frames_missing: u32,
}

// =============================================================================
// Face State Tracker
// =============================================================================

/// GRU-based face state tracker with persistent identity.
///
/// For each tracked face:
/// 1. Merge backbone ROI features + event features → input vector
/// 2. GRU step with face's hidden state → updated state
/// 3. Decode: bbox delta, face score, velocity update
///
/// New faces are spawned when unmatched detections appear.
/// Faces decay when not matched for multiple frames.
pub struct FaceStateTracker {
    /// Merge backbone (48) + event (32) features → 64-dim input.
    merge_linear: Linear,
    /// GRU cell for temporal state update.
    gru: GRUCell,
    /// Decode hidden state → bbox delta [4].
    bbox_head: Linear,
    /// Decode hidden state → face score [1].
    score_head: Linear,
    /// Decode hidden state → velocity [2].
    velocity_head: Linear,
    /// Face classifier from event features → is_face probability.
    face_classifier: Linear,

    /// Active face slots.
    faces: Vec<FaceState>,
    /// Next tracking ID to assign.
    next_id: u64,

    /// Hidden state dimension.
    hidden_size: usize,
    /// Confidence decay per missing frame.
    pub decay_rate: f32,
    /// Maximum frames missing before removal.
    pub max_missing: u32,
    /// IoU threshold for matching detections to tracked faces.
    pub match_iou_threshold: f32,
}

impl FaceStateTracker {
    /// Create a new face state tracker.
    ///
    /// - `backbone_feat_dim`: Dimension of backbone ROI features (default 48).
    /// - `event_feat_dim`: Dimension of event features (default 32).
    /// - `hidden_size`: GRU hidden dimension (default 64).
    pub fn new(backbone_feat_dim: usize, event_feat_dim: usize, hidden_size: usize) -> Self {
        let input_dim = backbone_feat_dim + event_feat_dim;

        Self {
            merge_linear: Linear::new(input_dim, hidden_size),
            gru: GRUCell::new(hidden_size, hidden_size),
            bbox_head: Linear::new(hidden_size, 4),
            score_head: Linear::new(hidden_size, 1),
            velocity_head: Linear::new(hidden_size, 2),
            face_classifier: Linear::new(event_feat_dim, 1),
            faces: Vec::new(),
            next_id: 1,
            hidden_size,
            decay_rate: 0.95,
            max_missing: 15,
            match_iou_threshold: 0.3,
        }
    }

    /// Default tracker with standard dimensions.
    pub fn default_config() -> Self {
        Self::new(48, 32, 64)
    }

    /// Update tracked faces with new features.
    ///
    /// # Arguments
    /// - `backbone_feats`: ROI features from backbone for detected regions [N, 48]
    /// - `event_feats`: Event features for detected regions [N, 32]
    /// - `detected_bboxes`: Detected bounding boxes [N, 4]
    ///
    /// # Returns
    /// Updated face states as output detections.
    pub fn update(
        &mut self,
        backbone_feats: &Variable,
        event_feats: &Variable,
        detected_bboxes: &[[f32; 4]],
    ) -> Vec<crate::ops::PhantomFaceDetection> {
        let n = detected_bboxes.len();

        // Match detections to existing faces via IoU
        let mut matched_det = vec![false; n];
        let mut matched_face = vec![false; self.faces.len()];

        // Collect matches first (avoid borrow conflict with update_face)
        let mut matches: Vec<(usize, usize)> = Vec::new();

        if !self.faces.is_empty() && n > 0 {
            for fi in 0..self.faces.len() {
                let mut best_iou = 0.0f32;
                let mut best_det = None;

                for (di, det_bbox) in detected_bboxes.iter().enumerate() {
                    if matched_det[di] {
                        continue;
                    }
                    let iou = compute_iou(&self.faces[fi].bbox, det_bbox);
                    if iou > best_iou && iou > self.match_iou_threshold {
                        best_iou = iou;
                        best_det = Some(di);
                    }
                }

                if let Some(di) = best_det {
                    matched_det[di] = true;
                    matched_face[fi] = true;
                    matches.push((fi, di));
                }
            }
        }

        // Apply updates after matching
        for (fi, di) in matches {
            self.update_face(fi, backbone_feats, event_feats, di, &detected_bboxes[di]);
        }

        // Decay unmatched faces
        let mut to_remove = Vec::new();
        for (fi, face) in self.faces.iter_mut().enumerate() {
            if !matched_face.get(fi).copied().unwrap_or(false) {
                face.frames_missing += 1;
                face.confidence *= self.decay_rate;
                if face.frames_missing > self.max_missing {
                    to_remove.push(fi);
                }
            }
        }

        // Remove dead faces (reverse order to preserve indices)
        for &fi in to_remove.iter().rev() {
            self.faces.remove(fi);
        }

        // Spawn new faces for unmatched detections
        for (di, _bbox) in detected_bboxes.iter().enumerate() {
            if !matched_det[di] {
                // Check if event features indicate a face
                let event_feat = extract_row(event_feats, di);
                let face_score = self.classify_face(&event_feat);

                if face_score > 0.3 {
                    self.spawn_face(&detected_bboxes[di]);
                }
            }
        }

        // Collect output
        self.faces
            .iter()
            .map(|f| crate::ops::PhantomFaceDetection {
                bbox: f.bbox,
                confidence: f.confidence,
                tracking_id: f.id,
                velocity: f.velocity,
                frames_tracked: f.frames_tracked,
            })
            .collect()
    }

    /// Update a matched face with new features.
    fn update_face(
        &mut self,
        face_idx: usize,
        backbone_feats: &Variable,
        event_feats: &Variable,
        det_idx: usize,
        new_bbox: &[f32; 4],
    ) {
        let bb_feat = extract_row(backbone_feats, det_idx);
        let ev_feat = extract_row(event_feats, det_idx);

        // Concatenate features along dim 1, preserving gradient flow
        let merged = Variable::cat(&[&bb_feat, &ev_feat], 1);

        // Project to hidden dim
        let projected = self.merge_linear.forward(&merged).relu();

        // GRU step
        let hidden = &self.faces[face_idx].hidden;
        let new_hidden = self.gru.forward_step(&projected, hidden);

        // Decode
        let bbox_delta = self.bbox_head.forward(&new_hidden);
        let score = self.score_head.forward(&new_hidden);
        let velocity = self.velocity_head.forward(&new_hidden);

        let delta = bbox_delta.data().to_vec();
        let score_val = score.data().to_vec()[0].tanh() * 0.5 + 0.5; // Map to [0, 1]
        let vel = velocity.data().to_vec();

        let face = &mut self.faces[face_idx];
        face.bbox = [
            new_bbox[0] + delta[0] * 0.1,
            new_bbox[1] + delta[1] * 0.1,
            new_bbox[2] + delta[2] * 0.1,
            new_bbox[3] + delta[3] * 0.1,
        ];
        face.confidence = (face.confidence * 0.8 + score_val * 0.2).min(1.0);
        face.velocity = [vel[0], vel[1]];
        face.hidden = new_hidden;
        face.frames_tracked += 1;
        face.frames_missing = 0;
    }

    /// Classify whether event features belong to a face.
    fn classify_face(&self, event_feat: &Variable) -> f32 {
        let out = self.face_classifier.forward(event_feat);
        let val = out.data().to_vec()[0];
        1.0 / (1.0 + (-val).exp()) // sigmoid
    }

    /// Spawn a new face tracking slot.
    fn spawn_face(&mut self, bbox: &[f32; 4]) {
        let id = self.next_id;
        self.next_id += 1;

        let hidden = Variable::new(
            Tensor::from_vec(vec![0.0; self.hidden_size], &[1, self.hidden_size]).unwrap(),
            false,
        );

        self.faces.push(FaceState {
            id,
            bbox: *bbox,
            confidence: 0.5,
            velocity: [0.0, 0.0],
            hidden,
            frames_tracked: 1,
            frames_missing: 0,
        });
    }

    /// Get currently tracked faces.
    pub fn tracked_faces(&self) -> &[FaceState] {
        &self.faces
    }

    /// Number of currently tracked faces.
    pub fn num_tracked(&self) -> usize {
        self.faces.len()
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.faces.clear();
        self.next_id = 1;
    }

    /// Get parameters for optimization.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.merge_linear.parameters());
        p.extend(self.gru.parameters());
        p.extend(self.bbox_head.parameters());
        p.extend(self.score_head.parameters());
        p.extend(self.velocity_head.parameters());
        p.extend(self.face_classifier.parameters());
        p
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Compute IoU between two bounding boxes [x1, y1, x2, y2].
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

/// Extract a single row from a batched Variable, preserving the computational graph.
fn extract_row(var: &Variable, idx: usize) -> Variable {
    // Use narrow to slice a single row along dim 0, keeping grad flow intact
    var.narrow(0, idx, 1)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_state_tracker_creation() {
        let tracker = FaceStateTracker::default_config();
        assert_eq!(tracker.num_tracked(), 0);
    }

    #[test]
    fn test_spawn_and_track() {
        let mut tracker = FaceStateTracker::default_config();

        let bb_feats = Variable::new(
            Tensor::from_vec(vec![0.1; 48], &[1, 48]).unwrap(),
            false,
        );
        let ev_feats = Variable::new(
            Tensor::from_vec(vec![0.5; 32], &[1, 32]).unwrap(),
            false,
        );
        let bboxes = vec![[10.0, 10.0, 50.0, 50.0]];

        let detections = tracker.update(&bb_feats, &ev_feats, &bboxes);

        // Should have spawned or rejected based on face classifier
        // With random weights, result varies — just check it doesn't crash
        assert!(detections.len() <= 1);
    }

    #[test]
    fn test_face_decay() {
        let mut tracker = FaceStateTracker::default_config();
        tracker.max_missing = 3;

        // Manually spawn a face
        tracker.spawn_face(&[10.0, 10.0, 50.0, 50.0]);
        assert_eq!(tracker.num_tracked(), 1);

        // Update with no detections → face should decay
        let empty_bb = Variable::new(Tensor::from_vec(vec![], &[0, 48]).unwrap(), false);
        let empty_ev = Variable::new(Tensor::from_vec(vec![], &[0, 32]).unwrap(), false);

        for _ in 0..4 {
            tracker.update(&empty_bb, &empty_ev, &[]);
        }

        // After 4 frames missing (max_missing=3), face should be removed
        assert_eq!(tracker.num_tracked(), 0);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = FaceStateTracker::default_config();
        tracker.spawn_face(&[0.0, 0.0, 10.0, 10.0]);
        tracker.spawn_face(&[20.0, 20.0, 30.0, 30.0]);
        assert_eq!(tracker.num_tracked(), 2);

        tracker.reset();
        assert_eq!(tracker.num_tracked(), 0);
    }

    #[test]
    fn test_compute_iou() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [0.0, 0.0, 10.0, 10.0];
        assert!((compute_iou(&a, &b) - 1.0).abs() < 1e-5);

        let c = [20.0, 20.0, 30.0, 30.0];
        assert!(compute_iou(&a, &c) < 1e-5);

        let d = [5.0, 5.0, 15.0, 15.0];
        let expected = 25.0 / 175.0;
        assert!((compute_iou(&a, &d) - expected).abs() < 1e-4);
    }

    #[test]
    fn test_tracker_param_count() {
        let tracker = FaceStateTracker::default_config();
        let total: usize = tracker.parameters().iter().map(|p| p.numel()).sum();
        // Should be compact: merge(80*64) + GRU(3*64*(64+64)) + heads
        assert!(total < 50_000, "Tracker too large: {total} params");
        assert!(total > 5_000, "Tracker too small: {total} params");
    }
}
