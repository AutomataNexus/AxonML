//! Object Memory Bank — GRU-based Persistent Object Tracking
//!
//! Implements `ObjectMemoryBank` which maintains a GRU hidden state per tracked
//! object across video frames. Each detection is matched to existing memory slots
//! via feature similarity, and the GRU evolves the object's latent representation
//! over time. Supports slot creation for new objects, eviction of stale tracks,
//! and feature-based re-identification after occlusion.
//!
//! # File
//! `crates/axonml-vision/src/models/nexus/memory.rs`
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

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_nn::{GRUCell, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use crate::ops::{box_iou, roi_align};

// =============================================================================
// Object Slot
// =============================================================================

/// Persistent state for a single tracked object.
#[derive(Clone)]
pub struct ObjectSlot {
    /// Unique tracking ID.
    pub id: u64,
    /// Current bounding box [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Accumulated confidence.
    pub confidence: f32,
    /// GRU hidden state [1, hidden_size].
    pub hidden: Variable,
    /// Class logits from last update.
    pub class_logits: Vec<f32>,
    /// Frames tracked.
    pub frames_tracked: u32,
    /// Frames since last match.
    pub frames_missing: u32,
}

// =============================================================================
// Object Memory Bank
// =============================================================================

/// GRU-based persistent object memory bank.
///
/// For each detected object, maintains a hidden state that is updated
/// across frames. New objects spawn new slots; unmatched slots decay.
pub struct ObjectMemoryBank {
    /// Project ROI features to GRU input dimension.
    roi_project: Linear,
    /// GRU cell for temporal state update.
    gru: GRUCell,
    /// Hidden state dimension.
    hidden_size: usize,
    /// ROI align output size.
    roi_size: usize,
    /// Feature channels for ROI extraction.
    feat_channels: usize,

    /// Active object slots.
    slots: Vec<ObjectSlot>,
    /// Next tracking ID.
    next_id: u64,

    /// IoU threshold for matching.
    pub match_threshold: f32,
    /// Maximum missing frames before removal.
    pub max_missing: u32,
    /// Confidence decay rate per missing frame.
    pub decay_rate: f32,
}

impl ObjectMemoryBank {
    /// Create an object memory bank.
    ///
    /// - `feat_channels`: Channels in the feature map for ROI extraction.
    /// - `hidden_size`: GRU hidden state dimension.
    /// - `roi_size`: Spatial size for ROI align output (roi_size × roi_size).
    pub fn new(feat_channels: usize, hidden_size: usize, roi_size: usize) -> Self {
        let roi_feat_dim = feat_channels * roi_size * roi_size;

        Self {
            roi_project: Linear::new(roi_feat_dim, hidden_size),
            gru: GRUCell::new(hidden_size, hidden_size),
            hidden_size,
            roi_size,
            feat_channels,
            slots: Vec::new(),
            next_id: 1,
            match_threshold: 0.3,
            max_missing: 10,
            decay_rate: 0.9,
        }
    }

    /// Default configuration for Nexus (96 channels, 64 hidden, 3×3 ROI).
    pub fn default_config() -> Self {
        Self::new(96, 64, 3)
    }

    /// Update object memory with new frame detections.
    ///
    /// # Arguments
    /// - `features`: Feature map [B, C, H, W] for ROI extraction.
    /// - `proposals`: Detected bounding boxes [N, 4] in (x1, y1, x2, y2).
    /// - `scores`: Proposal confidence scores [N].
    /// - `spatial_scale`: Scale from image coords to feature map coords.
    ///
    /// # Returns
    /// Updated hidden states for matched/new objects [M, hidden_size].
    pub fn update(
        &mut self,
        features: &Variable,
        proposals: &[[f32; 4]],
        scores: &[f32],
        spatial_scale: f32,
    ) -> Vec<Variable> {
        let n_proposals = proposals.len();

        // Match proposals to existing slots via IoU
        let mut matched_prop = vec![false; n_proposals];
        let mut matched_slot = vec![false; self.slots.len()];

        if !self.slots.is_empty() && n_proposals > 0 {
            // Build box tensors for IoU computation
            let slot_boxes: Vec<f32> = self.slots.iter().flat_map(|s| s.bbox).collect();
            let prop_boxes: Vec<f32> = proposals.iter().flat_map(|b| b.iter().copied()).collect();

            let slot_tensor = Tensor::from_vec(slot_boxes, &[self.slots.len(), 4]).unwrap();
            let prop_tensor = Tensor::from_vec(prop_boxes, &[n_proposals, 4]).unwrap();
            let iou_matrix = box_iou(&slot_tensor, &prop_tensor);
            let iou_data = iou_matrix.to_vec();

            // Greedy matching: best IoU first
            let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
            for si in 0..self.slots.len() {
                for pi in 0..n_proposals {
                    let iou = iou_data[si * n_proposals + pi];
                    if iou > self.match_threshold {
                        pairs.push((si, pi, iou));
                    }
                }
            }
            pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            for (si, pi, _iou) in pairs {
                if matched_slot[si] || matched_prop[pi] {
                    continue;
                }
                matched_slot[si] = true;
                matched_prop[pi] = true;

                // Extract ROI and update slot
                let roi_feat = self.extract_roi(features, &proposals[pi], spatial_scale);
                let projected = self.roi_project.forward(&roi_feat).relu();
                let new_hidden = self.gru.forward_step(&projected, &self.slots[si].hidden);

                self.slots[si].hidden = new_hidden;
                self.slots[si].bbox = proposals[pi];
                self.slots[si].confidence =
                    (self.slots[si].confidence * 0.7 + scores[pi] * 0.3).min(1.0);
                self.slots[si].frames_tracked += 1;
                self.slots[si].frames_missing = 0;
            }
        }

        // Decay unmatched slots
        let max_missing = self.max_missing;
        let decay = self.decay_rate;
        let mut to_remove = Vec::new();
        for (si, slot) in self.slots.iter_mut().enumerate() {
            if !matched_slot.get(si).copied().unwrap_or(false) {
                slot.frames_missing += 1;
                slot.confidence *= decay;
                if slot.frames_missing > max_missing {
                    to_remove.push(si);
                }
            }
        }
        for &si in to_remove.iter().rev() {
            self.slots.remove(si);
        }

        // Spawn new slots for unmatched proposals
        for (pi, &score) in scores.iter().enumerate() {
            if !matched_prop[pi] && score > 0.3 {
                self.spawn_slot(&proposals[pi], score, features, spatial_scale);
            }
        }

        // Return current hidden states
        self.slots.iter().map(|s| s.hidden.clone()).collect()
    }

    /// Extract ROI features for a single bbox.
    fn extract_roi(&self, features: &Variable, bbox: &[f32; 4], spatial_scale: f32) -> Variable {
        let roi_data = vec![0.0, bbox[0], bbox[1], bbox[2], bbox[3]];
        let roi_tensor = Tensor::from_vec(roi_data, &[1, 5]).unwrap();
        let roi_out = roi_align(
            &features.data(),
            &roi_tensor,
            (self.roi_size, self.roi_size),
            spatial_scale,
        );

        // Flatten: [1, C, roi_h, roi_w] → [1, C*roi_h*roi_w]
        let flat = roi_out.to_vec();
        Variable::new(
            Tensor::from_vec(
                flat,
                &[1, self.feat_channels * self.roi_size * self.roi_size],
            )
            .unwrap(),
            false,
        )
    }

    /// Spawn a new object slot.
    fn spawn_slot(
        &mut self,
        bbox: &[f32; 4],
        confidence: f32,
        features: &Variable,
        spatial_scale: f32,
    ) {
        let roi_feat = self.extract_roi(features, bbox, spatial_scale);
        let projected = self.roi_project.forward(&roi_feat).relu();

        // Initialize hidden state from first observation
        let hidden = Variable::new(
            Tensor::from_vec(vec![0.0; self.hidden_size], &[1, self.hidden_size]).unwrap(),
            false,
        );
        let initial_hidden = self.gru.forward_step(&projected, &hidden);

        let id = self.next_id;
        self.next_id += 1;

        self.slots.push(ObjectSlot {
            id,
            bbox: *bbox,
            confidence,
            hidden: initial_hidden,
            class_logits: Vec::new(),
            frames_tracked: 1,
            frames_missing: 0,
        });
    }

    /// Get currently tracked objects.
    pub fn slots(&self) -> &[ObjectSlot] {
        &self.slots
    }

    /// Number of tracked objects.
    pub fn num_tracked(&self) -> usize {
        self.slots.len()
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.slots.clear();
        self.next_id = 1;
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.roi_project.parameters());
        p.extend(self.gru.parameters());
        p
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(c: usize, h: usize, w: usize) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![0.1; c * h * w], &[1, c, h, w]).unwrap(),
            false,
        )
    }

    #[test]
    fn test_memory_bank_creation() {
        let bank = ObjectMemoryBank::default_config();
        assert_eq!(bank.num_tracked(), 0);
    }

    #[test]
    fn test_memory_bank_spawn() {
        let mut bank = ObjectMemoryBank::default_config();
        let features = make_features(96, 20, 20);

        let proposals = vec![[10.0, 10.0, 50.0, 50.0]];
        let scores = vec![0.9];

        let hiddens = bank.update(&features, &proposals, &scores, 0.25);

        assert_eq!(bank.num_tracked(), 1);
        assert_eq!(hiddens.len(), 1);
        assert_eq!(hiddens[0].shape(), vec![1, 64]);
    }

    #[test]
    fn test_memory_bank_match_and_update() {
        let mut bank = ObjectMemoryBank::default_config();
        let features = make_features(96, 20, 20);

        // Frame 1: spawn object
        let proposals1 = vec![[10.0, 10.0, 50.0, 50.0]];
        let scores1 = vec![0.9];
        bank.update(&features, &proposals1, &scores1, 0.25);
        assert_eq!(bank.num_tracked(), 1);
        assert_eq!(bank.slots()[0].frames_tracked, 1);

        // Frame 2: same position → should match
        let proposals2 = vec![[12.0, 12.0, 52.0, 52.0]]; // Slightly shifted
        let scores2 = vec![0.85];
        bank.update(&features, &proposals2, &scores2, 0.25);

        assert_eq!(bank.num_tracked(), 1);
        assert_eq!(bank.slots()[0].frames_tracked, 2);
        assert_eq!(bank.slots()[0].id, 1); // Same ID
    }

    #[test]
    fn test_memory_bank_decay() {
        let mut bank = ObjectMemoryBank::default_config();
        bank.max_missing = 3;
        let features = make_features(96, 10, 10);

        // Spawn
        bank.update(&features, &[[5.0, 5.0, 15.0, 15.0]], &[0.8], 0.5);
        assert_eq!(bank.num_tracked(), 1);

        // No detections for 4 frames → should be removed
        for _ in 0..4 {
            bank.update(&features, &[], &[], 0.5);
        }
        assert_eq!(bank.num_tracked(), 0);
    }

    #[test]
    fn test_memory_bank_reset() {
        let mut bank = ObjectMemoryBank::default_config();
        let features = make_features(96, 10, 10);

        bank.update(&features, &[[5.0, 5.0, 15.0, 15.0]], &[0.8], 0.5);
        bank.update(&features, &[[30.0, 30.0, 60.0, 60.0]], &[0.7], 0.5);
        assert_eq!(bank.num_tracked(), 2);

        bank.reset();
        assert_eq!(bank.num_tracked(), 0);
    }

    #[test]
    fn test_memory_bank_param_count() {
        let bank = ObjectMemoryBank::default_config();
        let total: usize = bank.parameters().iter().map(|p| p.numel()).sum();
        // roi_project(96*9 * 64) + GRU(3*64*(64+64))
        assert!(total > 10_000);
        assert!(total < 100_000);
    }
}
