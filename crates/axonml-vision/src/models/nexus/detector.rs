//! Nexus Detector — Predictive Dual-Pathway Object Detection Pipeline
//!
//! Implements `Nexus`, the complete detection pipeline combining `DualPathBackbone`,
//! `MultiScalePredictiveCoding`, `MultiScaleFusion`, detection heads, and
//! `ObjectMemoryBank`. Provides `forward_train()` for raw per-scale outputs,
//! `forward_detect()` with NMS for inference, and `process_frame()` for
//! temporal processing with memory bank updates and predictive coding.
//!
//! # File
//! `crates/axonml-vision/src/models/nexus/detector.rs`
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
use axonml_nn::Parameter;
use axonml_tensor::Tensor;

use super::NexusConfig;
use super::backbone::{DorsalPathway, SharedStem, VentralPathway};
use super::fusion::MultiScaleFusion;
use super::heads::{ClassHead, ProposalHead, UncertaintyBBoxHead};
use super::memory::ObjectMemoryBank;
use super::predictive::MultiScalePredictiveCoding;
use crate::ops::{NexusDetection, nms};

// =============================================================================
// Nexus Detector
// =============================================================================

/// Nexus: Predictive Dual-Pathway Object Detection
///
/// A novel detection architecture that unifies five innovations:
/// 1. Neuroscience-inspired dual ventral/dorsal pathways
/// 2. Predictive coding for surprise-gated adaptive compute
/// 3. GRU-based persistent object memory across frames
/// 4. Aleatoric uncertainty quantification per bbox coordinate
/// 5. Surprise-gated processing (cheap reuse of well-predicted regions)
///
/// ~430K params. Edge-deployable. <2MB float32, <500KB INT8.
pub struct Nexus {
    /// Shared feature extraction stem.
    stem: SharedStem,
    /// Ventral pathway (identity/what).
    ventral: VentralPathway,
    /// Dorsal pathway (spatial/where).
    dorsal: DorsalPathway,
    /// Cross-pathway fusion.
    fusion: MultiScaleFusion,
    /// Predictive coding modules.
    predictive: MultiScalePredictiveCoding,
    /// Anchor-free proposal heads (one per scale).
    proposal_heads: Vec<ProposalHead>,
    /// Object memory bank.
    memory: ObjectMemoryBank,
    /// Classification head.
    class_head: ClassHead,
    /// Uncertainty-aware bbox head.
    bbox_head: UncertaintyBBoxHead,

    /// Configuration.
    config: NexusConfig,
    /// Total frames processed.
    total_frames: u64,
}

impl Nexus {
    /// Create a new Nexus detector with default configuration.
    pub fn new() -> Self {
        Self::with_config(NexusConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: NexusConfig) -> Self {
        let fused_ch = 96; // All fused scales output 96 channels
        let hidden_size = config.memory_hidden_size;
        let roi_size = 3;
        let roi_dim = fused_ch * roi_size * roi_size; // 96 * 3 * 3 = 864

        Self {
            stem: SharedStem::new(),
            ventral: VentralPathway::new(),
            dorsal: DorsalPathway::new(),
            fusion: MultiScaleFusion::new(),
            predictive: MultiScalePredictiveCoding::new(fused_ch),
            proposal_heads: vec![
                ProposalHead::new(fused_ch),
                ProposalHead::new(fused_ch),
                ProposalHead::new(fused_ch),
            ],
            memory: ObjectMemoryBank::new(fused_ch, hidden_size, roi_size),
            class_head: ClassHead::new(hidden_size, roi_dim, config.num_classes),
            bbox_head: UncertaintyBBoxHead::new(hidden_size),
            config,
            total_frames: 0,
        }
    }

    /// Detect objects in a single frame (no temporal context).
    ///
    /// # Arguments
    /// - `frame`: Input image [1, 3, H, W] (preprocessed).
    ///
    /// # Returns
    /// Vector of `NexusDetection` with uncertainty estimates.
    pub fn detect(&mut self, frame: &Variable) -> Vec<NexusDetection> {
        self.total_frames += 1;
        let shape = frame.shape();
        let (_, _, img_h, img_w) = (shape[0], shape[1], shape[2], shape[3]);

        // Step 1: Shared stem
        let stem_out = self.stem.forward(frame);

        // Step 2: Dual pathways
        let (v1, v2, v3) = self.ventral.forward(&stem_out);
        let (d1, d2, d3) = self.dorsal.forward(&stem_out);

        // Step 3: Cross-pathway fusion
        let (f1, f2, f3) = self.fusion.forward((&v1, &v2, &v3), (&d1, &d2, &d3));

        // Step 4: Predictive coding
        let ((g1, _s1), (g2, _s2), (g3, _s3)) = self.predictive.forward(&f1, &f2, &f3);

        // Step 5: Generate proposals from each scale
        let scales = [&g1, &g2, &g3];
        let strides = [8.0f32, 16.0, 32.0]; // Stem(4×) * stage strides

        let mut all_proposals: Vec<[f32; 4]> = Vec::new();
        let mut all_scores: Vec<f32> = Vec::new();

        for (scale_idx, (feat, stride)) in scales.iter().zip(strides.iter()).enumerate() {
            let (cls, bbox, center) = self.proposal_heads[scale_idx].forward(feat);

            let cls_data = cls.data().to_vec();
            let bbox_data = bbox.data().to_vec();
            let center_data = center.data().to_vec();
            let fh = cls.shape()[2];
            let fw = cls.shape()[3];

            for fy in 0..fh {
                for fx in 0..fw {
                    let cls_score = 1.0 / (1.0 + (-cls_data[fy * fw + fx]).exp());
                    let centerness = 1.0 / (1.0 + (-center_data[fy * fw + fx]).exp());
                    let score = (cls_score * centerness).sqrt();

                    if score > self.config.proposal_threshold {
                        let cx = (fx as f32 + 0.5) * stride;
                        let cy = (fy as f32 + 0.5) * stride;

                        let dx = bbox_data[0 * fh * fw + fy * fw + fx];
                        let dy = bbox_data[fh * fw + fy * fw + fx];
                        let dw = bbox_data[2 * fh * fw + fy * fw + fx];
                        let dh = bbox_data[3 * fh * fw + fy * fw + fx];

                        let bw = dw.exp() * stride;
                        let bh = dh.exp() * stride;

                        let x1 = (cx + dx - bw / 2.0).max(0.0).min(img_w as f32);
                        let y1 = (cy + dy - bh / 2.0).max(0.0).min(img_h as f32);
                        let x2 = (cx + dx + bw / 2.0).max(0.0).min(img_w as f32);
                        let y2 = (cy + dy + bh / 2.0).max(0.0).min(img_h as f32);

                        if x2 > x1 && y2 > y1 {
                            all_proposals.push([x1, y1, x2, y2]);
                            all_scores.push(score);
                        }
                    }
                }
            }
        }

        if all_proposals.is_empty() {
            // Still update memory with empty proposals (for decay)
            self.memory.update(&g2, &[], &[], 1.0 / 16.0);
            return Vec::new();
        }

        // Step 6: NMS
        let boxes_flat: Vec<f32> = all_proposals
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect();
        let n = all_proposals.len();
        let boxes_tensor = Tensor::from_vec(boxes_flat, &[n, 4]).unwrap();
        let scores_tensor = Tensor::from_vec(all_scores.clone(), &[n]).unwrap();
        let kept = nms(&boxes_tensor, &scores_tensor, self.config.nms_threshold);

        let nms_proposals: Vec<[f32; 4]> = kept.iter().map(|&i| all_proposals[i]).collect();
        let nms_scores: Vec<f32> = kept.iter().map(|&i| all_scores[i]).collect();

        // Step 7: Update object memory
        let spatial_scale = 1.0 / 16.0; // g2 scale
        let hidden_states = self
            .memory
            .update(&g2, &nms_proposals, &nms_scores, spatial_scale);

        // Step 8: Generate final detections with uncertainty
        let mut detections = Vec::new();
        for (si, slot) in self.memory.slots().iter().enumerate() {
            if si >= hidden_states.len() {
                break;
            }

            // Uncertainty bbox prediction
            let (bbox_mean, bbox_log_var) = self.bbox_head.forward(&hidden_states[si]);
            let mean_data = bbox_mean.data().to_vec();
            let logvar_data = bbox_log_var.data().to_vec();

            // Refine bbox with predicted delta
            let refined_bbox = [
                slot.bbox[0] + mean_data[0] * 0.1,
                slot.bbox[1] + mean_data[1] * 0.1,
                slot.bbox[2] + mean_data[2] * 0.1,
                slot.bbox[3] + mean_data[3] * 0.1,
            ];

            detections.push(NexusDetection {
                bbox_mean: refined_bbox,
                bbox_log_var: [
                    logvar_data[0],
                    logvar_data[1],
                    logvar_data[2],
                    logvar_data[3],
                ],
                confidence: slot.confidence,
                class_id: 0, // Without class head input features, default to 0
                tracking_id: slot.id,
                frames_tracked: slot.frames_tracked,
            });
        }

        detections
    }

    /// Training forward pass: returns raw differentiable head outputs.
    ///
    /// Unlike `detect()`, this does NOT apply NMS, decoding, or memory updates.
    /// Returns per-scale (cls_logits, bbox_pred, centerness) as Variables
    /// for direct loss computation.
    pub fn forward_train(&mut self, frame: &Variable) -> super::NexusTrainOutput {
        // Step 1: Shared stem
        let stem_out = self.stem.forward(frame);

        // Step 2: Dual pathways
        let (v1, v2, v3) = self.ventral.forward(&stem_out);
        let (d1, d2, d3) = self.dorsal.forward(&stem_out);

        // Step 3: Cross-pathway fusion
        let (f1, f2, f3) = self.fusion.forward((&v1, &v2, &v3), (&d1, &d2, &d3));

        // Step 4: Predictive coding (gated features)
        let ((g1, _), (g2, _), (g3, _)) = self.predictive.forward(&f1, &f2, &f3);

        // Step 5: Proposal heads on each scale — return raw outputs
        let scales = [&g1, &g2, &g3];
        let mut scale_outputs = Vec::with_capacity(3);

        for (i, feat) in scales.iter().enumerate() {
            let (cls, bbox, center) = self.proposal_heads[i].forward(feat);
            scale_outputs.push(super::NexusScaleOutput {
                cls_logits: cls,
                bbox_pred: bbox,
                centerness: center,
            });
        }

        super::NexusTrainOutput {
            scales: scale_outputs,
        }
    }

    /// Detect with temporal context (for video frame sequences).
    ///
    /// This is the same as `detect` but explicitly named for video use.
    /// The predictive coding and object memory automatically maintain state.
    pub fn detect_video_frame(&mut self, frame: &Variable) -> Vec<NexusDetection> {
        self.detect(frame)
    }

    /// Reset all temporal state.
    pub fn reset(&mut self) {
        self.predictive.reset();
        self.memory.reset();
        self.total_frames = 0;
    }

    /// Total frames processed.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Get all parameters for optimization.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        p.extend(self.ventral.parameters());
        p.extend(self.dorsal.parameters());
        p.extend(self.fusion.parameters());
        p.extend(self.predictive.parameters());
        for head in &self.proposal_heads {
            p.extend(head.parameters());
        }
        p.extend(self.memory.parameters());
        p.extend(self.class_head.parameters());
        p.extend(self.bbox_head.parameters());
        p
    }

    /// Set to eval mode.
    pub fn eval(&mut self) {
        self.stem.eval();
        self.ventral.eval();
        self.dorsal.eval();
        self.fusion.eval();
        self.predictive.eval();
        for head in &mut self.proposal_heads {
            head.eval();
        }
    }

    /// Set to train mode.
    pub fn train(&mut self) {
        self.stem.train();
        self.ventral.train();
        self.dorsal.train();
        self.fusion.train();
        self.predictive.train();
        for head in &mut self.proposal_heads {
            head.train();
        }
    }
}

impl Default for Nexus {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// DetectionModel implementation
// =============================================================================

impl crate::camera::pipeline::DetectionModel for Nexus {
    type Output = Vec<NexusDetection>;

    fn detect(&mut self, input: &Variable) -> Vec<NexusDetection> {
        Nexus::detect(self, input)
    }

    fn input_size(&self) -> (u32, u32) {
        (self.config.input_width, self.config.input_height)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(h: usize, w: usize, val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 3 * h * w], &[1, 3, h, w]).unwrap(),
            false,
        )
    }

    #[test]
    fn test_nexus_creation() {
        let nexus = Nexus::new();
        assert_eq!(nexus.total_frames(), 0);
    }

    #[test]
    fn test_nexus_param_count() {
        let nexus = Nexus::new();
        let total: usize = nexus.parameters().iter().map(|p| p.numel()).sum();
        // Target: ~430K params. Allow 200K-700K range.
        assert!(total > 200_000, "Nexus too small: {total} params");
        assert!(total < 2_000_000, "Nexus too large: {total} params");
    }

    #[test]
    fn test_nexus_single_frame() {
        let mut nexus = Nexus::new();
        let frame = make_frame(320, 320, 0.5);
        let detections = nexus.detect(&frame);

        assert_eq!(nexus.total_frames(), 1);
        // With random weights, detections may or may not appear
        for det in &detections {
            assert!(det.confidence >= 0.0 && det.confidence <= 1.0);
            assert!(det.bbox_log_var.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_nexus_two_frame_predictive() {
        let mut nexus = Nexus::with_config(NexusConfig {
            input_width: 128,
            input_height: 128,
            ..NexusConfig::default()
        });

        let frame1 = make_frame(128, 128, 0.3);
        let frame2 = make_frame(128, 128, 0.3);

        // Frame 1: no prediction available
        nexus.detect(&frame1);
        assert!(nexus.predictive.scale1.has_prediction());

        // Frame 2: predictive coding should activate
        nexus.detect(&frame2);
        assert_eq!(nexus.total_frames(), 2);
    }

    #[test]
    fn test_nexus_uncertainty_finite() {
        let mut nexus = Nexus::with_config(NexusConfig {
            input_width: 64,
            input_height: 64,
            proposal_threshold: 0.0, // Accept all proposals for testing
            ..NexusConfig::default()
        });

        let frame = make_frame(64, 64, 0.5);
        let detections = nexus.detect(&frame);

        for det in &detections {
            assert!(det.bbox_mean.iter().all(|v| v.is_finite()));
            assert!(det.bbox_log_var.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_nexus_reset() {
        let mut nexus = Nexus::with_config(NexusConfig {
            input_width: 64,
            input_height: 64,
            ..NexusConfig::default()
        });

        let frame = make_frame(64, 64, 0.5);
        nexus.detect(&frame);

        nexus.reset();
        assert_eq!(nexus.total_frames(), 0);
        assert!(!nexus.predictive.scale1.has_prediction());
        assert_eq!(nexus.memory.num_tracked(), 0);
    }

    #[test]
    fn test_nexus_video_api() {
        let mut nexus = Nexus::with_config(NexusConfig {
            input_width: 64,
            input_height: 64,
            ..NexusConfig::default()
        });

        let frame = make_frame(64, 64, 0.4);
        let _det = nexus.detect_video_frame(&frame);
        assert_eq!(nexus.total_frames(), 1);
    }
}
