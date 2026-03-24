//! Phantom Detector — Temporal Event-Driven Face Detection
//!
//! # File
//! `crates/axonml-vision/src/models/phantom/detector.rs`
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
use axonml_nn::{Conv2d, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use super::PhantomConfig;
use super::backbone::{EventFeatureExtractor, PhantomBackbone};
use super::events::EventEncoder;
use super::tracker::FaceStateTracker;
use crate::ops::PhantomFaceDetection;

// =============================================================================
// Phantom Detector
// =============================================================================

/// Phantom: Temporal Event-Driven Face Detection
///
/// A novel face detection architecture that mimics neuromorphic event cameras
/// on standard hardware. Key innovations:
///
/// 1. **Pseudo-events**: Frame differencing generates sparse event maps
/// 2. **Sparse processing**: Only event-active regions get heavy computation
/// 3. **Predictive tracking**: GRU state per face predicts next location
/// 4. **Implicit identity**: Tracking ID from temporal continuity
/// 5. **Confidence accumulation**: Long-tracked faces = higher confidence
///
/// ~126K parameters. <500KB float32. Runs on Pi at 30+ FPS for static scenes.
pub struct Phantom {
    /// Event encoder for pseudo-event generation.
    event_encoder: EventEncoder,
    /// Lightweight backbone for multi-scale features.
    backbone: PhantomBackbone,
    /// Event patch feature extractor.
    event_extractor: EventFeatureExtractor,
    /// GRU-based face state tracker.
    tracker: FaceStateTracker,

    /// Face detection head on backbone P2 features.
    face_conv: Conv2d,
    face_cls: Conv2d,
    face_bbox: Conv2d,

    /// Merge P2 ROI features to tracker input dimension.
    roi_project: Linear,

    /// Configuration.
    config: PhantomConfig,
    /// Total frames processed.
    total_frames: u64,
    /// Frames that used cached backbone (sparse path).
    cached_frames: u64,
}

impl Phantom {
    /// Create a new Phantom detector with default configuration.
    pub fn new() -> Self {
        Self::with_config(PhantomConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: PhantomConfig) -> Self {
        let mut backbone = PhantomBackbone::new();
        backbone.refresh_interval = config.backbone_refresh_interval;

        // Face detection head operates on P2 features (32 channels)
        let face_conv = Conv2d::with_options(32, 32, (3, 3), (1, 1), (1, 1), true);
        let face_cls = Conv2d::with_options(32, 1, (1, 1), (1, 1), (0, 0), true);
        let face_bbox = Conv2d::with_options(32, 4, (1, 1), (1, 1), (0, 0), true);

        // Project P2 ROI features (32ch, pooled to fixed size) to tracker input
        // Flatten 32 * 3 * 3 = 288 → 48
        let roi_project = Linear::new(32 * 3 * 3, 48);

        let tracker = FaceStateTracker::new(48, 32, config.tracker_hidden_size);

        Self {
            event_encoder: EventEncoder::new(),
            backbone,
            event_extractor: EventFeatureExtractor::new(),
            tracker,
            face_conv,
            face_cls,
            face_bbox,
            roi_project,
            config,
            total_frames: 0,
            cached_frames: 0,
        }
    }

    /// Detect faces in a single frame.
    ///
    /// # Arguments
    /// - `frame`: Input frame [1, 3, H, W] (preprocessed, normalized).
    ///
    /// # Returns
    /// Vector of `PhantomFaceDetection` with tracking IDs and confidence.
    pub fn detect_frame(&mut self, frame: &Variable) -> Vec<PhantomFaceDetection> {
        self.total_frames += 1;
        let shape = frame.shape();
        let (_b, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // Step 1: Generate events
        let (_density, _active_cells, event_map) = self.event_encoder.encode(frame);
        let _is_cold = self.event_encoder.is_cold_start()
            || !self.backbone.has_cache()
            || self.total_frames == 1;

        // Step 2: Get backbone features (full run or cached)
        let (features, was_full) = self.backbone.get_features(frame);
        if !was_full {
            self.cached_frames += 1;
        }

        // Step 3: Run face detection head on P2 features
        let p2 = &features[1]; // [1, 32, H/4, W/4]
        let face_feat = self.face_conv.forward(p2).relu();
        let cls_map = self.face_cls.forward(&face_feat); // [1, 1, H/4, W/4]
        let bbox_map = self.face_bbox.forward(&face_feat); // [1, 4, H/4, W/4]

        // Step 4: Decode face proposals
        let cls_data = cls_map.data().to_vec();
        let bbox_data = bbox_map.data().to_vec();
        let fh = cls_map.shape()[2];
        let fw = cls_map.shape()[3];

        let stride = 4.0; // P2 stride relative to backbone input (stem s=2, stage2 s=2)
        let mut detected_bboxes = Vec::new();
        let mut roi_features = Vec::new();

        for fy in 0..fh {
            for fx in 0..fw {
                let score = 1.0 / (1.0 + (-cls_data[fy * fw + fx]).exp()); // sigmoid

                if score > self.config.detection_threshold {
                    // Decode bbox relative to cell center
                    let cx = (fx as f32 + 0.5) * stride;
                    let cy = (fy as f32 + 0.5) * stride;

                    let dx = bbox_data[0 * fh * fw + fy * fw + fx];
                    let dy = bbox_data[fh * fw + fy * fw + fx];
                    let dw = bbox_data[2 * fh * fw + fy * fw + fx];
                    let dh = bbox_data[3 * fh * fw + fy * fw + fx];

                    let box_w = dw.exp() * stride;
                    let box_h = dh.exp() * stride;

                    let x1 = (cx + dx - box_w / 2.0).max(0.0);
                    let y1 = (cy + dy - box_h / 2.0).max(0.0);
                    let x2 = (cx + dx + box_w / 2.0).min(w as f32);
                    let y2 = (cy + dy + box_h / 2.0).min(h as f32);

                    detected_bboxes.push([x1, y1, x2, y2]);

                    // Extract ROI feature (simple: use 3x3 around detection cell)
                    let mut roi = vec![0.0f32; 32 * 3 * 3];
                    let p2_data = p2.data().to_vec();
                    let p2h = p2.shape()[2];
                    let p2w = p2.shape()[3];

                    for c in 0..32 {
                        for dy_r in 0..3i32 {
                            for dx_r in 0..3i32 {
                                let sy = (fy as i32 + dy_r - 1).clamp(0, p2h as i32 - 1) as usize;
                                let sx = (fx as i32 + dx_r - 1).clamp(0, p2w as i32 - 1) as usize;
                                roi[c * 9 + dy_r as usize * 3 + dx_r as usize] =
                                    p2_data[c * p2h * p2w + sy * p2w + sx];
                            }
                        }
                    }
                    roi_features.push(roi);
                }
            }
        }

        let n_det = detected_bboxes.len();
        if n_det == 0 {
            // No new detections — still update tracker (decay missing faces)
            let empty_bb = Variable::new(Tensor::from_vec(vec![], &[0, 48]).unwrap(), false);
            let empty_ev = Variable::new(Tensor::from_vec(vec![], &[0, 32]).unwrap(), false);
            return self.tracker.update(&empty_bb, &empty_ev, &[]);
        }

        // Step 5: Project ROI features → [N, 48]
        let flat_roi: Vec<f32> = roi_features.into_iter().flatten().collect();
        let roi_var = Variable::new(
            Tensor::from_vec(flat_roi, &[n_det, 32 * 3 * 3]).unwrap(),
            false,
        );
        let backbone_feats = self.roi_project.forward(&roi_var).relu(); // [N, 48]

        // Step 6: Extract event features for detected patches
        // Create 4-channel patches (RGB + motion) at each detection
        let event_feats = self.extract_event_features(frame, &event_map, &detected_bboxes);

        // Step 7: Update tracker
        self.tracker
            .update(&backbone_feats, &event_feats, &detected_bboxes)
    }

    /// Extract event features for detected face regions.
    fn extract_event_features(
        &self,
        frame: &Variable,
        event_map: &Variable,
        bboxes: &[[f32; 4]],
    ) -> Variable {
        let n = bboxes.len();
        if n == 0 {
            return Variable::new(Tensor::from_vec(vec![0.0f32; 0], &[0, 32]).unwrap(), false);
        }

        let shape = frame.shape();
        let (_b, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let frame_data = frame.data().to_vec();
        let event_data = event_map.data().to_vec();

        // For each bbox, extract 48×48 patch with 4 channels (RGB + motion)
        let patch_size = 48;
        let mut all_patches = vec![0.0f32; n * 4 * patch_size * patch_size];

        for (di, bbox) in bboxes.iter().enumerate() {
            let bx1 = bbox[0].max(0.0) as usize;
            let by1 = bbox[1].max(0.0) as usize;
            let bx2 = (bbox[2] as usize).min(w);
            let by2 = (bbox[3] as usize).min(h);

            let bw = bx2.saturating_sub(bx1).max(1);
            let bh = by2.saturating_sub(by1).max(1);

            let base = di * 4 * patch_size * patch_size;

            for py in 0..patch_size {
                for px in 0..patch_size {
                    let src_x = bx1 + px * bw / patch_size;
                    let src_y = by1 + py * bh / patch_size;
                    let src_x = src_x.min(w.saturating_sub(1));
                    let src_y = src_y.min(h.saturating_sub(1));

                    // RGB channels
                    for c in 0..3 {
                        all_patches[base + c * patch_size * patch_size + py * patch_size + px] =
                            frame_data[c * h * w + src_y * w + src_x];
                    }
                    // Motion channel
                    all_patches[base + 3 * patch_size * patch_size + py * patch_size + px] =
                        event_data[src_y * w + src_x];
                }
            }
        }

        let patches = Variable::new(
            Tensor::from_vec(all_patches, &[n, 4, patch_size, patch_size]).unwrap(),
            false,
        );

        self.event_extractor.forward(&patches)
    }

    /// Training forward pass: returns raw differentiable head outputs.
    ///
    /// Always runs full backbone (no caching during training).
    /// Returns (face_cls, face_bbox) as Variables for loss computation.
    pub fn forward_train(&mut self, frame: &Variable) -> super::PhantomTrainOutput {
        // Always run full backbone during training (no caching)
        let features = self.backbone.forward_full(frame);

        // Face detection head on P2 features
        let p2 = &features[1]; // [1, 32, H/4, W/4]
        let face_feat = self.face_conv.forward(p2).relu();
        let face_cls = self.face_cls.forward(&face_feat);
        let face_bbox = self.face_bbox.forward(&face_feat);

        super::PhantomTrainOutput {
            face_cls,
            face_bbox,
        }
    }

    /// Get total frames processed.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Get fraction of frames that used cached backbone (compute savings).
    pub fn cache_hit_ratio(&self) -> f32 {
        if self.total_frames > 0 {
            self.cached_frames as f32 / self.total_frames as f32
        } else {
            0.0
        }
    }

    /// Reset all state (events, backbone cache, tracker).
    pub fn reset(&mut self) {
        self.event_encoder.reset();
        self.backbone.invalidate_cache();
        self.tracker.reset();
        self.total_frames = 0;
        self.cached_frames = 0;
    }

    /// Get all parameters for optimization.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters());
        p.extend(self.event_extractor.parameters());
        p.extend(self.face_conv.parameters());
        p.extend(self.face_cls.parameters());
        p.extend(self.face_bbox.parameters());
        p.extend(self.roi_project.parameters());
        p.extend(self.tracker.parameters());
        p
    }

    /// Set to eval mode.
    pub fn eval(&mut self) {
        self.backbone.eval();
        self.event_extractor.eval();
    }

    /// Set to train mode.
    pub fn train(&mut self) {
        self.backbone.train();
        self.event_extractor.train();
    }

    /// Get the tracker (for inspection/testing).
    pub fn tracker(&self) -> &FaceStateTracker {
        &self.tracker
    }
}

impl Default for Phantom {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// DetectionModel implementation
// =============================================================================

impl crate::camera::pipeline::DetectionModel for Phantom {
    type Output = Vec<PhantomFaceDetection>;

    fn detect(&mut self, input: &Variable) -> Vec<PhantomFaceDetection> {
        self.detect_frame(input)
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
    fn test_phantom_creation() {
        let phantom = Phantom::new();
        assert_eq!(phantom.total_frames(), 0);
        assert_eq!(phantom.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_phantom_param_count() {
        let phantom = Phantom::new();
        let total: usize = phantom.parameters().iter().map(|p| p.numel()).sum();
        assert!(total < 200_000, "Phantom too large: {total} params");
        assert!(total > 20_000, "Phantom too small: {total} params");
    }

    #[test]
    fn test_phantom_single_frame() {
        let mut phantom = Phantom::new();
        let frame = make_frame(128, 128, 0.5);
        let detections = phantom.detect_frame(&frame);

        // First frame: cold start, random weights → may or may not detect
        assert_eq!(phantom.total_frames(), 1);
        // Detections are valid structs
        for det in &detections {
            assert!(det.confidence >= 0.0 && det.confidence <= 1.0);
            assert!(det.frames_tracked >= 1);
        }
    }

    #[test]
    fn test_phantom_two_frames_identical() {
        let mut phantom = Phantom::new();
        let frame = make_frame(64, 64, 0.3);

        phantom.detect_frame(&frame);
        let _det2 = phantom.detect_frame(&frame);

        assert_eq!(phantom.total_frames(), 2);
    }

    #[test]
    fn test_phantom_static_scene_efficiency() {
        let mut phantom = Phantom::new();
        phantom.backbone.refresh_interval = 10;

        let frame = make_frame(64, 64, 0.3);

        // Run 10 frames of static scene
        for _ in 0..10 {
            phantom.detect_frame(&frame);
        }

        // Most frames should use cached backbone
        assert!(phantom.cache_hit_ratio() > 0.5);
    }

    #[test]
    fn test_phantom_reset() {
        let mut phantom = Phantom::new();
        let frame = make_frame(64, 64, 0.5);
        phantom.detect_frame(&frame);

        phantom.reset();
        assert_eq!(phantom.total_frames(), 0);
        assert_eq!(phantom.cache_hit_ratio(), 0.0);
        assert_eq!(phantom.tracker().num_tracked(), 0);
    }
}
