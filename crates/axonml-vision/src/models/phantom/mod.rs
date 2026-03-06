//! Phantom — Temporal Event-Driven Face Detection
//!
//! A novel face detection architecture that generates pseudo-events from
//! standard camera frames (no neuromorphic hardware needed), processes only
//! changed regions, and maintains persistent face identity via GRU state.
//!
//! # Key Innovations
//!
//! 1. **Pseudo-event generation** on standard cameras via frame differencing
//! 2. **Sparse processing** — only event-active regions get heavy compute
//! 3. **Predictive tracking** — GRU state per face predicts next location
//! 4. **Implicit identity** — tracking ID from temporal continuity
//! 5. **Confidence accumulation** — faces tracked over time gain higher confidence
//!
//! # Efficiency Profile
//!
//! - Frame 1: 100% compute (full backbone, cold start)
//! - Frame 5: ~30% compute (only event regions processed)
//! - Frame 30: ~5% compute (predictions accurate, minimal events)
//! - Static scene: ~0% compute (cached backbone, no events)
//!
//! ~126K params. <500KB float32. <130KB INT8. Trivially runs on Pi.
//!
//! @version 0.1.0

pub mod backbone;
pub mod detector;
pub mod events;
pub mod tracker;

pub use detector::Phantom;
pub use events::{EventConfig, EventEncoder};
pub use tracker::FaceStateTracker;

use axonml_autograd::Variable;

/// Training output from Phantom (raw head outputs, no decoding).
pub struct PhantomTrainOutput {
    /// Face classification logits [1, 1, H/4, W/4].
    pub face_cls: Variable,
    /// Face bounding box predictions [1, 4, H/4, W/4].
    pub face_bbox: Variable,
}

/// Configuration for the Phantom detector.
#[derive(Debug, Clone)]
pub struct PhantomConfig {
    /// Input image width.
    pub input_width: u32,
    /// Input image height.
    pub input_height: u32,
    /// How often to run full backbone (in frames).
    pub backbone_refresh_interval: u32,
    /// GRU hidden state dimension for face tracker.
    pub tracker_hidden_size: usize,
    /// Face detection confidence threshold.
    pub detection_threshold: f32,
}

impl Default for PhantomConfig {
    fn default() -> Self {
        Self {
            input_width: 128,
            input_height: 128,
            backbone_refresh_interval: 30,
            tracker_hidden_size: 64,
            detection_threshold: 0.5,
        }
    }
}
