//! Phantom — Temporal Event-Driven Face Detection
//!
//! # File
//! `crates/axonml-vision/src/models/phantom/mod.rs`
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
