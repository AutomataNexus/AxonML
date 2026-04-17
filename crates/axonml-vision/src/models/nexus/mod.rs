//! Nexus — Predictive Dual-Pathway Object Detection
//!
//! Top-level module for the Nexus detector. Re-exports `Nexus` (full pipeline),
//! `MultiScaleFusion`, `ObjectMemoryBank`, and `MultiScalePredictiveCoding`.
//! Defines `NexusConfig` (input size, num_classes, GRU memory hidden dimension,
//! proposal/NMS thresholds), `NexusScaleOutput` (per-scale cls logits, bbox
//! predictions, and centerness), and `NexusTrainOutput` (3-scale raw outputs
//! for loss computation). The Nexus architecture combines a dual-pathway
//! backbone, predictive coding surprise gating, attention-gated fusion, and
//! GRU-based object memory for temporal object tracking.
//!
//! # File
//! `crates/axonml-vision/src/models/nexus/mod.rs`
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

// =============================================================================
// Sub-Modules and Re-Exports
// =============================================================================

pub mod backbone;
pub mod detector;
pub mod fusion;
pub mod heads;
pub mod memory;
pub mod predictive;

pub use detector::Nexus;
pub use fusion::MultiScaleFusion;
pub use memory::ObjectMemoryBank;
pub use predictive::MultiScalePredictiveCoding;

use axonml_autograd::Variable;

// =============================================================================
// Training Output Types
// =============================================================================

/// Per-scale training outputs from Nexus.
pub struct NexusScaleOutput {
    /// Classification logits [1, 1, H, W].
    pub cls_logits: Variable,
    /// Bounding box predictions [1, 4, H, W].
    pub bbox_pred: Variable,
    /// Centerness predictions [1, 1, H, W].
    pub centerness: Variable,
}

/// Training output from Nexus (raw head outputs, no NMS).
pub struct NexusTrainOutput {
    /// Per-scale outputs (3 scales).
    pub scales: Vec<NexusScaleOutput>,
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Nexus detector.
#[derive(Debug, Clone)]
pub struct NexusConfig {
    /// Input image width.
    pub input_width: u32,
    /// Input image height.
    pub input_height: u32,
    /// Number of object classes.
    pub num_classes: usize,
    /// GRU hidden dimension for object memory.
    pub memory_hidden_size: usize,
    /// Minimum proposal score threshold.
    pub proposal_threshold: f32,
    /// NMS IoU threshold.
    pub nms_threshold: f32,
}

impl Default for NexusConfig {
    fn default() -> Self {
        Self {
            input_width: 320,
            input_height: 320,
            num_classes: 20,
            memory_hidden_size: 64,
            proposal_threshold: 0.3,
            nms_threshold: 0.5,
        }
    }
}
