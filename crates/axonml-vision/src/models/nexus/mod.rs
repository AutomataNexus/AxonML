//! Nexus — Predictive Dual-Pathway Object Detection
//!
//! A novel detection architecture that unifies five innovations never
//! combined before:
//!
//! # Key Innovations
//!
//! 1. **Dual-pathway processing** — Separate ventral ("what", identity) and
//!    dorsal ("where", spatial) streams inspired by primate visual cortex
//! 2. **Predictive coding** — Maintains prediction of current features;
//!    only prediction errors get full compute (free-energy principle)
//! 3. **Persistent object memory** — GRU hidden state per detected object
//!    across frames (refine, don't re-detect)
//! 4. **Uncertainty quantification** — Every bbox outputs mean + log-variance
//!    (aleatoric uncertainty for localization)
//! 5. **Surprise-gated adaptive compute** — High prediction-error regions get
//!    deep processing; well-predicted regions get cheap reuse
//!
//! ~430K params. <2MB float32. <500KB INT8. Edge-deployable.
//!
//! @version 0.1.0

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
