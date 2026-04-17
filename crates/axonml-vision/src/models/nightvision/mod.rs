//! NightVision — Multi-Domain Infrared Object Detection
//!
//! A domain-adaptive infrared detection model for wildlife monitoring, human
//! detection (SAR/security), and astronomical thermal signatures. Uses a thermal
//! stem (adaptive 1/3-channel input), CSP backbone for multi-scale feature
//! extraction, a thermal FPN for feature pyramid fusion, and decoupled YOLOX-
//! style detection heads with optional domain tagging. Handles unique IR
//! characteristics including inverted contrast, thermal bloom, and varying
//! emissivity.
//!
//! # File
//! `crates/axonml-vision/src/models/nightvision/mod.rs`
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

//! NightVision is a multi-domain infrared object detection model designed for
//! deployment across wildly different thermal imaging scenarios:
//!
//! - **Wildlife monitoring** — detect animals in thermal camera footage
//! - **Human detection** — search and rescue, perimeter security, surveillance
//! - **Interstellar/astronomical** — detect thermal signatures in space imagery
//!
//! The model uses a domain-adaptive backbone with thermal-aware feature extraction
//! that handles the unique characteristics of IR imagery: no color, inverted
//! contrast, thermal bloom, low spatial resolution, and varying emissivity.
//!
//! # Architecture
//!
//! ```text
//! IR Image [B, 1, H, W]  (single-channel thermal)
//!      or  [B, 3, H, W]  (pseudo-color / multi-band IR)
//!     |
//! ┌───────────────────────────┐
//! │ Thermal Stem               │  Adaptive input (1 or 3 ch) → 32 channels
//! │ Conv2d + BN + SiLU         │  Thermal normalization preprocessing
//! └───────────────────────────┘
//!     |
//! ┌───────────────────────────┐
//! │ CSP Backbone               │  Cross-Stage Partial blocks
//! │ Stage 1: 32 → 64, s=2    │  Progressive feature extraction
//! │ Stage 2: 64 → 128, s=2   │  Multi-scale outputs for FPN
//! │ Stage 3: 128 → 256, s=2  │
//! └───────────────────────────┘
//!     |
//! ┌───────────────────────────┐
//! │ Thermal FPN (Feature       │  Feature Pyramid Network
//! │ Pyramid Network)           │  Fuses multi-scale features
//! │ P3 (small) + P4 + P5 (lg) │  Top-down + lateral connections
//! └───────────────────────────┘
//!     |
//! ┌───────────────────────────┐
//! │ Detection Head             │  Per-scale: cls + bbox + objectness
//! │ Decoupled heads (YOLOX)    │  + optional domain tag
//! └───────────────────────────┘
//!     |
//!  Detections: [class, x, y, w, h, confidence, domain]
//! ```

// =============================================================================
// Sub-Modules and Re-Exports
// =============================================================================

pub mod backbone;
pub mod detector;
pub mod head;
pub mod neck;

pub use detector::{NightVision, NightVisionConfig, NightVisionLoss, ThermalDomain};
