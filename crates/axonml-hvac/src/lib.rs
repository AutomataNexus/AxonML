//! Axonml HVAC — Domain-Specific HVAC Diagnostic Models
//!
//! Top-level crate module aggregating nine named neural-network models for
//! HVAC fault detection and diagnostic reasoning on top of the AxonML deep
//! learning framework. `apollo` hosts the primary fault classifier, `aquilo`
//! the airflow anomaly detector, `boreas` the cold-side cooling specialist,
//! `colossus` a large transformer diagnostician, `gaia` an environmental
//! context encoder, `naiad` the water-side hydronic specialist, `panoptes`
//! the observability / multi-signal fusion model, `vulcan` the heat-side
//! specialist, and `zephyrus` the temporal predictor / autoencoder.
//! Supporting modules are `data` (the `HvacSensorData`, `HvacLabels`,
//! `PipelineOutput`, and `SyntheticHvacGenerator` types), `panoptes_datagen`
//! (`PanoptesTrainingData` + the `WarrenSimulator` HVAC scenario engine), and
//! `pipeline` which wires the models into the end-to-end `HvacPipeline`. The
//! crate re-exports each model struct alongside these helpers as the public
//! surface. Clippy lints for the cast, doc, naming, and arity families are
//! selectively relaxed at the crate root to accommodate the numeric-heavy
//! training code.
//!
//! # File
//! `crates/axonml-hvac/src/lib.rs`
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
// Crate-Level Lints
// =============================================================================

#![warn(clippy::all)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod apollo;
pub mod aquilo;
pub mod boreas;
pub mod colossus;
pub mod data;
pub mod gaia;
pub mod naiad;
pub mod panoptes;
pub mod panoptes_datagen;
pub mod pipeline;
pub mod vulcan;
pub mod zephyrus;

// =============================================================================
// Public Re-exports
// =============================================================================

pub use apollo::Apollo;
pub use aquilo::Aquilo;
pub use boreas::Boreas;
pub use colossus::Colossus;
pub use data::{HvacLabels, HvacSensorData, PipelineOutput, SyntheticHvacGenerator};
pub use gaia::Gaia;
pub use naiad::Naiad;
pub use panoptes::Panoptes;
pub use panoptes_datagen::{PanoptesTrainingData, WarrenSimulator};
pub use pipeline::HvacPipeline;
pub use vulcan::Vulcan;
pub use zephyrus::Zephyrus;
