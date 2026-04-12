//! Axonml HVAC — Domain-Specific HVAC Diagnostic Models
//!
//! # File
//! `crates/axonml-hvac/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Overview
//!
//! Nine models for HVAC fault detection and diagnostic reasoning, built on
//! the AxonML deep learning framework:
//!
//! | Model | Purpose |
//! |-------|---------|
//! | [`apollo`]   | Primary fault classifier |
//! | [`aquilo`]   | Airflow anomaly detector |
//! | [`boreas`]   | Cold-side (cooling) specialist |
//! | [`colossus`] | Large transformer diagnostician |
//! | [`gaia`]     | Environmental context encoder |
//! | [`naiad`]    | Water-side (hydronic) specialist |
//! | [`panoptes`] | Observability / multi-signal fusion |
//! | [`vulcan`]   | Heat-side specialist |
//! | [`zephyrus`] | Temporal predictor / autoencoder |
//!
//! Plus supporting modules:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`data`]             | Sensor data types, synthetic generator |
//! | [`panoptes_datagen`] | Warren HVAC simulator for Panoptes training |
//! | [`pipeline`]         | End-to-end 8-model diagnostic pipeline |
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

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

// Re-exports
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
