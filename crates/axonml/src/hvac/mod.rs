//! HVAC 8-Model Diagnostic System
//!
//! # File
//! `crates/axonml/src/hvac/mod.rs`
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
