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

pub mod data;
pub mod aquilo;
pub mod boreas;
pub mod naiad;
pub mod vulcan;
pub mod zephyrus;
pub mod colossus;
pub mod gaia;
pub mod apollo;
pub mod pipeline;

// Re-exports
pub use aquilo::Aquilo;
pub use boreas::Boreas;
pub use naiad::Naiad;
pub use vulcan::Vulcan;
pub use zephyrus::Zephyrus;
pub use colossus::Colossus;
pub use gaia::Gaia;
pub use apollo::Apollo;
pub use pipeline::HvacPipeline;
pub use data::{HvacSensorData, HvacLabels, PipelineOutput, SyntheticHvacGenerator};
