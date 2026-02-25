//! HVAC 8-Model Diagnostic System
//!
//! A complete HVAC fault diagnosis pipeline with 8 neural network models
//! organized in a 4-stage hierarchy:
//!
//! **Stage 1 — Specialists** (5 models, run in parallel):
//! - `Aquilo` — Electrical systems (~608K params)
//! - `Boreas` — Refrigeration systems (~1.2M params)
//! - `Naiad` — Water systems (~533K params)
//! - `Vulcan` — Mechanical systems (~1.1M params)
//! - `Zephyrus` — Airflow systems (~845K params)
//!
//! **Stage 2 — Aggregator**:
//! - `Colossus` — Cross-specialist fusion (~1.5M params)
//!
//! **Stage 3 — Safety**:
//! - `Gaia` — Safety validation with adversarial robustness (~896K params)
//!
//! **Stage 4 — Coordinator**:
//! - `Apollo` — Master diagnosis and action planning (~1.8M params)
//!
//! Total: ~8.6M parameters
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

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
