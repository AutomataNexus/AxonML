//! TUI Views — Screen-Level View Components
//!
//! Groups the top-level view widgets rendered by the AxonML TUI. Declares
//! the six view submodules — `data`, `files`, `graphs`, `help`, `model`,
//! `training` — and re-exports one public view struct from each:
//! `DataView`, `FilesView`, `GraphsView`, `HelpView`, `ModelView`,
//! `TrainingView`. Each of those types is the entry point the TUI app's
//! router uses to paint a full-screen view.
//!
//! # File
//! `crates/axonml-tui/src/views/mod.rs`
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
// Sub-Modules
// =============================================================================

mod data;
mod files;
mod graphs;
mod help;
mod model;
mod training;

// =============================================================================
// Re-Exports
// =============================================================================

pub use data::DataView;
pub use files::FilesView;
pub use graphs::GraphsView;
pub use help::HelpView;
pub use model::ModelView;
pub use training::TrainingView;
