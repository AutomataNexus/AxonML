//! Views - TUI View Components
//!
//! # File
//! `crates/axonml-tui/src/views/mod.rs`
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

mod data;
mod files;
mod graphs;
mod help;
mod model;
mod training;

pub use data::DataView;
pub use files::FilesView;
pub use graphs::GraphsView;
pub use help::HelpView;
pub use model::ModelView;
pub use training::TrainingView;
