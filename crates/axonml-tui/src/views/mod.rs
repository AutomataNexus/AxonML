//! Views - TUI View Components
//!
//! Each view represents a tab/screen in the Axonml TUI.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

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
