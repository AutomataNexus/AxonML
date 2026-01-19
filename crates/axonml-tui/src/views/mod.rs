//! Views - TUI View Components
//!
//! Each view represents a tab/screen in the Axonml TUI.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

mod model;
mod data;
mod training;
mod graphs;
mod files;
mod help;

pub use model::ModelView;
pub use data::DataView;
pub use training::TrainingView;
pub use graphs::GraphsView;
pub use files::FilesView;
pub use help::HelpView;
