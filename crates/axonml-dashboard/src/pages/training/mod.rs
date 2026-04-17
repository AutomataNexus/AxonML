//! Training Pages — Module Root for Training and Notebook Views
//!
//! Module root that re-exports the training-subsection page components.
//! Contains two feature groupings:
//!
//! - Classic training runs: `list` (run history), `detail` (live run
//!   inspection), `new` (launch a new training run).
//! - Jupyter-style notebooks: `notebook_list` (notebook registry),
//!   `notebook_editor` (cell editor / executor), `notebook_import`
//!   (import .ipynb uploads).
//!
//! # File
//! `crates/axonml-dashboard/src/pages/training/mod.rs`
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
// Submodules
// =============================================================================

pub mod detail;
pub mod list;
pub mod new;
pub mod notebook_editor;
pub mod notebook_import;
pub mod notebook_list;

// =============================================================================
// Re-Exports
// =============================================================================

pub use detail::*;
pub use list::*;
pub use new::*;
pub use notebook_editor::*;
pub use notebook_import::*;
pub use notebook_list::*;
