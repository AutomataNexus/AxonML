//! Models Pages — Module Root for Model Management Views
//!
//! Module root that re-exports the model management dashboard pages:
//! `list` (model registry listing), `detail` (individual model inspection),
//! and `upload` (model upload form). Each submodule defines Leptos components
//! rendered by the dashboard router under the `/models` route tree.
//!
//! # File
//! `crates/axonml-dashboard/src/pages/models/mod.rs`
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
pub mod upload;

// =============================================================================
// Re-Exports
// =============================================================================

pub use detail::*;
pub use list::*;
pub use upload::*;
