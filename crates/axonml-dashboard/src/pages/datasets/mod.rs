//! Datasets Pages — Module Root For The Datasets Section
//!
//! Aggregates the dataset management sub-pages of the dashboard. Declares the
//! `analyze`, `builtin`, `kaggle`, `list`, and `upload` sub-modules and glob
//! re-exports their public page components so they can be referenced as
//! `crate::pages::datasets::<PageName>`.
//!
//! # File
//! `crates/axonml-dashboard/src/pages/datasets/mod.rs`
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
// Sub-Module Declarations
// =============================================================================

pub mod analyze;
pub mod builtin;
pub mod kaggle;
pub mod list;
pub mod upload;

// =============================================================================
// Re-Exports
// =============================================================================

pub use analyze::*;
pub use builtin::*;
pub use kaggle::*;
pub use list::*;
pub use upload::*;
