//! Dashboard Pages — Module Root And Re-Exports
//!
//! Top-level module aggregator for the Leptos/WASM dashboard pages. Declares
//! the page sub-modules (`admin`, `dashboard`, `datasets`, `hub`, `inference`,
//! `landing`, `models`, `settings`, `system`, `training`) and re-exports the
//! `dashboard` and `landing` namespaces so their components are reachable
//! directly from `crate::pages`.
//!
//! # File
//! `crates/axonml-dashboard/src/pages/mod.rs`
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

pub mod admin;
pub mod dashboard;
pub mod datasets;
pub mod hub;
pub mod inference;
pub mod landing;
pub mod models;
pub mod settings;
pub mod system;
pub mod training;

// =============================================================================
// Re-Exports
// =============================================================================

pub use dashboard::*;
pub use landing::*;
