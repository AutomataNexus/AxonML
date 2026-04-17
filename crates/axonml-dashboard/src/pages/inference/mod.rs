//! Inference Pages — Module Root For Inference Dashboard Sub-Pages
//!
//! Module aggregator for the Inference section. Declares the `endpoints`,
//! `metrics`, and `overview` sub-modules and glob re-exports their public
//! page components so they are reachable as `crate::pages::inference::*`.
//!
//! # File
//! `crates/axonml-dashboard/src/pages/inference/mod.rs`
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

pub mod endpoints;
pub mod metrics;
pub mod overview;

// =============================================================================
// Re-Exports
// =============================================================================

pub use endpoints::*;
pub use metrics::*;
pub use overview::*;
