//! System Pages — Module Root for System Status Views
//!
//! Module root for the dashboard's system-status subsection. Currently
//! re-exports only the `overview` submodule, which contains the
//! `SystemOverviewPage` component showing storage usage, worker status,
//! and service health metrics.
//!
//! # File
//! `crates/axonml-dashboard/src/pages/system/mod.rs`
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
// Submodules and Re-Exports
// =============================================================================

pub mod overview;

pub use overview::*;
