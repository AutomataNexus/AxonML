//! Dashboard Utilities — Sub-Module Aggregator
//!
//! Declares the dashboard's cross-cutting utility modules. `js_helpers`
//! hosts thin wrappers around `web_sys`/`js_sys` that are reused by many
//! components; `webauthn` provides the high-level credential-creation and
//! assertion helpers used by the MFA enrollment and challenge pages.
//!
//! # File
//! `crates/axonml-dashboard/src/utils/mod.rs`
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

pub mod js_helpers;
pub mod webauthn;
