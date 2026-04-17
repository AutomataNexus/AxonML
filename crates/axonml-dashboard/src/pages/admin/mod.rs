//! Admin Pages Module — Declarations And Re-Exports
//!
//! Module entry point for the admin section of the dashboard. Declares the
//! private `system` and `users` sub-modules and re-exports their two page
//! components: `SystemStatsPage` (platform health and resource stats) and
//! `UserManagementPage` (CRUD over user accounts).
//!
//! # File
//! `crates/axonml-dashboard/src/pages/admin/mod.rs`
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

mod system;
mod users;

// =============================================================================
// Re-Exports
// =============================================================================

pub use system::SystemStatsPage;
pub use users::UserManagementPage;
