//! Authentication Module — Sub-Module Aggregator
//!
//! Aggregates the dashboard's authentication surface into a single `auth`
//! namespace. Declares and re-exports the `login` (login + registration
//! pages), `mfa` (multi-factor challenge views), `mfa_setup` (TOTP, WebAuthn,
//! and recovery-code enrollment pages), and `session` (session initializer +
//! `ProtectedRoute` guard) sub-modules so callers elsewhere in the crate can
//! `use auth::*` without reaching into the module paths.
//!
//! # File
//! `crates/axonml-dashboard/src/auth/mod.rs`
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

pub mod login;
pub mod mfa;
pub mod mfa_setup;
pub mod session;

// =============================================================================
// Re-Exports
// =============================================================================

pub use login::*;
pub use mfa::*;
pub use mfa_setup::*;
pub use session::*;
