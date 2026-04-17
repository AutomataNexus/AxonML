//! Shared Constants for AxonML Dashboard — Storage Keys and Validation Bounds
//!
//! Centralizes browser storage keys and password validation limits used across
//! the dashboard modules. Access tokens are stored in sessionStorage (cleared
//! on browser close) to limit XSS exposure, while refresh tokens persist in
//! localStorage for session persistence across tabs. Also exposes the minimum
//! and maximum password length constants consumed by client-side form
//! validation.
//!
//! # File
//! `crates/axonml-dashboard/src/constants.rs`
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
//!
//! SECURITY: Access tokens are stored in sessionStorage (cleared on browser
//! close) to limit XSS exposure. Refresh tokens remain in localStorage
//! for session persistence across tabs. User data is in sessionStorage.

// =============================================================================
// Storage Keys
// =============================================================================

/// sessionStorage key for the access token (short-lived, per-tab).
pub const ACCESS_TOKEN_KEY: &str = "access_token";
/// localStorage key for the refresh token (persistent across tabs/close).
pub const REFRESH_TOKEN_KEY: &str = "refresh_token";
/// sessionStorage key for the serialized User object.
pub const USER_KEY: &str = "user";

// =============================================================================
// Password Validation Bounds
// =============================================================================

/// Minimum password length for client-side validation.
pub const MIN_PASSWORD_LENGTH: usize = 8;
/// Maximum password length.
pub const MAX_PASSWORD_LENGTH: usize = 128;
