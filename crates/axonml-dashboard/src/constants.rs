//! Shared Constants for AxonML Dashboard
//!
//! Centralizes storage keys and other shared constants
//! to prevent duplication across modules.
//!
//! SECURITY: Access tokens are stored in sessionStorage (cleared on browser
//! close) to limit XSS exposure. Refresh tokens remain in localStorage
//! for session persistence across tabs. User data is in sessionStorage.

/// sessionStorage key for the access token (short-lived, per-tab).
pub const ACCESS_TOKEN_KEY: &str = "access_token";
/// localStorage key for the refresh token (persistent across tabs/close).
pub const REFRESH_TOKEN_KEY: &str = "refresh_token";
/// sessionStorage key for the serialized User object.
pub const USER_KEY: &str = "user";

/// Minimum password length for client-side validation.
pub const MIN_PASSWORD_LENGTH: usize = 8;
/// Maximum password length.
pub const MAX_PASSWORD_LENGTH: usize = 128;
