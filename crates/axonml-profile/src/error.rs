//! Profile Error Types — Failures for Compute and Memory Profiling
//!
//! Defines `ProfileError`, the `thiserror`-derived error enum used throughout
//! `axonml-profile`. Variants cover a missing-operation lookup, invalid
//! profiler state, `std::io::Error` (via `#[from]`) for report export
//! failures, serialization errors, and timer errors. `ProfileResult<T>` is
//! the corresponding `Result` alias exported to the rest of the crate.
//!
//! # File
//! `crates/axonml-profile/src/error.rs`
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
// Error Types
// =============================================================================

use thiserror::Error;

/// Result type for profiling operations.
pub type ProfileResult<T> = Result<T, ProfileError>;

/// Error types for profiling operations.
#[derive(Error, Debug)]
pub enum ProfileError {
    /// Operation not found in profiler
    #[error("Operation not found: {0}")]
    OperationNotFound(String),

    /// Invalid profiler state
    #[error("Invalid profiler state: {0}")]
    InvalidState(String),

    /// IO error during report export
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Timer error
    #[error("Timer error: {0}")]
    TimerError(String),
}
