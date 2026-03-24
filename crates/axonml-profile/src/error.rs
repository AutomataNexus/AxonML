//! Error types for the profiling module.
//!
//! # File
//! `crates/axonml-profile/src/error.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

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
