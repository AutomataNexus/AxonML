//! Error types for the profiling module.

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
