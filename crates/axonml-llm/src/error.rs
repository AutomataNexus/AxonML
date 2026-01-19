//! Error types for the LLM module.

use thiserror::Error;

/// Result type for LLM operations.
pub type LLMResult<T> = Result<T, LLMError>;

/// Error types for LLM operations.
#[derive(Error, Debug)]
pub enum LLMError {
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Expected shape
        expected: String,
        /// Actual shape
        actual: String,
    },

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Generation error
    #[error("Generation error: {0}")]
    GenerationError(String),

    /// Model loading error
    #[error("Model loading error: {0}")]
    LoadError(String),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] axonml_core::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
