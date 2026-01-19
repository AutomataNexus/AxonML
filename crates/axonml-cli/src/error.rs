//! Error - CLI Error Types
//!
//! Defines error types for CLI operations.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

/// CLI-specific errors
#[derive(Error, Debug)]
pub enum CliError {
    /// Configuration file error
    #[error("Configuration error: {0}")]
    Config(String),

    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Model loading error
    #[error("Model error: {0}")]
    Model(String),

    /// Training error
    #[error("Training error: {0}")]
    Training(String),

    /// Data/dataset error
    #[error("Data error: {0}")]
    Data(String),

    /// Conversion error
    #[error("Conversion error: {0}")]
    Conversion(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Project already exists
    #[error("Project already exists at {0}")]
    ProjectExists(String),

    /// Checkpoint not found
    #[error("Checkpoint not found: {0}")]
    CheckpointNotFound(String),

    /// Unsupported format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// GPU error
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Generic error with message
    #[error("{0}")]
    Other(String),
}

/// Result type for CLI operations
pub type CliResult<T> = Result<T, CliError>;

// =============================================================================
// Error Conversion
// =============================================================================

impl From<toml::de::Error> for CliError {
    fn from(e: toml::de::Error) -> Self {
        CliError::Config(e.to_string())
    }
}

impl From<serde_json::Error> for CliError {
    fn from(e: serde_json::Error) -> Self {
        CliError::Serialization(e.to_string())
    }
}

impl From<anyhow::Error> for CliError {
    fn from(e: anyhow::Error) -> Self {
        CliError::Other(e.to_string())
    }
}
