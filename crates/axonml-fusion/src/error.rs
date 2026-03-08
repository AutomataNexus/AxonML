//! Fusion Error Types
//!
//! # File
//! `crates/axonml-fusion/src/error.rs`
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

/// Result type for fusion operations.
pub type FusionResult<T> = Result<T, FusionError>;

/// Errors that can occur during kernel fusion.
#[derive(Error, Debug)]
pub enum FusionError {
    /// Invalid input shape.
    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// Pattern not fusable.
    #[error("Pattern not fusable: {0}")]
    NotFusable(String),

    /// Invalid fusion configuration.
    #[error("Invalid fusion configuration: {0}")]
    InvalidConfig(String),

    /// Execution error.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Tensor conversion error.
    #[error("Tensor conversion error: {0}")]
    TensorError(String),
}
