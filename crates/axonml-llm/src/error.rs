//! LLM Error Types — Failure Modes for Transformer Operations
//!
//! Defines `LLMError`, the `thiserror`-derived error enum for the `axonml-llm`
//! crate. Variants cover invalid configuration, shape mismatch (expected vs
//! actual string pair), invalid input, generation failures, model loading
//! errors, core framework errors (`#[from] axonml_core::Error`), IO errors
//! (stored as `String` to avoid duplicate `From` impls), network/parse errors,
//! missing-model and missing-weight errors, unsupported-format errors,
//! tensor-layer errors, and a hub error variant. A manual `From<HubError>`
//! impl stringifies hub errors into `LLMError::HubError`. The `LLMResult<T>`
//! alias is the corresponding `Result` type used across the crate.
//!
//! # File
//! `crates/axonml-llm/src/error.rs`
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

    /// IO error (string to avoid duplicate From impl)
    #[error("IO error: {0}")]
    IoError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Weight not found
    #[error("Weight not found: {0}")]
    WeightNotFound(String),

    /// Unsupported format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Tensor error
    #[error("Tensor error: {0}")]
    TensorError(String),

    /// Hub error
    #[error("Hub error: {0}")]
    HubError(String),
}

// =============================================================================
// Conversions
// =============================================================================

impl From<crate::hub::HubError> for LLMError {
    fn from(e: crate::hub::HubError) -> Self {
        LLMError::HubError(e.to_string())
    }
}
