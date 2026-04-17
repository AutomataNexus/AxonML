//! Fusion Error Types — Kernel Fusion Failure Modes
//!
//! Defines `FusionError`, the `thiserror`-derived error enum used throughout
//! `axonml-fusion`. Variants cover input shape mismatches (expected vs actual
//! `Vec<usize>`), unfusable patterns, invalid fusion configuration, execution
//! errors, and tensor-conversion errors. `FusionResult<T>` is the
//! corresponding `Result` alias used by public APIs in the crate.
//!
//! # File
//! `crates/axonml-fusion/src/error.rs`
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
