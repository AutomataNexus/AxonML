//! Quantization Error Types — Block, Shape, and Calibration Failures
//!
//! Defines `QuantError`, the `thiserror`-derived error enum for the
//! `axonml-quant` crate. Variants cover invalid block sizes, unknown
//! quantization-type strings, shape mismatch (expected/actual `Vec<usize>`),
//! data-length mismatch (expected/actual byte counts), calibration errors,
//! numerical overflow during quantization, invalid encoded data, and
//! tensor-conversion errors. `QuantResult<T>` is the corresponding `Result`
//! alias used throughout the quantization pipeline.
//!
//! # File
//! `crates/axonml-quant/src/error.rs`
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

/// Result type for quantization operations.
pub type QuantResult<T> = Result<T, QuantError>;

/// Errors that can occur during quantization.
#[derive(Error, Debug)]
pub enum QuantError {
    /// Invalid block size.
    #[error("Invalid block size: {0}")]
    InvalidBlockSize(usize),

    /// Invalid quantization type.
    #[error("Invalid quantization type: {0}")]
    InvalidQuantType(String),

    /// Shape mismatch during quantization.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// Data length mismatch.
    #[error("Data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Calibration error.
    #[error("Calibration error: {0}")]
    CalibrationError(String),

    /// Numerical overflow during quantization.
    #[error("Numerical overflow during quantization")]
    Overflow,

    /// Invalid quantized data.
    #[error("Invalid quantized data: {0}")]
    InvalidData(String),

    /// Tensor conversion error.
    #[error("Tensor conversion error: {0}")]
    TensorConversion(String),
}
