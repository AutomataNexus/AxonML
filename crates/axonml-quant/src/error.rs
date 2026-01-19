//! Quantization Error Types
//!
//! Error types for quantization operations.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

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
