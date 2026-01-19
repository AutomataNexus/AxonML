//! Error Types - Axonml Core Error Handling
//!
//! Provides comprehensive error types for all operations within the Axonml
//! framework, including device errors, memory allocation failures, and
//! type mismatches.
//!
//! # Key Features
//! - Unified error type for all Axonml operations
//! - Detailed error context for debugging
//! - Integration with `std::error::Error`
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use thiserror::Error;

use crate::device::Device;
use crate::dtype::DType;

// =============================================================================
// Error Types
// =============================================================================

/// The main error type for Axonml operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum Error {
    /// Shape mismatch between tensors.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// The expected shape.
        expected: Vec<usize>,
        /// The actual shape.
        actual: Vec<usize>,
    },

    /// Data type mismatch between tensors.
    #[error("DType mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch {
        /// The expected data type.
        expected: DType,
        /// The actual data type.
        actual: DType,
    },

    /// Device mismatch between tensors.
    #[error("Device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        /// The expected device.
        expected: Device,
        /// The actual device.
        actual: Device,
    },

    /// Invalid dimension index.
    #[error("Invalid dimension: index {index} for tensor with {ndim} dimensions")]
    InvalidDimension {
        /// The invalid dimension index.
        index: i64,
        /// Number of dimensions in the tensor.
        ndim: usize,
    },

    /// Index out of bounds.
    #[error("Index out of bounds: index {index} for dimension of size {size}")]
    IndexOutOfBounds {
        /// The invalid index.
        index: usize,
        /// The size of the dimension.
        size: usize,
    },

    /// Memory allocation failed.
    #[error("Memory allocation failed: requested {size} bytes on {device:?}")]
    AllocationFailed {
        /// The requested size in bytes.
        size: usize,
        /// The device on which allocation failed.
        device: Device,
    },

    /// Device not available.
    #[error("Device not available: {device:?}")]
    DeviceNotAvailable {
        /// The unavailable device.
        device: Device,
    },

    /// Invalid operation for the given tensor.
    #[error("Invalid operation: {message}")]
    InvalidOperation {
        /// Description of why the operation is invalid.
        message: String,
    },

    /// Broadcasting failed between shapes.
    #[error("Cannot broadcast shapes {shape1:?} and {shape2:?}")]
    BroadcastError {
        /// The first shape.
        shape1: Vec<usize>,
        /// The second shape.
        shape2: Vec<usize>,
    },

    /// Empty tensor error.
    #[error("Operation not supported on empty tensor")]
    EmptyTensor,

    /// Contiguous tensor required.
    #[error("Operation requires contiguous tensor")]
    NotContiguous,

    /// Gradient computation error.
    #[error("Gradient error: {message}")]
    GradientError {
        /// Description of the gradient error.
        message: String,
    },

    /// Serialization/deserialization error.
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Description of the serialization error.
        message: String,
    },

    /// Internal error (should not happen).
    #[error("Internal error: {message}")]
    InternalError {
        /// Description of the internal error.
        message: String,
    },
}

// =============================================================================
// Result Type
// =============================================================================

/// A specialized Result type for Axonml operations.
pub type Result<T> = core::result::Result<T, Error>;

// =============================================================================
// Helper Functions
// =============================================================================

impl Error {
    /// Creates a new shape mismatch error.
    #[must_use]
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        Self::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Creates a new invalid operation error.
    #[must_use]
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::InvalidOperation {
            message: message.into(),
        }
    }

    /// Creates a new internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::shape_mismatch(&[2, 3], &[2, 4]);
        assert!(err.to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = Error::EmptyTensor;
        let err2 = Error::EmptyTensor;
        assert_eq!(err1, err2);
    }
}
