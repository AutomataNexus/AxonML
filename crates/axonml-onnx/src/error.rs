//! ONNX Error Types
//!
//! Error types for ONNX import/export operations.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use thiserror::Error;

/// Result type for ONNX operations.
pub type OnnxResult<T> = Result<T, OnnxError>;

/// Errors that can occur during ONNX operations.
#[derive(Error, Debug)]
pub enum OnnxError {
    /// Failed to read ONNX file.
    #[error("Failed to read ONNX file: {0}")]
    FileRead(#[from] std::io::Error),

    /// Failed to parse ONNX protobuf.
    #[error("Failed to parse ONNX protobuf: {0}")]
    ProtobufParse(String),

    /// Unsupported ONNX opset version.
    #[error("Unsupported ONNX opset version: {0}")]
    UnsupportedOpset(i64),

    /// Unsupported ONNX operator.
    #[error("Unsupported ONNX operator: {0}")]
    UnsupportedOperator(String),

    /// Invalid tensor shape.
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    /// Invalid tensor data type.
    #[error("Invalid tensor data type: {0}")]
    InvalidDataType(i32),

    /// Missing required attribute.
    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),

    /// Invalid attribute value.
    #[error("Invalid attribute value for {0}: {1}")]
    InvalidAttribute(String, String),

    /// Missing initializer (weight tensor).
    #[error("Missing initializer: {0}")]
    MissingInitializer(String),

    /// Graph validation error.
    #[error("Graph validation error: {0}")]
    GraphValidation(String),

    /// Tensor conversion error.
    #[error("Tensor conversion error: {0}")]
    TensorConversion(String),

    /// Model export error.
    #[error("Model export error: {0}")]
    Export(String),

    /// Axonml core error.
    #[error("Axonml error: {0}")]
    Axonml(String),
}

impl From<prost::DecodeError> for OnnxError {
    fn from(err: prost::DecodeError) -> Self {
        OnnxError::ProtobufParse(err.to_string())
    }
}
