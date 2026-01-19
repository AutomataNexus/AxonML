//! JIT Error Types
//!
//! Error handling for JIT compilation operations.

use std::fmt;

/// Result type for JIT operations.
pub type JitResult<T> = Result<T, JitError>;

/// JIT compilation errors.
#[derive(Debug, Clone)]
pub enum JitError {
    /// Invalid graph structure.
    InvalidGraph(String),
    /// Type mismatch in operations.
    TypeMismatch {
        /// Expected type.
        expected: String,
        /// Actual type.
        found: String,
    },
    /// Shape mismatch in operations.
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        found: Vec<usize>,
    },
    /// Unsupported operation for JIT.
    UnsupportedOp(String),
    /// Code generation failed.
    CodegenError(String),
    /// Runtime execution error.
    RuntimeError(String),
    /// Input not found.
    InputNotFound(String),
    /// Output not found.
    OutputNotFound(String),
    /// Compilation failed.
    CompilationFailed(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(msg) => write!(f, "Invalid graph: {msg}"),
            Self::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {expected}, found {found}")
            }
            Self::ShapeMismatch { expected, found } => {
                write!(f, "Shape mismatch: expected {expected:?}, found {found:?}")
            }
            Self::UnsupportedOp(op) => write!(f, "Unsupported operation: {op}"),
            Self::CodegenError(msg) => write!(f, "Code generation error: {msg}"),
            Self::RuntimeError(msg) => write!(f, "Runtime error: {msg}"),
            Self::InputNotFound(name) => write!(f, "Input not found: {name}"),
            Self::OutputNotFound(name) => write!(f, "Output not found: {name}"),
            Self::CompilationFailed(msg) => write!(f, "Compilation failed: {msg}"),
        }
    }
}

impl std::error::Error for JitError {}

impl From<String> for JitError {
    fn from(msg: String) -> Self {
        Self::RuntimeError(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = JitError::TypeMismatch {
            expected: "f32".to_string(),
            found: "i32".to_string(),
        };
        assert!(err.to_string().contains("Type mismatch"));
    }

    #[test]
    fn test_shape_mismatch() {
        let err = JitError::ShapeMismatch {
            expected: vec![2, 3],
            found: vec![3, 2],
        };
        assert!(err.to_string().contains("Shape mismatch"));
    }
}
