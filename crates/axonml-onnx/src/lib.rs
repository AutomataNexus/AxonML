//! Axonml ONNX - ONNX Import/Export for ML Models
//!
//! This crate provides support for importing and exporting models in the
//! ONNX (Open Neural Network Exchange) format, enabling interoperability
//! with PyTorch, TensorFlow, and other ML frameworks.
//!
//! # Features
//! - Import ONNX models and convert to Axonml modules
//! - Export Axonml models to ONNX format
//! - Support for common operators (Conv, MatMul, ReLU, etc.)
//! - Weight loading from ONNX initializers
//!
//! # Example
//! ```ignore
//! use axonml_onnx::{OnnxModel, import_onnx};
//!
//! // Import an ONNX model
//! let model = import_onnx("model.onnx")?;
//!
//! // Run inference
//! let output = model.forward(&input);
//!
//! // Export back to ONNX
//! model.export_onnx("model_out.onnx")?;
//! ```
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod error;
pub mod model;
pub mod operators;
pub mod parser;
pub mod proto;
pub mod export;

pub use error::{OnnxError, OnnxResult};
pub use model::OnnxModel;
pub use parser::{import_onnx, import_onnx_bytes};
pub use export::export_onnx;

// =============================================================================
// Re-exports for convenience
// =============================================================================

/// ONNX opset version supported by this crate.
pub const SUPPORTED_OPSET_VERSION: i64 = 17;

/// ONNX IR version.
pub const ONNX_IR_VERSION: i64 = 8;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(SUPPORTED_OPSET_VERSION > 0);
        assert!(ONNX_IR_VERSION > 0);
    }
}
