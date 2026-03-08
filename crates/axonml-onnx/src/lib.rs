//! Axonml ONNX - ONNX Import/Export for ML Models
//!
//! # File
//! `crates/axonml-onnx/src/lib.rs`
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

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod error;
pub mod export;
pub mod model;
pub mod operators;
pub mod parser;
pub mod proto;

pub use error::{OnnxError, OnnxResult};
pub use export::export_onnx;
pub use model::OnnxModel;
pub use parser::{import_onnx, import_onnx_bytes};

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
