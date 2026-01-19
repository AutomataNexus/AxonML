//! Axonml Quant - Model Quantization Library
//!
//! Provides quantization support for reducing model size and improving
//! inference performance. Supports multiple quantization formats:
//!
//! - **Q8_0**: 8-bit quantization (block size 32)
//! - **Q4_0**: 4-bit quantization (block size 32)
//! - **Q4_1**: 4-bit quantization with min/max (block size 32)
//! - **F16**: Half-precision floating point
//!
//! # Example
//! ```ignore
//! use axonml_quant::{quantize_tensor, QuantType};
//!
//! let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
//! let quantized = quantize_tensor(&tensor, QuantType::Q8_0)?;
//! let dequantized = dequantize_tensor(&quantized)?;
//! ```
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]

pub mod error;
pub mod types;
pub mod quantize;
pub mod dequantize;
pub mod calibration;

pub use error::{QuantError, QuantResult};
pub use types::{QuantType, QuantizedTensor, QuantizedBlock};
pub use quantize::{quantize_tensor, quantize_model};
pub use dequantize::{dequantize_tensor, dequantize_block};
pub use calibration::{CalibrationData, calibrate};

// =============================================================================
// Constants
// =============================================================================

/// Default block size for quantization.
pub const DEFAULT_BLOCK_SIZE: usize = 32;

/// Maximum block size supported.
pub const MAX_BLOCK_SIZE: usize = 256;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(DEFAULT_BLOCK_SIZE > 0);
        assert!(MAX_BLOCK_SIZE >= DEFAULT_BLOCK_SIZE);
    }
}
