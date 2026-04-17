//! Model quantization for AxonML — GGUF formats + BitNet I2_S ternary.
//!
//! `types` (QuantType enum, block structs for Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/F16),
//! `quantize` (tensor/model quantization with RMSE error analysis),
//! `dequantize` (block/tensor reconstruction to f32), `bitnet` (I2_S 1.58-bit
//! ternary — 128-weight blocks, fused add-only matmul, int8 activation
//! quantizer, AVX-VNNI dispatch scaffolded), `calibration` (MinMax/Percentile/
//! MeanStd/Entropy methods), `inference` (QuantizedLinear drop-in layer,
//! QuantizedModel wrapper), `error` (QuantError/QuantResult).
//!
//! # File
//! `crates/axonml-quant/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
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

pub mod bitnet;
pub mod calibration;
pub mod dequantize;
pub mod error;
pub mod inference;
pub mod quantize;
pub mod types;

pub use calibration::{CalibrationData, calibrate};
pub use dequantize::{dequantize_block, dequantize_tensor};
pub use error::{QuantError, QuantResult};
pub use inference::{QuantizedLinear, QuantizedModel, deserialize_quantized, serialize_quantized};
pub use quantize::{quantize_model, quantize_tensor};
pub use types::{QuantType, QuantizedBlock, QuantizedTensor};

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
