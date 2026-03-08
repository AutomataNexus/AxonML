//! Axonml Quant - Model Quantization Library
//!
//! # File
//! `crates/axonml-quant/src/lib.rs`
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

pub mod calibration;
pub mod dequantize;
pub mod error;
pub mod quantize;
pub mod types;

pub use calibration::{calibrate, CalibrationData};
pub use dequantize::{dequantize_block, dequantize_tensor};
pub use error::{QuantError, QuantResult};
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
