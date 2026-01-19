//! Axonml Fusion - Kernel Fusion Library
//!
//! Provides kernel fusion support for combining multiple operations into
//! single optimized kernels. Common fusion patterns include:
//!
//! - **MatMul + Bias + Activation**: Fused dense layer
//! - **Conv + BatchNorm + ReLU**: Fused convolution block
//! - **Elementwise chains**: Multiple elementwise ops in one pass
//! - **Reduction + Transform**: Softmax, LayerNorm patterns
//!
//! # Example
//! ```ignore
//! use axonml_fusion::{FusedOp, fuse_matmul_bias_relu};
//!
//! let fused = fuse_matmul_bias_relu(&weight, &bias);
//! let output = fused.execute(&input);
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
pub mod patterns;
pub mod elementwise;
pub mod linear;
pub mod optimizer;

pub use error::{FusionError, FusionResult};
pub use patterns::{FusionPattern, detect_patterns};
pub use elementwise::{FusedElementwise, fuse_elementwise};
pub use linear::{FusedLinear, fuse_matmul_bias_relu};
pub use optimizer::{FusionOptimizer, optimize_graph};

// =============================================================================
// Fused Operation Trait
// =============================================================================

use axonml_tensor::Tensor;
use std::fmt::Debug;

/// Trait for fused operations.
pub trait FusedOp: Debug + Send + Sync {
    /// Executes the fused operation.
    fn execute(&self, inputs: &[&Tensor<f32>]) -> FusionResult<Tensor<f32>>;

    /// Returns the name of the fused operation.
    fn name(&self) -> &str;

    /// Returns the number of operations fused.
    fn num_ops(&self) -> usize;

    /// Returns estimated speedup from fusion.
    fn estimated_speedup(&self) -> f32 {
        // Default: 1.0 + 0.2 per additional op fused
        1.0 + 0.2 * (self.num_ops() - 1) as f32
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        assert!(true);
    }
}
