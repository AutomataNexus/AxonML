//! Axonml Fusion - Kernel Fusion Library
//!
//! # File
//! `crates/axonml-fusion/src/lib.rs`
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

pub mod elementwise;
pub mod error;
pub mod linear;
pub mod optimizer;
pub mod patterns;

pub use elementwise::{fuse_elementwise, FusedElementwise};
pub use error::{FusionError, FusionResult};
pub use linear::{fuse_matmul_bias_relu, FusedLinear};
pub use optimizer::{optimize_graph, FusionOptimizer};
pub use patterns::{detect_patterns, FusionPattern};

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
