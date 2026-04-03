//! Axonml Fusion - Kernel Fusion Optimization
//!
//! # File
//! `crates/axonml-fusion/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 25, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]

// =============================================================================
// Modules
// =============================================================================

pub mod elementwise;
pub mod error;
pub mod linear;
pub mod optimizer;
pub mod patterns;

// =============================================================================
// Re-exports
// =============================================================================

pub use elementwise::{ElementwiseOp, FusedElementwise};
pub use error::{FusionError, FusionResult};
pub use linear::{Activation, FusedLinear};
pub use optimizer::{FusionConfig, FusionOptimizer, OptimizationStats, optimize_graph};
pub use patterns::{FusionPattern, OpType, detect_patterns};

use axonml_tensor::Tensor;

/// Trait for fused operations that combine multiple ops into one.
pub trait FusedOp: std::fmt::Debug + Send + Sync {
    /// Executes the fused operation on the given inputs.
    fn execute(&self, inputs: &[&Tensor<f32>]) -> FusionResult<Tensor<f32>>;

    /// Returns the name of this fused operation.
    fn name(&self) -> &str;

    /// Returns the number of original operations fused.
    fn num_ops(&self) -> usize;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use axonml_tensor::Tensor;

    #[test]
    fn test_detect_elementwise_chain() {
        use crate::patterns::{OpType, detect_patterns};

        let ops = vec![OpType::Add, OpType::Mul, OpType::Relu];
        let patterns = detect_patterns(&ops);
        assert!(!patterns.is_empty(), "Should detect patterns");
    }

    #[test]
    fn test_detect_add_relu() {
        use crate::patterns::{FusionPattern, OpType, detect_patterns};

        let ops = vec![OpType::Add, OpType::Relu];
        let patterns = detect_patterns(&ops);
        assert!(
            patterns
                .iter()
                .any(|(p, _, _)| *p == FusionPattern::AddRelu),
            "Should detect AddRelu, got: {:?}",
            patterns
        );
    }

    #[test]
    fn test_fused_elementwise() {
        use crate::elementwise::{ElementwiseOp, FusedElementwise};

        let chain = FusedElementwise::new(vec![ElementwiseOp::Relu, ElementwiseOp::Neg]);
        let input =
            Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[4]).expect("tensor creation failed");
        let output = chain.forward(&input).expect("forward failed");
        assert_eq!(output.to_vec(), vec![0.0, -2.0, 0.0, -4.0]);
    }

    #[test]
    fn test_fused_linear() {
        use crate::linear::{Activation, FusedLinear};

        let weight =
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("tensor creation failed");
        let bias = Tensor::from_vec(vec![0.1, 0.2], &[2]).expect("tensor creation failed");
        let input = Tensor::from_vec(vec![3.0, 4.0], &[1, 2]).expect("tensor creation failed");

        let fl = FusedLinear::new(weight, Some(bias), Activation::Relu).expect("new failed");
        let output = fl.forward(&input).expect("forward failed");
        let out = output.to_vec();
        assert!((out[0] - 3.1).abs() < 1e-4, "got {}", out[0]);
        assert!((out[1] - 4.2).abs() < 1e-4, "got {}", out[1]);
    }

    #[test]
    fn test_optimize_graph() {
        use crate::optimizer::optimize_graph;
        use crate::patterns::OpType;

        let ops = vec![OpType::MatMul, OpType::Add, OpType::Relu];
        let (patterns, stats) = optimize_graph(&ops, None).expect("optimize failed");
        assert!(!patterns.is_empty());
        assert!(stats.fusions_applied > 0);
    }
}
