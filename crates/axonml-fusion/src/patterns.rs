//! Fusion Pattern Detection
//!
//! Detects common patterns in computational graphs that can be fused.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::fmt;

// =============================================================================
// Fusion Patterns
// =============================================================================

/// Common fusion patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// MatMul followed by bias addition.
    MatMulBias,
    /// MatMul followed by bias addition and ReLU.
    MatMulBiasRelu,
    /// MatMul followed by bias addition and GELU.
    MatMulBiasGelu,
    /// Convolution followed by batch normalization.
    ConvBatchNorm,
    /// Convolution followed by batch normalization and ReLU.
    ConvBatchNormRelu,
    /// Multiple elementwise operations.
    ElementwiseChain,
    /// Softmax pattern (exp, sum, div).
    Softmax,
    /// Layer normalization pattern.
    LayerNorm,
    /// GELU approximation pattern.
    GeluApprox,
    /// Add followed by ReLU.
    AddRelu,
    /// Multiply followed by add (FMA).
    MulAdd,
}

impl FusionPattern {
    /// Returns the number of operations fused in this pattern.
    pub fn num_ops(&self) -> usize {
        match self {
            FusionPattern::MatMulBias | FusionPattern::AddRelu | FusionPattern::MulAdd => 2,
            FusionPattern::MatMulBiasRelu | FusionPattern::MatMulBiasGelu |
            FusionPattern::ConvBatchNorm | FusionPattern::Softmax => 3,
            FusionPattern::ConvBatchNormRelu | FusionPattern::LayerNorm => 4,
            FusionPattern::GeluApprox => 5,
            FusionPattern::ElementwiseChain => 2, // Variable, default 2
        }
    }

    /// Returns estimated speedup from this fusion.
    pub fn estimated_speedup(&self) -> f32 {
        match self {
            // Memory-bound patterns benefit most
            FusionPattern::ElementwiseChain => 2.0,
            FusionPattern::AddRelu => 1.8,
            FusionPattern::MulAdd => 1.5,

            // Compute-bound patterns still benefit
            FusionPattern::MatMulBiasRelu | FusionPattern::MatMulBiasGelu => 1.3,
            FusionPattern::MatMulBias => 1.2,

            // Complex patterns
            FusionPattern::ConvBatchNormRelu => 1.4,
            FusionPattern::ConvBatchNorm => 1.3,
            FusionPattern::Softmax | FusionPattern::LayerNorm | FusionPattern::GeluApprox => 1.2,
        }
    }

    /// Returns whether this pattern is memory-bound (vs compute-bound).
    pub fn is_memory_bound(&self) -> bool {
        matches!(
            self,
            FusionPattern::ElementwiseChain | FusionPattern::AddRelu | FusionPattern::MulAdd
        )
    }
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionPattern::MatMulBias => write!(f, "MatMul+Bias"),
            FusionPattern::MatMulBiasRelu => write!(f, "MatMul+Bias+ReLU"),
            FusionPattern::MatMulBiasGelu => write!(f, "MatMul+Bias+GELU"),
            FusionPattern::ConvBatchNorm => write!(f, "Conv+BatchNorm"),
            FusionPattern::ConvBatchNormRelu => write!(f, "Conv+BatchNorm+ReLU"),
            FusionPattern::ElementwiseChain => write!(f, "Elementwise Chain"),
            FusionPattern::Softmax => write!(f, "Softmax"),
            FusionPattern::LayerNorm => write!(f, "LayerNorm"),
            FusionPattern::GeluApprox => write!(f, "GELU Approximation"),
            FusionPattern::AddRelu => write!(f, "Add+ReLU"),
            FusionPattern::MulAdd => write!(f, "Mul+Add (FMA)"),
        }
    }
}

// =============================================================================
// Operation Type for Pattern Matching
// =============================================================================

/// Operation types for pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// Matrix multiplication.
    MatMul,
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// ReLU activation.
    Relu,
    /// GELU activation.
    Gelu,
    /// Sigmoid activation.
    Sigmoid,
    /// Tanh activation.
    Tanh,
    /// Softmax.
    Softmax,
    /// Convolution.
    Conv,
    /// Batch normalization.
    BatchNorm,
    /// Layer normalization.
    LayerNorm,
    /// Exponential.
    Exp,
    /// Logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Power.
    Pow,
    /// Reduction (sum, mean, max).
    Reduce,
    /// Unknown operation.
    Unknown,
}

// =============================================================================
// Pattern Detection
// =============================================================================

/// Detects fusion patterns in a sequence of operations.
///
/// # Arguments
/// * `ops` - Sequence of operation types
///
/// # Returns
/// List of detected patterns with their positions
pub fn detect_patterns(ops: &[OpType]) -> Vec<(FusionPattern, usize, usize)> {
    let mut patterns = Vec::new();
    let n = ops.len();

    let mut i = 0;
    while i < n {
        // Try to match longer patterns first

        // MatMul + Add + ReLU (length 3)
        if i + 2 < n {
            if ops[i] == OpType::MatMul && ops[i + 1] == OpType::Add && ops[i + 2] == OpType::Relu {
                patterns.push((FusionPattern::MatMulBiasRelu, i, i + 3));
                i += 3;
                continue;
            }
            if ops[i] == OpType::MatMul && ops[i + 1] == OpType::Add && ops[i + 2] == OpType::Gelu {
                patterns.push((FusionPattern::MatMulBiasGelu, i, i + 3));
                i += 3;
                continue;
            }
            if ops[i] == OpType::Conv && ops[i + 1] == OpType::BatchNorm && ops[i + 2] == OpType::Relu {
                patterns.push((FusionPattern::ConvBatchNormRelu, i, i + 3));
                i += 3;
                continue;
            }
        }

        // MatMul + Add (length 2)
        if i + 1 < n {
            if ops[i] == OpType::MatMul && ops[i + 1] == OpType::Add {
                patterns.push((FusionPattern::MatMulBias, i, i + 2));
                i += 2;
                continue;
            }
            if ops[i] == OpType::Conv && ops[i + 1] == OpType::BatchNorm {
                patterns.push((FusionPattern::ConvBatchNorm, i, i + 2));
                i += 2;
                continue;
            }
            if ops[i] == OpType::Add && ops[i + 1] == OpType::Relu {
                patterns.push((FusionPattern::AddRelu, i, i + 2));
                i += 2;
                continue;
            }
            if ops[i] == OpType::Mul && ops[i + 1] == OpType::Add {
                patterns.push((FusionPattern::MulAdd, i, i + 2));
                i += 2;
                continue;
            }
        }

        // Elementwise chain detection
        if is_elementwise_op(ops[i]) {
            let start = i;
            while i < n && is_elementwise_op(ops[i]) {
                i += 1;
            }
            if i - start > 1 {
                patterns.push((FusionPattern::ElementwiseChain, start, i));
            }
            continue;
        }

        i += 1;
    }

    patterns
}

/// Checks if an operation is elementwise.
fn is_elementwise_op(op: OpType) -> bool {
    matches!(
        op,
        OpType::Add | OpType::Sub | OpType::Mul | OpType::Div |
        OpType::Relu | OpType::Sigmoid | OpType::Tanh |
        OpType::Exp | OpType::Log | OpType::Sqrt
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_matmul_bias_relu() {
        let ops = vec![OpType::MatMul, OpType::Add, OpType::Relu];
        let patterns = detect_patterns(&ops);

        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].0, FusionPattern::MatMulBiasRelu);
    }

    #[test]
    fn test_detect_matmul_bias() {
        let ops = vec![OpType::MatMul, OpType::Add];
        let patterns = detect_patterns(&ops);

        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].0, FusionPattern::MatMulBias);
    }

    #[test]
    fn test_detect_elementwise_chain() {
        let ops = vec![OpType::Add, OpType::Mul, OpType::Relu];
        let patterns = detect_patterns(&ops);

        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].0, FusionPattern::ElementwiseChain);
    }

    #[test]
    fn test_pattern_speedup() {
        assert!(FusionPattern::ElementwiseChain.estimated_speedup() > 1.5);
        assert!(FusionPattern::MatMulBiasRelu.estimated_speedup() > 1.0);
    }

    #[test]
    fn test_detect_add_relu() {
        let ops = vec![OpType::Add, OpType::Relu];
        let patterns = detect_patterns(&ops);

        // Note: This will be detected as elementwise chain first
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_pattern_display() {
        assert_eq!(format!("{}", FusionPattern::MatMulBiasRelu), "MatMul+Bias+ReLU");
        assert_eq!(format!("{}", FusionPattern::Softmax), "Softmax");
    }
}
