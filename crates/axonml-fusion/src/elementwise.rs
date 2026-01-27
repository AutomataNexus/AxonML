//! Fused Elementwise Operations
//!
//! Combines multiple elementwise operations into a single fused kernel.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_tensor::Tensor;
use rayon::prelude::*;

use crate::error::{FusionError, FusionResult};
use crate::FusedOp;

// =============================================================================
// Elementwise Operation Type
// =============================================================================

/// Elementwise operation variants.
#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    /// Add constant.
    AddConst(f32),
    /// Multiply by constant.
    MulConst(f32),
    /// ReLU activation.
    Relu,
    /// Leaky ReLU with alpha.
    LeakyRelu(f32),
    /// Sigmoid activation.
    Sigmoid,
    /// Tanh activation.
    Tanh,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Square.
    Square,
    /// Clamp to range.
    Clamp(f32, f32),
    /// Negate.
    Neg,
    /// Absolute value.
    Abs,
}

impl ElementwiseOp {
    /// Applies this operation to a single value.
    #[inline(always)]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ElementwiseOp::AddConst(c) => x + c,
            ElementwiseOp::MulConst(c) => x * c,
            ElementwiseOp::Relu => x.max(0.0),
            ElementwiseOp::LeakyRelu(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    x * alpha
                }
            }
            ElementwiseOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ElementwiseOp::Tanh => x.tanh(),
            ElementwiseOp::Exp => x.exp(),
            ElementwiseOp::Log => x.ln(),
            ElementwiseOp::Sqrt => x.sqrt(),
            ElementwiseOp::Square => x * x,
            ElementwiseOp::Clamp(min, max) => x.clamp(*min, *max),
            ElementwiseOp::Neg => -x,
            ElementwiseOp::Abs => x.abs(),
        }
    }
}

// =============================================================================
// Fused Elementwise Operation
// =============================================================================

/// A fused chain of elementwise operations.
#[derive(Debug, Clone)]
pub struct FusedElementwise {
    /// Chain of operations to apply.
    ops: Vec<ElementwiseOp>,
}

impl FusedElementwise {
    /// Creates a new fused elementwise operation.
    pub fn new(ops: Vec<ElementwiseOp>) -> Self {
        Self { ops }
    }

    /// Creates a builder for fused elementwise operations.
    pub fn builder() -> FusedElementwiseBuilder {
        FusedElementwiseBuilder::new()
    }

    /// Applies the fused operation to a single value.
    #[inline(always)]
    fn apply_chain(&self, mut x: f32) -> f32 {
        for op in &self.ops {
            x = op.apply(x);
        }
        x
    }

    /// Executes the fused operation on a tensor.
    pub fn forward(&self, input: &Tensor<f32>) -> FusionResult<Tensor<f32>> {
        let data = input.to_vec();

        let result: Vec<f32> = data.par_iter().map(|&x| self.apply_chain(x)).collect();

        Tensor::from_vec(result, input.shape())
            .map_err(|e| FusionError::TensorError(format!("{:?}", e)))
    }
}

impl FusedOp for FusedElementwise {
    fn execute(&self, inputs: &[&Tensor<f32>]) -> FusionResult<Tensor<f32>> {
        let input = inputs
            .first()
            .ok_or_else(|| FusionError::InvalidConfig("No input provided".to_string()))?;
        self.forward(input)
    }

    fn name(&self) -> &str {
        "FusedElementwise"
    }

    fn num_ops(&self) -> usize {
        self.ops.len()
    }
}

// =============================================================================
// Builder Pattern
// =============================================================================

/// Builder for fused elementwise operations.
#[derive(Debug, Default)]
pub struct FusedElementwiseBuilder {
    ops: Vec<ElementwiseOp>,
}

impl FusedElementwiseBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Adds an addition by constant.
    pub fn add(mut self, c: f32) -> Self {
        self.ops.push(ElementwiseOp::AddConst(c));
        self
    }

    /// Adds a multiplication by constant.
    pub fn mul(mut self, c: f32) -> Self {
        self.ops.push(ElementwiseOp::MulConst(c));
        self
    }

    /// Adds ReLU activation.
    pub fn relu(mut self) -> Self {
        self.ops.push(ElementwiseOp::Relu);
        self
    }

    /// Adds Leaky ReLU activation.
    pub fn leaky_relu(mut self, alpha: f32) -> Self {
        self.ops.push(ElementwiseOp::LeakyRelu(alpha));
        self
    }

    /// Adds sigmoid activation.
    pub fn sigmoid(mut self) -> Self {
        self.ops.push(ElementwiseOp::Sigmoid);
        self
    }

    /// Adds tanh activation.
    pub fn tanh(mut self) -> Self {
        self.ops.push(ElementwiseOp::Tanh);
        self
    }

    /// Adds exponential.
    pub fn exp(mut self) -> Self {
        self.ops.push(ElementwiseOp::Exp);
        self
    }

    /// Adds natural logarithm.
    pub fn log(mut self) -> Self {
        self.ops.push(ElementwiseOp::Log);
        self
    }

    /// Adds square root.
    pub fn sqrt(mut self) -> Self {
        self.ops.push(ElementwiseOp::Sqrt);
        self
    }

    /// Adds square operation.
    pub fn square(mut self) -> Self {
        self.ops.push(ElementwiseOp::Square);
        self
    }

    /// Adds clamp operation.
    pub fn clamp(mut self, min: f32, max: f32) -> Self {
        self.ops.push(ElementwiseOp::Clamp(min, max));
        self
    }

    /// Adds negation.
    pub fn neg(mut self) -> Self {
        self.ops.push(ElementwiseOp::Neg);
        self
    }

    /// Adds absolute value.
    pub fn abs(mut self) -> Self {
        self.ops.push(ElementwiseOp::Abs);
        self
    }

    /// Builds the fused operation.
    pub fn build(self) -> FusedElementwise {
        FusedElementwise::new(self.ops)
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Creates a fused elementwise operation from a list of ops.
pub fn fuse_elementwise(ops: Vec<ElementwiseOp>) -> FusedElementwise {
    FusedElementwise::new(ops)
}

/// Creates a fused add + relu operation.
pub fn fused_add_relu(bias: f32) -> FusedElementwise {
    FusedElementwise::builder().add(bias).relu().build()
}

/// Creates a fused multiply + add operation (FMA).
pub fn fused_mul_add(scale: f32, bias: f32) -> FusedElementwise {
    FusedElementwise::builder().mul(scale).add(bias).build()
}

/// Creates a fused scale + bias + relu (common in normalization).
pub fn fused_scale_bias_relu(scale: f32, bias: f32) -> FusedElementwise {
    FusedElementwise::builder()
        .mul(scale)
        .add(bias)
        .relu()
        .build()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_relu() {
        let fused = FusedElementwise::builder().relu().build();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let output = fused.forward(&input).unwrap();

        assert_eq!(output.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_fused_chain() {
        let fused = FusedElementwise::builder().mul(2.0).add(1.0).relu().build();

        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0], &[4]).unwrap();
        let output = fused.forward(&input).unwrap();

        // -2 * 2 + 1 = -3 -> relu -> 0
        // -1 * 2 + 1 = -1 -> relu -> 0
        //  0 * 2 + 1 =  1 -> relu -> 1
        //  1 * 2 + 1 =  3 -> relu -> 3
        assert_eq!(output.to_vec(), vec![0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn test_fused_sigmoid() {
        let fused = FusedElementwise::builder().sigmoid().build();
        let input = Tensor::from_vec(vec![0.0], &[1]).unwrap();
        let output = fused.forward(&input).unwrap();

        assert!((output.to_vec()[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_fused_mul_add() {
        let fused = fused_mul_add(2.0, 1.0);
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = fused.forward(&input).unwrap();

        assert_eq!(output.to_vec(), vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_fused_op_trait() {
        let fused = FusedElementwise::builder().relu().mul(2.0).build();
        assert_eq!(fused.num_ops(), 2);
        assert_eq!(fused.name(), "FusedElementwise");
    }

    #[test]
    fn test_clamp() {
        let fused = FusedElementwise::builder().clamp(-1.0, 1.0).build();

        let input = Tensor::from_vec(vec![-5.0, 0.0, 5.0], &[3]).unwrap();
        let output = fused.forward(&input).unwrap();

        assert_eq!(output.to_vec(), vec![-1.0, 0.0, 1.0]);
    }
}
