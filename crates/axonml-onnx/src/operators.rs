//! ONNX Operator Implementations
//!
//! Provides implementations of common ONNX operators using Axonml tensors.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use crate::error::{OnnxError, OnnxResult};
use crate::proto::NodeProto;
use axonml_tensor::creation::zeros;
use axonml_tensor::ops::{clamp, eq, gelu, gt, leaky_relu, lt, softmax};
use axonml_tensor::view::cat;
use axonml_tensor::Tensor;
use std::fmt::Debug;

// =============================================================================
// Helper Functions
// =============================================================================

/// Performs reduction along specified axes.
///
/// # Arguments
/// * `input` - Input tensor
/// * `axes` - Axes to reduce along (can be negative for indexing from end)
/// * `keepdims` - Whether to keep the reduced dimensions as size 1
/// * `reduce_fn` - Function to reduce a slice of values to a single value
fn reduce_along_axes<F>(
    input: &Tensor<f32>,
    axes: &[i64],
    keepdims: bool,
    reduce_fn: F,
) -> OnnxResult<Tensor<f32>>
where
    F: Fn(&[f32]) -> f32,
{
    let shape = input.shape();
    let ndim = shape.len();
    let data = input.to_vec();

    // Normalize negative axes
    let mut norm_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    norm_axes.sort_unstable();
    norm_axes.dedup();

    // Compute output shape
    let mut output_shape: Vec<usize> = Vec::new();
    for (i, &dim) in shape.iter().enumerate() {
        if norm_axes.contains(&i) {
            if keepdims {
                output_shape.push(1);
            }
        } else {
            output_shape.push(dim);
        }
    }

    // Handle case where all dimensions are reduced
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    // Compute strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Group elements by output position and reduce
    let output_numel: usize = output_shape.iter().product();
    let mut output_data = vec![Vec::new(); output_numel];

    for (flat_idx, &val) in data.iter().enumerate() {
        // Convert flat index to multi-index
        let mut multi_idx = vec![0usize; ndim];
        let mut remaining = flat_idx;
        for d in 0..ndim {
            multi_idx[d] = remaining / strides[d];
            remaining %= strides[d];
        }

        // Compute output index (zeroing reduced dims)
        let mut out_multi_idx: Vec<usize> = Vec::new();
        for (i, &idx) in multi_idx.iter().enumerate() {
            if norm_axes.contains(&i) {
                if keepdims {
                    out_multi_idx.push(0);
                }
            } else {
                out_multi_idx.push(idx);
            }
        }

        // Convert output multi-index to flat index
        let mut out_strides = vec![1usize; output_shape.len()];
        for i in (0..output_shape.len().saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * output_shape[i + 1];
        }

        let out_flat_idx: usize = out_multi_idx
            .iter()
            .zip(out_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();

        if out_flat_idx < output_data.len() {
            output_data[out_flat_idx].push(val);
        }
    }

    // Apply reduction function
    let reduced: Vec<f32> = output_data.iter().map(|v| reduce_fn(v)).collect();

    Tensor::from_vec(reduced, &output_shape)
        .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
}

// =============================================================================
// Operator Trait
// =============================================================================

/// Trait for ONNX operator implementations.
pub trait OnnxOperator: Debug + Send + Sync {
    /// Executes the operator with the given inputs.
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>>;

    /// Returns the operator name.
    fn name(&self) -> &str;
}

// =============================================================================
// Operator Factory
// =============================================================================

/// Creates an operator from an ONNX node.
pub fn create_operator(node: &NodeProto) -> OnnxResult<Box<dyn OnnxOperator>> {
    match node.op_type.as_str() {
        // Activation functions
        "Relu" => Ok(Box::new(ReluOp)),
        "Sigmoid" => Ok(Box::new(SigmoidOp)),
        "Tanh" => Ok(Box::new(TanhOp)),
        "Softmax" => Ok(Box::new(SoftmaxOp::from_node(node)?)),
        "LeakyRelu" => Ok(Box::new(LeakyReluOp::from_node(node)?)),
        "Gelu" => Ok(Box::new(GeluOp)),

        // Math operations
        "Add" => Ok(Box::new(AddOp)),
        "Sub" => Ok(Box::new(SubOp)),
        "Mul" => Ok(Box::new(MulOp)),
        "Div" => Ok(Box::new(DivOp)),
        "MatMul" => Ok(Box::new(MatMulOp)),
        "Gemm" => Ok(Box::new(GemmOp::from_node(node)?)),
        "Sqrt" => Ok(Box::new(SqrtOp)),
        "Pow" => Ok(Box::new(PowOp)),
        "Exp" => Ok(Box::new(ExpOp)),
        "Log" => Ok(Box::new(LogOp)),

        // Shape operations
        "Reshape" => Ok(Box::new(ReshapeOp)),
        "Transpose" => Ok(Box::new(TransposeOp::from_node(node)?)),
        "Flatten" => Ok(Box::new(FlattenOp::from_node(node)?)),
        "Squeeze" => Ok(Box::new(SqueezeOp::from_node(node)?)),
        "Unsqueeze" => Ok(Box::new(UnsqueezeOp::from_node(node)?)),
        "Concat" => Ok(Box::new(ConcatOp::from_node(node)?)),
        "Gather" => Ok(Box::new(GatherOp::from_node(node)?)),

        // Reduction operations
        "ReduceSum" => Ok(Box::new(ReduceSumOp::from_node(node)?)),
        "ReduceMean" => Ok(Box::new(ReduceMeanOp::from_node(node)?)),
        "ReduceMax" => Ok(Box::new(ReduceMaxOp::from_node(node)?)),

        // Neural network operations
        "Conv" => Ok(Box::new(ConvOp::from_node(node)?)),
        "MaxPool" => Ok(Box::new(MaxPoolOp::from_node(node)?)),
        "AveragePool" => Ok(Box::new(AvgPoolOp::from_node(node)?)),
        "BatchNormalization" => Ok(Box::new(BatchNormOp::from_node(node)?)),
        "Dropout" => Ok(Box::new(DropoutOp)),

        // Constant
        "Constant" => Ok(Box::new(ConstantOp::from_node(node)?)),
        "Identity" => Ok(Box::new(IdentityOp)),
        "Cast" => Ok(Box::new(CastOp)),
        "Shape" => Ok(Box::new(ShapeOp)),

        // Comparison
        "Equal" => Ok(Box::new(EqualOp)),
        "Greater" => Ok(Box::new(GreaterOp)),
        "Less" => Ok(Box::new(LessOp)),

        // Clip
        "Clip" => Ok(Box::new(ClipOp::from_node(node)?)),

        _ => Err(OnnxError::UnsupportedOperator(node.op_type.clone())),
    }
}

// =============================================================================
// Activation Operators
// =============================================================================

/// ReLU activation: max(0, x)
#[derive(Debug)]
pub struct ReluOp;

impl OnnxOperator for ReluOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![input.relu()])
    }

    fn name(&self) -> &str {
        "Relu"
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[derive(Debug)]
pub struct SigmoidOp;

impl OnnxOperator for SigmoidOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![input.sigmoid()])
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Tanh activation
#[derive(Debug)]
pub struct TanhOp;

impl OnnxOperator for TanhOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![input.tanh()])
    }

    fn name(&self) -> &str {
        "Tanh"
    }
}

/// Softmax activation
#[derive(Debug)]
pub struct SoftmaxOp {
    axis: i64,
}

impl SoftmaxOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axis = node.get_int("axis").unwrap_or(-1);
        Ok(Self { axis })
    }
}

impl OnnxOperator for SoftmaxOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        softmax(input, self.axis)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Softmax"
    }
}

/// LeakyReLU activation
#[derive(Debug)]
pub struct LeakyReluOp {
    alpha: f32,
}

impl LeakyReluOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let alpha = node.get_float("alpha").unwrap_or(0.01);
        Ok(Self { alpha })
    }
}

impl OnnxOperator for LeakyReluOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![leaky_relu(input, self.alpha)])
    }

    fn name(&self) -> &str {
        "LeakyRelu"
    }
}

/// GELU activation
#[derive(Debug)]
pub struct GeluOp;

impl OnnxOperator for GeluOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![gelu(input)])
    }

    fn name(&self) -> &str {
        "Gelu"
    }
}

// =============================================================================
// Math Operators
// =============================================================================

/// Element-wise addition
#[derive(Debug)]
pub struct AddOp;

impl OnnxOperator for AddOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        a.add(b)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Add"
    }
}

/// Element-wise subtraction
#[derive(Debug)]
pub struct SubOp;

impl OnnxOperator for SubOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        a.sub(b)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Sub"
    }
}

/// Element-wise multiplication
#[derive(Debug)]
pub struct MulOp;

impl OnnxOperator for MulOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        a.mul(b)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Mul"
    }
}

/// Element-wise division
#[derive(Debug)]
pub struct DivOp;

impl OnnxOperator for DivOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        a.div(b)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Div"
    }
}

/// Matrix multiplication
#[derive(Debug)]
pub struct MatMulOp;

impl OnnxOperator for MatMulOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        a.matmul(b)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "MatMul"
    }
}

/// Gemm: Y = alpha * A @ B + beta * C
#[derive(Debug)]
pub struct GemmOp {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl GemmOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        Ok(Self {
            alpha: node.get_float("alpha").unwrap_or(1.0),
            beta: node.get_float("beta").unwrap_or(1.0),
            trans_a: node.get_int("transA").unwrap_or(0) != 0,
            trans_b: node.get_int("transB").unwrap_or(0) != 0,
        })
    }
}

impl OnnxOperator for GemmOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        let a_t = if self.trans_a {
            a.transpose(0, 1)
                .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?
        } else {
            a.clone()
        };

        let b_t = if self.trans_b {
            b.transpose(0, 1)
                .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?
        } else {
            b.clone()
        };

        let mut result = a_t
            .matmul(&b_t)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;

        if (self.alpha - 1.0).abs() > 1e-6 {
            result = result.mul_scalar(self.alpha);
        }

        // Add bias if present
        if let Some(Some(c)) = inputs.get(2) {
            let bias_scaled = if (self.beta - 1.0).abs() > 1e-6 {
                c.mul_scalar(self.beta)
            } else {
                (*c).clone()
            };
            result = result
                .add(&bias_scaled)
                .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;
        }

        Ok(vec![result])
    }

    fn name(&self) -> &str {
        "Gemm"
    }
}

/// Square root
#[derive(Debug)]
pub struct SqrtOp;

impl OnnxOperator for SqrtOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![input.sqrt()])
    }

    fn name(&self) -> &str {
        "Sqrt"
    }
}

/// Power
#[derive(Debug)]
pub struct PowOp;

impl OnnxOperator for PowOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let base = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("base".to_string()))?;
        let exp = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("exponent".to_string()))?;

        // Get exponent value (assume scalar or broadcast)
        let exp_val = exp.to_vec()[0];
        Ok(vec![base.pow(exp_val)])
    }

    fn name(&self) -> &str {
        "Pow"
    }
}

/// Exponential
#[derive(Debug)]
pub struct ExpOp;

impl OnnxOperator for ExpOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![input.exp()])
    }

    fn name(&self) -> &str {
        "Exp"
    }
}

/// Natural logarithm
#[derive(Debug)]
pub struct LogOp;

impl OnnxOperator for LogOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![input.ln()])
    }

    fn name(&self) -> &str {
        "Log"
    }
}

// =============================================================================
// Shape Operators
// =============================================================================

/// Reshape operator
#[derive(Debug)]
pub struct ReshapeOp;

impl OnnxOperator for ReshapeOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let data = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("data".to_string()))?;
        let shape_tensor = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("shape".to_string()))?;

        let shape: Vec<isize> = shape_tensor.to_vec().iter().map(|&x| x as isize).collect();

        data.reshape(&shape)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Reshape"
    }
}

/// Transpose operator
#[derive(Debug)]
pub struct TransposeOp {
    perm: Option<Vec<i64>>,
}

impl TransposeOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let perm = node.get_ints("perm").map(|p| p.to_vec());
        Ok(Self { perm })
    }
}

impl OnnxOperator for TransposeOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        let perm: Vec<usize> = if let Some(ref p) = self.perm {
            p.iter().map(|&x| x as usize).collect()
        } else {
            // Default: reverse all dimensions
            (0..input.ndim()).rev().collect()
        };

        input
            .permute(&perm)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Transpose"
    }
}

/// Flatten operator
#[derive(Debug)]
pub struct FlattenOp {
    axis: i64,
}

impl FlattenOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axis = node.get_int("axis").unwrap_or(1);
        Ok(Self { axis })
    }
}

impl OnnxOperator for FlattenOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        let axis = if self.axis < 0 {
            (input.ndim() as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let shape = input.shape();
        let dim0: usize = shape[..axis].iter().product();
        let dim1: usize = shape[axis..].iter().product();

        input
            .reshape(&[dim0 as isize, dim1 as isize])
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Flatten"
    }
}

/// Squeeze operator
#[derive(Debug)]
pub struct SqueezeOp {
    axes: Option<Vec<i64>>,
}

impl SqueezeOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axes = node.get_ints("axes").map(|a| a.to_vec());
        Ok(Self { axes })
    }
}

impl OnnxOperator for SqueezeOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        // If axes tensor provided (ONNX opset >= 13), use it
        let axes_from_input = inputs
            .get(1)
            .and_then(|i| *i)
            .map(|t| t.to_vec().iter().map(|&x| x as i64).collect::<Vec<_>>());

        let axes = axes_from_input.or_else(|| self.axes.clone());

        if let Some(axes) = axes {
            let mut result = input.clone();
            for &axis in axes.iter().rev() {
                result = result
                    .squeeze(Some(axis))
                    .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;
            }
            Ok(vec![result])
        } else {
            // Squeeze all dimensions of size 1
            input
                .squeeze(None)
                .map(|t| vec![t])
                .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
        }
    }

    fn name(&self) -> &str {
        "Squeeze"
    }
}

/// Unsqueeze operator
#[derive(Debug)]
pub struct UnsqueezeOp {
    axes: Option<Vec<i64>>,
}

impl UnsqueezeOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axes = node.get_ints("axes").map(|a| a.to_vec());
        Ok(Self { axes })
    }
}

impl OnnxOperator for UnsqueezeOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        let axes_from_input = inputs
            .get(1)
            .and_then(|i| *i)
            .map(|t| t.to_vec().iter().map(|&x| x as i64).collect::<Vec<_>>());

        let axes = axes_from_input
            .or_else(|| self.axes.clone())
            .ok_or_else(|| OnnxError::MissingAttribute("axes".to_string()))?;

        let mut result = input.clone();
        let mut sorted_axes = axes.clone();
        sorted_axes.sort();

        for &axis in &sorted_axes {
            result = result
                .unsqueeze(axis)
                .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;
        }

        Ok(vec![result])
    }

    fn name(&self) -> &str {
        "Unsqueeze"
    }
}

/// Concat operator
#[derive(Debug)]
pub struct ConcatOp {
    axis: i64,
}

impl ConcatOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axis = node
            .get_int("axis")
            .ok_or_else(|| OnnxError::MissingAttribute("axis".to_string()))?;
        Ok(Self { axis })
    }
}

impl OnnxOperator for ConcatOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let tensors: Vec<Tensor<f32>> =
            inputs.iter().filter_map(|i| i.map(|t| t.clone())).collect();

        if tensors.is_empty() {
            return Err(OnnxError::MissingAttribute("inputs".to_string()));
        }

        let axis = if self.axis < 0 {
            (tensors[0].ndim() as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        cat(&tensors, axis)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Concat"
    }
}

/// Gather operator
#[derive(Debug)]
pub struct GatherOp {
    axis: i64,
}

impl GatherOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axis = node.get_int("axis").unwrap_or(0);
        Ok(Self { axis })
    }
}

impl OnnxOperator for GatherOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let data = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("data".to_string()))?;
        let indices_f32 = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("indices".to_string()))?;

        // Convert f32 indices to i64
        let indices_data: Vec<i64> = indices_f32.to_vec().iter().map(|&x| x as i64).collect();
        let indices = Tensor::<i64>::from_vec(indices_data, indices_f32.shape())
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;

        let axis = if self.axis < 0 {
            (data.ndim() as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        data.gather(axis, &indices)
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Gather"
    }
}

// =============================================================================
// Reduction Operators
// =============================================================================

/// ReduceSum operator
#[derive(Debug)]
pub struct ReduceSumOp {
    axes: Option<Vec<i64>>,
    keepdims: bool,
}

impl ReduceSumOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axes = node.get_ints("axes").map(|a| a.to_vec());
        let keepdims = node.get_int("keepdims").unwrap_or(1) != 0;
        Ok(Self { axes, keepdims })
    }
}

impl OnnxOperator for ReduceSumOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        // Check for axes from second input (opset >= 13)
        let axes_from_input = inputs
            .get(1)
            .and_then(|i| *i)
            .map(|t| t.to_vec().iter().map(|&x| x as i64).collect::<Vec<_>>());

        let axes = axes_from_input.or_else(|| self.axes.clone());

        if let Some(ref axes_vec) = axes {
            if !axes_vec.is_empty() {
                // Perform dimension-specific reduction
                let result =
                    reduce_along_axes(input, axes_vec, self.keepdims, |data| data.iter().sum())?;
                return Ok(vec![result]);
            }
        }
        // Sum all elements
        Ok(vec![input.sum()])
    }

    fn name(&self) -> &str {
        "ReduceSum"
    }
}

/// ReduceMean operator
#[derive(Debug)]
pub struct ReduceMeanOp {
    axes: Option<Vec<i64>>,
    keepdims: bool,
}

impl ReduceMeanOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axes = node.get_ints("axes").map(|a| a.to_vec());
        let keepdims = node.get_int("keepdims").unwrap_or(1) != 0;
        Ok(Self { axes, keepdims })
    }
}

impl OnnxOperator for ReduceMeanOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        if let Some(ref axes_vec) = self.axes {
            if !axes_vec.is_empty() {
                // Perform dimension-specific reduction
                let result = reduce_along_axes(input, axes_vec, self.keepdims, |data| {
                    if data.is_empty() {
                        0.0
                    } else {
                        data.iter().sum::<f32>() / data.len() as f32
                    }
                })?;
                return Ok(vec![result]);
            }
        }
        // Global mean
        input
            .mean()
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "ReduceMean"
    }
}

/// ReduceMax operator
#[derive(Debug)]
pub struct ReduceMaxOp {
    axes: Option<Vec<i64>>,
    keepdims: bool,
}

impl ReduceMaxOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let axes = node.get_ints("axes").map(|a| a.to_vec());
        let keepdims = node.get_int("keepdims").unwrap_or(1) != 0;
        Ok(Self { axes, keepdims })
    }
}

impl OnnxOperator for ReduceMaxOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        if let Some(ref axes_vec) = self.axes {
            if !axes_vec.is_empty() {
                // Perform dimension-specific reduction
                let result = reduce_along_axes(input, axes_vec, self.keepdims, |data| {
                    data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                })?;
                return Ok(vec![result]);
            }
        }
        // Global max
        input
            .max()
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "ReduceMax"
    }
}

// =============================================================================
// Neural Network Operators
// =============================================================================

/// Conv operator
#[derive(Debug)]
pub struct ConvOp {
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
    pads: Vec<i64>,
    dilations: Vec<i64>,
    group: i64,
}

impl ConvOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        Ok(Self {
            kernel_shape: node
                .get_ints("kernel_shape")
                .map(|k| k.to_vec())
                .unwrap_or_default(),
            strides: node
                .get_ints("strides")
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![1, 1]),
            pads: node
                .get_ints("pads")
                .map(|p| p.to_vec())
                .unwrap_or_else(|| vec![0, 0, 0, 0]),
            dilations: node
                .get_ints("dilations")
                .map(|d| d.to_vec())
                .unwrap_or_else(|| vec![1, 1]),
            group: node.get_int("group").unwrap_or(1),
        })
    }
}

impl OnnxOperator for ConvOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("X".to_string()))?;
        let weight = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("W".to_string()))?;
        let bias = inputs.get(2).and_then(|i| *i);

        // Input shape: [batch, in_channels, height, width]
        // Weight shape: [out_channels, in_channels/group, kernel_h, kernel_w]
        let input_shape = input.shape();
        let weight_shape = weight.shape();

        if input_shape.len() != 4 || weight_shape.len() != 4 {
            return Err(OnnxError::InvalidShape(
                "Conv requires 4D input and weight tensors".to_string(),
            ));
        }

        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let out_channels = weight_shape[0];
        let kernel_h = self
            .kernel_shape
            .get(0)
            .copied()
            .unwrap_or(weight_shape[2] as i64) as usize;
        let kernel_w = self
            .kernel_shape
            .get(1)
            .copied()
            .unwrap_or(weight_shape[3] as i64) as usize;

        let stride_h = self.strides.get(0).copied().unwrap_or(1) as usize;
        let stride_w = self.strides.get(1).copied().unwrap_or(1) as usize;

        let pad_h_begin = self.pads.get(0).copied().unwrap_or(0) as usize;
        let pad_w_begin = self.pads.get(1).copied().unwrap_or(0) as usize;
        let pad_h_end = self.pads.get(2).copied().unwrap_or(pad_h_begin as i64) as usize;
        let pad_w_end = self.pads.get(3).copied().unwrap_or(pad_w_begin as i64) as usize;

        let dilation_h = self.dilations.get(0).copied().unwrap_or(1) as usize;
        let dilation_w = self.dilations.get(1).copied().unwrap_or(1) as usize;

        let group = self.group as usize;

        // Calculate output dimensions
        let out_h =
            (in_h + pad_h_begin + pad_h_end - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        let out_w =
            (in_w + pad_w_begin + pad_w_end - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        let input_data = input.to_vec();
        let weight_data = weight.to_vec();
        let bias_data = bias.map(|b| b.to_vec());

        let mut output = vec![0.0f32; batch * out_channels * out_h * out_w];

        let in_channels_per_group = in_channels / group;
        let out_channels_per_group = out_channels / group;

        // Perform grouped convolution
        for b in 0..batch {
            for g in 0..group {
                for oc in 0..out_channels_per_group {
                    let out_c = g * out_channels_per_group + oc;

                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;

                            for ic in 0..in_channels_per_group {
                                let in_c = g * in_channels_per_group + ic;

                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let ih = (oh * stride_h + kh * dilation_h) as isize
                                            - pad_h_begin as isize;
                                        let iw = (ow * stride_w + kw * dilation_w) as isize
                                            - pad_w_begin as isize;

                                        if ih >= 0
                                            && ih < in_h as isize
                                            && iw >= 0
                                            && iw < in_w as isize
                                        {
                                            let input_idx = b * in_channels * in_h * in_w
                                                + in_c * in_h * in_w
                                                + ih as usize * in_w
                                                + iw as usize;

                                            let weight_idx = out_c
                                                * (in_channels_per_group * kernel_h * kernel_w)
                                                + ic * kernel_h * kernel_w
                                                + kh * kernel_w
                                                + kw;

                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }

                            // Add bias if present
                            if let Some(ref bias) = bias_data {
                                sum += bias[out_c];
                            }

                            let output_idx = b * out_channels * out_h * out_w
                                + out_c * out_h * out_w
                                + oh * out_w
                                + ow;
                            output[output_idx] = sum;
                        }
                    }
                }
            }
        }

        Tensor::from_vec(output, &[batch, out_channels, out_h, out_w])
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Conv"
    }
}

/// MaxPool operator - 2D max pooling
#[derive(Debug)]
pub struct MaxPoolOp {
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
    pads: Vec<i64>,
}

impl MaxPoolOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        Ok(Self {
            kernel_shape: node
                .get_ints("kernel_shape")
                .map(|k| k.to_vec())
                .unwrap_or_default(),
            strides: node
                .get_ints("strides")
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![1, 1]),
            pads: node
                .get_ints("pads")
                .map(|p| p.to_vec())
                .unwrap_or_else(|| vec![0, 0, 0, 0]),
        })
    }
}

impl OnnxOperator for MaxPoolOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let x = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("X".to_string()))?;

        let shape = x.shape();
        if shape.len() != 4 {
            return Err(OnnxError::InvalidShape(format!(
                "MaxPool requires 4D input [N,C,H,W], got {:?}",
                shape
            )));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let kh = self.kernel_shape.get(0).copied().unwrap_or(2) as usize;
        let kw = self.kernel_shape.get(1).copied().unwrap_or(2) as usize;
        let sh = self.strides.get(0).copied().unwrap_or(1) as usize;
        let sw = self.strides.get(1).copied().unwrap_or(1) as usize;
        let pad_top = self.pads.get(0).copied().unwrap_or(0) as usize;
        let pad_left = self.pads.get(1).copied().unwrap_or(0) as usize;
        let pad_bottom = self.pads.get(2).copied().unwrap_or(0) as usize;
        let pad_right = self.pads.get(3).copied().unwrap_or(0) as usize;

        let out_h = (h + pad_top + pad_bottom - kh) / sh + 1;
        let out_w = (w + pad_left + pad_right - kw) / sw + 1;

        let x_data = x.to_vec();
        let mut output = vec![f32::NEG_INFINITY; n * c * out_h * out_w];

        for batch in 0..n {
            for channel in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = (oh * sh + khi) as isize - pad_top as isize;
                                let iw = (ow * sw + kwi) as isize - pad_left as isize;

                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let idx = batch * c * h * w
                                        + channel * h * w
                                        + ih as usize * w
                                        + iw as usize;
                                    max_val = max_val.max(x_data[idx]);
                                }
                            }
                        }

                        let out_idx =
                            batch * c * out_h * out_w + channel * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }

        Tensor::from_vec(output, &[n, c, out_h, out_w])
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "MaxPool"
    }
}

/// AveragePool operator - 2D average pooling
#[derive(Debug)]
pub struct AvgPoolOp {
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
    pads: Vec<i64>,
    count_include_pad: bool,
}

impl AvgPoolOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        Ok(Self {
            kernel_shape: node
                .get_ints("kernel_shape")
                .map(|k| k.to_vec())
                .unwrap_or_default(),
            strides: node
                .get_ints("strides")
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![1, 1]),
            pads: node
                .get_ints("pads")
                .map(|p| p.to_vec())
                .unwrap_or_else(|| vec![0, 0, 0, 0]),
            count_include_pad: node.get_int("count_include_pad").unwrap_or(0) != 0,
        })
    }
}

impl OnnxOperator for AvgPoolOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let x = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("X".to_string()))?;

        let shape = x.shape();
        if shape.len() != 4 {
            return Err(OnnxError::InvalidShape(format!(
                "AveragePool requires 4D input [N,C,H,W], got {:?}",
                shape
            )));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let kh = self.kernel_shape.get(0).copied().unwrap_or(2) as usize;
        let kw = self.kernel_shape.get(1).copied().unwrap_or(2) as usize;
        let sh = self.strides.get(0).copied().unwrap_or(1) as usize;
        let sw = self.strides.get(1).copied().unwrap_or(1) as usize;
        let pad_top = self.pads.get(0).copied().unwrap_or(0) as usize;
        let pad_left = self.pads.get(1).copied().unwrap_or(0) as usize;
        let pad_bottom = self.pads.get(2).copied().unwrap_or(0) as usize;
        let pad_right = self.pads.get(3).copied().unwrap_or(0) as usize;

        let out_h = (h + pad_top + pad_bottom - kh) / sh + 1;
        let out_w = (w + pad_left + pad_right - kw) / sw + 1;

        let x_data = x.to_vec();
        let mut output = vec![0.0f32; n * c * out_h * out_w];

        for batch in 0..n {
            for channel in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0usize;

                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = (oh * sh + khi) as isize - pad_top as isize;
                                let iw = (ow * sw + kwi) as isize - pad_left as isize;

                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let idx = batch * c * h * w
                                        + channel * h * w
                                        + ih as usize * w
                                        + iw as usize;
                                    sum += x_data[idx];
                                    count += 1;
                                } else if self.count_include_pad {
                                    count += 1;
                                }
                            }
                        }

                        let out_idx =
                            batch * c * out_h * out_w + channel * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        Tensor::from_vec(output, &[n, c, out_h, out_w])
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "AveragePool"
    }
}

/// BatchNormalization operator
#[derive(Debug)]
pub struct BatchNormOp {
    epsilon: f32,
    /// Momentum is only used during training for updating running stats
    _momentum: f32,
}

impl BatchNormOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        Ok(Self {
            epsilon: node.get_float("epsilon").unwrap_or(1e-5),
            _momentum: node.get_float("momentum").unwrap_or(0.9),
        })
    }
}

impl OnnxOperator for BatchNormOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let x = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("X".to_string()))?;
        let scale = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("scale".to_string()))?;
        let bias = inputs
            .get(2)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;
        let mean = inputs
            .get(3)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("mean".to_string()))?;
        let var = inputs
            .get(4)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("var".to_string()))?;

        // Batch norm: y = scale * (x - mean) / sqrt(var + eps) + bias
        // Simplified implementation for inference mode
        let x_centered = x
            .sub(mean)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;

        let std = var.add_scalar(self.epsilon).sqrt();

        let x_norm = x_centered
            .div(&std)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;

        let scaled = x_norm
            .mul(scale)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;

        let result = scaled
            .add(bias)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;

        Ok(vec![result])
    }

    fn name(&self) -> &str {
        "BatchNormalization"
    }
}

/// Dropout operator (inference mode - identity)
#[derive(Debug)]
pub struct DropoutOp;

impl OnnxOperator for DropoutOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("data".to_string()))?;

        // In inference mode, dropout is identity
        Ok(vec![input.clone()])
    }

    fn name(&self) -> &str {
        "Dropout"
    }
}

// =============================================================================
// Utility Operators
// =============================================================================

/// Constant operator
#[derive(Debug)]
pub struct ConstantOp {
    value: Tensor<f32>,
}

impl ConstantOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        let attr = node
            .get_attribute("value")
            .ok_or_else(|| OnnxError::MissingAttribute("value".to_string()))?;

        let tensor_proto = attr.t.as_ref().ok_or_else(|| {
            OnnxError::InvalidAttribute("value".to_string(), "not a tensor".to_string())
        })?;

        let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();
        let data = tensor_proto.get_float_data();

        let value = if data.is_empty() {
            zeros(&shape)
        } else {
            Tensor::from_vec(data, &shape)
                .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?
        };

        Ok(Self { value })
    }
}

impl OnnxOperator for ConstantOp {
    fn execute(&self, _inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        Ok(vec![self.value.clone()])
    }

    fn name(&self) -> &str {
        "Constant"
    }
}

/// Identity operator
#[derive(Debug)]
pub struct IdentityOp;

impl OnnxOperator for IdentityOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        Ok(vec![input.clone()])
    }

    fn name(&self) -> &str {
        "Identity"
    }
}

/// Cast operator (simplified - assumes f32)
#[derive(Debug)]
pub struct CastOp;

impl OnnxOperator for CastOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        // For now, just return the input (already f32)
        Ok(vec![input.clone()])
    }

    fn name(&self) -> &str {
        "Cast"
    }
}

/// Shape operator
#[derive(Debug)]
pub struct ShapeOp;

impl OnnxOperator for ShapeOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("data".to_string()))?;

        let shape_data: Vec<f32> = input.shape().iter().map(|&d| d as f32).collect();
        let shape_len = shape_data.len();

        Tensor::from_vec(shape_data, &[shape_len])
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Shape"
    }
}

// =============================================================================
// Comparison Operators
// =============================================================================

/// Equal comparison
#[derive(Debug)]
pub struct EqualOp;

impl OnnxOperator for EqualOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        let result = eq(a, b).map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;
        let float_result: Vec<f32> = result.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        Tensor::from_vec(float_result, a.shape())
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Equal"
    }
}

/// Greater comparison
#[derive(Debug)]
pub struct GreaterOp;

impl OnnxOperator for GreaterOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        let result = gt(a, b).map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;
        let float_result: Vec<f32> = result.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        Tensor::from_vec(float_result, a.shape())
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Greater"
    }
}

/// Less comparison
#[derive(Debug)]
pub struct LessOp;

impl OnnxOperator for LessOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let a = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("A".to_string()))?;
        let b = inputs
            .get(1)
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("B".to_string()))?;

        let result = lt(a, b).map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))?;
        let float_result: Vec<f32> = result.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        Tensor::from_vec(float_result, a.shape())
            .map(|t| vec![t])
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
    }

    fn name(&self) -> &str {
        "Less"
    }
}

/// Clip operator
#[derive(Debug)]
pub struct ClipOp {
    min: Option<f32>,
    max: Option<f32>,
}

impl ClipOp {
    fn from_node(node: &NodeProto) -> OnnxResult<Self> {
        Ok(Self {
            min: node.get_float("min"),
            max: node.get_float("max"),
        })
    }
}

impl OnnxOperator for ClipOp {
    fn execute(&self, inputs: &[Option<&Tensor<f32>>]) -> OnnxResult<Vec<Tensor<f32>>> {
        let input = inputs
            .first()
            .and_then(|i| *i)
            .ok_or_else(|| OnnxError::MissingAttribute("input".to_string()))?;

        // Get min/max from inputs (opset >= 11) or attributes
        let min_val = inputs
            .get(1)
            .and_then(|i| *i)
            .map(|t| t.to_vec()[0])
            .or(self.min)
            .unwrap_or(f32::NEG_INFINITY);

        let max_val = inputs
            .get(2)
            .and_then(|i| *i)
            .map(|t| t.to_vec()[0])
            .or(self.max)
            .unwrap_or(f32::INFINITY);

        Ok(vec![clamp(input, min_val, max_val)])
    }

    fn name(&self) -> &str {
        "Clip"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_op() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let op = ReluOp;
        let result = op.execute(&[Some(&input)]).unwrap();
        let output = result[0].to_vec();
        assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_add_op() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let op = AddOp;
        let result = op.execute(&[Some(&a), Some(&b)]).unwrap();
        let output = result[0].to_vec();
        assert_eq!(output, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul_op() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let op = MatMulOp;
        let result = op.execute(&[Some(&a), Some(&b)]).unwrap();
        assert_eq!(result[0].shape(), &[2, 2]);
    }
}
