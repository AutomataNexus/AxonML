//! Model Conversion Utilities
//!
//! Helpers for converting between Axonml and other formats (`PyTorch`, ONNX).

use crate::{StateDict, TensorData};
use std::collections::HashMap;

// =============================================================================
// PyTorch Conversion
// =============================================================================

/// Convert a PyTorch-style key to Axonml format.
///
/// `PyTorch` uses keys like:
/// - "module.layer1.weight"
/// - "`encoder.layers.0.self_attn.q_proj.weight`"
///
/// This function normalizes them for Axonml.
#[must_use]
pub fn from_pytorch_key(key: &str) -> String {
    let mut result = key.to_string();

    // Remove common prefixes
    if result.starts_with("module.") {
        result = result.strip_prefix("module.").unwrap().to_string();
    }
    if result.starts_with("_orig_mod.") {
        result = result.strip_prefix("_orig_mod.").unwrap().to_string();
    }

    result
}

/// Convert a Axonml key to `PyTorch` format.
#[must_use]
pub fn to_pytorch_key(key: &str) -> String {
    // Add "module." prefix if not present (for DDP models)
    key.to_string()
}

/// Map of `PyTorch` layer names to Axonml equivalents.
#[must_use]
pub fn pytorch_layer_mapping() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    // Linear layers
    map.insert("fc", "linear");
    map.insert("dense", "linear");

    // Convolutions
    map.insert("conv", "conv");

    // Normalization
    map.insert("bn", "batch_norm");
    map.insert("batch_norm", "batch_norm");
    map.insert("layer_norm", "layer_norm");
    map.insert("ln", "layer_norm");

    // Attention
    map.insert("self_attn", "attention");
    map.insert("multihead_attn", "attention");

    map
}

// =============================================================================
// ONNX Conversion
// =============================================================================

/// Convert a shape to ONNX format (with batch dimension handling).
#[must_use]
pub fn to_onnx_shape(shape: &[usize], include_batch: bool) -> Vec<i64> {
    if include_batch {
        // ONNX uses -1 for dynamic batch size
        std::iter::once(-1i64)
            .chain(shape.iter().map(|&d| d as i64))
            .collect()
    } else {
        shape.iter().map(|&d| d as i64).collect()
    }
}

/// Convert from ONNX shape (handling -1 for dynamic dimensions).
#[must_use]
pub fn from_onnx_shape(shape: &[i64], default_dynamic: usize) -> Vec<usize> {
    shape
        .iter()
        .map(|&d| if d < 0 { default_dynamic } else { d as usize })
        .collect()
}

/// ONNX operator type mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxOpType {
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Matrix multiplication.
    MatMul,
    /// General matrix multiplication (with bias).
    Gemm,
    /// Rectified Linear Unit activation.
    Relu,
    /// Sigmoid activation.
    Sigmoid,
    /// Hyperbolic tangent activation.
    Tanh,
    /// Softmax activation.
    Softmax,
    /// Gaussian Error Linear Unit activation.
    Gelu,
    /// Convolution operation.
    Conv,
    /// Transposed convolution (deconvolution).
    ConvTranspose,
    /// Max pooling operation.
    MaxPool,
    /// Average pooling operation.
    AveragePool,
    /// Global average pooling operation.
    GlobalAveragePool,
    /// Batch normalization.
    BatchNormalization,
    /// Layer normalization.
    LayerNormalization,
    /// Reshape tensor dimensions.
    Reshape,
    /// Transpose tensor dimensions.
    Transpose,
    /// Flatten tensor to 2D.
    Flatten,
    /// Remove dimensions of size 1.
    Squeeze,
    /// Add dimension of size 1.
    Unsqueeze,
    /// Concatenate tensors along axis.
    Concat,
    /// Sum reduction along axis.
    ReduceSum,
    /// Mean reduction along axis.
    ReduceMean,
    /// Max reduction along axis.
    ReduceMax,
    /// Min reduction along axis.
    ReduceMin,
    /// Dropout layer (training regularization).
    Dropout,
    /// Constant tensor.
    Constant,
    /// Identity pass-through.
    Identity,
    /// Unknown or unsupported operator.
    Unknown,
}

impl OnnxOpType {
    /// Parse ONNX operator type from string.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s {
            "Add" => Self::Add,
            "Sub" => Self::Sub,
            "Mul" => Self::Mul,
            "Div" => Self::Div,
            "MatMul" => Self::MatMul,
            "Gemm" => Self::Gemm,
            "Relu" => Self::Relu,
            "Sigmoid" => Self::Sigmoid,
            "Tanh" => Self::Tanh,
            "Softmax" => Self::Softmax,
            "Gelu" => Self::Gelu,
            "Conv" => Self::Conv,
            "ConvTranspose" => Self::ConvTranspose,
            "MaxPool" => Self::MaxPool,
            "AveragePool" => Self::AveragePool,
            "GlobalAveragePool" => Self::GlobalAveragePool,
            "BatchNormalization" => Self::BatchNormalization,
            "LayerNormalization" => Self::LayerNormalization,
            "Reshape" => Self::Reshape,
            "Transpose" => Self::Transpose,
            "Flatten" => Self::Flatten,
            "Squeeze" => Self::Squeeze,
            "Unsqueeze" => Self::Unsqueeze,
            "Concat" => Self::Concat,
            "ReduceSum" => Self::ReduceSum,
            "ReduceMean" => Self::ReduceMean,
            "ReduceMax" => Self::ReduceMax,
            "ReduceMin" => Self::ReduceMin,
            "Dropout" => Self::Dropout,
            "Constant" => Self::Constant,
            "Identity" => Self::Identity,
            _ => Self::Unknown,
        }
    }

    /// Get the ONNX operator name.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::MatMul => "MatMul",
            Self::Gemm => "Gemm",
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::Softmax => "Softmax",
            Self::Gelu => "Gelu",
            Self::Conv => "Conv",
            Self::ConvTranspose => "ConvTranspose",
            Self::MaxPool => "MaxPool",
            Self::AveragePool => "AveragePool",
            Self::GlobalAveragePool => "GlobalAveragePool",
            Self::BatchNormalization => "BatchNormalization",
            Self::LayerNormalization => "LayerNormalization",
            Self::Reshape => "Reshape",
            Self::Transpose => "Transpose",
            Self::Flatten => "Flatten",
            Self::Squeeze => "Squeeze",
            Self::Unsqueeze => "Unsqueeze",
            Self::Concat => "Concat",
            Self::ReduceSum => "ReduceSum",
            Self::ReduceMean => "ReduceMean",
            Self::ReduceMax => "ReduceMax",
            Self::ReduceMin => "ReduceMin",
            Self::Dropout => "Dropout",
            Self::Constant => "Constant",
            Self::Identity => "Identity",
            Self::Unknown => "Unknown",
        }
    }
}

// =============================================================================
// State Dict Conversion
// =============================================================================

/// Convert a state dict from `PyTorch` naming conventions.
#[must_use]
pub fn convert_from_pytorch(state_dict: &StateDict) -> StateDict {
    let mut converted = StateDict::new();

    for (key, entry) in state_dict.entries() {
        let new_key = from_pytorch_key(key);
        converted.insert_entry(new_key, entry.clone());
    }

    converted
}

/// Transpose weights if needed for format conversion.
///
/// `PyTorch` Linear: [`out_features`, `in_features`]
/// Some frameworks: [`in_features`, `out_features`]
#[must_use]
pub fn transpose_linear_weights(data: &TensorData) -> TensorData {
    if data.shape.len() != 2 {
        return data.clone();
    }

    let (rows, cols) = (data.shape[0], data.shape[1]);
    let mut transposed = vec![0.0; data.values.len()];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = data.values[i * cols + j];
        }
    }

    TensorData {
        shape: vec![cols, rows],
        values: transposed,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_pytorch_key() {
        assert_eq!(from_pytorch_key("module.layer1.weight"), "layer1.weight");
        assert_eq!(from_pytorch_key("layer1.weight"), "layer1.weight");
        assert_eq!(
            from_pytorch_key("_orig_mod.encoder.weight"),
            "encoder.weight"
        );
    }

    #[test]
    fn test_to_onnx_shape() {
        assert_eq!(to_onnx_shape(&[3, 4], false), vec![3, 4]);
        assert_eq!(to_onnx_shape(&[3, 4], true), vec![-1, 3, 4]);
    }

    #[test]
    fn test_from_onnx_shape() {
        assert_eq!(from_onnx_shape(&[3, 4], 1), vec![3, 4]);
        assert_eq!(from_onnx_shape(&[-1, 3, 4], 8), vec![8, 3, 4]);
    }

    #[test]
    fn test_onnx_op_type() {
        assert_eq!(OnnxOpType::from_str("Relu"), OnnxOpType::Relu);
        assert_eq!(OnnxOpType::from_str("MatMul"), OnnxOpType::MatMul);
        assert_eq!(OnnxOpType::from_str("Unknown"), OnnxOpType::Unknown);

        assert_eq!(OnnxOpType::Relu.as_str(), "Relu");
    }

    #[test]
    fn test_transpose_linear_weights() {
        let data = TensorData {
            shape: vec![2, 3],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let transposed = transpose_linear_weights(&data);
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_convert_from_pytorch() {
        let mut pytorch_dict = StateDict::new();
        pytorch_dict.insert(
            "module.linear.weight".to_string(),
            TensorData {
                shape: vec![10, 5],
                values: vec![0.0; 50],
            },
        );

        let converted = convert_from_pytorch(&pytorch_dict);
        assert!(converted.contains("linear.weight"));
        assert!(!converted.contains("module.linear.weight"));
    }
}
