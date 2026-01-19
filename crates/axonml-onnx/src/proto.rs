//! ONNX Protocol Buffer Definitions
//!
//! Manual Rust implementations of ONNX protobuf structures.
//! These match the ONNX specification for parsing .onnx files.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Data Types
// =============================================================================

/// ONNX tensor element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum TensorDataType {
    /// Undefined type.
    Undefined = 0,
    /// 32-bit float.
    Float = 1,
    /// 8-bit unsigned int.
    Uint8 = 2,
    /// 8-bit signed int.
    Int8 = 3,
    /// 16-bit unsigned int.
    Uint16 = 4,
    /// 16-bit signed int.
    Int16 = 5,
    /// 32-bit signed int.
    Int32 = 6,
    /// 64-bit signed int.
    Int64 = 7,
    /// String type.
    String = 8,
    /// Boolean type.
    Bool = 9,
    /// 16-bit float (half precision).
    Float16 = 10,
    /// 64-bit float (double).
    Double = 11,
    /// 32-bit unsigned int.
    Uint32 = 12,
    /// 64-bit unsigned int.
    Uint64 = 13,
    /// Complex 64-bit float.
    Complex64 = 14,
    /// Complex 128-bit float.
    Complex128 = 15,
    /// BFloat16.
    Bfloat16 = 16,
}

impl TensorDataType {
    /// Returns the size in bytes for this data type.
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorDataType::Undefined => 0,
            TensorDataType::Bool | TensorDataType::Int8 | TensorDataType::Uint8 => 1,
            TensorDataType::Float16 | TensorDataType::Bfloat16 | TensorDataType::Int16 | TensorDataType::Uint16 => 2,
            TensorDataType::Float | TensorDataType::Int32 | TensorDataType::Uint32 => 4,
            TensorDataType::Double | TensorDataType::Int64 | TensorDataType::Uint64 | TensorDataType::Complex64 => 8,
            TensorDataType::Complex128 => 16,
            TensorDataType::String => 0, // Variable
        }
    }

    /// Creates from i32 value.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(TensorDataType::Undefined),
            1 => Some(TensorDataType::Float),
            2 => Some(TensorDataType::Uint8),
            3 => Some(TensorDataType::Int8),
            4 => Some(TensorDataType::Uint16),
            5 => Some(TensorDataType::Int16),
            6 => Some(TensorDataType::Int32),
            7 => Some(TensorDataType::Int64),
            8 => Some(TensorDataType::String),
            9 => Some(TensorDataType::Bool),
            10 => Some(TensorDataType::Float16),
            11 => Some(TensorDataType::Double),
            12 => Some(TensorDataType::Uint32),
            13 => Some(TensorDataType::Uint64),
            14 => Some(TensorDataType::Complex64),
            15 => Some(TensorDataType::Complex128),
            16 => Some(TensorDataType::Bfloat16),
            _ => None,
        }
    }
}

// =============================================================================
// Tensor Shape
// =============================================================================

/// A dimension in a tensor shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimension {
    /// Fixed dimension value (if known).
    pub dim_value: Option<i64>,
    /// Symbolic dimension name (if dynamic).
    pub dim_param: Option<String>,
}

impl Dimension {
    /// Creates a fixed dimension.
    pub fn fixed(value: i64) -> Self {
        Self {
            dim_value: Some(value),
            dim_param: None,
        }
    }

    /// Creates a dynamic dimension with a symbolic name.
    pub fn dynamic(name: &str) -> Self {
        Self {
            dim_value: None,
            dim_param: Some(name.to_string()),
        }
    }

    /// Returns the dimension value if it's fixed.
    pub fn value(&self) -> Option<i64> {
        self.dim_value
    }
}

/// Tensor shape information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShape {
    /// Dimensions of the tensor.
    pub dims: Vec<Dimension>,
}

impl TensorShape {
    /// Creates a new tensor shape from fixed dimensions.
    pub fn from_dims(dims: &[i64]) -> Self {
        Self {
            dims: dims.iter().map(|&d| Dimension::fixed(d)).collect(),
        }
    }

    /// Returns the shape as a vector of Option<i64>.
    pub fn to_vec(&self) -> Vec<Option<i64>> {
        self.dims.iter().map(|d| d.dim_value).collect()
    }
}

// =============================================================================
// Type Information
// =============================================================================

/// Tensor type information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorType {
    /// Element data type.
    pub elem_type: i32,
    /// Shape information.
    pub shape: Option<TensorShape>,
}

/// Type information for a value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeProto {
    /// Tensor type.
    pub tensor_type: Option<TensorType>,
}

// =============================================================================
// Value Information
// =============================================================================

/// Information about a graph input, output, or intermediate value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueInfo {
    /// Name of the value.
    pub name: String,
    /// Type information.
    pub r#type: Option<TypeProto>,
    /// Documentation string.
    pub doc_string: Option<String>,
}

// =============================================================================
// Tensor (Initializer)
// =============================================================================

/// A tensor constant (initializer/weight).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorProto {
    /// Name of the tensor.
    pub name: String,
    /// Dimensions (shape).
    pub dims: Vec<i64>,
    /// Data type.
    pub data_type: i32,
    /// Raw data bytes.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub raw_data: Vec<u8>,
    /// Float data (if not using raw_data).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub float_data: Vec<f32>,
    /// Int32 data.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub int32_data: Vec<i32>,
    /// Int64 data.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub int64_data: Vec<i64>,
    /// Double data.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub double_data: Vec<f64>,
    /// Documentation string.
    pub doc_string: Option<String>,
}

impl TensorProto {
    /// Creates a new float tensor.
    pub fn float(name: &str, dims: &[i64], data: Vec<f32>) -> Self {
        Self {
            name: name.to_string(),
            dims: dims.to_vec(),
            data_type: TensorDataType::Float as i32,
            raw_data: Vec::new(),
            float_data: data,
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
            doc_string: None,
        }
    }

    /// Returns the number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Extracts float data from the tensor.
    pub fn get_float_data(&self) -> Vec<f32> {
        if !self.float_data.is_empty() {
            return self.float_data.clone();
        }

        if !self.raw_data.is_empty() {
            // Parse raw data as f32
            let numel = self.numel();
            let mut result = vec![0.0f32; numel];
            let bytes_needed = numel * 4;
            if self.raw_data.len() >= bytes_needed {
                for (i, chunk) in self.raw_data[..bytes_needed].chunks_exact(4).enumerate() {
                    result[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            return result;
        }

        Vec::new()
    }

    /// Extracts int64 data from the tensor.
    pub fn get_int64_data(&self) -> Vec<i64> {
        if !self.int64_data.is_empty() {
            return self.int64_data.clone();
        }

        if !self.raw_data.is_empty() {
            let numel = self.numel();
            let mut result = vec![0i64; numel];
            let bytes_needed = numel * 8;
            if self.raw_data.len() >= bytes_needed {
                for (i, chunk) in self.raw_data[..bytes_needed].chunks_exact(8).enumerate() {
                    result[i] = i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]);
                }
            }
            return result;
        }

        Vec::new()
    }
}

// =============================================================================
// Attributes
// =============================================================================

/// Attribute type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AttributeType {
    /// Undefined attribute.
    Undefined = 0,
    /// Float value.
    Float = 1,
    /// Integer value.
    Int = 2,
    /// String value.
    String = 3,
    /// Tensor value.
    Tensor = 4,
    /// Graph value.
    Graph = 5,
    /// Sparse tensor value.
    SparseTensor = 11,
    /// Float array.
    Floats = 6,
    /// Integer array.
    Ints = 7,
    /// String array.
    Strings = 8,
    /// Tensor array.
    Tensors = 9,
    /// Graph array.
    Graphs = 10,
    /// Sparse tensor array.
    SparseTensors = 12,
}

/// An attribute of an ONNX operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeProto {
    /// Attribute name.
    pub name: String,
    /// Attribute type.
    pub r#type: i32,
    /// Float value.
    pub f: Option<f32>,
    /// Integer value.
    pub i: Option<i64>,
    /// String value.
    pub s: Option<Vec<u8>>,
    /// Tensor value.
    pub t: Option<TensorProto>,
    /// Float array.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub floats: Vec<f32>,
    /// Integer array.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub ints: Vec<i64>,
    /// String array.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub strings: Vec<Vec<u8>>,
    /// Documentation string.
    pub doc_string: Option<String>,
}

impl AttributeProto {
    /// Creates an integer attribute.
    pub fn int(name: &str, value: i64) -> Self {
        Self {
            name: name.to_string(),
            r#type: AttributeType::Int as i32,
            f: None,
            i: Some(value),
            s: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            doc_string: None,
        }
    }

    /// Creates a float attribute.
    pub fn float(name: &str, value: f32) -> Self {
        Self {
            name: name.to_string(),
            r#type: AttributeType::Float as i32,
            f: Some(value),
            i: None,
            s: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            doc_string: None,
        }
    }

    /// Creates an integer array attribute.
    pub fn ints(name: &str, values: Vec<i64>) -> Self {
        Self {
            name: name.to_string(),
            r#type: AttributeType::Ints as i32,
            f: None,
            i: None,
            s: None,
            t: None,
            floats: Vec::new(),
            ints: values,
            strings: Vec::new(),
            doc_string: None,
        }
    }

    /// Gets the string value.
    pub fn get_string(&self) -> Option<String> {
        self.s.as_ref().and_then(|bytes| String::from_utf8(bytes.clone()).ok())
    }
}

// =============================================================================
// Node (Operator)
// =============================================================================

/// A node in the ONNX computation graph (operator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProto {
    /// Input tensor names.
    pub input: Vec<String>,
    /// Output tensor names.
    pub output: Vec<String>,
    /// Node name (optional, for debugging).
    pub name: Option<String>,
    /// Operator type (e.g., "Conv", "Relu", "MatMul").
    pub op_type: String,
    /// ONNX domain (empty for default ONNX ops).
    pub domain: Option<String>,
    /// Operator attributes.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub attribute: Vec<AttributeProto>,
    /// Documentation string.
    pub doc_string: Option<String>,
}

impl NodeProto {
    /// Gets an attribute by name.
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeProto> {
        self.attribute.iter().find(|a| a.name == name)
    }

    /// Gets an integer attribute by name.
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.get_attribute(name).and_then(|a| a.i)
    }

    /// Gets a float attribute by name.
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.get_attribute(name).and_then(|a| a.f)
    }

    /// Gets an integer array attribute by name.
    pub fn get_ints(&self, name: &str) -> Option<&[i64]> {
        self.get_attribute(name).map(|a| a.ints.as_slice())
    }
}

// =============================================================================
// Graph
// =============================================================================

/// An ONNX computation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphProto {
    /// Nodes (operators) in the graph.
    pub node: Vec<NodeProto>,
    /// Graph name.
    pub name: Option<String>,
    /// Initializers (weights/constants).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub initializer: Vec<TensorProto>,
    /// Sparse initializers.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub sparse_initializer: Vec<TensorProto>,
    /// Graph inputs.
    pub input: Vec<ValueInfo>,
    /// Graph outputs.
    pub output: Vec<ValueInfo>,
    /// Intermediate value information.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub value_info: Vec<ValueInfo>,
    /// Documentation string.
    pub doc_string: Option<String>,
}

impl GraphProto {
    /// Gets an initializer by name.
    pub fn get_initializer(&self, name: &str) -> Option<&TensorProto> {
        self.initializer.iter().find(|i| i.name == name)
    }

    /// Returns a map of initializer name to tensor.
    pub fn initializer_map(&self) -> HashMap<String, &TensorProto> {
        self.initializer.iter().map(|t| (t.name.clone(), t)).collect()
    }
}

// =============================================================================
// Opset Import
// =============================================================================

/// Opset import declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorSetIdProto {
    /// Domain (empty for default ONNX ops).
    pub domain: Option<String>,
    /// Opset version.
    pub version: i64,
}

// =============================================================================
// Model
// =============================================================================

/// An ONNX model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProto {
    /// ONNX IR version.
    pub ir_version: i64,
    /// Opset imports.
    pub opset_import: Vec<OperatorSetIdProto>,
    /// Producer name.
    pub producer_name: Option<String>,
    /// Producer version.
    pub producer_version: Option<String>,
    /// Domain.
    pub domain: Option<String>,
    /// Model version.
    pub model_version: Option<i64>,
    /// Documentation string.
    pub doc_string: Option<String>,
    /// The computation graph.
    pub graph: Option<GraphProto>,
    /// Metadata properties.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub metadata_props: Vec<StringStringEntry>,
}

/// String-string key-value pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringStringEntry {
    /// Key.
    pub key: String,
    /// Value.
    pub value: String,
}

impl ModelProto {
    /// Gets the default opset version.
    pub fn opset_version(&self) -> i64 {
        self.opset_import
            .iter()
            .find(|o| o.domain.is_none() || o.domain.as_deref() == Some(""))
            .map(|o| o.version)
            .unwrap_or(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_type() {
        assert_eq!(TensorDataType::Float.size_bytes(), 4);
        assert_eq!(TensorDataType::Double.size_bytes(), 8);
        assert_eq!(TensorDataType::Int64.size_bytes(), 8);
        assert_eq!(TensorDataType::from_i32(1), Some(TensorDataType::Float));
    }

    #[test]
    fn test_tensor_proto() {
        let tensor = TensorProto::float("weight", &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.get_float_data().len(), 6);
    }

    #[test]
    fn test_attribute_proto() {
        let attr = AttributeProto::int("kernel_size", 3);
        assert_eq!(attr.i, Some(3));

        let attr = AttributeProto::ints("pads", vec![1, 1, 1, 1]);
        assert_eq!(attr.ints.len(), 4);
    }
}
