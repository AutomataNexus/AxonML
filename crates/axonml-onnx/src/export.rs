//! ONNX Export
//!
//! # File
//! `crates/axonml-onnx/src/export.rs`
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

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::SUPPORTED_OPSET_VERSION;
use crate::error::{OnnxError, OnnxResult};
use crate::proto::{
    AttributeProto, Dimension, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorDataType, TensorProto, TensorShape, TensorType, TypeProto, ValueInfo,
};
use axonml_tensor::Tensor;

// =============================================================================
// Public API
// =============================================================================

/// Exports a model to ONNX format.
///
/// # Arguments
/// * `exporter` - The model exporter containing graph information
/// * `path` - Output path for the .onnx file
///
/// # Example
/// ```ignore
/// use axonml_onnx::export::OnnxExporter;
///
/// let mut exporter = OnnxExporter::new("my_model");
/// exporter.add_input("input", &[1, 3, 224, 224], TensorDataType::Float);
/// exporter.add_node("Relu", &["input"], &["output"], HashMap::new());
/// exporter.add_output("output", &[1, 3, 224, 224], TensorDataType::Float);
/// exporter.export("model.onnx")?;
/// ```
pub fn export_onnx<P: AsRef<Path>>(exporter: &OnnxExporter, path: P) -> OnnxResult<()> {
    let proto = exporter.to_proto()?;
    let bytes = serialize_model(&proto)?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    Ok(())
}

/// Exports a model to ONNX bytes.
pub fn export_onnx_bytes(exporter: &OnnxExporter) -> OnnxResult<Vec<u8>> {
    let proto = exporter.to_proto()?;
    serialize_model(&proto)
}

// =============================================================================
// ONNX Exporter
// =============================================================================

/// Builder for constructing ONNX models for export.
#[derive(Debug, Clone)]
pub struct OnnxExporter {
    /// Model name.
    pub name: String,
    /// Producer name.
    pub producer_name: String,
    /// Producer version.
    pub producer_version: String,
    /// Model inputs.
    inputs: Vec<ValueInfo>,
    /// Model outputs.
    outputs: Vec<ValueInfo>,
    /// Graph nodes.
    nodes: Vec<NodeProto>,
    /// Initializers (weights).
    initializers: Vec<TensorProto>,
    /// Optional documentation.
    doc_string: Option<String>,
}

impl OnnxExporter {
    /// Creates a new ONNX exporter with the given model name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            producer_name: "Axonml".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            initializers: Vec::new(),
            doc_string: None,
        }
    }

    /// Sets the producer name.
    pub fn with_producer(mut self, name: &str, version: &str) -> Self {
        self.producer_name = name.to_string();
        self.producer_version = version.to_string();
        self
    }

    /// Sets the documentation string.
    pub fn with_doc_string(mut self, doc: &str) -> Self {
        self.doc_string = Some(doc.to_string());
        self
    }

    /// Adds an input to the model.
    pub fn add_input(&mut self, name: &str, shape: &[i64], dtype: TensorDataType) {
        let value_info = create_value_info(name, shape, dtype);
        self.inputs.push(value_info);
    }

    /// Adds an output to the model.
    pub fn add_output(&mut self, name: &str, shape: &[i64], dtype: TensorDataType) {
        let value_info = create_value_info(name, shape, dtype);
        self.outputs.push(value_info);
    }

    /// Adds a node (operator) to the graph.
    pub fn add_node(
        &mut self,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attributes: HashMap<String, AttributeValue>,
    ) {
        let node = NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: None,
            op_type: op_type.to_string(),
            domain: None,
            attribute: attributes
                .into_iter()
                .map(|(k, v)| create_attribute(&k, v))
                .collect(),
            doc_string: None,
        };
        self.nodes.push(node);
    }

    /// Adds a named node to the graph.
    pub fn add_named_node(
        &mut self,
        name: &str,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attributes: HashMap<String, AttributeValue>,
    ) {
        let node = NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: Some(name.to_string()),
            op_type: op_type.to_string(),
            domain: None,
            attribute: attributes
                .into_iter()
                .map(|(k, v)| create_attribute(&k, v))
                .collect(),
            doc_string: None,
        };
        self.nodes.push(node);
    }

    /// Adds an initializer (weight tensor) to the model.
    pub fn add_initializer(&mut self, name: &str, tensor: &Tensor<f32>) {
        let proto = tensor_to_proto(name, tensor);
        self.initializers.push(proto);
    }

    /// Adds an initializer with explicit data.
    pub fn add_initializer_data(&mut self, name: &str, shape: &[i64], data: &[f32]) {
        let proto = TensorProto {
            name: name.to_string(),
            dims: shape.to_vec(),
            data_type: TensorDataType::Float as i32,
            float_data: data.to_vec(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
            raw_data: Vec::new(),
            doc_string: None,
        };
        self.initializers.push(proto);
    }

    /// Converts the exporter state to a ModelProto.
    pub fn to_proto(&self) -> OnnxResult<ModelProto> {
        if self.inputs.is_empty() {
            return Err(OnnxError::Export("Model has no inputs".to_string()));
        }
        if self.outputs.is_empty() {
            return Err(OnnxError::Export("Model has no outputs".to_string()));
        }

        let graph = GraphProto {
            node: self.nodes.clone(),
            name: Some(self.name.clone()),
            initializer: self.initializers.clone(),
            sparse_initializer: Vec::new(),
            input: self.inputs.clone(),
            output: self.outputs.clone(),
            value_info: Vec::new(),
            doc_string: self.doc_string.clone(),
        };

        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: None,
                version: SUPPORTED_OPSET_VERSION,
            }],
            producer_name: Some(self.producer_name.clone()),
            producer_version: Some(self.producer_version.clone()),
            domain: None,
            model_version: Some(1),
            doc_string: self.doc_string.clone(),
            graph: Some(graph),
            metadata_props: Vec::new(),
        };

        Ok(model)
    }

    /// Exports the model to a file.
    pub fn export<P: AsRef<Path>>(&self, path: P) -> OnnxResult<()> {
        export_onnx(self, path)
    }

    /// Exports the model to bytes.
    pub fn to_bytes(&self) -> OnnxResult<Vec<u8>> {
        export_onnx_bytes(self)
    }
}

// =============================================================================
// Attribute Values
// =============================================================================

/// Attribute values for ONNX nodes.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// Float value.
    Float(f32),
    /// Integer value.
    Int(i64),
    /// String value.
    String(String),
    /// Tensor value.
    Tensor(TensorProto),
    /// List of floats.
    Floats(Vec<f32>),
    /// List of integers.
    Ints(Vec<i64>),
    /// List of strings.
    Strings(Vec<String>),
}

fn create_attribute(name: &str, value: AttributeValue) -> AttributeProto {
    match value {
        AttributeValue::Float(f) => AttributeProto {
            name: name.to_string(),
            r#type: 1,
            f: Some(f),
            i: None,
            s: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            doc_string: None,
        },
        AttributeValue::Int(i) => AttributeProto {
            name: name.to_string(),
            r#type: 2,
            f: None,
            i: Some(i),
            s: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            doc_string: None,
        },
        AttributeValue::String(s) => AttributeProto {
            name: name.to_string(),
            r#type: 3,
            f: None,
            i: None,
            s: Some(s.into_bytes()),
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            doc_string: None,
        },
        AttributeValue::Tensor(t) => AttributeProto {
            name: name.to_string(),
            r#type: 4,
            f: None,
            i: None,
            s: None,
            t: Some(t),
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            doc_string: None,
        },
        AttributeValue::Floats(fs) => AttributeProto {
            name: name.to_string(),
            r#type: 6,
            f: None,
            i: None,
            s: None,
            t: None,
            floats: fs,
            ints: Vec::new(),
            strings: Vec::new(),
            doc_string: None,
        },
        AttributeValue::Ints(is) => AttributeProto {
            name: name.to_string(),
            r#type: 7,
            f: None,
            i: None,
            s: None,
            t: None,
            floats: Vec::new(),
            ints: is,
            strings: Vec::new(),
            doc_string: None,
        },
        AttributeValue::Strings(ss) => AttributeProto {
            name: name.to_string(),
            r#type: 8,
            f: None,
            i: None,
            s: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: ss.into_iter().map(|s| s.into_bytes()).collect(),
            doc_string: None,
        },
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_value_info(name: &str, shape: &[i64], dtype: TensorDataType) -> ValueInfo {
    let dims: Vec<Dimension> = shape
        .iter()
        .map(|&d| Dimension {
            dim_value: Some(d),
            dim_param: None,
        })
        .collect();

    ValueInfo {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorType {
                elem_type: dtype as i32,
                shape: Some(TensorShape { dims }),
            }),
        }),
        doc_string: None,
    }
}

fn tensor_to_proto(name: &str, tensor: &Tensor<f32>) -> TensorProto {
    let dims: Vec<i64> = tensor.shape().iter().map(|&d| d as i64).collect();
    let data = tensor.to_vec();

    TensorProto {
        name: name.to_string(),
        dims,
        data_type: TensorDataType::Float as i32,
        float_data: data,
        int32_data: Vec::new(),
        int64_data: Vec::new(),
        double_data: Vec::new(),
        raw_data: Vec::new(),
        doc_string: None,
    }
}

/// Serializes a ModelProto to standard ONNX binary protobuf format.
///
/// Uses prost::Message::encode for proper protobuf wire-format output,
/// producing files compatible with ONNX Runtime, TensorRT, and other
/// standard ONNX consumers.
fn serialize_model(proto: &ModelProto) -> OnnxResult<Vec<u8>> {
    use prost::Message;

    // Convert our serde ModelProto to prost-compatible structs
    let raw = model_to_prost(proto);

    let mut buf = Vec::with_capacity(raw.encoded_len());
    raw.encode(&mut buf)
        .map_err(|e| OnnxError::Export(format!("Protobuf encode error: {}", e)))?;
    Ok(buf)
}

// =============================================================================
// Prost Message Structs for ONNX Binary Encoding
// =============================================================================

// These mirror the ONNX protobuf schema with correct field tags.
// Used exclusively for binary serialization via prost::Message::encode.

#[derive(Clone, PartialEq, prost::Message)]
struct ProstModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<ProstOpsetImport>,
    #[prost(string, optional, tag = "2")]
    producer_name: Option<String>,
    #[prost(string, optional, tag = "3")]
    producer_version: Option<String>,
    #[prost(string, optional, tag = "4")]
    domain: Option<String>,
    #[prost(int64, optional, tag = "5")]
    model_version: Option<i64>,
    #[prost(string, optional, tag = "6")]
    doc_string: Option<String>,
    #[prost(message, optional, tag = "7")]
    graph: Option<ProstGraphProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstOpsetImport {
    #[prost(string, optional, tag = "1")]
    domain: Option<String>,
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstGraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<ProstNodeProto>,
    #[prost(string, optional, tag = "2")]
    name: Option<String>,
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<ProstTensorProto>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ProstValueInfo>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ProstValueInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstNodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, optional, tag = "3")]
    name: Option<String>,
    #[prost(string, optional, tag = "4")]
    op_type: Option<String>,
    #[prost(string, optional, tag = "7")]
    domain: Option<String>,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<ProstAttributeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstAttributeProto {
    #[prost(string, optional, tag = "1")]
    name: Option<String>,
    #[prost(int32, optional, tag = "20")]
    r#type: Option<i32>,
    #[prost(float, optional, tag = "2")]
    f: Option<f32>,
    #[prost(int64, optional, tag = "3")]
    i: Option<i64>,
    #[prost(bytes, optional, tag = "4")]
    s: Option<Vec<u8>>,
    #[prost(float, repeated, tag = "7")]
    floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    ints: Vec<i64>,
    #[prost(bytes, repeated, tag = "9")]
    strings: Vec<Vec<u8>>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstTensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, optional, tag = "2")]
    data_type: Option<i32>,
    #[prost(float, repeated, tag = "4")]
    float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    int32_data: Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    int64_data: Vec<i64>,
    #[prost(string, optional, tag = "8")]
    name: Option<String>,
    #[prost(bytes, optional, tag = "13")]
    raw_data: Option<Vec<u8>>,
    #[prost(double, repeated, tag = "10")]
    double_data: Vec<f64>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstValueInfo {
    #[prost(string, optional, tag = "1")]
    name: Option<String>,
    #[prost(message, optional, tag = "2")]
    r#type: Option<ProstTypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstTypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<ProstTensorTypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstTensorTypeProto {
    #[prost(int32, tag = "1")]
    elem_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<ProstTensorShapeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstTensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<ProstDimension>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ProstDimension {
    #[prost(int64, optional, tag = "1")]
    dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    dim_param: Option<String>,
}

// =============================================================================
// Conversion: Serde ModelProto → Prost Message
// =============================================================================

fn model_to_prost(proto: &ModelProto) -> ProstModelProto {
    ProstModelProto {
        ir_version: proto.ir_version,
        opset_import: proto
            .opset_import
            .iter()
            .map(|o| ProstOpsetImport {
                domain: o.domain.clone(),
                version: o.version,
            })
            .collect(),
        producer_name: proto.producer_name.clone(),
        producer_version: proto.producer_version.clone(),
        domain: proto.domain.clone(),
        model_version: proto.model_version,
        doc_string: proto.doc_string.clone(),
        graph: proto.graph.as_ref().map(graph_to_prost),
    }
}

fn graph_to_prost(graph: &GraphProto) -> ProstGraphProto {
    ProstGraphProto {
        name: graph.name.clone(),
        node: graph.node.iter().map(node_to_prost).collect(),
        initializer: graph.initializer.iter().map(tensor_to_prost_msg).collect(),
        input: graph.input.iter().map(value_info_to_prost).collect(),
        output: graph.output.iter().map(value_info_to_prost).collect(),
    }
}

fn node_to_prost(node: &NodeProto) -> ProstNodeProto {
    ProstNodeProto {
        input: node.input.clone(),
        output: node.output.clone(),
        name: node.name.clone(),
        op_type: Some(node.op_type.clone()),
        domain: node.domain.clone(),
        attribute: node.attribute.iter().map(attr_to_prost).collect(),
    }
}

fn attr_to_prost(attr: &AttributeProto) -> ProstAttributeProto {
    ProstAttributeProto {
        name: Some(attr.name.clone()),
        r#type: Some(attr.r#type),
        f: attr.f,
        i: attr.i,
        s: attr.s.clone(),
        floats: attr.floats.clone(),
        ints: attr.ints.clone(),
        strings: attr.strings.clone(),
    }
}

fn tensor_to_prost_msg(tensor: &TensorProto) -> ProstTensorProto {
    ProstTensorProto {
        dims: tensor.dims.clone(),
        data_type: Some(tensor.data_type),
        float_data: tensor.float_data.clone(),
        int32_data: tensor.int32_data.clone(),
        int64_data: tensor.int64_data.clone(),
        name: Some(tensor.name.clone()),
        raw_data: if tensor.raw_data.is_empty() {
            None
        } else {
            Some(tensor.raw_data.clone())
        },
        double_data: tensor.double_data.clone(),
    }
}

fn value_info_to_prost(vi: &ValueInfo) -> ProstValueInfo {
    ProstValueInfo {
        name: Some(vi.name.clone()),
        r#type: vi.r#type.as_ref().map(|t| ProstTypeProto {
            tensor_type: t.tensor_type.as_ref().map(|tt| ProstTensorTypeProto {
                elem_type: tt.elem_type,
                shape: tt.shape.as_ref().map(|s| ProstTensorShapeProto {
                    dim: s
                        .dims
                        .iter()
                        .map(|d| ProstDimension {
                            dim_value: d.dim_value,
                            dim_param: d.dim_param.clone(),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

// =============================================================================
// Module-Level Export Helpers
// =============================================================================

/// Helper to export a simple feedforward network.
pub fn export_feedforward(
    name: &str,
    layers: &[(usize, usize)], // (in_features, out_features)
    weights: &[(&str, &Tensor<f32>)],
    biases: &[(&str, &Tensor<f32>)],
) -> OnnxResult<OnnxExporter> {
    let mut exporter = OnnxExporter::new(name);

    if layers.is_empty() {
        return Err(OnnxError::Export("No layers specified".to_string()));
    }

    // Add input
    let (first_in, _) = layers[0];
    exporter.add_input("input", &[1, first_in as i64], TensorDataType::Float);

    // Add layers
    let mut current_input = "input".to_string();

    for (i, ((_in_f, _out_f), ((w_name, w_tensor), (b_name, b_tensor)))) in layers
        .iter()
        .zip(weights.iter().zip(biases.iter()))
        .enumerate()
    {
        let output_name = if i == layers.len() - 1 {
            "output".to_string()
        } else {
            format!("layer_{}_out", i)
        };

        // Add weight and bias initializers
        exporter.add_initializer(w_name, w_tensor);
        exporter.add_initializer(b_name, b_tensor);

        // Add Gemm node (General Matrix Multiply)
        let gemm_output = format!("gemm_{}", i);
        let mut attrs = HashMap::new();
        attrs.insert("alpha".to_string(), AttributeValue::Float(1.0));
        attrs.insert("beta".to_string(), AttributeValue::Float(1.0));
        attrs.insert("transB".to_string(), AttributeValue::Int(1));

        exporter.add_node(
            "Gemm",
            &[&current_input, w_name, b_name],
            &[&gemm_output],
            attrs,
        );

        // Add activation (ReLU) except for last layer
        if i < layers.len() - 1 {
            exporter.add_node("Relu", &[&gemm_output], &[&output_name], HashMap::new());
            current_input = output_name;
        } else {
            // Rename gemm output to final output
            exporter.add_node("Identity", &[&gemm_output], &[&output_name], HashMap::new());
        }
    }

    // Add output
    let (_, last_out) = layers[layers.len() - 1];
    exporter.add_output("output", &[1, last_out as i64], TensorDataType::Float);

    Ok(exporter)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exporter_basic() {
        let mut exporter = OnnxExporter::new("test_model");
        exporter.add_input("input", &[1, 10], TensorDataType::Float);
        exporter.add_output("output", &[1, 10], TensorDataType::Float);
        exporter.add_node("Relu", &["input"], &["output"], HashMap::new());

        let proto = exporter.to_proto().unwrap();
        assert_eq!(proto.graph.as_ref().unwrap().node.len(), 1);
    }

    #[test]
    fn test_exporter_with_attributes() {
        let mut exporter = OnnxExporter::new("test_model");
        exporter.add_input("input", &[1, 10], TensorDataType::Float);
        exporter.add_output("output", &[1, 10], TensorDataType::Float);

        let mut attrs = HashMap::new();
        attrs.insert("alpha".to_string(), AttributeValue::Float(0.01));
        exporter.add_node("LeakyRelu", &["input"], &["output"], attrs);

        let proto = exporter.to_proto().unwrap();
        let node = &proto.graph.as_ref().unwrap().node[0];
        assert_eq!(node.attribute.len(), 1);
    }

    #[test]
    fn test_exporter_no_inputs_fails() {
        let exporter = OnnxExporter::new("test_model");
        assert!(exporter.to_proto().is_err());
    }

    #[test]
    fn test_attribute_value_creation() {
        let attr = create_attribute("test_float", AttributeValue::Float(1.5));
        assert_eq!(attr.name, "test_float");
        assert_eq!(attr.f, Some(1.5));

        let attr = create_attribute("test_int", AttributeValue::Int(42));
        assert_eq!(attr.i, Some(42));

        let attr = create_attribute("test_ints", AttributeValue::Ints(vec![1, 2, 3]));
        assert_eq!(attr.ints, vec![1, 2, 3]);
    }

    #[test]
    fn test_serialize_produces_protobuf_binary() {
        let mut exporter = OnnxExporter::new("test_binary");
        exporter.add_input("x", &[1, 4], TensorDataType::Float);
        exporter.add_output("y", &[1, 4], TensorDataType::Float);
        exporter.add_node("Relu", &["x"], &["y"], HashMap::new());

        let proto = exporter.to_proto().unwrap();
        let bytes = serialize_model(&proto).unwrap();

        // Protobuf binary should NOT start with '{' (that would be JSON)
        assert_ne!(
            bytes.first(),
            Some(&b'{'),
            "Output should be protobuf binary, not JSON"
        );
        // Should be non-empty
        assert!(
            bytes.len() > 10,
            "Protobuf output should have substantial content"
        );
    }

    #[test]
    fn test_export_roundtrip_binary() {
        // Export a model to binary protobuf bytes
        let mut exporter = OnnxExporter::new("roundtrip_model");
        exporter.add_input("input", &[1, 3, 8, 8], TensorDataType::Float);
        exporter.add_output("output", &[1, 3, 8, 8], TensorDataType::Float);
        exporter.add_node("Relu", &["input"], &["output"], HashMap::new());

        let bytes = export_onnx_bytes(&exporter).unwrap();

        // Re-import using the parser
        let model = crate::parser::import_onnx_bytes(&bytes).unwrap();

        // Verify the model structure survived the roundtrip
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.outputs.len(), 1);
        assert_eq!(model.inputs[0].name, "input");
        assert_eq!(model.outputs[0], "output");
    }

    #[test]
    fn test_export_to_file_and_reimport() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let mut exporter = OnnxExporter::new("file_test");
        exporter.add_input("x", &[2, 10], TensorDataType::Float);
        exporter.add_output("y", &[2, 5], TensorDataType::Float);

        let mut attrs = HashMap::new();
        attrs.insert("alpha".to_string(), AttributeValue::Float(0.01));
        exporter.add_node("LeakyRelu", &["x"], &["y"], attrs);

        export_onnx(&exporter, path).unwrap();

        // File should exist and be non-empty
        let metadata = std::fs::metadata(path).unwrap();
        assert!(metadata.len() > 0);

        // Re-import
        let model = crate::parser::import_onnx(path).unwrap();
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.outputs.len(), 1);
        assert_eq!(model.inputs[0].name, "x");
        assert_eq!(model.outputs[0], "y");
    }
}
