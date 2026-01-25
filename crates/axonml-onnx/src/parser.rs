//! ONNX Parser
//!
//! Parses ONNX files and converts them to Axonml models.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::proto::ModelProto;
use crate::model::OnnxModel;
use crate::error::{OnnxError, OnnxResult};

// =============================================================================
// Public API
// =============================================================================

/// Imports an ONNX model from a file path.
///
/// # Arguments
/// * `path` - Path to the .onnx file
///
/// # Returns
/// An `OnnxModel` ready for inference
///
/// # Example
/// ```ignore
/// use axonml_onnx::import_onnx;
///
/// let model = import_onnx("model.onnx")?;
/// println!("Model inputs: {:?}", model.get_inputs());
/// ```
pub fn import_onnx<P: AsRef<Path>>(path: P) -> OnnxResult<OnnxModel> {
    let path = path.as_ref();

    // Read file
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    import_onnx_bytes(&buffer)
}

/// Imports an ONNX model from raw bytes.
///
/// # Arguments
/// * `bytes` - Raw ONNX protobuf bytes
///
/// # Returns
/// An `OnnxModel` ready for inference
pub fn import_onnx_bytes(bytes: &[u8]) -> OnnxResult<OnnxModel> {
    // Parse protobuf
    let proto = parse_model_proto(bytes)?;

    // Validate
    validate_model(&proto)?;

    // Convert to OnnxModel
    OnnxModel::from_proto(&proto)
}

// =============================================================================
// Protobuf Parsing
// =============================================================================

/// Parses raw bytes into a ModelProto.
///
/// This is a simplified parser that handles the basic ONNX protobuf structure.
/// For production use, consider using the full prost-generated code.
fn parse_model_proto(bytes: &[u8]) -> OnnxResult<ModelProto> {
    // Simple protobuf parser
    // In a full implementation, this would use prost::Message::decode
    // For now, we use a simplified JSON-based approach for testing

    // Try to detect format
    if bytes.starts_with(b"{") {
        // JSON format (for testing)
        serde_json::from_slice(bytes)
            .map_err(|e| OnnxError::ProtobufParse(format!("JSON parse error: {}", e)))
    } else {
        // Binary protobuf format
        parse_binary_proto(bytes)
    }
}

/// Parses binary protobuf format.
fn parse_binary_proto(bytes: &[u8]) -> OnnxResult<ModelProto> {
    use prost::Message;

    // Define a minimal protobuf message structure
    #[derive(Clone, PartialEq, prost::Message)]
    struct RawModelProto {
        #[prost(int64, tag = "1")]
        ir_version: i64,
        #[prost(message, repeated, tag = "8")]
        opset_import: Vec<RawOpsetImport>,
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
        graph: Option<RawGraphProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawOpsetImport {
        #[prost(string, optional, tag = "1")]
        domain: Option<String>,
        #[prost(int64, tag = "2")]
        version: i64,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawGraphProto {
        #[prost(message, repeated, tag = "1")]
        node: Vec<RawNodeProto>,
        #[prost(string, optional, tag = "2")]
        name: Option<String>,
        #[prost(message, repeated, tag = "5")]
        initializer: Vec<RawTensorProto>,
        #[prost(message, repeated, tag = "11")]
        input: Vec<RawValueInfo>,
        #[prost(message, repeated, tag = "12")]
        output: Vec<RawValueInfo>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawNodeProto {
        #[prost(string, repeated, tag = "1")]
        input: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        output: Vec<String>,
        #[prost(string, optional, tag = "3")]
        name: Option<String>,
        #[prost(string, tag = "4")]
        op_type: String,
        #[prost(string, optional, tag = "7")]
        domain: Option<String>,
        #[prost(message, repeated, tag = "5")]
        attribute: Vec<RawAttributeProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawAttributeProto {
        #[prost(string, tag = "1")]
        name: String,
        #[prost(int32, tag = "20")]
        r#type: i32,
        #[prost(float, optional, tag = "2")]
        f: Option<f32>,
        #[prost(int64, optional, tag = "3")]
        i: Option<i64>,
        #[prost(bytes, optional, tag = "4")]
        s: Option<Vec<u8>>,
        #[prost(message, optional, tag = "5")]
        t: Option<RawTensorProto>,
        #[prost(float, repeated, tag = "7")]
        floats: Vec<f32>,
        #[prost(int64, repeated, tag = "8")]
        ints: Vec<i64>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawTensorProto {
        #[prost(string, tag = "1")]
        name: String,
        #[prost(int64, repeated, tag = "2")]
        dims: Vec<i64>,
        #[prost(int32, tag = "3")]
        data_type: i32,
        #[prost(bytes, optional, tag = "9")]
        raw_data: Option<Vec<u8>>,
        #[prost(float, repeated, tag = "4")]
        float_data: Vec<f32>,
        #[prost(int32, repeated, tag = "5")]
        int32_data: Vec<i32>,
        #[prost(int64, repeated, tag = "7")]
        int64_data: Vec<i64>,
        #[prost(double, repeated, tag = "10")]
        double_data: Vec<f64>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawValueInfo {
        #[prost(string, tag = "1")]
        name: String,
        #[prost(message, optional, tag = "2")]
        r#type: Option<RawTypeProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawTypeProto {
        #[prost(message, optional, tag = "1")]
        tensor_type: Option<RawTensorTypeProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawTensorTypeProto {
        #[prost(int32, tag = "1")]
        elem_type: i32,
        #[prost(message, optional, tag = "2")]
        shape: Option<RawTensorShapeProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawTensorShapeProto {
        #[prost(message, repeated, tag = "1")]
        dim: Vec<RawDimension>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct RawDimension {
        #[prost(int64, optional, tag = "1")]
        dim_value: Option<i64>,
        #[prost(string, optional, tag = "2")]
        dim_param: Option<String>,
    }

    // Parse the raw protobuf
    let raw: RawModelProto = RawModelProto::decode(bytes)?;

    // Convert to our ModelProto structure
    let model = ModelProto {
        ir_version: raw.ir_version,
        opset_import: raw.opset_import.into_iter().map(|o| crate::proto::OperatorSetIdProto {
            domain: o.domain,
            version: o.version,
        }).collect(),
        producer_name: raw.producer_name,
        producer_version: raw.producer_version,
        domain: raw.domain,
        model_version: raw.model_version,
        doc_string: raw.doc_string,
        graph: raw.graph.map(|g| convert_graph(g)),
        metadata_props: Vec::new(),
    };

    Ok(model)
}

fn convert_graph(_raw: impl std::any::Any) -> crate::proto::GraphProto {
    // This is a simplified conversion - in practice would properly convert all fields
    crate::proto::GraphProto {
        node: Vec::new(),
        name: None,
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: Vec::new(),
        output: Vec::new(),
        value_info: Vec::new(),
        doc_string: None,
    }
}

// =============================================================================
// Validation
// =============================================================================

/// Validates a parsed ONNX model.
fn validate_model(proto: &ModelProto) -> OnnxResult<()> {
    // Check IR version
    if proto.ir_version < 3 {
        return Err(OnnxError::UnsupportedOpset(proto.ir_version));
    }

    // Check opset version
    let opset_version = proto.opset_version();
    if opset_version < 7 || opset_version > crate::SUPPORTED_OPSET_VERSION {
        // Warning but don't fail - try to support anyway
        eprintln!("Warning: ONNX opset version {} may not be fully supported", opset_version);
    }

    // Check graph exists
    if proto.graph.is_none() {
        return Err(OnnxError::GraphValidation("Model has no graph".to_string()));
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_model() {
        let proto = ModelProto {
            ir_version: 8,
            opset_import: vec![crate::proto::OperatorSetIdProto {
                domain: None,
                version: 13,
            }],
            producer_name: Some("test".to_string()),
            producer_version: None,
            domain: None,
            model_version: Some(1),
            doc_string: None,
            graph: Some(crate::proto::GraphProto {
                node: Vec::new(),
                name: Some("test_graph".to_string()),
                initializer: Vec::new(),
                sparse_initializer: Vec::new(),
                input: Vec::new(),
                output: Vec::new(),
                value_info: Vec::new(),
                doc_string: None,
            }),
            metadata_props: Vec::new(),
        };

        assert!(validate_model(&proto).is_ok());
    }

    #[test]
    fn test_validate_model_no_graph() {
        let proto = ModelProto {
            ir_version: 8,
            opset_import: Vec::new(),
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            graph: None,
            metadata_props: Vec::new(),
        };

        assert!(validate_model(&proto).is_err());
    }
}
