//! ONNX Model Representation
//!
//! Provides the `OnnxModel` struct for representing imported ONNX models
//! and executing inference with Axonml tensors.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;
use axonml_tensor::Tensor;
use axonml_tensor::creation::zeros;
use crate::proto::{GraphProto, ModelProto, TensorProto};
use crate::operators::OnnxOperator;
use crate::error::{OnnxError, OnnxResult};

// =============================================================================
// ONNX Model
// =============================================================================

/// A loaded ONNX model ready for inference.
#[derive(Debug)]
pub struct OnnxModel {
    /// Model name.
    pub name: String,
    /// Opset version.
    pub opset_version: i64,
    /// Input names and shapes.
    pub inputs: Vec<ModelInput>,
    /// Output names.
    pub outputs: Vec<String>,
    /// Operators in topological order.
    pub operators: Vec<CompiledOp>,
    /// Initializers (weights).
    pub initializers: HashMap<String, Tensor<f32>>,
    /// Producer name.
    pub producer: Option<String>,
}

/// Model input specification.
#[derive(Debug, Clone)]
pub struct ModelInput {
    /// Input name.
    pub name: String,
    /// Input shape (None for dynamic dimensions).
    pub shape: Vec<Option<i64>>,
    /// Data type.
    pub dtype: String,
}

/// A compiled operator ready for execution.
#[derive(Debug)]
pub struct CompiledOp {
    /// Operator type.
    pub op_type: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Operator implementation.
    pub operator: Box<dyn OnnxOperator>,
}

impl OnnxModel {
    /// Creates a new ONNX model from a parsed model proto.
    pub fn from_proto(proto: &ModelProto) -> OnnxResult<Self> {
        let graph = proto.graph.as_ref()
            .ok_or_else(|| OnnxError::GraphValidation("Model has no graph".to_string()))?;

        // Extract inputs
        let inputs = extract_inputs(graph)?;

        // Extract outputs
        let outputs: Vec<String> = graph.output.iter()
            .map(|o| o.name.clone())
            .collect();

        // Load initializers (weights)
        let initializers = load_initializers(graph)?;

        // Compile operators
        let operators = compile_operators(graph, proto.opset_version())?;

        Ok(Self {
            name: graph.name.clone().unwrap_or_else(|| "model".to_string()),
            opset_version: proto.opset_version(),
            inputs,
            outputs,
            operators,
            initializers,
            producer: proto.producer_name.clone(),
        })
    }

    /// Runs inference on the model with the given inputs.
    pub fn forward(&self, inputs: HashMap<String, Tensor<f32>>) -> OnnxResult<HashMap<String, Tensor<f32>>> {
        let mut values: HashMap<String, Tensor<f32>> = HashMap::new();

        // Add inputs
        for (name, tensor) in inputs {
            values.insert(name, tensor);
        }

        // Add initializers
        for (name, tensor) in &self.initializers {
            values.insert(name.clone(), tensor.clone());
        }

        // Execute operators in order
        for op in &self.operators {
            // Gather inputs
            let op_inputs: Vec<Option<&Tensor<f32>>> = op.inputs.iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        values.get(name)
                    }
                })
                .collect();

            // Execute operator
            let outputs = op.operator.execute(&op_inputs)?;

            // Store outputs
            for (name, tensor) in op.outputs.iter().zip(outputs) {
                values.insert(name.clone(), tensor);
            }
        }

        // Collect outputs
        let mut result = HashMap::new();
        for name in &self.outputs {
            if let Some(tensor) = values.remove(name) {
                result.insert(name.clone(), tensor);
            }
        }

        Ok(result)
    }

    /// Returns the input specifications.
    pub fn get_inputs(&self) -> &[ModelInput] {
        &self.inputs
    }

    /// Returns the output names.
    pub fn get_outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Returns the number of parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.initializers.values()
            .map(|t| t.numel())
            .sum()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn extract_inputs(graph: &GraphProto) -> OnnxResult<Vec<ModelInput>> {
    let initializer_names: std::collections::HashSet<_> = graph.initializer
        .iter()
        .map(|i| i.name.as_str())
        .collect();

    let mut inputs = Vec::new();

    for input in &graph.input {
        // Skip initializers (they're weights, not inputs)
        if initializer_names.contains(input.name.as_str()) {
            continue;
        }

        let shape = input.r#type.as_ref()
            .and_then(|t| t.tensor_type.as_ref())
            .and_then(|tt| tt.shape.as_ref())
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let dtype = input.r#type.as_ref()
            .and_then(|t| t.tensor_type.as_ref())
            .map(|tt| format!("{:?}", crate::proto::TensorDataType::from_i32(tt.elem_type)))
            .unwrap_or_else(|| "float32".to_string());

        inputs.push(ModelInput {
            name: input.name.clone(),
            shape,
            dtype,
        });
    }

    Ok(inputs)
}

fn load_initializers(graph: &GraphProto) -> OnnxResult<HashMap<String, Tensor<f32>>> {
    let mut initializers = HashMap::new();

    for init in &graph.initializer {
        let tensor = tensor_from_proto(init)?;
        initializers.insert(init.name.clone(), tensor);
    }

    Ok(initializers)
}

fn tensor_from_proto(proto: &TensorProto) -> OnnxResult<Tensor<f32>> {
    let shape: Vec<usize> = proto.dims.iter().map(|&d| d as usize).collect();
    let numel = if shape.is_empty() { 1 } else { shape.iter().product::<usize>() };

    // Handle empty/scalar tensor
    if numel == 0 || shape.is_empty() {
        // Scalar or empty - still try to read data
        if shape.is_empty() {
            // Scalar tensor
            if !proto.float_data.is_empty() {
                return Tensor::from_vec(vec![proto.float_data[0]], &[])
                    .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
            }
            if !proto.raw_data.is_empty() && proto.raw_data.len() >= 4 {
                let val = f32::from_le_bytes([
                    proto.raw_data[0], proto.raw_data[1], proto.raw_data[2], proto.raw_data[3]
                ]);
                return Tensor::from_vec(vec![val], &[])
                    .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
            }
            // Return scalar zero
            return Ok(zeros(&[]));
        }
        return Ok(zeros(&shape));
    }

    // Try float_data first
    if !proto.float_data.is_empty() {
        return Tensor::from_vec(proto.float_data.clone(), &shape)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
    }

    // Try double_data
    if !proto.double_data.is_empty() {
        let float_data: Vec<f32> = proto.double_data.iter().map(|&x| x as f32).collect();
        return Tensor::from_vec(float_data, &shape)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
    }

    // Try int64_data
    if !proto.int64_data.is_empty() {
        let float_data: Vec<f32> = proto.int64_data.iter().map(|&x| x as f32).collect();
        return Tensor::from_vec(float_data, &shape)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
    }

    // Try int32_data
    if !proto.int32_data.is_empty() {
        let float_data: Vec<f32> = proto.int32_data.iter().map(|&x| x as f32).collect();
        return Tensor::from_vec(float_data, &shape)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
    }

    // Try raw_data based on data_type
    if !proto.raw_data.is_empty() {
        let data_type = proto.data_type;
        let float_data = match data_type {
            1 => {
                // FLOAT (f32)
                let mut result = vec![0.0f32; numel];
                let bytes_needed = numel * 4;
                if proto.raw_data.len() >= bytes_needed {
                    for (i, chunk) in proto.raw_data[..bytes_needed].chunks_exact(4).enumerate() {
                        result[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                }
                result
            }
            6 => {
                // INT32
                let mut result = vec![0.0f32; numel];
                let bytes_needed = numel * 4;
                if proto.raw_data.len() >= bytes_needed {
                    for (i, chunk) in proto.raw_data[..bytes_needed].chunks_exact(4).enumerate() {
                        let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        result[i] = val as f32;
                    }
                }
                result
            }
            7 => {
                // INT64
                let mut result = vec![0.0f32; numel];
                let bytes_needed = numel * 8;
                if proto.raw_data.len() >= bytes_needed {
                    for (i, chunk) in proto.raw_data[..bytes_needed].chunks_exact(8).enumerate() {
                        let val = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        result[i] = val as f32;
                    }
                }
                result
            }
            10 => {
                // FLOAT16 (f16) - convert to f32
                let mut result = vec![0.0f32; numel];
                let bytes_needed = numel * 2;
                if proto.raw_data.len() >= bytes_needed {
                    for (i, chunk) in proto.raw_data[..bytes_needed].chunks_exact(2).enumerate() {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        result[i] = half_to_f32(bits);
                    }
                }
                result
            }
            11 => {
                // DOUBLE (f64)
                let mut result = vec![0.0f32; numel];
                let bytes_needed = numel * 8;
                if proto.raw_data.len() >= bytes_needed {
                    for (i, chunk) in proto.raw_data[..bytes_needed].chunks_exact(8).enumerate() {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        result[i] = val as f32;
                    }
                }
                result
            }
            16 => {
                // BFLOAT16 - convert to f32
                let mut result = vec![0.0f32; numel];
                let bytes_needed = numel * 2;
                if proto.raw_data.len() >= bytes_needed {
                    for (i, chunk) in proto.raw_data[..bytes_needed].chunks_exact(2).enumerate() {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        // BF16 is just top 16 bits of f32
                        let f32_bits = (bits as u32) << 16;
                        result[i] = f32::from_bits(f32_bits);
                    }
                }
                result
            }
            _ => {
                return Err(OnnxError::TensorConversion(
                    format!("Unsupported tensor data type {} for {}", data_type, proto.name)
                ));
            }
        };

        return Tensor::from_vec(float_data, &shape)
            .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
    }

    // If no data found, this could be a tensor with external data or just empty
    // For now, initialize with zeros and log a warning
    eprintln!("Warning: No data found for tensor {} (dims={:?}, dtype={}), initializing with zeros",
        proto.name, proto.dims, proto.data_type);
    Ok(zeros(&shape))
}

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            // Zero
            sign << 31
        } else {
            // Subnormal - convert to normalized f32
            let mut e = 0u32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp32 = 127 - 15 - e;
            let mant32 = (m & 0x3ff) << 13;
            (sign << 31) | (exp32 << 23) | mant32
        }
    } else if exp == 31 {
        // Inf or NaN
        let mant32 = mant << 13;
        (sign << 31) | (0xff << 23) | mant32
    } else {
        // Normal
        let exp32 = exp + 127 - 15;
        let mant32 = mant << 13;
        (sign << 31) | (exp32 << 23) | mant32
    };

    f32::from_bits(f32_bits)
}

fn compile_operators(graph: &GraphProto, _opset_version: i64) -> OnnxResult<Vec<CompiledOp>> {
    let mut operators = Vec::new();

    for node in &graph.node {
        let operator = crate::operators::create_operator(node)?;

        operators.push(CompiledOp {
            op_type: node.op_type.clone(),
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            operator,
        });
    }

    Ok(operators)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_input() {
        let input = ModelInput {
            name: "input".to_string(),
            shape: vec![Some(1), Some(3), Some(224), Some(224)],
            dtype: "float32".to_string(),
        };

        assert_eq!(input.name, "input");
        assert_eq!(input.shape.len(), 4);
    }
}
