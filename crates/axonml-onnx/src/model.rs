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
    let data = proto.get_float_data();

    if data.is_empty() {
        // Handle other data types by converting to f32
        let int64_data = proto.get_int64_data();
        if !int64_data.is_empty() {
            let float_data: Vec<f32> = int64_data.iter().map(|&x| x as f32).collect();
            return Tensor::from_vec(float_data, &shape)
                .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)));
        }

        // Empty tensor
        if proto.numel() == 0 {
            return Ok(zeros(&shape));
        }

        return Err(OnnxError::TensorConversion(
            format!("Unsupported tensor data type for {}", proto.name)
        ));
    }

    Tensor::from_vec(data, &shape)
        .map_err(|e| OnnxError::TensorConversion(format!("{:?}", e)))
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
