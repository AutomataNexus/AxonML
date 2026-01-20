# axonml-onnx

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part%20of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

`axonml-onnx` provides ONNX (Open Neural Network Exchange) import and export functionality for the AxonML machine learning framework. It enables interoperability with PyTorch, TensorFlow, and other ML frameworks by supporting ONNX model loading and saving up to opset version 17.

## Features

- **ONNX Import** - Load ONNX models and convert to executable AxonML representation
- **ONNX Export** - Export AxonML models to ONNX format for deployment
- **Operator Support** - Implementations for 40+ common ONNX operators including activations, math, and neural network ops
- **Protobuf Parsing** - Full ONNX protobuf structure support with JSON fallback for testing
- **Graph Execution** - Execute imported models with automatic operator dispatch
- **Feedforward Export** - Helper utilities for exporting simple feedforward networks
- **Error Handling** - Comprehensive error types for debugging import/export issues

## Modules

| Module | Description |
|--------|-------------|
| `parser` | ONNX file parsing and model import functionality |
| `export` | OnnxExporter builder and export utilities |
| `model` | OnnxModel representation for inference |
| `operators` | ONNX operator implementations (Relu, MatMul, Conv, etc.) |
| `proto` | ONNX protobuf structure definitions |
| `error` | Error types for ONNX operations |

## Supported Operators

| Category | Operators |
|----------|-----------|
| Activations | Relu, Sigmoid, Tanh, Softmax, LeakyRelu, Gelu |
| Math | Add, Sub, Mul, Div, MatMul, Gemm, Sqrt, Pow, Exp, Log |
| Shape | Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Gather |
| Reduction | ReduceSum, ReduceMean, ReduceMax |
| Neural Network | Conv*, MaxPool*, AveragePool*, BatchNormalization, Dropout |
| Comparison | Equal, Greater, Less, Clip |
| Utility | Constant, Identity, Cast, Shape |

*\* Partial implementation - basic structure only*

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-onnx = "0.1.0"
```

### Importing an ONNX Model

```rust
use axonml_onnx::{import_onnx, OnnxModel};
use axonml_tensor::Tensor;
use std::collections::HashMap;

// Import model from file
let model = import_onnx("model.onnx")?;

// Inspect model
println!("Inputs: {:?}", model.get_inputs());
println!("Outputs: {:?}", model.get_outputs());
println!("Parameters: {}", model.num_parameters());

// Run inference
let mut inputs = HashMap::new();
let input_tensor = Tensor::from_vec(vec![1.0; 784], &[1, 784]).unwrap();
inputs.insert("input".to_string(), input_tensor);

let outputs = model.forward(inputs)?;
let output = outputs.get("output").unwrap();
```

### Importing from Bytes

```rust
use axonml_onnx::import_onnx_bytes;

let bytes = std::fs::read("model.onnx")?;
let model = import_onnx_bytes(&bytes)?;
```

### Exporting a Model

```rust
use axonml_onnx::export::{OnnxExporter, AttributeValue};
use axonml_onnx::proto::TensorDataType;
use axonml_tensor::Tensor;
use std::collections::HashMap;

// Create exporter
let mut exporter = OnnxExporter::new("my_model")
    .with_producer("MyApp", "1.0.0")
    .with_doc_string("A simple ReLU model");

// Add input
exporter.add_input("input", &[1, 10], TensorDataType::Float);

// Add nodes (operators)
exporter.add_node("Relu", &["input"], &["relu_out"], HashMap::new());

// Add output
exporter.add_output("output", &[1, 10], TensorDataType::Float);

// Export to file
exporter.export("model.onnx")?;

// Or export to bytes
let bytes = exporter.to_bytes()?;
```

### Adding Weights to Export

```rust
use axonml_onnx::export::{OnnxExporter, AttributeValue};
use axonml_onnx::proto::TensorDataType;
use axonml_tensor::Tensor;

let mut exporter = OnnxExporter::new("linear_model");

// Add input
exporter.add_input("input", &[1, 10], TensorDataType::Float);

// Add weight initializer
let weights = Tensor::from_vec(vec![0.1; 50], &[10, 5]).unwrap();
exporter.add_initializer("weight", &weights);

// Add bias initializer
let bias = Tensor::from_vec(vec![0.0; 5], &[5]).unwrap();
exporter.add_initializer("bias", &bias);

// Add Gemm node with attributes
let mut attrs = HashMap::new();
attrs.insert("transB".to_string(), AttributeValue::Int(1));
exporter.add_node("Gemm", &["input", "weight", "bias"], &["output"], attrs);

// Add output
exporter.add_output("output", &[1, 5], TensorDataType::Float);
```

### Exporting Feedforward Networks

```rust
use axonml_onnx::export::export_feedforward;
use axonml_tensor::Tensor;

// Define layers (in_features, out_features)
let layers = vec![(784, 256), (256, 128), (128, 10)];

// Prepare weights and biases
let weights = vec![
    ("fc1_weight", &Tensor::from_vec(vec![0.01; 784 * 256], &[256, 784]).unwrap()),
    ("fc2_weight", &Tensor::from_vec(vec![0.01; 256 * 128], &[128, 256]).unwrap()),
    ("fc3_weight", &Tensor::from_vec(vec![0.01; 128 * 10], &[10, 128]).unwrap()),
];

let biases = vec![
    ("fc1_bias", &Tensor::from_vec(vec![0.0; 256], &[256]).unwrap()),
    ("fc2_bias", &Tensor::from_vec(vec![0.0; 128], &[128]).unwrap()),
    ("fc3_bias", &Tensor::from_vec(vec![0.0; 10], &[10]).unwrap()),
];

let exporter = export_feedforward("mlp", &layers, &weights, &biases)?;
exporter.export("mlp.onnx")?;
```

### Working with Attributes

```rust
use axonml_onnx::export::AttributeValue;
use std::collections::HashMap;

let mut attrs = HashMap::new();

// Float attribute
attrs.insert("alpha".to_string(), AttributeValue::Float(0.01));

// Integer attribute
attrs.insert("axis".to_string(), AttributeValue::Int(-1));

// Integer array attribute
attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));

// Float array attribute
attrs.insert("scales".to_string(), AttributeValue::Floats(vec![1.0, 2.0]));

exporter.add_node("LeakyRelu", &["input"], &["output"], attrs);
```

### Error Handling

```rust
use axonml_onnx::{import_onnx, OnnxError};

match import_onnx("model.onnx") {
    Ok(model) => {
        println!("Model loaded successfully");
    }
    Err(OnnxError::FileRead(e)) => {
        eprintln!("Could not read file: {}", e);
    }
    Err(OnnxError::UnsupportedOperator(op)) => {
        eprintln!("Operator not supported: {}", op);
    }
    Err(OnnxError::InvalidShape(msg)) => {
        eprintln!("Invalid tensor shape: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Constants

```rust
use axonml_onnx::{SUPPORTED_OPSET_VERSION, ONNX_IR_VERSION};

println!("Supported opset: {}", SUPPORTED_OPSET_VERSION);  // 17
println!("IR version: {}", ONNX_IR_VERSION);                // 8
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-onnx
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
