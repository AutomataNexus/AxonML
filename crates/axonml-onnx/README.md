# axonml-onnx

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.6.1-green.svg" alt="Version 0.6.1">
  <img src="https://img.shields.io/badge/part%20of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

`axonml-onnx` provides ONNX (Open Neural Network Exchange) import and export
for AxonML. Imported models are executed by a dispatch-driven graph runtime
over `Tensor<f32>`; exported models are built via a fluent `OnnxExporter`
builder. Target opset is 17, IR version 8. Protobuf parsing is done
in-crate (no `prost-derive` generated types — the structs live in
`proto.rs`; `prost-build` is a build dep used for optional schema work).

## Features

- **ONNX Import** — `import_onnx(path)` and `import_onnx_bytes(bytes)` produce a ready-to-run `OnnxModel`
- **ONNX Export** — `OnnxExporter` builder (`add_input`, `add_output`, `add_node`, `add_initializer`, `to_proto`, `to_bytes`, `export`) + `export_feedforward` helper
- **Operator Support** — 40+ operators with dispatch via `create_operator(node)` returning `Box<dyn OnnxOperator>`
- **Initializers** — f32 weight tensors threaded through graph execution via `initializer_map`
- **State-Dict Bridge** — `OnnxModel::to_state_dict()` emits an `axonml_serialize::StateDict`
- **Dynamic Shapes** — `ModelInput::shape` is `Vec<Option<i64>>` with named dimensions supported via `Dimension::dynamic`
- **Error Handling** — rich `OnnxError` enum (file I/O, protobuf, opset, operator, shape, dtype, attribute, initializer, graph validation, tensor conversion, export, axonml-core)

## Modules

| Module | Description |
|--------|-------------|
| `parser` | `import_onnx`, `import_onnx_bytes`, `parse_model_proto`, `validate_model` |
| `export` | `OnnxExporter` builder, `AttributeValue` enum, `export_onnx`, `export_onnx_bytes`, `export_feedforward` |
| `model` | `OnnxModel`, `ModelInput`, `CompiledOp`, `forward`, `to_state_dict` |
| `operators` | `OnnxOperator` trait + 40+ operator structs and `create_operator` dispatch |
| `proto` | Hand-written ONNX protobuf types: `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, `AttributeProto`, `ValueInfo`, `TensorDataType`, `AttributeType`, `Dimension`, `TensorShape`, `OperatorSetIdProto`, `TypeProto`, `StringStringEntry` |
| `error` | `OnnxError` / `OnnxResult` |

## Supported Operators

All dispatched via `operators::create_operator`:

| Category | Operators |
|----------|-----------|
| Activations | `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyRelu`, `Gelu` |
| Math | `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Gemm`, `Sqrt`, `Pow`, `Exp`, `Log` |
| Shape | `Reshape`, `Transpose`, `Flatten`, `Squeeze`, `Unsqueeze`, `Concat`, `Gather` |
| Reduction | `ReduceSum`, `ReduceMean`, `ReduceMax` |
| Neural Network | `Conv`, `MaxPool`, `AveragePool`, `BatchNormalization`, `Dropout` |
| Comparison | `Equal`, `Greater`, `Less`, `Clip` |
| Utility | `Constant`, `Identity`, `Cast`, `Shape` |

Operators with attributes (`Softmax`, `LeakyRelu`, `Gemm`, `Transpose`,
`Flatten`, `Squeeze`, `Unsqueeze`, `Concat`, `Gather`, `ReduceSum`,
`ReduceMean`, `ReduceMax`, `Conv`, `MaxPool`, `AveragePool`,
`BatchNormalization`, `Constant`, `Clip`) parse their attributes via
`from_node`.

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-onnx = "0.6.1"
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
use std::collections::HashMap;

let mut exporter = OnnxExporter::new("linear_model");
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

exporter.add_output("output", &[1, 5], TensorDataType::Float);
```

### Exporting Feedforward Networks

```rust
use axonml_onnx::export::export_feedforward;
use axonml_tensor::Tensor;

// Define layers (in_features, out_features)
let layers = vec![(784, 256), (256, 128), (128, 10)];

let w1 = Tensor::from_vec(vec![0.01; 784 * 256], &[256, 784]).unwrap();
let w2 = Tensor::from_vec(vec![0.01; 256 * 128], &[128, 256]).unwrap();
let w3 = Tensor::from_vec(vec![0.01; 128 * 10], &[10, 128]).unwrap();
let b1 = Tensor::from_vec(vec![0.0; 256], &[256]).unwrap();
let b2 = Tensor::from_vec(vec![0.0; 128], &[128]).unwrap();
let b3 = Tensor::from_vec(vec![0.0; 10], &[10]).unwrap();

let weights = vec![("fc1_weight", &w1), ("fc2_weight", &w2), ("fc3_weight", &w3)];
let biases  = vec![("fc1_bias", &b1),  ("fc2_bias", &b2),  ("fc3_bias", &b3)];

let exporter = export_feedforward("mlp", &layers, &weights, &biases)?;
exporter.export("mlp.onnx")?;
```

### Working with Attributes

```rust
use axonml_onnx::export::AttributeValue;
use std::collections::HashMap;

let mut attrs = HashMap::new();

// Float / Int / arrays
attrs.insert("alpha".to_string(),        AttributeValue::Float(0.01));
attrs.insert("axis".to_string(),         AttributeValue::Int(-1));
attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));
attrs.insert("scales".to_string(),       AttributeValue::Floats(vec![1.0, 2.0]));
```

### Error Handling

```rust
use axonml_onnx::{import_onnx, OnnxError};

match import_onnx("model.onnx") {
    Ok(model) => println!("Model loaded successfully"),
    Err(OnnxError::FileRead(e))               => eprintln!("Could not read file: {}", e),
    Err(OnnxError::UnsupportedOperator(op))   => eprintln!("Operator not supported: {}", op),
    Err(OnnxError::UnsupportedOpset(v))       => eprintln!("Opset not supported: {}", v),
    Err(OnnxError::InvalidShape(msg))         => eprintln!("Invalid tensor shape: {}", msg),
    Err(OnnxError::MissingInitializer(name))  => eprintln!("Missing initializer: {}", name),
    Err(OnnxError::GraphValidation(msg))      => eprintln!("Graph error: {}", msg),
    Err(e)                                    => eprintln!("Error: {}", e),
}
```

### State-Dict Bridge

```rust
use axonml_onnx::import_onnx;

let model = import_onnx("model.onnx")?;
let state_dict = model.to_state_dict();
// Feed into axonml-serialize save/load
```

## Constants

```rust
use axonml_onnx::{SUPPORTED_OPSET_VERSION, ONNX_IR_VERSION};

println!("Supported opset: {}", SUPPORTED_OPSET_VERSION);  // 17
println!("IR version: {}", ONNX_IR_VERSION);                // 8
```

## Tests

```bash
cargo test -p axonml-onnx
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
