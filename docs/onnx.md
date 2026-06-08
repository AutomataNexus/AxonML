---
layout: default
title: ONNX
nav_order: 7
description: "ONNX model import and export with AxonML"
---

# ONNX Import/Export
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

`axonml-onnx` provides ONNX (Open Neural Network Exchange) import and export for interoperability with PyTorch, TensorFlow, and other frameworks. Protobuf parsing is implemented via `prost` (the `proto` module). The current supported ONNX opset version and IR version are:

```rust
use axonml_onnx::{SUPPORTED_OPSET_VERSION, ONNX_IR_VERSION};

assert_eq!(SUPPORTED_OPSET_VERSION, 17);
assert_eq!(ONNX_IR_VERSION, 8);
```

**Supported:**
- 40+ ONNX operators
- Import from PyTorch-exported models
- Export AxonML computation graphs to ONNX via `OnnxExporter`
- Dynamic input shapes (unknown dims become `None` in `ModelInput::shape`)

## Importing ONNX Models

### Basic Import

Two entry points: from a path, or from a byte slice.

```rust
use std::collections::HashMap;
use axonml_onnx::{import_onnx, import_onnx_bytes, OnnxModel};
use axonml_tensor::Tensor;

// From a path
let model: OnnxModel = import_onnx("model.onnx")?;

// From bytes (e.g. loaded from S3 / hub)
let bytes = std::fs::read("model.onnx")?;
let model = import_onnx_bytes(&bytes)?;

// Inspect the model
println!("Name: {}", model.name);
println!("Opset: {}", model.opset_version);
println!("Inputs: {:?}", model.get_inputs());
println!("Outputs: {:?}", model.get_outputs());
println!("Parameters: {}", model.num_parameters());

// Run inference — OnnxModel::forward takes a HashMap<input_name, Tensor<f32>>
// and returns a HashMap<output_name, Tensor<f32>>.
let mut inputs = HashMap::new();
inputs.insert("input".to_string(), Tensor::<f32>::randn(&[1, 3, 224, 224]));
let outputs = model.forward(inputs)?;
```

## Exporting to ONNX

Export is explicit — you build an `OnnxExporter` and add inputs, outputs, nodes, and initializers. The free function `export_onnx` then writes the resulting `ModelProto` to disk.

```rust
use axonml_onnx::{export_onnx, export_onnx_bytes};
use axonml_onnx::export::OnnxExporter;
use axonml_onnx::proto::TensorDataType;
use axonml_tensor::Tensor;

let mut exporter = OnnxExporter::new("my_model")
    .with_producer("axonml", "0.6.5")
    .with_doc_string("Two-layer MLP exported from AxonML");

// Declare IO
exporter.add_input("input", &[1, 784], TensorDataType::Float);
exporter.add_output("logits", &[1, 10], TensorDataType::Float);

// Add initializers (weights)
let w1 = Tensor::<f32>::randn(&[784, 256]);
exporter.add_initializer("fc1.weight", &w1);

// Add a node (op_type, inputs, outputs, attributes via the add_node signature
// — see crates/axonml-onnx/src/export.rs for the full attribute encoding)
// exporter.add_node(...);

// Write to disk
export_onnx(&exporter, "my_model.onnx")?;

// Or serialize to bytes
let bytes = export_onnx_bytes(&exporter)?;
```

For feedforward models, `axonml_onnx::export::export_feedforward` automates wiring the common `Linear → Activation → Linear` patterns.

## Converting ONNX Weights to a StateDict

`OnnxModel::to_state_dict()` extracts the model's initializers (weights) to an `axonml_serialize::StateDict`, which you can feed to AxonML modules whose tensors are keyed by the same names:

```rust
let model = import_onnx("model.onnx")?;
let state_dict: axonml_serialize::StateDict = model.to_state_dict();
```

## Supported Operators

The parser/operator registry is in `axonml_onnx::operators`. 40+ ONNX ops are implemented, grouped as follows:

### Math

`Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Gemm`, `Pow`, `Sqrt`, `Exp`, `Log`

### Tensor

`Reshape`, `Transpose`, `Concat`, `Split`, `Slice`, `Gather`, `Squeeze`, `Unsqueeze`, `Flatten`

### Reduction

`ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd`

### NN

`Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool`, `BatchNormalization`, `Dropout` (inference mode), `Softmax`, `LogSoftmax`

### Activations

`Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `Elu`, `Gelu`, `Silu`

### RNN

`LSTM`, `GRU`

(Run `grep -n "register" /opt/AxonML/crates/axonml-onnx/src/operators.rs` for the authoritative up-to-date list.)

## Working with PyTorch Models

### Export from PyTorch

```python
import torch

model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=17,                         # Match AxonML's SUPPORTED_OPSET_VERSION
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

### Import in AxonML

```rust
use std::collections::HashMap;
use axonml_onnx::import_onnx;
use axonml_tensor::Tensor;

let model = import_onnx("resnet18.onnx")?;

let mut inputs = HashMap::new();
inputs.insert("input".to_string(), Tensor::<f32>::randn(&[1, 3, 224, 224]));
let outputs = model.forward(inputs)?;
let logits = outputs.get("output").unwrap();
```

## Error Handling

All imports return `OnnxResult<T>` (alias for `Result<T, OnnxError>`):

```rust
use axonml_onnx::{OnnxError, OnnxResult};

match import_onnx("model.onnx") {
    Ok(model) => { /* ... */ }
    Err(OnnxError::GraphValidation(msg)) => eprintln!("graph error: {msg}"),
    Err(e) => eprintln!("other ONNX error: {e}"),
}
```

---

*Last updated: 2026-06-06 (v0.6.5)*
