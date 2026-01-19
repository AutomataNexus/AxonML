# axonml-onnx

[![Crates.io](https://img.shields.io/crates/v/axonml-onnx.svg)](https://crates.io/crates/axonml-onnx)
[![Docs.rs](https://docs.rs/axonml-onnx/badge.svg)](https://docs.rs/axonml-onnx)
[![Downloads](https://img.shields.io/crates/d/axonml-onnx.svg)](https://crates.io/crates/axonml-onnx)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> ONNX import/export for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-onnx` provides ONNX model import and export capabilities for interoperability with other ML frameworks like PyTorch, TensorFlow, and various inference runtimes.

## Features

### Import
- **Load ONNX models** - Run inference on ONNX models
- **40+ operators** - Wide operator coverage
- **Opset 17** - Latest ONNX opset support

### Export
- **Export Axonml models** - Convert to ONNX format
- **Optimizations** - Graph optimizations on export
- **Metadata** - Custom metadata embedding

### Operators
- Arithmetic: Add, Sub, Mul, Div, Pow
- Matrix: MatMul, Gemm
- Activations: Relu, Sigmoid, Tanh, Softmax, Gelu
- Pooling: MaxPool, AveragePool, GlobalAveragePool
- Normalization: BatchNormalization, LayerNormalization
- Shape: Reshape, Transpose, Squeeze, Unsqueeze
- And more...

## Installation

```toml
[dependencies]
axonml-onnx = "0.1"
```

## Usage

### Load ONNX Model

```rust
use axonml_onnx::OnnxModel;

// Load ONNX model
let model = OnnxModel::load("model.onnx")?;

// Print model info
println!("Inputs: {:?}", model.inputs());
println!("Outputs: {:?}", model.outputs());

// Run inference
let input = randn(&[1, 3, 224, 224]);
let output = model.forward(&[("input", &input)])?;
```

### Export to ONNX

```rust
use axonml_onnx::export_onnx;
use axonml_nn::Module;

let model = create_model();

// Export with sample input for tracing
let sample_input = randn(&[1, 784]);
export_onnx(
    &model,
    &[("input", &sample_input)],
    "model.onnx",
    17,  // opset version
)?;
```

### Model Inspection

```rust
use axonml_onnx::OnnxModel;

let model = OnnxModel::load("model.onnx")?;

// Get model graph
for node in model.graph().nodes() {
    println!("Op: {}, Inputs: {:?}", node.op_type(), node.inputs());
}

// Get model metadata
if let Some(producer) = model.producer_name() {
    println!("Produced by: {}", producer);
}
```

### Dynamic Shapes

```rust
use axonml_onnx::OnnxModel;

let model = OnnxModel::load("model.onnx")?;

// Model supports dynamic batch size
let batch_1 = model.forward(&[("input", &randn(&[1, 784]))])?;
let batch_32 = model.forward(&[("input", &randn(&[32, 784]))])?;
```

## Supported Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log |
| Matrix | MatMul, Gemm |
| Activation | Relu, LeakyRelu, Sigmoid, Tanh, Softmax, Gelu |
| Pooling | MaxPool, AveragePool, GlobalAveragePool |
| Normalization | BatchNormalization, LayerNormalization |
| Shape | Reshape, Transpose, Squeeze, Unsqueeze, Flatten |
| Reduction | ReduceSum, ReduceMean, ReduceMax, ReduceMin |
| Comparison | Equal, Greater, Less |
| Logic | And, Or, Not |
| Tensor | Concat, Split, Slice, Gather, Pad |

## Part of Axonml

```toml
[dependencies]
axonml = "0.1"  # Includes ONNX support
```

## License

MIT OR Apache-2.0
