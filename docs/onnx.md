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

AxonML supports ONNX (Open Neural Network Exchange) for interoperability with PyTorch, TensorFlow, and other frameworks.

**Supported Features:**
- 40+ ONNX operators
- Import from PyTorch-exported models
- Export AxonML models to ONNX
- Dynamic input shapes
- Custom operators

## Importing ONNX Models

### Basic Import

```rust
use axonml::onnx::import_onnx;

// Load ONNX model
let model = import_onnx("model.onnx")?;

// Get model information
println!("Inputs: {:?}", model.inputs());
println!("Outputs: {:?}", model.outputs());

// Run inference
let input = Tensor::randn(&[1, 3, 224, 224]);
let output = model.forward(&input)?;
```

### From Bytes

```rust
use axonml::onnx::OnnxModel;

let bytes = std::fs::read("model.onnx")?;
let model = OnnxModel::from_bytes(&bytes)?;
```

### With Options

```rust
use axonml::onnx::{import_onnx_with_options, ImportOptions};

let options = ImportOptions::new()
    .device(Device::CUDA(0))
    .dtype(DType::F16)
    .optimize(true);

let model = import_onnx_with_options("model.onnx", options)?;
```

## Exporting to ONNX

### Basic Export

```rust
use axonml::onnx::export_onnx;

let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 10));

// Export with example input shape
export_onnx(&model, "my_model.onnx", &[1, 784])?;
```

### With Dynamic Axes

```rust
use axonml::onnx::{export_onnx_with_options, ExportOptions};

let options = ExportOptions::new()
    .dynamic_axes(vec![("input", vec![0])])  // Batch dimension is dynamic
    .opset_version(13);

export_onnx_with_options(&model, "model.onnx", &[1, 784], options)?;
```

## Supported Operators

### Math Operations

| ONNX Op | Status | Notes |
|:--------|:-------|:------|
| Add | ✅ | With broadcasting |
| Sub | ✅ | With broadcasting |
| Mul | ✅ | With broadcasting |
| Div | ✅ | With broadcasting |
| MatMul | ✅ | 2D and batched |
| Gemm | ✅ | General matrix multiply |
| Pow | ✅ | Element-wise |
| Sqrt | ✅ | |
| Exp | ✅ | |
| Log | ✅ | |

### Tensor Operations

| ONNX Op | Status | Notes |
|:--------|:-------|:------|
| Reshape | ✅ | |
| Transpose | ✅ | |
| Concat | ✅ | Any axis |
| Split | ✅ | |
| Slice | ✅ | |
| Gather | ✅ | |
| Squeeze | ✅ | |
| Unsqueeze | ✅ | |
| Flatten | ✅ | |

### Reduction Operations

| ONNX Op | Status | Notes |
|:--------|:-------|:------|
| ReduceSum | ✅ | |
| ReduceMean | ✅ | |
| ReduceMax | ✅ | |
| ReduceMin | ✅ | |
| ReduceProd | ✅ | |

### Neural Network Layers

| ONNX Op | Status | Notes |
|:--------|:-------|:------|
| Conv | ✅ | 1D and 2D |
| ConvTranspose | ✅ | |
| MaxPool | ✅ | |
| AveragePool | ✅ | |
| GlobalAveragePool | ✅ | |
| BatchNormalization | ✅ | |
| Dropout | ✅ | Inference mode |
| Softmax | ✅ | |
| LogSoftmax | ✅ | |

### Activations

| ONNX Op | Status | Notes |
|:--------|:-------|:------|
| Relu | ✅ | |
| LeakyRelu | ✅ | |
| Sigmoid | ✅ | |
| Tanh | ✅ | |
| Elu | ✅ | |
| Gelu | ✅ | |
| Silu | ✅ | |

### RNN

| ONNX Op | Status | Notes |
|:--------|:-------|:------|
| LSTM | ✅ | Unidirectional |
| GRU | ✅ | |

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
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

### Import in AxonML

```rust
use axonml::onnx::import_onnx;

let model = import_onnx("resnet18.onnx")?;

// Run inference
let input = Tensor::randn(&[1, 3, 224, 224]);
let output = model.forward(&input)?;

// Get predictions
let predictions = output.argmax(1);
```

## Custom Operators

### Register Custom Op

```rust
use axonml::onnx::{register_custom_op, CustomOp};

struct MyCustomOp;

impl CustomOp for MyCustomOp {
    fn name(&self) -> &str {
        "MyCustomOp"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let x = inputs[0];
        let y = x.mul(&Tensor::full(x.shape(), 2.0));
        Ok(vec![y])
    }
}

// Register before importing
register_custom_op(MyCustomOp);

let model = import_onnx("model_with_custom_op.onnx")?;
```

## Model Optimization

### Constant Folding

```rust
use axonml::onnx::{optimize_model, OptimizationPass};

let model = import_onnx("model.onnx")?;

let optimized = optimize_model(&model, &[
    OptimizationPass::ConstantFolding,
    OptimizationPass::EliminateDeadNodes,
    OptimizationPass::FuseOperations,
])?;
```

### Quantization

```rust
use axonml::onnx::quantize_model;

let model = import_onnx("model.onnx")?;
let quantized = quantize_model(&model, DType::I8)?;

// Save quantized model
export_onnx(&quantized, "model_int8.onnx", &[1, 3, 224, 224])?;
```

## Validation

```rust
use axonml::onnx::validate_model;

// Check model is valid ONNX
let result = validate_model("model.onnx");
match result {
    Ok(()) => println!("Model is valid"),
    Err(e) => println!("Validation error: {}", e),
}
```

## Model Information

```rust
let model = import_onnx("model.onnx")?;

// Input/output info
for input in model.inputs() {
    println!("Input: {} - {:?}", input.name, input.shape);
}

for output in model.outputs() {
    println!("Output: {} - {:?}", output.name, output.shape);
}

// Graph info
println!("Nodes: {}", model.graph().nodes().len());
println!("Opset version: {}", model.opset_version());
```
