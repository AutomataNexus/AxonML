# axonml-serialize

<p align="center">
  <!-- Logo placeholder -->
  <img src="../../assets/logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part%20of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

`axonml-serialize` provides model serialization functionality for the AxonML machine learning framework. It supports saving and loading trained models, including state dictionaries, training checkpoints, and format conversion utilities for interoperability with PyTorch and ONNX.

## Features

- **Multiple Formats** - Support for AxonML native binary (.axonml), JSON (.json), and SafeTensors (.safetensors) formats
- **State Dictionaries** - PyTorch-style state_dict for storing and loading model parameters
- **Training Checkpoints** - Save complete training state including model, optimizer, epoch, and metrics
- **Format Detection** - Automatic format detection from file extensions and magic bytes
- **PyTorch Conversion** - Utilities for converting between PyTorch and AxonML naming conventions
- **ONNX Shape Utilities** - Helper functions for ONNX shape conversion with dynamic dimension support
- **Metadata Support** - Attach custom metadata to state dictionaries and checkpoints

## Modules

| Module | Description |
|--------|-------------|
| `state_dict` | StateDict and TensorData for storing model parameters by name |
| `checkpoint` | Checkpoint and TrainingState for saving/resuming training sessions |
| `format` | Format enum and detection utilities for different serialization formats |
| `convert` | Conversion utilities for PyTorch and ONNX interoperability |

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-serialize = "0.1.0"
```

### Saving and Loading Models

```rust
use axonml_serialize::{save_model, load_state_dict};
use axonml_nn::Linear;

// Save a model (format detected from extension)
let model = Linear::new(10, 5);
save_model(&model, "model.axonml")?;  // Binary format
save_model(&model, "model.json")?;     // JSON format

// Load state dictionary
let state_dict = load_state_dict("model.axonml")?;
println!("Parameters: {}", state_dict.total_params());
println!("Size: {} bytes", state_dict.size_bytes());
```

### Working with State Dictionaries

```rust
use axonml_serialize::{StateDict, TensorData};

// Create a state dictionary
let mut state_dict = StateDict::new();

let weights = TensorData {
    shape: vec![10, 5],
    values: vec![0.0; 50],
};
state_dict.insert("linear.weight".to_string(), weights);

let bias = TensorData {
    shape: vec![5],
    values: vec![0.0; 5],
};
state_dict.insert("linear.bias".to_string(), bias);

// Query the state dictionary
assert!(state_dict.contains("linear.weight"));
println!("{}", state_dict.summary());

// Filter by prefix
let linear_params = state_dict.filter_prefix("linear.");

// Strip prefix from keys
let stripped = state_dict.strip_prefix("linear.");
assert!(stripped.contains("weight"));
```

### Training Checkpoints

```rust
use axonml_serialize::{Checkpoint, TrainingState, save_checkpoint, load_checkpoint};

// Track training state
let mut training_state = TrainingState::new();
training_state.record_loss(0.5);
training_state.record_loss(0.3);
training_state.update_best("loss", 0.3, false);  // lower is better

training_state.next_epoch();
training_state.next_step();

// Create checkpoint with builder pattern
let checkpoint = Checkpoint::builder()
    .model_state(model_state_dict)
    .optimizer_state(optimizer_state_dict)
    .training_state(training_state)
    .epoch(10)
    .global_step(5000)
    .config("learning_rate", "0.001")
    .config("batch_size", "32")
    .build();

// Save and load checkpoints
save_checkpoint(&checkpoint, "checkpoint.ckpt")?;
let loaded = load_checkpoint("checkpoint.ckpt")?;

println!("Resuming from epoch {}", loaded.epoch());
println!("Best metric: {:?}", loaded.best_metric());
```

### Format Detection

```rust
use axonml_serialize::{detect_format, detect_format_from_bytes, Format};

// Detect from file extension
let format = detect_format("model.json");
assert_eq!(format, Format::Json);

let format = detect_format("model.safetensors");
assert_eq!(format, Format::SafeTensors);

// Detect from file contents
let bytes = b"{\"key\": \"value\"}";
let format = detect_format_from_bytes(bytes);
assert_eq!(format, Some(Format::Json));

// Format properties
assert!(Format::Axonml.is_binary());
assert!(!Format::Json.is_binary());
```

### PyTorch Conversion

```rust
use axonml_serialize::{from_pytorch_key, convert_from_pytorch, transpose_linear_weights};

// Convert PyTorch key naming to AxonML
let key = from_pytorch_key("module.layer1.weight");
assert_eq!(key, "layer1.weight");

// Convert entire state dictionary
let axonml_dict = convert_from_pytorch(&pytorch_dict);

// Transpose linear weights if needed (PyTorch uses [out, in])
let transposed = transpose_linear_weights(&weight_data);
```

### ONNX Shape Utilities

```rust
use axonml_serialize::{to_onnx_shape, from_onnx_shape, OnnxOpType};

// Convert to ONNX shape (with dynamic batch)
let onnx_shape = to_onnx_shape(&[3, 224, 224], true);
assert_eq!(onnx_shape, vec![-1, 3, 224, 224]);

// Convert from ONNX shape (replace -1 with default)
let shape = from_onnx_shape(&[-1, 3, 224, 224], 1);
assert_eq!(shape, vec![1, 3, 224, 224]);

// ONNX operator type mapping
let op = OnnxOpType::from_str("Relu");
assert_eq!(op.as_str(), "Relu");
```

### State Dictionary Metadata

```rust
use axonml_serialize::StateDict;

let mut state_dict = StateDict::new();
state_dict.set_metadata("framework_version", "0.1.0");
state_dict.set_metadata("model_architecture", "ResNet50");

if let Some(version) = state_dict.get_metadata("framework_version") {
    println!("Saved with version: {}", version);
}
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-serialize
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
