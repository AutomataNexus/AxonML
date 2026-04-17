# axonml-serialize

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

`axonml-serialize` handles model state I/O for AxonML: named-parameter
`StateDict`s, `Checkpoint`s with full training state, and format conversion
for PyTorch / ONNX interop. The native `.axonml` format is bincode-encoded
binary; JSON and SafeTensors (behind the `safetensors` feature) are also
supported. Format is detected from the file extension with magic-byte
fallback.

## Features

- **Multiple Formats** — native bincode `.axonml`, `.json`, `.safetensors` (feature-gated)
- **State Dictionaries** — `StateDict` with `from_module`, `insert`, `get`, `entries`, `keys`, `merge`, `filter_prefix`, `strip_prefix`, `add_prefix`, `remove`, `set_metadata` / `get_metadata`, `total_params`, `size_bytes`, `summary`
- **Training Checkpoints** — `Checkpoint` + `CheckpointBuilder` + `TrainingState` with loss / val-loss / lr / custom metric history, best-metric tracking, epoch / step counters, ISO-8601 timestamp, config map
- **Format Detection** — `detect_format(path)` by extension, `detect_format_from_bytes(bytes)` by magic bytes; `Format::is_binary`, `Format::supports_streaming`, `Format::extension`, `Format::name`, `Format::all`
- **PyTorch Conversion** — `from_pytorch_key`, `to_pytorch_key`, `pytorch_layer_mapping`, `convert_from_pytorch`, `transpose_linear_weights`
- **ONNX Utilities** — `to_onnx_shape` / `from_onnx_shape` (dynamic batch dim handling), `OnnxOpType` with `parse_op` / `as_str`
- **High-Level API** — `save_model(&model, path)` / `load_model(&model, path)` (name-matched param load with positional fallback), `save_state_dict` / `load_state_dict`, `save_checkpoint` / `load_checkpoint`

## Feature Flags

| Flag | Effect |
|------|--------|
| `safetensors` | Enables `.safetensors` save/load (f32 / f16 / bf16 / f64 input), pulls `safetensors = "0.3"` and `half` |

## Modules

| Module | Description |
|--------|-------------|
| `state_dict` | `TensorData`, `StateDictEntry`, `StateDict` |
| `checkpoint` | `Checkpoint`, `CheckpointBuilder`, `TrainingState` |
| `format` | `Format` enum and detection helpers |
| `convert` | PyTorch / ONNX conversion utilities, `OnnxOpType` |

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-serialize = "0.6.1"

# Or with SafeTensors:
axonml-serialize = { version = "0.6.1", features = ["safetensors"] }
```

### Saving and Loading Models

```rust
use axonml_serialize::{save_model, load_model, load_state_dict};
use axonml_nn::Linear;

// Save a model (format detected from extension)
let model = Linear::new(10, 5);
save_model(&model, "model.axonml")?;        // Binary format
save_model(&model, "model.json")?;          // JSON format
// save_model(&model, "model.safetensors")?; // Requires `safetensors` feature

// Inspect the state dict directly
let sd = load_state_dict("model.axonml")?;
println!("Parameters: {}", sd.total_params());
println!("Size: {} bytes", sd.size_bytes());

// Or load weights back into a model (name-matched, positional fallback)
let target = Linear::new(10, 5);
let loaded = load_model(&target, "model.axonml")?;
println!("Loaded {loaded} parameters");
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

// Filter / rename
let linear_params = state_dict.filter_prefix("linear.");
let stripped       = state_dict.strip_prefix("linear.");
let prefixed       = state_dict.add_prefix("module.");
```

### Training Checkpoints

```rust
use axonml_serialize::{Checkpoint, TrainingState, save_checkpoint, load_checkpoint};

// Track training state
let mut training_state = TrainingState::new();
training_state.record_loss(0.5);
training_state.record_loss(0.3);
training_state.record_val_loss(0.35);
training_state.record_lr(1e-3);
training_state.record_metric("accuracy", 0.92);
training_state.update_best("loss", 0.3, false);  // lower is better

training_state.next_epoch();
training_state.next_step();

// Average last N losses
let smoothed = training_state.avg_loss(10);

// Build checkpoint
let checkpoint = Checkpoint::builder()
    .model_state(model_state_dict)
    .optimizer_state(optimizer_state_dict)
    .training_state(training_state)
    .rng_state(rng_bytes)
    .epoch(10)
    .global_step(5000)
    .config("learning_rate", "0.001")
    .config("batch_size", "32")
    .build();

// Save and load checkpoints (bincode)
save_checkpoint(&checkpoint, "checkpoint.ckpt")?;
let loaded = load_checkpoint("checkpoint.ckpt")?;

println!("Resuming from epoch {}", loaded.epoch());
println!("Best metric: {:?}", loaded.best_metric());
```

### Format Detection

```rust
use axonml_serialize::{detect_format, detect_format_from_bytes, Format};

// Detect from file extension
assert_eq!(detect_format("model.json"),        Format::Json);
assert_eq!(detect_format("model.safetensors"), Format::SafeTensors);
assert_eq!(detect_format("model.bin"),         Format::Axonml); // default

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
use axonml_serialize::{
    from_pytorch_key, to_pytorch_key, pytorch_layer_mapping,
    convert_from_pytorch, transpose_linear_weights,
};

// Convert PyTorch key naming to AxonML
let key = from_pytorch_key("module.layer1.weight");

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

// ONNX operator name mapping
let op = OnnxOpType::parse_op("Relu");
assert_eq!(op.as_str(), "Relu");
```

### State Dictionary Metadata

```rust
use axonml_serialize::StateDict;

let mut state_dict = StateDict::new();
state_dict.set_metadata("framework_version", "0.6.1");
state_dict.set_metadata("model_architecture", "ResNet50");

if let Some(version) = state_dict.get_metadata("framework_version") {
    println!("Saved with version: {}", version);
}
```

## Tests

```bash
cargo test -p axonml-serialize
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
