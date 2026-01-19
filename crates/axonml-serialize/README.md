# axonml-serialize

[![Crates.io](https://img.shields.io/crates/v/axonml-serialize.svg)](https://crates.io/crates/axonml-serialize)
[![Docs.rs](https://docs.rs/axonml-serialize/badge.svg)](https://docs.rs/axonml-serialize)
[![Downloads](https://img.shields.io/crates/d/axonml-serialize.svg)](https://crates.io/crates/axonml-serialize)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Model serialization for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-serialize` provides model saving and loading capabilities including checkpoint management, state dictionaries, and multiple format support. Compatible with SafeTensors format for secure model distribution.

## Features

### Formats
- **Axonml native** - Optimized binary format (.axonml)
- **SafeTensors** - Safe tensor serialization
- **JSON** - Human-readable metadata
- **Bincode** - Compact binary encoding

### Checkpoint Management
- **Save/load checkpoints** - Full training state
- **Best model tracking** - Save best validation model
- **Periodic saving** - Auto-save every N epochs

### State Dictionaries
- **Model state** - All model parameters
- **Optimizer state** - Optimizer internal state
- **Scheduler state** - Learning rate scheduler state

## Installation

```toml
[dependencies]
axonml-serialize = "0.1"
```

## Usage

### Save and Load Model

```rust
use axonml_serialize::{save_model, load_model};

// Save model
let model = create_model();
save_model(&model, "model.axonml")?;

// Load model
let loaded_model = load_model::<MyModel>("model.axonml")?;
```

### State Dictionary

```rust
use axonml_serialize::StateDict;

// Get state dict from model
let state_dict = model.state_dict();

// Save state dict
state_dict.save("weights.safetensors")?;

// Load into model
let loaded_state = StateDict::load("weights.safetensors")?;
model.load_state_dict(&loaded_state)?;
```

### Training Checkpoint

```rust
use axonml_serialize::Checkpoint;

// Save full training state
let checkpoint = Checkpoint::new()
    .model(&model)
    .optimizer(&optimizer)
    .scheduler(&scheduler)
    .epoch(epoch)
    .best_loss(best_loss)
    .metadata("experiment", "v1");

checkpoint.save("checkpoint.pt")?;

// Resume training
let checkpoint = Checkpoint::load("checkpoint.pt")?;
model.load_state_dict(checkpoint.model_state())?;
optimizer.load_state_dict(checkpoint.optimizer_state())?;
let start_epoch = checkpoint.epoch();
```

### Checkpoint Manager

```rust
use axonml_serialize::CheckpointManager;

let manager = CheckpointManager::new("checkpoints/")
    .keep_last(5)        // Keep last 5 checkpoints
    .save_best(true)     // Save best model separately
    .metric("val_loss")  // Track this metric
    .mode("min");        // Lower is better

for epoch in 0..100 {
    let train_loss = train_epoch();
    let val_loss = validate();

    // Automatically manages checkpoints
    manager.save(&model, &optimizer, epoch, val_loss)?;
}

// Load best model
let best_state = manager.load_best()?;
```

### SafeTensors Format

```rust
use axonml_serialize::safetensors::{save_safetensors, load_safetensors};

// Save as SafeTensors (secure, no arbitrary code execution)
save_safetensors(&model.state_dict(), "model.safetensors")?;

// Load SafeTensors
let state_dict = load_safetensors("model.safetensors")?;
model.load_state_dict(&state_dict)?;
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `save_model(model, path)` | Save complete model |
| `load_model(path)` | Load complete model |
| `save_state_dict(dict, path)` | Save state dict |
| `load_state_dict(path)` | Load state dict |

### StateDict Methods

| Method | Description |
|--------|-------------|
| `new()` | Create empty state dict |
| `insert(name, tensor)` | Add tensor |
| `get(name)` | Get tensor |
| `keys()` | List all keys |
| `save(path)` | Save to file |
| `load(path)` | Load from file |

### Checkpoint Methods

| Method | Description |
|--------|-------------|
| `model(&m)` | Add model state |
| `optimizer(&o)` | Add optimizer state |
| `epoch(e)` | Set epoch number |
| `save(path)` | Save checkpoint |
| `load(path)` | Load checkpoint |

## Part of Axonml

```toml
[dependencies]
axonml = "0.1"  # Includes serialization
```

## License

MIT OR Apache-2.0
