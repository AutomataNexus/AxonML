<p align="center">
  <img src="../../assets/axonml-logo.png" alt="AxonML Logo" width="200"/>
</p>

<h1 align="center">axonml</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Crates.io: 0.1.0"></a>
  <a href="https://github.com/automatanexus/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-teal.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

**axonml** is the umbrella crate for the AxonML machine learning framework - a complete ML/AI toolkit written in pure Rust. It provides PyTorch-equivalent functionality including tensors, automatic differentiation, neural networks, optimizers, data loading, and domain-specific utilities for vision, text, and audio processing.

This crate re-exports all AxonML sub-crates under a unified API, allowing you to build and train deep learning models with a single import.

---

## Features

- **Tensors** - N-dimensional arrays with broadcasting, views, slicing, and BLAS-accelerated operations
- **Automatic Differentiation** - Computational graph construction with backward pass for gradient computation
- **Neural Networks** - Linear, Conv2d, BatchNorm, LayerNorm, Attention, RNN/LSTM/GRU, and more
- **Optimizers** - SGD, Adam, AdamW, RMSprop with learning rate schedulers (Step, Exponential, Cosine)
- **Data Loading** - Dataset trait, DataLoader with batching, samplers, and transforms
- **Vision** - Image transforms, MNIST/CIFAR datasets, CNN architectures (LeNet, SimpleCNN)
- **Text** - Tokenizers (BPE, WordPiece, Whitespace), vocabularies, text datasets
- **Audio** - Spectrograms, MFCC, audio transforms, synthetic audio datasets
- **Distributed** - DDP, all-reduce, broadcast, barrier, process groups
- **Profiling** - Memory and compute profilers, timeline analysis, bottleneck detection
- **LLM Architectures** - BERT and GPT-2 implementations with generation utilities
- **JIT Compilation** - Graph tracing and optimization for compiled inference

---

## Modules (Re-exports)

| Module | Description | Feature Flag |
|--------|-------------|--------------|
| `core` | Error types, Device, DType definitions | `core` |
| `tensor` | N-dimensional tensor operations | `core` |
| `autograd` | Automatic differentiation and Variables | `core` |
| `nn` | Neural network layers and modules | `nn` |
| `optim` | Optimizers and learning rate schedulers | `nn` |
| `data` | DataLoader, Dataset trait, samplers, transforms | `data` |
| `vision` | Image transforms and vision datasets | `vision` |
| `text` | Tokenizers, vocabularies, text datasets | `text` |
| `audio` | Audio transforms and audio datasets | `audio` |
| `distributed` | Distributed training utilities (DDP) | `distributed` |
| `profile` | Profiling and bottleneck analysis | `profile` |
| `llm` | BERT and GPT-2 architectures | `llm` |
| `jit` | JIT compilation and graph optimization | `jit` |

---

## Usage

Add `axonml` to your `Cargo.toml`:

```toml
[dependencies]
axonml = "0.1.0"
```

### Using the Prelude

The prelude module re-exports commonly used types for quick access:

```rust
use axonml::prelude::*;

fn main() -> axonml::core::Result<()> {
    // Create a tensor
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    // Wrap in a Variable for autograd
    let var = Variable::new(x, true);

    // Build a simple model
    let model = Sequential::new()
        .add(Linear::new(784, 128))
        .add(ReLU)
        .add(Linear::new(128, 10));

    // Create an optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    Ok(())
}
```

### Training Loop Example

```rust
use axonml::prelude::*;

fn train() -> axonml::core::Result<()> {
    // Create model
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Dropout::new(0.2))
        .add(Linear::new(256, 10));

    // Optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    // Dataset and loader
    let dataset = SyntheticMNIST::new(1000);
    let loader = DataLoader::new(dataset, 32);

    // Training loop
    for epoch in 0..10 {
        for batch in loader.iter() {
            // Forward pass
            let output = model.forward(&batch.data);
            let loss = CrossEntropyLoss::new().forward(&output, &batch.labels);

            // Backward pass
            loss.backward();

            // Update weights
            optimizer.step();
            optimizer.zero_grad();
        }
        println!("Epoch {} complete", epoch + 1);
    }

    Ok(())
}
```

### Feature Flags

Select only the features you need:

```toml
[dependencies]
# Full framework (default)
axonml = "0.1.0"

# Core only (tensors + autograd)
axonml = { version = "0.1.0", default-features = false, features = ["core"] }

# Neural networks without domain-specific modules
axonml = { version = "0.1.0", default-features = false, features = ["nn", "data"] }

# Vision pipeline
axonml = { version = "0.1.0", default-features = false, features = ["vision"] }

# NLP pipeline
axonml = { version = "0.1.0", default-features = false, features = ["text", "llm"] }
```

---

## Feature Flag Reference

| Feature | Includes | Description |
|---------|----------|-------------|
| `full` | All features | Complete framework (default) |
| `core` | tensor, autograd | Core tensor operations and autodiff |
| `nn` | core + nn, optim | Neural network layers and optimizers |
| `data` | core + data | DataLoader and dataset utilities |
| `vision` | nn, data + vision | Image processing and vision datasets |
| `text` | nn, data + text | Tokenizers and text processing |
| `audio` | nn, data + audio | Audio transforms and datasets |
| `distributed` | nn + distributed | Distributed training (DDP) |
| `profile` | core + profile | Profiling and analysis tools |
| `llm` | nn + llm | BERT and GPT-2 architectures |
| `jit` | core + jit | JIT compilation and tracing |

---

## Version Information

```rust
use axonml::{version, features};

fn main() {
    println!("AxonML version: {}", version());
    println!("Enabled features: {}", features());
}
```

---

## Tests

Run the full test suite:

```bash
cargo test -p axonml
```

Run tests for specific features:

```bash
cargo test -p axonml --features "core nn data"
```

Run integration tests:

```bash
cargo test -p axonml --test integration_test
```

---

## Examples

The crate includes example programs:

```bash
# Simple training example
cargo run -p axonml --example simple_training

# MNIST digit classification
cargo run -p axonml --example mnist_training

# NLP and audio feature test
cargo run -p axonml --example nlp_audio_test
```

---

## Sub-Crate Structure

```
axonml/
  axonml-core       - Error types, Device, DType
  axonml-tensor     - N-dimensional tensor operations
  axonml-autograd   - Automatic differentiation
  axonml-nn         - Neural network layers
  axonml-optim      - Optimizers and schedulers
  axonml-data       - DataLoader and Dataset trait
  axonml-vision     - Image transforms and datasets
  axonml-text       - Tokenizers and text utilities
  axonml-audio      - Audio transforms and datasets
  axonml-distributed- Distributed training (DDP)
  axonml-profile    - Profiling and analysis
  axonml-llm        - BERT and GPT-2 models
  axonml-jit        - JIT compilation
  axonml-serialize  - Model serialization
  axonml-tui        - Terminal user interface
  axonml (this)     - Umbrella crate
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
