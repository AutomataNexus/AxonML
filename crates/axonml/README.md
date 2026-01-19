# Axonml

[![Crates.io](https://img.shields.io/crates/v/axonml.svg)](https://crates.io/crates/axonml)
[![Docs.rs](https://docs.rs/axonml/badge.svg)](https://docs.rs/axonml)
[![Downloads](https://img.shields.io/crates/d/axonml.svg)](https://crates.io/crates/axonml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> A complete, PyTorch-equivalent machine learning framework in pure Rust.

## Overview

Axonml is a comprehensive machine learning framework providing everything you need to build, train, and deploy neural networks in Rust. From basic tensors to large language models, Axonml offers a PyTorch-like API with Rust's safety and performance.

## Features

### Core
- **Tensors** - N-dimensional arrays with broadcasting
- **Autograd** - Automatic differentiation with computational graph
- **GPU Support** - CUDA and Vulkan backends

### Neural Networks
- **Layers** - Linear, Conv1d/2d/3d, RNN, LSTM, GRU, Attention
- **Activations** - ReLU, GELU, Sigmoid, Tanh, Softmax, SiLU
- **Normalization** - BatchNorm, LayerNorm, GroupNorm
- **Pooling** - MaxPool, AvgPool, AdaptivePool

### Optimizers
- **SGD** - With momentum and Nesterov
- **Adam** - Adaptive moment estimation
- **AdamW** - Adam with decoupled weight decay
- **RMSprop** - Root mean square propagation

### Data Loading
- **Dataset trait** - Custom dataset support
- **DataLoader** - Batching, shuffling, parallel loading
- **Transforms** - Image, audio, and text transforms

### Domain-Specific
- **Vision** - ResNet, VGG, ViT, image transforms, MNIST/CIFAR
- **Audio** - Spectrogram, MFCC, mel filterbanks
- **Text** - Tokenization, vocabularies, embeddings
- **LLM** - BERT, GPT-2, LLaMA architectures

### Production
- **Serialization** - Save/load models, SafeTensors
- **ONNX** - Import/export ONNX models
- **Quantization** - INT8/INT4 compression
- **JIT** - Just-in-time compilation

## Installation

```toml
[dependencies]
axonml = "0.1"
```

### Feature Flags

```toml
# Full installation with all features
axonml = { version = "0.1", features = ["full"] }

# Select specific features
axonml = { version = "0.1", features = ["cuda", "vision", "llm"] }
```

| Feature | Description |
|---------|-------------|
| `cuda` | NVIDIA CUDA backend |
| `vulkan` | Vulkan compute backend |
| `vision` | Computer vision (torchvision equivalent) |
| `audio` | Audio processing |
| `text` | NLP utilities |
| `llm` | Large language models |
| `distributed` | Multi-GPU/node training |
| `jit` | JIT compilation |
| `profile` | Performance profiling |
| `full` | All features |

## Quick Start

### Basic Tensor Operations

```rust
use axonml::prelude::*;

// Create tensors
let a = Tensor::randn(&[3, 4]);
let b = Tensor::randn(&[4, 5]);

// Operations
let c = a.matmul(&b);      // Matrix multiplication
let d = c.relu();          // Activation
let e = d.softmax(-1);     // Softmax along last dim

println!("Shape: {:?}", e.shape());  // [3, 5]
```

### Training a Neural Network

```rust
use axonml::prelude::*;

// Define model
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Dropout::new(0.2))
    .add(Linear::new(256, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

// Optimizer
let mut optimizer = Adam::new(model.parameters(), 0.001);

// Training loop
for epoch in 0..num_epochs {
    for batch in train_loader.iter() {
        let output = model.forward(&batch.data);
        let loss = output.cross_entropy(&batch.targets);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
```

### Computer Vision

```rust
use axonml::prelude::*;
use axonml::vision::{ResNet18, transforms, datasets};

// Load pretrained model
let model = ResNet18::pretrained()?;
model.eval();

// Image transforms
let transform = transforms::Compose::new(vec![
    transforms::Resize::new(256),
    transforms::CenterCrop::new(224),
    transforms::ToTensor::new(),
    transforms::Normalize::imagenet(),
]);

// Load and transform image
let image = transforms::load_image("cat.jpg")?;
let input = transform.apply(&image).unsqueeze(0);

// Inference
let output = model.forward(&input);
let prediction = output.argmax(-1);
```

### Large Language Models

```rust
use axonml::llm::{GPT2, GPT2Tokenizer, GenerationConfig};

// Load model
let model = GPT2::from_pretrained("gpt2")?;
let tokenizer = GPT2Tokenizer::from_pretrained("gpt2")?;

// Generate text
let input_ids = tokenizer.encode("The future of AI is")?;

let config = GenerationConfig {
    max_new_tokens: 50,
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};

let output_ids = model.generate(&input_ids, &config)?;
println!("{}", tokenizer.decode(&output_ids)?);
```

### Quantization

```rust
use axonml::prelude::*;
use axonml::quant::{quantize, QuantFormat};

// Quantize model to INT4
let quantized = quantize(&model, QuantFormat::Q4_0)?;

// ~8x smaller, faster inference
println!("Original: {} MB", model.size_mb());
println!("Quantized: {} MB", quantized.size_mb());

quantized.save("model_q4.axonml")?;
```

### Distributed Training

```rust
use axonml::distributed::{init_process_group, DistributedDataParallel};

// Initialize
init_process_group("nccl", "env://")?;

// Wrap model in DDP
let model = create_model().to_device(get_rank());
let ddp_model = DistributedDataParallel::new(model);

// Training (gradients auto-synchronized)
for batch in dataloader.iter() {
    let output = ddp_model.forward(&batch.data);
    let loss = compute_loss(&output, &batch.targets);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

## Crate Structure

Axonml is composed of modular crates that can be used independently:

| Crate | Description |
|-------|-------------|
| `axonml-core` | Device, storage, dtypes |
| `axonml-tensor` | Tensor operations |
| `axonml-autograd` | Automatic differentiation |
| `axonml-nn` | Neural network modules |
| `axonml-optim` | Optimizers |
| `axonml-data` | Data loading |
| `axonml-vision` | Computer vision |
| `axonml-audio` | Audio processing |
| `axonml-text` | NLP utilities |
| `axonml-llm` | Large language models |
| `axonml-serialize` | Model serialization |
| `axonml-onnx` | ONNX support |
| `axonml-quant` | Quantization |
| `axonml-fusion` | Kernel fusion |
| `axonml-distributed` | Distributed training |
| `axonml-jit` | JIT compilation |
| `axonml-profile` | Profiling |
| `axonml-cli` | Command-line interface |
| `axonml-tui` | Terminal UI |

## CLI Usage

```bash
# Install CLI
cargo install axonml-cli

# Train model
axonml train config.toml

# Run inference
axonml run model.axonml --input data.tensor

# Quantize model
axonml quant convert model.axonml --type q4_0 -o model_q4.axonml

# Benchmark
axonml benchmark model.axonml --batch-size 32
```

## Documentation

- [API Documentation](https://docs.rs/axonml)
- [GitHub Repository](https://github.com/AutomataNexus/AxonML)
- [Examples](https://github.com/AutomataNexus/AxonML/tree/main/examples)

## License

MIT OR Apache-2.0
