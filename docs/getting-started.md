---
layout: default
title: Getting Started
nav_order: 2
description: "Installation and setup guide for AxonML"
---

# Getting Started
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before using AxonML, ensure you have:

- **Rust** 1.75+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Cargo** (included with Rust)

Optional for GPU acceleration:
- **CUDA Toolkit** 12.0+ (for NVIDIA GPUs)
- **Vulkan SDK** (for cross-platform GPU)

## Installation

### As a Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml = "0.2.8"
```

### Feature Flags

AxonML uses feature flags to control what gets compiled:

| Feature | Description | Default |
|:--------|:------------|:--------|
| `full` | All features enabled | Yes |
| `core` | Core tensor operations | Yes |
| `nn` | Neural network modules | Yes |
| `vision` | Computer vision (MNIST, CIFAR, ResNet) | Yes |
| `audio` | Audio processing (MelSpectrogram, MFCC) | Yes |
| `text` | Text processing (Tokenizers, BPE) | Yes |
| `llm` | Large language models (BERT, GPT-2) | Yes |
| `distributed` | Distributed training (DDP, FSDP) | Yes |
| `cuda` | CUDA GPU backend | No |
| `wgpu` | WebGPU/Vulkan backend | No |

Example with specific features:

```toml
[dependencies]
axonml = { version = "0.2.8", default-features = false, features = ["core", "nn", "cuda"] }
```

### CLI Installation

Install the AxonML CLI:

```bash
cargo install axonml-cli
```

## Your First Model

### 1. Create a New Project

```bash
cargo new my_ml_project
cd my_ml_project
```

### 2. Add Dependencies

Edit `Cargo.toml`:

```toml
[dependencies]
axonml = "0.2.8"
```

### 3. Write Your Model

Edit `src/main.rs`:

```rust
use axonml::prelude::*;

fn main() {
    // Create random training data
    let x_train = Tensor::randn(&[100, 2]);  // 100 samples, 2 features
    let y_train = Tensor::randn(&[100, 1]);  // 100 labels

    // Define a simple neural network
    let model = Sequential::new()
        .add(Linear::new(2, 16))
        .add(ReLU)
        .add(Linear::new(16, 1));

    // Create optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.01);

    // Loss function
    let loss_fn = MSELoss::new();

    // Training loop
    println!("Training...");
    for epoch in 0..100 {
        // Forward pass
        let x = Variable::new(x_train.clone(), false);
        let y = Variable::new(y_train.clone(), false);
        let pred = model.forward(&x);

        // Compute loss
        let loss = loss_fn.compute(&pred, &y);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.data().item());
        }
    }

    println!("Training complete!");
}
```

### 4. Run Your Model

```bash
cargo run --release
```

Expected output:
```
Training...
Epoch 0: Loss = 1.234567
Epoch 10: Loss = 0.567890
Epoch 20: Loss = 0.234567
...
Epoch 90: Loss = 0.012345
Training complete!
```

## MNIST Example

A complete MNIST digit classification example:

```rust
use axonml::prelude::*;
use axonml::vision::{MNIST, transforms};
use axonml::data::DataLoader;

fn main() {
    // Load MNIST dataset
    let train_dataset = MNIST::new("./data", true)
        .transform(transforms::Normalize::new(0.1307, 0.3081));

    let test_dataset = MNIST::new("./data", false)
        .transform(transforms::Normalize::new(0.1307, 0.3081));

    let train_loader = DataLoader::new(train_dataset, 64, true);
    let test_loader = DataLoader::new(test_dataset, 64, false);

    // Define CNN model
    let model = Sequential::new()
        .add(Conv2d::new(1, 32, 3).padding(1))
        .add(ReLU)
        .add(MaxPool2d::new(2))
        .add(Conv2d::new(32, 64, 3).padding(1))
        .add(ReLU)
        .add(MaxPool2d::new(2))
        .add(Flatten)
        .add(Linear::new(64 * 7 * 7, 128))
        .add(ReLU)
        .add(Linear::new(128, 10));

    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::new();

    // Training
    for epoch in 0..10 {
        model.train();
        let mut total_loss = 0.0;

        for (batch_idx, (images, labels)) in train_loader.iter().enumerate() {
            let x = Variable::new(images, false);
            let y = labels;

            let output = model.forward(&x);
            let loss = loss_fn.compute(&output, &y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data().item();
        }

        // Evaluation
        model.eval();
        let mut correct = 0;
        let mut total = 0;

        for (images, labels) in test_loader.iter() {
            let x = Variable::new(images, false);
            let output = model.forward(&x);
            let predictions = output.data().argmax(1);

            for (pred, label) in predictions.iter().zip(labels.iter()) {
                if pred == label {
                    correct += 1;
                }
                total += 1;
            }
        }

        let accuracy = 100.0 * correct as f32 / total as f32;
        println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%",
                 epoch, total_loss, accuracy);
    }
}
```

## GPU Acceleration

### CUDA

Enable CUDA in `Cargo.toml`:

```toml
axonml = { version = "0.2.8", features = ["cuda"] }
```

Use GPU in code:

```rust
use axonml::core::Device;

// Move tensor to GPU
let x = Tensor::randn(&[1000, 1000]);
let x_gpu = x.to(Device::CUDA(0));

// Operations run on GPU
let y_gpu = x_gpu.matmul(&x_gpu.t());

// Move back to CPU
let y_cpu = y_gpu.to(Device::CPU);
```

### WebGPU (Cross-platform)

```toml
axonml = { version = "0.2.8", features = ["wgpu"] }
```

```rust
let device = Device::WebGPU(0);
let x = Tensor::randn(&[1000, 1000]).to(device);
```

## Next Steps

- [Tensor Operations]({% link tensors.md %}) - Deep dive into tensor API
- [Neural Networks]({% link neural-networks.md %}) - Building complex models
- [Training]({% link training.md %}) - Optimizers, schedulers, mixed precision
- [Distributed]({% link distributed.md %}) - Multi-GPU training
- [ONNX]({% link onnx.md %}) - Import/export models
