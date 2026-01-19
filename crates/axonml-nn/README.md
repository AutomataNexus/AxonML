# axonml-nn

[![Crates.io](https://img.shields.io/crates/v/axonml-nn.svg)](https://crates.io/crates/axonml-nn)
[![Docs.rs](https://docs.rs/axonml-nn/badge.svg)](https://docs.rs/axonml-nn)
[![Downloads](https://img.shields.io/crates/d/axonml-nn.svg)](https://crates.io/crates/axonml-nn)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Neural network modules for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-nn` provides PyTorch-style neural network building blocks for the Axonml framework. It includes layers, activation functions, loss functions, and container modules that can be combined to build complex neural network architectures.

## Features

### Layer Types
- **Linear** - Fully connected layers with bias
- **Conv1d/Conv2d** - 1D and 2D convolutions with padding, stride, dilation
- **MaxPool/AvgPool** - Pooling layers for downsampling
- **BatchNorm/LayerNorm** - Normalization layers
- **Dropout** - Regularization via random zeroing

### Recurrent Layers
- **RNN** - Vanilla recurrent neural network
- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit

### Attention
- **MultiHeadAttention** - Transformer-style attention mechanism
- **Embedding** - Learnable embedding tables

### Activations
- **ReLU, LeakyReLU, ELU** - Rectified linear variants
- **Sigmoid, Tanh** - Classic activations
- **GELU, SiLU/Swish** - Modern smooth activations
- **Softmax, LogSoftmax** - Probability outputs

### Loss Functions
- **MSELoss** - Mean squared error for regression
- **CrossEntropyLoss** - Classification with softmax
- **BCELoss/BCEWithLogitsLoss** - Binary classification
- **L1Loss** - Mean absolute error
- **NLLLoss** - Negative log likelihood

### Containers
- **Sequential** - Chain modules in sequence
- **ModuleList** - List of modules
- **ModuleDict** - Dictionary of named modules

## Installation

```toml
[dependencies]
axonml-nn = "0.1"
```

## Usage

### Building a Simple MLP

```rust
use axonml_nn::{Sequential, Linear, ReLU, Module};
use axonml_tensor::randn;

// Build a 3-layer MLP
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

// Forward pass
let input = randn::<f32>(&[32, 784]);  // Batch of 32
let output = model.forward(&input);     // Shape: [32, 10]
```

### Convolutional Neural Network

```rust
use axonml_nn::{Sequential, Conv2d, MaxPool2d, Linear, ReLU, Flatten, Module};

let cnn = Sequential::new()
    // Conv block 1: 1 -> 32 channels
    .add(Conv2d::new(1, 32, 3).padding(1))
    .add(ReLU)
    .add(MaxPool2d::new(2))

    // Conv block 2: 32 -> 64 channels
    .add(Conv2d::new(32, 64, 3).padding(1))
    .add(ReLU)
    .add(MaxPool2d::new(2))

    // Classifier
    .add(Flatten)
    .add(Linear::new(64 * 7 * 7, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

// Input: [batch, channels, height, width]
let images = randn::<f32>(&[16, 1, 28, 28]);
let logits = cnn.forward(&images);  // Shape: [16, 10]
```

### LSTM for Sequence Processing

```rust
use axonml_nn::{LSTM, Linear, Module};
use axonml_tensor::randn;

// LSTM: input_size=100, hidden_size=256, num_layers=2
let lstm = LSTM::new(100, 256, 2)
    .bidirectional(true)
    .dropout(0.1);

let classifier = Linear::new(512, 10);  // 256*2 for bidirectional

// Input: [seq_len, batch, features]
let sequence = randn::<f32>(&[50, 32, 100]);

// Get LSTM output
let (output, (h_n, c_n)) = lstm.forward(&sequence, None);
// output: [50, 32, 512], h_n: [4, 32, 256], c_n: [4, 32, 256]

// Use last hidden state for classification
let last_hidden = h_n.select(0, -1);  // Last layer
let logits = classifier.forward(&last_hidden);
```

### Multi-Head Attention

```rust
use axonml_nn::{MultiHeadAttention, Module};
use axonml_tensor::randn;

// 8 attention heads, 512 embedding dimension
let attention = MultiHeadAttention::new(512, 8)
    .dropout(0.1);

// Self-attention
let x = randn::<f32>(&[32, 100, 512]);  // [batch, seq, embed]
let (attn_output, attn_weights) = attention.forward(&x, &x, &x, None);
// attn_output: [32, 100, 512]
```

### Using Normalization

```rust
use axonml_nn::{Linear, BatchNorm1d, LayerNorm, ReLU, Sequential, Module};

// BatchNorm for MLPs
let mlp_with_bn = Sequential::new()
    .add(Linear::new(784, 256))
    .add(BatchNorm1d::new(256))
    .add(ReLU)
    .add(Linear::new(256, 10));

// LayerNorm for Transformers
let layer_norm = LayerNorm::new(&[512]);  // Normalize last dim
```

### Dropout for Regularization

```rust
use axonml_nn::{Linear, Dropout, ReLU, Sequential, Module};

let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Dropout::new(0.5))  // 50% dropout
    .add(Linear::new(256, 10));

// Set training mode (dropout active)
model.train();
let train_output = model.forward(&input);

// Set eval mode (dropout disabled)
model.eval();
let eval_output = model.forward(&input);
```

### Computing Loss

```rust
use axonml_nn::{CrossEntropyLoss, MSELoss, Module};
use axonml_tensor::{randn, Tensor};

// Classification loss
let ce_loss = CrossEntropyLoss::new();
let logits = randn::<f32>(&[32, 10]);
let targets = Tensor::<i64>::from_vec((0..32).map(|i| i % 10).collect(), &[32]).unwrap();
let loss = ce_loss.forward(&logits, &targets);

// Regression loss
let mse_loss = MSELoss::new();
let predictions = randn::<f32>(&[32, 1]);
let targets = randn::<f32>(&[32, 1]);
let loss = mse_loss.forward(&predictions, &targets);
```

### Accessing Parameters

```rust
use axonml_nn::{Linear, Module};

let layer = Linear::new(100, 50);

// Get all parameters (for optimizer)
let params = layer.parameters();
println!("Number of parameters: {}", params.len());

// Get named parameters
for (name, param) in layer.named_parameters() {
    println!("{}: shape {:?}", name, param.shape());
}

// Count total parameters
let total: usize = params.iter().map(|p| p.numel()).sum();
println!("Total parameters: {}", total);
```

### Custom Module

```rust
use axonml_nn::{Module, Linear, ReLU, Parameter};
use axonml_tensor::Tensor;
use axonml_autograd::Variable;

struct MyModule {
    fc1: Linear,
    fc2: Linear,
}

impl MyModule {
    fn new(in_features: usize, hidden: usize, out_features: usize) -> Self {
        Self {
            fc1: Linear::new(in_features, hidden),
            fc2: Linear::new(hidden, out_features),
        }
    }
}

impl Module for MyModule {
    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let h = self.fc1.forward(x).relu();
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}
```

## API Reference

### Layers

| Layer | Description |
|-------|-------------|
| `Linear` | Fully connected layer: y = xW + b |
| `Conv1d` | 1D convolution for sequences |
| `Conv2d` | 2D convolution for images |
| `Conv3d` | 3D convolution for volumes |
| `ConvTranspose2d` | Transposed convolution (upsampling) |

### Pooling

| Layer | Description |
|-------|-------------|
| `MaxPool1d/2d/3d` | Max pooling |
| `AvgPool1d/2d/3d` | Average pooling |
| `AdaptiveMaxPool2d` | Adaptive max pool to target size |
| `AdaptiveAvgPool2d` | Adaptive avg pool to target size |

### Normalization

| Layer | Description |
|-------|-------------|
| `BatchNorm1d/2d/3d` | Batch normalization |
| `LayerNorm` | Layer normalization |
| `GroupNorm` | Group normalization |
| `InstanceNorm2d` | Instance normalization |

### Recurrent

| Layer | Description |
|-------|-------------|
| `RNN` | Vanilla RNN |
| `LSTM` | Long Short-Term Memory |
| `GRU` | Gated Recurrent Unit |

### Attention

| Layer | Description |
|-------|-------------|
| `MultiHeadAttention` | Multi-head self/cross attention |
| `Embedding` | Learnable lookup table |

### Activations

| Activation | Formula |
|------------|---------|
| `ReLU` | max(0, x) |
| `LeakyReLU` | max(αx, x) |
| `ELU` | x if x>0, α(e^x-1) otherwise |
| `Sigmoid` | 1/(1+e^-x) |
| `Tanh` | tanh(x) |
| `GELU` | x·Φ(x) |
| `SiLU` | x·σ(x) |
| `Softmax` | e^x / Σe^x |

### Loss Functions

| Loss | Use Case |
|------|----------|
| `MSELoss` | Regression |
| `L1Loss` | Robust regression |
| `CrossEntropyLoss` | Multi-class classification |
| `BCELoss` | Binary classification |
| `BCEWithLogitsLoss` | Binary with numerical stability |
| `NLLLoss` | When using LogSoftmax |

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

```toml
[dependencies]
axonml = "0.1"  # Includes axonml-nn
```

## License

MIT OR Apache-2.0
