# axonml-nn

<p align="center">
  <!-- Logo placeholder -->
  <img src="../../assets/logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

**axonml-nn** provides neural network building blocks for the AxonML framework. It includes layers, activation functions, loss functions, and utilities for constructing and training deep learning models with a PyTorch-like API.

## Features

- **Module Trait** - Core interface for all neural network components with parameter management and train/eval modes.

- **Comprehensive Layers** - Linear, Conv1d/Conv2d, RNN/LSTM/GRU, Embedding, BatchNorm, LayerNorm, Dropout, and MultiHeadAttention.

- **Activation Functions** - ReLU, Sigmoid, Tanh, GELU, SiLU, ELU, LeakyReLU, Softmax, and LogSoftmax.

- **Loss Functions** - MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, NLLLoss, L1Loss, and SmoothL1Loss.

- **Weight Initialization** - Xavier/Glorot, Kaiming/He, orthogonal, sparse, and custom initialization schemes.

- **Sequential Container** - Easy model composition by chaining layers together.

## Modules

| Module | Description |
|--------|-------------|
| `module` | Core `Module` trait and `ModuleList` container for neural network components |
| `parameter` | `Parameter` wrapper for learnable weights with gradient tracking |
| `sequential` | `Sequential` container for chaining modules in order |
| `layers` | Neural network layers (Linear, Conv, RNN, Attention, Norm, Pooling, Embedding, Dropout) |
| `activation` | Activation function modules (ReLU, Sigmoid, Tanh, GELU, etc.) |
| `loss` | Loss function modules (MSE, CrossEntropy, BCE, etc.) |
| `init` | Weight initialization functions (Xavier, Kaiming, orthogonal, etc.) |
| `functional` | Stateless functional versions of operations |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-nn = "0.1.0"
```

### Building a Simple MLP

```rust
use axonml_nn::prelude::*;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// Build model using Sequential
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

// Create input
let input = Variable::new(
    Tensor::from_vec(vec![0.5; 784], &[1, 784]).unwrap(),
    false
);

// Forward pass
let output = model.forward(&input);
assert_eq!(output.shape(), vec![1, 10]);

// Get all parameters
let params = model.parameters();
println!("Total parameters: {}", model.num_parameters());
```

### Convolutional Neural Network

```rust
use axonml_nn::prelude::*;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let model = Sequential::new()
    .add(Conv2d::new(1, 32, 3))      // [B, 1, 28, 28] -> [B, 32, 26, 26]
    .add(ReLU)
    .add(MaxPool2d::new(2))          // -> [B, 32, 13, 13]
    .add(Conv2d::new(32, 64, 3))     // -> [B, 64, 11, 11]
    .add(ReLU)
    .add(MaxPool2d::new(2));         // -> [B, 64, 5, 5]

let input = Variable::new(
    Tensor::from_vec(vec![0.5; 784], &[1, 1, 28, 28]).unwrap(),
    false
);
let features = model.forward(&input);
```

### Recurrent Neural Network

```rust
use axonml_nn::prelude::*;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// LSTM for sequence modeling
let lstm = LSTM::new(
    64,   // input_size
    128,  // hidden_size
    2     // num_layers
);

// Input: [batch, seq_len, input_size]
let input = Variable::new(
    Tensor::from_vec(vec![0.5; 640], &[2, 5, 64]).unwrap(),
    false
);
let output = lstm.forward(&input);  // [2, 5, 128]
```

### Transformer Attention

```rust
use axonml_nn::prelude::*;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let attention = MultiHeadAttention::new(
    512,  // embed_dim
    8     // num_heads
);

// Input: [batch, seq_len, embed_dim]
let input = Variable::new(
    Tensor::from_vec(vec![0.5; 5120], &[2, 5, 512]).unwrap(),
    false
);
let output = attention.forward(&input);  // [2, 5, 512]
```

### Loss Functions

```rust
use axonml_nn::prelude::*;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// Cross Entropy Loss for classification
let logits = Variable::new(
    Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap(),
    true
);
let targets = Variable::new(
    Tensor::from_vec(vec![2.0, 0.0], &[2]).unwrap(),
    false
);

let loss_fn = CrossEntropyLoss::new();
let loss = loss_fn.compute(&logits, &targets);
loss.backward();

// MSE Loss for regression
let mse = MSELoss::new();
let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), true);
let target = Variable::new(Tensor::from_vec(vec![1.5, 2.5], &[2]).unwrap(), false);
let loss = mse.compute(&pred, &target);
```

### Weight Initialization

```rust
use axonml_nn::init::*;

// Xavier/Glorot initialization
let weights = xavier_uniform(256, 128);
let weights = xavier_normal(256, 128);

// Kaiming/He initialization (for ReLU networks)
let weights = kaiming_uniform(256, 128);
let weights = kaiming_normal(256, 128);

// Other initializations
let zeros_tensor = zeros(&[3, 3]);
let ones_tensor = ones(&[3, 3]);
let eye_tensor = eye(4);
let ortho_tensor = orthogonal(64, 64);
```

### Training/Evaluation Mode

```rust
use axonml_nn::prelude::*;

let mut model = Sequential::new()
    .add(Linear::new(10, 5))
    .add(Dropout::new(0.5))
    .add(Linear::new(5, 2));

// Training mode (dropout active)
model.train();
assert!(model.is_training());

// Evaluation mode (dropout disabled)
model.eval();
assert!(!model.is_training());

// Zero gradients before backward pass
model.zero_grad();
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-nn
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
