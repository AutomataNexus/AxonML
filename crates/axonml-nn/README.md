# axonml-nn

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/version-0.6.5-green.svg" alt="Version 0.6.5">
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

**axonml-nn** provides neural network building blocks for the AxonML framework: the `Module` trait, the `Parameter` wrapper, the `Sequential` container, and a broad catalog of layers (dense, conv, pooling, norm, recurrent, attention, transformer, MoE, ternary, sparse, graph, FFT/STFT), activations, losses, and weight initializers.

## Features

- **Module Trait** - Core interface for all neural network components with parameter collection, train/eval mode, and `zero_grad`.

- **Dense and Conv Layers** - `Linear`, `Conv1d`, `Conv2d`, `ConvTranspose2d`, `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `AdaptiveAvgPool2d`.

- **Normalization** - `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `InstanceNorm2d`.

- **Recurrent Networks** - `RNN`, `LSTM`, `GRU` plus single-step cell variants (`RNNCell`, `LSTMCell`, `GRUCell`). Timesteps processed with batched matmul to avoid per-step allocation overhead.

- **Attention and Transformers** - `MultiHeadAttention`, `CrossAttention`, `DifferentialAttention`, `TransformerEncoderLayer` / `TransformerDecoderLayer`, `TransformerEncoder` / `TransformerDecoder`, and `Seq2SeqTransformer` end-to-end.

- **Mixture of Experts** - `MoELayer`, `MoERouter`, and `Expert` for sparse expert routing.

- **1.58-bit ternary quantized linear (BitNet b1.58)** - ternary weights in {−1, 0, +1} for ~16× weight compression. GPU forward + backward kernels with GPU-resident ternary quantization, plus `saved_input` host-staging during backward to cut peak VRAM at billion-parameter scale.

- **Graph Neural Networks** - `GCNConv` and `GATConv` for graph convolution / attention.

- **Spectral Layers** - `FFT1d` and `STFT` via rustfft for frequency-domain processing.

- **Differentiable Structured Sparsity** - `SparseLinear` (soft-thresholded magnitude pruning), `GroupSparsity` (row/col L1 regularization), `LotteryTicket` (snapshot/prune/rewind).

- **Other Building Blocks** - `Embedding`, `Dropout`, `Dropout2d`, `ResidualBlock`.

- **Activations** - `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `ELU`, `Softmax`, `LogSoftmax`, `Identity`, `Flatten`.

- **Losses** - `MSELoss`, `L1Loss`, `SmoothL1Loss`, `CrossEntropyLoss`, `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, with `Reduction` (`Mean` / `Sum` / `None`).

- **Weight Initialization** - `xavier_uniform`, `xavier_normal`, `glorot_uniform`, `glorot_normal`, `kaiming_uniform`, `kaiming_normal`, `he_uniform`, `he_normal`, `orthogonal`, `sparse`, `uniform`, `uniform_range`, `normal`, `constant`, `zeros`, `ones`, `eye`, `diag`, plus the `InitMode` enum.

- **Sequential Container** - `Sequential::new().add(...)` for quick model composition; also `ModuleList` for heterogeneous collections.

## Modules

| Module | Description |
|--------|-------------|
| `module` | `Module` trait and `ModuleList` container |
| `parameter` | `Parameter` wrapper for learnable weights with gradient tracking |
| `sequential` | `Sequential` container for chaining modules |
| `layers` | All layer types (see `layers/` submodules: `linear`, `conv`, `pooling`, `norm`, `rnn`, `attention`, `diff_attention`, `transformer`, `embedding`, `dropout`, `residual`, `moe`, `sparse`, `graph`, `fft`) |
| `activation` | Activation function modules |
| `loss` | Loss function modules and `Reduction` enum |
| `init` | Weight initialization functions and `InitMode` |
| `functional` | Stateless functional versions of common operations |

## Cargo Features

| Feature | Purpose |
|---------|---------|
| `cuda` | Forwards CUDA support to tensor / core |
| `cudnn` | Forwards cuDNN support (implies `cuda`) |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-nn = "0.6.5"
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

### Differentiable Structured Sparsity

Learn which weights to prune end-to-end — the pruning mask is differentiable.

```rust
use axonml_nn::layers::sparse::{SparseLinear, GroupSparsity, LotteryTicket};
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// SparseLinear: soft-thresholded magnitude pruning
let sparse = SparseLinear::new(256, 128, 0.5, 10.0); // threshold=0.5, temperature=10.0
let input = Variable::new(Tensor::randn(&[4, 256]), true);
let output = sparse.forward(&input); // [4, 128]

// Check actual sparsity
let sparsity = sparse.sparsity(); // fraction of weights effectively pruned
println!("Sparsity: {:.1}%", sparsity * 100.0);

// Group sparsity regularization
let group_reg = GroupSparsity::new(0.01, "row"); // L1 penalty on row norms
let reg_loss = group_reg.compute(&sparse);

// Lottery Ticket Hypothesis
let mut ticket = LotteryTicket::new(&sparse);
ticket.snapshot(); // save initial weights
// ... train for a while ...
ticket.prune(0.2); // prune bottom 20% by magnitude
ticket.rewind(&mut sparse); // rewind to initial weights with discovered mask
```

### Mixture of Experts

```rust
use axonml_nn::layers::moe::MoELayer;

// Sparse top-k expert routing
let moe = MoELayer::new(
    /* in_features */   512,
    /* out_features */  512,
    /* num_experts */    8,
    /* top_k */          2,
);
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

---

_Last updated: 2026-06-06 (v0.6.5)_
