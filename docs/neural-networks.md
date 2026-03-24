---
layout: default
title: Neural Networks
nav_order: 4
description: "Building neural networks with AxonML"
---

# Neural Networks
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Module Trait

All neural network layers implement the `Module` trait:

```rust
pub trait Module {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<Parameter>;
    fn train(&mut self);
    fn eval(&mut self);
    fn is_training(&self) -> bool;
}
```

## Linear Layers

### Linear (Fully Connected)

```rust
use axonml::nn::Linear;

// Create linear layer: input=784, output=256
let linear = Linear::new(784, 256);

// With bias disabled
let linear = Linear::new(784, 256).bias(false);

// Forward pass
let x = Variable::new(Tensor::randn(&[32, 784]), false);
let y = linear.forward(&x);  // Shape: [32, 256]
```

## Convolutional Layers

### Conv1d

```rust
use axonml::nn::Conv1d;

// 1D convolution: in_channels=32, out_channels=64, kernel_size=3
let conv = Conv1d::new(32, 64, 3);

// With options
let conv = Conv1d::new(32, 64, 3)
    .stride(2)
    .padding(1)
    .dilation(1)
    .groups(1)
    .bias(true);

let x = Variable::new(Tensor::randn(&[16, 32, 100]), false);
let y = conv.forward(&x);  // Shape: [16, 64, 50]
```

### Conv2d

```rust
use axonml::nn::Conv2d;

let conv = Conv2d::new(3, 64, 3)
    .stride(1)
    .padding(1);

let x = Variable::new(Tensor::randn(&[16, 3, 224, 224]), false);
let y = conv.forward(&x);  // Shape: [16, 64, 224, 224]
```

## Pooling Layers

```rust
use axonml::nn::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d};

// Max pooling
let pool = MaxPool2d::new(2).stride(2);

// Average pooling
let pool = AvgPool2d::new(2).stride(2);

// Adaptive average pooling (output size)
let pool = AdaptiveAvgPool2d::new(1, 1);
```

## Normalization Layers

### BatchNorm

```rust
use axonml::nn::{BatchNorm1d, BatchNorm2d};

// For 1D inputs (e.g., after Linear)
let bn = BatchNorm1d::new(256);

// For 2D inputs (e.g., after Conv2d)
let bn = BatchNorm2d::new(64);

// With custom momentum and epsilon
let bn = BatchNorm2d::new(64)
    .momentum(0.1)
    .eps(1e-5);
```

### LayerNorm

```rust
use axonml::nn::LayerNorm;

// Normalize over last dimension
let ln = LayerNorm::new(256);

// Normalize over multiple dimensions
let ln = LayerNorm::new_dims(&[256, 256]);
```

### GroupNorm

```rust
use axonml::nn::GroupNorm;

// 32 groups, 256 channels
let gn = GroupNorm::new(32, 256);
```

### InstanceNorm

```rust
use axonml::nn::InstanceNorm2d;

// Instance normalization for style transfer
let in_norm = InstanceNorm2d::new(64);
```

## Activation Layers

```rust
use axonml::nn::{ReLU, Sigmoid, Tanh, Softmax, GELU, SiLU, LeakyReLU, ELU};

let relu = ReLU;
let sigmoid = Sigmoid;
let tanh = Tanh;
let softmax = Softmax::new(1);  // dim=1
let gelu = GELU;
let silu = SiLU;
let leaky = LeakyReLU::new(0.01);
let elu = ELU::new(1.0);
```

## Dropout

```rust
use axonml::nn::{Dropout, Dropout2d};

// Standard dropout
let dropout = Dropout::new(0.5);

// Spatial dropout (for conv layers)
let dropout2d = Dropout2d::new(0.5);

// Only active during training
model.train();
let y = dropout.forward(&x);  // Dropout applied

model.eval();
let y = dropout.forward(&x);  // Dropout disabled
```

## Recurrent Layers

### LSTM

```rust
use axonml::nn::LSTM;

// LSTM: input_size=256, hidden_size=512
let lstm = LSTM::new(256, 512);

// With options
let lstm = LSTM::new(256, 512)
    .num_layers(2)
    .bidirectional(true)
    .dropout(0.1)
    .batch_first(true);

let x = Variable::new(Tensor::randn(&[32, 100, 256]), false);
let (output, (h_n, c_n)) = lstm.forward_with_state(&x, None);
```

### GRU

```rust
use axonml::nn::GRU;

let gru = GRU::new(256, 512)
    .num_layers(2)
    .bidirectional(false);

let (output, h_n) = gru.forward_with_state(&x, None);
```

## Attention Mechanism

### Multi-Head Attention

```rust
use axonml::nn::MultiHeadAttention;

// embed_dim=512, num_heads=8
let attn = MultiHeadAttention::new(512, 8);

// With options
let attn = MultiHeadAttention::new(512, 8)
    .dropout(0.1)
    .bias(true);

let q = Variable::new(Tensor::randn(&[32, 100, 512]), false);
let k = q.clone();
let v = q.clone();

let (output, weights) = attn.forward(&q, &k, &v, None);
```

### Transformer Layers

```rust
use axonml::nn::{TransformerEncoderLayer, TransformerDecoderLayer};

// Encoder layer
let encoder_layer = TransformerEncoderLayer::new(512, 8)
    .dim_feedforward(2048)
    .dropout(0.1);

// Decoder layer
let decoder_layer = TransformerDecoderLayer::new(512, 8)
    .dim_feedforward(2048)
    .dropout(0.1);
```

## Building Models

### Sequential

```rust
use axonml::nn::Sequential;

let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Dropout::new(0.5))
    .add(Linear::new(256, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

let x = Variable::new(Tensor::randn(&[32, 784]), false);
let y = model.forward(&x);  // Shape: [32, 10]
```

### Custom Models

```rust
use axonml::nn::{Module, Linear, ReLU, BatchNorm1d};

struct MyMLP {
    fc1: Linear,
    bn1: BatchNorm1d,
    fc2: Linear,
    fc3: Linear,
    training: bool,
}

impl MyMLP {
    fn new(in_features: usize, hidden: usize, out_features: usize) -> Self {
        Self {
            fc1: Linear::new(in_features, hidden),
            bn1: BatchNorm1d::new(hidden),
            fc2: Linear::new(hidden, hidden),
            fc3: Linear::new(hidden, out_features),
            training: true,
        }
    }
}

impl Module for MyMLP {
    fn forward(&self, x: &Variable) -> Variable {
        let h = self.fc1.forward(x);
        let h = self.bn1.forward(&h);
        let h = h.relu();
        let h = self.fc2.forward(&h);
        let h = h.relu();
        self.fc3.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.bn1.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.bn1.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}
```

## Loss Functions

```rust
use axonml::nn::{MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, L1Loss};

// Mean Squared Error
let loss_fn = MSELoss::new();
let loss = loss_fn.compute(&predictions, &targets);

// Cross Entropy (for classification)
let loss_fn = CrossEntropyLoss::new();

// Binary Cross Entropy
let loss_fn = BCELoss::new();

// BCE with Logits (includes sigmoid)
let loss_fn = BCEWithLogitsLoss::new();

// L1 Loss (MAE)
let loss_fn = L1Loss::new();
```

## Saving and Loading

```rust
use axonml::serialize::{save_model, load_model};

// Save model
save_model(&model, "model.safetensors")?;

// Load model
let loaded_model = load_model::<MyMLP>("model.safetensors")?;
```
