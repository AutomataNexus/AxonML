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

> **Performance (0.6.5):** layer forward and backward run device-native — GPU
> tensors stay on-device, and on CPU the matmul / norm / activation / attention
> math and the full GradFn backward family are rayon-parallel (see
> [Training → Device-Native Execution](training.md)). The 1.58-bit ternary
> quantized linear (BitNet b1.58) additionally has GPU forward + backward
> kernels with host-staged activations for billion-parameter training.

## Module Trait

All neural network layers implement the `Module` trait in `axonml_nn::module`:

```rust
use axonml_autograd::Variable;
use axonml_nn::Parameter;
use std::collections::HashMap;

pub trait Module: Send + Sync {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<Parameter> { Vec::new() }
    fn named_parameters(&self) -> HashMap<String, Parameter> { HashMap::new() }
    fn num_parameters(&self) -> usize { /* sum of requires_grad params */ 0 }
    fn train(&mut self) { self.set_training(true); }
    fn eval(&mut self) { self.set_training(false); }
    fn set_training(&mut self, _training: bool) { /* no-op by default */ }
    fn is_training(&self) -> bool { true }
    fn zero_grad(&self) { /* zero all parameter grads */ }
}
```

Stateless modules (Linear, Conv, activations) leave `set_training` / `is_training` at their defaults. Dropout and BatchNorm override both to track the mode.

## Linear Layers

### Linear (Fully Connected)

```rust
use axonml_nn::{Linear, Module};
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// in_features=784, out_features=256 (with bias)
let linear = Linear::new(784, 256);

// No bias — note: the method is `with_bias`, not a `.bias(false)` builder
let linear_no_bias = Linear::with_bias(784, 256, false);

// From pre-computed weights (e.g. loaded from a checkpoint)
// let linear_loaded = Linear::from_weights(weight_tensor, Some(bias_tensor));

let x = Variable::new(Tensor::<f32>::randn(&[32, 784]), false);
let y = linear.forward(&x);                        // [32, 256]
```

## Convolutional Layers

`Conv1d`, `Conv2d`, and `ConvTranspose2d` all follow the same shape: `new(in, out, kernel)` for sensible defaults, or `with_options(...)` for stride / padding / bias.

### Conv1d

```rust
use axonml_nn::Conv1d;

// in=32, out=64, kernel=3 (stride=1, padding=0, bias=true)
let conv = Conv1d::new(32, 64, 3);

// All options (stride=2, padding=1, bias=true)
let conv = Conv1d::with_options(32, 64, 3, 2, 1, true);
```

### Conv2d

```rust
use axonml_nn::Conv2d;

let conv = Conv2d::new(3, 64, 3);
let conv_padded = Conv2d::with_options(3, 64, 3, /*stride=*/1, /*padding=*/1, /*bias=*/true);

let x = Variable::new(Tensor::<f32>::randn(&[16, 3, 224, 224]), false);
let y = conv_padded.forward(&x);                   // [16, 64, 224, 224]
```

`Conv2d` has a GPU-resident cuDNN / cuBLAS convolution fast path when the `cuda` feature is enabled and inputs / weights live on a GPU device.

## Pooling Layers

```rust
use axonml_nn::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d};

// Max pool kernel=2 (stride=kernel, padding=0)
let pool = MaxPool2d::new(2);
let pool_full = MaxPool2d::with_options(2, /*stride=*/2, /*padding=*/0);

// Avg pool
let avg = AvgPool2d::new(2);

// Adaptive — output size (H, W)
let gap = AdaptiveAvgPool2d::new((1, 1));
let square_gap = AdaptiveAvgPool2d::square(1);
```

`MaxPool1d`, `AvgPool1d` exist as well.

## Normalization Layers

```rust
use axonml_nn::{BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm2d};

let bn1 = BatchNorm1d::new(256);
let bn1_opts = BatchNorm1d::with_options(256, /*eps=*/1e-5, /*momentum=*/0.1);

let bn2 = BatchNorm2d::new(64);

// LayerNorm takes the shape it normalizes over
let ln = LayerNorm::new(vec![256]);
let ln_multidim = LayerNorm::new(vec![256, 256]);

// 32 groups, 256 channels
let gn = GroupNorm::new(32, 256);
let gn_opts = GroupNorm::with_options(32, 256, /*eps=*/1e-5, /*affine=*/true);

let in_norm = InstanceNorm2d::new(64);
```

`RMSNorm` is also available (used by LLaMA / Mistral).

## Activation Layers

Activations are constructed with `new()` (or a variant that takes parameters). They are unit structs in some cases and stateful configs in others, so construct them with `::new()` rather than just the bare name.

```rust
use axonml_nn::{ReLU, Sigmoid, Tanh, Softmax, LogSoftmax, GELU, SiLU, LeakyReLU, ELU, Flatten, Identity};

let relu   = ReLU;                     // unit struct — available bare too
let sig    = Sigmoid;
let tanh   = Tanh;
let gelu   = GELU;
let silu   = SiLU;
let id     = Identity;
let flat   = Flatten::new();

let leaky  = LeakyReLU::new();                    // default slope
let leaky2 = LeakyReLU::with_slope(0.01);

let elu    = ELU::new();                          // default alpha=1.0
let elu2   = ELU::with_alpha(0.5);

let sm     = Softmax::new(1);                     // dim 1 (i64)
let lsm    = LogSoftmax::new(1);
```

Activations are also available as methods on `Variable` (e.g. `x.relu()`, `x.gelu()`, `x.softmax(1)`).

## Dropout

```rust
use axonml_nn::Dropout;

let dropout = Dropout::new(0.5);

// Training / eval mode switches are on the Module trait
// dropout.train();
// dropout.eval();
```

`Dropout2d` (spatial) lives in the `axonml_nn::layers::dropout` module.

## Recurrent Layers

`RNN`, `LSTM`, `GRU` all take `(input_size, hidden_size, num_layers)` for the multi-layer constructor, with a single-layer `Cell` variant also available.

```rust
use axonml_nn::{LSTM, GRU, Module};

let lstm = LSTM::new(256, 512, /*num_layers=*/2);
let gru = GRU::new(256, 512, 1);

// batch_first / bidirectional toggles live on `with_options`
// (see crates/axonml-nn/src/layers/rnn.rs)

let x = Variable::new(Tensor::<f32>::randn(&[32, 100, 256]), false);
let y = lstm.forward(&x);                          // default `Module::forward`: hidden states
```

The Cell types (`LSTMCell`, `GRUCell`, `RNNCell`) step one timestep at a time.

## Attention

```rust
use axonml_nn::{MultiHeadAttention, CrossAttention, DifferentialAttention};

// Self-attention — embed_dim=512, num_heads=8
let attn = MultiHeadAttention::new(512, 8);

// Cross-attention (separate Q vs K/V projections)
let cross = CrossAttention::new(512, 8);

// Differential attention (DiffTransformer variant)
let diff = DifferentialAttention::new(512, 8);
```

## Transformer Blocks

```rust
use axonml_nn::{TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Seq2SeqTransformer};

let enc_layer = TransformerEncoderLayer::new(/*d_model=*/512, /*nhead=*/8);
let dec_layer = TransformerDecoderLayer::new(512, 8);
```

## Advanced Layers

The `axonml_nn::layers` module also provides:

- `Embedding` — lookup table
- 1.58-bit ternary quantized Linear (BitNet b1.58)
- `SparseLinear`, `GroupSparsity`, `LotteryTicket` — differentiable structured sparsity (novel to AxonML)
- `MoELayer`, `MoERouter`, `Expert` — mixture-of-experts
- `GCNConv`, `GATConv` — graph neural networks
- `ResidualBlock`
- `FFT1d`, `STFT` — spectral layers

## Building Models

### Sequential

`Sequential` can hold mixed module types and is built via `.add(...)`:

```rust
use axonml_nn::{Sequential, Linear, ReLU, Dropout};

let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Dropout::new(0.5))
    .add(Linear::new(256, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));
```

### Custom Modules

```rust
use axonml_nn::{Module, Linear, BatchNorm1d, Parameter};
use axonml_autograd::Variable;

struct MyMLP {
    fc1: Linear,
    bn1: BatchNorm1d,
    fc2: Linear,
    fc3: Linear,
    training: bool,
}

impl MyMLP {
    fn new(in_features: usize, hidden: usize, out: usize) -> Self {
        Self {
            fc1: Linear::new(in_features, hidden),
            bn1: BatchNorm1d::new(hidden),
            fc2: Linear::new(hidden, hidden),
            fc3: Linear::new(hidden, out),
            training: true,
        }
    }
}

impl Module for MyMLP {
    fn forward(&self, x: &Variable) -> Variable {
        let h = self.fc1.forward(x);
        let h = self.bn1.forward(&h);
        let h = h.relu();
        let h = self.fc2.forward(&h).relu();
        self.fc3.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.fc1.parameters();
        p.extend(self.bn1.parameters());
        p.extend(self.fc2.parameters());
        p.extend(self.fc3.parameters());
        p
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.bn1.set_training(training);
    }

    fn is_training(&self) -> bool { self.training }
}
```

## Loss Functions

All losses are in `axonml_nn::loss` and re-exported:

```rust
use axonml_nn::{MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, L1Loss, SmoothL1Loss, NLLLoss, Reduction};

let mse = MSELoss::new();
let loss = mse.compute(&predictions, &targets);

let ce = CrossEntropyLoss::new();                  // classification
let bce = BCELoss::new();                          // binary
let bcel = BCEWithLogitsLoss::new();               // binary, numerically-stable
let l1 = L1Loss::new();                            // MAE
let s1 = SmoothL1Loss::new();                      // Huber, beta=1.0
let s1b = SmoothL1Loss::with_beta(0.1);
let nll = NLLLoss::new();
```

Specialty detection losses live in `axonml-vision::losses` (`FocalLoss`, `GIoULoss`, `UncertaintyLoss`, etc.). See [Object Detection Training]({% link detection.md %}) and [Training]({% link training.md %}).

## Saving and Loading

Model serialization lives in the `axonml-serialize` crate (StateDict + SafeTensors). See the crate-level docs and [ONNX]({% link onnx.md %}) for interchange with external frameworks.

---

*Last updated: 2026-06-06 (v0.6.5)*
