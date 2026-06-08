# axonml-nn Documentation

> Neural network modules for the AxonML ML framework.

## Overview

`axonml-nn` provides a PyTorch-style `Module` trait and a large library of
layers, activations, loss functions, initializers, and container modules.
Every layer implements `Module`, with forward pass, parameter aggregation,
train/eval mode, and device placement.

`TernaryLinear` (1.58-bit) has GPU forward + backward kernels with GPU-resident
ternary quantization and host-staging of saved activations during backward to
cut peak VRAM at billion-parameter scale (this is what powers Trident b1.58
training). On CPU, the layer/GradFn backward paths are rayon-parallel as of
0.6.5 — see [autograd](../autograd/README.md).

## Core Concepts

### The `Module` trait

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<Parameter> { Vec::new() }
    fn named_parameters(&self) -> HashMap<String, Parameter> { HashMap::new() }
    fn train(&mut self);
    fn eval(&mut self);
    fn is_training(&self) -> bool;
    fn zero_grad(&mut self);
    fn name(&self) -> &str;
    fn to_device(&mut self, device: Device);
}
```

`ModuleList` is a heterogeneous `Vec<Box<dyn Module>>` that runs modules
sequentially and aggregates parameters + mode switches.

### `Parameter`

`Parameter` is a named, gradient-tracked weight wrapping a `Variable`.

## Modules

### Containers

- `Sequential` — chain of modules (builder API via `.add(...)`)
- `ModuleList` — heterogeneous owned list

### Layers (in `layers/`)

| Category        | Items                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------|
| Linear          | `Linear`                                                                                |
| Convolution     | `Conv1d`, `Conv2d`, `ConvTranspose2d`                                                   |
| Pooling         | `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `AdaptiveAvgPool2d`                 |
| Normalization   | `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `InstanceNorm2d` (RMSNorm lives in `axonml-llm::llama`, not this crate) |
| Dropout         | `Dropout`, `Dropout2d`                                                                  |
| Recurrent       | `RNN`, `RNNCell`, `LSTM`, `LSTMCell`, `GRU`, `GRUCell`                                  |
| Attention       | `MultiHeadAttention`, `CrossAttention`, `DifferentialAttention`                         |
| Transformer     | `TransformerEncoder(Layer)`, `TransformerDecoder(Layer)`, `Seq2SeqTransformer`          |
| Embedding       | `Embedding`                                                                             |
| Residual        | `ResidualBlock`                                                                         |
| MoE             | `MoELayer`, `MoERouter`, `Expert`                                                       |
| Quantized       | `TernaryLinear`, `PackedTernaryWeights` (BitNet b1.58)                                  |
| Graph           | `GCNConv`, `GATConv`                                                                    |
| Spectral        | `FFT1d`, `STFT`                                                                         |
| Sparse          | `SparseLinear`, `GroupSparsity`, `LotteryTicket` (differentiable structured sparsity)   |

### Activations (module form)

`ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `ELU`, `LeakyReLU`, `Softmax`,
`LogSoftmax`, `Flatten`, `Identity`. Available both as `Module` structs and
as `Variable` methods (`x.relu()`, `x.gelu()`, ...).

### Losses

`MSELoss`, `CrossEntropyLoss`, `BCELoss`, `BCEWithLogitsLoss`, `L1Loss`,
`SmoothL1Loss`, `NLLLoss`. All share a `Reduction` enum (`Mean`, `Sum`,
`None`).

Higher-level losses (`CTCLoss`, `FocalLoss`, `TripletLoss`, `ArcFaceLoss`)
are also available in the `loss` module.

### `init`

Weight initialization.

```rust
use axonml_nn::init::*;

xavier_uniform(&mut tensor);
xavier_normal(&mut tensor);
glorot_uniform(&mut tensor);
glorot_normal(&mut tensor);
kaiming_uniform(&mut tensor, nonlinearity);
kaiming_normal(&mut tensor, nonlinearity);
he_uniform(&mut tensor, nonlinearity);
he_normal(&mut tensor, nonlinearity);

zeros(&mut tensor);
ones(&mut tensor);
constant(&mut tensor, value);
eye(&mut tensor);
diag(&mut tensor, value);
normal(&mut tensor, mean, std);
uniform(&mut tensor, low, high);
uniform_range(&mut tensor, r);
orthogonal(&mut tensor, gain);
sparse(&mut tensor, sparsity, std);
randn(&mut tensor);
```

`InitMode` enum captures the choice when storing configuration.

### `functional`

Stateless function wrappers (`functional::relu(&x)`, `functional::linear(...)`,
etc.) — the module-free API for one-off ops.

## Usage Examples

### Simple MLP

```rust
use axonml::prelude::*;

let model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(ReLU)
    .add(Dropout::new(0.5))
    .add(Linear::new(128, 10));

let output = model.forward(&input);
```

### CNN

```rust
struct CNN {
    conv1: Conv2d,
    conv2: Conv2d,
    pool: MaxPool2d,
    fc: Linear,
}

impl Module for CNN {
    fn forward(&self, x: &Variable) -> Variable {
        let x = self.conv1.forward(x).relu();
        let x = self.pool.forward(&x);
        let x = self.conv2.forward(&x).relu();
        let x = self.pool.forward(&x);
        let x = x.flatten();
        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.fc.parameters(),
        ].concat()
    }
    // (remaining Module methods)
    # fn train(&mut self) {}
    # fn eval(&mut self) {}
    # fn is_training(&self) -> bool { false }
    # fn zero_grad(&mut self) {}
    # fn name(&self) -> &str { "CNN" }
    # fn to_device(&mut self, _d: axonml_core::Device) {}
}
```

### Train / eval mode

```rust
let mut model = create_model();

model.train(); // dropout active, batchnorm uses batch stats
let y = model.forward(&train_input);

model.eval();  // dropout off, batchnorm uses running stats
let y = model.forward(&test_input);
```

### Differentiable structured sparsity

`SparseLinear` applies a soft-thresholded, differentiable pruning mask:

```rust
use axonml_nn::layers::sparse::{SparseLinear, GroupSparsity, LotteryTicket};

let mut sparse = SparseLinear::new(256, 128, 0.5, 10.0);
let output = sparse.forward(&input);
println!("sparsity: {:.1}%", sparse.sparsity() * 100.0);

let group_reg = GroupSparsity::new(0.01, "row");
let reg_loss = group_reg.compute(&sparse);

let mut ticket = LotteryTicket::new(&sparse);
ticket.snapshot();
// ... train ...
ticket.prune(0.2);           // prune 20% by magnitude
ticket.rewind(&mut sparse);  // rewind to init weights with mask
```

Mask formula: `sigmoid((|weight| - threshold) * temperature)` — gradients
flow through the pruning decision.

## Related Modules

- [Autograd](../autograd/README.md) — gradient computation
- [Optimizers](../optim/README.md) — parameter updates

## Last updated

0.6.5 (2026-06-06)
