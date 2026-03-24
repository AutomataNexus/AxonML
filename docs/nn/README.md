# axonml-nn Documentation

> Neural network modules for the Axonml ML framework.

## Overview

`axonml-nn` provides a PyTorch-like module system for building neural networks. It includes common layers, activation functions, loss functions, and container modules.

## Core Concepts

### Module Trait

All neural network components implement the `Module` trait:

```rust
pub trait Module {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<Parameter>;
    fn train(&mut self);
    fn eval(&mut self);
    fn is_training(&self) -> bool;
}
```

### Parameters

`Parameter` wraps a `Variable` that should be optimized:

```rust
pub struct Parameter {
    variable: Variable,
    name: String,
}
```

## Modules

### Layers

#### linear.rs - Fully Connected Layer

```rust
let linear = Linear::new(in_features, out_features);
let output = linear.forward(&input);  // [batch, out_features]
```

- Computes `y = xW^T + b`
- Weight shape: `[out_features, in_features]`
- Bias shape: `[out_features]`

#### conv.rs - Convolutional Layers

```rust
// 2D Convolution
let conv = Conv2d::new(in_channels, out_channels, kernel_size);
let output = conv.forward(&input);  // [batch, out_channels, H', W']

// With padding and stride
let conv = Conv2d::with_options(in_ch, out_ch, kernel, stride, padding);
```

**Available:**
- `Conv1d` - 1D convolution (sequences)
- `Conv2d` - 2D convolution (images)
- `Conv3d` - 3D convolution (video/volumetric)

#### pooling.rs - Pooling Layers

```rust
// Max pooling
let pool = MaxPool2d::new(kernel_size);
let pool = MaxPool2d::with_stride(kernel_size, stride);

// Average pooling
let pool = AvgPool2d::new(kernel_size);

// Global pooling
let pool = GlobalAvgPool2d::new();
```

#### norm.rs - Normalization Layers

```rust
// Batch normalization
let bn = BatchNorm1d::new(num_features);
let bn = BatchNorm2d::new(num_features);

// Layer normalization
let ln = LayerNorm::new(normalized_shape);
```

#### dropout.rs - Regularization

```rust
let dropout = Dropout::new(p);  // Drop probability

// In training: randomly zero elements
// In eval: pass through unchanged
let output = dropout.forward(&input);
```

#### rnn.rs - Recurrent Layers

```rust
// Simple RNN
let rnn = RNN::new(input_size, hidden_size);

// LSTM
let lstm = LSTM::new(input_size, hidden_size);

// GRU
let gru = GRU::new(input_size, hidden_size);

// Multi-layer with dropout
let lstm = LSTM::with_options(input_size, hidden_size, num_layers, dropout);
```

#### attention.rs - Attention Mechanisms

```rust
let attention = MultiHeadAttention::new(embed_dim, num_heads);
let output = attention.forward_qkv(&query, &key, &value, mask);
```

#### embedding.rs - Embeddings

```rust
let embedding = Embedding::new(vocab_size, embed_dim);
let output = embedding.forward(&indices);  // [batch, seq, embed_dim]
```

### Activations

```rust
use axonml_nn::*;

// As modules
let relu = ReLU;
let sigmoid = Sigmoid;
let tanh = Tanh;
let softmax = Softmax::new(dim);
let leaky_relu = LeakyReLU::new(negative_slope);
let gelu = GELU;
let silu = SiLU;

// Applied to variables
let y = x.relu();
let y = x.sigmoid();
let y = x.tanh();
```

### Loss Functions

```rust
use axonml_nn::*;

// Mean Squared Error
let loss = MSELoss::new();
let l = loss.forward(&prediction, &target);

// Cross Entropy (for classification)
let loss = CrossEntropyLoss::new();
let l = loss.forward(&logits, &labels);

// Binary Cross Entropy
let loss = BCELoss::new();
let l = loss.forward(&prediction, &target);

// L1 Loss
let loss = L1Loss::new();
let l = loss.forward(&prediction, &target);
```

### Containers

#### sequential.rs

Chain modules together:

```rust
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

let output = model.forward(&input);
```

### Initialization

```rust
use axonml_nn::init::*;

// Xavier/Glorot initialization
xavier_uniform(&mut tensor);
xavier_normal(&mut tensor);

// Kaiming/He initialization
kaiming_uniform(&mut tensor, nonlinearity);
kaiming_normal(&mut tensor, nonlinearity);

// Others
zeros(&mut tensor);
ones(&mut tensor);
constant(&mut tensor, value);
normal(&mut tensor, mean, std);
uniform(&mut tensor, low, high);
```

## Usage Examples

### Simple MLP

```rust
use axonml::prelude::*;

// Build model
let model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(ReLU)
    .add(Dropout::new(0.5))
    .add(Linear::new(128, 10));

// Forward pass
let output = model.forward(&input);
```

### Convolutional Network

```rust
use axonml::prelude::*;

struct CNN {
    conv1: Conv2d,
    conv2: Conv2d,
    pool: MaxPool2d,
    fc: Linear,
}

impl CNN {
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, 3),
            conv2: Conv2d::new(32, 64, 3),
            pool: MaxPool2d::new(2),
            fc: Linear::new(64 * 5 * 5, 10),
        }
    }
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
}
```

### Training/Eval Mode

```rust
let mut model = create_model();

// Training mode (dropout active, batchnorm uses batch stats)
model.train();
let output = model.forward(&train_input);

// Eval mode (dropout inactive, batchnorm uses running stats)
model.eval();
let output = model.forward(&test_input);
```

### Differentiable Structured Sparsity *(novel)*

#### sparse.rs - Learnable Pruning

`SparseLinear` applies a differentiable magnitude-based pruning mask:

```rust
use axonml_nn::layers::sparse::{SparseLinear, GroupSparsity, LotteryTicket};

// Soft-thresholded magnitude pruning
let sparse = SparseLinear::new(256, 128, 0.5, 10.0);
let output = sparse.forward(&input);
println!("Sparsity: {:.1}%", sparse.sparsity() * 100.0);

// Group sparsity regularization (row, column, or block)
let group_reg = GroupSparsity::new(0.01, "row");
let reg_loss = group_reg.compute(&sparse);

// Lottery Ticket Hypothesis: snapshot → train → prune → rewind
let mut ticket = LotteryTicket::new(&sparse);
ticket.snapshot();
// ... train ...
ticket.prune(0.2); // prune 20% by magnitude
ticket.rewind(&mut sparse); // rewind to init weights with mask
```

**Mask formula:** `sigmoid((|weight| - threshold) * temperature)`

This makes the mask continuous and differentiable — gradients flow through the pruning decision. PyTorch's pruning (`torch.nn.utils.prune`) uses binary masks that are not differentiable.

### Additional Layers

The following layers are also available in `axonml-nn`:

| Layer | Module | Description |
|:------|:-------|:------------|
| `DiffAttention` | `layers::diff_attention` | Differential attention (used by Chimera SLM) |
| `MoELayer` | `layers::moe` | Mixture of Experts with top-k routing |
| `TernaryLinear` | `layers::ternary` | 1.58-bit ternary weight layer (BitNet b1.58) |
| `ResidualBlock` | `layers::residual` | Pre-configured residual skip connection |
| `GCNConv` | `layers::graph` | Graph Convolutional Network layer |
| `GATConv` | `layers::graph` | Graph Attention Network layer |
| `FFT1d` / `STFT` | `layers::fft` | Fourier transform layers |

These layers are used as building blocks by higher-level models in `axonml-vision` (NightVision, Biometric Suite) and `axonml-llm` (Chimera, Trident).

## Related Modules

- [Autograd](../autograd/README.md) - Gradient computation
- [Optimizers](../optim/README.md) - Parameter updates
- [Data](../data/README.md) - Data loading

@version 0.4.1
@author AutomataNexus Development Team
