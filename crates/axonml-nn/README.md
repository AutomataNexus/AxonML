# axonml-nn

Neural network modules for the [Axonml](https://github.com/AutomataNexus/AxonML) ML framework.

## Overview

PyTorch-style neural network building blocks:

- **Layers** - Linear, Conv1d/2d, BatchNorm, LayerNorm, Dropout
- **Recurrent** - RNN, LSTM, GRU
- **Attention** - MultiHeadAttention
- **Activations** - ReLU, Sigmoid, Tanh, GELU, SiLU
- **Loss Functions** - MSE, CrossEntropy, BCE, L1

## Usage

```rust
use axonml_nn::{Sequential, Linear, ReLU, Module};

let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 10));

let output = model.forward(&input);
```

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

## License

MIT OR Apache-2.0
