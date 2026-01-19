# axonml-tensor

N-dimensional tensor operations for the [Axonml](https://github.com/AutomataNexus/AxonML) ML framework.

## Overview

`axonml-tensor` provides efficient tensor operations including:

- **N-dimensional Arrays** - Arbitrary shapes and strides
- **Broadcasting** - NumPy-compatible broadcasting rules
- **Arithmetic** - Add, subtract, multiply, divide, matmul
- **Reductions** - Sum, mean, max, min along dimensions
- **Activations** - ReLU, Sigmoid, Tanh, Softmax, GELU

## Usage

```rust
use axonml_tensor::{Tensor, zeros, ones, randn};

// Create tensors
let a = zeros::<f32>(&[2, 3]);
let b = ones::<f32>(&[2, 3]);
let c = randn::<f32>(&[2, 3]);

// Operations
let d = a.add(&b).unwrap();
let e = c.matmul(&b.t()).unwrap();
let f = e.relu();
```

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

## License

MIT OR Apache-2.0
