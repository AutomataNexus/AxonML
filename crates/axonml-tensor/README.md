# axonml-tensor

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

**axonml-tensor** provides the core `Tensor` type for the AxonML framework. Tensors are N-dimensional arrays with support for automatic broadcasting, efficient memory sharing through views, and device-agnostic operations for machine learning computations.

## Features

- **N-Dimensional Arrays** - Create tensors of arbitrary shape with generic element types (f32, f64, i32, etc.).

- **Automatic Broadcasting** - NumPy-style broadcasting for element-wise operations between tensors of different shapes.

- **Efficient Views** - Zero-copy slicing, transposing, and reshaping through stride manipulation without data copying.

- **Device Agnostic** - Transparent tensor operations across CPU, CUDA, Vulkan, Metal, and WebGPU backends.

- **Rich Operations** - Comprehensive arithmetic, reduction, activation, and matrix operations including matmul with batching support.

- **Factory Functions** - Convenient tensor creation with `zeros`, `ones`, `rand`, `randn`, `arange`, `linspace`, and more.

## Modules

| Module | Description |
|--------|-------------|
| `tensor` | Core `Tensor` struct with arithmetic, reduction, activation, and shape operations |
| `shape` | Shape and stride utilities including broadcasting, reshape, and index computation |
| `creation` | Factory functions for tensor initialization (zeros, ones, rand, randn, arange, linspace, eye) |
| `view` | Slicing, indexing, and view operations (select, narrow, chunk, split) |
| `ops` | Additional operations including softmax, GELU, comparisons, and clipping |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-tensor = "0.1.0"
```

### Basic Example

```rust
use axonml_tensor::{Tensor, zeros, ones, randn};

// Create tensors
let a = zeros::<f32>(&[2, 3]);
let b = ones::<f32>(&[2, 3]);
let c = randn::<f32>(&[2, 3]);

// Arithmetic operations
let sum = a.add(&b).unwrap();
let product = b.mul(&c).unwrap();
let scaled = c.mul_scalar(2.0);

// Reductions
let total = scaled.sum();
let average = scaled.mean().unwrap();
let maximum = scaled.max().unwrap();
```

### Shape Operations

```rust
use axonml_tensor::Tensor;

let t = Tensor::<f32>::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    &[2, 3]
).unwrap();

// Reshape
let flat = t.reshape(&[-1]).unwrap();  // [6]
let reshaped = t.reshape(&[3, 2]).unwrap();

// Transpose
let transposed = t.t().unwrap();  // [3, 2]

// Squeeze and unsqueeze
let unsqueezed = t.unsqueeze(0).unwrap();  // [1, 2, 3]
let squeezed = unsqueezed.squeeze(Some(0)).unwrap();  // [2, 3]
```

### Matrix Operations

```rust
use axonml_tensor::Tensor;

// Matrix multiplication
let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
let c = a.matmul(&b).unwrap();  // [2, 2]

// Batched matmul
let batch_a = randn::<f32>(&[4, 2, 3]);
let batch_b = randn::<f32>(&[4, 3, 5]);
let batch_c = batch_a.matmul(&batch_b).unwrap();  // [4, 2, 5]

// Dot product
let v1 = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
let v2 = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
let dot = v1.dot(&v2).unwrap();  // Scalar tensor
```

### Activation Functions

```rust
use axonml_tensor::Tensor;

let x = Tensor::<f32>::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();

let relu_out = x.relu();      // [0.0, 0.0, 1.0, 2.0]
let sigmoid_out = x.sigmoid();
let tanh_out = x.tanh();
let gelu_out = x.gelu();
let softmax_out = x.softmax(-1);
```

### Broadcasting

```rust
use axonml_tensor::Tensor;

// Automatic broadcasting
let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
let b = Tensor::<f32>::from_vec(vec![10.0], &[1]).unwrap();
let c = a.add(&b).unwrap();  // [11.0, 12.0, 13.0]

// 2D broadcasting
let matrix = Tensor::<f32>::from_vec(vec![1.0; 6], &[2, 3]).unwrap();
let row = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
let result = matrix.add(&row).unwrap();  // [2, 3]
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-tensor
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
