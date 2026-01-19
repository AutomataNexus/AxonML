# axonml-tensor

[![Crates.io](https://img.shields.io/crates/v/axonml-tensor.svg)](https://crates.io/crates/axonml-tensor)
[![Docs.rs](https://docs.rs/axonml-tensor/badge.svg)](https://docs.rs/axonml-tensor)
[![Downloads](https://img.shields.io/crates/d/axonml-tensor.svg)](https://crates.io/crates/axonml-tensor)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> N-dimensional tensor operations for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-tensor` provides the core tensor data structure and operations for the Axonml ML framework. It implements efficient N-dimensional arrays with automatic broadcasting, SIMD-accelerated operations, and seamless device support. This is the computational workhorse of the framework.

## Features

### Tensor Operations
- **N-dimensional arrays** - Arbitrary rank tensors with dynamic shapes
- **Automatic broadcasting** - NumPy-compatible broadcasting rules
- **Zero-copy views** - Efficient slicing, reshaping, and transposing
- **SIMD acceleration** - Vectorized operations for CPU performance

### Arithmetic Operations
- **Element-wise** - Add, subtract, multiply, divide, power
- **Matrix multiplication** - Optimized matmul with BLAS integration
- **Reductions** - Sum, mean, max, min, prod along axes
- **Comparisons** - Greater, less, equal, not equal

### Activation Functions
- **Standard** - ReLU, Sigmoid, Tanh
- **Modern** - GELU, SiLU/Swish, Softmax
- **Variants** - LeakyReLU, ELU, Softplus

### Shape Operations
- **Reshape** - Change tensor shape without copying
- **Transpose/Permute** - Reorder dimensions
- **Squeeze/Unsqueeze** - Remove or add dimensions
- **Concatenate/Stack** - Combine tensors

## Installation

```toml
[dependencies]
axonml-tensor = "0.1"
```

## Usage

### Creating Tensors

```rust
use axonml_tensor::{Tensor, zeros, ones, rand, randn, arange, linspace};

// Zeros and ones
let z = zeros::<f32>(&[2, 3, 4]);  // Shape: [2, 3, 4]
let o = ones::<f64>(&[5, 5]);       // Shape: [5, 5]

// Random tensors
let r = rand::<f32>(&[10, 10]);     // Uniform [0, 1)
let n = randn::<f32>(&[10, 10]);    // Normal(0, 1)

// Ranges
let a = arange::<f32>(0.0, 10.0, 1.0);   // [0, 1, 2, ..., 9]
let l = linspace::<f32>(0.0, 1.0, 100);  // 100 points from 0 to 1

// From data
let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

// Special matrices
let eye = eye::<f32>(4);                 // 4x4 identity
let diag = diag(&[1.0, 2.0, 3.0]);       // Diagonal matrix
```

### Arithmetic Operations

```rust
use axonml_tensor::{Tensor, ones, randn};

let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[3, 4]);

// Element-wise operations
let sum = &a + &b;
let diff = &a - &b;
let prod = &a * &b;
let quot = &a / &b;

// Scalar operations
let scaled = &a * 2.0;
let shifted = &a + 1.0;

// In-place operations
let mut c = a.clone();
c += &b;
c *= 2.0;

// Power and mathematical functions
let squared = a.pow(2.0);
let sqrt = a.sqrt();
let exp = a.exp();
let log = a.log();
```

### Matrix Operations

```rust
use axonml_tensor::{Tensor, randn};

let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[4, 5]);

// Matrix multiplication
let c = a.matmul(&b).unwrap();  // Shape: [3, 5]

// Batch matrix multiplication
let batch_a = randn::<f32>(&[8, 3, 4]);  // 8 matrices of 3x4
let batch_b = randn::<f32>(&[8, 4, 5]);  // 8 matrices of 4x5
let batch_c = batch_a.matmul(&batch_b).unwrap();  // Shape: [8, 3, 5]

// Transpose
let at = a.t();  // Shape: [4, 3]

// Dot product
let v1 = randn::<f32>(&[100]);
let v2 = randn::<f32>(&[100]);
let dot = v1.dot(&v2);
```

### Reductions

```rust
use axonml_tensor::{Tensor, randn};

let t = randn::<f32>(&[3, 4, 5]);

// Global reductions
let total = t.sum();              // Sum of all elements
let average = t.mean().unwrap();  // Mean of all elements
let maximum = t.max();            // Maximum value
let minimum = t.min();            // Minimum value

// Axis reductions
let row_sum = t.sum_dim(0, false);    // Sum along axis 0
let col_mean = t.mean_dim(1, true);   // Mean along axis 1, keep dim
let last_max = t.max_dim(2, false);   // Max along last axis

// Argmax/Argmin
let indices = t.argmax(1);  // Indices of max values along axis 1
```

### Broadcasting

```rust
use axonml_tensor::{Tensor, ones, randn};

// Automatic broadcasting (NumPy rules)
let a = randn::<f32>(&[3, 4, 5]);
let b = randn::<f32>(&[4, 5]);      // Broadcasts to [3, 4, 5]
let c = randn::<f32>(&[5]);         // Broadcasts to [3, 4, 5]

let result = &a + &b;  // Shape: [3, 4, 5]
let result2 = &a * &c; // Shape: [3, 4, 5]

// Explicit broadcasting
let row = randn::<f32>(&[1, 5]);
let col = randn::<f32>(&[3, 1]);
let matrix = &row + &col;  // Shape: [3, 5]
```

### Shape Manipulation

```rust
use axonml_tensor::{Tensor, randn};

let t = randn::<f32>(&[2, 3, 4]);

// Reshape
let flat = t.flatten();                      // Shape: [24]
let reshaped = t.reshape(&[6, 4]).unwrap();  // Shape: [6, 4]
let auto = t.reshape(&[-1, 4]).unwrap();     // Shape: [6, 4] (infer -1)

// Transpose and permute
let transposed = t.permute(&[2, 0, 1]).unwrap();  // Shape: [4, 2, 3]

// Squeeze and unsqueeze
let expanded = t.unsqueeze(0).unwrap();           // Shape: [1, 2, 3, 4]
let squeezed = expanded.squeeze(Some(0)).unwrap(); // Shape: [2, 3, 4]

// Slicing
let slice = t.slice_dim0(0, 1).unwrap();  // First element along dim 0
let narrow = t.narrow(1, 0, 2).unwrap();  // Narrow dim 1 to indices 0-1
```

### Activation Functions

```rust
use axonml_tensor::{Tensor, randn};

let t = randn::<f32>(&[10, 10]);

// Standard activations
let relu = t.relu();
let sigmoid = t.sigmoid();
let tanh_out = t.tanh();

// Modern activations
let gelu = t.gelu();
let silu = t.silu();          // Also known as Swish
let softmax = t.softmax(1);   // Along axis 1

// Variants
let leaky = t.leaky_relu(0.01);
let elu = t.elu(1.0);
let softplus = t.softplus();
```

### Concatenation and Stacking

```rust
use axonml_tensor::{Tensor, randn, cat, stack};

let a = randn::<f32>(&[2, 3]);
let b = randn::<f32>(&[2, 3]);
let c = randn::<f32>(&[2, 3]);

// Concatenate along existing dimension
let concat = cat(&[&a, &b, &c], 0).unwrap();  // Shape: [6, 3]

// Stack creates new dimension
let stacked = stack(&[&a, &b, &c], 0).unwrap();  // Shape: [3, 2, 3]
```

## API Reference

### Creation Functions

| Function | Description |
|----------|-------------|
| `zeros(shape)` | Create tensor filled with zeros |
| `ones(shape)` | Create tensor filled with ones |
| `full(shape, value)` | Create tensor filled with value |
| `rand(shape)` | Uniform random [0, 1) |
| `randn(shape)` | Normal distribution (0, 1) |
| `arange(start, end, step)` | Range of values |
| `linspace(start, end, n)` | N evenly spaced values |
| `eye(n)` | Identity matrix |
| `diag(values)` | Diagonal matrix |

### Tensor Methods

| Method | Description |
|--------|-------------|
| `shape()` | Get tensor shape |
| `ndim()` | Number of dimensions |
| `numel()` | Total number of elements |
| `dtype()` | Data type |
| `device()` | Device (CPU, CUDA, etc.) |
| `is_contiguous()` | Check memory layout |
| `contiguous()` | Make contiguous copy if needed |

### Operations

| Category | Operations |
|----------|------------|
| Arithmetic | `+`, `-`, `*`, `/`, `pow`, `sqrt`, `exp`, `log` |
| Comparison | `gt`, `lt`, `ge`, `le`, `eq`, `ne` |
| Reduction | `sum`, `mean`, `max`, `min`, `prod`, `argmax`, `argmin` |
| Matrix | `matmul`, `dot`, `t`, `transpose` |
| Shape | `reshape`, `flatten`, `squeeze`, `unsqueeze`, `permute` |
| Activation | `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softmax` |

## Performance

- **SIMD vectorization** - Uses platform SIMD (SSE, AVX, NEON)
- **Cache-friendly** - Optimized memory access patterns
- **Parallel execution** - Multi-threaded via Rayon
- **BLAS integration** - Accelerated matrix operations

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

```toml
[dependencies]
axonml = "0.1"  # Includes axonml-tensor
```

## License

MIT OR Apache-2.0
