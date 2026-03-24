---
layout: default
title: Tensor Operations
nav_order: 3
description: "Working with tensors in AxonML"
---

# Tensor Operations
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Creating Tensors

### From Data

```rust
use axonml::tensor::Tensor;

// From a vector
let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

// From a slice
let data = [1.0, 2.0, 3.0];
let t = Tensor::from_slice(&data, &[3]).unwrap();
```

### Random Initialization

```rust
// Standard normal distribution
let t = Tensor::randn(&[3, 4]);  // Shape: [3, 4]

// Uniform [0, 1)
let t = Tensor::rand(&[3, 4]);

// Xavier/Glorot initialization
let t = Tensor::xavier(&[784, 256]);

// Kaiming/He initialization
let t = Tensor::kaiming(&[256, 128]);
```

### Special Tensors

```rust
// Zeros
let t = Tensor::zeros(&[3, 4]);

// Ones
let t = Tensor::ones(&[3, 4]);

// Identity matrix
let t = Tensor::eye(3);

// Filled with value
let t = Tensor::full(&[3, 4], 3.14);

// Range
let t = Tensor::arange(0.0, 10.0, 1.0);  // [0, 1, 2, ..., 9]

// Linspace
let t = Tensor::linspace(0.0, 1.0, 5);  // [0.0, 0.25, 0.5, 0.75, 1.0]
```

## Shape Operations

### Reshape and View

```rust
let t = Tensor::randn(&[2, 3, 4]);

// Reshape (may copy data)
let r = t.reshape(&[6, 4]);

// View (zero-copy, must be contiguous)
let v = t.view(&[2, 12]);

// Flatten
let f = t.flatten();  // Shape: [24]

// Squeeze (remove dimensions of size 1)
let t = Tensor::randn(&[1, 3, 1, 4]);
let s = t.squeeze();  // Shape: [3, 4]

// Unsqueeze (add dimension)
let t = Tensor::randn(&[3, 4]);
let u = t.unsqueeze(0);  // Shape: [1, 3, 4]
```

### Transpose and Permute

```rust
let t = Tensor::randn(&[2, 3, 4]);

// Transpose last two dimensions
let tr = t.t();  // Shape: [2, 4, 3]

// Permute dimensions
let p = t.permute(&[2, 0, 1]);  // Shape: [4, 2, 3]

// Contiguous (ensure memory layout)
let c = t.contiguous();
```

## Indexing and Slicing

### Basic Indexing

```rust
let t = Tensor::randn(&[4, 5, 6]);

// Get single element
let val = t.get(&[0, 1, 2]);

// Slice
let s = t.slice(&[0..2, 1..4, ..]);  // Shape: [2, 3, 6]

// Select along dimension
let s = t.select(0, 1);  // Shape: [5, 6]

// Narrow
let n = t.narrow(1, 1, 3);  // Shape: [4, 3, 6]
```

### Advanced Indexing

```rust
// Gather
let indices = Tensor::from_vec(vec![0, 2, 1], &[3]).unwrap();
let g = t.gather(1, &indices);

// Scatter
let src = Tensor::ones(&[3, 3]);
let indices = Tensor::from_vec(vec![0, 1, 2], &[3]).unwrap();
let s = Tensor::zeros(&[3, 5]).scatter(1, &indices, &src);

// Index select
let indices = Tensor::from_vec(vec![0, 2], &[2]).unwrap();
let s = t.index_select(0, &indices);  // Shape: [2, 5, 6]

// Masked select
let mask = t.gt(&Tensor::zeros(&[4, 5, 6]));
let s = t.masked_select(&mask);
```

## Arithmetic Operations

### Element-wise Operations

```rust
let a = Tensor::randn(&[3, 4]);
let b = Tensor::randn(&[3, 4]);

// Addition
let c = &a + &b;
let c = a.add(&b);

// Subtraction
let c = &a - &b;

// Multiplication
let c = &a * &b;

// Division
let c = &a / &b;

// Power
let c = a.pow(2.0);
let c = a.pow_tensor(&b);

// Square root
let c = a.sqrt();

// Absolute value
let c = a.abs();

// Negation
let c = -&a;
```

### Matrix Operations

```rust
let a = Tensor::randn(&[3, 4]);
let b = Tensor::randn(&[4, 5]);

// Matrix multiplication
let c = a.matmul(&b);  // Shape: [3, 5]

// Batch matrix multiplication
let a = Tensor::randn(&[2, 3, 4]);
let b = Tensor::randn(&[2, 4, 5]);
let c = a.bmm(&b);  // Shape: [2, 3, 5]

// Dot product
let a = Tensor::randn(&[100]);
let b = Tensor::randn(&[100]);
let c = a.dot(&b);  // Scalar
```

### Broadcasting

```rust
let a = Tensor::randn(&[3, 4, 5]);
let b = Tensor::randn(&[5]);      // Broadcasts to [3, 4, 5]
let c = Tensor::randn(&[4, 1]);   // Broadcasts to [3, 4, 5]

let d = &a + &b;  // Shape: [3, 4, 5]
let e = &a * &c;  // Shape: [3, 4, 5]
```

## Reduction Operations

```rust
let t = Tensor::randn(&[3, 4, 5]);

// Sum
let s = t.sum();                    // Scalar
let s = t.sum_dim(1, true);         // Shape: [3, 1, 5]
let s = t.sum_dims(&[0, 2], false); // Shape: [4]

// Mean
let m = t.mean();
let m = t.mean_dim(1, true);

// Max/Min
let (max_val, max_idx) = t.max_dim(1);
let (min_val, min_idx) = t.min_dim(1);

// Product
let p = t.prod();

// Variance and Standard Deviation
let v = t.var(1, true);   // Unbiased variance
let s = t.std(1, false);  // Biased std
```

## Sorting and Searching

```rust
let t = Tensor::randn(&[3, 4]);

// Sort
let (sorted, indices) = t.sort(1, false);  // Ascending
let (sorted, indices) = t.sort(1, true);   // Descending

// Argsort
let indices = t.argsort(1, false);

// Top-k
let (values, indices) = t.topk(3, 1, true);  // Top 3, descending

// Unique
let (unique, inverse, counts) = t.unique(true, true);

// Nonzero
let indices = t.nonzero();
```

## Activation Functions

```rust
let t = Tensor::randn(&[3, 4]);

// ReLU
let r = t.relu();

// Sigmoid
let s = t.sigmoid();

// Tanh
let th = t.tanh();

// Softmax
let sm = t.softmax(1);

// Log Softmax
let lsm = t.log_softmax(1);

// GELU
let g = t.gelu();

// SiLU / Swish
let si = t.silu();

// Leaky ReLU
let lr = t.leaky_relu(0.01);

// ELU
let e = t.elu(1.0);
```

## Device Management

```rust
use axonml::core::Device;

let t = Tensor::randn(&[1000, 1000]);

// Move to GPU
let t_gpu = t.to(Device::CUDA(0));

// Move back to CPU
let t_cpu = t_gpu.to(Device::CPU);

// Check device
assert!(t_cpu.device() == Device::CPU);

// Create directly on device
let t = Tensor::randn_on(&[1000, 1000], Device::CUDA(0));
```

## Data Type Conversion

```rust
use axonml::core::DType;

let t = Tensor::randn(&[3, 4]);  // Default: F32

// Convert to different types
let t_f64 = t.to_dtype(DType::F64);
let t_f16 = t.to_dtype(DType::F16);
let t_i32 = t.to_dtype(DType::I32);

// Check dtype
assert!(t.dtype() == DType::F32);
```
