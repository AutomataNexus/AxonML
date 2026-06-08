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

## Overview

`axonml-tensor::Tensor<T>` is the N-dimensional array type, generic over `Scalar` element types (f16, f32, f64, i8–i64, u8/u32/u64, bool). Supports NumPy-style broadcasting, strided zero-copy views, CPU + CUDA GPU matmul (cuBLAS with a GEMV fast path for `m=1` decode), quantized matmul dispatch (Q4_K / Q5_K / Q6_K / Q8_0 dequant-in-shader on GPU), lazy evaluation with algebraic optimization (see `axonml_tensor::lazy`), and sparse COO tensors.

**CPU parallelism (0.6.5):** on the CPU backend, matmul (all layouts incl. 3D/4D batched), reductions, SwiGLU, RMSNorm, RoPE, `layer_norm`, `gelu`, and the elementwise/activation family are rayon-parallel above a work threshold, with contiguous/offset-0 fast-paths that avoid copying before the parallel backend. Tiny ops stay serial; results are identical to the single-threaded path.

`axonml-tensor` re-exports `Device`, `DType`, `Error`, `Result` from `axonml-core`.

## Creating Tensors

### From Data

```rust
use axonml_tensor::Tensor;

// From a vector
let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

// From a slice
let data = [1.0_f32, 2.0, 3.0];
let t = Tensor::from_slice(&data, &[3]).unwrap();

// Scalar (0-dim tensor)
let s = Tensor::<f32>::scalar(3.14);
```

### Factory Functions

The top-level factory functions live in `axonml_tensor::creation` and are re-exported from the crate root (also available via the `prelude`):

```rust
use axonml_tensor::{zeros, ones, full, randn, rand, uniform, normal, arange, linspace, logspace, eye, diag};

// Zeros / Ones
let z = zeros::<f32>(&[2, 3]);
let o = ones::<f32>(&[2, 3]);

// Filled
let f = full(&[3, 4], 3.14_f32);

// Random
let r = randn::<f32>(&[3, 4]);                     // Standard normal
let u = rand::<f32>(&[3, 4]);                      // Uniform [0, 1)
let un = uniform::<f32>(&[3, 4], -1.0, 1.0);       // Uniform [low, high)
let no = normal::<f32>(&[3, 4], 0.0, 0.5);         // Normal(mean, std)

// Ranges
let a = arange::<f32>(0.0, 10.0, 1.0);             // [0, 1, ..., 9]
let l = linspace::<f32>(0.0, 1.0, 5);              // 5 points, inclusive endpoints
let g = logspace::<f32>(0.0, 3.0, 4, 10.0);        // 10^0, 10^1, 10^2, 10^3

// Special matrices
let i = eye::<f32>(4);
let d = diag::<f32>(&[1.0, 2.0, 3.0]);
```

Also available on the `Tensor` type directly: `Tensor::zeros`, `Tensor::ones` (requires `Numeric`), `Tensor::full`, `Tensor::randn` (requires `Float`), `Tensor::rand` (requires `Float`).

## Shape Operations

```rust
let t = randn::<f32>(&[2, 3, 4]);

// Shape queries
assert_eq!(t.shape(), &[2, 3, 4]);
assert_eq!(t.ndim(), 3);
assert_eq!(t.numel(), 24);
assert_eq!(t.size(1).unwrap(), 3);
assert!(!t.is_empty());

// Reshape — uses -1 as an inferred dim. Takes &[isize].
let r = t.reshape(&[6, 4]).unwrap();
let inferred = t.reshape(&[-1, 4]).unwrap();

// Flatten to 1D
let f = t.flatten();

// Squeeze / Unsqueeze
let s = randn::<f32>(&[1, 3, 1, 4]);
let sq = s.squeeze(None).unwrap();                 // Remove all size-1 dims -> [3, 4]
let sq1 = s.squeeze(Some(0)).unwrap();             // Remove dim 0 only -> [3, 1, 4]
let u = t.unsqueeze(0).unwrap();                   // [1, 2, 3, 4]

// Transpose / Permute
let tr = t.transpose(0, 2).unwrap();               // swap dims 0 and 2
let p = t.permute(&[2, 0, 1]).unwrap();            // [4, 2, 3]
let rev = t.t().unwrap();                          // Transpose last two dims (2-D only)

// Contiguous
let c = t.contiguous();
assert!(c.is_contiguous());
```

## Indexing and Slicing

```rust
let t = randn::<f32>(&[4, 5, 6]);

// Get / Set single element
let val: f32 = t.get(&[0, 1, 2]).unwrap();
t.set(&[0, 1, 2], 3.5).unwrap();

// Item — scalar tensor (numel == 1) only
let s = Tensor::<f32>::scalar(7.0);
let v = s.item().unwrap();

// Slice over ranges (one per dim)
let s = t.slice(&[0..2, 1..4, 0..6]);              // Shape: [2, 3, 6]

// Concatenation
let a = zeros::<f32>(&[2, 3]);
let b = ones::<f32>(&[2, 3]);
let cat = Tensor::cat(&[&a, &b], 0).unwrap();      // [4, 3]

// Broadcasting a tensor to a target shape
let b1 = ones::<f32>(&[1, 3]);
let broadcasted = b1.broadcast_to(&[4, 3]);        // [4, 3]
```

## Arithmetic Operations

### Element-wise

```rust
let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[3, 4]);

// Elementwise add/sub/mul/div — return Result (broadcasting may fail)
let sum = a.add(&b).unwrap();
let dif = a.sub(&b).unwrap();
let prd = a.mul(&b).unwrap();
let quo = a.div(&b).unwrap();

// Scalar ops
let s1 = a.add_scalar(1.0);
let s2 = a.mul_scalar(2.0);
let n = a.neg();

// Unary
let sq = a.sqrt();
let ex = a.exp();
let lg = a.ln();
let pw = a.pow(2.0_f32);
```

The `std::ops` operators (`+`, `-`, `*`, `/`) are implemented for references and panic on error (e.g. shape mismatch). Prefer the explicit methods for fallible broadcast contexts.

### Matrix Operations

```rust
let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[4, 5]);

// Matrix multiplication (routes to cuBLAS on GPU; GEMV fast path for m=1)
let c = a.matmul(&b).unwrap();                     // [3, 5]

// Batched matmul works via matmul with 3-D operands: [B, M, K] @ [B, K, N]
let ba = randn::<f32>(&[2, 3, 4]);
let bb = randn::<f32>(&[2, 4, 5]);
let bc = ba.matmul(&bb).unwrap();                  // [2, 3, 5]

// 1-D dot product
let v1 = randn::<f32>(&[100]);
let v2 = randn::<f32>(&[100]);
let dp = v1.dot(&v2).unwrap();                     // scalar tensor
```

### Broadcasting

NumPy rules apply:

```rust
let a = randn::<f32>(&[3, 4, 5]);
let b = randn::<f32>(&[5]);                        // broadcasts to [3, 4, 5]
let c = randn::<f32>(&[4, 1]);                     // broadcasts to [3, 4, 5]

let d = a.add(&b).unwrap();
let e = a.mul(&c).unwrap();
```

## Reduction Operations

```rust
let t = randn::<f32>(&[3, 4, 5]);

// Full reductions
let s  = t.sum();                                  // 0-D tensor
let m  = t.mean().unwrap();
let p  = t.prod();
let mx = t.max().unwrap();
let mn = t.min().unwrap();

// argmax / argmin (global, return usize index)
let ai = t.argmax().unwrap();
let aj = t.argmin().unwrap();

// Dim reductions — take `i32` dim (negative indexes from the end)
// and `keepdim: bool`.
let sd = t.sum_dim(1, true);                       // [3, 1, 5]
let md = t.mean_dim(1, false);                     // [3, 5]
let vd = t.var_dim(1, true);                       // [3, 1, 5]
```

## Activation Functions

```rust
let t = randn::<f32>(&[3, 4]);

let r   = t.relu();
let s   = t.sigmoid();
let th  = t.tanh();
let sm  = t.softmax(1);                            // dim 1
let lsm = t.log_softmax(1);
let g   = t.gelu();
let si  = t.silu();
```

## Device Management

```rust
use axonml_core::Device;

// Create on CPU, transfer to another device
let t = randn::<f32>(&[1000, 1000]);
let t_gpu = t.to_device(Device::Cuda(0)).unwrap();

// Transfer back to CPU (same as to_device(Device::Cpu))
let t_cpu = t_gpu.cpu().unwrap();

// Device / layout queries
assert_eq!(t_cpu.device(), Device::Cpu);
assert!(t_cpu.is_contiguous());
```

`Tensor::to_device` and `Tensor::cpu` both return `Result<Self>`. Moving tensors between mismatched devices is a runtime check, not a compile-time one.

## Data Type

```rust
use axonml_core::DType;

assert_eq!(<f32 as axonml_core::dtype::Scalar>::DTYPE, DType::F32);
assert_eq!(DType::F32.size_of(), 4);
assert!(DType::F32.is_float());
assert!(DType::I64.is_signed());
assert_eq!(DType::default_float(), DType::F32);
```

Per-element dtype conversion on `Tensor<T>` is done by iterating and constructing a new `Tensor<U>`. The runtime `DType` enum exists for ONNX, serialization, and backend dispatch. For f16/f32 round-trip checks the tensor type provides:

```rust
let f = randn::<f32>(&[10]);
let h = f.to_f16_precision();                      // round to f16 then back
let roundtrip_error = f.has_f16_rounding_error();
```

## Lazy and Sparse Tensors

- `axonml_tensor::lazy::{LazyTensor, LazyOp}` — defers execution and applies algebraic simplification (constant folding, identity elimination, inverse cancellation, scalar folding) before realization. No external JIT needed.
- `axonml_tensor::sparse` — COO-format sparse tensors.

---

*Last updated: 2026-06-06 (v0.6.5)*
