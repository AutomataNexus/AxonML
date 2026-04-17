---
layout: default
title: Error
parent: Core
nav_order: 3
description: "AxonML Error enum and Result alias"
---

# Error Module

Unified error handling for AxonML (`axonml_core::error`).

## Overview

`Error` is a `thiserror`-derived enum covering every failure mode the tensor / device / autograd stack can produce. All fallible tensor ops return `axonml_core::Result<T>` (i.e. `std::result::Result<T, Error>`).

## Error Enum

Authoritative definition from `crates/axonml-core/src/error.rs`:

```rust
use axonml_core::{Device, DType};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum Error {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("DType mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch {
        expected: DType,
        actual: DType,
    },

    #[error("Device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        expected: Device,
        actual: Device,
    },

    #[error("Invalid dimension: index {index} for tensor with {ndim} dimensions")]
    InvalidDimension { index: i64, ndim: usize },

    #[error("Index out of bounds: index {index} for dimension of size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Memory allocation failed: requested {size} bytes on {device:?}")]
    AllocationFailed { size: usize, device: Device },

    #[error("Device not available: {device:?}")]
    DeviceNotAvailable { device: Device },

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    #[error("Cannot broadcast shapes {shape1:?} and {shape2:?}")]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    #[error("Operation not supported on empty tensor")]
    EmptyTensor,

    #[error("Operation requires contiguous tensor")]
    NotContiguous,

    #[error("Gradient error: {message}")]
    GradientError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}
```

## Result Type

```rust
pub type Result<T> = core::result::Result<T, Error>;
```

## Constructors

Helper functions for the most common variants:

```rust
use axonml_core::Error;

let e1 = Error::shape_mismatch(&[2, 3], &[2, 4]);
let e2 = Error::invalid_operation("cannot invert a singular matrix");
let e3 = Error::internal("this should never happen");
```

## Usage

### Matching Specific Variants

```rust
use axonml_core::{Error, Result};
use axonml_tensor::Tensor;

fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    a.add(b)
}

match add(&t1, &t2) {
    Ok(out) => { /* use out */ }
    Err(Error::ShapeMismatch { expected, actual }) => {
        eprintln!("shape mismatch: {expected:?} vs {actual:?}");
    }
    Err(Error::BroadcastError { shape1, shape2 }) => {
        eprintln!("can't broadcast {shape1:?} with {shape2:?}");
    }
    Err(Error::DeviceMismatch { expected, actual }) => {
        eprintln!("device mismatch: {expected:?} vs {actual:?}");
    }
    Err(e) => eprintln!("other error: {e}"),
}
```

### Propagation

Every tensor method that can fail uses `Result<T>`, so `?` composes naturally:

```rust
use axonml_core::Result;
use axonml_tensor::Tensor;

fn two_step(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    let sum = a.add(b)?;
    let out = sum.matmul(b)?;
    Ok(out)
}
```

## Common Errors and Causes

| Variant | Typical Cause |
|:--------|:--------------|
| `ShapeMismatch` | Feeding a tensor with the wrong shape into a Linear / Conv / loss. |
| `BroadcastError` | Trying to `add` / `mul` tensors whose shapes are not broadcast-compatible. |
| `DeviceMismatch` | Model is on GPU, input is still on CPU (or vice-versa). Move inputs with `Tensor::to_device` each batch. |
| `DTypeMismatch` | Mixing `Tensor<f32>` and `Tensor<f64>` in one op. |
| `InvalidDimension` | `sum_dim(3, ...)` on a 2-D tensor, or a negative dim that resolves out of range. |
| `IndexOutOfBounds` | `.get(&[5])` on a length-3 axis. |
| `AllocationFailed` | OOM — check `Device::capabilities().available_memory`. |
| `DeviceNotAvailable` | Asked for `Device::Cuda(1)` when only one GPU is present, or the `cuda` feature is off. |
| `NotContiguous` | An op that requires contiguous layout got a strided view. Insert `.contiguous()`. |
| `EmptyTensor` | Reducing or indexing a zero-element tensor. |
| `GradientError` | Disconnected autograd graph (e.g. constructing a fresh `Variable::new(tensor.to_vec() ...)` in the middle of a forward pass severs the graph — use framework built-ins instead). |

---

*Last updated: 2026-04-16 (v0.6.1)*
