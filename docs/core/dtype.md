---
layout: default
title: DType
parent: Core
nav_order: 2
description: "Data type enum and Scalar/Numeric/Float trait hierarchy"
---

# DType Module

Type system for AxonML's generic tensor operations (`axonml_core::dtype`).

## Overview

Two layers:

1. **`DType`** — a runtime enum for dtype dispatch (ONNX, serialization, backend kernels).
2. **`Scalar` / `Numeric` / `Float` traits** — compile-time generics so `Tensor<T>` is monomorphized for each element type.

## DType Enum

```rust
use axonml_core::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    F16,
    #[default]
    F32,
    F64,
    I8, I16, I32, I64,
    U8, U32, U64,
    Bool,
}
```

### Methods

```rust
use axonml_core::DType;

// Size (const fn)
assert_eq!(DType::F32.size_of(), 4);
assert_eq!(DType::F64.size_of(), 8);
assert_eq!(DType::Bool.size_of(), 1);
assert_eq!(DType::F16.size_of(), 2);

// Classification
assert!(DType::F32.is_float());
assert!(DType::I32.is_integer());
assert!(DType::I32.is_signed());
assert!(!DType::U32.is_signed());

// Name
assert_eq!(DType::F32.name(), "f32");
assert_eq!(format!("{}", DType::F32), "f32");      // Display uses name()

// Defaults
assert_eq!(DType::default(),      DType::F32);
assert_eq!(DType::default_float(), DType::F32);
assert_eq!(DType::default_int(),   DType::I64);
```

Note: `bf16` is not in the enum — only `F16`. Integer widths are I8/I16/I32/I64 + U8/U32/U64 (no `U16`).

## Trait Hierarchy

Three compile-time traits let generic tensor code dispatch without boxing:

```rust
use axonml_core::dtype::{Scalar, Numeric, Float};

// Scalar: any storable element (Pod + Zeroable + Send + Sync)
pub trait Scalar: Copy + Clone + Debug + Default + Send + Sync + Pod + Zeroable + 'static {
    const DTYPE: DType;
    fn dtype() -> DType { Self::DTYPE }
}

// Numeric: adds arithmetic (Num + NumCast + PartialOrd + Zero + One)
pub trait Numeric: Scalar + /* ... */ {
    const ZERO: Self;
    const ONE: Self;
    fn min_value() -> Self;
    fn max_value() -> Self;
}

// Float: adds exp/ln/pow/sqrt/sin/cos/tanh + NaN/Inf/EPSILON constants
pub trait Float: Numeric + /* num_traits::Float */ {
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const EPSILON: Self;
    fn is_nan_value(self) -> bool;
    fn is_infinite_value(self) -> bool;
    fn exp_value(self) -> Self;
    fn ln_value(self) -> Self;
    fn pow_value(self, exp: Self) -> Self;
    fn sqrt_value(self) -> Self;
    fn sin_value(self) -> Self;
    fn cos_value(self) -> Self;
    fn tanh_value(self) -> Self;
}
```

### Implementations

- `Scalar`: `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u32`, `u64`, plus wrapper types `F16Wrapper(half::f16)` and `BoolWrapper(u8)` — these wrappers exist because `bytemuck` does not implement `Pod` for `half::f16` or `bool` directly.
- `Numeric`: `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`.
- `Float`: `f32`, `f64`.

### Wrapper Types

```rust
use axonml_core::dtype::{F16Wrapper, BoolWrapper};

// f16
let h = F16Wrapper(half::f16::from_f32(3.14));

// Bool (u8-backed)
let b: BoolWrapper = true.into();
let raw: bool = b.into();
```

## Usage

### Query a generic `T`'s runtime dtype

```rust
use axonml_core::dtype::Scalar;

fn name_of<T: Scalar>() -> &'static str {
    T::dtype().name()
}

assert_eq!(name_of::<f32>(), "f32");
assert_eq!(name_of::<i64>(), "i64");
```

### Memory Layout

| DType | Size (bytes) |
|-------|--------------|
| Bool, I8, U8 | 1 |
| F16, I16     | 2 |
| F32, I32, U32 | 4 |
| F64, I64, U64 | 8 |

Alignment matches size (Pod + Zeroable invariants).

## Best Practices

1. **Use f32 for training** — good precision/memory balance.
2. **Use f16 via `F16Wrapper` for inference** — half the memory; `axonml-autograd::amp::autocast(DType::F16, ...)` handles the mixed-precision pattern.
3. **Use i64 for indices** — matches PyTorch and ONNX conventions.
4. **Avoid unnecessary conversions** — per-element conversion copies the tensor.

## Related

- [Device]({% link core/device.md %}) — where tensors live
- [Tensor operations]({% link tensors.md %}) — where these element types are used

---

*Last updated: 2026-04-16 (v0.6.1)*
