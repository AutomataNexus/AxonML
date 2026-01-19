# DType Module

Data type system for tensor element types.

## Overview

The dtype module defines the supported data types for tensor elements. This provides type safety and enables efficient memory layout.

## DType Enum

```rust
pub enum DType {
    // Floating point
    F16,    // 16-bit float (half precision)
    F32,    // 32-bit float (single precision)
    F64,    // 64-bit float (double precision)
    BF16,   // Brain float 16

    // Integers
    I8,     // 8-bit signed
    I16,    // 16-bit signed
    I32,    // 32-bit signed
    I64,    // 64-bit signed

    // Unsigned integers
    U8,     // 8-bit unsigned
    U16,    // 16-bit unsigned
    U32,    // 32-bit unsigned
    U64,    // 64-bit unsigned

    // Boolean
    Bool,
}
```

## Usage Examples

### Getting Element Size

```rust
use axonml_core::DType;

let dtype = DType::F32;
assert_eq!(dtype.size_bytes(), 4);

let dtype64 = DType::F64;
assert_eq!(dtype64.size_bytes(), 8);
```

### Type Checking

```rust
use axonml_core::DType;

let dtype = DType::F32;

assert!(dtype.is_floating_point());
assert!(!dtype.is_integer());
assert!(!dtype.is_signed()); // floating point is neither signed nor unsigned int
```

### Default Types

```rust
use axonml_core::DType;

// Default floating point type
let default_float = DType::default_float(); // F32

// Default integer type
let default_int = DType::default_int(); // I64
```

## Type Conversion

### Promotion Rules

When operating on tensors with different dtypes, automatic promotion follows these rules:

1. `Bool` promotes to any numeric type
2. Integers promote to floats when mixed
3. Smaller types promote to larger types
4. Signed and unsigned of same size: both promote to next larger signed

```rust
// Example promotions:
// F32 + F64 -> F64
// I32 + F32 -> F32
// I32 + I64 -> I64
// Bool + F32 -> F32
```

### Explicit Casting

```rust
use axonml::prelude::*;

let float_tensor = Tensor::from_vec(vec![1.5, 2.7, 3.2], &[3]).unwrap();

// Cast to integer (truncates)
let int_tensor = float_tensor.to_dtype(DType::I32);
// Result: [1, 2, 3]
```

## Memory Layout

Each dtype has a specific memory layout:

| DType | Size (bytes) | Alignment |
|-------|--------------|-----------|
| F16   | 2            | 2         |
| F32   | 4            | 4         |
| F64   | 8            | 8         |
| BF16  | 2            | 2         |
| I8    | 1            | 1         |
| I16   | 2            | 2         |
| I32   | 4            | 4         |
| I64   | 8            | 8         |
| Bool  | 1            | 1         |

## Best Practices

1. **Use F32 for training** - Good balance of precision and memory
2. **Use F16/BF16 for inference** - Faster with minimal accuracy loss
3. **Use I64 for indices** - Handles large datasets
4. **Avoid unnecessary casts** - Each cast has overhead

## Related

- [Tensor](../tensor/tensor.md) - Tensors using dtypes
- [Storage](storage.md) - Raw storage with dtype awareness

@version 0.1.0
@author AutomataNexus Development Team
