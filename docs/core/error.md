# Error Module

Unified error handling for the Axonml framework.

## Overview

The error module provides a comprehensive error type that covers all possible failure modes across the framework. This enables consistent error handling and helpful error messages.

## Error Enum

```rust
pub enum Error {
    /// Shape mismatch in operation
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Invalid shape specification
    InvalidShape(String),

    /// Index out of bounds
    IndexOutOfBounds {
        index: usize,
        size: usize,
    },

    /// Device mismatch between tensors
    DeviceMismatch {
        expected: Device,
        got: Device,
    },

    /// Data type mismatch
    DTypeMismatch {
        expected: DType,
        got: DType,
    },

    /// Dimension mismatch
    DimensionMismatch {
        expected: usize,
        got: usize,
    },

    /// Invalid operation
    InvalidOperation(String),

    /// Gradient computation error
    GradientError(String),

    /// I/O error
    IoError(std::io::Error),

    /// Backend-specific error
    BackendError(String),
}
```

## Result Type

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

## Usage Examples

### Creating Errors

```rust
use axonml_core::{Error, Result};

fn check_shape(expected: &[usize], got: &[usize]) -> Result<()> {
    if expected != got {
        return Err(Error::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        });
    }
    Ok(())
}
```

### Handling Errors

```rust
use axonml::prelude::*;

fn process_tensor(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<f32>> {
    let tensor = Tensor::from_vec(data, shape)?;
    Ok(tensor)
}

fn main() {
    match process_tensor(vec![1.0, 2.0, 3.0], &[2, 2]) {
        Ok(t) => println!("Success: {:?}", t.shape()),
        Err(Error::ShapeMismatch { expected, got }) => {
            println!("Shape mismatch: expected {:?}, got {:?}", expected, got);
        }
        Err(e) => println!("Other error: {}", e),
    }
}
```

### Error Propagation

```rust
use axonml::prelude::*;

fn complex_operation(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Errors automatically propagate with ?
    let sum = a.add(b)?;
    let result = sum.matmul(b)?;
    Ok(result)
}
```

## Error Messages

All errors implement `Display` for human-readable messages:

```rust
use axonml_core::Error;

let err = Error::ShapeMismatch {
    expected: vec![2, 3],
    got: vec![3, 2],
};

println!("{}", err);
// Output: "Shape mismatch: expected [2, 3], got [3, 2]"
```

## Error Conversion

The Error type implements `From` for common error types:

```rust
// From std::io::Error
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IoError(e)
    }
}
```

This enables using `?` with standard library functions:

```rust
use axonml_core::Result;
use std::fs::File;

fn load_data(path: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;  // IoError automatically converted
    // ...
}
```

## Best Practices

1. **Use Result<T>** - Always return Result for fallible operations
2. **Propagate with ?** - Use the `?` operator for clean error propagation
3. **Add context** - Use `.map_err()` to add context when needed
4. **Match specifically** - Match on specific error variants when handling

## Related

- [Tensor](../tensor/tensor.md) - Operations that return Result
- [Variable](../autograd/variable.md) - Autograd operations with error handling

@version 0.1.0
@author AutomataNexus Development Team
