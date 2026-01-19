# axonml-tensor Documentation

> N-dimensional tensor operations for the Axonml ML framework.

## Overview

`axonml-tensor` provides the core `Tensor` type that serves as the foundation for all machine learning operations in Axonml. Tensors are multi-dimensional arrays with support for automatic broadcasting, device placement, and efficient memory sharing through views.

## Modules

### tensor.rs

The main `Tensor<T>` struct and its operations.

```rust
pub struct Tensor<T: Scalar> {
    storage: Storage<T>,
    shape: Shape,
    strides: Strides,
    offset: usize,
}
```

**Key methods:**

Shape Information:
- `shape()` - Get tensor dimensions
- `ndim()` - Number of dimensions
- `numel()` - Total number of elements
- `size(dim)` - Size of specific dimension

Data Access:
- `get(indices)` - Get element at indices
- `set(indices, value)` - Set element at indices
- `item()` - Get scalar value (for single-element tensors)
- `to_vec()` - Convert to vector

Shape Operations:
- `reshape(shape)` - Change shape
- `flatten()` - Flatten to 1D
- `transpose(d0, d1)` - Swap dimensions
- `squeeze(dim)` - Remove size-1 dimensions
- `unsqueeze(dim)` - Add size-1 dimension
- `permute(dims)` - Reorder dimensions
- `contiguous()` - Make contiguous copy

### shape.rs

Shape and stride utilities.

**Types:**
- `Shape` - `SmallVec<[usize; 6]>` for dimensions
- `Strides` - `SmallVec<[isize; 6]>` for strides

**Functions:**
- `numel(shape)` - Total elements
- `contiguous_strides(shape)` - Row-major strides
- `is_contiguous(shape, strides)` - Check contiguity
- `broadcast_shape(s1, s2)` - Compute broadcast shape
- `reshape(old, new)` - Compute reshape
- `squeeze(shape, dim)` - Remove size-1 dims
- `unsqueeze(shape, dim)` - Add size-1 dim

### creation.rs

Tensor factory functions.

**Zero/One:**
```rust
zeros::<f32>(&[2, 3])    // All zeros
ones::<f32>(&[2, 3])     // All ones
full::<f32>(&[2, 3], v)  // Fill with value
eye::<f32>(n)            // Identity matrix
diag(&[1.0, 2.0, 3.0])   // Diagonal matrix
```

**Random:**
```rust
rand::<f32>(&[10])       // Uniform [0, 1)
randn::<f32>(&[10])      // Normal(0, 1)
uniform(&[10], lo, hi)   // Uniform [lo, hi)
normal(&[10], mu, std)   // Normal(mu, std)
randint(&[10], lo, hi)   // Random integers
```

**Ranges:**
```rust
arange(start, end, step) // Range with step
linspace(start, end, n)  // N evenly spaced
logspace(s, e, n, base)  // Log-spaced
```

### view.rs

Slicing and indexing operations.

**Methods:**
- `slice_dim0(start, end)` - Slice first dimension
- `select(dim, index)` - Select single index
- `narrow(dim, start, len)` - Narrow a dimension
- `chunk(n, dim)` - Split into chunks
- `split(sizes, dim)` - Split by sizes
- `gather(dim, indices)` - Gather by indices
- `masked_select(mask)` - Boolean masking

**Standalone:**
- `cat(tensors, dim)` - Concatenate
- `stack(tensors, dim)` - Stack along new dimension

### ops/mod.rs

Additional operations.

**Activations:**
- `softmax(x, dim)` - Softmax
- `log_softmax(x, dim)` - Log-softmax
- `gelu(x)` - GELU activation
- `leaky_relu(x, neg_slope)` - Leaky ReLU
- `elu(x, alpha)` - ELU activation
- `silu(x)` - SiLU/Swish

**Clipping:**
- `clamp(x, min, max)` - Clamp to range
- `clamp_min(x, min)` - Clamp minimum
- `clamp_max(x, max)` - Clamp maximum

**Comparisons:**
- `eq(a, b)` - Element-wise equality
- `lt(a, b)` - Less than
- `gt(a, b)` - Greater than

**Selection:**
- `where_cond(cond, x, y)` - Conditional selection

## Usage Examples

### Basic Operations

```rust
use axonml_tensor::prelude::*;

// Create tensors
let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[3, 4]);

// Arithmetic
let c = &a + &b;
let d = &a * &b;
let e = a.matmul(&b.t()?)?;

// Reductions
let sum = c.sum();
let mean = c.mean()?;
let max = c.max()?;
```

### Broadcasting

```rust
let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[4]);     // Will broadcast

let c = &a + &b;  // Shape: [3, 4]
```

### Shape Manipulation

```rust
let a = randn::<f32>(&[2, 3, 4]);

let b = a.reshape(&[6, 4])?;
let c = a.flatten();
let d = a.transpose(0, 2)?;
let e = a.permute(&[2, 0, 1])?;
```

### Slicing

```rust
let a = arange::<f32>(0.0, 24.0, 1.0).reshape(&[4, 6])?;

let row = a.select(0, 0)?;        // First row
let col = a.select(1, 0)?;        // First column
let sub = a.narrow(0, 1, 2)?;     // Rows 1-2
let chunks = a.chunk(2, 0)?;      // Split into 2
```

## Feature Flags

- `std` (default) - Enable standard library
