# axonml-tensor Documentation

> N-dimensional tensor operations for the AxonML ML framework.

## Overview

`axonml-tensor` provides `Tensor<T>`, AxonML's generic strided
multi-dimensional array. Features: NumPy-style broadcasting, zero-copy
strided views, CPU and CUDA backends, quantized matmul dispatch (Q4_K /
Q5_K / Q6_K / Q8_0 in-shader dequant), lazy tensors with algebraic
optimization, sparse COO tensors, and the full factory-function suite.

**Device-native CPU parallelism (0.6.5):** the CPU backend is rayon-parallel
above a work threshold across matmul (regular, B-transposed GGUF `[out,in]`,
f64, generic-tiled, and 3D/4D batched), reductions
(sum/mean/prod/max/min/argmax/argmin), SwiGLU, RMSNorm (+heads/batched), RoPE,
`layer_norm`, `gelu`, and residual adds — with contiguous/offset-0 fast-paths
that skip the `to_vec` copy before the backend. Tiny ops stay serial with
identical semantics; the GPU path is unchanged.

## Modules

### `tensor`

The core `Tensor<T>` struct.

```rust
pub struct Tensor<T: Scalar> {
    storage: Storage<T>,
    shape: Shape,
    strides: Strides,
    offset: usize,
}
```

**Shape info:** `shape()`, `ndim()`, `numel()`, `size(dim)`
**Data access:** `get(indices)`, `set(indices, value)`, `item()`, `to_vec()`
**Shape ops:** `reshape(shape)`, `flatten()`, `transpose(d0, d1)`,
`squeeze(dim)`, `unsqueeze(dim)`, `permute(dims)`, `contiguous()`
**Arithmetic:** `+ - * /` operators, `matmul`, `neg`, `abs`, `pow`
**Reductions:** `sum`, `mean`, `max`, `min`, `var_dim`

### `shape`

`Shape` (`SmallVec<[usize; 6]>`) and `Strides` (`SmallVec<[isize; 6]>`)
aliases plus stride/broadcast utilities: `numel`, `contiguous_strides`,
`is_contiguous`, `broadcast_shape`, `reshape`, `squeeze`, `unsqueeze`.

### `creation`

Tensor factory functions.

**Zero / one / full:**

```rust
zeros::<f32>(&[2, 3])
ones::<f32>(&[2, 3])
full::<f32>(&[2, 3], v)
eye::<f32>(n)
diag(&[1.0, 2.0, 3.0])
```

**Random:**

```rust
rand::<f32>(&[10])              // Uniform [0, 1)
randn::<f32>(&[10])             // Normal(0, 1)
uniform(&[10], lo, hi)
normal(&[10], mu, std)
randint(&[10], lo, hi)
```

**Ranges:**

```rust
arange(start, end, step)
linspace(start, end, n)
logspace(start, end, n, base)
```

### `view`

Slicing, indexing, and splitting.

- `slice_dim0(start, end)` — slice first dimension
- `select(dim, index)` — select single index
- `narrow(dim, start, len)` — narrow a dimension
- `chunk(n, dim)` — split into `n` equal chunks
- `split(sizes, dim)` — split by explicit sizes
- `gather(dim, indices)` — gather by indices
- `masked_select(mask)` — boolean mask
- Standalone: `cat(tensors, dim)`, `stack(tensors, dim)`

### `ops`

Higher-level free functions (1133 lines).

**Comparisons:** `eq`, `lt`, `gt`, and the mask variants `eq_mask`,
`lt_mask`, `gt_mask`
**Softmax:** `softmax(x, dim)`, `log_softmax(x, dim)` (numerically stable)
**Activations:** `gelu`, `leaky_relu`, `elu`, `silu`, `mish`
**Clipping:** `clamp`, `clamp_min`, `clamp_max`
**Selection:** `where_cond(cond, x, y)`
**Stats:** `var_dim` (Welford single-pass variance)
**Training:** `dropout`, `layer_norm`, `batch_norm` (with running mean/var
+ affine)

### `lazy`

Deferred computation with algebraic optimization (976 lines).

**Types:**
- `LazyTensor` — a tensor wrapped in a deferred expression tree
- `LazyOp` — op nodes: `Tensor`, unary (`Neg`/`Relu`/`Sigmoid`/`Tanh`/`Exp`
  /`Log`/`Sqrt`/`Abs`), binary (`Add`/`Sub`/`Mul`/`Div`/`MatMul`/`Pow`),
  reductions (`Sum`/`Mean`), shape (`Reshape`/`Transpose`), scalar
  (`Scalar`/`AddScalar`/`MulScalar`)

**Usage:**

```rust
use axonml_tensor::lazy::LazyTensor;

let a = LazyTensor::from_tensor(tensor_a);
let b = LazyTensor::from_tensor(tensor_b);
let result = a.add(&b).mul_scalar(2.0).neg().neg();

let optimized = result.optimize();
let concrete = optimized.materialize();
```

`optimize()` performs: constant folding, identity elimination (`x+0`,
`x*1`, `x-0`), inverse cancellation (`neg(neg)`, `exp(log)`, `log(exp)`),
and scalar folding (`(x*2)*3` → `x*6`).

### `sparse`

`SparseCOO` — coordinate-format sparse tensor (f32 values), with
`from_dense`, `to_dense`, `nnz`, `density`, sparse+sparse and sparse+dense
add/mul, `spmm` (sparse × dense → dense), `coalesce` (sort + dedup), and
`transpose`. `SparseFormat` tags the COO/CSR/CSC variants (COO is the
implemented one).

### `cuda_ops` *(feature = `cuda`)*

3215 lines, 34 GPU methods on `Tensor<f32>`, dispatched through the
`CudaBackend` singleton:

- Placement: `to_device`, `contiguous_gpu`, `to_vec` for GPU tensors
- Elementwise: add/sub/mul/div/scalar, neg, abs, pow
- Activations: relu, sigmoid, tanh, gelu, silu, elu, leaky_relu, softmax,
  log_softmax
- Reductions: sum, mean, max, min
- Linear algebra: matmul (cuBLAS GEMM), layernorm, RMSNorm, transpose
- Quantized matmul: `q4k_gemv_cuda`, `q4k_gemm_cuda`, `q6k_gemv_cuda`,
  `q6k_gemm_cuda` (in-shader Q4_K / Q6_K dequant)
- Other: `embedding_gather`, `dropout`

Re-exports `pool_alloc` and `get_cuda_backend` for other crates.

## Usage Examples

### Basic

```rust
use axonml_tensor::prelude::*;

let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[3, 4]);

let c = &a + &b;
let d = &a * &b;
let e = a.matmul(&b.t()?)?;

let sum = c.sum();
let mean = c.mean()?;
let max = c.max()?;
```

### Broadcasting

```rust
let a = randn::<f32>(&[3, 4]);
let b = randn::<f32>(&[4]);
let c = &a + &b; // [3, 4]
```

### Shape manipulation

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
let row    = a.select(0, 0)?;
let col    = a.select(1, 0)?;
let sub    = a.narrow(0, 1, 2)?;
let chunks = a.chunk(2, 0)?;
```

## Feature Flags

- `std` (default) — standard library
- `cuda` — enables `cuda_ops` and CUDA-backed tensor storage
- `cudnn` — cuDNN conv2d (implies `cuda`)

## Last updated

0.6.5 (2026-06-06)
