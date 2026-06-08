# axonml-fusion

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-fusion"><img src="https://img.shields.io/badge/crates.io-0.6.5-green.svg" alt="Version: 0.6.5"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-fusion` provides kernel fusion for AxonML: combining sequences of ops
into a single pass to cut memory traffic and kernel-launch overhead. It
exposes a `FusedOp` trait, a `FusedLinear` (matmul + bias + activation) kernel
backed by `matrixmultiply`, an `ElementwiseOp` chain with a fluent builder,
and a `FusionOptimizer` that runs pattern detection over an `[OpType]` graph
and reports statistics.

## Features

- **Pattern Detection** — `detect_patterns(&[OpType])` returns fusion opportunities with start/end indices
- **Linear Fusion** — `FusedLinear` (MatMul + optional bias + activation) with `None`, `Relu`, `Gelu` (tanh approximation), `Sigmoid`, `Tanh`, `Silu`
- **Elementwise Fusion** — `FusedElementwise` chain with `FusedElementwise::builder()` fluent API
- **Graph Optimizer** — `FusionOptimizer` with `FusionConfig::all_enabled()` / `FusionConfig::conservative()`; `OptimizationStats` with `fusions_applied`, `ops_eliminated`, `estimated_speedup`
- **Convenience Constructors** — `fuse_matmul_bias_relu`, `fuse_matmul_bias_gelu`, `fuse_matmul_bias`, `fused_add_relu`, `fused_mul_add`, `fused_scale_bias_relu`, `fuse_elementwise`, `optimize_graph`, `estimate_speedup`
- **Parallel Execution** — Rayon for tensor operations; `matrixmultiply` for the inner GEMM

## Modules

| Module | Description |
|--------|-------------|
| `patterns` | `FusionPattern` enum, `OpType` enum, `detect_patterns` |
| `elementwise` | `ElementwiseOp`, `FusedElementwise`, `FusedElementwiseBuilder`, convenience constructors |
| `linear` | `Activation` enum, `FusedLinear` (MatMul + Bias + Activation), `fuse_matmul_bias*` constructors |
| `optimizer` | `FusionConfig`, `FusionOptimizer`, `OptimizationStats`, `optimize_graph`, `estimate_speedup` |
| `error` | `FusionError` / `FusionResult` |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-fusion = "0.6.5"
```

### Fused Linear Operations

```rust
use axonml_fusion::{fuse_matmul_bias_relu, FusedLinear, Activation};
use axonml_tensor::Tensor;

// Create weight and bias tensors
let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let bias   = Tensor::from_vec(vec![0.5, 0.5], &[2])?;

// Create fused MatMul + Bias + ReLU operation
let fused = fuse_matmul_bias_relu(&weight, &bias)?;

// Execute fused operation
let input  = Tensor::from_vec(vec![1.0, 1.0], &[1, 2])?;
let output = fused.forward(&input)?;

// Or construct directly
let fl = FusedLinear::new(weight, Some(bias), Activation::Gelu)?;
```

`FusedLinear::forward` accepts 1D or 2D inputs (batch × in_features).

### Fused Elementwise Operations

```rust
use axonml_fusion::{FusedElementwise, ElementwiseOp};

// Build a fused elementwise chain using the builder
let fused = FusedElementwise::builder()
    .mul(2.0)      // Scale by 2
    .add(1.0)      // Add bias
    .relu()        // Apply ReLU
    .build();

let output = fused.forward(&input)?;

// Or construct from an explicit op list
let f = FusedElementwise::new(vec![
    ElementwiseOp::MulConst(2.0),
    ElementwiseOp::AddConst(1.0),
    ElementwiseOp::Relu,
]);
```

### Graph Optimization

```rust
use axonml_fusion::{optimize_graph, patterns::OpType};

// Define operation sequence
let ops = vec![
    OpType::MatMul,
    OpType::Add,
    OpType::Relu,
    OpType::Add,
    OpType::Mul,
];

// Optimize with default configuration (None uses all-enabled defaults)
let (patterns, stats) = optimize_graph(&ops, None)?;

println!("Fusions applied:      {}", stats.fusions_applied);
println!("Operations eliminated: {}", stats.ops_eliminated);
println!("Estimated speedup:    {:.2}x", stats.estimated_speedup);
```

### Custom Fusion Configuration

```rust
use axonml_fusion::{FusionOptimizer, FusionConfig};

// Preset configurations
let config = FusionConfig::conservative();
let config = FusionConfig::all_enabled();

// Or customize
let config = FusionConfig {
    fuse_elementwise: true,
    fuse_linear: true,
    fuse_conv: false,
    min_elementwise_chain: 3,
    aggressive: false,
};

let optimizer = FusionOptimizer::with_config(config);
let patterns  = optimizer.analyze(&ops);
```

## Supported Fusion Patterns

All variants of `FusionPattern` with their `num_ops()` and `estimated_speedup()`:

| Pattern | Operations | num_ops | Estimated Speedup |
|---------|------------|---------|-------------------|
| `MatMulBias` | MatMul, Add | 2 | 1.2x |
| `MatMulBiasRelu` | MatMul, Add, ReLU | 3 | 1.3x |
| `MatMulBiasGelu` | MatMul, Add, GELU | 3 | 1.3x |
| `ConvBatchNorm` | Conv, BatchNorm | 3 | 1.3x |
| `ConvBatchNormRelu` | Conv, BatchNorm, ReLU | 4 | 1.4x |
| `ElementwiseChain` | 2+ elementwise ops | variable | 2.0x |
| `Softmax` | 3 ops | 3 | 1.2x |
| `LayerNorm` | 4 ops | 4 | 1.2x |
| `GeluApprox` | 5 ops | 5 | 1.2x |
| `AddRelu` | Add, ReLU | 2 | 1.8x |
| `MulAdd` | Mul, Add (FMA) | 2 | 1.5x |

## Elementwise Operations

`FusedElementwise::builder()` methods:

| Method | Op |
|--------|----|
| `add(c: f32)` | `ElementwiseOp::AddConst(c)` |
| `mul(c: f32)` | `ElementwiseOp::MulConst(c)` |
| `relu()` | `ElementwiseOp::Relu` |
| `leaky_relu(alpha: f32)` | `ElementwiseOp::LeakyRelu(alpha)` |
| `sigmoid()` | `ElementwiseOp::Sigmoid` |
| `tanh()` | `ElementwiseOp::Tanh` |
| `exp()` | `ElementwiseOp::Exp` |
| `log()` | `ElementwiseOp::Log` (natural log) |
| `sqrt()` | `ElementwiseOp::Sqrt` |
| `square()` | `ElementwiseOp::Square` |
| `clamp(min, max)` | `ElementwiseOp::Clamp(min, max)` |
| `neg()` | `ElementwiseOp::Neg` |
| `abs()` | `ElementwiseOp::Abs` |

## Convenience Constructors

- `fuse_matmul_bias(weight, bias)` — `Activation::None`
- `fuse_matmul_bias_relu(weight, bias)` — `Activation::Relu`
- `fuse_matmul_bias_gelu(weight, bias)` — `Activation::Gelu`
- `fused_add_relu(bias: f32)` — add then ReLU
- `fused_mul_add(scale, bias)` — FMA-style chain
- `fused_scale_bias_relu(scale, bias)` — norm-style chain
- `fuse_elementwise(ops: Vec<ElementwiseOp>)` — explicit op chain

## Tests

```bash
cargo test -p axonml-fusion
```

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
