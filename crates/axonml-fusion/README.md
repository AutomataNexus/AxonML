# axonml-fusion

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-fusion"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version: 0.1.0"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-fusion` provides kernel fusion support for combining multiple operations into single optimized kernels. By reducing memory bandwidth requirements and kernel launch overhead, fusion significantly improves performance for neural network inference and training workloads.

## Features

- **Pattern Detection**: Automatic detection of fusible operation patterns in computational graphs
- **Linear Fusion**: Fused MatMul + Bias + Activation operations (ReLU, GELU, Sigmoid, Tanh, SiLU)
- **Elementwise Fusion**: Chain multiple elementwise operations into single memory-efficient kernels
- **Graph Optimizer**: Configurable fusion optimizer with conservative and aggressive modes
- **Builder Pattern**: Fluent API for constructing fused elementwise operation chains
- **Performance Statistics**: Track fusions applied, operations eliminated, and estimated speedup
- **Parallel Execution**: Leverages Rayon for parallel tensor operations

## Modules

| Module | Description |
|--------|-------------|
| `patterns` | Fusion pattern definitions and detection algorithms for MatMul, Conv, and Elementwise patterns |
| `elementwise` | Fused elementwise operations with builder pattern for chaining Add, Mul, ReLU, Sigmoid, etc. |
| `linear` | Fused linear layer operations combining MatMul + Bias + Activation |
| `optimizer` | Graph fusion optimizer with configurable passes and statistics tracking |
| `error` | Error types and Result alias for fusion operations |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-fusion = "0.1.0"
```

### Fused Linear Operations

```rust
use axonml_fusion::{fuse_matmul_bias_relu, FusedLinear, Activation};
use axonml_tensor::Tensor;

// Create weight and bias tensors
let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let bias = Tensor::from_vec(vec![0.5, 0.5], &[2])?;

// Create fused MatMul + Bias + ReLU operation
let fused = fuse_matmul_bias_relu(&weight, &bias)?;

// Execute fused operation
let input = Tensor::from_vec(vec![1.0, 1.0], &[2])?;
let output = fused.forward(&input)?;
```

### Fused Elementwise Operations

```rust
use axonml_fusion::{FusedElementwise, fused_scale_bias_relu};

// Build a fused elementwise chain using the builder
let fused = FusedElementwise::builder()
    .mul(2.0)      // Scale by 2
    .add(1.0)      // Add bias
    .relu()        // Apply ReLU
    .build();

let output = fused.forward(&input)?;

// Or use convenience functions
let scale_bias_relu = fused_scale_bias_relu(2.0, 1.0);
```

### Graph Optimization

```rust
use axonml_fusion::{optimize_graph, FusionConfig, OpType};

// Define operation sequence
let ops = vec![
    OpType::MatMul,
    OpType::Add,
    OpType::Relu,
    OpType::Add,
    OpType::Mul,
];

// Optimize with default configuration
let (patterns, stats) = optimize_graph(&ops, None)?;

println!("Fusions applied: {}", stats.fusions_applied);
println!("Operations eliminated: {}", stats.ops_eliminated);
println!("Estimated speedup: {:.2}x", stats.estimated_speedup);
```

### Custom Fusion Configuration

```rust
use axonml_fusion::{FusionOptimizer, FusionConfig};

// Create conservative configuration
let config = FusionConfig::conservative();

// Or customize specific settings
let config = FusionConfig {
    fuse_elementwise: true,
    fuse_linear: true,
    fuse_conv: false,
    min_elementwise_chain: 3,
    aggressive: false,
};

let mut optimizer = FusionOptimizer::with_config(config);
let patterns = optimizer.analyze(&ops);
```

## Supported Fusion Patterns

| Pattern | Operations | Estimated Speedup |
|---------|------------|-------------------|
| MatMul + Bias | MatMul, Add | 1.2x |
| MatMul + Bias + ReLU | MatMul, Add, ReLU | 1.3x |
| MatMul + Bias + GELU | MatMul, Add, GELU | 1.3x |
| Conv + BatchNorm | Conv, BatchNorm | 1.3x |
| Conv + BatchNorm + ReLU | Conv, BatchNorm, ReLU | 1.4x |
| Elementwise Chain | Multiple elementwise ops | 2.0x |
| Add + ReLU | Add, ReLU | 1.8x |
| Mul + Add (FMA) | Mul, Add | 1.5x |

## Elementwise Operations

The `FusedElementwise` builder supports:

- `add(f32)` - Add constant
- `mul(f32)` - Multiply by constant
- `relu()` - ReLU activation
- `leaky_relu(f32)` - Leaky ReLU with alpha
- `sigmoid()` - Sigmoid activation
- `tanh()` - Hyperbolic tangent
- `exp()` - Exponential
- `log()` - Natural logarithm
- `sqrt()` - Square root
- `square()` - Square
- `clamp(f32, f32)` - Clamp to range
- `neg()` - Negation
- `abs()` - Absolute value

## Tests

Run the test suite:

```bash
cargo test -p axonml-fusion
```

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
