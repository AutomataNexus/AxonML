# axonml-fusion

[![Crates.io](https://img.shields.io/crates/v/axonml-fusion.svg)](https://crates.io/crates/axonml-fusion)
[![Docs.rs](https://docs.rs/axonml-fusion/badge.svg)](https://docs.rs/axonml-fusion)
[![Downloads](https://img.shields.io/crates/d/axonml-fusion.svg)](https://crates.io/crates/axonml-fusion)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Kernel fusion optimization for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-fusion` provides automatic kernel fusion to reduce memory bandwidth and improve performance. Fuses common operation patterns like MatMul+Bias+Activation into single optimized kernels.

## Features

### Fusion Patterns
- **FusedLinear** - MatMul + Bias + Activation
- **FusedConv** - Convolution + BatchNorm + Activation
- **FusedElementwise** - Chains of element-wise operations
- **FusedAttention** - QKV projection + attention + output

### Benefits
- **Up to 2x speedup** for memory-bound operations
- **Reduced memory traffic** - Fewer intermediate tensors
- **Automatic pattern detection** - No manual optimization needed

## Installation

```toml
[dependencies]
axonml-fusion = "0.1"
```

## Usage

### Automatic Fusion

```rust
use axonml_fusion::optimize_graph;

let model = create_model();

// Automatically detect and apply fusions
let optimized = optimize_graph(&model)?;

// Benchmark improvement
let original_time = benchmark(&model);
let optimized_time = benchmark(&optimized);
println!("Speedup: {:.2}x", original_time / optimized_time);
```

### FusedLinear

```rust
use axonml_fusion::FusedLinear;

// Fuses: y = relu(x @ W + b)
let layer = FusedLinear::new(784, 256)
    .activation(Activation::ReLU);

// Single kernel instead of 3 separate operations
let output = layer.forward(&input);
```

### FusedElementwise

```rust
use axonml_fusion::FusedElementwise;

// Fuses: y = (x + 1) * 2 - 0.5
let fused = FusedElementwise::new()
    .add(1.0)
    .mul(2.0)
    .sub(0.5);

let output = fused.forward(&input);
```

### Pattern Detection

```rust
use axonml_fusion::{FusionOptimizer, FusionPattern};

let optimizer = FusionOptimizer::new()
    .enable(FusionPattern::LinearBiasRelu)
    .enable(FusionPattern::ConvBnRelu)
    .enable(FusionPattern::ElementwiseChain);

let optimized = optimizer.optimize(&model)?;

// Print detected patterns
for pattern in optimizer.detected_patterns() {
    println!("Found: {:?}", pattern);
}
```

## Supported Patterns

| Pattern | Operations | Speedup |
|---------|------------|---------|
| LinearBiasRelu | MatMul + Bias + ReLU | ~1.5x |
| LinearBiasGelu | MatMul + Bias + GELU | ~1.5x |
| ConvBnRelu | Conv2d + BatchNorm + ReLU | ~1.8x |
| ElementwiseChain | Multiple element-wise ops | ~2x |
| SoftmaxCrossEntropy | Softmax + CrossEntropy | ~1.3x |

## Part of Axonml

```toml
[dependencies]
axonml = "0.1"  # Includes fusion
```

## License

MIT OR Apache-2.0
