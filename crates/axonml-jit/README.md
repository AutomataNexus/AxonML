# axonml-jit

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-jit"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version: 0.1.0"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-jit` provides Just-In-Time compilation for tensor operations, enabling significant performance improvements through operation tracing, graph optimization, and compiled function caching. It builds computation graphs from traced operations and optimizes them before execution.

## Features

- **Operation Tracing**: Record tensor operations to build computation graphs automatically
- **Graph Optimization**: Constant folding, dead code elimination, algebraic simplification, and CSE
- **Function Caching**: LRU cache for compiled functions with configurable size
- **Comprehensive IR**: Rich intermediate representation supporting 40+ tensor operations
- **Shape Inference**: Automatic shape propagation including broadcast semantics
- **Native Compilation**: Prepared for Cranelift code generation (interpreter fallback available)
- **Thread-Local Tracing**: Safe concurrent tracing with thread-local state

## Modules

| Module | Description |
|--------|-------------|
| `ir` | Graph-based intermediate representation with Node, Op, Shape, and DataType definitions |
| `trace` | Operation tracing functionality with TracedValue and Tracer for graph construction |
| `optimize` | Optimization passes including constant folding, DCE, CSE, and algebraic simplification |
| `codegen` | JIT compiler and compiled function execution with interpreter fallback |
| `cache` | Function cache with LRU eviction and graph hashing |
| `error` | Error types and Result alias for JIT operations |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-jit = "0.1.0"
```

### Basic Tracing and Compilation

```rust
use axonml_jit::{trace, JitCompiler};

// Trace operations to build a computation graph
let graph = trace(|tracer| {
    let a = tracer.input("a", &[2, 3]);
    let b = tracer.input("b", &[2, 3]);
    let c = a.add(&b);
    let d = c.mul_scalar(2.0);
    tracer.output("result", d)
});

// Compile the graph
let compiler = JitCompiler::new();
let compiled = compiler.compile(&graph)?;

// Execute with real data
let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let b_data = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
let result = compiled.run(&[("a", &a_data), ("b", &b_data)])?;
```

### Traced Operations

```rust
use axonml_jit::trace;

let graph = trace(|tracer| {
    let x = tracer.input("x", &[4, 4]);

    // Unary operations
    let y = x.relu()
             .mul_scalar(2.0)
             .add_scalar(1.0);

    // Activation functions
    let z = y.sigmoid().tanh().gelu();

    // Reductions
    let mean = z.mean_axis(1, true);

    // Shape operations
    let reshaped = mean.reshape(&[-1]);

    tracer.output("output", reshaped)
});
```

### Custom Optimization

```rust
use axonml_jit::{Optimizer, OptimizationPass, JitCompiler};

// Create optimizer with custom passes
let mut optimizer = Optimizer::new();
optimizer.add_pass(OptimizationPass::ConstantFolding);
optimizer.add_pass(OptimizationPass::AlgebraicSimplification);
optimizer.add_pass(OptimizationPass::DeadCodeElimination);
optimizer.add_pass(OptimizationPass::CommonSubexpressionElimination);

// Apply optimizations
let optimized_graph = optimizer.optimize(graph);

// Compile optimized graph
let compiler = JitCompiler::with_optimizer(optimizer);
let compiled = compiler.compile(&graph)?;
```

### Cache Management

```rust
use axonml_jit::JitCompiler;

let compiler = JitCompiler::new();

// Compile multiple graphs
let _ = compiler.compile(&graph1)?;
let _ = compiler.compile(&graph2)?;

// Check cache statistics
let stats = compiler.cache_stats();
println!("Cached functions: {}", stats.entries);
println!("Cache utilization: {:.1}%", stats.utilization());

// Clear cache if needed
compiler.clear_cache();
```

## Supported Operations

### Binary Operations
- `add`, `sub`, `mul`, `div`, `pow`, `max`, `min`

### Unary Operations
- `neg`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`

### Activations
- `relu`, `sigmoid`, `gelu`, `silu`

### Scalar Operations
- `add_scalar`, `mul_scalar`

### Reductions
- `sum`, `mean`, `sum_axis`, `mean_axis`

### Shape Operations
- `reshape`, `transpose`, `squeeze`, `unsqueeze`

### Matrix Operations
- `matmul`

### Comparison Operations
- `gt`, `lt`, `eq`, `where`

## Optimization Passes

| Pass | Description |
|------|-------------|
| `ConstantFolding` | Evaluate constant expressions at compile time |
| `DeadCodeElimination` | Remove nodes that do not contribute to outputs |
| `AlgebraicSimplification` | Simplify expressions (x * 1 = x, x + 0 = x, etc.) |
| `CommonSubexpressionElimination` | Reuse identical subexpressions |
| `ElementwiseFusion` | Fuse consecutive elementwise operations |
| `StrengthReduction` | Replace expensive ops with cheaper equivalents |

## Tests

Run the test suite:

```bash
cargo test -p axonml-jit
```

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
