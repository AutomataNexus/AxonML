# axonml-jit

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-jit"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Version: 0.6.1"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-jit` is the AxonML tracing JIT. A `Tracer` records tensor operations
into a typed `Graph` IR; an `Optimizer` runs a stack of passes (constant
folding, DCE, CSE, algebraic simplification, elementwise fusion, strength
reduction); a `JitCompiler` emits either an interpreter-backed
`CompiledFunction` or a native Cranelift-compiled one; and a graph-hash
`FunctionCache` (LRU) lets repeated compilations hit cache. Cranelift
0.111 is the code-gen backend.

## Features

- **Operation Tracing** — `trace(|tracer| { ... })` and `Tracer` APIs build a `Graph` from recorded operations using thread-local state
- **Typed IR** — `Graph`, `Node`, `NodeId`, `Op` (40+ variants), `Shape` (with broadcast checks and broadcast-shape computation), `DataType`
- **Optimizer** — `Optimizer::default_passes()` plus six `OptimizationPass` variants (`ConstantFolding`, `DeadCodeElimination`, `CommonSubexpressionElimination`, `AlgebraicSimplification`, `ElementwiseFusion`, `StrengthReduction`)
- **JIT Compiler** — `JitCompiler` with interpreter execution and optional Cranelift native codegen (`enable_native(true)`)
- **Higher-Level Facade** — `compile_fn`, `compile_fn_with_config`, `compile_graph`, `compile_graph_with_config`, `CompiledModel`, `LazyCompiled` (deferred compilation) with `CompileConfig` (`Mode::{Default, ReduceOverhead, MaxAutotune}`, `Backend::{Default, Eager, AOT, ONNX}`, `fullgraph`, `dynamic`, `disable`, custom passes)
- **Function Caching** — `FunctionCache` with LRU eviction and `Self::hash_graph`-based keying; `CacheStats` with `utilization`
- **Shape Inference** — automatic shape propagation including broadcast semantics (`Shape::broadcast_shape`)
- **Thread-Local Tracing** — safe concurrent tracing via per-thread tracer state

## Modules

| Module | Description |
|--------|-------------|
| `ir` | `Graph`, `Node`, `NodeId`, `Op`, `Shape`, `DataType`, topological order, validation |
| `trace` | `Tracer`, `TracedValue`, `trace` entry point, thread-local state |
| `optimize` | `Optimizer`, `OptimizationPass` (6 variants), `default_passes` |
| `codegen` | `JitCompiler`, `CompiledFunction` (Interpreted + Cranelift Native kinds) |
| `compile` | `compile_fn`, `compile_graph`, `CompiledModel`, `LazyCompiled`, `CompileConfig`, `CompileStats`, `Mode`, `Backend` |
| `cache` | `FunctionCache` (LRU), `CacheStats` |
| `error` | `JitError`, `JitResult` |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-jit = "0.6.1"
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

// Compile the graph (interpreter-backed by default)
let compiler = JitCompiler::new();
let compiled = compiler.compile(&graph)?;

// Execute with real data — inputs are name/slice tuples
let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let b_data = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
let result = compiled.run(&[("a", &a_data[..]), ("b", &b_data[..])])?;
```

### Cranelift Native Codegen

```rust
use axonml_jit::JitCompiler;

let mut compiler = JitCompiler::new();
compiler.enable_native(true);  // opt in to Cranelift codegen
let compiled = compiler.compile(&graph)?;
```

### Traced Operations

```rust
use axonml_jit::trace;

let graph = trace(|tracer| {
    let x = tracer.input("x", &[4, 4]);

    // Elementwise + scalar ops
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

// Build a custom pass pipeline
let mut optimizer = Optimizer::new();
optimizer.add_pass(OptimizationPass::ConstantFolding);
optimizer.add_pass(OptimizationPass::AlgebraicSimplification);
optimizer.add_pass(OptimizationPass::DeadCodeElimination);
optimizer.add_pass(OptimizationPass::CommonSubexpressionElimination);

// Apply optimizations directly
let optimized_graph = optimizer.optimize(graph.clone());

// Or hand the optimizer to the compiler
let compiler = JitCompiler::with_optimizer(optimizer);
let compiled = compiler.compile(&graph)?;
```

### Higher-Level `compile_fn` / `CompiledModel`

```rust
use axonml_jit::{compile_fn, compile_fn_with_config, CompileConfig, Mode, Backend};
use std::collections::HashMap;

// Zero-config
let model = compile_fn(|t| {
    let x = t.input("x", &[8]);
    t.output("y", x.relu())
})?;

// With config
let cfg = CompileConfig::new()
    .mode(Mode::MaxAutotune)
    .backend(Backend::Default)
    .fullgraph(true);

let model = compile_fn_with_config(|t| {
    let x = t.input("x", &[8]);
    t.output("y", x.gelu())
}, cfg)?;

// CompiledModel runs on HashMap<String, Vec<f32>>
let mut inputs = HashMap::new();
inputs.insert("x".to_string(), vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let outputs = model.run(&inputs)?;

// Inspect compilation
println!("{} -> {} nodes ({:.1}% reduction)",
         model.stats().original_nodes,
         model.stats().optimized_nodes,
         model.stats().optimization_ratio() * 100.0);
```

### Lazy Compilation

```rust
use axonml_jit::LazyCompiled;

let lazy = LazyCompiled::new(|t| {
    let x = t.input("x", &[3]);
    t.output("y", x.exp())
});
// Compiled on first call, cached thereafter.
let outputs = lazy.run(&inputs)?;
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
println!("Cache utilization: {:.1}%", stats.utilization() * 100.0);

// Clear cache if needed
compiler.clear_cache();
```

## Supported Operations

All recorded via `Op` enum variants:

### Binary Operations
- `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Max`, `Min`

### Unary Operations
- `Neg`, `Abs`, `Sqrt`, `Exp`, `Log`, `Sin`, `Cos`, `Tanh`

### Activations
- `Relu`, `Sigmoid`, `Gelu`, `Silu`

### Scalar Operations
- `AddScalar`, `MulScalar`

### Reductions
- `Sum`, `SumAxis`, `Mean`, `MeanAxis`, `MaxAxis`

### Shape Operations
- `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Broadcast`

### Matrix Operations
- `MatMul`

### Comparison / Conditional
- `Gt`, `Lt`, `Eq`, `Where`

### Special
- `Cast` (change `DataType`), `Contiguous`, `Input`, `Output`, `Constant`

## Optimization Passes

| Pass | Description |
|------|-------------|
| `ConstantFolding` | Evaluate constant expressions at compile time |
| `DeadCodeElimination` | Remove nodes that don't feed an output |
| `CommonSubexpressionElimination` | Reuse identical subexpressions |
| `AlgebraicSimplification` | `x * 1 = x`, `x + 0 = x`, etc. |
| `ElementwiseFusion` | Fuse consecutive elementwise ops |
| `StrengthReduction` | Replace expensive ops with cheaper equivalents |

Default `CompileConfig` enables `ConstantFolding`, `DeadCodeElimination`, and
`CommonSubexpressionElimination`. `Mode::MaxAutotune` additionally appends
`ElementwiseFusion` and `AlgebraicSimplification`.

## Tests

```bash
cargo test -p axonml-jit
```

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
