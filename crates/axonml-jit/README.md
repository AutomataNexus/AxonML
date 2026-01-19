# axonml-jit

[![Crates.io](https://img.shields.io/crates/v/axonml-jit.svg)](https://crates.io/crates/axonml-jit)
[![Docs.rs](https://docs.rs/axonml-jit/badge.svg)](https://docs.rs/axonml-jit)
[![Downloads](https://img.shields.io/crates/d/axonml-jit.svg)](https://crates.io/crates/axonml-jit)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Just-in-time compilation for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-jit` provides JIT compilation capabilities to optimize model execution at runtime. Traces computational graphs and generates optimized code for your specific hardware and input patterns.

## Features

### Tracing
- **Graph tracing** - Record operations during forward pass
- **Dynamic shapes** - Handle variable-sized inputs
- **Control flow** - Support for conditionals and loops

### Compilation
- **LLVM backend** - Native code generation
- **Cranelift backend** - Fast compilation, cross-platform
- **WASM target** - Compile models for web deployment

### Optimizations
- **Operator fusion** - Combine multiple ops into one
- **Memory planning** - Minimize allocations
- **Constant folding** - Evaluate constant expressions
- **Dead code elimination** - Remove unused operations

## Installation

```toml
[dependencies]
axonml-jit = "0.1"
```

## Usage

### Basic Tracing

```rust
use axonml_jit::{trace, JitModule};

let model = create_model();
let sample_input = randn(&[1, 784]);

// Trace the model
let traced = trace(&model, &sample_input)?;

// Run traced model (optimized)
let output = traced.forward(&input);
```

### JIT Compilation

```rust
use axonml_jit::{JitCompiler, Backend};

let compiler = JitCompiler::new(Backend::Cranelift);

// Compile traced module
let compiled = compiler.compile(&traced)?;

// Much faster execution
let output = compiled.forward(&input);

// Benchmark
let traced_time = benchmark(|| traced.forward(&input));
let compiled_time = benchmark(|| compiled.forward(&input));
println!("Speedup: {:.2}x", traced_time / compiled_time);
```

### Export to WASM

```rust
use axonml_jit::{JitCompiler, Backend};

let compiler = JitCompiler::new(Backend::Wasm);
let wasm_module = compiler.compile(&traced)?;

// Save as .wasm file
wasm_module.save("model.wasm")?;

// Use in browser with wasm-bindgen
```

### Dynamic Shapes

```rust
use axonml_jit::{trace_dynamic, DynamicDim};

// Mark batch dimension as dynamic
let traced = trace_dynamic(
    &model,
    &[(DynamicDim::Dynamic, 784)],  // (batch, features)
)?;

// Works with any batch size
let out1 = traced.forward(&randn(&[1, 784]));
let out32 = traced.forward(&randn(&[32, 784]));
let out128 = traced.forward(&randn(&[128, 784]));
```

### Optimization Levels

```rust
use axonml_jit::{JitCompiler, OptLevel};

let compiler = JitCompiler::new(Backend::LLVM)
    .opt_level(OptLevel::O3)       // Maximum optimization
    .enable_fusion(true)           // Operator fusion
    .enable_vectorization(true);   // SIMD vectorization

let compiled = compiler.compile(&traced)?;
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `trace(model, input)` | Trace model with sample input |
| `trace_dynamic(model, shape)` | Trace with dynamic dimensions |
| `script(model)` | Convert model to script (preserves control flow) |

### JitCompiler Methods

| Method | Description |
|--------|-------------|
| `new(backend)` | Create compiler with backend |
| `opt_level(level)` | Set optimization level |
| `compile(traced)` | Compile traced module |
| `enable_fusion(bool)` | Enable/disable operator fusion |

### Backend Options

| Backend | Description |
|---------|-------------|
| `LLVM` | Maximum performance, slower compile |
| `Cranelift` | Fast compile, good performance |
| `Wasm` | WebAssembly output |

## CLI Usage

```bash
# Trace and compile a model
axonml jit compile model.axonml -o model.jit

# Export to WASM
axonml jit compile model.axonml --target wasm -o model.wasm

# Benchmark JIT vs eager
axonml jit benchmark model.axonml --input sample.tensor
```

## Part of Axonml

```toml
[dependencies]
axonml = { version = "0.1", features = ["jit"] }
```

## License

MIT OR Apache-2.0
