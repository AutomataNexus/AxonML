# axonml-core

Core abstractions for the [Axonml](https://github.com/AutomataNexus/AxonML) ML framework.

## Overview

`axonml-core` provides the foundational layer for the Axonml machine learning framework, including:

- **Device Abstraction** - CPU, CUDA, Vulkan, Metal, WebGPU backends
- **Storage Management** - Reference-counted memory with copy-on-write
- **Data Types** - f32, f64, i32, i64, f16, bf16, and more
- **Backend Traits** - Unified interface for compute backends

## Usage

```rust
use axonml_core::{Device, Storage, DType};

// Create storage on CPU
let storage = Storage::<f32>::zeros(1024, Device::Cpu);

// Check device capabilities
let device = Device::Cpu;
println!("Device: {:?}", device);
```

## Features

- `cuda` - Enable CUDA/cuBLAS backend
- `vulkan` - Enable Vulkan compute backend
- `metal` - Enable Metal backend (macOS)
- `wgpu` - Enable WebGPU backend

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework. For most use cases, depend on the main `axonml` crate instead.

## License

MIT OR Apache-2.0
