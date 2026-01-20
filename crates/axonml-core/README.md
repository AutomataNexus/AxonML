# axonml-core

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

**axonml-core** is the foundational layer of the AxonML machine learning framework. It provides core abstractions for device management, memory storage, data types, and pluggable backend implementations that underpin all tensor operations across CPU and GPU devices.

## Features

- **Device Abstraction** - Unified interface for managing compute devices including CPU, CUDA, Vulkan, Metal, and WebGPU backends with seamless tensor transfer between devices.

- **Type-Safe Data Types** - Comprehensive type system supporting f16, f32, f64, i8, i16, i32, i64, u8, u32, u64, and bool with automatic type promotion rules.

- **Efficient Memory Storage** - Reference-counted storage with zero-copy slicing, automatic memory cleanup, and device-agnostic operations.

- **Pluggable Backend Architecture** - Extensible backend system with a common `Backend` trait enabling device-agnostic tensor operations.

- **Memory Allocator** - Flexible allocator trait with default CPU implementation and support for custom memory pools.

- **Device Capabilities** - Query device capabilities including memory, f16/f64 support, and compute capability for optimal resource utilization.

## Modules

| Module | Description |
|--------|-------------|
| `device` | Device abstraction (CPU, CUDA, Vulkan, Metal, WebGPU) with availability checking and capability queries |
| `dtype` | Data type definitions with `Scalar`, `Numeric`, and `Float` traits for type-safe operations |
| `storage` | Reference-counted memory storage with views, slicing, and device transfer |
| `allocator` | Memory allocation traits and default CPU allocator implementation |
| `backends` | Device-specific backend implementations for compute operations |
| `error` | Comprehensive error types for shape mismatches, device errors, and memory allocation failures |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-core = "0.1.0"
```

### Basic Example

```rust
use axonml_core::{Device, DType, Storage};

// Check device availability
let device = Device::Cpu;
assert!(device.is_available());

// Create storage on CPU
let storage = Storage::<f32>::zeros(1024, device);
assert_eq!(storage.len(), 1024);

// Create storage from data
let data = vec![1.0f32, 2.0, 3.0, 4.0];
let storage = Storage::from_vec(data, Device::Cpu);

// Create a view (zero-copy slice)
let view = storage.slice(1, 2).unwrap();
assert_eq!(view.len(), 2);
```

### Device Capabilities

```rust
use axonml_core::Device;

let device = Device::Cpu;
let caps = device.capabilities();

println!("Device: {}", caps.name);
println!("Total Memory: {} bytes", caps.total_memory);
println!("Supports f16: {}", caps.supports_f16);
println!("Supports f64: {}", caps.supports_f64);
```

### Data Types

```rust
use axonml_core::{DType, Scalar, Numeric, Float};

// Query dtype properties
assert!(DType::F32.is_float());
assert_eq!(DType::F32.size_of(), 4);

// Use type traits
fn process<T: Float>(data: &[T]) -> T {
    data.iter().fold(T::ZERO, |acc, &x| acc + x)
}
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-core
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
