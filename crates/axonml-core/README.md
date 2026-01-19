# axonml-core

[![Crates.io](https://img.shields.io/crates/v/axonml-core.svg)](https://crates.io/crates/axonml-core)
[![Docs.rs](https://docs.rs/axonml-core/badge.svg)](https://docs.rs/axonml-core)
[![Downloads](https://img.shields.io/crates/d/axonml-core.svg)](https://crates.io/crates/axonml-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Core foundation layer for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-core` provides the foundational abstractions that underpin the entire Axonml machine learning framework. It handles device management, memory storage, data types, error handling, and backend implementations. This is the lowest-level crate in the Axonml stack - all other crates build on top of it.

## Features

### Device Abstraction
- **Multi-backend support** - CPU, CUDA, Vulkan, Metal, WebGPU
- **Unified API** - Same code works across all devices
- **Device queries** - Check capabilities, memory, compute units
- **Device transfer** - Move data between devices seamlessly

### Storage Management
- **Reference-counted memory** - Efficient sharing without copies
- **Copy-on-write semantics** - Automatic optimization for memory efficiency
- **Aligned allocations** - SIMD-friendly memory layout
- **Custom allocators** - Pluggable allocation strategies

### Data Types
- **Floating point** - f32, f64, f16 (half), bf16 (bfloat16)
- **Integer** - i8, i16, i32, i64, u8, u16, u32, u64
- **Boolean** - bool for masks and conditions
- **Type traits** - Scalar, Numeric, Float for generic programming

### Error Handling
- **Typed errors** - ShapeError, DeviceError, AllocationError, etc.
- **Result type** - Consistent error propagation
- **Descriptive messages** - Clear error diagnostics

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-core = "0.1"
```

Or with specific features:

```toml
[dependencies]
axonml-core = { version = "0.1", features = ["cuda", "vulkan"] }
```

## Usage

### Basic Device Operations

```rust
use axonml_core::{Device, DType};

// Default CPU device
let cpu = Device::Cpu;
println!("Using device: {:?}", cpu);

// Check if CUDA is available (when feature enabled)
#[cfg(feature = "cuda")]
{
    let cuda = Device::Cuda(0);
    println!("CUDA device 0: {:?}", cuda);
}
```

### Storage Creation and Management

```rust
use axonml_core::{Device, Storage};

// Create zeroed storage
let zeros = Storage::<f32>::zeros(1024, Device::Cpu);
println!("Created {} elements", zeros.len());

// Create storage from data
let data = vec![1.0f32, 2.0, 3.0, 4.0];
let storage = Storage::from_vec(data, Device::Cpu);

// Access data
let slice = storage.as_slice();
println!("First element: {}", slice[0]);

// Mutable access (copy-on-write if shared)
let mut storage = storage;
let slice_mut = storage.as_slice_mut();
slice_mut[0] = 42.0;
```

### Data Type System

```rust
use axonml_core::{DType, Scalar, Numeric, Float};

// Check data type properties
let dtype = DType::F32;
println!("Size: {} bytes", dtype.size_of());
println!("Is float: {}", dtype.is_float());
println!("Is signed: {}", dtype.is_signed());

// Generic programming with type traits
fn process_numeric<T: Numeric>(values: &[T]) -> T {
    values.iter().fold(T::zero(), |acc, &x| acc + x)
}

fn process_float<T: Float>(values: &[T]) -> T {
    let sum: T = values.iter().fold(T::zero(), |acc, &x| acc + x);
    sum / T::from_usize(values.len())
}
```

### Custom Allocators

```rust
use axonml_core::{Allocator, DefaultAllocator, Device};

// Use the default allocator
let allocator = DefaultAllocator::new();

// Allocate aligned memory
let layout = std::alloc::Layout::from_size_align(1024, 64).unwrap();
let ptr = allocator.allocate(layout, Device::Cpu).unwrap();

// ... use memory ...

// Deallocate when done
unsafe { allocator.deallocate(ptr, layout, Device::Cpu); }
```

### Error Handling

```rust
use axonml_core::{Error, Result, Device, Storage};

fn create_large_storage() -> Result<Storage<f32>> {
    // This might fail if not enough memory
    let storage = Storage::<f32>::zeros(1_000_000_000, Device::Cpu);
    Ok(storage)
}

match create_large_storage() {
    Ok(storage) => println!("Created storage with {} elements", storage.len()),
    Err(Error::Allocation(msg)) => eprintln!("Allocation failed: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## API Reference

### Modules

| Module | Description |
|--------|-------------|
| `device` | Device enumeration and capabilities |
| `storage` | Memory storage with reference counting |
| `dtype` | Data types and numeric traits |
| `error` | Error types and Result alias |
| `allocator` | Memory allocation traits and implementations |
| `backends` | Backend-specific implementations |

### Key Types

| Type | Description |
|------|-------------|
| `Device` | Enum representing compute devices (Cpu, Cuda, Vulkan, etc.) |
| `Storage<T>` | Reference-counted memory storage for type T |
| `DType` | Enum representing data types (F32, F64, I32, etc.) |
| `Error` | Error type for all core operations |
| `Result<T>` | Type alias for `std::result::Result<T, Error>` |
| `Allocator` | Trait for custom memory allocators |

### Traits

| Trait | Description |
|-------|-------------|
| `Scalar` | Base trait for all scalar types |
| `Numeric` | Trait for numeric types (supports arithmetic) |
| `Float` | Trait for floating-point types |

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Enable standard library support | Yes |
| `cuda` | Enable CUDA/cuBLAS backend | No |
| `vulkan` | Enable Vulkan compute backend | No |
| `metal` | Enable Metal backend (macOS/iOS) | No |
| `wgpu` | Enable WebGPU backend | No |

## Architecture

```
axonml-core
├── device.rs      # Device enum and capabilities
├── storage.rs     # Reference-counted memory storage
├── dtype.rs       # Data types and numeric traits
├── error.rs       # Error types
├── allocator.rs   # Memory allocation
└── backends/
    ├── cpu.rs     # CPU backend implementation
    ├── cuda.rs    # CUDA backend (feature-gated)
    └── vulkan.rs  # Vulkan backend (feature-gated)
```

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) machine learning framework. The dependency hierarchy is:

```
axonml-core (this crate)
    └── axonml-tensor
        └── axonml-autograd
            └── axonml-nn
                └── axonml (umbrella)
```

For most use cases, depend on the main `axonml` crate which re-exports everything:

```toml
[dependencies]
axonml = "0.1"
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
