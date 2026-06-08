# axonml-core

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/version-0.6.5-green.svg" alt="Version 0.6.5">
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

**axonml-core** is the foundational layer of the AxonML machine learning framework. It provides the `Device` abstraction, the `Scalar`/`Numeric`/`Float` trait hierarchy, reference-counted `Storage<T>` with pooled GPU allocations, and five compute backends (CPU, CUDA, Vulkan, Metal, WebGPU) that underpin every tensor operation in the framework.

## Features

- **Device Abstraction** - `Device` enum (Cpu, Cuda, Vulkan, Metal, Wgpu) with per-variant device index, runtime availability checks, and `best_available_backend()` selector (CUDA > Metal > Vulkan > WebGPU > CPU).

- **Type-Safe Data Types** - `DType` runtime enum covering F16, F32, F64, I8, I16, I32, I64, U8, U32, U64, Bool with `size_of` / `is_float` / `is_signed` / `is_integer` queries. Compile-time `Scalar` / `Numeric` / `Float` trait hierarchy for zero-cost generic dispatch.

- **Reference-Counted Storage** - `Storage<T>` wraps either a host `Vec<T>` or a `PooledCudaSlice` behind `Arc<RwLock<...>>`. Supports zero-copy views via offset+len slicing, `to_device()` for CPU<->GPU transfer, deep copy, and RAII `as_slice()` / `as_slice_mut()` guards.

- **Five Compute Backends** - CPU (rayon-parallel, matrixmultiply GEMM/GEMV, always available), CUDA (cuBLAS + 15+ custom PTX kernel modules), Vulkan (ash + gpu-allocator, SPIR-V compute), Metal (Apple Silicon, compute pipelines), WebGPU (wgpu for browser/cross-platform). As of 0.6.5 the CPU backend's matmul (all layouts incl. batched), reductions, and the SwiGLU/RMSNorm/RoPE math are rayon-parallel above a work threshold, so single-node CPU inference and training fallbacks scale across cores.

- **GPU Memory Pool** - `cuda_pool` returns freed CUDA allocations to a size-bucketed free list instead of calling `cudaFree`, amortising allocator cost across training steps.

- **Device Capabilities** - `DeviceCapabilities` exposes name, total/available memory, f16/f64 support, max threads per block, and CUDA compute capability.

- **Allocator Trait** - `Allocator` extension point with a `DefaultAllocator` that performs 64-byte-aligned host allocations and reports system memory via sysinfo.

## Modules

| Module | Description |
|--------|-------------|
| `device` | `Device` enum (Cpu, Cuda, Vulkan, Metal, Wgpu) + `DeviceCapabilities` with availability and capability queries |
| `dtype` | `DType` runtime enum and `Scalar` / `Numeric` / `Float` trait hierarchy; `F16Wrapper` and `BoolWrapper` adapters |
| `storage` | Reference-counted `Storage<T>` with zero-copy views, device transfer, and pooled GPU slices |
| `allocator` | `Allocator` trait and `DefaultAllocator` (64-byte-aligned CPU allocator) |
| `backends` | `Backend` trait, `BackendType`, `GpuMemory`, `GpuStream`, plus CPU/CUDA/Vulkan/Metal/WGPU implementations |
| `error` | `Error` / `Result` types for shape mismatches, device errors, and allocation failures |

### Backends (under `backends/`)

| Backend | File | Status |
|---------|------|--------|
| CPU | `cpu.rs` | Always compiled; rayon-parallel ops, matrixmultiply GEMM |
| CUDA | `cuda.rs` + `cuda_kernels/` + `cuda_pool.rs` | Feature `cuda`; cuBLAS + PTX kernels for elementwise, activations, attention, Q4_K/Q6_K dequant-in-shader matmul, softmax, layernorm, RMSNorm, transpose, embedding gather |
| cuDNN | `cudnn_ops.rs` | Feature `cudnn`; conv2d forward/backward via cuDNN |
| Vulkan | `vulkan.rs` | Feature `vulkan`; ash + gpu-allocator, full buffer/pipeline/dispatch (~982 lines) |
| Metal | `metal.rs` | Feature `metal`; full buffer/pipeline/dispatch on Apple Silicon (~769 lines) |
| WebGPU | `wgpu_backend.rs` | Feature `wgpu`; full buffer/pipeline/dispatch via wgpu (~1710 lines) |

## Cargo Features

| Feature | Pulls In | Purpose |
|---------|----------|---------|
| `std` (default) | — | Standard library support |
| `cuda` | `cudarc` | NVIDIA CUDA backend |
| `cudnn` | `cuda` + cudarc cuDNN | cuDNN conv ops |
| `vulkan` | `ash`, `gpu-allocator` | Vulkan compute backend |
| `metal` | `metal`, `objc` (macOS only) | Apple Metal backend |
| `wgpu` | `wgpu`, `pollster` | WebGPU / cross-platform backend |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-core = "0.6.5"
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

### Picking a Backend

```rust
use axonml_core::backends::{best_available_backend, gpu_count, BackendType};

let backend = best_available_backend();
match backend {
    BackendType::Cpu => println!("Falling back to CPU"),
    _ => println!("Using {} GPU(s)", gpu_count()),
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

---

_Last updated: 2026-06-06 (v0.6.5)_
