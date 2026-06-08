---
layout: default
title: Device
parent: Core
nav_order: 1
description: "Device enum, availability, capabilities"
---

# Device Module

Device abstraction for compute target specification (`axonml_core::device`).

## Overview

`Device` is a `Copy + Eq + Hash` enum that names where a tensor lives. The backend implementations for each variant are in `axonml_core::backends::{cpu, cuda, vulkan, metal, wgpu_backend}`. All four GPU backends are full implementations (CUDA: 3,706 lines, Vulkan: 982 lines, Metal: 769 lines, WebGPU: 1,710 lines), not stubs.

## Device Enum

```rust
use axonml_core::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU (always available; the default).
    #[default]
    Cpu,

    /// NVIDIA CUDA GPU by index.
    #[cfg(feature = "cuda")]
    Cuda(usize),

    /// Vulkan GPU by index (cross-platform).
    #[cfg(feature = "vulkan")]
    Vulkan(usize),

    /// Apple Metal GPU by index.
    #[cfg(feature = "metal")]
    Metal(usize),

    /// WebGPU device by index (WASM / browser).
    #[cfg(feature = "wgpu")]
    Wgpu(usize),
}
```

Each GPU variant is gated behind a Cargo feature, so if a feature is off the variant doesn't exist.

## Methods

```rust
use axonml_core::Device;

// Construction
let cpu = Device::cpu();                 // const
#[cfg(feature = "cuda")]
let gpu = Device::cuda(0);               // const; Cuda(0)

// Queries (const where possible)
assert!(Device::Cpu.is_cpu());
assert!(!Device::Cpu.is_gpu());
assert_eq!(Device::Cpu.index(), 0);       // GPU variants return their device index
assert_eq!(Device::Cpu.device_type(), "cpu");
assert!(Device::Cpu.is_available());      // walks the backend's is_device_available

// Display
assert_eq!(format!("{}", Device::Cpu), "cpu");
// format!("{}", Device::Cuda(1)) -> "cuda:1"
```

### Best-available Selection

```rust
use axonml_core::device::best_available_backend;

// Preference order: CUDA > Metal > Vulkan > WebGPU > CPU
let dev = best_available_backend();
```

### Device Counts

```rust
#[cfg(feature = "cuda")]
use axonml_core::device::cuda_device_count;
#[cfg(feature = "vulkan")]
use axonml_core::device::vulkan_device_count;
```

## Capabilities

```rust
use axonml_core::device::{Device, DeviceCapabilities};

let caps: DeviceCapabilities = Device::Cpu.capabilities();
println!("Name: {}", caps.name);
println!("Total mem: {} bytes", caps.total_memory);
println!("Available mem: {} bytes", caps.available_memory);
println!("Supports f16: {}", caps.supports_f16);
println!("Supports f64: {}", caps.supports_f64);
println!("Max threads/block: {}", caps.max_threads_per_block);
println!("Compute capability: {:?}", caps.compute_capability); // Option<(major, minor)> for CUDA
```

`DeviceCapabilities::supports_f32` is `const fn` and always `true` — every backend handles f32.

## Usage

### Transfer Tensors Between Devices

```rust
use axonml_core::Device;
use axonml_tensor::Tensor;

let cpu_t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

#[cfg(feature = "cuda")]
let gpu_t = cpu_t.to_device(Device::Cuda(0)).unwrap();

// `cpu()` is shorthand for `to_device(Device::Cpu)`
let back = gpu_t.cpu().unwrap();
```

### Checking Where a Tensor Lives

```rust
match tensor.device() {
    Device::Cpu => { /* ... */ }
    #[cfg(feature = "cuda")]
    Device::Cuda(idx) => { /* ... */ }
    _ => {}
}
```

## Backend Notes

- **CPU** — OpenBLAS when available, native Rust fallback, SIMD where possible.
- **CUDA** — `axonml-core/src/backends/cuda.rs` (3,706 lines): cuBLAS matmul, custom PTX kernels (Q4_K / Q6_K dequant-in-shader GEMV, fused flash-decode attention, fused prefill attention — used by `nexus-serve`). Requires CUDA toolkit; compute capability 5.2+ (PTX targets).
- **Vulkan** — `vulkan.rs` (982 lines): compute shaders, cross-platform.
- **Metal** — `metal.rs` (769 lines): MSL kernels, Apple Silicon.
- **WebGPU** — `wgpu_backend.rs` (1,710 lines): WGSL kernels via the `wgpu` crate; works in WASM and on native via Vulkan / Metal / D3D12.

## Best Practices

1. Default to CPU for development; move to GPU for throughput.
2. Move **both** model parameters and inputs to the same device; `Error::DeviceMismatch` is the most common GPU training error.
3. Check `.is_available()` before allocating if the device is optional.
4. Batch transfers — each host↔device copy has a per-call overhead.

---

*Last updated: 2026-06-06 (v0.6.5)*
