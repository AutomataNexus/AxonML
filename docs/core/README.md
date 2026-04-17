# axonml-core Documentation

> Core abstractions for the AxonML ML framework.

## Overview

`axonml-core` is the foundation layer every other AxonML crate builds on. It
provides device abstractions with runtime capability queries, the
`Scalar`/`Numeric`/`Float` trait hierarchy for generic tensor element
dispatch, reference-counted `Storage<T>` with pooled GPU allocations, and the
five compute backends (CPU, CUDA, Vulkan, Metal, WebGPU).

## Modules

### `device`

Defines the `Device` enum and `DeviceCapabilities`.

```rust
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]   Cuda(usize),
    #[cfg(feature = "vulkan")] Vulkan(usize),
    #[cfg(feature = "metal")]  Metal(usize),
    #[cfg(feature = "wgpu")]   Wgpu(usize),
}
```

**Key methods:**
- `Device::is_available()` — runtime check via backend `is_device_available`
- `Device::is_cpu()` / `Device::is_gpu()` — device-type predicates
- `Device::index()` — GPU device index (0 for CPU)
- `Device::capabilities()` — returns `DeviceCapabilities` (memory, f16/f64
  support, compute capability)

The `best_available_backend()` selector prefers CUDA > Metal > Vulkan >
WebGPU > CPU.

### `dtype`

Runtime and compile-time type system for tensor elements.

**`DType` enum:** F16, F32, F64, I8, I16, I32, I64, U8, U32, U64, Bool.
Exposes `size_of()`, `is_float()`, `is_int()`, name/category queries.

**Trait hierarchy (compile-time, zero-cost generic dispatch):**
- `Scalar` — any storable element (`Pod + Zeroable + Send + Sync + 'static`)
- `Numeric: Scalar` — adds arithmetic (`Num + NumCast + Zero + One + PartialOrd`)
- `Float: Numeric` — adds `exp`/`ln`/`pow`/`sqrt`/`sin`/`cos`/`tanh`,
  NaN/Inf constants, epsilon

Implementations cover `f32`, `f64`, `i8..i64`, `u8`/`u32`/`u64`, plus
wrapper types `F16Wrapper` (bytemuck-safe `half::f16`) and `BoolWrapper`
(u8-backed).

### `storage`

Reference-counted raw memory management.

```rust
pub struct Storage<T: Scalar> { /* Arc<RwLock<StorageInner<T>>> */ }
```

CPU storage is a `Vec<T>`; GPU storage is a `PooledCudaSlice` wrapping
`cudarc::driver::CudaSlice<f32>` that returns memory to the size-bucketed
pool in `cuda_pool` on drop instead of calling `cudaFree`.

**Key methods:**
- `Storage::zeros(len, device)` — zero-initialised storage
- `Storage::from_vec(vec, device)` — from an existing `Vec<T>`
- `Storage::as_slice()` / `as_slice_mut()` — RAII guard accessors
- `Storage::slice(offset, len)` — zero-copy view
- `Storage::to_device(device)` — CPU<->GPU transfer
- `Storage::as_cuda_slice()` — direct GPU kernel access (f32-only)

### `error`

The `Error` enum + `Result<T>` alias used throughout the framework.

**Variants:**
`ShapeMismatch`, `DTypeMismatch`, `DeviceMismatch`, `InvalidDimension`,
`IndexOutOfBounds`, `AllocationFailed`, `DeviceNotAvailable`,
`InvalidOperation`, `BroadcastError`, `EmptyTensor`, `NotContiguous`,
`GradientError`, `SerializationError`, `InternalError`.

### `allocator`

`Allocator` trait plus the `DefaultAllocator` CPU implementation. Backs
host-side allocations with 64-byte-aligned system memory and exposes
`allocate<T>`, `deallocate<T>`, `copy<T>`, `zero<T>`, plus `total_memory()`
and `free_memory()` via `sysinfo`.

### `backends`

Compute backends and the shared `Backend` trait.

| Module           | Feature    | Notes                                                       |
|------------------|------------|-------------------------------------------------------------|
| `cpu`            | always on  | rayon-parallel GEMM/GEMV via `matrixmultiply`               |
| `cuda`           | `cuda`     | cuBLAS + custom PTX kernels via `cudarc`                    |
| `cuda_kernels`   | `cuda`     | PTX module registry (activations, attention, Q4_K/Q6_K, LSTM, pooling, elementwise) |
| `cuda_pool`      | always on  | Size-bucketed GPU memory pool                               |
| `cudnn_ops`      | `cudnn`    | cuDNN conv2d bindings                                       |
| `vulkan`         | `vulkan`   | `ash` + `gpu-allocator`, SPIR-V compute                     |
| `metal`          | `metal`    | Apple Metal compute pipelines via `objc`                    |
| `wgpu_backend`   | `wgpu`     | `wgpu` for browser / cross-platform WebGPU                  |
| `gpu_tests`      | always on  | Backend correctness infrastructure                          |

Top-level dispatch helpers in `backends::mod`:
- `Backend` trait — `allocate`, `deallocate`, `copy_to_device`,
  `copy_to_host`, `copy_device_to_device`, `synchronize`, `capabilities`
- `GpuMemory` — pointer + size + device index + `BackendType`
- `GpuStream` — per-backend stream handle with `synchronize()`
- `best_available_backend()`, `gpu_count()`

## Usage

```rust
use axonml_core::prelude::*;

let device = Device::Cpu;
assert!(device.is_available());

let storage = Storage::<f32>::zeros(100, device);
{
    let data = storage.as_slice();
    println!("First element: {}", data[0]);
}
```

## Feature Flags

- `std` (default) — standard library
- `cuda` — NVIDIA CUDA backend (cuBLAS + PTX kernels)
- `cudnn` — cuDNN ops (implies `cuda`)
- `vulkan` — Vulkan compute backend
- `metal` — Apple Metal backend (macOS only)
- `wgpu` — WebGPU backend

## Last updated

0.6.1 (2026-04-16)
