# axonml-core Documentation

> Core abstractions for the Axonml ML framework.

## Overview

`axonml-core` provides the foundational types and abstractions that all other Axonml crates build upon. It handles device management, memory storage, data types, and backend implementations.

## Modules

### device.rs

Defines the `Device` enum for representing compute devices.

```rust
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")] Cuda(usize),
    #[cfg(feature = "vulkan")] Vulkan(usize),
    // ...
}
```

**Key functions:**
- `Device::is_available()` - Check if device is available
- `Device::is_cpu()` / `Device::is_gpu()` - Device type checks
- `Device::capabilities()` - Get device capabilities

### dtype.rs

Defines the type system for tensor elements.

**Types:**
- `DType` - Runtime type representation
- `Scalar` - Base trait for tensor element types
- `Numeric` - Trait for types supporting arithmetic
- `Float` - Trait for floating point types

**Supported DTypes:**
- F16, F32, F64 (floating point)
- I8, I16, I32, I64 (signed integers)
- U8 (unsigned integer)
- Bool (boolean)

### storage.rs

Reference-counted memory storage for tensors.

```rust
pub struct Storage<T: Scalar> {
    // Reference-counted inner storage
}
```

**Key methods:**
- `Storage::zeros(len, device)` - Create zeroed storage
- `Storage::from_vec(vec, device)` - Create from vector
- `Storage::as_slice()` / `Storage::as_slice_mut()` - Access data
- `Storage::slice(offset, len)` - Create view
- `Storage::to_device(device)` - Transfer to device

### error.rs

Comprehensive error types for Axonml operations.

**Error variants:**
- `ShapeMismatch` - Incompatible tensor shapes
- `DTypeMismatch` - Incompatible data types
- `DeviceMismatch` - Tensors on different devices
- `InvalidDimension` - Invalid dimension index
- `IndexOutOfBounds` - Index exceeds size
- `AllocationFailed` - Memory allocation failure
- `BroadcastError` - Incompatible broadcast shapes

### allocator.rs

Memory allocator traits and implementations.

**Traits:**
- `Allocator` - Device memory allocator trait

**Implementations:**
- `DefaultAllocator` - CPU allocator using system memory

### backends/

Backend-specific implementations.

**Available:**
- `cpu.rs` - CPU backend with basic operations

**Planned:**
- `cuda.rs` - NVIDIA CUDA backend
- `vulkan.rs` - Vulkan compute backend
- `metal.rs` - Apple Metal backend
- `wgpu.rs` - WebGPU backend

## Usage

```rust
use axonml_core::prelude::*;

// Device management
let device = Device::Cpu;
assert!(device.is_available());

// Create storage
let storage = Storage::<f32>::zeros(100, device);

// Access data
{
    let data = storage.as_slice();
    println!("First element: {}", data[0]);
}
```

## Feature Flags

- `std` (default) - Enable standard library
- `cuda` - Enable CUDA backend
- `vulkan` - Enable Vulkan backend
- `metal` - Enable Metal backend
- `wgpu` - Enable WebGPU backend
