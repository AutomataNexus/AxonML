# Device Module

Device abstraction for compute target specification.

## Overview

The device module provides a unified interface for specifying where computations should run. This abstraction allows code to be written once and executed on different hardware backends.

## Device Enum

```rust
pub enum Device {
    /// CPU computation (default)
    Cpu,

    /// CUDA GPU computation
    Cuda(usize),  // device index

    /// Vulkan GPU computation (cross-platform)
    Vulkan(usize),

    /// Metal GPU computation (Apple platforms)
    Metal(usize),

    /// WebGPU computation (web/cross-platform)
    WebGpu(usize),
}
```

## Usage Examples

### Basic Device Selection

```rust
use axonml::prelude::*;

// Create tensor on CPU (default)
let cpu_tensor = Tensor::zeros(&[2, 3]);

// Specify CPU explicitly
let cpu_explicit = Tensor::zeros_on(&[2, 3], Device::Cpu);

// Create on CUDA GPU 0
let cuda_tensor = Tensor::zeros_on(&[2, 3], Device::Cuda(0));
```

### Device Transfer

```rust
use axonml::prelude::*;

let cpu_tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

// Transfer to GPU
let gpu_tensor = cpu_tensor.to(Device::Cuda(0));

// Transfer back to CPU
let back_to_cpu = gpu_tensor.to(Device::Cpu);
```

### Checking Device

```rust
use axonml::prelude::*;

let tensor = Tensor::zeros(&[2, 3]);

match tensor.device() {
    Device::Cpu => println!("On CPU"),
    Device::Cuda(idx) => println!("On CUDA GPU {}", idx),
    _ => println!("Other device"),
}
```

## Implementation Details

### CPU Backend

The CPU backend is always available and uses:
- OpenBLAS for matrix operations (when available)
- Native Rust implementations as fallback
- SIMD optimizations where possible

### CUDA Backend (Feature-gated)

Enabled with the `cuda` feature:
- Requires CUDA toolkit installed
- Uses cuBLAS for matrix operations
- Supports compute capability 3.5+

### Vulkan Backend (Feature-gated)

Enabled with the `vulkan` feature:
- Cross-platform GPU compute
- Uses compute shaders
- Works on Windows, Linux, macOS (via MoltenVK)

## Best Practices

1. **Default to CPU** - Start development on CPU for easier debugging
2. **Batch transfers** - Minimize CPU-GPU transfers by batching operations
3. **Check availability** - Use `Device::is_available()` before using GPU backends
4. **Consistent devices** - Keep tensors in an operation on the same device

## Related

- [Storage](storage.md) - Memory storage tied to devices
- [Tensor](../tensor/tensor.md) - Tensors with device awareness

@version 0.1.0
@author AutomataNexus Development Team
