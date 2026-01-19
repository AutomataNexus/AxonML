//! Device Abstraction - Hardware Backend Management
//!
//! Provides a unified interface for managing compute devices including CPU,
//! CUDA GPUs, Vulkan, Metal, and WebGPU backends. Tensors can be moved between
//! devices transparently.
//!
//! # Key Features
//! - Unified device abstraction across backends
//! - Device availability checking
//! - Device capability queries
//! - Seamless tensor transfer between devices
//!
//! # Example
//! ```rust
//! use axonml_core::Device;
//!
//! let cpu = Device::Cpu;
//! assert!(cpu.is_available());
//! assert!(cpu.is_cpu());
//!
//! // Use default device (CPU)
//! let device = Device::default();
//! assert_eq!(device, Device::Cpu);
//! ```
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use core::fmt;
use sysinfo::System;

// =============================================================================
// Device Enum
// =============================================================================

/// Represents a compute device where tensors can be allocated and operations executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device (always available).
    Cpu,

    /// NVIDIA CUDA GPU device with device index.
    #[cfg(feature = "cuda")]
    Cuda(usize),

    /// Vulkan GPU device with device index (cross-platform).
    #[cfg(feature = "vulkan")]
    Vulkan(usize),

    /// Apple Metal GPU device with device index.
    #[cfg(feature = "metal")]
    Metal(usize),

    /// WebGPU device with device index (for WASM/browser).
    #[cfg(feature = "wgpu")]
    Wgpu(usize),
}

impl Device {
    /// Returns true if this device is available on the current system.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Cpu => true,
            #[cfg(feature = "cuda")]
            Self::Cuda(idx) => crate::backends::cuda::is_device_available(idx),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(idx) => crate::backends::vulkan::is_device_available(idx),
            #[cfg(feature = "metal")]
            Self::Metal(idx) => crate::backends::metal::is_device_available(idx),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(idx) => crate::backends::wgpu::is_device_available(idx),
        }
    }

    /// Returns true if this is a CPU device.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns true if this is a GPU device.
    #[must_use]
    pub const fn is_gpu(self) -> bool {
        !self.is_cpu()
    }

    /// Returns the device index for GPU devices, or 0 for CPU.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Cpu => 0,
            #[cfg(feature = "cuda")]
            Self::Cuda(idx) => idx,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(idx) => idx,
            #[cfg(feature = "metal")]
            Self::Metal(idx) => idx,
            #[cfg(feature = "wgpu")]
            Self::Wgpu(idx) => idx,
        }
    }

    /// Returns the name of this device type.
    #[must_use]
    pub const fn device_type(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => "vulkan",
            #[cfg(feature = "metal")]
            Self::Metal(_) => "metal",
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => "wgpu",
        }
    }

    /// Returns the default CPU device.
    #[must_use]
    pub const fn cpu() -> Self {
        Self::Cpu
    }

    /// Returns a CUDA device with the given index.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub const fn cuda(index: usize) -> Self {
        Self::Cuda(index)
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Self::Cuda(idx) => write!(f, "cuda:{idx}"),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(idx) => write!(f, "vulkan:{idx}"),
            #[cfg(feature = "metal")]
            Self::Metal(idx) => write!(f, "metal:{idx}"),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(idx) => write!(f, "wgpu:{idx}"),
        }
    }
}

// =============================================================================
// Device Capabilities
// =============================================================================

/// Information about a device's capabilities.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Name of the device.
    pub name: String,
    /// Total memory in bytes.
    pub total_memory: usize,
    /// Available memory in bytes.
    pub available_memory: usize,
    /// Whether the device supports f16.
    pub supports_f16: bool,
    /// Whether the device supports f64.
    pub supports_f64: bool,
    /// Maximum threads per block (for GPU).
    pub max_threads_per_block: usize,
    /// Compute capability version (for CUDA).
    pub compute_capability: Option<(usize, usize)>,
}

impl Device {
    /// Returns the capabilities of this device.
    #[must_use]
    pub fn capabilities(self) -> DeviceCapabilities {
        match self {
            Self::Cpu => DeviceCapabilities {
                name: "CPU".to_string(),
                total_memory: get_system_memory(),
                available_memory: get_available_memory(),
                supports_f16: true,
                supports_f64: true,
                max_threads_per_block: num_cpus(),
                compute_capability: None,
            },
            #[cfg(feature = "cuda")]
            Self::Cuda(idx) => crate::backends::cuda::get_capabilities(idx),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(idx) => crate::backends::vulkan::get_capabilities(idx),
            #[cfg(feature = "metal")]
            Self::Metal(idx) => crate::backends::metal::get_capabilities(idx),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(idx) => crate::backends::wgpu::get_capabilities(idx),
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Returns the total system memory in bytes.
fn get_system_memory() -> usize {
    let sys = System::new_all();
    sys.total_memory() as usize
}

/// Returns the available system memory in bytes.
fn get_available_memory() -> usize {
    let sys = System::new_all();
    sys.available_memory() as usize
}

/// Returns the number of CPU cores.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

impl DeviceCapabilities {
    /// Returns true if the device supports f32.
    #[must_use]
    pub const fn supports_f32(&self) -> bool {
        true // All devices support f32
    }
}

// =============================================================================
// Device Count Functions
// =============================================================================

/// Returns the number of available CUDA devices.
#[cfg(feature = "cuda")]
#[must_use]
pub fn cuda_device_count() -> usize {
    crate::backends::cuda::device_count()
}

/// Returns the number of available Vulkan devices.
#[cfg(feature = "vulkan")]
#[must_use]
pub fn vulkan_device_count() -> usize {
    crate::backends::vulkan::device_count()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device() {
        let device = Device::Cpu;
        assert!(device.is_cpu());
        assert!(!device.is_gpu());
        assert!(device.is_available());
        assert_eq!(device.device_type(), "cpu");
    }

    #[test]
    fn test_device_display() {
        let cpu = Device::Cpu;
        assert_eq!(format!("{cpu}"), "cpu");
    }

    #[test]
    fn test_device_default() {
        let device = Device::default();
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn test_device_capabilities() {
        let caps = Device::Cpu.capabilities();
        assert_eq!(caps.name, "CPU");
        assert!(caps.supports_f32());
    }
}
