//! Backends - Device-Specific Implementations
//!
//! # File
//! `crates/axonml-core/src/backends/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use crate::device::DeviceCapabilities;

// =============================================================================
// Backend Modules
// =============================================================================

pub mod cpu;

pub mod cuda;

#[cfg(feature = "cuda")]
pub mod cuda_kernels;

pub mod cuda_pool;

#[cfg(feature = "cudnn")]
pub mod cudnn_ops;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

// GPU testing infrastructure
pub mod gpu_tests;

// =============================================================================
// Re-exports
// =============================================================================

pub use cpu::CpuBackend;

pub use cuda::CudaBackend;

#[cfg(feature = "vulkan")]
pub use vulkan::VulkanBackend;

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

#[cfg(feature = "wgpu")]
pub use wgpu_backend::WgpuBackend;

// =============================================================================
// Backend Trait
// =============================================================================

/// Common trait for all compute backends.
///
/// This trait defines the interface that all backends must implement,
/// enabling device-agnostic tensor operations.
pub trait Backend: Send + Sync {
    /// Returns the name of this backend.
    fn name(&self) -> &'static str;

    /// Returns whether this backend is available on the current system.
    fn is_available(&self) -> bool;

    /// Returns the device capabilities.
    fn capabilities(&self) -> DeviceCapabilities;

    /// Allocates memory on this backend.
    fn allocate(&self, size: usize) -> *mut u8;

    /// Deallocates memory on this backend.
    fn deallocate(&self, ptr: *mut u8, size: usize);

    /// Copies data from host to device.
    fn copy_to_device(&self, dst: *mut u8, src: *const u8, size: usize);

    /// Copies data from device to host.
    fn copy_to_host(&self, dst: *mut u8, src: *const u8, size: usize);

    /// Copies data within the device.
    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize);

    /// Synchronizes the device (waits for all operations to complete).
    fn synchronize(&self);
}

// =============================================================================
// GPU Memory Management
// =============================================================================

/// GPU memory handle for safe memory management.
#[derive(Debug)]
pub struct GpuMemory {
    ptr: *mut u8,
    size: usize,
    device_index: usize,
    backend_type: BackendType,
}

/// Type of backend for a GPU memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CPU backend.
    Cpu,
    /// CUDA backend.
    #[cfg(feature = "cuda")]
    Cuda,
    /// Vulkan backend.
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// Metal backend.
    #[cfg(feature = "metal")]
    Metal,
    /// WebGPU backend.
    #[cfg(feature = "wgpu")]
    Wgpu,
}

impl GpuMemory {
    /// Creates a new GPU memory handle.
    pub fn new(ptr: *mut u8, size: usize, device_index: usize, backend_type: BackendType) -> Self {
        Self {
            ptr,
            size,
            device_index,
            backend_type,
        }
    }

    /// Returns the raw pointer.
    #[must_use]
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the device index.
    #[must_use]
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Returns the backend type.
    #[must_use]
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }
}

// =============================================================================
// GPU Stream/Queue Abstraction
// =============================================================================

/// GPU execution stream for async operations.
#[derive(Debug)]
pub struct GpuStream {
    /// Stream handle (backend-specific).
    handle: usize,
    /// Device index.
    device_index: usize,
    /// Backend type.
    backend_type: BackendType,
}

impl GpuStream {
    /// Creates a new GPU stream.
    #[must_use]
    pub fn new(handle: usize, device_index: usize, backend_type: BackendType) -> Self {
        Self {
            handle,
            device_index,
            backend_type,
        }
    }

    /// Returns the stream handle.
    #[must_use]
    pub fn handle(&self) -> usize {
        self.handle
    }

    /// Returns the device index.
    #[must_use]
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Synchronizes this stream (waits for all operations to complete).
    ///
    /// # Backend-specific behavior
    /// - **CPU**: No-op (CPU operations are synchronous)
    /// - **CUDA**: No-op at stream level; use `CudaBackend::synchronize()` for device sync
    /// - **Vulkan**: Waits for queue to become idle
    /// - **Metal**: Waits for command buffer completion
    /// - **WebGPU**: Submits pending commands to queue
    ///
    /// For CUDA, proper synchronization should be done through `CudaBackend::synchronize()`
    /// which performs device-level synchronization.
    pub fn synchronize(&self) {
        match self.backend_type {
            BackendType::Cpu => {} // No-op for CPU (synchronous)
            #[cfg(feature = "cuda")]
            BackendType::Cuda => cuda::stream_synchronize(self.handle),
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => vulkan::queue_wait_idle(self.handle),
            #[cfg(feature = "metal")]
            BackendType::Metal => metal::command_buffer_wait(self.handle),
            #[cfg(feature = "wgpu")]
            BackendType::Wgpu => wgpu_backend::queue_submit(self.handle),
        }
    }
}

// =============================================================================
// Device Selection Utilities
// =============================================================================

/// Returns the best available GPU backend.
#[must_use]
pub fn best_available_backend() -> BackendType {
    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        return BackendType::Cuda;
    }

    #[cfg(feature = "metal")]
    if metal::is_available() {
        return BackendType::Metal;
    }

    #[cfg(feature = "vulkan")]
    if vulkan::is_available() {
        return BackendType::Vulkan;
    }

    #[cfg(feature = "wgpu")]
    if wgpu_backend::is_available() {
        return BackendType::Wgpu;
    }

    BackendType::Cpu
}

/// Returns the number of available GPUs across all backends.
#[must_use]
pub fn gpu_count() -> usize {
    #[allow(unused_mut)]
    let mut count = 0_usize;

    #[cfg(feature = "cuda")]
    {
        count += cuda::device_count();
    }

    #[cfg(feature = "vulkan")]
    {
        count += vulkan::device_count();
    }

    #[cfg(feature = "metal")]
    {
        count += metal::device_count();
    }

    #[cfg(feature = "wgpu")]
    {
        count += wgpu_backend::device_count();
    }

    count
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_memory_creation() {
        let mem = GpuMemory::new(std::ptr::null_mut(), 1024, 0, BackendType::Cpu);
        assert_eq!(mem.size(), 1024);
        assert_eq!(mem.device_index(), 0);
        assert_eq!(mem.backend_type(), BackendType::Cpu);
        assert!(mem.ptr().is_null());
    }

    #[test]
    fn test_gpu_memory_nonzero_ptr() {
        let mut data = vec![0u8; 256];
        let ptr = data.as_mut_ptr();
        let mem = GpuMemory::new(ptr, 256, 0, BackendType::Cpu);
        assert_eq!(mem.ptr(), ptr);
        assert_eq!(mem.size(), 256);
    }

    #[test]
    fn test_gpu_stream_creation() {
        let stream = GpuStream::new(42, 0, BackendType::Cpu);
        assert_eq!(stream.handle(), 42);
        assert_eq!(stream.device_index(), 0);
    }

    #[test]
    fn test_gpu_stream_cpu_sync() {
        let stream = GpuStream::new(0, 0, BackendType::Cpu);
        // CPU sync is a no-op — should not panic
        stream.synchronize();
    }

    #[test]
    fn test_backend_type_equality() {
        assert_eq!(BackendType::Cpu, BackendType::Cpu);
        #[cfg(feature = "cuda")]
        assert_ne!(BackendType::Cpu, BackendType::Cuda);
    }

    #[test]
    fn test_best_available_backend() {
        let best = best_available_backend();
        // Should always return something (at minimum CPU)
        // Just verify it doesn't panic
        let _ = best;
    }

    #[test]
    fn test_gpu_count() {
        let count = gpu_count();
        // On a machine with a GPU this should be >= 1
        // On CI without GPU it's 0 — both are valid
        assert!(count < 1000, "Sanity check: unreasonable GPU count");
    }

    #[test]
    fn test_cpu_backend_is_available() {
        let cpu = CpuBackend::new();
        assert!(cpu.is_available());
        assert_eq!(cpu.name(), "cpu");
    }

    #[test]
    fn test_cpu_backend_allocate_deallocate() {
        let cpu = CpuBackend::new();
        let ptr = cpu.allocate(256);
        assert!(!ptr.is_null());
        cpu.deallocate(ptr, 256);
    }

    #[test]
    fn test_cpu_backend_zero_alloc() {
        let cpu = CpuBackend::new();
        let ptr = cpu.allocate(0);
        assert!(ptr.is_null());
    }

    #[test]
    fn test_cpu_backend_copy_round_trip() {
        let cpu = CpuBackend::new();
        let src: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let dst_ptr = cpu.allocate(16); // 4 f32s

        cpu.copy_to_device(dst_ptr, src.as_ptr() as *const u8, 16);

        let mut result = [0.0f32; 4];
        cpu.copy_to_host(result.as_mut_ptr() as *mut u8, dst_ptr as *const u8, 16);

        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
        cpu.deallocate(dst_ptr, 16);
    }

    #[test]
    fn test_cpu_backend_capabilities() {
        let cpu = CpuBackend::new();
        let caps = cpu.capabilities();
        assert!(caps.supports_f16);
        assert!(caps.supports_f64);
        assert!(caps.total_memory > 0);
    }
}
