//! CUDA Backend - NVIDIA GPU Operations
//!
//! Provides the CUDA implementation for tensor operations on NVIDIA GPUs.
//! This backend requires the `cuda` feature and NVIDIA CUDA toolkit.
//!
//! # Key Features
//! - cuBLAS integration for linear algebra
//! - Async execution with CUDA streams
//! - Multi-GPU support
//! - Unified memory support
//!
//! # Requirements
//! - NVIDIA GPU with compute capability 3.5+
//! - CUDA Toolkit 11.0+
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#[cfg(feature = "cuda")]
use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, GemmConfig};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr};

use super::Backend;
use crate::device::DeviceCapabilities;
use std::sync::Arc;

// =============================================================================
// CUDA Backend Struct
// =============================================================================

/// CUDA backend for tensor operations on NVIDIA GPUs.
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device_index: usize,
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    stream: CudaStream,
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaBackend {
    device_index: usize,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_index", &self.device_index)
            .finish()
    }
}

impl CudaBackend {
    /// Creates a new CUDA backend for the specified device.
    #[cfg(feature = "cuda")]
    pub fn new(device_index: usize) -> Option<Self> {
        let device = CudaDevice::new(device_index).ok()?;
        let device = Arc::new(device);
        let blas = CudaBlas::new(device.clone()).ok()?;
        let stream = device.fork_default_stream().ok()?;

        Some(Self {
            device_index,
            device,
            blas,
            stream,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(device_index: usize) -> Option<Self> {
        let _ = device_index;
        None // CUDA not available without feature
    }

    /// Returns the device index.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Returns the underlying CUDA device.
    #[cfg(feature = "cuda")]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Returns the cuBLAS handle.
    #[cfg(feature = "cuda")]
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// Returns the CUDA stream.
    #[cfg(feature = "cuda")]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Allocates a typed buffer on the GPU.
    #[cfg(feature = "cuda")]
    pub fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, CudaError> {
        self.device.alloc_zeros(len).map_err(CudaError::from)
    }

    /// Copies data from host to device.
    #[cfg(feature = "cuda")]
    pub fn htod_copy<T: DeviceRepr>(&self, src: &[T]) -> Result<CudaSlice<T>, CudaError> {
        self.device.htod_copy(src.to_vec()).map_err(CudaError::from)
    }

    /// Copies data from device to host.
    #[cfg(feature = "cuda")]
    pub fn dtoh_copy<T: DeviceRepr>(&self, src: &CudaSlice<T>) -> Result<Vec<T>, CudaError> {
        self.device.dtoh_sync_copy(src).map_err(CudaError::from)
    }
}

// =============================================================================
// Backend Trait Implementation
// =============================================================================

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    #[cfg(feature = "cuda")]
    fn is_available(&self) -> bool {
        true
    }

    #[cfg(not(feature = "cuda"))]
    fn is_available(&self) -> bool {
        false
    }

    #[cfg(feature = "cuda")]
    fn capabilities(&self) -> DeviceCapabilities {
        // Query actual device properties
        let name = format!("CUDA Device {}", self.device_index);

        // Get memory info
        let (free, total) = self.device.mem_info().unwrap_or((0, 0));

        DeviceCapabilities {
            name,
            total_memory: total as u64,
            available_memory: free as u64,
            supports_f16: true,
            supports_f64: true,
            max_threads_per_block: 1024,
            compute_capability: None, // Would need to query this
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            name: format!("CUDA Device {} (unavailable)", self.device_index),
            total_memory: 0,
            available_memory: 0,
            supports_f16: false,
            supports_f64: false,
            max_threads_per_block: 0,
            compute_capability: None,
        }
    }

    #[cfg(feature = "cuda")]
    fn allocate(&self, size: usize) -> *mut u8 {
        match self.device.alloc_zeros::<u8>(size) {
            Ok(slice) => {
                let ptr = *slice.device_ptr() as *mut u8;
                std::mem::forget(slice); // Don't drop, we're managing memory manually
                ptr
            }
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn allocate(&self, _size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    #[cfg(feature = "cuda")]
    fn deallocate(&self, ptr: *mut u8, size: usize) {
        if !ptr.is_null() {
            // Reconstruct the CudaSlice to properly free
            unsafe {
                let slice: CudaSlice<u8> = self
                    .device
                    .upgrade_device_ptr(DevicePtr::from(ptr as u64), size);
                drop(slice);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn deallocate(&self, _ptr: *mut u8, _size: usize) {}

    #[cfg(feature = "cuda")]
    fn copy_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let src_slice = std::slice::from_raw_parts(src, size);
            let _ =
                cudarc::driver::result::memcpy_htod_sync(DevicePtr::from(dst as u64), src_slice);
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn copy_to_device(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    #[cfg(feature = "cuda")]
    fn copy_to_host(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let dst_slice = std::slice::from_raw_parts_mut(dst, size);
            let _ =
                cudarc::driver::result::memcpy_dtoh_sync(dst_slice, DevicePtr::from(src as u64));
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn copy_to_host(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    #[cfg(feature = "cuda")]
    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let _ = cudarc::driver::result::memcpy_dtod_sync(
                DevicePtr::from(dst as u64),
                DevicePtr::from(src as u64),
                size,
            );
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn copy_device_to_device(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    #[cfg(feature = "cuda")]
    fn synchronize(&self) {
        let _ = self.device.synchronize();
    }

    #[cfg(not(feature = "cuda"))]
    fn synchronize(&self) {}
}

// =============================================================================
// CUDA Error Type
// =============================================================================

/// CUDA-specific error type
#[derive(Debug)]
pub enum CudaError {
    DeviceNotFound,
    AllocationFailed,
    CopyFailed,
    KernelLaunchFailed,
    BlasError(String),
    DriverError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceNotFound => write!(f, "CUDA device not found"),
            CudaError::AllocationFailed => write!(f, "CUDA memory allocation failed"),
            CudaError::CopyFailed => write!(f, "CUDA memory copy failed"),
            CudaError::KernelLaunchFailed => write!(f, "CUDA kernel launch failed"),
            CudaError::BlasError(s) => write!(f, "cuBLAS error: {}", s),
            CudaError::DriverError(s) => write!(f, "CUDA driver error: {}", s),
        }
    }
}

impl std::error::Error for CudaError {}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        CudaError::DriverError(e.to_string())
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::cublas::result::CublasError> for CudaError {
    fn from(e: cudarc::cublas::result::CublasError) -> Self {
        CudaError::BlasError(format!("{:?}", e))
    }
}

// =============================================================================
// CUDA Runtime Functions
// =============================================================================

/// Returns whether CUDA is available on this system.
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        CudaDevice::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Returns the number of available CUDA devices.
pub fn device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::result::device::get_count().unwrap_or(0) as usize
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Returns whether a specific CUDA device is available.
pub fn is_device_available(index: usize) -> bool {
    index < device_count()
}

/// Returns the capabilities of a CUDA device.
pub fn get_capabilities(index: usize) -> DeviceCapabilities {
    #[cfg(feature = "cuda")]
    {
        if let Some(backend) = CudaBackend::new(index) {
            return backend.capabilities();
        }
    }

    DeviceCapabilities {
        name: format!("CUDA Device {}", index),
        total_memory: 0,
        available_memory: 0,
        supports_f16: true,
        supports_f64: true,
        max_threads_per_block: 1024,
        compute_capability: None,
    }
}

// =============================================================================
// CUDA Stream Functions
// =============================================================================

/// CUDA stream wrapper for async operations
#[cfg(feature = "cuda")]
pub struct Stream {
    inner: CudaStream,
}

#[cfg(feature = "cuda")]
impl Stream {
    /// Creates a new CUDA stream.
    pub fn new(backend: &CudaBackend) -> Result<Self, CudaError> {
        let inner = backend.device.fork_default_stream()?;
        Ok(Self { inner })
    }

    /// Synchronizes the stream.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.inner.synchronize().map_err(CudaError::from)
    }
}

// =============================================================================
// cuBLAS Operations
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Performs matrix multiplication using cuBLAS: C = alpha * A @ B + beta * C
    pub fn gemm_f32(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        lda: usize,
        b: &CudaSlice<f32>,
        ldb: usize,
        beta: f32,
        c: &mut CudaSlice<f32>,
        ldc: usize,
    ) -> Result<(), CudaError> {
        use cudarc::cublas::Gemm;

        let cfg = GemmConfig {
            transa: if transa {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            },
            transb: if transb {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            },
            m: m as i32,
            n: n as i32,
            k: k as i32,
            alpha,
            lda: lda as i32,
            ldb: ldb as i32,
            beta,
            ldc: ldc as i32,
        };

        unsafe { self.blas.gemm(cfg, a, b, c).map_err(CudaError::from) }
    }

    /// Performs batched matrix multiplication.
    pub fn gemm_batched_f32(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_array: &[&CudaSlice<f32>],
        lda: usize,
        b_array: &[&CudaSlice<f32>],
        ldb: usize,
        beta: f32,
        c_array: &mut [&mut CudaSlice<f32>],
        ldc: usize,
        batch_count: usize,
    ) -> Result<(), CudaError> {
        // Execute batched gemm by iterating (cudarc doesn't expose batched directly)
        for i in 0..batch_count {
            let cfg = GemmConfig {
                transa: if transa {
                    cublasOperation_t::CUBLAS_OP_T
                } else {
                    cublasOperation_t::CUBLAS_OP_N
                },
                transb: if transb {
                    cublasOperation_t::CUBLAS_OP_T
                } else {
                    cublasOperation_t::CUBLAS_OP_N
                },
                m: m as i32,
                n: n as i32,
                k: k as i32,
                alpha,
                lda: lda as i32,
                ldb: ldb as i32,
                beta,
                ldc: ldc as i32,
            };

            unsafe {
                self.blas
                    .gemm(cfg, a_array[i], b_array[i], c_array[i])
                    .map_err(CudaError::from)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// Tensor Operations
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Element-wise addition: dst = a + b
    pub fn add_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        use cudarc::cublas::Axpy;

        // Copy a to dst
        self.device.dtod_copy(a, dst)?;

        // dst = dst + 1.0 * b (axpy)
        unsafe { self.blas.axpy(1.0f32, b, dst).map_err(CudaError::from) }
    }

    /// Scalar multiplication: dst = alpha * a
    pub fn scale_f32(&self, dst: &mut CudaSlice<f32>, alpha: f32) -> Result<(), CudaError> {
        use cudarc::cublas::Scal;

        unsafe { self.blas.scal(alpha, dst).map_err(CudaError::from) }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        let available = is_available();
        println!("CUDA available: {}", available);
    }

    #[test]
    fn test_device_count() {
        let count = device_count();
        println!("CUDA device count: {}", count);
        assert!(count <= 16);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_backend_creation() {
        if is_available() {
            let backend = CudaBackend::new(0);
            assert!(backend.is_some());
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_memory_operations() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        // Test allocation
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu_data = backend.htod_copy(&data).unwrap();

        // Test copy back
        let result = backend.dtoh_copy(&gpu_data).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_gemm() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        // 2x3 @ 3x2 = 2x2
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let mut c: Vec<f32> = vec![0.0; 4]; // 2x2

        let a_gpu = backend.htod_copy(&a).unwrap();
        let b_gpu = backend.htod_copy(&b).unwrap();
        let mut c_gpu = backend.htod_copy(&c).unwrap();

        backend
            .gemm_f32(
                false, false, 2, 2, 3,   // m, n, k
                1.0, // alpha
                &a_gpu, 3, // A, lda
                &b_gpu, 2,   // B, ldb
                0.0, // beta
                &mut c_gpu, 2, // C, ldc
            )
            .unwrap();

        let result = backend.dtoh_copy(&c_gpu).unwrap();
        // Expected: [[22, 28], [49, 64]]
        assert!((result[0] - 22.0).abs() < 1e-5);
        assert!((result[3] - 64.0).abs() < 1e-5);
    }
}
