//! CUDA Backend - NVIDIA GPU Operations
//!
//! Provides the CUDA implementation for tensor operations on NVIDIA GPUs.
//! This backend requires the `cuda` feature and NVIDIA CUDA toolkit.
//!
//! # Key Features
//! - cuBLAS integration for linear algebra
//! - Multi-GPU support
//!
//! # Requirements
//! - NVIDIA GPU with compute capability 3.5+
//! - CUDA Toolkit 11.0+
//!
//! @version 0.2.0
//! @author AutomataNexus Development Team

#[cfg(feature = "cuda")]
use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};

use super::Backend;
use crate::device::DeviceCapabilities;
use std::sync::Arc;

// =============================================================================
// CUDA Backend Struct
// =============================================================================

/// CUDA backend for tensor operations on NVIDIA GPUs.
///
/// Note: CudaStream is not Send+Sync, so we don't store it in the struct.
/// Instead, we use synchronous operations and the device's default stream.
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device_index: usize,
    device: Arc<CudaDevice>,
    blas: CudaBlas,
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaBackend {
    device_index: usize,
}

// Implement Send and Sync for CudaBackend
// Safe because CudaDevice and CudaBlas are internally synchronized
#[cfg(feature = "cuda")]
unsafe impl Send for CudaBackend {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaBackend {}

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
        // CudaDevice::new returns Result<Arc<CudaDevice>, _>
        let device = CudaDevice::new(device_index).ok()?;
        let blas = CudaBlas::new(device.clone()).ok()?;

        Some(Self {
            device_index,
            device,
            blas,
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

    /// Allocates a typed buffer on the GPU initialized to zeros.
    #[cfg(feature = "cuda")]
    pub fn alloc<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, CudaError> {
        self.device.alloc_zeros(len).map_err(CudaError::from)
    }

    /// Allocates uninitialized memory on the GPU.
    #[cfg(feature = "cuda")]
    pub fn alloc_uninit<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, CudaError> {
        unsafe { self.device.alloc(len).map_err(CudaError::from) }
    }

    /// Copies data from host to device.
    #[cfg(feature = "cuda")]
    pub fn htod_copy<T: DeviceRepr + Clone + Unpin>(
        &self,
        src: &[T],
    ) -> Result<CudaSlice<T>, CudaError> {
        self.device.htod_copy(src.to_vec()).map_err(CudaError::from)
    }

    /// Copies data from device to host.
    #[cfg(feature = "cuda")]
    pub fn dtoh_copy<T: DeviceRepr + Clone + Default + Unpin>(
        &self,
        src: &CudaSlice<T>,
    ) -> Result<Vec<T>, CudaError> {
        self.device.dtoh_sync_copy(src).map_err(CudaError::from)
    }
}

// =============================================================================
// Backend Trait Implementation
// =============================================================================

#[cfg(feature = "cuda")]
impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn capabilities(&self) -> DeviceCapabilities {
        // Query actual device properties
        let name = format!("CUDA Device {}", self.device_index);

        // Get memory info via CUDA driver API
        let (free, total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));

        DeviceCapabilities {
            name,
            total_memory: total,
            available_memory: free,
            supports_f16: true,
            supports_f64: true,
            max_threads_per_block: 1024,
            compute_capability: None, // Would need to query this
        }
    }

    fn allocate(&self, size: usize) -> *mut u8 {
        match self.device.alloc_zeros::<u8>(size) {
            Ok(slice) => {
                // Get the raw device pointer
                use cudarc::driver::DevicePtr;
                let ptr = *slice.device_ptr() as *mut u8;
                std::mem::forget(slice); // Don't drop, we're managing memory manually
                ptr
            }
            Err(_) => std::ptr::null_mut(),
        }
    }

    fn deallocate(&self, ptr: *mut u8, size: usize) {
        if !ptr.is_null() {
            // Reconstruct the CudaSlice to properly free
            unsafe {
                let slice: CudaSlice<u8> = self.device.upgrade_device_ptr(ptr as u64, size);
                drop(slice);
            }
        }
    }

    fn copy_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let src_slice = std::slice::from_raw_parts(src, size);
            let _ = cudarc::driver::result::memcpy_htod_sync(dst as u64, src_slice);
        }
    }

    fn copy_to_host(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let dst_slice = std::slice::from_raw_parts_mut(dst, size);
            let _ = cudarc::driver::result::memcpy_dtoh_sync(dst_slice, src as u64);
        }
    }

    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let _ = cudarc::driver::result::memcpy_dtod_sync(dst as u64, src as u64, size);
        }
    }

    fn synchronize(&self) {
        let _ = self.device.synchronize();
    }
}

#[cfg(not(feature = "cuda"))]
impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        false
    }

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

    fn allocate(&self, _size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    fn deallocate(&self, _ptr: *mut u8, _size: usize) {}

    fn copy_to_device(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn copy_to_host(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn copy_device_to_device(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn synchronize(&self) {}
}

// =============================================================================
// CUDA Error Type
// =============================================================================

/// CUDA-specific error type
#[derive(Debug)]
pub enum CudaError {
    /// CUDA device was not found
    DeviceNotFound,
    /// Memory allocation on the GPU failed
    AllocationFailed,
    /// Memory copy operation failed
    CopyFailed,
    /// CUDA kernel launch failed
    KernelLaunchFailed,
    /// cuBLAS operation error
    BlasError(String),
    /// CUDA driver error
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
    #[allow(unreachable_code)]
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

/// Synchronizes a CUDA stream by handle.
///
/// # Design Note
/// This function exists for API compatibility with the `GpuStream` abstraction.
/// However, AxonML's CUDA backend uses the device's default stream exclusively
/// (CudaStream is not Send+Sync, so explicit stream management is avoided).
///
/// For proper synchronization:
/// - Use `CudaBackend::synchronize()` which calls `cudaDeviceSynchronize()`
/// - This synchronizes all pending operations on the device
///
/// The handle parameter is accepted but not used because cudarc manages
/// streams internally and doesn't expose raw stream handles.
///
/// # Arguments
/// * `_handle` - Stream handle (unused, kept for API compatibility)
#[cfg(feature = "cuda")]
pub fn stream_synchronize(_handle: usize) {
    // AxonML uses CudaDevice's default stream for all operations.
    // Stream-level synchronization requires a CudaDevice reference.
    // Use CudaBackend::synchronize() for device-level synchronization.
    //
    // Without a global device registry, we cannot synchronize here.
    // This is intentional: synchronization should be explicit via CudaBackend.
}

#[cfg(not(feature = "cuda"))]
pub fn stream_synchronize(_handle: usize) {
    // No-op when CUDA is not available
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

    /// Element-wise addition using device-to-device copy and manual computation.
    pub fn add_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        _len: usize,
    ) -> Result<(), CudaError> {
        // Copy a to host, b to host, add, copy back
        let a_host = self.dtoh_copy(a)?;
        let b_host = self.dtoh_copy(b)?;

        let result: Vec<f32> = a_host
            .iter()
            .zip(b_host.iter())
            .map(|(x, y)| x + y)
            .collect();

        // Copy result to dst
        let result_gpu = self.htod_copy(&result)?;
        self.device.dtod_copy(&result_gpu, dst)?;

        Ok(())
    }

    /// Scalar multiplication.
    pub fn scale_f32(&self, dst: &mut CudaSlice<f32>, alpha: f32) -> Result<(), CudaError> {
        // Copy to host, scale, copy back
        let host_data = self.dtoh_copy(dst)?;
        let scaled: Vec<f32> = host_data.iter().map(|x| x * alpha).collect();
        let scaled_gpu = self.htod_copy(&scaled)?;
        self.device.dtod_copy(&scaled_gpu, dst)?;
        Ok(())
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

        // cuBLAS uses column-major order
        // To compute C = A @ B where:
        //   A is 2x3 (m=2, k=3) and B is 3x2 (k=3, n=2), C is 2x2 (m=2, n=2)
        // In column-major: lda >= m, ldb >= k, ldc >= m
        //
        // A in column-major (2x3):
        // | a00 a01 a02 |    stored as: [a00, a10, a01, a11, a02, a12]
        // | a10 a11 a12 |
        let a: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // column-major 2x3
                                                              // B in column-major (3x2):
                                                              // | b00 b01 |    stored as: [b00, b10, b20, b01, b11, b21]
                                                              // | b10 b11 |
                                                              // | b20 b21 |
        let b: Vec<f32> = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // column-major 3x2
        let c: Vec<f32> = vec![0.0; 4]; // 2x2

        let a_gpu = backend.htod_copy(&a).unwrap();
        let b_gpu = backend.htod_copy(&b).unwrap();
        let mut c_gpu = backend.htod_copy(&c).unwrap();

        // C = A @ B
        // m=2 (rows of A, rows of C)
        // n=2 (cols of B, cols of C)
        // k=3 (cols of A, rows of B)
        // lda=2 (leading dimension of A, >= m)
        // ldb=3 (leading dimension of B, >= k)
        // ldc=2 (leading dimension of C, >= m)
        backend
            .gemm_f32(
                false, false, 2, 2, 3,   // m, n, k
                1.0, // alpha
                &a_gpu, 2, // A, lda
                &b_gpu, 3,   // B, ldb
                0.0, // beta
                &mut c_gpu, 2, // C, ldc
            )
            .unwrap();

        let result = backend.dtoh_copy(&c_gpu).unwrap();
        // C = A @ B (in matrix form, row-major interpretation):
        // A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
        // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        // Column-major result: [22, 49, 28, 64]
        assert!((result[0] - 22.0).abs() < 1e-5, "result[0] = {}", result[0]);
        assert!((result[1] - 49.0).abs() < 1e-5, "result[1] = {}", result[1]);
        assert!((result[2] - 28.0).abs() < 1e-5, "result[2] = {}", result[2]);
        assert!((result[3] - 64.0).abs() < 1e-5, "result[3] = {}", result[3]);
    }
}
