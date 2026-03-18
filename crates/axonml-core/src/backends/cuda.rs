//! CUDA Backend - NVIDIA GPU Operations
//!
//! # File
//! `crates/axonml-core/src/backends/cuda.rs`
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

#[cfg(feature = "cuda")]
use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};

#[cfg(feature = "cuda")]
use super::cuda_kernels::{self, CudaKernels, BLOCK_SIZE};
use super::Backend;
use crate::device::DeviceCapabilities;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

// =============================================================================
// Global CUDA Backend Singleton
// =============================================================================

#[cfg(feature = "cuda")]
static CUDA_BACKEND: OnceLock<Option<CudaBackend>> = OnceLock::new();

/// Get the global CUDA backend singleton (initialized lazily on first call).
#[cfg(feature = "cuda")]
pub fn get_cuda_backend() -> Option<&'static CudaBackend> {
    CUDA_BACKEND
        .get_or_init(|| {
            let backend = CudaBackend::new(0);
            if backend.is_some() {
                eprintln!("[AxonML] CUDA backend initialized (GPU 0)");
            }
            backend
        })
        .as_ref()
}

/// Get the global CUDA backend singleton (stub when cuda feature disabled).
#[cfg(not(feature = "cuda"))]
pub fn get_cuda_backend() -> Option<&'static CudaBackend> {
    None
}

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
    kernels: CudaKernels,
}

/// CUDA backend stub when the `cuda` feature is disabled.
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
        let kernels = match CudaKernels::load(device.clone()) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[AxonML CUDA] Kernel loading failed: {:?}", e);
                return None;
            }
        };

        Some(Self {
            device_index,
            device,
            blas,
            kernels,
        })
    }

    /// Creates a new CUDA backend (stub, always returns None without the `cuda` feature).
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

/// Synchronize the CUDA device (wait for all GPU operations to complete).
/// Returns true if sync was performed, false if CUDA is not available.
#[cfg(feature = "cuda")]
pub fn cuda_sync() -> bool {
    if let Some(backend) = get_cuda_backend() {
        let _ = backend.device.synchronize();
        true
    } else {
        false
    }
}

/// Synchronize the CUDA device (no-op without the `cuda` feature).
#[cfg(not(feature = "cuda"))]
pub fn cuda_sync() -> bool {
    false
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
    /// PTX module loading failed
    ModuleLoadFailed(String),
    /// Kernel function not found in module
    KernelNotFound(String),
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
            CudaError::ModuleLoadFailed(s) => write!(f, "CUDA module load failed: {}", s),
            CudaError::KernelNotFound(s) => write!(f, "CUDA kernel not found: {}", s),
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

/// Synchronize a CUDA stream (no-op without the `cuda` feature).
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

    /// Strided batched GEMM using cublasSgemmStridedBatched.
    /// All batch data in contiguous GPU memory with fixed strides between batches.
    /// C[i] = alpha * A[i] @ B[i] + beta * C[i] for i in 0..batch_count
    pub fn gemm_strided_batched_f32(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        lda: usize,
        stride_a: i64,
        b: &CudaSlice<f32>,
        ldb: usize,
        stride_b: i64,
        beta: f32,
        c: &mut CudaSlice<f32>,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<(), CudaError> {
        use cudarc::cublas::result::sgemm_strided_batched;
        use cudarc::driver::safe::DevicePtr;

        let op_a = if transa {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };

        let a_ptr = *a.device_ptr() as *const f32;
        let b_ptr = *b.device_ptr() as *const f32;
        let c_ptr = *c.device_ptr() as *mut f32;

        unsafe {
            sgemm_strided_batched(
                *self.blas.handle(),
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32,
                a_ptr,
                lda as i32,
                stride_a,
                b_ptr,
                ldb as i32,
                stride_b,
                &beta as *const f32,
                c_ptr,
                ldc as i32,
                stride_c,
                batch_count as i32,
            )
            .map_err(CudaError::from)
        }
    }

    /// Element-wise addition using CUDA kernel.
    pub fn add_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("add_f32")
            .ok_or_else(|| CudaError::KernelNotFound("add_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Scalar multiplication using CUDA kernel.
    pub fn scale_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        alpha: f32,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("scale_f32")
            .ok_or_else(|| CudaError::KernelNotFound("scale_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (dst, alpha, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise multiplication using CUDA kernel.
    pub fn mul_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("mul_f32")
            .ok_or_else(|| CudaError::KernelNotFound("mul_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// ReLU activation using CUDA kernel.
    pub fn relu_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("relu_f32")
            .ok_or_else(|| CudaError::KernelNotFound("relu_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Sigmoid activation using CUDA kernel.
    pub fn sigmoid_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sigmoid_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sigmoid_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Tanh activation using CUDA kernel.
    pub fn tanh_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("tanh_f32")
            .ok_or_else(|| CudaError::KernelNotFound("tanh_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise subtraction using CUDA kernel.
    pub fn sub_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sub_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sub_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise division using CUDA kernel.
    pub fn div_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("div_f32")
            .ok_or_else(|| CudaError::KernelNotFound("div_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    // =========================================================================
    // Broadcast Element-wise Operations
    // =========================================================================

    /// Broadcast addition: out[i] = a[i] + b[i % b_len]
    /// `a` is the larger tensor (n elements), `b` is broadcast (b_len elements).
    pub fn broadcast_add_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        b_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_add_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_add_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, b_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Broadcast subtraction: out[i] = a[i] - b[i % b_len]
    pub fn broadcast_sub_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        b_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_sub_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_sub_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, b_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Broadcast multiplication: out[i] = a[i] * b[i % b_len]
    pub fn broadcast_mul_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        b_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_mul_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_mul_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, b_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Broadcast division: out[i] = a[i] / b[i % b_len]
    pub fn broadcast_div_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        b_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_div_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_div_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, b_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Reverse broadcast addition: out[i] = a[i % a_len] + b[i]
    pub fn broadcast_add_rev_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        a_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_add_rev_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_add_rev_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, a_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Reverse broadcast subtraction: out[i] = a[i % a_len] - b[i]
    pub fn broadcast_sub_rev_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        a_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_sub_rev_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_sub_rev_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, a_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Reverse broadcast multiplication: out[i] = a[i % a_len] * b[i]
    pub fn broadcast_mul_rev_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        a_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_mul_rev_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_mul_rev_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, a_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Reverse broadcast division: out[i] = a[i % a_len] / b[i]
    pub fn broadcast_div_rev_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
        a_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_div_rev_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_div_rev_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, n as u32, a_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise negation using CUDA kernel.
    pub fn neg_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("neg_f32")
            .ok_or_else(|| CudaError::KernelNotFound("neg_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise power using CUDA kernel: dst[i] = a[i] ^ b[i].
    pub fn pow_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("pow_f32")
            .ok_or_else(|| CudaError::KernelNotFound("pow_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (a, b, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise power with scalar exponent: dst[i] = src[i] ^ exp.
    pub fn pow_scalar_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        exp: f32,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("pow_scalar_f32")
            .ok_or_else(|| CudaError::KernelNotFound("pow_scalar_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, exp, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise exp using CUDA kernel.
    pub fn exp_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("exp_f32")
            .ok_or_else(|| CudaError::KernelNotFound("exp_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise log using CUDA kernel.
    pub fn log_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("log_f32")
            .ok_or_else(|| CudaError::KernelNotFound("log_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Element-wise sqrt using CUDA kernel.
    pub fn sqrt_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sqrt_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sqrt_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// GELU activation using CUDA kernel.
    pub fn gelu_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("gelu_f32")
            .ok_or_else(|| CudaError::KernelNotFound("gelu_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// SiLU activation using CUDA kernel.
    pub fn silu_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("silu_f32")
            .ok_or_else(|| CudaError::KernelNotFound("silu_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Scalar addition: dst[i] = src[i] + scalar.
    pub fn add_scalar_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        scalar: f32,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("add_scalar_f32")
            .ok_or_else(|| CudaError::KernelNotFound("add_scalar_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (src, scalar, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// ReLU backward using CUDA kernel.
    pub fn relu_backward_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("relu_backward_f32")
            .ok_or_else(|| CudaError::KernelNotFound("relu_backward_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (grad_output, input, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Sigmoid backward using CUDA kernel.
    pub fn sigmoid_backward_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sigmoid_backward_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sigmoid_backward_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (grad_output, output, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Tanh backward using CUDA kernel.
    pub fn tanh_backward_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("tanh_backward_f32")
            .ok_or_else(|| CudaError::KernelNotFound("tanh_backward_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            func.clone()
                .launch(cfg, (grad_output, output, dst, len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Sum along a dimension. Tensor viewed as [outer_size, dim_size, inner_size].
    /// Output has outer_size * inner_size elements.
    pub fn sum_dim_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        outer_size: usize,
        dim_size: usize,
        inner_size: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sum_dim_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sum_dim_f32".to_string()))?;
        let out_len = outer_size * inner_size;
        let cfg = cuda_kernels::launch_config(out_len);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        src,
                        dst,
                        outer_size as u32,
                        dim_size as u32,
                        inner_size as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Softmax along last dimension, in-place.
    /// Data layout: num_rows x row_size, each row gets softmax independently.
    /// One block per row, 256 threads per block.
    pub fn softmax_row_f32(
        &self,
        data: &mut CudaSlice<f32>,
        num_rows: usize,
        row_size: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("softmax_row_f32")
            .ok_or_else(|| CudaError::KernelNotFound("softmax_row_f32".to_string()))?;
        // One block per row
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * 4,
        };
        unsafe {
            func.clone()
                .launch(cfg, (data, num_rows as u32, row_size as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Broadcast copy: out[i] = src[i % src_len], for n output elements.
    pub fn broadcast_copy_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        n: usize,
        src_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("broadcast_copy_f32")
            .ok_or_else(|| CudaError::KernelNotFound("broadcast_copy_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (src, dst, n as u32, src_len as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// LayerNorm: per-row normalization with affine transform on GPU.
    /// One block per row, 256 threads. Computes mean, variance, normalize, apply gamma/beta.
    pub fn layer_norm_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        norm_size: usize,
        eps: f32,
        num_rows: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("layer_norm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("layer_norm_f32".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * 4,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        input,
                        gamma,
                        beta,
                        dst,
                        norm_size as u32,
                        eps,
                        num_rows as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Softmax backward: per-row backward pass.
    /// result[i] = softmax[i] * (grad[i] - dot), where dot = sum(softmax * grad) per row.
    /// One block per row, 256 threads.
    pub fn softmax_backward_row_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        softmax_output: &CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        num_rows: usize,
        row_size: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("softmax_backward_row_f32")
            .ok_or_else(|| CudaError::KernelNotFound("softmax_backward_row_f32".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * 4,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        softmax_output,
                        grad_output,
                        dst,
                        num_rows as u32,
                        row_size as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// LayerNorm backward: compute d_input on GPU.
    /// One block per row, 256 threads. Computes mean, var, sum_dy, sum_dy_xhat, then d_input.
    pub fn layer_norm_backward_dinput_f32(
        &self,
        d_input: &mut CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        norm_size: usize,
        eps: f32,
        num_rows: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("layer_norm_backward_dinput_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("layer_norm_backward_dinput_f32".to_string())
            })?;
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * 4 * 2, // two shared arrays
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        grad_output,
                        input,
                        gamma,
                        d_input,
                        norm_size as u32,
                        eps,
                        num_rows as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// LayerNorm backward: compute d_weight and d_bias on GPU.
    /// One thread per element in norm_size. Each thread loops over all rows.
    pub fn layer_norm_backward_dweight_dbias_f32(
        &self,
        d_weight: &mut CudaSlice<f32>,
        d_bias: &mut CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        norm_size: usize,
        eps: f32,
        num_rows: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("layer_norm_backward_dweight_dbias_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("layer_norm_backward_dweight_dbias_f32".to_string())
            })?;
        let cfg = cuda_kernels::launch_config(norm_size);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        grad_output,
                        input,
                        d_weight,
                        d_bias,
                        norm_size as u32,
                        eps,
                        num_rows as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Gather elements from src using index array: out[i] = src[indices[i]]
    pub fn gather_contiguous_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        indices: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("gather_contiguous_f32")
            .ok_or_else(|| CudaError::KernelNotFound("gather_contiguous_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (src, indices, dst, n as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Embedding scatter-add: atomically accumulates gradients into weight_grad.
    /// Each thread handles one element of grad_src (total = num_indices * emb_dim).
    pub fn embedding_scatter_add_f32(
        &self,
        grad_src: &CudaSlice<f32>,
        indices: &CudaSlice<u32>,
        weight_grad: &mut CudaSlice<f32>,
        total_n: usize,
        emb_dim: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("embedding_scatter_add_f32")
            .ok_or_else(|| CudaError::KernelNotFound("embedding_scatter_add_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total_n);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        grad_src,
                        indices,
                        weight_grad,
                        total_n as u32,
                        emb_dim as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused Adam optimizer step: updates param, exp_avg, exp_avg_sq in-place on GPU.
    /// Eliminates the GPU->CPU->GPU copy in standard Adam.
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_f32(
        &self,
        param: &mut CudaSlice<f32>,
        grad: &CudaSlice<f32>,
        exp_avg: &mut CudaSlice<f32>,
        exp_avg_sq: &mut CudaSlice<f32>,
        n: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("adam_step_f32")
            .ok_or_else(|| CudaError::KernelNotFound("adam_step_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        param,
                        grad,
                        exp_avg,
                        exp_avg_sq,
                        n as u32,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        bias_correction1,
                        bias_correction2,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Compute sum of squares of all elements (for gradient norm).
    /// Result is atomically accumulated into output[0].
    pub fn grad_norm_sq_f32(
        &self,
        data: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("grad_norm_sq_f32")
            .ok_or_else(|| CudaError::KernelNotFound("grad_norm_sq_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (data, output, n as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Scale all elements in-place: data[i] *= scale
    pub fn grad_scale_f32(
        &self,
        data: &mut CudaSlice<f32>,
        n: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("grad_scale_f32")
            .ok_or_else(|| CudaError::KernelNotFound("grad_scale_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (data, n as u32, scale))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// CrossEntropy forward: fused softmax + NLL loss.
    /// One block per batch item, 256 threads per block.
    /// Returns per-sample losses and softmax probabilities (for backward).
    pub fn cross_entropy_fwd_f32(
        &self,
        logits: &CudaSlice<f32>,
        targets: &CudaSlice<f32>,
        losses: &mut CudaSlice<f32>,
        softmax_out: &mut CudaSlice<f32>,
        batch_size: usize,
        num_classes: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("cross_entropy_fwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("cross_entropy_fwd_f32".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * 4,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (logits, targets, losses, softmax_out, num_classes as u32),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// CrossEntropy backward: grad = (softmax - one_hot(target)) * grad_output.
    /// Elementwise kernel, one thread per element.
    pub fn cross_entropy_bwd_f32(
        &self,
        softmax_probs: &CudaSlice<f32>,
        targets: &CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        grad_input: &mut CudaSlice<f32>,
        batch_size: usize,
        num_classes: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("cross_entropy_bwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("cross_entropy_bwd_f32".to_string()))?;
        let total = batch_size * num_classes;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        softmax_probs,
                        targets,
                        grad_output,
                        grad_input,
                        batch_size as u32,
                        num_classes as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Zero-fills a GPU allocation using cudaMemset.
    #[cfg(feature = "cuda")]
    pub fn memset_zeros_f32(&self, dst: &mut CudaSlice<f32>) -> Result<(), CudaError> {
        self.device
            .memset_zeros(dst)
            .map_err(|e| CudaError::DriverError(e.to_string()))
    }

    /// Device-to-device copy of `count` f32 elements with source and destination offsets.
    /// Copies src[src_offset..src_offset+count] → dst[dst_offset..dst_offset+count].
    #[cfg(feature = "cuda")]
    pub fn memcpy_dtod_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        dst_offset: usize,
        src: &CudaSlice<f32>,
        src_offset: usize,
        count: usize,
    ) -> Result<(), CudaError> {
        use cudarc::driver::safe::{DevicePtr, DevicePtrMut};
        let src_ptr = *src.device_ptr() as u64 + (src_offset * std::mem::size_of::<f32>()) as u64;
        let dst_ptr =
            *dst.device_ptr_mut() as u64 + (dst_offset * std::mem::size_of::<f32>()) as u64;
        let size = count * std::mem::size_of::<f32>();
        unsafe {
            cudarc::driver::result::memcpy_dtod_sync(dst_ptr, src_ptr, size)
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }
}

// =============================================================================
// Attention Mask Expansion GPU Operations
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Expand causal mask [T, S] → [B, H, T, S] with 0→-1e9 conversion, entirely on GPU.
    pub fn mask_expand_causal_f32(
        &self,
        mask: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        total_n: usize,
        tgt_len: usize,
        src_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("mask_expand_causal_f32")
            .ok_or_else(|| CudaError::KernelNotFound("mask_expand_causal_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total_n);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (mask, output, total_n as u32, tgt_len as u32, src_len as u32),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Expand padding mask [B, S] → [B, H, T, S] with 0→-1e9 conversion, entirely on GPU.
    pub fn mask_expand_padding_f32(
        &self,
        mask: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        total_n: usize,
        num_heads: usize,
        tgt_len: usize,
        src_len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("mask_expand_padding_f32")
            .ok_or_else(|| CudaError::KernelNotFound("mask_expand_padding_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total_n);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        mask,
                        output,
                        total_n as u32,
                        num_heads as u32,
                        tgt_len as u32,
                        src_len as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }
}

// =============================================================================
// Strided Gather (GPU-native contiguous)
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Gather elements from a strided tensor layout into contiguous output on GPU.
    /// Replaces the CPU index computation in contiguous_gpu().
    pub fn strided_gather_f32(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        strides: &CudaSlice<i64>,
        shape: &CudaSlice<u32>,
        ndim: usize,
        offset: usize,
        total_n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("strided_gather_f32")
            .ok_or_else(|| CudaError::KernelNotFound("strided_gather_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(total_n);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        src,
                        dst,
                        strides,
                        shape,
                        ndim as u32,
                        offset as u32,
                        total_n as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    // =========================================================================
    // Fused LSTM Gate Kernel
    // =========================================================================

    /// Fused LSTM gate computation on GPU.
    ///
    /// Takes pre-computed gates (ih + hh from cuBLAS GEMM) and c_prev,
    /// applies sigmoid/tanh activations and cell/hidden state update
    /// in a single kernel launch.
    ///
    /// - `gates`: [batch, 4*hidden] = x@W_ih^T + b_ih + h@W_hh^T + b_hh
    /// - `c_prev`: [batch, hidden]
    /// - `h_new`: [batch, hidden] output
    /// - `c_new`: [batch, hidden] output
    pub fn lstm_gates_f32(
        &self,
        gates: &CudaSlice<f32>,
        c_prev: &CudaSlice<f32>,
        h_new: &mut CudaSlice<f32>,
        c_new: &mut CudaSlice<f32>,
        hidden_size: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("lstm_gates_f32")
            .ok_or_else(|| CudaError::KernelNotFound("lstm_gates_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        gates,
                        c_prev,
                        h_new,
                        c_new,
                        hidden_size as u32,
                        total as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    // =========================================================================
    // Fused GRU Gate Kernel
    // =========================================================================

    /// Fused GRU gate computation on GPU.
    ///
    /// - `gates_ih`: [batch, 3*hidden] = x@W_ih^T + b_ih
    /// - `gates_hh`: [batch, 3*hidden] = h@W_hh^T + b_hh
    /// - `h_prev`: [batch, hidden]
    /// - `h_new`: [batch, hidden] output
    pub fn gru_gates_f32(
        &self,
        gates_ih: &CudaSlice<f32>,
        gates_hh: &CudaSlice<f32>,
        h_prev: &CudaSlice<f32>,
        h_new: &mut CudaSlice<f32>,
        hidden_size: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("gru_gates_f32")
            .ok_or_else(|| CudaError::KernelNotFound("gru_gates_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        gates_ih,
                        gates_hh,
                        h_prev,
                        h_new,
                        hidden_size as u32,
                        total as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    // =========================================================================
    // Fused BatchNorm Forward Kernels
    // =========================================================================

    /// BatchNorm pass 1: compute per-channel sum and sum_sq via atomics.
    pub fn batchnorm_stats_f32(
        &self,
        x: &CudaSlice<f32>,
        sum_out: &mut CudaSlice<f32>,
        sum_sq_out: &mut CudaSlice<f32>,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("batchnorm_stats_f32")
            .ok_or_else(|| CudaError::KernelNotFound("batchnorm_stats_f32".to_string()))?;
        let total = n * c * spatial;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (x, sum_out, sum_sq_out, n as u32, c as u32, spatial as u32),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// BatchNorm pass 2: normalize + affine transform using pre-computed mean/var.
    pub fn batchnorm_norm_f32(
        &self,
        x: &CudaSlice<f32>,
        mean: &CudaSlice<f32>,
        var: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        eps: f32,
        c: usize,
        spatial: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("batchnorm_norm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("batchnorm_norm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        x,
                        mean,
                        var,
                        gamma,
                        beta,
                        y,
                        eps,
                        c as u32,
                        spatial as u32,
                        total as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }
}

// =============================================================================
// Conv2d GPU Operations (im2col + GEMM)
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Launch the GPU im2col kernel.
    ///
    /// Unfolds one batch element's input patches into a column matrix.
    /// - `input`: device buffer for one batch element [C_in, H, W]
    /// - `col`: output device buffer [C_in*kH*kW, out_H*out_W]
    /// - `params`: device buffer with u32[10] = {H, W, kH, kW, pH, pW, sH, sW, oH, oW}
    pub fn im2col_f32(
        &self,
        input: &CudaSlice<f32>,
        col: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("im2col_f32")
            .ok_or_else(|| CudaError::KernelNotFound("im2col_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (input, col, params, n as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the GPU col2im kernel (reverse of im2col).
    ///
    /// Scatters column matrix back to input spatial positions using atomicAdd.
    /// The output buffer MUST be zero-initialized before calling this.
    pub fn col2im_f32(
        &self,
        col: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("col2im_f32")
            .ok_or_else(|| CudaError::KernelNotFound("col2im_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (col, output, params, n as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the GPU bias_add_channels kernel (in-place).
    ///
    /// Adds bias per output channel: data[i] += bias[i / spatial_size]
    pub fn bias_add_channels_f32(
        &self,
        data: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        spatial: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("bias_add_channels_f32")
            .ok_or_else(|| CudaError::KernelNotFound("bias_add_channels_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            func.clone()
                .launch(cfg, (data, bias, spatial as u32, n as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Full GPU conv2d forward: im2col on GPU → cuBLAS GEMM → bias add on GPU.
    ///
    /// Handles groups=1 only. Returns output as flat Vec<f32> in NCHW layout.
    /// Returns None if any GPU operation fails (caller falls back to CPU).
    pub fn conv2d_forward(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        batch_size: usize,
        in_channels: usize,
        in_height: usize,
        in_width: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Option<Vec<f32>> {
        let out_h = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        let col_h = in_channels * kernel_h * kernel_w;
        let col_w = out_h * out_w;
        let col_n = col_h * col_w;
        let spatial = out_h * out_w;
        let out_per_batch = out_channels * spatial;
        let in_per_batch = in_channels * in_height * in_width;

        use super::cuda_pool::pool_alloc;

        // Upload weight [out_channels, col_h] to GPU (once for all batches)
        let weight_gpu = self.htod_copy(weight).ok()?;

        // Upload bias if present
        let bias_gpu = bias.and_then(|b| self.htod_copy(b).ok());

        // Upload im2col parameters as u32 buffer (reused across batches)
        let im2col_params: [u32; 10] = [
            in_height as u32,
            in_width as u32,
            kernel_h as u32,
            kernel_w as u32,
            pad_h as u32,
            pad_w as u32,
            stride_h as u32,
            stride_w as u32,
            out_h as u32,
            out_w as u32,
        ];
        let params_gpu = self.htod_copy(&im2col_params[..]).ok()?;

        // Pool-allocate col buffer on GPU (reused across batches)
        let mut col_gpu = pool_alloc(col_n).ok()?;

        // Pool-allocate output buffer on GPU
        let mut batch_out_gpu = pool_alloc(out_per_batch).ok()?;

        let mut output = vec![0.0f32; batch_size * out_per_batch];

        for b in 0..batch_size {
            // Upload input for this batch element
            let input_slice = &input[b * in_per_batch..(b + 1) * in_per_batch];
            let input_gpu = self.htod_copy(input_slice).ok()?;

            // GPU im2col: input [C_in, H, W] → col [col_h, col_w]
            self.im2col_f32(&input_gpu, &mut col_gpu, &params_gpu, col_n)
                .ok()?;

            // GPU GEMM: out = weight @ col
            // weight: [out_channels, col_h] (row-major)
            // col: [col_h, col_w] (row-major)
            // result: [out_channels, col_w] (row-major)
            //
            // cuBLAS column-major: C^T = B^T @ A^T
            // m=col_w, n=out_channels, k=col_h
            self.gemm_f32(
                false,
                false,
                col_w,
                out_channels,
                col_h,
                1.0,
                &col_gpu,
                col_w,
                &weight_gpu,
                col_h,
                0.0,
                &mut batch_out_gpu,
                col_w,
            )
            .ok()?;

            // GPU bias add (in-place on batch_out_gpu)
            if let Some(ref bg) = bias_gpu {
                self.bias_add_channels_f32(&mut batch_out_gpu, bg, spatial, out_per_batch)
                    .ok()?;
            }

            // Download output for this batch
            let batch_result = self.dtoh_copy(&batch_out_gpu).ok()?;
            output[b * out_per_batch..(b + 1) * out_per_batch]
                .copy_from_slice(&batch_result[..out_per_batch]);
        }

        Some(output)
    }
}

/// Public GPU conv2d forward — callable from other crates.
///
/// Returns Some(output_vec) on success, None if CUDA unavailable or operation fails.
/// Only handles groups=1. Caller should fall back to CPU for grouped convolution.
#[cfg(feature = "cuda")]
pub fn cuda_conv2d_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch_size: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Option<Vec<f32>> {
    let cuda = get_cuda_backend()?;
    cuda.conv2d_forward(
        input,
        weight,
        bias,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    )
}

/// Stub when CUDA feature is disabled.
#[cfg(not(feature = "cuda"))]
pub fn cuda_conv2d_forward(
    _input: &[f32],
    _weight: &[f32],
    _bias: Option<&[f32]>,
    _batch_size: usize,
    _in_channels: usize,
    _in_height: usize,
    _in_width: usize,
    _out_channels: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _stride_h: usize,
    _stride_w: usize,
    _pad_h: usize,
    _pad_w: usize,
) -> Option<Vec<f32>> {
    None
}

// =============================================================================
// Pooling GPU Operations (MaxPool2d + AvgPool2d)
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Launch MaxPool2d forward kernel on GPU (device-resident).
    ///
    /// - `input`: GPU slice [N*C*H*W]
    /// - `output`: GPU slice [N*C*out_h*out_w] (pre-allocated, zero-init)
    /// - `indices`: GPU slice [N*C*out_h*out_w] (pre-allocated, i32)
    /// - `params`: GPU u32[8] = {H, W, kH, kW, sH, sW, pH, pW}
    pub fn maxpool2d_fwd_f32(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        params: &CudaSlice<u32>,
        channels: usize,
        out_h: usize,
        out_w: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("maxpool2d_fwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("maxpool2d_fwd_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        input,
                        output,
                        indices,
                        params,
                        channels as u32,
                        out_h as u32,
                        out_w as u32,
                        total as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch MaxPool2d backward kernel on GPU (device-resident).
    ///
    /// Scatters grad_output to grad_input at max index positions using atomicAdd.
    /// `grad_input` must be zero-initialized.
    pub fn maxpool2d_bwd_f32(
        &self,
        grad_output: &CudaSlice<f32>,
        indices: &CudaSlice<i32>,
        grad_input: &mut CudaSlice<f32>,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("maxpool2d_bwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("maxpool2d_bwd_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(cfg, (grad_output, indices, grad_input, total as u32))
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch AvgPool2d forward kernel on GPU (device-resident).
    ///
    /// - `params`: GPU u32[9] = {H, W, kH, kW, sH, sW, pH, pW, count_include_pad}
    pub fn avgpool2d_fwd_f32(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        channels: usize,
        out_h: usize,
        out_w: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("avgpool2d_fwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("avgpool2d_fwd_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        input,
                        output,
                        params,
                        channels as u32,
                        out_h as u32,
                        out_w as u32,
                        total as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch AvgPool2d backward kernel on GPU (device-resident).
    ///
    /// `grad_input` must be zero-initialized.
    pub fn avgpool2d_bwd_f32(
        &self,
        grad_output: &CudaSlice<f32>,
        grad_input: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        channels: usize,
        out_h: usize,
        out_w: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("avgpool2d_bwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("avgpool2d_bwd_f32".to_string()))?;

        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        grad_output,
                        grad_input,
                        params,
                        channels as u32,
                        out_h as u32,
                        out_w as u32,
                        total as u32,
                    ),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }
}

// =============================================================================
// Pinned (Page-Locked) Host Memory
// =============================================================================

/// A page-locked (pinned) host memory buffer for fast CPU-to-GPU transfers.
///
/// Pinned memory is allocated via `cuMemAllocHost` and is not subject to
/// OS paging, enabling the GPU to DMA directly from the host buffer. This
/// typically provides 2-3x faster host-to-device transfer compared to
/// pageable (regular) memory.
///
/// # Usage
/// ```ignore
/// use axonml_core::backends::cuda::PinnedBuffer;
///
/// let data = vec![1.0f32; 1024];
/// let pinned = PinnedBuffer::from_slice(&data).expect("pin failed");
/// // Use pinned.as_slice() as the source for htod transfers
/// ```
#[cfg(feature = "cuda")]
pub struct PinnedBuffer {
    /// Raw pointer to the pinned host allocation (from cuMemAllocHost).
    ptr: *mut f32,
    /// Number of f32 elements in the buffer.
    len: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for PinnedBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedBuffer {}

#[cfg(feature = "cuda")]
impl PinnedBuffer {
    /// Allocates a pinned host buffer and copies `data` into it.
    ///
    /// The returned buffer can be used as a source for fast CPU-to-GPU
    /// transfers. The memory is page-locked so the GPU can DMA from it
    /// without going through the OS page cache.
    ///
    /// # Errors
    /// Returns `CudaError` if pinned memory allocation fails (e.g., out of
    /// lockable memory, CUDA not initialized).
    pub fn from_slice(data: &[f32]) -> Result<Self, CudaError> {
        use cudarc::driver::sys::lib;
        use std::ptr;

        if data.is_empty() {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
            });
        }

        let byte_size = data.len() * std::mem::size_of::<f32>();
        let mut host_ptr: *mut std::ffi::c_void = ptr::null_mut();

        // Ensure CUDA is initialized before calling driver API
        let _ = get_cuda_backend().ok_or(CudaError::DeviceNotFound)?;

        unsafe {
            let result = lib().cuMemAllocHost_v2(&mut host_ptr, byte_size);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::AllocationFailed);
            }

            // Copy data into pinned buffer
            ptr::copy_nonoverlapping(data.as_ptr(), host_ptr as *mut f32, data.len());
        }

        Ok(Self {
            ptr: host_ptr as *mut f32,
            len: data.len(),
        })
    }

    /// Allocates an uninitialized pinned host buffer of the given length.
    ///
    /// # Safety
    /// The contents are uninitialized. Caller must write to the buffer
    /// before reading from it.
    ///
    /// # Errors
    /// Returns `CudaError` if pinned memory allocation fails.
    pub fn alloc(len: usize) -> Result<Self, CudaError> {
        use cudarc::driver::sys::lib;
        use std::ptr;

        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
            });
        }

        let byte_size = len * std::mem::size_of::<f32>();
        let mut host_ptr: *mut std::ffi::c_void = ptr::null_mut();

        let _ = get_cuda_backend().ok_or(CudaError::DeviceNotFound)?;

        unsafe {
            let result = lib().cuMemAllocHost_v2(&mut host_ptr, byte_size);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::AllocationFailed);
            }
        }

        Ok(Self {
            ptr: host_ptr as *mut f32,
            len,
        })
    }

    /// Returns a slice view of the pinned buffer.
    pub fn as_slice(&self) -> &[f32] {
        if self.ptr.is_null() || self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice view of the pinned buffer.
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        if self.ptr.is_null() || self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw host pointer.
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    /// Returns a mutable raw host pointer.
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// Transfers the pinned buffer contents to a GPU `CudaSlice`.
    ///
    /// This is the fast path: since the source memory is pinned, the GPU
    /// can DMA directly without staging through pageable memory.
    pub fn to_gpu(&self) -> Result<CudaSlice<f32>, CudaError> {
        let backend = get_cuda_backend().ok_or(CudaError::DeviceNotFound)?;
        backend.htod_copy(self.as_slice())
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let lib = cudarc::driver::sys::lib();
                let _ = lib.cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Convenience function: allocate pinned host memory and copy data into it.
///
/// This is a shorthand for `PinnedBuffer::from_slice(data)`.
///
/// # Errors
/// Returns `CudaError` if CUDA is not available or allocation fails.
#[cfg(feature = "cuda")]
pub fn pin_memory(data: &[f32]) -> Result<PinnedBuffer, CudaError> {
    PinnedBuffer::from_slice(data)
}

/// Stub when CUDA is not enabled - pinned memory is not available.
#[cfg(not(feature = "cuda"))]
pub fn pin_memory(_data: &[f32]) -> Result<(), CudaError> {
    Err(CudaError::DeviceNotFound)
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

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_add_kernel() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let a_gpu = backend.htod_copy(&a).unwrap();
        let b_gpu = backend.htod_copy(&b).unwrap();
        let mut c_gpu = backend.alloc::<f32>(4).unwrap();

        backend.add_f32(&mut c_gpu, &a_gpu, &b_gpu, 4).unwrap();

        let result = backend.dtoh_copy(&c_gpu).unwrap();
        assert!((result[0] - 6.0).abs() < 1e-5);
        assert!((result[1] - 8.0).abs() < 1e-5);
        assert!((result[2] - 10.0).abs() < 1e-5);
        assert!((result[3] - 12.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_mul_kernel() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];

        let a_gpu = backend.htod_copy(&a).unwrap();
        let b_gpu = backend.htod_copy(&b).unwrap();
        let mut c_gpu = backend.alloc::<f32>(4).unwrap();

        backend.mul_f32(&mut c_gpu, &a_gpu, &b_gpu, 4).unwrap();

        let result = backend.dtoh_copy(&c_gpu).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 6.0).abs() < 1e-5);
        assert!((result[2] - 12.0).abs() < 1e-5);
        assert!((result[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_scale_kernel() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut data_gpu = backend.htod_copy(&data).unwrap();

        backend.scale_f32(&mut data_gpu, 2.5, 4).unwrap();

        let result = backend.dtoh_copy(&data_gpu).unwrap();
        assert!((result[0] - 2.5).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
        assert!((result[2] - 7.5).abs() < 1e-5);
        assert!((result[3] - 10.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_relu_kernel() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input_gpu = backend.htod_copy(&input).unwrap();
        let mut output_gpu = backend.alloc::<f32>(5).unwrap();

        backend.relu_f32(&mut output_gpu, &input_gpu, 5).unwrap();

        let result = backend.dtoh_copy(&output_gpu).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 1.0).abs() < 1e-5);
        assert!((result[4] - 2.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_sigmoid_kernel() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        let input: Vec<f32> = vec![0.0, 1.0, -1.0];
        let input_gpu = backend.htod_copy(&input).unwrap();
        let mut output_gpu = backend.alloc::<f32>(3).unwrap();

        backend.sigmoid_f32(&mut output_gpu, &input_gpu, 3).unwrap();

        let result = backend.dtoh_copy(&output_gpu).unwrap();
        // sigmoid(0) = 0.5
        assert!((result[0] - 0.5).abs() < 1e-4);
        // sigmoid(1) ≈ 0.7311
        assert!((result[1] - 0.7311).abs() < 1e-3);
        // sigmoid(-1) ≈ 0.2689
        assert!((result[2] - 0.2689).abs() < 1e-3);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tanh_kernel() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        let input: Vec<f32> = vec![0.0, 1.0, -1.0];
        let input_gpu = backend.htod_copy(&input).unwrap();
        let mut output_gpu = backend.alloc::<f32>(3).unwrap();

        backend.tanh_f32(&mut output_gpu, &input_gpu, 3).unwrap();

        let result = backend.dtoh_copy(&output_gpu).unwrap();
        // tanh(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-5);
        // tanh(1) ≈ 0.7616
        assert!((result[1] - 0.7616).abs() < 1e-3);
        // tanh(-1) ≈ -0.7616
        assert!((result[2] - (-0.7616)).abs() < 1e-3);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_large_tensor_add() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        // Test with a large tensor (1M elements)
        let n = 1_000_000;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let a_gpu = backend.htod_copy(&a).unwrap();
        let b_gpu = backend.htod_copy(&b).unwrap();
        let mut c_gpu = backend.alloc::<f32>(n).unwrap();

        backend.add_f32(&mut c_gpu, &a_gpu, &b_gpu, n).unwrap();

        let result = backend.dtoh_copy(&c_gpu).unwrap();

        // Each element should equal n (i + (n-i) = n)
        assert!((result[0] - n as f32).abs() < 1e-3);
        assert!((result[n / 2] - n as f32).abs() < 1e-3);
        assert!((result[n - 1] - n as f32).abs() < 1e-3);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_conv2d_forward() {
        if !is_available() {
            return;
        }

        // 1x1 conv: 3 in_channels → 2 out_channels, input 4x4
        let input = vec![1.0f32; 1 * 3 * 4 * 4]; // all ones
        let mut weight = vec![0.0f32; 2 * 3 * 1 * 1];
        // out_ch0 = in_ch0 (weight[0]=1), out_ch1 = in_ch1 (weight[4]=1)
        weight[0] = 1.0;
        weight[4] = 1.0;
        let bias = vec![0.5f32; 2];

        let result = cuda_conv2d_forward(
            &input,
            &weight,
            Some(&bias),
            1,
            3,
            4,
            4,
            2,
            1,
            1,
            1,
            1,
            0,
            0,
        );

        let out = result.expect("CUDA conv2d should succeed");
        assert_eq!(out.len(), 2 * 4 * 4);
        // out_ch0 = 1.0*1 + 0.5 = 1.5
        assert!(
            (out[0] - 1.5).abs() < 0.01,
            "1x1 conv ch0: expected 1.5, got {}",
            out[0]
        );
        // out_ch1 = 1.0*1 + 0.5 = 1.5
        assert!(
            (out[16] - 1.5).abs() < 0.01,
            "1x1 conv ch1: expected 1.5, got {}",
            out[16]
        );

        // 3x3 conv with padding=1: all-ones input, all-ones weight
        let input2 = vec![1.0f32; 1 * 3 * 8 * 8];
        let weight2 = vec![1.0f32; 2 * 3 * 3 * 3]; // all 1s → each output = sum of 27 inputs
        let bias2 = vec![0.0f32; 2];

        let result2 = cuda_conv2d_forward(
            &input2,
            &weight2,
            Some(&bias2),
            1,
            3,
            8,
            8,
            2,
            3,
            3,
            1,
            1,
            1,
            1,
        );

        let out2 = result2.expect("CUDA 3x3 conv should succeed");
        assert_eq!(out2.len(), 2 * 8 * 8);
        // Center pixel (row 4, col 4) = 3 channels * 9 kernel positions * 1.0 = 27.0
        let center = 4 * 8 + 4;
        assert!(
            (out2[center] - 27.0).abs() < 0.1,
            "3x3 conv center: expected 27.0, got {}",
            out2[center]
        );
        // Corner pixel (0,0) with pad=1: only 2x2x3 = 12 valid positions
        assert!(
            (out2[0] - 12.0).abs() < 0.1,
            "3x3 conv corner: expected 12.0, got {}",
            out2[0]
        );
    }
}
