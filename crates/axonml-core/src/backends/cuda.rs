//! CUDA backend — 106 public methods for NVIDIA GPU tensor operations.
//!
//! Global `CudaBackend` singleton (via OnceLock) wrapping a cudarc
//! `CudaContext` + `CudaStream`. Exposes cuBLAS GEMM (regular + strided-
//! batched), 15 custom PTX kernel modules (loaded at init via
//! `CudaKernels::load`), elementwise ops (add/mul/scalar/neg/abs),
//! activations (relu/sigmoid/tanh/gelu/silu/elu/leaky_relu/softmax),
//! layernorm, RMSNorm, transpose, embedding gather, dropout, Q4_K/Q6_K
//! dequant-in-shader GEMV+GEMM (cooperative warp reduction), fused flash-
//! decode attention (online softmax, one warp per head, GQA + SWA aware),
//! fused flash-prefill attention (batched causal, one CTA per query×head),
//! and memory management (htod_copy, dtoh_copy, alloc, alloc_uninit).
//!
//! # File
//! `crates/axonml-core/src/backends/cuda.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#[cfg(feature = "cuda")]
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
#[cfg(feature = "cudnn")]
use cudarc::cudnn::Cudnn;
#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};

use super::Backend;
#[cfg(feature = "cuda")]
use super::cuda_kernels::{self, BLOCK_SIZE, CudaKernels};
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
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: CudaKernels,
    #[cfg(feature = "cudnn")]
    cudnn_handle: Option<Arc<Cudnn>>,
}

/// CUDA backend stub when the `cuda` feature is disabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaBackend {
    device_index: usize,
}

// Implement Send and Sync for CudaBackend
// Safe because CudaContext/CudaStream and CudaBlas are internally synchronized
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
        let ctx = CudaContext::new(device_index).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).ok()?;
        let kernels = match CudaKernels::load(ctx.clone()) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[AxonML CUDA] Kernel loading failed: {:?}", e);
                return None;
            }
        };

        #[cfg(feature = "cudnn")]
        let cudnn_handle = match Cudnn::new(stream.clone()) {
            Ok(handle) => {
                eprintln!("[AxonML] cuDNN handle initialized");
                Some(handle)
            }
            Err(e) => {
                eprintln!(
                    "[AxonML CUDA] cuDNN init failed: {:?} (falling back to im2col+GEMM)",
                    e
                );
                None
            }
        };

        Some(Self {
            device_index,
            ctx,
            stream,
            blas,
            kernels,
            #[cfg(feature = "cudnn")]
            cudnn_handle,
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

    /// Returns the underlying CUDA context.
    #[cfg(feature = "cuda")]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the underlying CUDA stream.
    #[cfg(feature = "cuda")]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Returns the cuBLAS handle.
    #[cfg(feature = "cuda")]
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// Returns the cuDNN handle, if available.
    #[cfg(feature = "cudnn")]
    pub fn cudnn(&self) -> Option<&Arc<Cudnn>> {
        self.cudnn_handle.as_ref()
    }

    /// Allocates a typed buffer on the GPU initialized to zeros.
    #[cfg(feature = "cuda")]
    pub fn alloc<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, CudaError> {
        self.stream.alloc_zeros(len).map_err(CudaError::from)
    }

    /// Allocates uninitialized memory on the GPU.
    #[cfg(feature = "cuda")]
    pub fn alloc_uninit<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, CudaError> {
        unsafe { self.stream.alloc(len).map_err(CudaError::from) }
    }

    /// Copies data from host to device.
    #[cfg(feature = "cuda")]
    pub fn htod_copy<T: DeviceRepr>(&self, src: &[T]) -> Result<CudaSlice<T>, CudaError> {
        self.stream.clone_htod(src).map_err(CudaError::from)
    }

    /// Copies data from device to host.
    #[cfg(feature = "cuda")]
    pub fn dtoh_copy<T: DeviceRepr>(&self, src: &CudaSlice<T>) -> Result<Vec<T>, CudaError> {
        self.stream.clone_dtoh(src).map_err(CudaError::from)
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
        match self.stream.alloc_zeros::<u8>(size) {
            Ok(slice) => {
                // Get the raw device pointer via leak
                let ptr = slice.leak() as *mut u8;
                ptr
            }
            Err(_) => std::ptr::null_mut(),
        }
    }

    fn deallocate(&self, ptr: *mut u8, size: usize) {
        if !ptr.is_null() {
            // Reconstruct the CudaSlice to properly free
            unsafe {
                let slice: CudaSlice<u8> = self
                    .stream
                    .upgrade_device_ptr(ptr as cudarc::driver::sys::CUdeviceptr, size);
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
            let _ = cudarc::driver::result::memcpy_htod_sync(
                dst as cudarc::driver::sys::CUdeviceptr,
                src_slice,
            );
        }
    }

    fn copy_to_host(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let dst_slice = std::slice::from_raw_parts_mut(dst, size);
            let _ = cudarc::driver::result::memcpy_dtoh_sync(
                dst_slice,
                src as cudarc::driver::sys::CUdeviceptr,
            );
        }
    }

    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        if dst.is_null() || src.is_null() || size == 0 {
            return;
        }
        unsafe {
            let _ = cudarc::driver::result::memcpy_dtod_sync(
                dst as cudarc::driver::sys::CUdeviceptr,
                src as cudarc::driver::sys::CUdeviceptr,
                size,
            );
        }
    }

    fn synchronize(&self) {
        let _ = self.stream.synchronize();
    }
}

/// Synchronize the CUDA device (wait for all GPU operations to complete).
/// Returns true if sync was performed, false if CUDA is not available.
#[cfg(feature = "cuda")]
pub fn cuda_sync() -> bool {
    if let Some(backend) = get_cuda_backend() {
        let _ = backend.stream.synchronize();
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
        CudaContext::new(0).is_ok()
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
    ///
    /// Uses raw cublasSgemm_v2 FFI to avoid GemmConfig abstraction issues
    /// with cuBLAS 12.9+ on Blackwell GPUs.
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
        use cudarc::cublas::result::sgemm;
        use cudarc::driver::DevicePtr as _;
        use cudarc::driver::DevicePtrMut as _;

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

        let (a_ptr, _ga) = a.device_ptr(&self.stream);
        let (b_ptr, _gb) = b.device_ptr(&self.stream);
        let (c_ptr, _gc) = c.device_ptr_mut(&self.stream);

        unsafe {
            sgemm(
                *self.blas.handle(),
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32,
                a_ptr as *const f32,
                lda as i32,
                b_ptr as *const f32,
                ldb as i32,
                &beta as *const f32,
                c_ptr as *mut f32,
                ldc as i32,
            )
            .map_err(CudaError::from)
        }
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
        use cudarc::driver::DevicePtr as _;
        use cudarc::driver::DevicePtrMut as _;

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

        let (a_devptr, _ga) = a.device_ptr(&self.stream);
        let (b_devptr, _gb) = b.device_ptr(&self.stream);
        let (c_devptr, _gc) = c.device_ptr_mut(&self.stream);
        let a_ptr = a_devptr as *const f32;
        let b_ptr = b_devptr as *const f32;
        let c_ptr = c_devptr as *mut f32;

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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(dst)
                .arg(&alpha)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q4_K GEMM: `c = a @ B^T` where `a` is `[m, in]` f32, B is device-side
    /// Q4_K bytes (physical shape `[out, in]`), and `c` is `[m, out]` f32.
    /// One thread per output element. Prefill uses this; decode uses GEMV.
    pub fn q4k_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q4_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemm_f32".to_string()))?;

        let total = m_dim * out_dim;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q4_K GEMM order-matched to `q4k_gemv_f32` — bit-identical output.
    pub fn q4k_gemm_matched_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q4_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemm_matched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemm_matched_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid_x = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, m_dim as u32, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q4_K GEMV: `c = a @ B^T` where B is stored on-device as Q4_K super-blocks.
    ///
    /// Shapes (all row-major):
    ///   - `w`: `out * in / 256 * 144` bytes of raw Q4_K data (physical `[out, in]` layout).
    ///   - `a`: f32 slice of length `in`.
    ///   - `c`: f32 slice of length `out`.
    ///
    /// Requirements:
    ///   - `in` must be a multiple of 256 (Q4_K super-block size).
    ///
    /// Each thread owns one output element and iterates `in / 256` blocks of its
    /// row of B, dequanting each block in registers. See `q4k_matmul.cu`.
    pub fn q4k_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q4_K GEMV requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemv_f32".to_string()))?;

        // v2 layout: `ROWS_PER_CTA` output rows per CTA × 2 warps/row × 32
        // threads = 64 * ROWS_PER_CTA threads/CTA. Two warps cooperate on
        // each output row (each handles half the super-blocks) and combine
        // their partial sums through shared memory. Vectorized qs (uint32)
        // and activation (float4) loads inside the warp.
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused Q4_K GEMV for QKV projections — one kernel launch produces
    /// Q, K, and V outputs from a shared input activation. Each weight
    /// matrix has the same input dimension but its own output dimension.
    /// Used by the nexus-serve decode path to collapse the three Q/K/V
    /// kernel launches per layer into a single grid.
    #[allow(clippy::too_many_arguments)]
    pub fn q4k_gemv_fused_qkv_f32(
        &self,
        q_w: &CudaSlice<u8>,
        k_w: &CudaSlice<u8>,
        v_w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        q_c: &mut CudaSlice<f32>,
        k_c: &mut CudaSlice<f32>,
        v_c: &mut CudaSlice<f32>,
        q_out: usize,
        k_out: usize,
        v_out: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            in_dim % 256 == 0,
            "fused QKV GEMV requires in_dim % 256 == 0"
        );
        let func = self
            .kernels
            .get("q4k_gemv_fused_qkv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemv_fused_qkv_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let total_out = (q_out + k_out + v_out) as u32;
        let grid = (total_out + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q_w)
                .arg(k_w)
                .arg(v_w)
                .arg(a)
                .arg(q_c)
                .arg(k_c)
                .arg(v_c)
                .arg(&(q_out as u32))
                .arg(&(k_out as u32))
                .arg(&(v_out as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused Q4_K QKV GEMV with per-section bias add at the output write.
    /// Extends `q4k_gemv_fused_qkv_f32` to absorb the three separate
    /// bias-add kernel launches per layer that Qwen2 / DeepSeek require
    /// (Qwen2 adds bias to Q, K, AND V projections). Saves three
    /// host→GPU launch cycles per layer.
    ///
    /// The bias buffers are mandatory — a caller without biases should
    /// route through `q4k_gemv_fused_qkv_f32` instead. Launch geometry
    /// matches the no-bias variant.
    #[allow(clippy::too_many_arguments)]
    pub fn q4k_gemv_fused_qkv_bias_f32(
        &self,
        q_w: &CudaSlice<u8>,
        k_w: &CudaSlice<u8>,
        v_w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        q_bias: &CudaSlice<f32>,
        k_bias: &CudaSlice<f32>,
        v_bias: &CudaSlice<f32>,
        q_c: &mut CudaSlice<f32>,
        k_c: &mut CudaSlice<f32>,
        v_c: &mut CudaSlice<f32>,
        q_out: usize,
        k_out: usize,
        v_out: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            in_dim % 256 == 0,
            "fused QKV+bias GEMV requires in_dim % 256 == 0"
        );
        let func = self
            .kernels
            .get("q4k_gemv_fused_qkv_bias_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemv_fused_qkv_bias_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        const STAGE_MAX_FLOATS: u32 = 8192;
        let _stage_cap = STAGE_MAX_FLOATS; // kept for parity; unused here
        let total_out = (q_out + k_out + v_out) as u32;
        let grid = (total_out + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let reduction_bytes = 8u32 * std::mem::size_of::<f32>() as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: reduction_bytes,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q_w)
                .arg(k_w)
                .arg(v_w)
                .arg(a)
                .arg(q_bias)
                .arg(k_bias)
                .arg(v_bias)
                .arg(q_c)
                .arg(k_c)
                .arg(v_c)
                .arg(&(q_out as u32))
                .arg(&(k_out as u32))
                .arg(&(v_out as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused Q4_K GEMV for SwiGLU / ReLU² gate+up projections — one kernel
    /// launch produces both outputs from a shared input activation. Both
    /// projections have the same `intermediate_size` output dimension.
    pub fn q4k_gemv_fused_gate_up_f32(
        &self,
        gate_w: &CudaSlice<u8>,
        up_w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        gate_c: &mut CudaSlice<f32>,
        up_c: &mut CudaSlice<f32>,
        inter: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            in_dim % 256 == 0,
            "fused gate/up GEMV requires in_dim % 256 == 0"
        );
        let func = self
            .kernels
            .get("q4k_gemv_fused_gate_up_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemv_fused_gate_up_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let total_out = (inter * 2) as u32;
        let grid = (total_out + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(gate_w)
                .arg(up_w)
                .arg(a)
                .arg(gate_c)
                .arg(up_c)
                .arg(&(inter as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused Q4_K GEMV + residual add: `x_out[j] = x_in[j] + matmul(a, w_j)`.
    /// Absorbs the matmul + element-wise residual_add into one kernel — one
    /// fewer launch per residual site and no intermediate projection buffer
    /// round trip.
    pub fn q4k_gemv_residual_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        x_in: &CudaSlice<f32>,
        x_out: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "q4k_gemv_residual: in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemv_residual_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemv_residual_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(x_in)
                .arg(x_out)
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused Q4_K gate/up + SwiGLU: writes `ffn[j] = silu(gate_row_j · a) *
    /// (up_row_j · a)` directly. Saves the two gate_c / up_c intermediate
    /// buffers and the separate swiglu launch.
    pub fn q4k_gemv_fused_gate_up_swiglu_f32(
        &self,
        gate_w: &CudaSlice<u8>,
        up_w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        ffn: &mut CudaSlice<f32>,
        inter: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "fused gate/up+swiglu: in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemv_fused_gate_up_swiglu_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("q4k_gemv_fused_gate_up_swiglu_f32".to_string())
            })?;

        // 4 warps per output row (2 gate + 2 up).
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 4;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((inter as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 4 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(gate_w)
                .arg(up_w)
                .arg(a)
                .arg(ffn)
                .arg(&(inter as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q6_K GEMM: `c = a @ B^T` where B is stored on-device as Q6_K super-blocks.
    /// Mirrors the Q4_K GEMM launcher — see `q6k_matmul.cu`.
    pub fn q6k_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q6_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q6k_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q6k_gemm_f32".to_string()))?;

        let total = m_dim * out_dim;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q6_K GEMM order-matched to `q6k_gemv_f32` — bit-identical output.
    pub fn q6k_gemm_matched_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q6_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q6k_gemm_matched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q6k_gemm_matched_f32".to_string()))?;
        const WARPS_PER_CTA: u32 = 4;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid_x = ((out_dim as u32) + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, m_dim as u32, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q5_K GEMV: `c = a @ B^T` with B stored on-device as Q5_K super-blocks
    /// (176 bytes per 256-element block). Same warp-cooperative layout as
    /// Q6_K — one warp per output row, 32 lanes handle 8 weights each per
    /// block. See `q5k_matmul.cu`.
    pub fn q5k_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q5_K GEMV requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q5k_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5k_gemv_f32".to_string()))?;

        // v2 layout: ROWS_PER_CTA output rows × 2 warps/row × 32 threads =
        // 64 * ROWS_PER_CTA threads per CTA. Two warps cooperate on each
        // output row (each handles half the super-blocks) and combine
        // partial sums through shared memory. Vectorized qs+qh (uint32)
        // and activation (float4) loads inside each warp. Same shape as
        // q4k_gemv_f32 launcher for consistency.
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q5_K fused QKV GEMV — one launch computes Q, K, V projections
    /// sharing the same activation. Phi-3's `attn_qkv` is Q5_K with
    /// split `[3072, 3072, 3072]`; this collapses 3 kernel launches
    /// per layer into 1.
    #[allow(clippy::too_many_arguments)]
    pub fn q5k_gemv_fused_qkv_f32(
        &self,
        q_w: &CudaSlice<u8>,
        k_w: &CudaSlice<u8>,
        v_w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        q_c: &mut CudaSlice<f32>,
        k_c: &mut CudaSlice<f32>,
        v_c: &mut CudaSlice<f32>,
        q_out: usize,
        k_out: usize,
        v_out: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            in_dim % 256 == 0,
            "fused QKV Q5_K GEMV requires in_dim % 256 == 0"
        );
        let func = self
            .kernels
            .get("q5k_gemv_fused_qkv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5k_gemv_fused_qkv_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let total_out = (q_out + k_out + v_out) as u32;
        let grid = (total_out + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q_w)
                .arg(k_w)
                .arg(v_w)
                .arg(a)
                .arg(q_c)
                .arg(k_c)
                .arg(v_c)
                .arg(&(q_out as u32))
                .arg(&(k_out as u32))
                .arg(&(v_out as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q5_K GEMM: `c = a @ B^T` where a is `[m, in]` f32 and B is Q5_K
    /// `[out, in]`. One thread per output element — naive but correct;
    /// the GEMV path (m=1) above is the hot decode case. See
    /// `q5k_matmul.cu`.
    pub fn q5k_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q5_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q5k_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5k_gemm_f32".to_string()))?;

        let total = m_dim * out_dim;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q5_K GEMM order-matched to `q5k_gemv_f32` — bit-identical output
    /// via 2D grid over `mi` dimension. Required for Phi-3 batched
    /// prefill K/V to match decode's single-query K/V.
    pub fn q5k_gemm_matched_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q5_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q5k_gemm_matched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5k_gemm_matched_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid_x = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let grid_y = m_dim as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q5_0 GEMV — one warp per output row, block_size=32, signed 5-bit
    /// `((lo | hi*16) - 16) * d`. Used by legacy Falcon's `attn_output`,
    /// `ffn_up`, and `token_embd` on Falcon-7B Q4_K_M exports.
    pub fn q5_0_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 32 == 0, "Q5_0 GEMV requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_0_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_0_gemv_f32".to_string()))?;
        // v2: two warps per row (split block range), rows_per_cta=4.
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q5_0 GEMM (m > 1 prefill). Naive one-thread-per-output.
    pub fn q5_0_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 32 == 0, "Q5_0 GEMM requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_0_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_0_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m_dim * out_dim);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q5_1 GEMV — unsigned 5-bit `(lo | hi*16) * d + m`. Used by legacy
    /// Falcon's `attn_qkv` (single merged Q/K/V tensor).
    pub fn q5_1_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 32 == 0, "Q5_1 GEMV requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_1_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_1_gemv_f32".to_string()))?;
        // v2: two warps per row (split block range), rows_per_cta=4.
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q5_1 GEMM (m > 1 prefill).
    pub fn q5_1_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 32 == 0, "Q5_1 GEMM requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_1_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_1_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m_dim * out_dim);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q5_1 fused QKV GEMV — one launch computes Q, K, V from a shared
    /// activation. Primary target: Falcon-7B's attn_qkv (MQA, K/V each
    /// only 64 rows — too small to fill the GPU on their own).
    pub fn q5_1_gemv_fused_qkv_f32(
        &self,
        q_w: &CudaSlice<u8>,
        k_w: &CudaSlice<u8>,
        v_w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        q_c: &mut CudaSlice<f32>,
        k_c: &mut CudaSlice<f32>,
        v_c: &mut CudaSlice<f32>,
        q_out: usize,
        k_out: usize,
        v_out: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            in_dim % 32 == 0,
            "fused QKV Q5_1 GEMV requires in_dim % 32 == 0"
        );
        let func = self
            .kernels
            .get("q5_1_gemv_fused_qkv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_1_gemv_fused_qkv_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let total_out = (q_out + k_out + v_out) as u32;
        let grid = (total_out + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q_w)
                .arg(k_w)
                .arg(v_w)
                .arg(a)
                .arg(q_c)
                .arg(k_c)
                .arg(v_c)
                .arg(&(q_out as u32))
                .arg(&(k_out as u32))
                .arg(&(v_out as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Q8_0 GEMV — same v2 two-warp-per-row layout as Q5_0. 34-byte block
    /// (f16 scale + 32 signed int8 quants). Primary consumer: Falcon-7B's
    /// Q8_0 LM head (4544 × 65024), which otherwise falls through to
    /// `cpu_dequant_matmul` on every decode token.
    pub fn q8_0_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 32 == 0, "Q8_0 GEMV requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q8_0_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q8_0_gemv_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((out_dim as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q8_0 GEMM (m > 1 prefill). Naive one-thread-per-output.
    pub fn q8_0_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m_dim: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 32 == 0, "Q8_0 GEMM requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q8_0_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q8_0_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m_dim * out_dim);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m_dim as u32))
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// BitNet I2_S (1.58-bit ternary) GEMV. Two warps per output row, each
    /// walks half the block range. Tensor-wide f32 scale passed separately
    /// (GGUF stores it in the last 4 bytes of the raw tensor buffer, but the
    /// GPU-side buffer holds packed bytes only — scale is sourced once at
    /// load time and passed here each call).
    pub fn i2s_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        scale: f32,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(k % 128 == 0, "I2_S GEMV requires k % 128 == 0");
        let func = self
            .kernels
            .get("i2s_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("i2s_gemv_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((n as u32) + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&scale)
                .arg(&(n as u32))
                .arg(&(k as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// BitNet I2_S GEMM (m > 1 prefill). Naive one-thread-per-output.
    pub fn i2s_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(k % 128 == 0, "I2_S GEMM requires k % 128 == 0");
        let func = self
            .kernels
            .get("i2s_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("i2s_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m * n);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&scale)
                .arg(&(m as u32))
                .arg(&(n as u32))
                .arg(&(k as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q6_K GEMV: one WARP per output row, lanes cooperate on each block.
    /// See `q6k_matmul.cu`.
    pub fn q6k_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(in_dim % 256 == 0, "Q6_K GEMV requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q6k_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q6k_gemv_f32".to_string()))?;

        const WARPS_PER_CTA: u32 = 4;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = ((out_dim as u32) + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(out_dim as u32))
                .arg(&(in_dim as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(b_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(b_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(b_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(b_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(a_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(a_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(a_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(a_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(&exp)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused SiLU backward. Replaces the 7-op chain in `SiluBackward::apply`
    /// (sigmoid + ones-H2D + sub + mul + add + mul + mul) with a single kernel
    /// launch — one pool_alloc, zero H2D, zero intermediate tensors.
    ///
    /// Math: grad_input[i] = grad_output[i] * σ(x[i]) * (1 + x[i] * (1 - σ(x[i])))
    pub fn silu_backward_f32(
        &self,
        grad_input: &mut CudaSlice<f32>,
        saved_input: &CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("silu_backward_f32")
            .ok_or_else(|| CudaError::KernelNotFound("silu_backward_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(saved_input)
                .arg(grad_output)
                .arg(grad_input)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(&scalar)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_output)
                .arg(input)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_output)
                .arg(output)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_output)
                .arg(output)
                .arg(dst)
                .arg(&(len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(outer_size as u32))
                .arg(&(dim_size as u32))
                .arg(&(inner_size as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(data)
                .arg(&(num_rows as u32))
                .arg(&(row_size as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(n as u32))
                .arg(&(src_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(gamma)
                .arg(beta)
                .arg(dst)
                .arg(&(norm_size as u32))
                .arg(&eps)
                .arg(&(num_rows as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(softmax_output)
                .arg(grad_output)
                .arg(dst)
                .arg(&(num_rows as u32))
                .arg(&(row_size as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_output)
                .arg(input)
                .arg(gamma)
                .arg(d_input)
                .arg(&(norm_size as u32))
                .arg(&eps)
                .arg(&(num_rows as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_output)
                .arg(input)
                .arg(d_weight)
                .arg(d_bias)
                .arg(&(norm_size as u32))
                .arg(&eps)
                .arg(&(num_rows as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(indices)
                .arg(dst)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_src)
                .arg(indices)
                .arg(weight_grad)
                .arg(&(total_n as u32))
                .arg(&(emb_dim as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(param)
                .arg(grad)
                .arg(exp_avg)
                .arg(exp_avg_sq)
                .arg(&(n as u32))
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&eps)
                .arg(&weight_decay)
                .arg(&bias_correction1)
                .arg(&bias_correction2)
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(data)
                .arg(output)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(data)
                .arg(&(n as u32))
                .arg(&scale)
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(logits)
                .arg(targets)
                .arg(losses)
                .arg(softmax_out)
                .arg(&(num_classes as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(softmax_probs)
                .arg(targets)
                .arg(grad_output)
                .arg(grad_input)
                .arg(&(batch_size as u32))
                .arg(&(num_classes as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Zero-fills a GPU allocation using cudaMemset.
    #[cfg(feature = "cuda")]
    pub fn memset_zeros_f32(&self, dst: &mut CudaSlice<f32>) -> Result<(), CudaError> {
        self.stream
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
        use cudarc::driver::DevicePtr as _;
        let (src_ptr, _guard_s) = src.device_ptr(&self.stream);
        let src_ptr =
            src_ptr + (src_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
        use cudarc::driver::DevicePtrMut as _;
        let (dst_ptr, _guard_d) = dst.device_ptr_mut(&self.stream);
        let dst_ptr =
            dst_ptr + (dst_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
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
            self.stream
                .launch_builder(func)
                .arg(mask)
                .arg(output)
                .arg(&(total_n as u32))
                .arg(&(tgt_len as u32))
                .arg(&(src_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(mask)
                .arg(output)
                .arg(&(total_n as u32))
                .arg(&(num_heads as u32))
                .arg(&(tgt_len as u32))
                .arg(&(src_len as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(strides)
                .arg(shape)
                .arg(&(ndim as u32))
                .arg(&(offset as u32))
                .arg(&(total_n as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(gates)
                .arg(c_prev)
                .arg(h_new)
                .arg(c_new)
                .arg(&(hidden_size as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    // =========================================================================
    // Fused LSTM Gate Backward Kernel
    // =========================================================================

    /// Fused LSTM gate backward computation on GPU.
    ///
    /// Given saved forward state and incoming gradients, computes gate gradients
    /// and cell gradient to previous timestep in a single kernel launch.
    ///
    /// - `gates`: [batch, 4*hidden] pre-activation gates from forward
    /// - `c_prev`: [batch, hidden] previous cell state
    /// - `c_new`: [batch, hidden] cell state from forward
    /// - `grad_h`: [batch, hidden] gradient from output
    /// - `grad_c_next`: [batch, hidden] gradient from next timestep cell
    /// - `grad_gates`: [batch, 4*hidden] output gate gradients
    /// - `grad_c_prev`: [batch, hidden] output cell gradient to prev timestep
    pub fn lstm_gates_backward_f32(
        &self,
        gates: &CudaSlice<f32>,
        c_prev: &CudaSlice<f32>,
        c_new: &CudaSlice<f32>,
        grad_h: &CudaSlice<f32>,
        grad_c_next: &CudaSlice<f32>,
        grad_gates: &mut CudaSlice<f32>,
        grad_c_prev: &mut CudaSlice<f32>,
        hidden_size: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("lstm_gates_backward_f32")
            .ok_or_else(|| CudaError::KernelNotFound("lstm_gates_backward_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(gates)
                .arg(c_prev)
                .arg(c_new)
                .arg(grad_h)
                .arg(grad_c_next)
                .arg(grad_gates)
                .arg(grad_c_prev)
                .arg(&(hidden_size as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(gates_ih)
                .arg(gates_hh)
                .arg(h_prev)
                .arg(h_new)
                .arg(&(hidden_size as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    // =========================================================================
    // Fused GRU Gate Backward Kernel
    // =========================================================================

    /// Fused GRU gate backward computation on GPU.
    ///
    /// Given saved forward state and incoming gradient, computes ih/hh gate
    /// gradients and hidden state gradient to previous timestep.
    ///
    /// - `gates_ih`: [batch, 3*hidden] pre-activation ih gates from forward
    /// - `gates_hh`: [batch, 3*hidden] pre-activation hh gates from forward
    /// - `h_prev`: [batch, hidden] previous hidden state
    /// - `grad_h_new`: [batch, hidden] gradient from output
    /// - `grad_gates_ih`: [batch, 3*hidden] output ih gate gradients
    /// - `grad_gates_hh`: [batch, 3*hidden] output hh gate gradients
    /// - `grad_h_prev`: [batch, hidden] output gradient to prev hidden
    pub fn gru_gates_backward_f32(
        &self,
        gates_ih: &CudaSlice<f32>,
        gates_hh: &CudaSlice<f32>,
        h_prev: &CudaSlice<f32>,
        grad_h_new: &CudaSlice<f32>,
        grad_gates_ih: &mut CudaSlice<f32>,
        grad_gates_hh: &mut CudaSlice<f32>,
        grad_h_prev: &mut CudaSlice<f32>,
        hidden_size: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("gru_gates_backward_f32")
            .ok_or_else(|| CudaError::KernelNotFound("gru_gates_backward_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(total);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(gates_ih)
                .arg(gates_hh)
                .arg(h_prev)
                .arg(grad_h_new)
                .arg(grad_gates_ih)
                .arg(grad_gates_hh)
                .arg(grad_h_prev)
                .arg(&(hidden_size as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(x)
                .arg(sum_out)
                .arg(sum_sq_out)
                .arg(&(n as u32))
                .arg(&(c as u32))
                .arg(&(spatial as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(x)
                .arg(mean)
                .arg(var)
                .arg(gamma)
                .arg(beta)
                .arg(y)
                .arg(&eps)
                .arg(&(c as u32))
                .arg(&(spatial as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }
}

// =============================================================================
// Fused Scaled Dot-Product Attention
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Fused attention forward: Q @ K^T * scale -> softmax -> @ V
    /// without materializing the full N*N attention matrix.
    ///
    /// Q: [B, H, Tq, D], K: [B, H, Tk, D], V: [B, H, Tk, D]
    /// Output: [B, H, Tq, D]
    pub fn fused_attention_fwd_f32(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        scale: f32,
        batch_size: usize,
        num_heads: usize,
        tgt_len: usize,
        src_len: usize,
        head_dim: usize,
        is_causal: bool,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("fused_attention_fwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("fused_attention_fwd_f32".to_string()))?;
        let total_rows = batch_size * num_heads * tgt_len;
        let cfg = cuda_kernels::launch_config(total_rows);
        let is_causal_u32: u32 = if is_causal { 1 } else { 0 };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(output)
                .arg(&scale)
                .arg(&(batch_size as u32))
                .arg(&(num_heads as u32))
                .arg(&(tgt_len as u32))
                .arg(&(src_len as u32))
                .arg(&(head_dim as u32))
                .arg(&is_causal_u32)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused flash-PREFILL attention: one CTA = one warp = one (query_row, head).
    /// Single launch handles all query rows with causal masking.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_attn_prefill_f32(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        total_kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        swa_window: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("fused_attn_prefill_f32")
            .ok_or_else(|| CudaError::KernelNotFound("fused_attn_prefill_f32".to_string()))?;

        let total_ctas = seq_len * n_heads;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_ctas as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(out)
                .arg(&(seq_len as u32))
                .arg(&(total_kv_len as u32))
                .arg(&(n_heads as u32))
                .arg(&(n_kv_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&(pos_offset as u32))
                .arg(&(swa_window as u32))
                .arg(&scale)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused flash-decode attention for inference: one CTA = one warp = one
    /// attention head, online softmax over the KV cache. See
    /// `attention.cu::fused_attn_decode_f32` for the algorithm.
    ///
    /// Shapes:
    ///   `q`       : `[n_heads,    head_dim]` f32
    ///   `k_cache` : `[kv_len, n_kv_heads, head_dim]` f32
    ///   `v_cache` : `[kv_len, n_kv_heads, head_dim]` f32
    ///   `out`     : `[n_heads,    head_dim]` f32
    ///
    /// `swa_window = 0` ⇒ full causal attention. Otherwise positions
    /// `< kv_len - swa_window` are masked out.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_attn_decode_f32(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        swa_window: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        debug_assert!(
            head_dim <= 512,
            "fused_attn_decode_f32: head_dim {head_dim} exceeds kernel MAX_DIMS budget"
        );
        debug_assert!(
            n_kv_heads > 0 && n_heads % n_kv_heads == 0,
            "fused_attn_decode_f32: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
        );

        let func = self
            .kernels
            .get("fused_attn_decode_f32")
            .ok_or_else(|| CudaError::KernelNotFound("fused_attn_decode_f32".to_string()))?;

        // One warp per head. n_heads CTAs, 32 threads each.
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(out)
                .arg(&(kv_len as u32))
                .arg(&(n_heads as u32))
                .arg(&(n_kv_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&(swa_window as u32))
                .arg(&scale)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Quantize one KV row (`n_kv_heads * head_dim` f32s) into the int8
    /// per-head-scale layout used by `fused_attn_decode_q8_f32`.
    ///
    /// Writes int8 values into `dst_q` at row `pos` and one f32 scale per
    /// head into `dst_scale[pos * n_kv_heads + kv_h]`. The full scale buffer
    /// holds `capacity * n_kv_heads` f32s; writing by logical `pos` avoids
    /// needing to pass the capacity to the kernel.
    ///
    /// See `attention.cu::quantize_kv_row_q8_f32` for the algorithm.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_kv_row_q8_f32(
        &self,
        src: &CudaSlice<f32>,
        dst_q: &mut CudaSlice<i8>,
        dst_scale: &mut CudaSlice<f32>,
        n_kv_heads: usize,
        head_dim: usize,
        pos: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            head_dim <= 512,
            "quantize_kv_row_q8_f32: head_dim {head_dim} exceeds DIMS_MAX budget"
        );
        let func = self
            .kernels
            .get("quantize_kv_row_q8_f32")
            .ok_or_else(|| CudaError::KernelNotFound("quantize_kv_row_q8_f32".to_string()))?;

        // One warp per head. n_kv_heads CTAs, 32 threads each.
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_kv_heads as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst_q)
                .arg(dst_scale)
                .arg(&(n_kv_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&(pos as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Fused flash-decode attention with TurboQuant Q8 KV cache. Same
    /// online-softmax algorithm as `fused_attn_decode_f32`, but reads int8
    /// K/V with per-(token,head) f32 scales. Dequant is inline.
    ///
    /// See `attention.cu::fused_attn_decode_q8_f32` for the algorithm.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_attn_decode_q8_f32(
        &self,
        q: &CudaSlice<f32>,
        k_q: &CudaSlice<i8>,
        k_scale: &CudaSlice<f32>,
        v_q: &CudaSlice<i8>,
        v_scale: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        swa_window: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        debug_assert!(
            head_dim <= 512,
            "fused_attn_decode_q8_f32: head_dim {head_dim} exceeds MAX_DIMS budget"
        );
        debug_assert!(
            n_kv_heads > 0 && n_heads % n_kv_heads == 0,
            "fused_attn_decode_q8_f32: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
        );
        let func = self
            .kernels
            .get("fused_attn_decode_q8_f32")
            .ok_or_else(|| CudaError::KernelNotFound("fused_attn_decode_q8_f32".to_string()))?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q)
                .arg(k_q)
                .arg(k_scale)
                .arg(v_q)
                .arg(v_scale)
                .arg(out)
                .arg(&(kv_len as u32))
                .arg(&(n_heads as u32))
                .arg(&(n_kv_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&(swa_window as u32))
                .arg(&scale)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }
}

// =============================================================================
// Transformer Per-Layer Ops (rms_norm, RoPE, SwiGLU, ReLU² gate)
//
// Decode-step launchers for the kernels in `transformer_ops.cu`. Used by
// `Tensor::rms_norm` / `apply_rope_split_halves` / `swiglu` / `relu2_gate`
// to keep activations on GPU through the whole layer in nexus-serve.
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// RMSNorm with a per-element weight scale.
    /// `out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)`.
    ///
    /// One CTA, 256 threads. Suitable for hidden sizes up to ~16 K (warp
    /// reduction inside the kernel handles arbitrary `n`).
    pub fn rms_norm_f32(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("rms_norm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rms_norm_f32".to_string()))?;
        let block: u32 = 256;
        let n_warps = (block + 31) / 32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_warps * 4, // one f32 per warp
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(x)
                .arg(weight)
                .arg(&(n as u32))
                .arg(&eps)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Single-token LayerNorm. `out[i] = (x[i] - mean) / sqrt(var + eps)
    /// * gamma[i] + beta[i]` over a single vector of length `n`. Used by
    /// legacy Falcon's decode path. Distinct from `layer_norm_f32` above
    /// (which takes a `num_rows` and operates on a batched `[rows, n]`
    /// input for training).
    ///
    /// Same single-CTA two-pass reduction as `rms_norm_f32`; shared-mem
    /// budget is doubled (two `n_warps * f32` arrays — mean then var).
    pub fn layer_norm_tokenwise_f32(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("layer_norm_tokenwise_f32")
            .ok_or_else(|| CudaError::KernelNotFound("layer_norm_tokenwise_f32".to_string()))?;
        let block: u32 = 256;
        let n_warps = (block + 31) / 32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_warps * 4 * 2, // mean + var
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(x)
                .arg(gamma)
                .arg(beta)
                .arg(&(n as u32))
                .arg(&eps)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// GELU with the tanh approximation —
    /// `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))`.
    /// Used by Falcon's MLP; element-wise, one thread per element.
    pub fn gelu_tanh_f32(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("gelu_tanh_f32")
            .ok_or_else(|| CudaError::KernelNotFound("gelu_tanh_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(x)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Element-wise `dst += src * scalar` (in-place). MoE expert
    /// accumulate — one kernel instead of `mul_scalar` + `add`.
    pub fn scaled_add_inplace_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        n: usize,
        scalar: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("scaled_add_inplace_f32")
            .ok_or_else(|| CudaError::KernelNotFound("scaled_add_inplace_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(dst)
                .arg(src)
                .arg(&(n as u32))
                .arg(&scalar)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Parallel-residual add: `x[i] = x[i] + attn[i] + ffn[i]`. Element-
    /// wise; fuses Falcon's two residual adds into one kernel launch.
    pub fn parallel_residual_add_f32(
        &self,
        x: &mut CudaSlice<f32>,
        attn: &CudaSlice<f32>,
        ffn: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("parallel_residual_add_f32")
            .ok_or_else(|| CudaError::KernelNotFound("parallel_residual_add_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(x)
                .arg(attn)
                .arg(ffn)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Per-head RMS_norm (Qwen3 QK-norm). Applies
    /// `x[h, :] = x[h, :] * rsqrt(mean(x[h,:]²) + eps) * weight` for
    /// every head `h`, where `weight` is a single `[head_dim]` vector
    /// broadcast across every head.
    ///
    /// One warp per head. `src` and `out` may alias for in-place normalize.
    pub fn rms_norm_heads_f32(
        &self,
        out: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("rms_norm_heads_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rms_norm_heads_f32".to_string()))?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(out)
                .arg(weight)
                .arg(&(head_dim as u32))
                .arg(&eps)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// RoPE in the LLaMA / Qwen / Mistral split-halves layout.
    ///
    /// Each query/key vector is laid out as `[head][dim]` and rotated by
    /// pairing dimension `d` with `d + head_dim/2`. One thread per pair
    /// per head. Operates in place on `x`.
    /// `src` and `out` may alias for in-place rotation.
    pub fn rope_split_halves_f32(
        &self,
        out: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            head_dim % 2 == 0,
            "head_dim must be even for split-halves RoPE"
        );
        let func = self
            .kernels
            .get("rope_split_halves_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rope_split_halves_f32".to_string()))?;
        let half = (head_dim / 2) as u32;
        let block: u32 = half.min(128); // small enough to fit; pairs are independent
        let grid_y = (half + block - 1) / block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(out)
                .arg(&(n_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&theta)
                .arg(&(pos as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Fused SwiGLU FFN gate: `out[i] = SiLU(gate[i]) * up[i]`.
    /// Eliminates the silu+mul kernel pair the unfused path runs.
    pub fn swiglu_f32(
        &self,
        out: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("swiglu_f32")
            .ok_or_else(|| CudaError::KernelNotFound("swiglu_f32".to_string()))?;
        let block: u32 = 256;
        let grid: u32 = ((n as u32) + block - 1) / block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(gate)
                .arg(up)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// BitNet b1.58 fused gate: `out[i] = ReLU(gate[i])² * up[i]`.
    pub fn relu2_gate_f32(
        &self,
        out: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("relu2_gate_f32")
            .ok_or_else(|| CudaError::KernelNotFound("relu2_gate_f32".to_string()))?;
        let block: u32 = 256;
        let grid: u32 = ((n as u32) + block - 1) / block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(gate)
                .arg(up)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Batched RMSNorm: `out[t, :] = rms_norm(x[t, :], weight)` for t in [0, m).
    /// x, out shape: [m, n] contiguous row-major.
    pub fn rms_norm_batched_f32(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("rms_norm_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rms_norm_batched_f32".to_string()))?;
        let block: u32 = 256;
        let n_warps = (block + 31) / 32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_warps * 4,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(x)
                .arg(weight)
                .arg(&(n as u32))
                .arg(&eps)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Batched per-head RMSNorm (Qwen3 QK-norm) over `m` tokens.
    /// In-place on x of shape [m, n_heads, head_dim] row-major.
    /// `src`/`out` may alias for in-place normalize.
    pub fn rms_norm_heads_batched_f32(
        &self,
        out: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        m: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("rms_norm_heads_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rms_norm_heads_batched_f32".to_string()))?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, m as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(out)
                .arg(weight)
                .arg(&(n_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&eps)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Batched split-halves RoPE. Rotates x[t, h, :] at position (pos_start + t)
    /// for t in [0, m). In-place on x of shape [m, n_heads, head_dim] row-major.
    /// `src`/`out` may alias for in-place rotation.
    pub fn rope_split_halves_batched_f32(
        &self,
        out: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        m: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Result<(), CudaError> {
        debug_assert!(
            head_dim % 2 == 0,
            "head_dim must be even for split-halves RoPE"
        );
        let func = self
            .kernels
            .get("rope_split_halves_batched_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("rope_split_halves_batched_f32".to_string())
            })?;
        let half = (head_dim / 2) as u32;
        let block: u32 = half.min(128);
        let grid_y = (half + block - 1) / block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, m as u32),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(out)
                .arg(&(n_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&theta)
                .arg(&(pos_start as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Broadcast per-column bias across m rows: `out[t, c] += bias[c]`.
    /// `out` shape: [m, n] contiguous row-major.
    pub fn add_bias_batched_f32(
        &self,
        out: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        m: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("add_bias_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("add_bias_batched_f32".to_string()))?;
        let total = (m * n) as u32;
        let block: u32 = 256;
        let grid: u32 = (total + block - 1) / block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(bias)
                .arg(&(m as u32))
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }
}

// =============================================================================
// Fused Attention Backward (recomputation-based, memory-efficient)
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Fused attention backward: recomputes attention weights from Q, K, O
    /// and computes grad_Q, grad_K, grad_V without materializing the N*N matrix.
    ///
    /// Q, K, V: [B, H, Tq/Tk, D]
    /// O: forward output [B, H, Tq, D]
    /// grad_O: gradient of loss w.r.t. output [B, H, Tq, D]
    /// grad_Q, grad_K, grad_V: output buffers (must be zero-initialized)
    pub fn fused_attention_bwd_f32(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        o: &CudaSlice<f32>,
        grad_o: &CudaSlice<f32>,
        grad_q: &mut CudaSlice<f32>,
        grad_k: &mut CudaSlice<f32>,
        grad_v: &mut CudaSlice<f32>,
        scale: f32,
        batch_size: usize,
        num_heads: usize,
        tgt_len: usize,
        src_len: usize,
        head_dim: usize,
        is_causal: bool,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("fused_attention_bwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("fused_attention_bwd_f32".to_string()))?;
        let total_rows = batch_size * num_heads * tgt_len;
        let cfg = cuda_kernels::launch_config(total_rows);
        let is_causal_u32: u32 = if is_causal { 1 } else { 0 };
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(o)
                .arg(grad_o)
                .arg(grad_q)
                .arg(grad_k)
                .arg(grad_v)
                .arg(&scale)
                .arg(&(batch_size as u32))
                .arg(&(num_heads as u32))
                .arg(&(tgt_len as u32))
                .arg(&(src_len as u32))
                .arg(&(head_dim as u32))
                .arg(&is_causal_u32)
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(col)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(col)
                .arg(output)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(data)
                .arg(bias)
                .arg(&(spatial as u32))
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(output)
                .arg(indices)
                .arg(params)
                .arg(&(channels as u32))
                .arg(&(out_h as u32))
                .arg(&(out_w as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_output)
                .arg(indices)
                .arg(grad_input)
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(output)
                .arg(params)
                .arg(&(channels as u32))
                .arg(&(out_h as u32))
                .arg(&(out_w as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
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
            self.stream
                .launch_builder(func)
                .arg(grad_output)
                .arg(grad_input)
                .arg(params)
                .arg(&(channels as u32))
                .arg(&(out_h as u32))
                .arg(&(out_w as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
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
            let result = cudarc::driver::sys::cuMemAllocHost_v2(&mut host_ptr, byte_size);
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
            let result = cudarc::driver::sys::cuMemAllocHost_v2(&mut host_ptr, byte_size);
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
                let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
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
