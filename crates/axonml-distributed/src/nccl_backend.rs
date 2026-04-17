//! NCCL Backend - Real NVIDIA NCCL Communication Backend
//!
//! # File
//! `crates/axonml-distributed/src/nccl_backend.rs`
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

use crate::backend::{Backend, ReduceOp};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

// =============================================================================
// NCCL FFI Types
// =============================================================================

/// NCCL result code.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum NcclResult {
    /// Success.
    Success = 0,
    /// Unhandled CUDA error.
    UnhandledCudaError = 1,
    /// System error.
    SystemError = 2,
    /// Internal error.
    InternalError = 3,
    /// Invalid argument.
    InvalidArgument = 4,
    /// Invalid usage.
    InvalidUsage = 5,
    /// Remote error.
    RemoteError = 6,
    /// In progress.
    InProgress = 7,
}

/// NCCL reduction operation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum NcclRedOp {
    /// Sum.
    Sum = 0,
    /// Product.
    Prod = 1,
    /// Maximum.
    Max = 2,
    /// Minimum.
    Min = 3,
    /// Average.
    Avg = 4,
}

/// NCCL data type.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum NcclDataType {
    /// 8-bit signed integer.
    Int8 = 0,
    /// 8-bit unsigned integer.
    Uint8 = 1,
    /// 32-bit signed integer.
    Int32 = 2,
    /// 32-bit unsigned integer.
    Uint32 = 3,
    /// 64-bit signed integer.
    Int64 = 4,
    /// 64-bit unsigned integer.
    Uint64 = 5,
    /// 16-bit float.
    Float16 = 6,
    /// 32-bit float.
    Float32 = 7,
    /// 64-bit float.
    Float64 = 8,
}

/// NCCL unique ID (128 bytes, matches `ncclUniqueId`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    /// Internal 128-byte identifier.
    pub internal: [c_char; 128],
}

impl Default for NcclUniqueId {
    fn default() -> Self {
        Self { internal: [0; 128] }
    }
}

impl std::fmt::Debug for NcclUniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclUniqueId")
            .field("internal", &"[128 bytes]")
            .finish()
    }
}

/// Opaque NCCL communicator handle.
type NcclComm = *mut c_void;

/// CUDA stream handle (opaque pointer from CUDA runtime).
type CudaStream = *mut c_void;

// =============================================================================
// NCCL Function Signatures
// =============================================================================

type FnGetVersion = unsafe extern "C" fn(*mut c_int) -> NcclResult;
type FnGetUniqueId = unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult;
type FnCommInitRank = unsafe extern "C" fn(*mut NcclComm, c_int, NcclUniqueId, c_int) -> NcclResult;
type FnCommDestroy = unsafe extern "C" fn(NcclComm) -> NcclResult;
type FnCommFinalize = unsafe extern "C" fn(NcclComm) -> NcclResult;
type FnGetErrorString = unsafe extern "C" fn(NcclResult) -> *const c_char;

type FnAllReduce = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    NcclRedOp,
    NcclComm,
    CudaStream,
) -> NcclResult;

type FnBroadcast = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    c_int,
    NcclComm,
    CudaStream,
) -> NcclResult;

type FnAllGather = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    NcclComm,
    CudaStream,
) -> NcclResult;

type FnReduceScatter = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    NcclRedOp,
    NcclComm,
    CudaStream,
) -> NcclResult;

type FnReduce = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    NcclRedOp,
    c_int,
    NcclComm,
    CudaStream,
) -> NcclResult;

type FnSend = unsafe extern "C" fn(
    *const c_void,
    usize,
    NcclDataType,
    c_int,
    NcclComm,
    CudaStream,
) -> NcclResult;

type FnRecv = unsafe extern "C" fn(
    *mut c_void,
    usize,
    NcclDataType,
    c_int,
    NcclComm,
    CudaStream,
) -> NcclResult;

type FnGroupStart = unsafe extern "C" fn() -> NcclResult;
type FnGroupEnd = unsafe extern "C" fn() -> NcclResult;

// =============================================================================
// CUDA Runtime FFI (for stream/memory management)
// =============================================================================

type FnCudaSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type FnCudaStreamCreate = unsafe extern "C" fn(*mut CudaStream) -> c_int;
type FnCudaStreamDestroy = unsafe extern "C" fn(CudaStream) -> c_int;
type FnCudaStreamSynchronize = unsafe extern "C" fn(CudaStream) -> c_int;
type FnCudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type FnCudaFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnCudaMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;

/// cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// =============================================================================
// NCCL Library Handle
// =============================================================================

/// Dynamically loaded NCCL library functions.
struct NcclLib {
    _lib: Library,
    _cuda_lib: Library,
    // NCCL functions
    get_version: FnGetVersion,
    get_unique_id: FnGetUniqueId,
    comm_init_rank: FnCommInitRank,
    comm_destroy: FnCommDestroy,
    comm_finalize: FnCommFinalize,
    get_error_string: FnGetErrorString,
    all_reduce: FnAllReduce,
    broadcast: FnBroadcast,
    all_gather: FnAllGather,
    reduce_scatter: FnReduceScatter,
    reduce: FnReduce,
    send: FnSend,
    recv: FnRecv,
    group_start: FnGroupStart,
    group_end: FnGroupEnd,
    // CUDA runtime functions
    cuda_set_device: FnCudaSetDevice,
    cuda_stream_create: FnCudaStreamCreate,
    cuda_stream_destroy: FnCudaStreamDestroy,
    cuda_stream_synchronize: FnCudaStreamSynchronize,
    cuda_malloc: FnCudaMalloc,
    cuda_free: FnCudaFree,
    cuda_memcpy: FnCudaMemcpy,
}

// Safety: NcclLib holds function pointers loaded from shared libraries.
// The NCCL and CUDA APIs are thread-safe for distinct communicators/streams.
unsafe impl Send for NcclLib {}
unsafe impl Sync for NcclLib {}

impl NcclLib {
    /// Load NCCL and CUDA runtime libraries dynamically.
    fn load() -> Result<Self, NcclError> {
        // Try common NCCL library paths
        let nccl_paths = [
            "libnccl.so.2",
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "/usr/local/lib/libnccl.so.2",
            "/usr/lib/libnccl.so.2",
            "libnccl.so",
        ];

        let lib = nccl_paths
            .iter()
            .find_map(|path| unsafe { Library::new(path).ok() })
            .ok_or(NcclError::LibraryNotFound)?;

        // Load CUDA runtime
        let cuda_paths = [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
        ];

        let cuda_lib = cuda_paths
            .iter()
            .find_map(|path| unsafe { Library::new(path).ok() })
            .ok_or(NcclError::CudaNotFound)?;

        unsafe {
            // Load all NCCL function pointers (extract raw pointers before moving lib)
            let fn_get_version = *lib
                .get::<FnGetVersion>(b"ncclGetVersion\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclGetVersion"))?;
            let fn_get_unique_id = *lib
                .get::<FnGetUniqueId>(b"ncclGetUniqueId\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclGetUniqueId"))?;
            let fn_comm_init_rank = *lib
                .get::<FnCommInitRank>(b"ncclCommInitRank\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclCommInitRank"))?;
            let fn_comm_destroy = *lib
                .get::<FnCommDestroy>(b"ncclCommDestroy\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclCommDestroy"))?;
            let fn_comm_finalize = *lib
                .get::<FnCommFinalize>(b"ncclCommFinalize\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclCommFinalize"))?;
            let fn_get_error_string = *lib
                .get::<FnGetErrorString>(b"ncclGetErrorString\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclGetErrorString"))?;
            let fn_all_reduce = *lib
                .get::<FnAllReduce>(b"ncclAllReduce\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclAllReduce"))?;
            let fn_broadcast = *lib
                .get::<FnBroadcast>(b"ncclBroadcast\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclBroadcast"))?;
            let fn_all_gather = *lib
                .get::<FnAllGather>(b"ncclAllGather\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclAllGather"))?;
            let fn_reduce_scatter = *lib
                .get::<FnReduceScatter>(b"ncclReduceScatter\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclReduceScatter"))?;
            let fn_reduce = *lib
                .get::<FnReduce>(b"ncclReduce\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclReduce"))?;
            let fn_send = *lib
                .get::<FnSend>(b"ncclSend\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclSend"))?;
            let fn_recv = *lib
                .get::<FnRecv>(b"ncclRecv\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclRecv"))?;
            let fn_group_start = *lib
                .get::<FnGroupStart>(b"ncclGroupStart\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclGroupStart"))?;
            let fn_group_end = *lib
                .get::<FnGroupEnd>(b"ncclGroupEnd\0")
                .map_err(|_| NcclError::SymbolNotFound("ncclGroupEnd"))?;

            // Load all CUDA runtime function pointers
            let fn_cuda_set_device = *cuda_lib
                .get::<FnCudaSetDevice>(b"cudaSetDevice\0")
                .map_err(|_| NcclError::SymbolNotFound("cudaSetDevice"))?;
            let fn_cuda_stream_create = *cuda_lib
                .get::<FnCudaStreamCreate>(b"cudaStreamCreate\0")
                .map_err(|_| NcclError::SymbolNotFound("cudaStreamCreate"))?;
            let fn_cuda_stream_destroy = *cuda_lib
                .get::<FnCudaStreamDestroy>(b"cudaStreamDestroy\0")
                .map_err(|_| NcclError::SymbolNotFound("cudaStreamDestroy"))?;
            let fn_cuda_stream_synchronize = *cuda_lib
                .get::<FnCudaStreamSynchronize>(b"cudaStreamSynchronize\0")
                .map_err(|_| NcclError::SymbolNotFound("cudaStreamSynchronize"))?;
            let fn_cuda_malloc = *cuda_lib
                .get::<FnCudaMalloc>(b"cudaMalloc\0")
                .map_err(|_| NcclError::SymbolNotFound("cudaMalloc"))?;
            let fn_cuda_free = *cuda_lib
                .get::<FnCudaFree>(b"cudaFree\0")
                .map_err(|_| NcclError::SymbolNotFound("cudaFree"))?;
            let fn_cuda_memcpy = *cuda_lib
                .get::<FnCudaMemcpy>(b"cudaMemcpy\0")
                .map_err(|_| NcclError::SymbolNotFound("cudaMemcpy"))?;

            // Now safe to move lib and cuda_lib since all Symbols are dropped
            Ok(Self {
                _lib: lib,
                _cuda_lib: cuda_lib,
                get_version: fn_get_version,
                get_unique_id: fn_get_unique_id,
                comm_init_rank: fn_comm_init_rank,
                comm_destroy: fn_comm_destroy,
                comm_finalize: fn_comm_finalize,
                get_error_string: fn_get_error_string,
                all_reduce: fn_all_reduce,
                broadcast: fn_broadcast,
                all_gather: fn_all_gather,
                reduce_scatter: fn_reduce_scatter,
                reduce: fn_reduce,
                send: fn_send,
                recv: fn_recv,
                group_start: fn_group_start,
                group_end: fn_group_end,
                cuda_set_device: fn_cuda_set_device,
                cuda_stream_create: fn_cuda_stream_create,
                cuda_stream_destroy: fn_cuda_stream_destroy,
                cuda_stream_synchronize: fn_cuda_stream_synchronize,
                cuda_malloc: fn_cuda_malloc,
                cuda_free: fn_cuda_free,
                cuda_memcpy: fn_cuda_memcpy,
            })
        }
    }

    /// Returns the NCCL version code.
    fn version(&self) -> Result<i32, NcclError> {
        let mut version: c_int = 0;
        let result = unsafe { (self.get_version)(&mut version) };
        check_nccl(result, self)?;
        Ok(version)
    }

    /// Returns a human-readable string for an NCCL error code.
    fn error_string(&self, result: NcclResult) -> String {
        let ptr = unsafe { (self.get_error_string)(result) };
        if ptr.is_null() {
            return format!("Unknown NCCL error: {:?}", result);
        }
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

// =============================================================================
// Error Type
// =============================================================================

/// Errors from NCCL operations.
#[derive(Debug)]
pub enum NcclError {
    /// NCCL shared library not found.
    LibraryNotFound,
    /// CUDA runtime library not found.
    CudaNotFound,
    /// A required symbol was not found in the library.
    SymbolNotFound(&'static str),
    /// An NCCL operation returned an error.
    NcclOp {
        /// The NCCL result code.
        code: NcclResult,
        /// Human-readable error description.
        message: String,
    },
    /// A CUDA runtime call failed.
    CudaError {
        /// The CUDA error code.
        code: c_int,
        /// Description of what failed.
        context: String,
    },
}

impl std::fmt::Display for NcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NcclError::LibraryNotFound => write!(f, "NCCL library (libnccl.so.2) not found"),
            NcclError::CudaNotFound => write!(f, "CUDA runtime library (libcudart.so) not found"),
            NcclError::SymbolNotFound(sym) => write!(f, "Symbol not found: {}", sym),
            NcclError::NcclOp { message, .. } => write!(f, "NCCL error: {}", message),
            NcclError::CudaError { code, context } => {
                write!(f, "CUDA error (code {}): {}", code, context)
            }
        }
    }
}

impl std::error::Error for NcclError {}

/// Check an NCCL result and convert to `Result`.
fn check_nccl(result: NcclResult, lib: &NcclLib) -> Result<(), NcclError> {
    if result == NcclResult::Success {
        Ok(())
    } else {
        Err(NcclError::NcclOp {
            code: result,
            message: lib.error_string(result),
        })
    }
}

/// Check a CUDA runtime result.
fn check_cuda(code: c_int, context: &str) -> Result<(), NcclError> {
    if code == 0 {
        Ok(())
    } else {
        Err(NcclError::CudaError {
            code,
            context: context.to_string(),
        })
    }
}

// =============================================================================
// Helper: Convert ReduceOp to NcclRedOp
// =============================================================================

fn to_nccl_op(op: ReduceOp) -> NcclRedOp {
    match op {
        ReduceOp::Sum => NcclRedOp::Sum,
        ReduceOp::Product => NcclRedOp::Prod,
        ReduceOp::Min => NcclRedOp::Min,
        ReduceOp::Max => NcclRedOp::Max,
        ReduceOp::Average => NcclRedOp::Avg,
    }
}

// =============================================================================
// GPU Buffer Helper
// =============================================================================

/// RAII wrapper for a GPU-allocated buffer.
struct GpuBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
    lib: Arc<NcclLib>,
}

impl GpuBuffer {
    /// Allocate `count` f32 elements on the GPU.
    fn alloc(lib: &Arc<NcclLib>, count: usize) -> Result<Self, NcclError> {
        let size_bytes = count * std::mem::size_of::<f32>();
        let mut ptr: *mut c_void = ptr::null_mut();
        let code = unsafe { (lib.cuda_malloc)(&mut ptr, size_bytes) };
        check_cuda(code, "cudaMalloc")?;
        Ok(Self {
            ptr,
            size_bytes,
            lib: Arc::clone(lib),
        })
    }

    /// Copy host data to this GPU buffer.
    fn copy_from_host(&self, data: &[f32]) -> Result<(), NcclError> {
        let size = (data.len() * std::mem::size_of::<f32>()).min(self.size_bytes);
        let code = unsafe {
            (self.lib.cuda_memcpy)(
                self.ptr,
                data.as_ptr() as *const c_void,
                size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        check_cuda(code, "cudaMemcpy H2D")
    }

    /// Copy from GPU buffer back to host slice.
    fn copy_to_host(&self, data: &mut [f32]) -> Result<(), NcclError> {
        let size = (data.len() * std::mem::size_of::<f32>()).min(self.size_bytes);
        let code = unsafe {
            (self.lib.cuda_memcpy)(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        check_cuda(code, "cudaMemcpy D2H")
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.lib.cuda_free)(self.ptr);
            }
        }
    }
}

// Safety: GPU buffers are accessed through NCCL collectives which are
// synchronized via CUDA streams.
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

// =============================================================================
// NcclBackend
// =============================================================================

/// Real NCCL backend for multi-GPU distributed training.
///
/// Uses NVIDIA's NCCL library for high-performance collective communication
/// over NVLink, PCIe, or InfiniBand. Requires CUDA GPUs and `libnccl.so.2`.
///
/// # Example
///
/// ```ignore
/// // On rank 0: generate unique ID and distribute to all ranks
/// let unique_id = NcclBackend::generate_unique_id()?;
/// // ... distribute unique_id to all ranks via MPI, TCP, etc. ...
///
/// // On each rank:
/// let backend = NcclBackend::new(unique_id, rank, world_size, gpu_device)?;
///
/// // Use as a Backend:
/// let mut data = vec![1.0f32; 1024];
/// backend.all_reduce(&mut data, ReduceOp::Sum);
/// ```
pub struct NcclBackend {
    lib: Arc<NcclLib>,
    comm: NcclComm,
    stream: CudaStream,
    rank: usize,
    world_size: usize,
    device: i32,
}

// Safety: NCCL communicators are thread-safe when used with distinct streams
// or properly synchronized. NcclBackend holds a unique communicator and stream
// per rank, and NCCL collectives internally handle synchronization.
unsafe impl Send for NcclBackend {}
unsafe impl Sync for NcclBackend {}

impl NcclBackend {
    /// Generate a unique ID for initializing a communicator group.
    ///
    /// This should be called on rank 0 and the resulting ID distributed
    /// to all other ranks before calling [`NcclBackend::new`].
    pub fn generate_unique_id() -> Result<NcclUniqueId, NcclError> {
        let lib = NcclLib::load()?;
        let mut id = NcclUniqueId::default();
        let result = unsafe { (lib.get_unique_id)(&mut id) };
        check_nccl(result, &lib)?;
        Ok(id)
    }

    /// Create a new NCCL backend for a single rank.
    ///
    /// # Arguments
    /// * `unique_id` - The communicator ID (from `generate_unique_id`, same for all ranks).
    /// * `rank` - This process's rank (0-indexed).
    /// * `world_size` - Total number of ranks in the group.
    /// * `device` - CUDA device ordinal to use for this rank.
    ///
    /// # Errors
    /// Returns `NcclError` if NCCL/CUDA initialization fails.
    pub fn new(
        unique_id: NcclUniqueId,
        rank: usize,
        world_size: usize,
        device: i32,
    ) -> Result<Self, NcclError> {
        let lib = Arc::new(NcclLib::load()?);

        // Set CUDA device for this rank
        let code = unsafe { (lib.cuda_set_device)(device) };
        check_cuda(code, "cudaSetDevice")?;

        // Create CUDA stream
        let mut stream: CudaStream = ptr::null_mut();
        let code = unsafe { (lib.cuda_stream_create)(&mut stream) };
        check_cuda(code, "cudaStreamCreate")?;

        // Initialize NCCL communicator
        let mut comm: NcclComm = ptr::null_mut();
        let result = unsafe {
            (lib.comm_init_rank)(&mut comm, world_size as c_int, unique_id, rank as c_int)
        };
        check_nccl(result, &lib)?;

        Ok(Self {
            lib,
            comm,
            stream,
            rank,
            world_size,
            device,
        })
    }

    /// Create multiple NCCL backends for all GPUs in a single process.
    ///
    /// This is a convenience function for single-node multi-GPU training,
    /// where all ranks run in the same process (different threads).
    ///
    /// # Arguments
    /// * `devices` - List of CUDA device ordinals to use. Length determines world size.
    ///
    /// # Errors
    /// Returns `NcclError` if initialization fails.
    pub fn create_world(devices: &[i32]) -> Result<Vec<Self>, NcclError> {
        let world_size = devices.len();
        let unique_id = Self::generate_unique_id()?;

        let mut backends = Vec::with_capacity(world_size);

        // Use NCCL group semantics to init all communicators from one thread
        let lib = Arc::new(NcclLib::load()?);
        let result = unsafe { (lib.group_start)() };
        check_nccl(result, &lib)?;

        let mut comms = vec![ptr::null_mut(); world_size];
        let mut streams = vec![ptr::null_mut(); world_size];

        for (rank, &device) in devices.iter().enumerate() {
            // Set device
            let code = unsafe { (lib.cuda_set_device)(device) };
            check_cuda(code, "cudaSetDevice")?;

            // Create stream on this device
            let code = unsafe { (lib.cuda_stream_create)(&mut streams[rank]) };
            check_cuda(code, "cudaStreamCreate")?;

            // Init communicator (grouped)
            let result = unsafe {
                (lib.comm_init_rank)(
                    &mut comms[rank],
                    world_size as c_int,
                    unique_id,
                    rank as c_int,
                )
            };
            check_nccl(result, &lib)?;
        }

        let result = unsafe { (lib.group_end)() };
        check_nccl(result, &lib)?;

        for (rank, &device) in devices.iter().enumerate() {
            backends.push(Self {
                lib: Arc::clone(&lib),
                comm: comms[rank],
                stream: streams[rank],
                rank,
                world_size,
                device,
            });
        }

        Ok(backends)
    }

    /// Returns the NCCL library version as (major, minor, patch).
    pub fn nccl_version(&self) -> Result<(i32, i32, i32), NcclError> {
        let code = self.lib.version()?;
        let major = code / 10000;
        let minor = (code % 10000) / 100;
        let patch = code % 100;
        Ok((major, minor, patch))
    }

    /// Returns the CUDA device ordinal used by this backend.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Synchronize the CUDA stream used by this backend.
    ///
    /// Blocks until all previously enqueued NCCL operations complete.
    pub fn synchronize(&self) -> Result<(), NcclError> {
        let code = unsafe { (self.lib.cuda_stream_synchronize)(self.stream) };
        check_cuda(code, "cudaStreamSynchronize")
    }

    /// Execute an NCCL collective using temporary GPU buffers.
    ///
    /// Handles: host->device copy, NCCL op, device->host copy, stream sync.
    fn with_gpu_buffers<F>(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        op: F,
    ) -> Result<(), NcclError>
    where
        F: FnOnce(*const c_void, *mut c_void) -> Result<(), NcclError>,
    {
        // Set device for this rank
        let code = unsafe { (self.lib.cuda_set_device)(self.device) };
        check_cuda(code, "cudaSetDevice")?;

        let send_buf = GpuBuffer::alloc(&self.lib, send_data.len())?;
        let recv_buf = GpuBuffer::alloc(&self.lib, recv_data.len())?;

        send_buf.copy_from_host(send_data)?;

        op(send_buf.ptr as *const c_void, recv_buf.ptr)?;

        // Synchronize before reading results back
        let code = unsafe { (self.lib.cuda_stream_synchronize)(self.stream) };
        check_cuda(code, "cudaStreamSynchronize")?;

        recv_buf.copy_to_host(recv_data)?;

        Ok(())
    }

    /// Execute an in-place NCCL collective (send and recv are same buffer).
    fn with_gpu_buffer_inplace<F>(&self, data: &mut [f32], op: F) -> Result<(), NcclError>
    where
        F: FnOnce(*mut c_void) -> Result<(), NcclError>,
    {
        let code = unsafe { (self.lib.cuda_set_device)(self.device) };
        check_cuda(code, "cudaSetDevice")?;

        let buf = GpuBuffer::alloc(&self.lib, data.len())?;
        buf.copy_from_host(data)?;

        op(buf.ptr)?;

        let code = unsafe { (self.lib.cuda_stream_synchronize)(self.stream) };
        check_cuda(code, "cudaStreamSynchronize")?;

        buf.copy_to_host(data)?;

        Ok(())
    }
}

impl Drop for NcclBackend {
    fn drop(&mut self) {
        // Finalize and destroy communicator
        if !self.comm.is_null() {
            unsafe {
                let _ = (self.lib.comm_finalize)(self.comm);
                let _ = (self.lib.comm_destroy)(self.comm);
            }
        }
        // Destroy stream
        if !self.stream.is_null() {
            unsafe {
                let _ = (self.lib.cuda_stream_destroy)(self.stream);
            }
        }
    }
}

// =============================================================================
// Backend Implementation
// =============================================================================

impl Backend for NcclBackend {
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "nccl"
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) {
        let nccl_op = to_nccl_op(op);
        let count = data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        self.with_gpu_buffer_inplace(data, |buf| {
            let result = unsafe {
                (lib.all_reduce)(
                    buf as *const c_void,
                    buf,
                    count,
                    NcclDataType::Float32,
                    nccl_op,
                    comm,
                    stream,
                )
            };
            check_nccl(result, lib)
        })
        .expect("NCCL all_reduce failed");
    }

    fn broadcast(&self, data: &mut [f32], src: usize) {
        let count = data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        self.with_gpu_buffer_inplace(data, |buf| {
            let result = unsafe {
                (lib.broadcast)(
                    buf as *const c_void,
                    buf,
                    count,
                    NcclDataType::Float32,
                    src as c_int,
                    comm,
                    stream,
                )
            };
            check_nccl(result, lib)
        })
        .expect("NCCL broadcast failed");
    }

    fn all_gather(&self, send_data: &[f32], recv_data: &mut [f32]) {
        let send_count = send_data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        self.with_gpu_buffers(send_data, recv_data, |send_buf, recv_buf| {
            let result = unsafe {
                (lib.all_gather)(
                    send_buf,
                    recv_buf,
                    send_count,
                    NcclDataType::Float32,
                    comm,
                    stream,
                )
            };
            check_nccl(result, lib)
        })
        .expect("NCCL all_gather failed");
    }

    fn reduce_scatter(&self, send_data: &[f32], recv_data: &mut [f32], op: ReduceOp) {
        let nccl_op = to_nccl_op(op);
        let recv_count = recv_data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        self.with_gpu_buffers(send_data, recv_data, |send_buf, recv_buf| {
            let result = unsafe {
                (lib.reduce_scatter)(
                    send_buf,
                    recv_buf,
                    recv_count,
                    NcclDataType::Float32,
                    nccl_op,
                    comm,
                    stream,
                )
            };
            check_nccl(result, lib)
        })
        .expect("NCCL reduce_scatter failed");
    }

    fn gather(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize) {
        // NCCL does not have a native gather. Implement via grouped send/recv.
        let send_count = send_data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        let code = unsafe { (lib.cuda_set_device)(self.device) };
        check_cuda(code, "cudaSetDevice").expect("CUDA set device failed");

        let send_buf =
            GpuBuffer::alloc(&self.lib, send_count).expect("GPU alloc failed for gather send");
        send_buf
            .copy_from_host(send_data)
            .expect("H2D copy failed for gather");

        // recv buffer only needed on dst rank, but allocate full size for simplicity
        let recv_buf =
            GpuBuffer::alloc(&self.lib, recv_data.len()).expect("GPU alloc failed for gather recv");

        unsafe {
            let result = (lib.group_start)();
            check_nccl(result, lib).expect("NCCL group_start failed");

            if self.rank == dst {
                // Destination receives from all ranks
                for r in 0..self.world_size {
                    let offset = r * send_count * std::mem::size_of::<f32>();
                    let recv_ptr = (recv_buf.ptr as *mut u8).add(offset) as *mut c_void;
                    let result = (lib.recv)(
                        recv_ptr,
                        send_count,
                        NcclDataType::Float32,
                        r as c_int,
                        comm,
                        stream,
                    );
                    check_nccl(result, lib).expect("NCCL recv in gather failed");
                }
            }

            // All ranks send to dst
            let result = (lib.send)(
                send_buf.ptr as *const c_void,
                send_count,
                NcclDataType::Float32,
                dst as c_int,
                comm,
                stream,
            );
            check_nccl(result, lib).expect("NCCL send in gather failed");

            let result = (lib.group_end)();
            check_nccl(result, lib).expect("NCCL group_end failed");
        }

        // Sync and copy back
        let code = unsafe { (lib.cuda_stream_synchronize)(self.stream) };
        check_cuda(code, "cudaStreamSynchronize").expect("CUDA sync failed");

        if self.rank == dst {
            recv_buf
                .copy_to_host(recv_data)
                .expect("D2H copy failed for gather");
        }
    }

    fn scatter(&self, send_data: &[f32], recv_data: &mut [f32], src: usize) {
        // NCCL does not have a native scatter. Implement via grouped send/recv.
        let recv_count = recv_data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        let code = unsafe { (lib.cuda_set_device)(self.device) };
        check_cuda(code, "cudaSetDevice").expect("CUDA set device failed");

        // Send buffer only needed on src rank
        let send_buf = GpuBuffer::alloc(&self.lib, send_data.len())
            .expect("GPU alloc failed for scatter send");
        if self.rank == src {
            send_buf
                .copy_from_host(send_data)
                .expect("H2D copy failed for scatter");
        }

        let recv_buf =
            GpuBuffer::alloc(&self.lib, recv_count).expect("GPU alloc failed for scatter recv");

        unsafe {
            let result = (lib.group_start)();
            check_nccl(result, lib).expect("NCCL group_start failed");

            if self.rank == src {
                // Source sends to all ranks
                for r in 0..self.world_size {
                    let offset = r * recv_count * std::mem::size_of::<f32>();
                    let send_ptr = (send_buf.ptr as *const u8).add(offset) as *const c_void;
                    let result = (lib.send)(
                        send_ptr,
                        recv_count,
                        NcclDataType::Float32,
                        r as c_int,
                        comm,
                        stream,
                    );
                    check_nccl(result, lib).expect("NCCL send in scatter failed");
                }
            }

            // All ranks receive from src
            let result = (lib.recv)(
                recv_buf.ptr,
                recv_count,
                NcclDataType::Float32,
                src as c_int,
                comm,
                stream,
            );
            check_nccl(result, lib).expect("NCCL recv in scatter failed");

            let result = (lib.group_end)();
            check_nccl(result, lib).expect("NCCL group_end failed");
        }

        // Sync and copy back
        let code = unsafe { (lib.cuda_stream_synchronize)(self.stream) };
        check_cuda(code, "cudaStreamSynchronize").expect("CUDA sync failed");

        recv_buf
            .copy_to_host(recv_data)
            .expect("D2H copy failed for scatter");
    }

    fn reduce(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize, op: ReduceOp) {
        let nccl_op = to_nccl_op(op);
        let count = send_data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        self.with_gpu_buffers(send_data, recv_data, |send_buf, recv_buf| {
            let result = unsafe {
                (lib.reduce)(
                    send_buf,
                    recv_buf,
                    count,
                    NcclDataType::Float32,
                    nccl_op,
                    dst as c_int,
                    comm,
                    stream,
                )
            };
            check_nccl(result, lib)
        })
        .expect("NCCL reduce failed");
    }

    fn barrier(&self) {
        // NCCL has no explicit barrier. Implement via zero-byte all-reduce.
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        let code = unsafe { (self.lib.cuda_set_device)(self.device) };
        check_cuda(code, "cudaSetDevice").expect("CUDA set device failed");

        // Allocate a tiny 1-element buffer for the all-reduce barrier
        let buf = GpuBuffer::alloc(&self.lib, 1).expect("GPU alloc failed for barrier");

        let result = unsafe {
            (lib.all_reduce)(
                buf.ptr as *const c_void,
                buf.ptr,
                1,
                NcclDataType::Float32,
                NcclRedOp::Sum,
                comm,
                stream,
            )
        };
        check_nccl(result, lib).expect("NCCL barrier (all_reduce) failed");

        let code = unsafe { (lib.cuda_stream_synchronize)(stream) };
        check_cuda(code, "cudaStreamSynchronize").expect("CUDA sync for barrier failed");
    }

    fn send(&self, data: &[f32], dst: usize, _tag: usize) {
        let count = data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        let code = unsafe { (self.lib.cuda_set_device)(self.device) };
        check_cuda(code, "cudaSetDevice").expect("CUDA set device failed");

        let send_buf = GpuBuffer::alloc(&self.lib, count).expect("GPU alloc failed for send");
        send_buf
            .copy_from_host(data)
            .expect("H2D copy failed for send");

        // NCCL send/recv must be paired in a group when called from the same thread
        let result = unsafe {
            (lib.send)(
                send_buf.ptr as *const c_void,
                count,
                NcclDataType::Float32,
                dst as c_int,
                comm,
                stream,
            )
        };
        check_nccl(result, lib).expect("NCCL send failed");

        let code = unsafe { (lib.cuda_stream_synchronize)(stream) };
        check_cuda(code, "cudaStreamSynchronize").expect("CUDA sync for send failed");
    }

    fn recv(&self, data: &mut [f32], src: usize, _tag: usize) {
        let count = data.len();
        let comm = self.comm;
        let stream = self.stream;
        let lib = &self.lib;

        let code = unsafe { (self.lib.cuda_set_device)(self.device) };
        check_cuda(code, "cudaSetDevice").expect("CUDA set device failed");

        let recv_buf = GpuBuffer::alloc(&self.lib, count).expect("GPU alloc failed for recv");

        let result = unsafe {
            (lib.recv)(
                recv_buf.ptr,
                count,
                NcclDataType::Float32,
                src as c_int,
                comm,
                stream,
            )
        };
        check_nccl(result, lib).expect("NCCL recv failed");

        let code = unsafe { (lib.cuda_stream_synchronize)(stream) };
        check_cuda(code, "cudaStreamSynchronize").expect("CUDA sync for recv failed");

        recv_buf
            .copy_to_host(data)
            .expect("D2H copy failed for recv");
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nccl_unique_id_default() {
        let id = NcclUniqueId::default();
        assert!(id.internal.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_nccl_unique_id_debug() {
        let id = NcclUniqueId::default();
        let debug = format!("{:?}", id);
        assert!(debug.contains("NcclUniqueId"));
    }

    #[test]
    fn test_reduce_op_conversion() {
        assert!(matches!(to_nccl_op(ReduceOp::Sum), NcclRedOp::Sum));
        assert!(matches!(to_nccl_op(ReduceOp::Product), NcclRedOp::Prod));
        assert!(matches!(to_nccl_op(ReduceOp::Min), NcclRedOp::Min));
        assert!(matches!(to_nccl_op(ReduceOp::Max), NcclRedOp::Max));
        assert!(matches!(to_nccl_op(ReduceOp::Average), NcclRedOp::Avg));
    }

    #[test]
    fn test_nccl_error_display() {
        let err = NcclError::LibraryNotFound;
        assert!(format!("{}", err).contains("libnccl.so.2"));

        let err = NcclError::CudaNotFound;
        assert!(format!("{}", err).contains("libcudart.so"));

        let err = NcclError::SymbolNotFound("ncclAllReduce");
        assert!(format!("{}", err).contains("ncclAllReduce"));

        let err = NcclError::CudaError {
            code: 2,
            context: "test".to_string(),
        };
        assert!(format!("{}", err).contains("test"));
    }

    #[test]
    fn test_nccl_result_enum() {
        assert_eq!(NcclResult::Success as i32, 0);
        assert_eq!(NcclResult::InternalError as i32, 3);
        assert_eq!(NcclResult::InvalidArgument as i32, 4);
    }

    #[test]
    fn test_nccl_data_type_enum() {
        assert_eq!(NcclDataType::Float32 as i32, 7);
        assert_eq!(NcclDataType::Float64 as i32, 8);
        assert_eq!(NcclDataType::Int32 as i32, 2);
    }
}
