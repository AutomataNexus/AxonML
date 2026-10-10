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

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::alloc_prelude::*;

#[cfg(feature = "cuda")]
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};

#[cfg(feature = "cudnn")]
use cudarc::cudnn::Cudnn;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
    ValidAsZeroBits,
};

use super::Backend;

#[cfg(feature = "cuda")]
use super::cuda_kernels::{self, BLOCK_SIZE, CudaKernels};

use crate::device::DeviceCapabilities;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

#[cfg(feature = "std")]
use core::sync::atomic::{AtomicU64, Ordering};

/// Device-to-host copy count since process start (copy-storm profiling).
#[cfg(feature = "std")]
pub static PROF_D2H_N: AtomicU64 = AtomicU64::new(0);

/// Device-to-host bytes since process start.
#[cfg(feature = "std")]
pub static PROF_D2H_B: AtomicU64 = AtomicU64::new(0);

/// Host-to-device copy count since process start.
#[cfg(feature = "std")]
pub static PROF_H2D_N: AtomicU64 = AtomicU64::new(0);

/// Host-to-device bytes since process start.
#[cfg(feature = "std")]
pub static PROF_H2D_B: AtomicU64 = AtomicU64::new(0);

/// Shape/stride scratch-buffer upload count since process start. Counts ACTUAL uploads: a request
/// whose contents already sit in the scratch buffer is served from cache and counted in
/// `PROF_SCRATCH_HIT_N` instead.
#[cfg(feature = "std")]
pub static PROF_SCRATCH_N: AtomicU64 = AtomicU64::new(0);

/// Shape/stride scratch requests served from the cache (memcpy elided) since process start.
#[cfg(feature = "std")]
pub static PROF_SCRATCH_HIT_N: AtomicU64 = AtomicU64::new(0);

/// Scratch requests served from cache (memcpy elided) since process start.
#[must_use]
#[cfg(feature = "std")]
pub fn prof_scratch_hits() -> u64 {
    PROF_SCRATCH_HIT_N.load(Ordering::Relaxed)
}

/// Returns (d2h_n, d2h_bytes, h2d_n, h2d_bytes, scratch_n) since process start.
/// Live vs reserved device memory of this process's default CUDA mempool (the
/// stream-ordered allocator): `(used_now, reserved_now, used_high_water)` in bytes.
/// `used` is what tensors hold right now; `reserved` is what the driver keeps; the
/// gap is fragmentation. A `reserved` at the card's ceiling with `used` far below it
/// is the WSL paging signature.
#[cfg(feature = "cuda")]
pub fn mempool_usage() -> Option<(u64, u64, u64)> {
    use cudarc::driver::sys as cs;
    let be = get_cuda_backend()?;
    let dev = be.context().cu_device();
    // SAFETY: read-only driver queries on the device this backend created its
    // context for; every out-parameter is a local the driver fills, and each
    // status is checked before the value is used.
    unsafe {
        let mut pool: cs::CUmemoryPool = std::ptr::null_mut();
        if cs::cuDeviceGetDefaultMemPool(&raw mut pool, dev) != cs::CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut out = [0u64; 3];
        for (i, attr) in [
            cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            cs::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        ]
        .into_iter()
        .enumerate()
        {
            let mut v: cs::cuuint64_t = 0;
            if cs::cuMemPoolGetAttribute(pool, attr, (&raw mut v).cast::<std::ffi::c_void>())
                != cs::CUresult::CUDA_SUCCESS
            {
                return None;
            }
            out[i] = v;
        }
        Some((out[0], out[1], out[2]))
    }
}

/// Stub without the `cuda` feature: there is no device pool to query.
#[cfg(not(feature = "cuda"))]
pub fn mempool_usage() -> Option<(u64, u64, u64)> {
    None
}

/// Copy-storm counters since process start: `(d2h_n, d2h_bytes, h2d_n, h2d_bytes, scratch_uploads)`.
#[cfg(feature = "std")]
pub fn prof_copy_snapshot() -> (u64, u64, u64, u64, u64) {
    (
        PROF_D2H_N.load(Ordering::Relaxed),
        PROF_D2H_B.load(Ordering::Relaxed),
        PROF_H2D_N.load(Ordering::Relaxed),
        PROF_H2D_B.load(Ordering::Relaxed),
        PROF_SCRATCH_N.load(Ordering::Relaxed),
    )
}

/// Owned host boxes kept alive for a captured graph's lifetime: their addresses
/// back the captured HtoD nodes, so freeing them mid-lifetime dangles replay.
#[cfg(feature = "cuda")]
pub enum HostKeep {
    /// Boxed u32 upload kept alive for the arena.
    U32(Box<[u32]>),
    /// Boxed i64 upload kept alive for the arena.
    I64(Box<[i64]>),
    /// Boxed f32 upload kept alive for the arena.
    F32(Box<[f32]>),
}

#[cfg(feature = "cuda")]
thread_local! {
    static CAPTURE_HOST_ARENA: std::cell::RefCell<Option<Vec<HostKeep>>> =
        const { std::cell::RefCell::new(None) };
}

/// Begin a capture-scoped host arena on the current thread — call immediately
/// before `graph_begin_capture`; scratch HtoDs source from arena-owned boxes.
#[cfg(feature = "cuda")]
pub fn capture_host_arena_begin() {
    CAPTURE_HOST_ARENA.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// End the arena and return the kept host boxes; hold them alive while the exec
/// graph may launch, then drop after `graph_destroy`.
#[cfg(feature = "cuda")]
pub fn capture_host_arena_take() -> Option<Vec<HostKeep>> {
    CAPTURE_HOST_ARENA.with(|c| c.borrow_mut().take())
}

#[cfg(feature = "cuda")]
fn capture_host_arena_active() -> bool {
    CAPTURE_HOST_ARENA.with(|c| c.borrow().is_some())
}

#[cfg(feature = "cuda")]
fn capture_host_arena_push(k: HostKeep) {
    CAPTURE_HOST_ARENA.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push(k);
        }
    });
}

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
    /// Runtime-JIT'd elementwise-chain kernels, keyed by chain signature. Interior mutability so
    /// the fuser can compile-on-first-use behind the `&self` backend handle. Compiled once per
    /// unique chain, then dispatched for free.
    jit_cache: parking_lot::Mutex<std::collections::HashMap<String, cudarc::driver::CudaFunction>>,
    /// Pre-allocated scratch buffer for shape-metadata uploads (up to 16
    /// u32 dims). Using `stream.memcpy_htod` into this pre-existing slice
    /// replaces the capture-breaking `stream.clone_htod` in hot paths like
    /// `contiguous_gpu`. Single-stream training serializes access —
    /// parking_lot mutex makes that explicit for the borrow checker.
    shape_scratch: parking_lot::Mutex<CudaSlice<u32>>,
    strides_scratch: parking_lot::Mutex<CudaSlice<i64>>,
    /// Last contents uploaded into each scratch buffer, so a repeat request elides the memcpy.
    /// Safe: all work is enqueued on the single stream and the caller holds the guard across the
    /// kernel launch, so a kernel reading the scratch is ordered before any later overwrite.
    shape_last: parking_lot::Mutex<Vec<u32>>,
    strides_last: parking_lot::Mutex<Vec<i64>>,
    #[cfg(feature = "cudnn")]
    cudnn_handle: Option<Arc<Cudnn>>,
}

/// CUDA backend stub when the `cuda` feature is disabled.
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
        let ctx = CudaContext::new(device_index).ok()?;
        // SAFETY: cudarc's three obligations are all inter-stream hazards. This
        // backend creates exactly one stream (tools/check_launches.py enforces
        // that count), and CUDA orders a single stream's work in issue order,
        // so no slice can be freed, read or written out of order with any use.
        unsafe {
            ctx.disable_event_tracking();
        }

        let dev_idx = device_index as i32;
        // SAFETY: raw driver calls with no memory obligation on our side. `pool`
        // is an out-parameter the driver fills before we read it, and its
        // address is passed on only if the get call reported success. The
        // attribute value is a u64 the driver copies by value.
        unsafe {
            use cudarc::driver::sys::{
                CUmemPool_attribute, cuDeviceGetDefaultMemPool, cuMemPoolSetAttribute,
            };
            let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
            if cuDeviceGetDefaultMemPool(&raw mut pool, dev_idx)
                == cudarc::driver::sys::CUresult::CUDA_SUCCESS
                && !pool.is_null()
            {
                // ── keep at most 80% of the card pooled: an unbounded threshold ratchets
                // reserved memory to the ceiling under fragmentation and WSL then pages
                // (measured: live 8.2 GB, reserved 11.5 GB, 0.85 s → 3 s/step drift) ──
                let threshold: u64 = ctx
                    .mem_get_info()
                    .map_or(u64::MAX, |(_, total)| (total as u64 / 10) * 8);
                let _ = cuMemPoolSetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &raw const threshold as *mut std::ffi::c_void,
                );
            }
        }
        let stream = ctx.new_stream().ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        // ── AXONML_TF32: run every cuBLAS GEMM on tf32 tensor cores (10-bit mantissa; ~1.7x fp32) ──
        if std::env::var("AXONML_TF32").is_ok() {
            // SAFETY: `blas.handle()` is the live cuBLAS handle created just
            // above on this context; setting its math mode has no memory
            // preconditions.
            let st = unsafe {
                cudarc::cublas::sys::cublasSetMathMode(
                    *blas.handle(),
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                )
            };
            eprintln!("[AxonML CUDA] cuBLAS math mode: TF32 tensor cores ({st:?})");
        }
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

        // SAFETY: cudarc marks alloc unsafe because the memory is uninitialised.
        // These are reachable only through upload_shape_scratch and
        // upload_strides_scratch, which memcpy_htod a prefix before handing the
        // buffer to a kernel, so nothing ever reads the uninitialised tail.
        let shape_scratch = unsafe { stream.alloc::<u32>(16).ok()? };
        // SAFETY: as above -- written by upload_strides_scratch before any read.
        let strides_scratch = unsafe { stream.alloc::<i64>(16).ok()? };

        Some(Self {
            device_index,
            ctx,
            stream,
            blas,
            kernels,
            jit_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            shape_scratch: parking_lot::Mutex::new(shape_scratch),
            strides_scratch: parking_lot::Mutex::new(strides_scratch),
            shape_last: parking_lot::Mutex::new(Vec::new()),
            strides_last: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "cudnn")]
            cudnn_handle,
        })
    }

    /// Upload a shape array into the pre-allocated scratch buffer and return
    /// a lock guard holding it. Caller passes the guard into kernel launchers
    /// via `.slice()`. Async memcpy — capture-safe when the stream is under
    /// capture. Fresh guard per call serializes access across training ops
    /// (all of which happen on our single stream anyway).
    #[cfg(feature = "cuda")]
    pub fn upload_shape_scratch(
        &self,
        shape: &[u32],
    ) -> parking_lot::MutexGuard<'_, CudaSlice<u32>> {
        assert!(
            shape.len() <= 16,
            "upload_shape_scratch: max 16 dims, got {}",
            shape.len()
        );
        let mut guard = self.shape_scratch.lock();
        let mut last = self.shape_last.lock();
        if !capture_host_arena_active() && last.as_slice() == shape {
            PROF_SCRATCH_HIT_N.fetch_add(1, Ordering::Relaxed);
            return guard;
        }
        PROF_SCRATCH_N.fetch_add(1, Ordering::Relaxed);
        last.clear();
        last.extend_from_slice(shape);
        if capture_host_arena_active() {
            last.clear();
            let boxed: Box<[u32]> = shape.to_vec().into_boxed_slice();
            self.stream
                .memcpy_htod(&boxed[..], &mut *guard)
                .expect("memcpy_htod shape scratch (arena)");
            capture_host_arena_push(HostKeep::U32(boxed));
            return guard;
        }
        self.stream
            .memcpy_htod(shape, &mut *guard)
            .expect("memcpy_htod shape scratch");
        guard
    }

    /// Companion to `upload_shape_scratch` for i64 stride arrays.
    #[cfg(feature = "cuda")]
    pub fn upload_strides_scratch(
        &self,
        strides: &[i64],
    ) -> parking_lot::MutexGuard<'_, CudaSlice<i64>> {
        assert!(
            strides.len() <= 16,
            "upload_strides_scratch: max 16 dims, got {}",
            strides.len()
        );
        let mut guard = self.strides_scratch.lock();
        let mut last = self.strides_last.lock();
        if !capture_host_arena_active() && last.as_slice() == strides {
            PROF_SCRATCH_HIT_N.fetch_add(1, Ordering::Relaxed);
            return guard;
        }
        PROF_SCRATCH_N.fetch_add(1, Ordering::Relaxed);
        last.clear();
        last.extend_from_slice(strides);
        if capture_host_arena_active() {
            last.clear();
            let boxed: Box<[i64]> = strides.to_vec().into_boxed_slice();
            self.stream
                .memcpy_htod(&boxed[..], &mut *guard)
                .expect("memcpy_htod strides scratch (arena)");
            capture_host_arena_push(HostKeep::I64(boxed));
            return guard;
        }
        self.stream
            .memcpy_htod(strides, &mut *guard)
            .expect("memcpy_htod strides scratch");
        guard
    }

    /// Creates a new CUDA backend (stub, always returns None without the `cuda` feature).
    #[cfg(not(feature = "cuda"))]
    pub fn new(device_index: usize) -> Option<Self> {
        let _ = device_index;
        None
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
        // SAFETY: the returned slice is uninitialised, which is what the name
        // promises. Every caller in this crate writes it in full -- as a kernel
        // output or a memcpy destination -- before reading; a caller that reads
        // first gets garbage but not UB, since CudaSlice never hands out a host
        // reference to device memory.
        unsafe { self.stream.alloc(len).map_err(CudaError::from) }
    }

    /// JIT + launch a fused elementwise-chain kernel. `key` identifies the chain (cache key);
    /// `expr` is a CUDA C expression over `x`, the input element. The entry
    /// `fused_chain(const float* in, float* out, unsigned int n)` and its `i < n` guard are
    /// emitted HERE, around `expr`, so the launch below matches the kernel by construction.
    /// Loads each input element once, applies the whole chain in registers, writes once.
    #[cfg(feature = "cuda")]
    pub fn fused_chain_unary_f32(
        &self,
        key: &str,
        expr: &str,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        assert!(
            input.len() >= n && output.len() >= n,
            "fused_chain_unary_f32: n={n} exceeds input {} / output {}",
            input.len(),
            output.len()
        );
        let mut cache = self.jit_cache.lock();
        if !cache.contains_key(key) {
            let src = format!(
                "extern \"C\" __global__ void fused_chain(const float* __restrict__ in, \
                 float* __restrict__ out, unsigned int n) {{\n\
                 \x20\x20unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n\
                 \x20\x20if (i >= n) return;\n\
                 \x20\x20float x = in[i];\n\
                 \x20\x20out[i] = {expr};\n}}\n"
            );
            let ptx = cudarc::nvrtc::compile_ptx(src)
                .map_err(|e| CudaError::ModuleLoadFailed(format!("nvrtc: {e}")))?;
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| CudaError::ModuleLoadFailed(e.to_string()))?;
            let func = module
                .load_function("fused_chain")
                .map_err(|e| CudaError::KernelNotFound(format!("fused_chain: {e}")))?;
            cache.insert(key.to_string(), func);
        }
        let func = cache.get(key).unwrap().clone();
        drop(cache);
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: the kernel text is assembled in this function: entry
        // `fused_chain(in, out, n)` guarded by `i < n`, three args in this
        // order; `input`/`output` hold >= n elements (asserted above) and
        // `output` is `&mut`. `expr` can only fail to compile, never change
        // the entry's parameter list. Ordering: single stream (see `new`).
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(input)
                .arg(output)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// JIT + launch a two-input fused elementwise chain. `expr` is a CUDA C expression over
    /// `x` (= `a[i]`) and `b[i]`; the entry `fused_chain(a, b, out, n)` and its `i < n` guard
    /// are emitted here around it.
    #[cfg(feature = "cuda")]
    pub fn fused_chain_binary_f32(
        &self,
        key: &str,
        expr: &str,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        assert!(
            a.len() >= n && b.len() >= n && output.len() >= n,
            "fused_chain_binary_f32: n={n} exceeds a {} / b {} / output {}",
            a.len(),
            b.len(),
            output.len()
        );
        let mut cache = self.jit_cache.lock();
        if !cache.contains_key(key) {
            let src = format!(
                "extern \"C\" __global__ void fused_chain(\
                 const float* __restrict__ a, const float* __restrict__ b, \
                 float* __restrict__ out, unsigned int n) {{\n\
                 \x20\x20unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n\
                 \x20\x20if (i >= n) return;\n\
                 \x20\x20float x = a[i];\n\
                 \x20\x20out[i] = {expr};\n}}\n"
            );
            let ptx = cudarc::nvrtc::compile_ptx(src)
                .map_err(|e| CudaError::ModuleLoadFailed(format!("nvrtc: {e}")))?;
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| CudaError::ModuleLoadFailed(e.to_string()))?;
            let func = module
                .load_function("fused_chain")
                .map_err(|e| CudaError::KernelNotFound(format!("fused_chain: {e}")))?;
            cache.insert(key.to_string(), func);
        }
        let func = cache.get(key).unwrap().clone();
        drop(cache);
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: the kernel text is assembled in this function: entry
        // `fused_chain(a, b, out, n)` guarded by `i < n`, four args in this
        // order; all three slices hold >= n elements (asserted above) and
        // `output` is `&mut`. `expr` can only fail to compile, never change
        // the entry's parameter list. Ordering: single stream (see `new`).
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(a)
                .arg(b)
                .arg(output)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Synchronize the backend's stream.
    #[cfg(feature = "cuda")]
    pub fn sync(&self) {
        let _ = self.stream.synchronize();
    }

    /// Synchronous D2H of `src[..n]` into `dst[..n]`.
    #[cfg(feature = "cuda")]
    pub fn dtoh_into_n(
        &self,
        src: &CudaSlice<f32>,
        n: usize,
        dst: &mut [f32],
    ) -> Result<(), CudaError> {
        use cudarc::driver::DevicePtr as _;
        assert!(
            src.len() >= n && dst.len() >= n,
            "dtoh_into_n: n={n} exceeds src {} / dst {}",
            src.len(),
            dst.len()
        );
        let (src_ptr, _guard) = src.device_ptr(&self.stream);
        // SAFETY: both sides hold >= n elements (asserted), `dst[..n]` is a
        // live &mut for the whole synchronous copy, and `_guard` keeps the
        // device slice's stream ordering for its duration.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(&mut dst[..n], src_ptr)
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Stream-ordered async D2H (no host sync) of `src[..n]` into the pinned
    /// `dst` at `dst_offset` — queued on the compute stream after prior work.
    /// The destination type guarantees pinned memory (a pageable slice would
    /// make the driver copy synchronously anyway). Call `sync()` before
    /// reading `dst`: event tracking is off on this context (see `new`).
    #[cfg(feature = "cuda")]
    pub fn dtoh_into_n_stream(
        &self,
        src: &CudaSlice<f32>,
        n: usize,
        dst: &mut PinnedBuffer,
        dst_offset: usize,
    ) -> Result<(), CudaError> {
        assert!(
            src.len() >= n && dst.len() >= dst_offset + n,
            "dtoh_into_n_stream: n={n} at {dst_offset} exceeds src {} / dst {}",
            src.len(),
            dst.len()
        );
        use cudarc::driver::DevicePtr as _;
        let inner = dst
            .inner
            .as_mut()
            .ok_or_else(|| CudaError::DriverError("pinned buffer already released".into()))?;
        let host = inner
            .as_mut_ptr()
            .map_err(|e| CudaError::DriverError(e.to_string()))?;
        let (src_ptr, _guard) = src.device_ptr(&self.stream);
        // SAFETY: `host` is the page-locked allocation `PinnedHostSlice` owns
        // (so the driver may DMA into it asynchronously), `dst_offset + n <=
        // dst.len()` and `n <= src.len()` are asserted above, and the copy is
        // queued on the backend's single stream behind the producers of
        // `src`. The write lands after this call returns; reading `dst`
        // before `sync()` is the documented contract, not a memory error,
        // and the buffer outlives the copy because `PinnedBuffer`'s drop is
        // itself stream-synchronised by cudarc.
        unsafe {
            let view = std::slice::from_raw_parts_mut(host.add(dst_offset), n);
            cudarc::driver::result::memcpy_dtoh_async(view, src_ptr, self.stream.cu_stream())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Strided device-to-device copy of a `height` x `width_elems` rectangle
    /// (element offsets and pitches), queued on the compute stream.
    #[cfg(feature = "cuda")]
    pub fn memcpy_2d_dtod_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        dst_offset_elems: usize,
        dst_pitch_elems: usize,
        src: &CudaSlice<f32>,
        src_offset_elems: usize,
        src_pitch_elems: usize,
        width_elems: usize,
        height: usize,
    ) -> Result<(), CudaError> {
        let extent = |off: usize, pitch: usize| -> Option<usize> {
            if height == 0 || width_elems == 0 {
                return Some(0);
            }
            pitch
                .checked_mul(height - 1)?
                .checked_add(width_elems)?
                .checked_add(off)
        };
        let (need_s, need_d) = (
            extent(src_offset_elems, src_pitch_elems),
            extent(dst_offset_elems, dst_pitch_elems),
        );
        assert!(
            src_pitch_elems >= width_elems
                && dst_pitch_elems >= width_elems
                && need_s.is_some_and(|x| x <= src.len())
                && need_d.is_some_and(|x| x <= dst.len()),
            "memcpy_2d_dtod_f32: {height} rows x {width_elems} at src {src_offset_elems}/{src_pitch_elems} (len {}) dst {dst_offset_elems}/{dst_pitch_elems} (len {}) out of bounds",
            src.len(),
            dst.len()
        );
        use cudarc::driver::DevicePtr as _;
        use cudarc::driver::DevicePtrMut as _;
        use cudarc::driver::sys;
        let esz = std::mem::size_of::<f32>();
        let (src_ptr, _gs) = src.device_ptr(&self.stream);
        let (dst_ptr, _gd) = dst.device_ptr_mut(&self.stream);
        let copy = sys::CUDA_MEMCPY2D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: src_ptr + (src_offset_elems * esz) as sys::CUdeviceptr,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch_elems * esz,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst_ptr + (dst_offset_elems * esz) as sys::CUdeviceptr,
            dstArray: std::ptr::null_mut(),
            dstPitch: dst_pitch_elems * esz,
            WidthInBytes: width_elems * esz,
            Height: height,
        };
        // SAFETY: the assert above bounds the last row's end (offset +
        // pitch*(height-1) + width) inside both slices and pitch >= width, so
        // the strided rectangle lies within memory the borrowed slices own;
        // the copy is queued on the backend's single stream.
        unsafe {
            let rc = sys::cuMemcpy2DAsync_v2(&raw const copy, self.stream.cu_stream());
            if rc != sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::DriverError(format!(
                    "cuMemcpy2DAsync_v2 -> {rc:?}"
                )));
            }
        }
        Ok(())
    }

    /// cuBLAS SGEMM with element offsets into `a`, `b` and `c` (column-major
    /// semantics as `gemm_f32`); bounds are checked past each offset.
    #[cfg(feature = "cuda")]
    pub fn gemm_f32_at(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        a_offset: usize,
        lda: usize,
        b: &CudaSlice<f32>,
        b_offset: usize,
        ldb: usize,
        beta: f32,
        c: &mut CudaSlice<f32>,
        c_offset: usize,
        ldc: usize,
    ) -> Result<(), CudaError> {
        use cudarc::cublas::result::sgemm;
        use cudarc::driver::DevicePtr as _;
        use cudarc::driver::DevicePtrMut as _;
        if a_offset > a.len() || b_offset > b.len() || c_offset > c.len() {
            return Err(CudaError::BlasError(format!(
                "gemm_f32_at: offsets {a_offset}/{b_offset}/{c_offset} exceed slices {}/{}/{}",
                a.len(),
                b.len(),
                c.len()
            )));
        }
        Self::check_gemm_bounds(
            transa,
            transb,
            m,
            n,
            k,
            a.len() - a_offset,
            lda,
            0,
            b.len() - b_offset,
            ldb,
            0,
            c.len() - c_offset,
            ldc,
            0,
            1,
        )?;
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
        let esz = std::mem::size_of::<f32>() as cudarc::driver::sys::CUdeviceptr;
        let (a_ptr, _ga) = a.device_ptr(&self.stream);
        let a_ptr = a_ptr + a_offset as cudarc::driver::sys::CUdeviceptr * esz;
        let (b_ptr, _gb) = b.device_ptr(&self.stream);
        let b_ptr = b_ptr + b_offset as cudarc::driver::sys::CUdeviceptr * esz;
        let (c_ptr, _gc) = c.device_ptr_mut(&self.stream);
        let c_ptr = c_ptr + c_offset as cudarc::driver::sys::CUdeviceptr * esz;
        // SAFETY: check_gemm_bounds ran on the slices' lengths past each
        // offset, so cuBLAS stays inside all three allocations; the slices are
        // borrowed for the call and the handle's stream is the backend's one.
        unsafe {
            sgemm(
                *self.blas.handle(),
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &raw const alpha,
                a_ptr as *const f32,
                lda as i32,
                b_ptr as *const f32,
                ldb as i32,
                &raw const beta,
                c_ptr as *mut f32,
                ldc as i32,
            )
            .map_err(CudaError::from)
        }
    }

    /// Copies data from host to device via `clone_htod` (allocates + syncs a
    /// new device slice; not capture-safe). Under capture, prefer `htod_into`.
    #[cfg(feature = "cuda")]
    /// Argmax along a dimension. Tensor viewed as [outer_size, dim_size, inner_size].
    /// Writes the winning index (cast to f32) for each of the outer_size*inner_size outputs.
    pub fn argmax_dim_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        outer_size: usize,
        dim_size: usize,
        inner_size: usize,
    ) -> Result<(), CudaError> {
        assert!(
            src.len() >= outer_size * dim_size * inner_size && dst.len() >= outer_size * inner_size,
            "argmax_dim_f32: src {} / dst {} too small for {outer_size}x{dim_size}x{inner_size}",
            src.len(),
            dst.len()
        );
        let func = self
            .kernels
            .get("argmax_dim_f32")
            .ok_or_else(|| CudaError::KernelNotFound("argmax_dim_f32".to_string()))?;
        let out_len = outer_size * inner_size;
        let cfg = cuda_kernels::launch_config(out_len);
        // SAFETY: `argmax_dim_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len); the kernel guards its index (see .cu).
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

    /// Argmin along a dimension. Same contract as argmax_dim_f32 but keeps the
    /// minimum — this is the direct VQ nearest-code primitive.
    #[cfg(feature = "cuda")]
    pub fn argmin_dim_f32(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        outer_size: usize,
        dim_size: usize,
        inner_size: usize,
    ) -> Result<(), CudaError> {
        assert!(
            src.len() >= outer_size * dim_size * inner_size && dst.len() >= outer_size * inner_size,
            "argmin_dim_f32: src {} / dst {} too small for {outer_size}x{dim_size}x{inner_size}",
            src.len(),
            dst.len()
        );
        let func = self
            .kernels
            .get("argmin_dim_f32")
            .ok_or_else(|| CudaError::KernelNotFound("argmin_dim_f32".to_string()))?;
        let out_len = outer_size * inner_size;
        let cfg = cuda_kernels::launch_config(out_len);
        // SAFETY: `argmin_dim_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len); the kernel guards its index (see .cu).
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

    /// Elapsed GPU milliseconds between two events recorded by `event_record`
    /// (synchronizes on `stop` first).
    #[cfg(feature = "cuda")]
    pub fn event_elapsed_ms(
        &self,
        start: &cudarc::driver::CudaEvent,
        stop: &cudarc::driver::CudaEvent,
    ) -> f32 {
        let _ = stop.synchronize();
        start.elapsed_ms(stop).unwrap_or(0.0)
    }

    /// Record a timing event on this stream. It timestamps the point the
    /// stream's GPU work reaches it — bracket a section with two of these and
    /// `event_elapsed_ms` to get its REAL GPU time (no LAUNCH_BLOCKING). The
    /// event is destroyed when dropped.
    #[cfg(feature = "cuda")]
    pub fn event_record(&self) -> Result<cudarc::driver::CudaEvent, CudaError> {
        self.stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| CudaError::DriverError(e.to_string()))
    }

    /// True only while the stream is in ACTIVE capture (false once an illegal op has
    /// INVALIDATED it). Poll before end/instantiate so a corrupt capture is never used.
    #[cfg(feature = "cuda")]
    pub fn stream_is_capturing(&self) -> bool {
        use cudarc::driver::sys;
        // SAFETY: a status query on this backend's own stream; `status` is a
        // local the driver fills and is read only on CUDA_SUCCESS.
        unsafe {
            let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            let rc = sys::cuStreamIsCapturing(self.stream.cu_stream(), &raw mut status);
            rc == sys::CUresult::CUDA_SUCCESS
                && status == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
        }
    }

    /// Uploads `src` into a fresh device slice (allocates + synchronous copy).
    #[cfg(feature = "cuda")]
    pub fn htod_copy<T: DeviceRepr>(&self, src: &[T]) -> Result<CudaSlice<T>, CudaError> {
        PROF_H2D_N.fetch_add(1, Ordering::Relaxed);
        PROF_H2D_B.fetch_add(std::mem::size_of_val(src) as u64, Ordering::Relaxed);
        self.stream.clone_htod(src).map_err(CudaError::from)
    }

    /// Async host→device copy into a pre-allocated destination. Capture-safe
    /// because there's no allocation inside the capture window — the caller
    /// owns `dst`, typically via `pool_alloc_uninit` which hits the pool
    /// cache on a warm training step.
    #[cfg(feature = "cuda")]
    pub fn htod_into<T: DeviceRepr>(
        &self,
        src: &[T],
        dst: &mut CudaSlice<T>,
    ) -> Result<(), CudaError> {
        self.stream.memcpy_htod(src, dst).map_err(CudaError::from)
    }

    /// Copies data from device to host.
    #[cfg(feature = "cuda")]
    pub fn dtoh_copy<T: DeviceRepr>(&self, src: &CudaSlice<T>) -> Result<Vec<T>, CudaError> {
        PROF_D2H_N.fetch_add(1, Ordering::Relaxed);
        PROF_D2H_B.fetch_add(std::mem::size_of_val(src) as u64, Ordering::Relaxed);
        self.stream.clone_dtoh(src).map_err(CudaError::from)
    }
}

#[cfg(feature = "cuda")]
impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let name = format!("CUDA Device {}", self.device_index);

        let mem = cudarc::driver::result::mem_get_info().ok();

        DeviceCapabilities {
            name,
            total_memory: mem.map(|(_, total)| total),
            available_memory: mem.map(|(free, _)| free),
            supports_f16: true,
            supports_f64: true,
            max_threads_per_block: 1024,
            compute_capability: None,
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
            total_memory: None,
            available_memory: None,
            supports_f16: false,
            supports_f64: false,
            max_threads_per_block: 0,
            compute_capability: None,
        }
    }

    fn synchronize(&self) {}
}

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

impl core::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

impl core::error::Error for CudaError {}

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
        total_memory: None,
        available_memory: None,
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
pub fn stream_synchronize(_handle: usize) {}

/// Synchronize a CUDA stream (no-op without the `cuda` feature).
#[cfg(not(feature = "cuda"))]
pub fn stream_synchronize(_handle: usize) {}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// The device-side twin of the CPU GEMM bound: cuBLAS is handed raw device
    /// pointers and `m`, `n`, `k` and the leading dimensions, and reads and
    /// writes exactly what those imply. Nothing in cudarc checks that against
    /// the slices, so this does, before any pointer leaves the safe API.
    ///
    /// Column-major, as cuBLAS is: an operand that is `rows x cols` after its
    /// transpose flag occupies `ld * cols` elements and needs `ld >= rows`.
    /// `batch` and the strides extend that to the strided-batched form; a
    /// batch of one with zero strides is the plain call.
    #[allow(clippy::too_many_arguments)]
    fn check_gemm_bounds(
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        a_len: usize,
        lda: usize,
        stride_a: usize,
        b_len: usize,
        ldb: usize,
        stride_b: usize,
        c_len: usize,
        ldc: usize,
        stride_c: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let (a_rows, a_cols) = if transa { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if transb { (n, k) } else { (k, n) };
        let need = |ld: usize, rows: usize, cols: usize, stride: usize| -> Option<usize> {
            if cols == 0 || batch == 0 {
                return Some(0);
            }
            if ld < rows {
                return None;
            }
            ld.checked_mul(cols)?
                .checked_add(stride.checked_mul(batch - 1)?)
        };
        let ok = matches!(need(lda, a_rows, a_cols, stride_a), Some(x) if x <= a_len)
            && matches!(need(ldb, b_rows, b_cols, stride_b), Some(x) if x <= b_len)
            && matches!(need(ldc, m, n, stride_c), Some(x) if x <= c_len);
        if ok {
            Ok(())
        } else {
            Err(CudaError::BlasError(format!(
                "GEMM dimensions exceed the slices given: m={m} n={n} k={k} lda={lda} \
                 ldb={ldb} ldc={ldc} batch={batch} with a={a_len} b={b_len} c={c_len}"
            )))
        }
    }

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

        Self::check_gemm_bounds(
            transa,
            transb,
            m,
            n,
            k,
            a.len(),
            lda,
            0,
            b.len(),
            ldb,
            0,
            c.len(),
            ldc,
            0,
            1,
        )?;

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

        // SAFETY: check_gemm_bounds above proved each slice holds every element
        // the operand shape, leading dimension and transpose flag imply, so
        // cuBLAS reads and writes inside the allocations. The device_ptr guards
        // keep the slices alive across the call, and the handle's stream is the
        // backend's single stream, so this is ordered with every other launch.
        unsafe {
            sgemm(
                *self.blas.handle(),
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &raw const alpha,
                a_ptr as *const f32,
                lda as i32,
                b_ptr as *const f32,
                ldb as i32,
                &raw const beta,
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
        if a_array.len() < batch_count || b_array.len() < batch_count || c_array.len() < batch_count
        {
            return Err(CudaError::BlasError(format!(
                "batched GEMM: batch_count {batch_count} exceeds the arrays given ({}, {}, {})",
                a_array.len(),
                b_array.len(),
                c_array.len()
            )));
        }
        for i in 0..batch_count {
            Self::check_gemm_bounds(
                transa,
                transb,
                m,
                n,
                k,
                a_array[i].len(),
                lda,
                0,
                b_array[i].len(),
                ldb,
                0,
                c_array[i].len(),
                ldc,
                0,
                1,
            )?;
        }

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

            // SAFETY: check_gemm_bounds ran for element i above, so cuBLAS
            // stays inside all three of this batch entry's allocations; the
            // slices are borrowed for the whole loop body, and the handle's
            // stream is the backend's single stream.
            unsafe {
                self.blas
                    .gemm(cfg, a_array[i], b_array[i], c_array[i])
                    .map_err(CudaError::from)?;
            }
        }
        Ok(())
    }

    /// `gemm_strided_batched_f32` with element offsets into `a`, `b` and `c`.
    ///
    /// Grouped conv needs a per-group base offset into the column buffer and the output while the
    /// batch dimension keeps a uniform stride; the shared per-group weight is passed with
    /// `stride_b = 0` and its own offset.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_strided_batched_f32_at(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        a_offset: usize,
        lda: usize,
        stride_a: i64,
        b: &CudaSlice<f32>,
        b_offset: usize,
        ldb: usize,
        stride_b: i64,
        beta: f32,
        c: &mut CudaSlice<f32>,
        c_offset: usize,
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

        if a_offset > a.len() || b_offset > b.len() || c_offset > c.len() {
            return Err(CudaError::BlasError(format!(
                "gemm_strided_batched_f32_at: offsets {a_offset}/{b_offset}/{c_offset} exceed slices {}/{}/{}",
                a.len(),
                b.len(),
                c.len()
            )));
        }
        // Negative strides never occur here; a negative one would walk below
        // the offset, which the bounds helper cannot express, so it is refused.
        let to_stride = |v: i64, what: &str| -> Result<usize, CudaError> {
            usize::try_from(v).map_err(|_| {
                CudaError::BlasError(format!(
                    "gemm_strided_batched_f32_at: negative {what} stride {v}"
                ))
            })
        };
        Self::check_gemm_bounds(
            transa,
            transb,
            m,
            n,
            k,
            a.len() - a_offset,
            lda,
            to_stride(stride_a, "a")?,
            b.len() - b_offset,
            ldb,
            to_stride(stride_b, "b")?,
            c.len() - c_offset,
            ldc,
            to_stride(stride_c, "c")?,
            batch_count,
        )?;
        let (a_devptr, _ga) = a.device_ptr(&self.stream);
        let (b_devptr, _gb) = b.device_ptr(&self.stream);
        let (c_devptr, _gc) = c.device_ptr_mut(&self.stream);
        let esz = std::mem::size_of::<f32>();
        let a_ptr = (a_devptr + (a_offset * esz) as cudarc::driver::sys::CUdeviceptr) as *const f32;
        let b_ptr = (b_devptr + (b_offset * esz) as cudarc::driver::sys::CUdeviceptr) as *const f32;
        let c_ptr = (c_devptr + (c_offset * esz) as cudarc::driver::sys::CUdeviceptr) as *mut f32;

        // SAFETY: check_gemm_bounds ran on the slices' lengths past each
        // offset with the batch strides, so every batch entry cuBLAS touches
        // lies inside the three allocations; the slices are borrowed for the
        // call and the handle's stream is the backend's one.
        unsafe {
            sgemm_strided_batched(
                *self.blas.handle(),
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &raw const alpha,
                a_ptr,
                lda as i32,
                stride_a,
                b_ptr,
                ldb as i32,
                stride_b,
                &raw const beta,
                c_ptr,
                ldc as i32,
                stride_c,
                batch_count as i32,
            )
            .map_err(CudaError::from)
        }
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
        Self::check_gemm_bounds(
            transa,
            transb,
            m,
            n,
            k,
            a.len(),
            lda,
            usize::try_from(stride_a).unwrap_or(usize::MAX),
            b.len(),
            ldb,
            usize::try_from(stride_b).unwrap_or(usize::MAX),
            c.len(),
            ldc,
            usize::try_from(stride_c).unwrap_or(usize::MAX),
            batch_count,
        )?;

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

        // SAFETY: check_gemm_bounds above proved each slice holds every element
        // implied by the operand shapes, leading dimensions, strides and batch
        // count, so every batch entry cuBLAS touches is inside the allocations.
        // The device_ptr guards keep the slices alive across the call, on the
        // backend's single stream.
        unsafe {
            sgemm_strided_batched(
                *self.blas.handle(),
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &raw const alpha,
                a_ptr,
                lda as i32,
                stride_a,
                b_ptr,
                ldb as i32,
                stride_b,
                &raw const beta,
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
        // SAFETY: `add_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// Per-output-channel LSQ fake-quant forward (GPU, QAT). Folded from vendored AxonML.
    #[allow(clippy::too_many_arguments)]
    pub fn fake_quant_pc_fwd_f32(
        &self,
        input: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        qn: f32,
        qp: f32,
        ch: u32,
        stride: u32,
        n: usize,
    ) -> Result<(), CudaError> {
        assert!(
            stride > 0 && scale.len() >= ch as usize && input.len() >= n && output.len() >= n,
            "fake_quant_pc_fwd_f32: n={n} ch={ch} stride={stride} vs input {} / scale {} / output {}",
            input.len(),
            scale.len(),
            output.len()
        );
        let func = self
            .kernels
            .get("fake_quant_pc_fwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("fake_quant_pc_fwd_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `fake_quant_pc_fwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len); the kernel guards its index (see .cu).
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(scale)
                .arg(output)
                .arg(&qn)
                .arg(&qp)
                .arg(&ch)
                .arg(&stride)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Per-output-channel LSQ fake-quant backward (GPU): STE grad_x + per-element step-size contribution.
    // ── multi-tensor training kernels: one launch over a pointer table (see train_fused.cu) ──
    pub fn mt_reduce_f32(
        &self,
        name: &str,
        ptrs: &CudaSlice<u64>,
        lens: &CudaSlice<u32>,
        bt: &CudaSlice<u32>,
        bo: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        n_blocks: usize,
    ) -> Result<(), CudaError> {
        // The per-block tables come from `MtPlan::new` (cuda_ops), which builds
        // them from the held tensors' own lengths and keeps those tensors alive.
        assert!(
            bt.len() >= n_blocks
                && bo.len() >= n_blocks
                && lens.len() == ptrs.len()
                && out.len()
                    >= if name == "mt_abssum_f32" {
                        lens.len()
                    } else {
                        1
                    },
            "mt_reduce_f32({name}): tables bt {} bo {} ptrs {} lens {} out {} for {n_blocks} blocks",
            bt.len(),
            bo.len(),
            ptrs.len(),
            lens.len(),
            out.len()
        );
        // Only the two reduce kernels share this 5-parameter shape; naming
        // them here keeps the launch statically checkable.
        let func = match name {
            "mt_sumsq_f32" => self.kernels.get("mt_sumsq_f32"),
            "mt_abssum_f32" => self.kernels.get("mt_abssum_f32"),
            other => return Err(CudaError::KernelNotFound(other.to_string())),
        }
        .ok_or_else(|| CudaError::KernelNotFound(name.to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (cuda_kernels::BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 4 * (cuda_kernels::BLOCK_SIZE / 32),
        };
        // SAFETY: `mt_abssum_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut.
        // `ptrs` is a read-only table of device pointers; the kernel
        // writes through it to buffers the caller owns and passed by address, which
        // are released only after this stream position (single stream).
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (n_blocks as u32, 1, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(ptrs)
                .arg(lens)
                .arg(bt)
                .arg(bo)
                .arg(out)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Multi-tensor ternary STE forward over an `MtPlan`: every projection in one launch.
    pub fn mt_ternarize_f32(
        &self,
        src: &CudaSlice<u64>,
        dst: &CudaSlice<u64>,
        lens: &CudaSlice<u32>,
        bt: &CudaSlice<u32>,
        bo: &CudaSlice<u32>,
        abssums: &CudaSlice<f32>,
        n_blocks: usize,
    ) -> Result<(), CudaError> {
        // The per-block tables come from `MtPlan::new` (cuda_ops), which builds
        // them from the held tensors' own lengths and keeps those tensors alive.
        assert!(
            bt.len() >= n_blocks
                && bo.len() >= n_blocks
                && src.len() == dst.len()
                && lens.len() == src.len()
                && abssums.len() >= lens.len(),
            "mt_ternarize_f32: tables bt {} bo {} src {} dst {} lens {} abssums {} for {n_blocks} blocks",
            bt.len(),
            bo.len(),
            src.len(),
            dst.len(),
            lens.len(),
            abssums.len()
        );
        let func = self
            .kernels
            .get("mt_ternarize_f32")
            .ok_or_else(|| CudaError::KernelNotFound("mt_ternarize_f32".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (cuda_kernels::BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `mt_ternarize_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // `src`, `dst` is a read-only table of device pointers; the kernel
        // writes through it to buffers the caller owns and passed by address, which
        // are released only after this stream position (single stream).
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (n_blocks as u32, 1, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(lens)
                .arg(bt)
                .arg(bo)
                .arg(abssums)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Multi-tensor in-place scale (gradient clipping) over an `MtPlan`, one launch.
    pub fn mt_scale_f32(
        &self,
        ptrs: &CudaSlice<u64>,
        lens: &CudaSlice<u32>,
        bt: &CudaSlice<u32>,
        bo: &CudaSlice<u32>,
        factor: &CudaSlice<f32>,
        n_blocks: usize,
    ) -> Result<(), CudaError> {
        // The per-block tables come from `MtPlan::new` (cuda_ops), which builds
        // them from the held tensors' own lengths and keeps those tensors alive.
        assert!(
            bt.len() >= n_blocks
                && bo.len() >= n_blocks
                && lens.len() == ptrs.len()
                && !factor.is_empty(),
            "mt_scale_f32: tables bt {} bo {} ptrs {} lens {} factor {} for {n_blocks} blocks",
            bt.len(),
            bo.len(),
            ptrs.len(),
            lens.len(),
            factor.len()
        );
        let func = self
            .kernels
            .get("mt_scale_f32")
            .ok_or_else(|| CudaError::KernelNotFound("mt_scale_f32".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (cuda_kernels::BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `mt_scale_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // `ptrs` is a read-only table of device pointers; the kernel
        // writes through it to buffers the caller owns and passed by address, which
        // are released only after this stream position (single stream).
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (n_blocks as u32, 1, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(ptrs)
                .arg(lens)
                .arg(bt)
                .arg(bo)
                .arg(factor)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    // ── trainable RMSNorm scale gradient (see train_fused.cu): per-row 1/rms, then column partials over row slices ──
    /// Per-row `1/rms` of an `m` x `n` matrix, one block per row.
    pub fn rms_inv_rows_f32(
        &self,
        inv_rms: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        assert!(
            x.len() >= m * n && inv_rms.len() >= m,
            "rms_inv_rows_f32: x {} / inv_rms {} too small for {m}x{n}",
            x.len(),
            inv_rms.len()
        );
        let func = self
            .kernels
            .get("rms_inv_rows_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rms_inv_rows_f32".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (cuda_kernels::BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 4 * (cuda_kernels::BLOCK_SIZE / 32),
        };
        // SAFETY: `rms_inv_rows_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `inv_rms`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (m as u32, 1, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(inv_rms)
                .arg(x)
                .arg(&(n as u32))
                .arg(&eps)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Partial `dL/dw` sums for a trainable RMSNorm scale, `splits` row bands per column.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_bwd_weight_partial_f32(
        &self,
        partial: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        inv_rms: &CudaSlice<f32>,
        m: usize,
        n: usize,
        rows_per_split: usize,
        splits: usize,
    ) -> Result<(), CudaError> {
        assert!(
            x.len() >= m * n
                && grad_out.len() >= m * n
                && inv_rms.len() >= m
                && partial.len() >= splits * n
                && rows_per_split * splits >= m,
            "rms_norm_bwd_weight_partial_f32: x {} grad_out {} inv_rms {} partial {} for m={m} n={n} splits={splits} rows_per_split={rows_per_split}",
            x.len(),
            grad_out.len(),
            inv_rms.len(),
            partial.len()
        );
        let func = self
            .kernels
            .get("rms_norm_bwd_weight_partial_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("rms_norm_bwd_weight_partial_f32".to_string())
            })?;
        let col_blocks = (n as u32).div_ceil(cuda_kernels::BLOCK_SIZE);
        let cfg = LaunchConfig {
            grid_dim: (col_blocks, splits as u32, 1),
            block_dim: (cuda_kernels::BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `rms_norm_bwd_weight_partial_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `partial`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (col_blocks, splits as u32, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(partial)
                .arg(x)
                .arg(grad_out)
                .arg(inv_rms)
                .arg(&(m as u32))
                .arg(&(n as u32))
                .arg(&(rows_per_split as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Upload a host table for the multi-tensor kernels.
    pub fn upload_u64(&self, v: &[u64]) -> Result<CudaSlice<u64>, CudaError> {
        self.stream
            .clone_htod(v)
            .map_err(|e| CudaError::DriverError(e.to_string()))
    }

    /// Uploads a `u32` table (multi-tensor plan metadata).
    pub fn upload_u32(&self, v: &[u32]) -> Result<CudaSlice<u32>, CudaError> {
        self.stream
            .clone_htod(v)
            .map_err(|e| CudaError::DriverError(e.to_string()))
    }

    /// Per-output-channel LSQ fake-quant backward (grad_x and the per-element scale contribution).
    #[allow(clippy::too_many_arguments)]
    pub fn fake_quant_pc_bwd_f32(
        &self,
        input: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        grad_x: &mut CudaSlice<f32>,
        gs_contrib: &mut CudaSlice<f32>,
        qn: f32,
        qp: f32,
        ch: u32,
        stride: u32,
        n: usize,
    ) -> Result<(), CudaError> {
        assert!(
            stride > 0
                && scale.len() >= ch as usize
                && input.len() >= n
                && grad_output.len() >= n
                && grad_x.len() >= n
                && gs_contrib.len() >= n,
            "fake_quant_pc_bwd_f32: n={n} ch={ch} stride={stride} vs input {} / scale {} / grad_output {} / grad_x {} / gs_contrib {}",
            input.len(),
            scale.len(),
            grad_output.len(),
            grad_x.len(),
            gs_contrib.len()
        );
        let func = self
            .kernels
            .get("fake_quant_pc_bwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("fake_quant_pc_bwd_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `fake_quant_pc_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len); the kernel guards its index (see .cu).
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(scale)
                .arg(grad_output)
                .arg(grad_x)
                .arg(gs_contrib)
                .arg(&qn)
                .arg(&qp)
                .arg(&ch)
                .arg(&stride)
                .arg(&(n as u32))
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
        // SAFETY: `scale_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `data`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `mul_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(in_dim % 256 == 0, "Q4_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemm_f32".to_string()))?;

        let total = m_dim * out_dim;
        let cfg = cuda_kernels::launch_config(total);
        // SAFETY: `q4k_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(in_dim % 256 == 0, "Q4_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemm_matched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemm_matched_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid_x = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, m_dim as u32, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q4k_gemm_matched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid_x` = (out_dim as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 256 == 0, "Q4_K GEMV requires in_dim % 256 == 0");
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
        let grid = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q4k_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = (out_dim as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
    /// Used by the axonml-serve decode path to collapse the three Q/K/V
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
        assert!(
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
        let grid = total_out.div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q4k_gemv_fused_qkv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `q_c`, `k_c`, `v_c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = total_out.div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(
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
        let total_out = (q_out + k_out + v_out) as u32;
        let grid = total_out.div_ceil(ROWS_PER_CTA);
        let reduction_bytes = 8u32 * std::mem::size_of::<f32>() as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: reduction_bytes,
        };
        // SAFETY: `q4k_gemv_fused_qkv_bias_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `q_c`, `k_c`, `v_c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = total_out.div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(
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
        let grid = total_out.div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q4k_gemv_fused_gate_up_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `gate_c`, `up_c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = total_out.div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 256 == 0, "q4k_gemv_residual: in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q4k_gemv_residual_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q4k_gemv_residual_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q4k_gemv_residual_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `x_out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = (out_dim as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 256 == 0, "fused gate/up+swiglu: in_dim % 256 == 0");
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
        let grid = (inter as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 4 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q4k_gemv_fused_gate_up_swiglu_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `ffn`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = (inter as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 256 == 0, "Q6_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q6k_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q6k_gemm_f32".to_string()))?;

        let total = m_dim * out_dim;
        let cfg = cuda_kernels::launch_config(total);
        // SAFETY: `q6k_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(in_dim % 256 == 0, "Q6_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q6k_gemm_matched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q6k_gemm_matched_f32".to_string()))?;
        const WARPS_PER_CTA: u32 = 4;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid_x = (out_dim as u32).div_ceil(WARPS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, m_dim as u32, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `q6k_gemm_matched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid_x` = (out_dim as u32).div_ceil(WARPS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 256 == 0, "Q5_K GEMV requires in_dim % 256 == 0");
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
        let grid = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q5k_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = (out_dim as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(
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
        let grid = total_out.div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q5k_gemv_fused_qkv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `q_c`, `k_c`, `v_c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = total_out.div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 256 == 0, "Q5_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q5k_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5k_gemm_f32".to_string()))?;

        let total = m_dim * out_dim;
        let cfg = cuda_kernels::launch_config(total);
        // SAFETY: `q5k_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(in_dim % 256 == 0, "Q5_K GEMM requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q5k_gemm_matched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5k_gemm_matched_f32".to_string()))?;

        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid_x = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let grid_y = m_dim as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q5k_gemm_matched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid_y` = m_dim as u32, so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 32 == 0, "Q5_0 GEMV requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_0_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_0_gemv_f32".to_string()))?;
        // v2: two warps per row (split block range), rows_per_cta=4.
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q5_0_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 32 == 0).
        // Grid: `grid` = (out_dim as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 32 == 0, "Q5_0 GEMM requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_0_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_0_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m_dim * out_dim);
        // SAFETY: `q5_0_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 32 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(in_dim % 32 == 0, "Q5_1 GEMV requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_1_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_1_gemv_f32".to_string()))?;
        // v2: two warps per row (split block range), rows_per_cta=4.
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q5_1_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 32 == 0).
        // Grid: `grid` = (out_dim as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 32 == 0, "Q5_1 GEMM requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q5_1_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q5_1_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m_dim * out_dim);
        // SAFETY: `q5_1_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 32 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(
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
        let grid = total_out.div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q5_1_gemv_fused_qkv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `q_c`, `k_c`, `v_c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 32 == 0).
        // Grid: `grid` = total_out.div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 32 == 0, "Q8_0 GEMV requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q8_0_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q8_0_gemv_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (out_dim as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q8_0_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 32 == 0).
        // Grid: `grid` = (out_dim as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        assert!(in_dim % 32 == 0, "Q8_0 GEMM requires in_dim % 32 == 0");
        let func = self
            .kernels
            .get("q8_0_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q8_0_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m_dim * out_dim);
        // SAFETY: `q8_0_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 32 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(k % 128 == 0, "I2_S GEMV requires k % 128 == 0");
        let func = self
            .kernels
            .get("i2s_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("i2s_gemv_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (n as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `i2s_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (k % 128 == 0).
        // Grid: `grid` = (n as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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

    /// PrismML Q1_0 (1-bit) GEMV — `w` is `n × n_blocks × 18` raw GGUF
    /// Q1_0 bytes (per-block fp16 scale embedded). Two warps per output
    /// row, lane-stride-32 activation reads. Same launch shape as
    /// `i2s_gemv_f32`.
    pub fn q1_0_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        assert!(k % 128 == 0, "Q1_0 GEMV requires k % 128 == 0");
        let func = self
            .kernels
            .get("q1_0_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q1_0_gemv_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (n as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q1_0_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (k % 128 == 0).
        // Grid: `grid` = (n as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(n as u32))
                .arg(&(k as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// PrismML Q1_0 DP4A path — online int8 activation quant.
    ///
    /// Quantizes a f32 activation row `a [k]` to int8 with per-32-chunk
    /// fp16 scales. One warp per 32-element chunk, lane-stride-1
    /// coalesced loads/stores. Produces buffers consumed by
    /// `q1_0_gemv_dp4a_f32`.
    pub fn q1_0_quantize_acts_q8(
        &self,
        a: &CudaSlice<f32>,
        a_q: &mut CudaSlice<u8>,
        a_d: &mut CudaSlice<u16>,
        k: usize,
    ) -> Result<(), CudaError> {
        assert!(k % 32 == 0, "q8 act-quant requires k % 32 == 0");
        let func = self
            .kernels
            .get("q1_0_quantize_acts_q8")
            .ok_or_else(|| CudaError::KernelNotFound("q1_0_quantize_acts_q8".to_string()))?;
        let n_chunks = (k / 32) as u32;
        const THREADS_PER_CTA: u32 = 128;
        let warps_per_cta: u32 = THREADS_PER_CTA / 32;
        let grid = n_chunks.div_ceil(warps_per_cta);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `q1_0_quantize_acts_q8`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `a_q_u`, `a_d`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (k % 32 == 0).
        // Grid: `grid` = n_chunks.div_ceil(warps_per_cta), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(a_q)
                .arg(a_d)
                .arg(&(k as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q1_0 DP4A GEMV — sign-bit weights × int8 activations via `__dp4a`.
    ///
    /// Same launch shape as `q1_0_gemv_f32` (two warps per output row,
    /// 4 rows per CTA). Caller must run `q1_0_quantize_acts_q8` on the
    /// activation row first to produce `a_q` + `a_d`.
    pub fn q1_0_gemv_dp4a_f32(
        &self,
        w: &CudaSlice<u8>,
        a_q: &CudaSlice<u8>,
        a_d: &CudaSlice<u16>,
        c: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        assert!(k % 128 == 0, "Q1_0 DP4A GEMV requires k % 128 == 0");
        let func = self
            .kernels
            .get("q1_0_gemv_dp4a_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q1_0_gemv_dp4a_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (n as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `q1_0_gemv_dp4a_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (k % 128 == 0).
        // Grid: `grid` = (n as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a_q)
                .arg(a_d)
                .arg(c)
                .arg(&(n as u32))
                .arg(&(k as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Q1_0 fused single-launch DP4A GEMV.
    ///
    /// Caller passes f32 activations; the kernel quantizes them into
    /// shared memory then runs the dp4a matmul. Same launch shape as
    /// `q1_0_gemv_dp4a_f32` (4 rows per CTA, 2 warps per row), with
    /// dynamic smem = `k + (k/32)*2 + rows_per_cta * 2 * 4` bytes.
    pub fn q1_0_gemv_fused_dp4a_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        assert!(k % 128 == 0, "Q1_0 fused DP4A GEMV requires k % 128 == 0");
        let func = self
            .kernels
            .get("q1_0_gemv_fused_dp4a_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q1_0_gemv_fused_dp4a_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (n as u32).div_ceil(ROWS_PER_CTA);
        // smem layout: k bytes int8 acts + (k/32)*2 bytes fp16 scales
        //            + rows_per_cta * 2 * 4 bytes partials
        let smem_bytes = (k as u32) + ((k as u32) / 32) * 2 + ROWS_PER_CTA * 2 * 4;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        // SAFETY: `q1_0_gemv_fused_dp4a_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (k % 128 == 0).
        // Grid: `grid` = (n as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(n as u32))
                .arg(&(k as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// PrismML Q1_0 GEMM (m > 1 prefill). Naive one-thread-per-output;
    /// same shape as `i2s_gemm_f32`.
    pub fn q1_0_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        assert!(k % 128 == 0, "Q1_0 GEMM requires k % 128 == 0");
        let func = self
            .kernels
            .get("q1_0_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q1_0_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m * n);
        // SAFETY: `q1_0_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (k % 128 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(a)
                .arg(c)
                .arg(&(m as u32))
                .arg(&(n as u32))
                .arg(&(k as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Quantize an f32 shadow weight tensor to ternary `{-1, 0, +1}` i8
    /// in-place on GPU. Two kernel launches: an abssum reduction
    /// followed by a per-element threshold using the resulting absmean.
    /// Returns the f32 scale (= absmean, clamped to ≥ 1e-8) the caller
    /// will multiply by during the matmul.
    ///
    /// `out_i8` must already be allocated with capacity ≥ `n` bytes.
    pub fn ternary_quantize_weights(
        &self,
        shadow: &CudaSlice<f32>,
        out_i8: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<f32, CudaError> {
        let reduce_func = self
            .kernels
            .get("f32_abssum_reduce")
            .ok_or_else(|| CudaError::KernelNotFound("f32_abssum_reduce".to_string()))?;
        let quant_func = self
            .kernels
            .get("f32_quantize_ternary")
            .ok_or_else(|| CudaError::KernelNotFound("f32_quantize_ternary".to_string()))?;

        const THREADS_PER_CTA: u32 = 256;
        let n_u32 = n as u32;
        let grid = n_u32.div_ceil(THREADS_PER_CTA);

        // Stage 1 — sum |w| into a single device scalar.
        let mut sum_buf: CudaSlice<f32> =
            self.stream.alloc_zeros::<f32>(1).map_err(CudaError::from)?;
        let cfg_reduce = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: (THREADS_PER_CTA / 32) * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `f32_abssum_reduce`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out_sum`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: `grid` = n_u32.div_ceil(THREADS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
        unsafe {
            self.stream
                .launch_builder(reduce_func)
                .arg(shadow)
                .arg(&mut sum_buf)
                .arg(&n_u32)
                .launch(cfg_reduce)
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }

        // Pull the single scalar back to host. `clone_dtoh` is implicitly
        // stream-ordered (replaces the deprecated `memcpy_dtov`).
        let sum_host: Vec<f32> = self.stream.clone_dtoh(&sum_buf).map_err(CudaError::from)?;
        let abs_mean = sum_host[0] / (n as f32);
        let scale = abs_mean.max(1e-8);

        // Stage 2 — threshold each element to ±1 / 0 i8.
        let cfg_quant = cuda_kernels::launch_config(n);
        // SAFETY: `f32_quantize_ternary`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out_i8`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: `grid` = n_u32.div_ceil(THREADS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
        unsafe {
            self.stream
                .launch_builder(quant_func)
                .arg(shadow)
                .arg(out_i8)
                .arg(&n_u32)
                .arg(&scale)
                .launch(cfg_quant)
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }

        Ok(scale)
    }

    /// Raw-i8 ternary GEMV (m=1 decode/training-decode).
    pub fn ternary_gemv_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        scale: f32,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("ternary_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("ternary_gemv_f32".to_string()))?;
        const ROWS_PER_CTA: u32 = 4;
        const WARPS_PER_CTA: u32 = ROWS_PER_CTA * 2;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (n as u32).div_ceil(ROWS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: ROWS_PER_CTA * 2 * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: `ternary_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: `grid` = (n as u32).div_ceil(ROWS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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

    /// Raw-i8 ternary GEMM (m>1 prefill / training fwd).
    pub fn ternary_gemm_f32(
        &self,
        w: &CudaSlice<u8>,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("ternary_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("ternary_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m * n);
        // SAFETY: `ternary_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// Raw-i8 ternary backward grad_input — `scale * ternary^T @ grad_output`.
    pub fn ternary_grad_input_f32(
        &self,
        w: &CudaSlice<u8>,
        grad_out: &CudaSlice<f32>,
        grad_in: &mut CudaSlice<f32>,
        scale: f32,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("ternary_grad_input_f32")
            .ok_or_else(|| CudaError::KernelNotFound("ternary_grad_input_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(batch_size * in_features);
        // SAFETY: `ternary_grad_input_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_in`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(w)
                .arg(grad_out)
                .arg(grad_in)
                .arg(&scale)
                .arg(&(batch_size as u32))
                .arg(&(in_features as u32))
                .arg(&(out_features as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Raw-i8 ternary backward grad_bias — sum over batch axis.
    pub fn ternary_grad_bias_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        grad_bias: &mut CudaSlice<f32>,
        batch_size: usize,
        out_features: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("ternary_grad_bias_f32")
            .ok_or_else(|| CudaError::KernelNotFound("ternary_grad_bias_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(out_features);
        // SAFETY: `ternary_grad_bias_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_bias`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(grad_bias)
                .arg(&(batch_size as u32))
                .arg(&(out_features as u32))
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
        assert!(k % 128 == 0, "I2_S GEMM requires k % 128 == 0");
        let func = self
            .kernels
            .get("i2s_gemm_f32")
            .ok_or_else(|| CudaError::KernelNotFound("i2s_gemm_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(m * n);
        // SAFETY: `i2s_gemm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (k % 128 == 0).
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        assert!(in_dim % 256 == 0, "Q6_K GEMV requires in_dim % 256 == 0");
        let func = self
            .kernels
            .get("q6k_gemv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("q6k_gemv_f32".to_string()))?;

        const WARPS_PER_CTA: u32 = 4;
        const THREADS_PER_CTA: u32 = WARPS_PER_CTA * 32;
        let grid = (out_dim as u32).div_ceil(WARPS_PER_CTA);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_CTA, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `q6k_gemv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `c`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (in_dim % 256 == 0).
        // Grid: `grid` = (out_dim as u32).div_ceil(WARPS_PER_CTA), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        // SAFETY: `relu_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `sigmoid_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `tanh_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `sub_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `div_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_add_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_sub_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_mul_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_div_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_add_rev_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_sub_rev_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_mul_rev_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `broadcast_div_rev_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `neg_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
            .get("pow_f32_c99")
            .ok_or_else(|| CudaError::KernelNotFound("pow_f32_c99".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        // SAFETY: `pow_f32_c99`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `dst`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
            .get("pow_scalar_f32_c99")
            .ok_or_else(|| CudaError::KernelNotFound("pow_scalar_f32_c99".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        // SAFETY: `pow_scalar_f32_c99`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `dst`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `exp_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `log_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `sqrt_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `gelu_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `silu_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `silu_backward_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `add_scalar_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `data`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `relu_backward_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `sigmoid_backward_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `tanh_backward_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `sum_dim_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `softmax_row_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
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
        // SAFETY: `broadcast_copy_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `layer_norm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
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
        // SAFETY: `softmax_backward_row_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
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
        assert!(
            d_input.len() >= num_rows * norm_size
                && grad_output.len() >= num_rows * norm_size
                && input.len() >= num_rows * norm_size
                && gamma.len() >= norm_size,
            "layer_norm_backward_dinput_f32: d_input {} grad_output {} input {} gamma {} for {num_rows}x{norm_size}",
            d_input.len(),
            grad_output.len(),
            input.len(),
            gamma.len()
        );
        let func = self
            .kernels
            .get("layer_norm_backward_dinput_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("layer_norm_backward_dinput_f32".to_string())
            })?;
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * 4 * 2,
        };
        // SAFETY: `layer_norm_backward_dinput_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (num_rows as u32, 1, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
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
        // SAFETY: `layer_norm_backward_dweight_dbias_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `gather_contiguous_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `embedding_scatter_add_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `adam_step_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `grad_norm_sq_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `grad_scale_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `cross_entropy_fwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
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
        // SAFETY: `cross_entropy_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        let src_end = src_offset.checked_add(count);
        let dst_end = dst_offset.checked_add(count);
        if !matches!(src_end, Some(e) if e <= src.len())
            || !matches!(dst_end, Some(e) if e <= dst.len())
        {
            return Err(CudaError::CopyFailed);
        }
        use cudarc::driver::DevicePtr as _;
        let (src_ptr, _guard_s) = src.device_ptr(&self.stream);
        let src_ptr =
            src_ptr + (src_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
        use cudarc::driver::DevicePtrMut as _;
        let (dst_ptr, _guard_d) = dst.device_ptr_mut(&self.stream);
        let dst_ptr =
            dst_ptr + (dst_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
        let size = count * std::mem::size_of::<f32>();
        // SAFETY: the range checks at the top of this fn proved
        // src_offset + count <= src.len() and dst_offset + count <= dst.len(),
        // so both offset pointers plus `size` bytes stay inside their slices.
        // The copy is synchronous and both slices are borrowed across it.
        unsafe {
            cudarc::driver::result::memcpy_dtod_sync(dst_ptr, src_ptr, size)
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Stream-ordered device-to-device copy on `self.stream` (async), unlike `memcpy_dtod_f32` which
    /// runs `memcpy_dtod_sync` on the NULL stream. Use this when the source is written by a compute-stream
    /// kernel (e.g. a gemm) and the copy must be ordered AFTER it — the null-stream sync copy does NOT wait
    /// for a non-blocking compute stream, so it reads STALE data once the pool is dirtied.
    pub fn memcpy_dtod_f32_stream(
        &self,
        dst: &mut CudaSlice<f32>,
        dst_offset: usize,
        src: &CudaSlice<f32>,
        src_offset: usize,
        count: usize,
    ) -> Result<(), CudaError> {
        use cudarc::driver::DevicePtr as _;
        use cudarc::driver::DevicePtrMut as _;
        let fits = |off: usize, len: usize| off.checked_add(count).is_some_and(|e| e <= len);
        assert!(
            fits(src_offset, src.len()) && fits(dst_offset, dst.len()),
            "memcpy_dtod_f32_stream: {count} at src {src_offset} (len {}) / dst {dst_offset} (len {}) out of bounds",
            src.len(),
            dst.len()
        );
        let (src_ptr, _guard_s) = src.device_ptr(&self.stream);
        let src_ptr =
            src_ptr + (src_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
        let (dst_ptr, _guard_d) = dst.device_ptr_mut(&self.stream);
        let dst_ptr =
            dst_ptr + (dst_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
        let size = count * std::mem::size_of::<f32>();
        // SAFETY: `offset + count` fits in both slices (asserted), the
        // borrows outlive the call, and the copy is queued on the backend's
        // single stream behind everything before it.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dst_ptr,
                src_ptr,
                size,
                self.stream.cu_stream(),
            )
            .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

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
        // SAFETY: `mask_expand_causal_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `mask_expand_padding_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `strided_gather_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `lstm_gates_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `h_new`, `c_new`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `lstm_gates_backward_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_gates`, `grad_c_prev`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `gru_gates_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `h_new`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `gru_gates_backward_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_gates_ih`, `grad_gates_hh`, `grad_h_prev`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `batchnorm_stats_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `sum_out`, `sum_sq_out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `batchnorm_norm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `y`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// BatchNorm2d backward pass 1: per-channel `sum_grad` / `sum_grad_xhat`.
    /// `sum_grad` and `sum_grad_xhat` MUST be zero-initialized before calling.
    pub fn batchnorm_bwd_reduce_f32(
        &self,
        grad: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        mean: &CudaSlice<f32>,
        var: &CudaSlice<f32>,
        sum_grad: &mut CudaSlice<f32>,
        sum_grad_xhat: &mut CudaSlice<f32>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<(), CudaError> {
        assert!(
            grad.len() >= n * c * spatial
                && x.len() >= n * c * spatial
                && mean.len() >= c
                && var.len() >= c
                && sum_grad.len() >= c
                && sum_grad_xhat.len() >= c,
            "batchnorm_bwd_reduce_f32: grad {} x {} mean {} var {} sum_grad {} sum_grad_xhat {} for {n}x{c}x{spatial}",
            grad.len(),
            x.len(),
            mean.len(),
            var.len(),
            sum_grad.len(),
            sum_grad_xhat.len()
        );
        let func = self
            .kernels
            .get("batchnorm_bwd_reduce_f32")
            .ok_or_else(|| CudaError::KernelNotFound("batchnorm_bwd_reduce_f32".to_string()))?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (c as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `batchnorm_bwd_reduce_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `sum_grad`, `sum_grad_xhat`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (c as u32, 1, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad)
                .arg(x)
                .arg(mean)
                .arg(var)
                .arg(sum_grad)
                .arg(sum_grad_xhat)
                .arg(&eps)
                .arg(&(n as u32))
                .arg(&(c as u32))
                .arg(&(spatial as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// BatchNorm2d backward pass 2: elementwise `grad_input`.
    pub fn batchnorm_bwd_input_f32(
        &self,
        grad: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        mean: &CudaSlice<f32>,
        var: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        sum_grad: &CudaSlice<f32>,
        sum_grad_xhat: &CudaSlice<f32>,
        grad_input: &mut CudaSlice<f32>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("batchnorm_bwd_input_f32")
            .ok_or_else(|| CudaError::KernelNotFound("batchnorm_bwd_input_f32".to_string()))?;
        let total = n * c * spatial;
        let cfg = cuda_kernels::launch_config(total);
        // SAFETY: `batchnorm_bwd_input_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad)
                .arg(x)
                .arg(mean)
                .arg(var)
                .arg(gamma)
                .arg(sum_grad)
                .arg(sum_grad_xhat)
                .arg(grad_input)
                .arg(&eps)
                .arg(&(n as u32))
                .arg(&(c as u32))
                .arg(&(spatial as u32))
                .arg(&(total as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

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
        let is_causal_u32: u32 = u32::from(is_causal);
        // SAFETY: `fused_attention_fwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `O`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

        // SAFETY: `fused_attn_prefill_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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
        assert!(
            head_dim <= 512,
            "fused_attn_decode_f32: head_dim {head_dim} exceeds kernel MAX_DIMS budget"
        );
        assert!(
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

        // SAFETY: `fused_attn_decode_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (head_dim <= 512; n_kv_heads > 0 && n_heads % n_kv_heads == 0).
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
        assert!(
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
        // SAFETY: `quantize_kv_row_q8_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `dst_q`, `dst_scale`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (head_dim <= 512).
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
        assert!(
            head_dim <= 512,
            "fused_attn_decode_q8_f32: head_dim {head_dim} exceeds MAX_DIMS budget"
        );
        assert!(
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
        // SAFETY: `fused_attn_decode_q8_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (head_dim <= 512; n_kv_heads > 0 && n_heads % n_kv_heads == 0).
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
        let n_warps = block.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_warps * 4,
        };
        // SAFETY: `rms_norm_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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

    /// Single-token LayerNorm over a single vector of length `n`:
    /// `out[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]`. Used by
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
        let n_warps = block.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_warps * 4 * 2,
        };
        // SAFETY: `layer_norm_tokenwise_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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
        // SAFETY: `gelu_tanh_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `scaled_add_inplace_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `dst`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `parallel_residual_add_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `x`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `rms_norm_heads_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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
        assert!(
            head_dim % 2 == 0,
            "head_dim must be even for split-halves RoPE"
        );
        let func = self
            .kernels
            .get("rope_split_halves_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rope_split_halves_f32".to_string()))?;
        let half = (head_dim / 2) as u32;
        let block: u32 = half.min(128);
        let grid_y = half.div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `rope_split_halves_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (head_dim % 2 == 0).
        // Grid: `grid_y` = half.div_ceil(block), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        let grid: u32 = (n as u32).div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `swiglu_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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

    /// SwiGLU backward: produces `grad_gate` and `grad_up` given saved
    /// forward inputs `gate`, `up` and upstream gradient `grad_out`. Replaces
    /// the separate SiluBackward + MulBackward kernel pair on the MLP path.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_bwd_f32(
        &self,
        grad_gate: &mut CudaSlice<f32>,
        grad_up: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("swiglu_bwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("swiglu_bwd_f32".to_string()))?;
        let block: u32 = 256;
        let grid: u32 = (n as u32).div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `swiglu_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_gate`, `grad_up`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_gate)
                .arg(grad_up)
                .arg(gate)
                .arg(up)
                .arg(grad_out)
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
        let grid: u32 = (n as u32).div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `relu2_gate_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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
        let n_warps = block.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_warps * 4,
        };
        // SAFETY: `rms_norm_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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

    /// Fused causal-scaled softmax. Replaces the `mul_scalar(scale) +
    /// broadcast_add(causal_mask) + softmax_row` kernel sequence with a
    /// single launch. `scores`/`out` shape is `[num_rows, tk]` flattened
    /// from `[B, H, Tq, Tk]` (num_rows = B*H*Tq). `q_pos = row_idx % tq`.
    pub fn softmax_causal_scaled_f32(
        &self,
        out: &mut CudaSlice<f32>,
        scores: &CudaSlice<f32>,
        num_rows: usize,
        tq: usize,
        tk: usize,
        offset: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("softmax_causal_scaled_f32")
            .ok_or_else(|| CudaError::KernelNotFound("softmax_causal_scaled_f32".to_string()))?;
        let block: u32 = 256;
        let n_warps = block.div_ceil(32);
        // Reuses one shmem buffer across two reductions; n_warps * 4 bytes is enough.
        let shmem = n_warps * 4;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shmem,
        };
        // SAFETY: `softmax_causal_scaled_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(scores)
                .arg(&(tq as u32))
                .arg(&(tk as u32))
                .arg(&(offset as u32))
                .arg(&scale)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Fused causal-scaled softmax backward wrt raw scores.
    pub fn softmax_causal_scaled_bwd_f32(
        &self,
        grad_scores: &mut CudaSlice<f32>,
        p: &CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        num_rows: usize,
        tk: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("softmax_causal_scaled_bwd_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("softmax_causal_scaled_bwd_f32".to_string())
            })?;
        let block: u32 = 256;
        let n_warps = block.div_ceil(32);
        let shmem = n_warps * 4;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shmem,
        };
        // SAFETY: `softmax_causal_scaled_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_scores`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_scores)
                .arg(p)
                .arg(grad_out)
                .arg(&(tk as u32))
                .arg(&scale)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Head-major split-halves RoPE backward. Inverse rotation of
    /// `rope_split_halves_bhsd_f32`. Input/output `[bs, n_heads, seq, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_split_halves_bhsd_bwd_f32(
        &self,
        grad_in: &mut CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        bs: usize,
        n_heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("rope_split_halves_bhsd_bwd_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("rope_split_halves_bhsd_bwd_f32".to_string())
            })?;
        let half = (head_dim / 2) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (seq as u32, n_heads as u32, bs as u32),
            block_dim: (half, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `rope_split_halves_bhsd_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_in`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(grad_in)
                .arg(&(seq as u32))
                .arg(&(n_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&theta)
                .arg(&(pos_start as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// GQA repeat_kv: duplicate each KV head `n_rep` times consecutively.
    /// Input shape `[bs, kv_heads, seq, head_dim]`, output
    /// `[bs, kv_heads * n_rep, seq, head_dim]`. Single kernel, no H2D/D2H.
    #[allow(clippy::too_many_arguments)]
    pub fn repeat_kv_f32(
        &self,
        out: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        bs: usize,
        kv_heads: usize,
        n_rep: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("repeat_kv_f32")
            .ok_or_else(|| CudaError::KernelNotFound("repeat_kv_f32".to_string()))?;
        let total = bs * kv_heads * n_rep * seq * head_dim;
        let block: u32 = 256;
        let grid: u32 = (total as u32).div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `repeat_kv_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(out)
                .arg(&(bs as u32))
                .arg(&(kv_heads as u32))
                .arg(&(n_rep as u32))
                .arg(&(seq as u32))
                .arg(&(head_dim as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Head-major split-halves RoPE for Qwen3/LLaMA training forward.
    /// Input/output shape `[bs, n_heads, seq, head_dim]` row-major on GPU.
    /// `src` and `out` may alias for in-place rotation.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_split_halves_bhsd_f32(
        &self,
        out: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        bs: usize,
        n_heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("rope_split_halves_bhsd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rope_split_halves_bhsd_f32".to_string()))?;
        let half = (head_dim / 2) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (seq as u32, n_heads as u32, bs as u32),
            block_dim: (half, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `rope_split_halves_bhsd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(out)
                .arg(&(seq as u32))
                .arg(&(n_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&theta)
                .arg(&(pos_start as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Fused residual-add + batched RMSNorm. For each row t in [0, m),
    /// computes `sum[t, :] = a[t, :] + b[t, :]` and
    /// `out[t, :] = sum[t, :] * weight / sqrt(mean(sum[t, :]²) + eps)`.
    /// Replaces the separate `broadcast_add_f32 + rms_norm_batched_f32`
    /// kernel pair in Qwen3's decoder layer residual path.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_batched_f32(
        &self,
        out: &mut CudaSlice<f32>,
        sum_out: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("add_rmsnorm_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("add_rmsnorm_batched_f32".to_string()))?;
        let block: u32 = 256;
        let n_warps = block.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_warps * 4,
        };
        // SAFETY: `add_rmsnorm_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`, `sum_out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(out)
                .arg(sum_out)
                .arg(a)
                .arg(b)
                .arg(weight)
                .arg(&(n as u32))
                .arg(&eps)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))
        }
    }

    /// Batched RMSNorm backward (grad_input only): for each row t in [0, m),
    /// computes `grad_x[t, :] = w/rms * grad_y[t, :] - x[t, :]/(rms³·n) · Σ(x·w·grad_y)`.
    /// x, grad_out shape [m, n]; weight [n]; output grad_input [m, n] all contiguous.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_bwd_batched_f32(
        &self,
        grad_input: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("rms_norm_bwd_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("rms_norm_bwd_batched_f32".to_string()))?;
        let block: u32 = 256;
        let n_warps = block.div_ceil(32);
        // Two reductions per row (sum_sq + dot), so 2 × n_warps × 4 bytes of shmem.
        let shmem = 2 * n_warps * 4;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shmem,
        };
        // SAFETY: `rms_norm_bwd_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_input)
                .arg(x)
                .arg(weight)
                .arg(grad_out)
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
        // SAFETY: `rms_norm_heads_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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
        assert!(
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
        let grid_y = half.div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, m as u32),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `rope_split_halves_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Bounds: asserted above (head_dim % 2 == 0).
        // Grid: `grid_y` = half.div_ceil(block), so every thread that indexes lands inside
        // buffers the caller sized to those same dimensions.
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
        let grid: u32 = total.div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `add_bias_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
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
        let is_causal_u32: u32 = u32::from(is_causal);
        // SAFETY: `fused_attention_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_Q`, `grad_K`, `grad_V`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `im2col_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// Batched im2col — whole batch in ONE launch.
    ///
    /// `input` is `[batch, C_in, H, W]`, `col` is `[batch, C_in*kH*kW, oH*oW]` (per-batch blocks are
    /// contiguous with stride `col_n`, identical to the single-image layout, so a strided-batched
    /// GEMM addresses them directly). `params` is `u32[11]` = the single-image `u32[10]` plus `C_in`.
    /// `n` = `batch * C_in*kH*kW*oH*oW`.
    pub fn im2col_batched_f32(
        &self,
        input: &CudaSlice<f32>,
        col: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("im2col_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("im2col_batched_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `im2col_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `col`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// Direct depthwise conv forward (groups == C_in == C_out). `params` is `u32[12]` =
    /// `{H, W, kH, kW, pH, pW, sH, sW, oH, oW, C, batch}`. Bias is applied separately.
    /// `n` = `batch * C * oH * oW`.
    pub fn depthwise_fwd_f32(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("depthwise_fwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("depthwise_fwd_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `depthwise_fwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(weight)
                .arg(output)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Direct depthwise grad_input. Gathers, so `grad_in` needs no pre-zeroing.
    /// `n` = `batch * C * H * W`.
    pub fn depthwise_grad_input_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        grad_in: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("depthwise_grad_input_f32")
            .ok_or_else(|| CudaError::KernelNotFound("depthwise_grad_input_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `depthwise_grad_input_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_in`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(weight)
                .arg(grad_in)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Direct depthwise grad_weight, ACCUMULATING into `grad_w`. `n` = `C * kH * kW`.
    pub fn depthwise_grad_weight_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        grad_w: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("depthwise_grad_weight_f32")
            .ok_or_else(|| CudaError::KernelNotFound("depthwise_grad_weight_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `depthwise_grad_weight_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_w`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(input)
                .arg(grad_w)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Grouped/depthwise im2col for the whole batch AND all groups in ONE launch.
    /// `col` is laid out group-major: `[groups][batch][icg*kH*kW][oH*oW]`.
    /// `params` is `u32[13]` = the single-image `u32[10]` plus `icg`, `C_in_total`, `batch`.
    pub fn im2col_group_batched_f32(
        &self,
        input: &CudaSlice<f32>,
        col: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("im2col_group_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("im2col_group_batched_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `im2col_group_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `col`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// Fused elementwise-mul backward: grad_lhs = grad_out*rhs, grad_rhs = grad_out*lhs, one launch.
    pub fn mul_backward_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        lhs: &CudaSlice<f32>,
        rhs: &CudaSlice<f32>,
        grad_lhs: &mut CudaSlice<f32>,
        grad_rhs: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("mul_backward_f32")
            .ok_or_else(|| CudaError::KernelNotFound("mul_backward_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `mul_backward_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_lhs`, `grad_rhs`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(lhs)
                .arg(rhs)
                .arg(grad_lhs)
                .arg(grad_rhs)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// ConvTranspose2d backward wrt input (thread per input element). params = u32[13].
    pub fn convtranspose2d_bwd_input_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        grad_in: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("convtranspose2d_bwd_input_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("convtranspose2d_bwd_input_f32".to_string())
            })?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `convtranspose2d_bwd_input_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_in`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(weight)
                .arg(grad_in)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// ConvTranspose2d backward wrt weight (thread per weight element). params = u32[13].
    pub fn convtranspose2d_bwd_weight_f32(
        &self,
        input: &CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        grad_w: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("convtranspose2d_bwd_weight_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("convtranspose2d_bwd_weight_f32".to_string())
            })?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `convtranspose2d_bwd_weight_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_w`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(grad_out)
                .arg(grad_w)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// GroupNorm backward pass 1: one block per (batch, group) -> stats[batch*groups][4]
    /// {mean, std_inv, sum_dy, sum_dy_xhat}.
    #[allow(clippy::too_many_arguments)]
    pub fn groupnorm_bwd_stats_f32(
        &self,
        input: &CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        dims: (usize, usize, usize, usize),
        eps: f32,
        stats: &mut CudaSlice<f32>,
        num_bg: usize,
    ) -> Result<(), CudaError> {
        // `dims` = (batch, channels, spatial, groups) is what `params` holds on
        // the device; the caller uploads both from the same values, and this
        // bounds every slice by them on the host.
        let (batch, channels, spatial, groups) = dims;
        assert!(
            groups > 0
                && channels % groups == 0
                && num_bg == batch * groups
                && params.len() >= 4
                && input.len() >= batch * channels * spatial
                && grad_out.len() >= batch * channels * spatial
                && weight.len() >= channels
                && stats.len() >= num_bg * 4,
            "groupnorm_bwd_stats_f32: input {} grad_out {} weight {} params {} stats {} for dims {dims:?}, num_bg {num_bg}",
            input.len(),
            grad_out.len(),
            weight.len(),
            params.len(),
            stats.len()
        );
        let func = self
            .kernels
            .get("groupnorm_bwd_stats_f32")
            .ok_or_else(|| CudaError::KernelNotFound("groupnorm_bwd_stats_f32".to_string()))?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_bg as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `groupnorm_bwd_stats_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `stats`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: (num_bg as u32, 1, 1); one block per row/tensor, the kernel loops to the length
        // argument it is given and never past it.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(grad_out)
                .arg(weight)
                .arg(params)
                .arg(&eps)
                .arg(stats)
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// GroupNorm backward pass 2: one thread per element -> d_input, atomicAdd d_weight/d_bias.
    /// d_weight and d_bias MUST be pre-zeroed.
    #[allow(clippy::too_many_arguments)]
    pub fn groupnorm_bwd_apply_f32(
        &self,
        input: &CudaSlice<f32>,
        grad_out: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        stats: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        d_input: &mut CudaSlice<f32>,
        d_weight: &mut CudaSlice<f32>,
        d_bias: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("groupnorm_bwd_apply_f32")
            .ok_or_else(|| CudaError::KernelNotFound("groupnorm_bwd_apply_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `groupnorm_bwd_apply_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `d_input`, `d_weight`, `d_bias`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(input)
                .arg(grad_out)
                .arg(weight)
                .arg(stats)
                .arg(params)
                .arg(d_input)
                .arg(d_weight)
                .arg(d_bias)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// AdaptiveAvgPool2d backward: one thread per input element gathers `grad_out/count` from its
    /// owning window. `params` = u32[6] {batch, channels, in_h, in_w, out_h, out_w}, `n` = input numel.
    pub fn adaptive_avgpool2d_bwd_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        grad_in: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("adaptive_avgpool2d_bwd_f32")
            .ok_or_else(|| CudaError::KernelNotFound("adaptive_avgpool2d_bwd_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `adaptive_avgpool2d_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_in`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(grad_in)
                .arg(params)
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// `grad_in[idx[o]] += grad_out[o]` for all outputs, in one launch. `grad_in` must be pre-zeroed.
    /// Backs index-scatter gradients (pooling backward). `n` = number of outputs.
    pub fn scatter_add_u32_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        grad_in: &mut CudaSlice<f32>,
        in_numel: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("scatter_add_u32_f32")
            .ok_or_else(|| CudaError::KernelNotFound("scatter_add_u32_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `scatter_add_u32_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_in`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(idx)
                .arg(grad_in)
                .arg(&(in_numel as u32))
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// `dst[o*block_dst + dst_offset + i] = src[o*block_src + i]` in ONE launch.
    /// Replaces a per-outer-block `memcpy_dtod` loop (see `narrow_backward_cuda`).
    /// `n` = `outer * block_src`.
    pub fn strided_block_copy_f32(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        block_src: usize,
        block_dst: usize,
        dst_offset: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("strided_block_copy_f32")
            .ok_or_else(|| CudaError::KernelNotFound("strided_block_copy_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `strided_block_copy_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `dst`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src)
                .arg(dst)
                .arg(&(block_src as u32))
                .arg(&(block_dst as u32))
                .arg(&(dst_offset as u32))
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Grouped/depthwise col2im for the whole batch AND all groups in ONE launch.
    /// `output` MUST be zero-initialised. Layout mirrors `im2col_group_batched_f32`.
    pub fn col2im_group_batched_f32(
        &self,
        col: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("col2im_group_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("col2im_group_batched_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `col2im_group_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// `sum_batch_f32` writing at an element offset in `out` — grouped grad_weight folds each
    /// group's per-batch partials into that group's slice of the full weight gradient.
    pub fn sum_batch_at_f32(
        &self,
        partial: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        out_offset: usize,
        len: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sum_batch_at_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sum_batch_at_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        // SAFETY: `sum_batch_at_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(partial)
                .arg(out)
                .arg(&(out_offset as u32))
                .arg(&(len as u32))
                .arg(&(batch as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched col2im — whole batch in ONE launch. `output` MUST be zero-initialised.
    pub fn col2im_batched_f32(
        &self,
        col: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        params: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("col2im_batched_f32")
            .ok_or_else(|| CudaError::KernelNotFound("col2im_batched_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `col2im_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

    /// Reduce per-batch partials into an accumulator: `out[i] += sum_b partial[b*len + i]`.
    /// A strided-batched GEMM must write disjoint `C` blocks, so grad_weight emits per-batch
    /// partials and folds them here — preserving the original `beta = 1.0` accumulate semantics.
    pub fn sum_batch_f32(
        &self,
        partial: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        len: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sum_batch_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sum_batch_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(len);
        // SAFETY: `sum_batch_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `out`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(partial)
                .arg(out)
                .arg(&(len as u32))
                .arg(&(batch as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched per-channel bias add over `[batch, C_out, spatial]`.
    /// `bias_add_channels_f32` computes `channel = i / spatial` with NO wrap, so it cannot be fed a
    /// whole batch — it would index `bias` past `C_out`. This wraps per batch element.
    pub fn bias_add_channels_batched_f32(
        &self,
        data: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        spatial: usize,
        out_channels: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("bias_add_channels_batched_f32")
            .ok_or_else(|| {
                CudaError::KernelNotFound("bias_add_channels_batched_f32".to_string())
            })?;
        let cfg = cuda_kernels::launch_config(n);
        // SAFETY: `bias_add_channels_batched_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `data`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(data)
                .arg(bias)
                .arg(&(spatial as u32))
                .arg(&(out_channels as u32))
                .arg(&(n as u32))
                .launch(cfg)
                .map(|_| ())
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
        }
        Ok(())
    }

    /// Reduce `grad_out [batch, C_out, spatial]` into `grad_bias [C_out]` (accumulating) on-stream.
    /// Replaces a host-side sum fed by an unsynchronised `grad_out.to_vec()`, which could read the
    /// tensor before the producing kernel had landed — intermittently wrong bias gradients.
    pub fn sum_bias_f32(
        &self,
        grad_out: &CudaSlice<f32>,
        grad_bias: &mut CudaSlice<f32>,
        spatial: usize,
        out_channels: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let func = self
            .kernels
            .get("sum_bias_f32")
            .ok_or_else(|| CudaError::KernelNotFound("sum_bias_f32".to_string()))?;
        let cfg = cuda_kernels::launch_config(out_channels);
        // SAFETY: `sum_bias_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_bias`; each arrives as &mut.
        // Event tracking is disabled on this context (see `new`), so ordering comes
        // from the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(grad_out)
                .arg(grad_bias)
                .arg(&(spatial as u32))
                .arg(&(out_channels as u32))
                .arg(&(batch as u32))
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
        // SAFETY: `col2im_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `bias_add_channels_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX by tools/check_launches.py on every CI run. This
        // kernel is inline PTX with no .cu, so which arguments it writes is read
        // from the PTX body, not from a const qualifier.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the PTX (setp/bra on the thread index)), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `maxpool2d_fwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`, `indices`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `maxpool2d_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `avgpool2d_fwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `output`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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
        // SAFETY: `avgpool2d_bwd_f32`. Argument count and pointer-vs-scalar width are
        // verified against the PTX, and write-vs-&mut against the .cu source, by
        // tools/check_launches.py on every CI run.
        // The kernel writes `grad_input`; each arrives as &mut. Event
        // tracking is disabled on this context (see `new`), so ordering comes from
        // the backend's single stream: every launch is serialised behind the last.
        // Grid: launch_config(len) covers exactly `len` elements and the kernel
        // guards the thread index against it (confirmed in the .cu source), so no
        // thread touches past the slices' length. check_launches.py enforces this.
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

/// A page-locked (pinned) host memory buffer for fast CPU-to-GPU transfers.
///
/// Pinned memory is allocated via `cuMemHostAlloc` and is not subject to OS
/// paging, so the GPU can DMA directly from it. This wraps cudarc's
/// [`PinnedHostSlice`], which owns the allocation and carries the CUDA event
/// that every host-side access waits on, so a read can never observe a copy
/// still in flight. That event is what makes an asynchronous pinned transfer
/// sound; a hand-rolled raw pointer has no way to express it.
///
/// There is no `unsafe impl Send`/`Sync` here and none is needed: the wrapped
/// type carries both, on the strength of owning its allocation and gating
/// mutation behind `&mut self`.
///
/// # Usage
/// ```ignore
/// use axonml_core::backends::cuda::PinnedBuffer;
///
/// let data = vec![1.0f32; 1024];
/// let pinned = PinnedBuffer::from_slice(&data).expect("pin failed");
/// let on_gpu = pinned.to_gpu().expect("copy failed");
/// ```
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct PinnedBuffer {
    inner: Option<PinnedHostSlice<f32>>,
}

// PinnedBuffer is Send + Sync because PinnedHostSlice is, not because anyone
// asserted it. This fails to compile the day a raw pointer comes back.
#[cfg(feature = "cuda")]
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PinnedBuffer>();
};

#[cfg(feature = "cuda")]
impl PinnedBuffer {
    /// Allocates a pinned host buffer and copies `data` into it.
    ///
    /// # Errors
    /// If no CUDA device is available or the pinned allocation fails.
    pub fn from_slice(data: &[f32]) -> Result<Self, CudaError> {
        let mut buf = Self::alloc(data.len())?;
        if let Some(inner) = buf.inner.as_mut() {
            inner
                .as_mut_slice()
                .map_err(CudaError::from)?
                .copy_from_slice(data);
        }
        Ok(buf)
    }

    /// Allocates an uninitialised pinned host buffer of `len` elements.
    ///
    /// The contents are zero-filled before this returns, so no caller can
    /// read uninitialised memory through `as_slice`.
    ///
    /// # Errors
    /// If no CUDA device is available or the pinned allocation fails.
    pub fn alloc(len: usize) -> Result<Self, CudaError> {
        if len == 0 {
            return Ok(Self { inner: None });
        }
        let backend = get_cuda_backend().ok_or(CudaError::DeviceNotFound)?;
        // SAFETY: cudarc marks alloc_pinned unsafe because the memory is unset
        // on return. It is filled with zeros on the next line, before the
        // buffer can escape, so nothing ever reads it uninitialised.
        let mut inner = unsafe { backend.context().alloc_pinned::<f32>(len) }
            .map_err(|_| CudaError::AllocationFailed)?;
        inner.as_mut_slice().map_err(CudaError::from)?.fill(0.0);
        Ok(Self { inner: Some(inner) })
    }

    /// The buffer contents. Waits for any in-flight transfer first.
    ///
    /// # Panics
    /// If the CUDA event wait fails, which means the driver is in an
    /// unrecoverable state.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        match &self.inner {
            Some(inner) => inner.as_slice().expect("pinned buffer event wait"),
            None => &[],
        }
    }

    /// Mutable buffer contents. Waits for any in-flight transfer first.
    ///
    /// # Panics
    /// If the CUDA event wait fails, which means the driver is in an
    /// unrecoverable state.
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        match self.inner.as_mut() {
            Some(inner) => inner.as_mut_slice().expect("pinned buffer event wait"),
            None => &mut [],
        }
    }

    /// Number of f32 elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, PinnedHostSlice::len)
    }

    /// Whether the buffer holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies the buffer to the GPU.
    ///
    /// The copy is asynchronous on the backend stream. cudarc records an event
    /// on this buffer, so a later `as_slice` waits for the transfer rather
    /// than reading memory the DMA engine may still be using -- which is the
    /// behaviour a hand-rolled raw pointer cannot provide, and the reason the
    /// previous implementation had to synchronize the whole stream instead.
    ///
    /// # Errors
    /// If no CUDA device is available or the copy fails.
    pub fn to_gpu(&self) -> Result<CudaSlice<f32>, CudaError> {
        let backend = get_cuda_backend().ok_or(CudaError::DeviceNotFound)?;
        match &self.inner {
            Some(inner) => backend.stream().clone_htod(inner).map_err(CudaError::from),
            None => backend.htod_copy(&[]),
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

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu_data = backend.htod_copy(&data).unwrap();

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

        let a: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b: Vec<f32> = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let c: Vec<f32> = vec![0.0; 4];

        let a_gpu = backend.htod_copy(&a).unwrap();
        let b_gpu = backend.htod_copy(&b).unwrap();
        let mut c_gpu = backend.htod_copy(&c).unwrap();

        backend
            .gemm_f32(
                false, false, 2, 2, 3, 1.0, &a_gpu, 2, &b_gpu, 3, 0.0, &mut c_gpu, 2,
            )
            .unwrap();

        let result = backend.dtoh_copy(&c_gpu).unwrap();
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
        assert!((result[0] - 0.5).abs() < 1e-4);
        assert!((result[1] - 0.7311).abs() < 1e-3);
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
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.7616).abs() < 1e-3);
        assert!((result[2] - (-0.7616)).abs() < 1e-3);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_large_tensor_add() {
        if !is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        let n = 1_000_000;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let a_gpu = backend.htod_copy(&a).unwrap();
        let b_gpu = backend.htod_copy(&b).unwrap();
        let mut c_gpu = backend.alloc::<f32>(n).unwrap();

        backend.add_f32(&mut c_gpu, &a_gpu, &b_gpu, n).unwrap();

        let result = backend.dtoh_copy(&c_gpu).unwrap();

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

        let input = vec![1.0f32; 1 * 3 * 4 * 4];
        let mut weight = vec![0.0f32; 2 * 3 * 1 * 1];
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
        assert!(
            (out[0] - 1.5).abs() < 0.01,
            "1x1 conv ch0: expected 1.5, got {}",
            out[0]
        );
        assert!(
            (out[16] - 1.5).abs() < 0.01,
            "1x1 conv ch1: expected 1.5, got {}",
            out[16]
        );

        let input2 = vec![1.0f32; 1 * 3 * 8 * 8];
        let weight2 = vec![1.0f32; 2 * 3 * 3 * 3];
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
        let center = 4 * 8 + 4;
        assert!(
            (out2[center] - 27.0).abs() < 0.1,
            "3x3 conv center: expected 27.0, got {}",
            out2[center]
        );
        assert!(
            (out2[0] - 12.0).abs() < 0.1,
            "3x3 conv corner: expected 12.0, got {}",
            out2[0]
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
mod pinned_buffer_tests {
    use super::PinnedBuffer;

    /// The behaviour that justifies pinned memory at all: a transfer is
    /// asynchronous, and a host read after it must see the data, not a copy
    /// still in flight. cudarc's event on the slice is what makes that true.
    #[test]
    fn round_trip_through_the_gpu_is_exact() {
        if !super::is_available() {
            return;
        }
        let data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5).collect();
        let pinned = PinnedBuffer::from_slice(&data).expect("pin");
        assert_eq!(pinned.len(), data.len());
        assert_eq!(pinned.as_slice(), &data[..]);

        let on_gpu = pinned.to_gpu().expect("htod");
        let backend = super::get_cuda_backend().expect("backend");
        let back = backend.dtoh_copy(&on_gpu).expect("dtoh");
        assert_eq!(back, data);
    }

    #[test]
    fn alloc_is_zeroed_and_writable() {
        if !super::is_available() {
            return;
        }
        let mut buf = PinnedBuffer::alloc(257).expect("alloc");
        assert!(buf.as_slice().iter().all(|&x| x == 0.0));
        buf.as_slice_mut()[256] = 7.0;
        assert_eq!(buf.as_slice()[256], 7.0);
    }

    #[test]
    fn an_empty_buffer_is_harmless() {
        if !super::is_available() {
            return;
        }
        let buf = PinnedBuffer::alloc(0).expect("alloc");
        assert!(buf.is_empty());
        assert!(buf.as_slice().is_empty());
        let empty = PinnedBuffer::from_slice(&[]).expect("pin");
        assert_eq!(empty.to_gpu().expect("htod").len(), 0);
    }
}
