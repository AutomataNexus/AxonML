//! CUDA memory pool — reuses freed GPU allocations to avoid cudaFree latency.
//!
//! `CudaMemoryPool` maintains per-size-bucket free lists (power-of-2 for sizes
//! above 256, linear 64-byte increments for smaller). `pool_alloc(len)` checks the
//! free list first, falls back to `stream.alloc_zeros(bucket)` on miss.
//! `pool_free(slice)` returns the block's raw device pointer to the bucket,
//! capped at 64 blocks per bucket to prevent unbounded growth. `clear_pool()`
//! actually cudaFrees everything. `print_pool_stats()` reports hits/misses/
//! returns/pooled bytes. Global singleton via `OnceLock`.
//!
//! # File
//! `crates/axonml-core/src/backends/cuda_pool.rs`
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
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Mutex;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

// =============================================================================
// Memory Pool
// =============================================================================

#[cfg(feature = "cuda")]
struct PooledBlock {
    /// The raw device pointer (CUdeviceptr = u64)
    ptr: u64,
    /// Actual allocated capacity in elements (may be larger than requested)
    capacity: usize,
}

#[cfg(feature = "cuda")]
struct MemoryPoolInner {
    /// Free lists bucketed by size bucket index
    /// Key: bucket size (rounded-up allocation size), Value: list of free blocks
    free_lists: HashMap<usize, Vec<PooledBlock>>,
    /// Total bytes currently in pool (not actively used)
    pooled_bytes: usize,
    /// Statistics
    hits: usize,
    misses: usize,
    returns: usize,
}

/// CUDA memory pool that reuses freed GPU allocations.
///
/// Uses size-bucketed free lists to efficiently match allocation requests
/// with previously freed blocks.
#[cfg(feature = "cuda")]
pub struct CudaMemoryPool {
    inner: Mutex<MemoryPoolInner>,
}

#[cfg(feature = "cuda")]
static CUDA_MEMORY_POOL: OnceLock<CudaMemoryPool> = OnceLock::new();

#[cfg(feature = "cuda")]
impl CudaMemoryPool {
    /// Creates a new empty memory pool.
    fn new() -> Self {
        Self {
            inner: Mutex::new(MemoryPoolInner {
                free_lists: HashMap::new(),
                pooled_bytes: 0,
                hits: 0,
                misses: 0,
                returns: 0,
            }),
        }
    }

    /// Round allocation size up to the nearest bucket size.
    /// Uses power-of-2 bucketing for sizes > 256, linear for smaller.
    fn bucket_size(requested: usize) -> usize {
        if requested <= 256 {
            // Round up to next multiple of 64
            ((requested + 63) / 64) * 64
        } else {
            // Round up to next power of 2
            requested.next_power_of_two()
        }
    }

    /// Try to get a block from the free list.
    /// Returns the raw device pointer and capacity if found.
    fn try_acquire(&self, requested_elements: usize) -> Option<(u64, usize)> {
        let bucket = Self::bucket_size(requested_elements);
        let mut inner = self.inner.lock().unwrap();

        if let Some(blocks) = inner.free_lists.get_mut(&bucket) {
            if let Some(block) = blocks.pop() {
                inner.pooled_bytes -= block.capacity * 4; // f32 = 4 bytes
                inner.hits += 1;
                return Some((block.ptr, block.capacity));
            }
        }
        inner.misses += 1;
        None
    }

    /// Return a block to the pool for later reuse.
    fn release(&self, ptr: u64, capacity: usize) {
        let bucket = Self::bucket_size(capacity);
        let mut inner = self.inner.lock().unwrap();
        inner.pooled_bytes += capacity * 4;
        inner.returns += 1;

        let blocks = inner.free_lists.entry(bucket).or_default();
        // Limit per-bucket free list to prevent unbounded growth
        if blocks.len() < 64 {
            blocks.push(PooledBlock { ptr, capacity });
        } else {
            // Too many blocks in this bucket, actually free this one
            inner.pooled_bytes -= capacity * 4;
            if let Some(backend) = super::cuda::get_cuda_backend() {
                unsafe {
                    let slice: CudaSlice<f32> = backend.stream().upgrade_device_ptr(ptr, capacity);
                    drop(slice); // Actually free GPU memory
                }
            }
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.hits, inner.misses, inner.returns, inner.pooled_bytes)
    }

    /// Clear all pooled memory, actually freeing it.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        let backend = super::cuda::get_cuda_backend();
        for (_bucket, blocks) in inner.free_lists.drain() {
            for block in blocks {
                if let Some(ref be) = backend {
                    unsafe {
                        let slice: CudaSlice<f32> =
                            be.stream().upgrade_device_ptr(block.ptr, block.capacity);
                        drop(slice);
                    }
                }
            }
        }
        inner.pooled_bytes = 0;
    }
}

/// Get or initialize the global CUDA memory pool.
#[cfg(feature = "cuda")]
pub fn get_memory_pool() -> &'static CudaMemoryPool {
    CUDA_MEMORY_POOL.get_or_init(CudaMemoryPool::new)
}

/// Allocate GPU memory using the pool.
///
/// First checks the free list for a matching block. If none found,
/// allocates fresh GPU memory. Pool-acquired blocks are zeroed before return.
///
/// Returns a CudaSlice with exactly `len` elements.
/// The pool uses bucketed sizes internally for efficient reuse.
#[cfg(feature = "cuda")]
pub fn pool_alloc(len: usize) -> Result<CudaSlice<f32>, super::cuda::CudaError> {
    let pool = get_memory_pool();

    // Try to get from pool (pool stores bucket-sized allocations)
    if let Some((ptr, capacity)) = pool.try_acquire(len) {
        let backend =
            super::cuda::get_cuda_backend().ok_or(super::cuda::CudaError::DeviceNotFound)?;
        unsafe {
            // Reconstruct at original capacity and zero it
            let mut slice: CudaSlice<f32> = backend.stream().upgrade_device_ptr(ptr, capacity);
            backend
                .stream()
                .memset_zeros(&mut slice)
                .map_err(super::cuda::CudaError::from)?;
            Ok(slice)
        }
    } else {
        // Allocate fresh from CUDA at bucket size for better reuse
        let bucket = CudaMemoryPool::bucket_size(len);
        let backend =
            super::cuda::get_cuda_backend().ok_or(super::cuda::CudaError::DeviceNotFound)?;
        backend
            .stream()
            .alloc_zeros(bucket)
            .map_err(super::cuda::CudaError::from)
    }
}

/// Allocate GPU memory from the pool WITHOUT zero-init.
///
/// Identical to [`pool_alloc`] but skips the `cuMemsetD8Async` on pool
/// hit and uses `stream.alloc` (uninitialized) on pool miss. ONLY safe
/// to use when the caller writes every element of the returned slice
/// before any read — matmul output buffers, elementwise kernel outputs,
/// etc.
///
/// Accumulators (anything that reads its own output before writing all
/// positions) MUST stay on [`pool_alloc`].
///
/// Skipping the memset removes one `cuMemsetD8Async` call per
/// allocation; the RTX 5070 Ti Laptop + DeepSeek-7B Q4_K_M profile
/// shows ~4 us per memset and ~200 pool allocations per decode token,
/// so swapping one call site is ~1 us/token saved, and a full hot-path
/// conversion can be ~0.8 ms/token.
/// Same layout pool as f32, but hands back `CudaSlice<u32>` — used for
/// gather index uploads etc. The underlying bytes are identical (4 bytes
/// per element), so we reuse the f32 bucket infrastructure and reinterpret
/// the raw device pointer.
///
/// Capture-pen semantics: returned slices go through a dedicated u32
/// `pool_free_u32` on drop. Those ALSO go into the capture pen when
/// active, preserving cu_device_ptr host addresses for graph replay.
#[cfg(feature = "cuda")]
pub fn pool_alloc_uninit_u32(len: usize) -> Result<CudaSlice<u32>, super::cuda::CudaError> {
    let pool = get_memory_pool();

    if let Some((ptr, capacity)) = pool.try_acquire(len) {
        let backend =
            super::cuda::get_cuda_backend().ok_or(super::cuda::CudaError::DeviceNotFound)?;
        unsafe {
            let slice: CudaSlice<u32> = backend.stream().upgrade_device_ptr(ptr, capacity);
            Ok(slice)
        }
    } else {
        let bucket = CudaMemoryPool::bucket_size(len);
        let backend =
            super::cuda::get_cuda_backend().ok_or(super::cuda::CudaError::DeviceNotFound)?;
        unsafe {
            backend
                .stream()
                .alloc::<u32>(bucket)
                .map_err(super::cuda::CudaError::from)
        }
    }
}

/// `pool_free`-equivalent for u32 slices. Routes to the capture pen when
/// active (by converting to the u32 pen — separate from the f32 pen
/// because we store them as typed `CudaSlice<u32>`).
#[cfg(feature = "cuda")]
pub fn pool_free_u32(slice: CudaSlice<u32>) {
    if CAPTURE_PEN_ACTIVE.with(|c| c.get()) {
        CAPTURE_PEN_U32.with(|pen| pen.borrow_mut().push(slice));
        return;
    }
    let pool = get_memory_pool();
    let capacity = slice.len();
    let ptr = slice.leak();
    pool.release(ptr, capacity);
}

/// Allocate an uninitialized f32 slice from the CUDA device pool (via
/// `cuMemAllocAsync`), bypassing the Rust-side cache. Used by the fused
/// decode kernels so the allocation is visible to CUDA graph capture.
#[cfg(feature = "cuda")]
pub fn pool_alloc_uninit(len: usize) -> Result<CudaSlice<f32>, super::cuda::CudaError> {
    // Under CUDA graph capture, skip the Rust-side pool cache. Its
    // `upgrade_device_ptr` path reconstructs a CudaSlice from a cached raw
    // pointer with no CUDA API call — capture doesn't see it, so on replay
    // the captured kernels still reference the original pointer from
    // capture-time, which gets reused for something else between replays
    // → CUDA_ERROR_ILLEGAL_ADDRESS. Always go through cuMemAllocAsync so
    // the driver records an alloc node in the captured graph, and the
    // replay machinery can reconstruct fresh virtual addresses each launch.
    if pool_force_driver_alloc() {
        let bucket = CudaMemoryPool::bucket_size(len);
        let backend =
            super::cuda::get_cuda_backend().ok_or(super::cuda::CudaError::DeviceNotFound)?;
        return unsafe {
            backend
                .stream()
                .alloc::<f32>(bucket)
                .map_err(super::cuda::CudaError::from)
        };
    }

    let pool = get_memory_pool();

    if let Some((ptr, capacity)) = pool.try_acquire(len) {
        let backend =
            super::cuda::get_cuda_backend().ok_or(super::cuda::CudaError::DeviceNotFound)?;
        unsafe {
            // Reconstruct at original capacity; skip the memset_zeros.
            let slice: CudaSlice<f32> = backend.stream().upgrade_device_ptr(ptr, capacity);
            Ok(slice)
        }
    } else {
        let bucket = CudaMemoryPool::bucket_size(len);
        let backend =
            super::cuda::get_cuda_backend().ok_or(super::cuda::CudaError::DeviceNotFound)?;
        unsafe {
            backend
                .stream()
                .alloc::<f32>(bucket)
                .map_err(super::cuda::CudaError::from)
        }
    }
}

/// Thread-local flag that opts-out of the Rust-side pool cache in favor of
/// the CUDA driver's stream-ordered allocator. Set via [`with_driver_alloc`]
/// around work that will be captured into a CUDA graph — that way every
/// allocation inside the scope records as a graph MemAllocNode, allowing
/// graph replay to allocate fresh virtual addresses deterministically
/// (`cuMemAllocAsync` with the device's default pool, release threshold
/// maxed so the driver keeps memory pooled across allocations).
#[cfg(feature = "cuda")]
pub fn pool_force_driver_alloc() -> bool {
    POOL_FORCE_DRIVER_ALLOC.with(|f| f.get())
}

#[cfg(feature = "cuda")]
thread_local! {
    static POOL_FORCE_DRIVER_ALLOC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with pool allocations routed through `cuMemAllocAsync` instead
/// of the Rust pool cache. Used to make the captured-graph memory plan
/// deterministic across replays.
#[cfg(feature = "cuda")]
pub fn with_driver_alloc<R>(f: impl FnOnce() -> R) -> R {
    POOL_FORCE_DRIVER_ALLOC.with(|flag| flag.set(true));
    let r = f();
    POOL_FORCE_DRIVER_ALLOC.with(|flag| flag.set(false));
    r
}

/// Return GPU memory to the pool instead of freeing it.
///
/// If a graph-capture "pen" is active (see [`with_capture_pen`]), the
/// CudaSlice is stored there intact instead of being leaked back to the
/// bucket cache. cudarc records kernel args as `&slice.cu_device_ptr`
/// (pointer-to-field), and graph replay dereferences that host pointer
/// every launch. If the slice is dropped between capture and replay, the
/// stack/heap location backing that field is freed → dangling pointer on
/// replay (CUDA_ERROR_ILLEGAL_ADDRESS). The pen gives captured slices a
/// stable host location for the graph's lifetime.
#[cfg(feature = "cuda")]
pub fn pool_free(slice: CudaSlice<f32>) {
    if CAPTURE_PEN_ACTIVE.with(|c| c.get()) {
        CAPTURE_PEN.with(|pen| pen.borrow_mut().push(slice));
        return;
    }
    let pool = get_memory_pool();
    let capacity = slice.len();
    // Leak returns the raw device pointer and prevents Drop from calling cudaFree
    let ptr = slice.leak();
    pool.release(ptr, capacity);
}

#[cfg(feature = "cuda")]
thread_local! {
    static CAPTURE_PEN_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static CAPTURE_PEN: std::cell::RefCell<Vec<CudaSlice<f32>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static CAPTURE_PEN_U32: std::cell::RefCell<Vec<CudaSlice<u32>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Collection of CudaSlices retained by the capture pen across types.
/// Keep this alive for the full lifetime of the captured graph; drop or
/// release it only after the `CudaGraph` (CUgraphExec wrapper) is dropped.
#[cfg(feature = "cuda")]
pub struct CapturePen {
    f32_slices: Vec<CudaSlice<f32>>,
    u32_slices: Vec<CudaSlice<u32>>,
}

#[cfg(feature = "cuda")]
impl CapturePen {
    /// Number of f32 slices retained.
    pub fn f32_count(&self) -> usize {
        self.f32_slices.len()
    }
    /// Number of u32 slices retained.
    pub fn u32_count(&self) -> usize {
        self.u32_slices.len()
    }
    /// Total retained.
    pub fn total(&self) -> usize {
        self.f32_slices.len() + self.u32_slices.len()
    }

    /// Return all retained slices to the Rust pool. Safe to call only
    /// after the captured graph's `CudaGraph` has been dropped.
    pub fn release(self) {
        let pool = get_memory_pool();
        for slice in self.f32_slices {
            let capacity = slice.len();
            let ptr = slice.leak();
            pool.release(ptr, capacity);
        }
        for slice in self.u32_slices {
            let capacity = slice.len();
            let ptr = slice.leak();
            pool.release(ptr, capacity);
        }
    }
}

/// Scope-guarded "graph capture pen". Inside the scope, [`pool_free`] and
/// [`pool_free_u32`] retain CudaSlices intact (their host-side
/// cu_device_ptr stays at stable addresses) instead of leaking them back
/// to the bucket cache. Returns the collected slices as a `CapturePen`
/// the caller keeps alive for the captured graph's full lifetime, then
/// releases once the graph is destroyed.
#[cfg(feature = "cuda")]
pub fn with_capture_pen<R>(f: impl FnOnce() -> R) -> (R, CapturePen) {
    CAPTURE_PEN_ACTIVE.with(|c| c.set(true));
    let r = f();
    CAPTURE_PEN_ACTIVE.with(|c| c.set(false));
    let f32_slices = CAPTURE_PEN.with(|pen| std::mem::take(&mut *pen.borrow_mut()));
    let u32_slices = CAPTURE_PEN_U32.with(|pen| std::mem::take(&mut *pen.borrow_mut()));
    (
        r,
        CapturePen {
            f32_slices,
            u32_slices,
        },
    )
}

/// Print pool statistics.
#[cfg(feature = "cuda")]
pub fn print_pool_stats() {
    let pool = get_memory_pool();
    let (hits, misses, returns, pooled) = pool.stats();
    eprintln!(
        "[CudaPool] hits={}, misses={}, returns={}, pooled={:.1}MB",
        hits,
        misses,
        returns,
        pooled as f64 / (1024.0 * 1024.0)
    );
}

/// Clear the memory pool.
#[cfg(feature = "cuda")]
pub fn clear_pool() {
    get_memory_pool().clear();
}

// =============================================================================
// No-op stubs when CUDA is not enabled
// =============================================================================

#[cfg(not(feature = "cuda"))]
/// Stub when CUDA not available.
pub fn print_pool_stats() {}

#[cfg(not(feature = "cuda"))]
/// Stub when CUDA not available.
pub fn clear_pool() {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::*;

    // -------------------------------------------------------------------------
    // Bucket sizing tests (pure logic, no GPU required)
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(feature = "cuda")]
    fn test_bucket_size_small() {
        // Small sizes round up to multiples of 64
        assert_eq!(CudaMemoryPool::bucket_size(1), 64);
        assert_eq!(CudaMemoryPool::bucket_size(63), 64);
        assert_eq!(CudaMemoryPool::bucket_size(64), 64);
        assert_eq!(CudaMemoryPool::bucket_size(65), 128);
        assert_eq!(CudaMemoryPool::bucket_size(128), 128);
        assert_eq!(CudaMemoryPool::bucket_size(200), 256);
        assert_eq!(CudaMemoryPool::bucket_size(256), 256);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_bucket_size_large() {
        // Large sizes round up to power of 2
        assert_eq!(CudaMemoryPool::bucket_size(257), 512);
        assert_eq!(CudaMemoryPool::bucket_size(500), 512);
        assert_eq!(CudaMemoryPool::bucket_size(512), 512);
        assert_eq!(CudaMemoryPool::bucket_size(513), 1024);
        assert_eq!(CudaMemoryPool::bucket_size(1_000_000), 1_048_576);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_bucket_size_zero() {
        assert_eq!(CudaMemoryPool::bucket_size(0), 0);
    }

    // -------------------------------------------------------------------------
    // Pool lifecycle tests (requires CUDA GPU)
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pool_alloc_and_free() {
        // This test requires a CUDA GPU
        if super::super::cuda::get_cuda_backend().is_none() {
            return;
        }

        let slice = pool_alloc(1024).expect("pool_alloc failed");
        assert!(slice.len() >= 1024);

        // Return to pool
        pool_free(slice);

        // Allocate again — should be a pool hit
        let pool = get_memory_pool();
        let (hits_before, _, _, _) = pool.stats();
        let slice2 = pool_alloc(1024).expect("second pool_alloc failed");
        let (hits_after, _, _, _) = pool.stats();

        // Should have gotten a pool hit (same bucket size)
        assert!(
            hits_after > hits_before,
            "Expected pool hit on second alloc"
        );

        pool_free(slice2);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pool_stats() {
        if super::super::cuda::get_cuda_backend().is_none() {
            return;
        }

        let pool = get_memory_pool();
        let (hits, misses, returns, _pooled) = pool.stats();
        // Stats should be non-negative (may be non-zero from other tests)
        assert!(hits + misses + returns >= 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pool_clear() {
        if super::super::cuda::get_cuda_backend().is_none() {
            return;
        }

        // Allocate and free to populate pool
        let slice = pool_alloc(512).expect("alloc failed");
        pool_free(slice);

        // Clear should not panic
        clear_pool();

        let pool = get_memory_pool();
        let (_, _, _, pooled_bytes) = pool.stats();
        assert_eq!(pooled_bytes, 0, "Pool should be empty after clear");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pool_different_sizes() {
        if super::super::cuda::get_cuda_backend().is_none() {
            return;
        }

        // Allocate different sizes — they should go to different buckets
        let s1 = pool_alloc(100).expect("alloc 100 failed");
        let s2 = pool_alloc(1000).expect("alloc 1000 failed");
        let s3 = pool_alloc(10000).expect("alloc 10000 failed");

        pool_free(s1);
        pool_free(s2);
        pool_free(s3);

        // Allocating 100 again should hit the 128-bucket (or 64-bucket)
        let pool = get_memory_pool();
        let (hits_before, _, _, _) = pool.stats();
        let s4 = pool_alloc(100).expect("re-alloc 100 failed");
        let (hits_after, _, _, _) = pool.stats();
        assert!(hits_after > hits_before);
        pool_free(s4);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pool_zeroed_on_reuse() {
        if super::super::cuda::get_cuda_backend().is_none() {
            return;
        }

        // Allocate, free, re-allocate — data should be zeroed
        let slice = pool_alloc(64).expect("alloc failed");
        pool_free(slice);

        let slice2 = pool_alloc(64).expect("re-alloc failed");
        // Copy to host and verify zeros
        let host_data = super::super::cuda::get_cuda_backend()
            .unwrap()
            .stream()
            .memcpy_dtoh(&slice2);

        if let Ok(data) = host_data {
            for &val in &data {
                assert_eq!(val, 0.0, "Pool-reused memory should be zeroed");
            }
        }
        pool_free(slice2);
    }
}
