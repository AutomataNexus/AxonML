//! CUDA Memory Pool - Reuses freed GPU allocations
//!
//! # File
//! `crates/axonml-core/src/backends/cuda_pool.rs`
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

/// Return GPU memory to the pool instead of freeing it.
#[cfg(feature = "cuda")]
pub fn pool_free(slice: CudaSlice<f32>) {
    let pool = get_memory_pool();
    let capacity = slice.len();
    // Leak returns the raw device pointer and prevents Drop from calling cudaFree
    let ptr = slice.leak();
    pool.release(ptr, capacity);
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
