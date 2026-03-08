//! Storage - Raw Memory Management for Tensors
//!
//! Provides efficient memory storage that underlies all tensor operations.
//! Storage is reference-counted for efficient sharing between tensor views.
//!
//! # Key Features
//! - Reference-counted memory for efficient views
//! - Device-agnostic storage interface (CPU and GPU)
//! - Zero-copy slicing through offset/length
//! - Automatic memory cleanup
//!
//! @version 0.2.0
//! @author `AutomataNexus` Development Team

use core::ops::{Deref, DerefMut};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::device::Device;
use crate::dtype::Scalar;
use crate::error::{Error, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::DeviceSlice;

// =============================================================================
// Storage Data Enum
// =============================================================================

/// Wrapper around CudaSlice that returns memory to the pool on drop
/// instead of calling cudaFree.
#[cfg(feature = "cuda")]
pub struct PooledCudaSlice {
    slice: Option<CudaSlice<f32>>,
    pool_managed: bool,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for PooledCudaSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledCudaSlice")
            .field("pool_managed", &self.pool_managed)
            .field("len", &self.slice.as_ref().map(|s| s.len()))
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl Drop for PooledCudaSlice {
    fn drop(&mut self) {
        if let Some(slice) = self.slice.take() {
            if self.pool_managed {
                crate::backends::cuda_pool::pool_free(slice);
            }
            // else: normal CudaSlice drop calls cudaFree
        }
    }
}

#[cfg(feature = "cuda")]
impl PooledCudaSlice {
    /// Create a new pool-managed CUDA slice.
    pub fn new(slice: CudaSlice<f32>, pool_managed: bool) -> Self {
        Self { slice: Some(slice), pool_managed }
    }

    /// Get a reference to the underlying CudaSlice.
    pub fn slice(&self) -> &CudaSlice<f32> {
        self.slice.as_ref().expect("CudaSlice already taken")
    }

    /// Get a mutable reference to the underlying CudaSlice.
    pub fn slice_mut(&mut self) -> &mut CudaSlice<f32> {
        self.slice.as_mut().expect("CudaSlice already taken")
    }
}

/// Holds either CPU or GPU data.
#[derive(Debug)]
enum StorageData<T: Scalar> {
    /// CPU data stored as a Vec.
    Cpu(Vec<T>),
    /// GPU data stored as a PooledCudaSlice (f32 only on GPU).
    /// Returns to memory pool on drop instead of calling cudaFree.
    #[cfg(feature = "cuda")]
    Cuda(PooledCudaSlice),
}

// =============================================================================
// Storage Struct
// =============================================================================

/// Raw memory storage for tensor data.
///
/// Storage is the fundamental building block for tensors. It manages a contiguous
/// block of memory on a specific device and is reference-counted to allow
/// efficient sharing between tensor views.
#[derive(Debug)]
pub struct Storage<T: Scalar> {
    /// The underlying data buffer.
    inner: Arc<RwLock<StorageInner<T>>>,
    /// Offset into the storage (for views).
    offset: usize,
    /// Number of elements in this view.
    len: usize,
}

/// Inner storage data that can be shared between views.
#[derive(Debug)]
struct StorageInner<T: Scalar> {
    /// Raw data (CPU or GPU).
    data: StorageData<T>,
    /// The device this storage resides on.
    device: Device,
}

impl<T: Scalar> Storage<T> {
    /// Creates new storage with the given capacity, initialized to zero.
    #[must_use]
    pub fn zeros(len: usize, device: Device) -> Self {
        let data = vec![T::zeroed(); len];
        Self::from_vec(data, device)
    }

    /// Creates storage from an existing vector (always on CPU).
    #[must_use]
    pub fn from_vec(data: Vec<T>, device: Device) -> Self {
        let len = data.len();
        Self {
            inner: Arc::new(RwLock::new(StorageInner {
                data: StorageData::Cpu(data),
                device,
            })),
            offset: 0,
            len,
        }
    }

    /// Creates storage from a slice by copying the data.
    #[must_use]
    pub fn from_slice(data: &[T], device: Device) -> Self {
        Self::from_vec(data.to_vec(), device)
    }

    /// Returns the number of elements in this storage view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the storage is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the offset into the underlying buffer.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the device this storage is on.
    #[must_use]
    pub fn device(&self) -> Device {
        self.inner.read().device
    }

    /// Returns true if data is on CPU.
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        matches!(self.inner.read().data, StorageData::Cpu(_))
    }

    /// Returns true if data is on GPU.
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Returns the size in bytes of this storage.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * core::mem::size_of::<T>()
    }

    /// Creates a view into a portion of this storage.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        if offset + len > self.len {
            return Err(Error::IndexOutOfBounds {
                index: offset + len,
                size: self.len,
            });
        }

        Ok(Self {
            inner: Arc::clone(&self.inner),
            offset: self.offset + offset,
            len,
        })
    }

    /// Returns true if this storage is uniquely owned (not shared).
    #[must_use]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Returns an immutable reference to the CPU data.
    ///
    /// # Panics
    /// Panics if the storage is on GPU. Use `to_vec()` for device-safe access.
    #[must_use]
    pub fn as_slice(&self) -> StorageReadGuard<'_, T> {
        StorageReadGuard {
            guard: self.inner.read(),
            offset: self.offset,
            len: self.len,
        }
    }

    /// Returns a mutable reference to the CPU data.
    ///
    /// # Panics
    /// Panics if the storage is on GPU.
    #[must_use]
    pub fn as_slice_mut(&self) -> StorageWriteGuard<'_, T> {
        StorageWriteGuard {
            guard: self.inner.write(),
            offset: self.offset,
            len: self.len,
        }
    }

    /// Copies data from another storage into this one.
    pub fn copy_from(&self, other: &Self) -> Result<()> {
        if self.len != other.len {
            return Err(Error::shape_mismatch(&[self.len], &[other.len]));
        }

        let src = other.as_slice();
        let mut dst = self.as_slice_mut();
        dst.copy_from_slice(&src);
        Ok(())
    }

    /// Makes a deep copy of this storage.
    ///
    /// For GPU storage, this only works for `Storage<f32>`.
    /// Other types will panic on GPU storage.
    #[must_use]
    pub fn deep_copy(&self) -> Self {
        let inner = self.inner.read();
        match &inner.data {
            StorageData::Cpu(cpu_data) => {
                let data = cpu_data[self.offset..self.offset + self.len].to_vec();
                Self::from_vec(data, inner.device)
            }
            #[cfg(feature = "cuda")]
            StorageData::Cuda(_) => {
                panic!("deep_copy() on GPU storage requires Storage<f32>. Use deep_copy_f32().");
            }
        }
    }

    /// Returns the data as a Vec from CPU storage.
    ///
    /// For GPU-resident f32 tensors, use the `Tensor::to_vec()` method which
    /// handles device-aware copies via `Storage<f32>::to_vec_f32()`.
    ///
    /// # Panics
    /// Panics if storage is on GPU (use `to_vec_f32()` on `Storage<f32>` instead).
    pub fn to_vec(&self) -> Vec<T> {
        let inner = self.inner.read();
        match &inner.data {
            StorageData::Cpu(cpu_data) => {
                cpu_data[self.offset..self.offset + self.len].to_vec()
            }
            #[cfg(feature = "cuda")]
            StorageData::Cuda(_) => {
                panic!("Cannot call to_vec() on GPU storage for generic T. Use to_vec_f32() on Storage<f32>.");
            }
        }
    }

    /// Transfers this storage to a different device.
    ///
    /// For GPU transfers, only `Storage<f32>` is supported. Use the
    /// `Storage<f32>::to_device()` specialization for CPU↔GPU transfers.
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }

        // Generic path: only CPU→CPU is supported
        if device.is_cpu() && self.device().is_cpu() {
            return Ok(self.deep_copy());
        }

        Err(Error::DeviceNotAvailable { device })
    }
}

// =============================================================================
// f32-specific Storage for GPU transfers
// =============================================================================

#[cfg(feature = "cuda")]
impl Storage<f32> {
    /// Transfers f32 storage between CPU and GPU.
    pub fn to_device_f32(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }

        let inner = self.inner.read();

        match (&inner.data, device) {
            // CPU → CPU: just deep copy
            (StorageData::Cpu(_), Device::Cpu) => {
                drop(inner);
                Ok(self.deep_copy())
            }
            // CPU → GPU: htod_copy
            (StorageData::Cpu(cpu_data), Device::Cuda(_idx)) => {
                let backend = crate::backends::cuda::get_cuda_backend()
                    .ok_or(Error::DeviceNotAvailable { device })?;
                let slice = &cpu_data[self.offset..self.offset + self.len];
                let cuda_slice = backend.htod_copy(slice)
                    .map_err(|_| Error::DeviceNotAvailable { device })?;
                let len = self.len;
                Ok(Self {
                    inner: Arc::new(RwLock::new(StorageInner {
                        data: StorageData::Cuda(PooledCudaSlice::new(cuda_slice, false)),
                        device,
                    })),
                    offset: 0,
                    len,
                })
            }
            // GPU → CPU: dtoh_copy
            (StorageData::Cuda(pooled), Device::Cpu) => {
                let backend = crate::backends::cuda::get_cuda_backend()
                    .ok_or(Error::DeviceNotAvailable { device: self.device() })?;
                let full_vec = backend.dtoh_copy(pooled.slice())
                    .map_err(|_| Error::DeviceNotAvailable { device })?;
                let end = self.offset + self.len;
                let sliced: Vec<f32> = if self.offset == 0 && self.len == full_vec.len() {
                    full_vec
                } else if end <= full_vec.len() {
                    full_vec[self.offset..end].to_vec()
                } else {
                    // CudaSlice is smaller than Storage.len — this indicates a bug
                    // but handle gracefully: copy what we have, zero-pad the rest
                    eprintln!(
                        "[storage] WARNING: CudaSlice len={} < Storage offset+len={} (offset={}, len={})",
                        full_vec.len(), end, self.offset, self.len
                    );
                    let available = if self.offset < full_vec.len() {
                        full_vec.len() - self.offset
                    } else {
                        0
                    };
                    let mut result = vec![0.0f32; self.len];
                    if available > 0 {
                        result[..available].copy_from_slice(&full_vec[self.offset..self.offset + available]);
                    }
                    result
                };
                Ok(Self::from_vec(sliced, Device::Cpu))
            }
            // GPU → GPU: D2H then H2D (simple path)
            (StorageData::Cuda(_), Device::Cuda(_)) => {
                drop(inner);
                let cpu_storage = self.to_device_f32(Device::Cpu)?;
                cpu_storage.to_device_f32(device)
            }
            _ => Err(Error::DeviceNotAvailable { device }),
        }
    }
}

// =============================================================================
// CUDA-specific Storage methods
// =============================================================================

#[cfg(feature = "cuda")]
impl Storage<f32> {
    /// Returns data as a Vec<f32>, performing D2H copy if on GPU.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        let inner = self.inner.read();
        match &inner.data {
            StorageData::Cpu(cpu_data) => {
                cpu_data[self.offset..self.offset + self.len].to_vec()
            }
            StorageData::Cuda(pooled) => {
                if let Some(backend) = crate::backends::cuda::get_cuda_backend() {
                    if let Ok(full_vec) = backend.dtoh_copy(pooled.slice()) {
                        if self.offset == 0 && self.len == full_vec.len() {
                            return full_vec;
                        }
                        return full_vec[self.offset..self.offset + self.len].to_vec();
                    }
                }
                vec![0.0f32; self.len]
            }
        }
    }

    /// Deep copy that works for both CPU and GPU f32 storage.
    pub fn deep_copy_f32(&self) -> Self {
        let device = self.device();
        let vec = self.to_vec_f32();
        if device.is_gpu() {
            if let Some(backend) = crate::backends::cuda::get_cuda_backend() {
                if let Ok(new_slice) = backend.htod_copy(&vec) {
                    return Self::from_cuda_slice_unmanaged(new_slice, self.len, device);
                }
            }
        }
        Self::from_vec(vec, device)
    }

    /// Creates storage from a pool-allocated CudaSlice.
    ///
    /// The slice will be returned to the CUDA memory pool on drop.
    /// Only use this for CudaSlices from `pool_alloc()` — the slice must be
    /// bucket-sized for correct pool reuse.
    pub fn from_cuda_slice(slice: CudaSlice<f32>, len: usize, device: Device) -> Self {
        Self {
            inner: Arc::new(RwLock::new(StorageInner {
                data: StorageData::Cuda(PooledCudaSlice::new(slice, true)),
                device,
            })),
            offset: 0,
            len,
        }
    }

    /// Creates storage from a non-pool CudaSlice (e.g., from htod_copy).
    ///
    /// The slice will be freed via cudaFree on drop (normal CUDA deallocation),
    /// NOT returned to the memory pool.
    pub fn from_cuda_slice_unmanaged(slice: CudaSlice<f32>, len: usize, device: Device) -> Self {
        Self {
            inner: Arc::new(RwLock::new(StorageInner {
                data: StorageData::Cuda(PooledCudaSlice::new(slice, false)),
                device,
            })),
            offset: 0,
            len,
        }
    }


    /// Returns a reference to the CudaSlice if on GPU.
    ///
    /// # Panics
    /// Panics if storage is not on GPU.
    pub fn as_cuda_slice(&self) -> CudaSliceReadGuard<'_> {
        CudaSliceReadGuard {
            guard: self.inner.read(),
        }
    }

    /// Returns a write guard providing mutable access to the underlying `CudaSlice<f32>`.
    ///
    /// # Panics
    /// Panics if storage is not on GPU.
    pub fn as_cuda_slice_mut(&self) -> CudaSliceWriteGuard<'_> {
        CudaSliceWriteGuard {
            guard: self.inner.write(),
        }
    }
}

/// Read guard that provides access to the CudaSlice.
#[cfg(feature = "cuda")]
pub struct CudaSliceReadGuard<'a> {
    guard: parking_lot::RwLockReadGuard<'a, StorageInner<f32>>,
}

#[cfg(feature = "cuda")]
impl<'a> CudaSliceReadGuard<'a> {
    /// Returns a reference to the CudaSlice.
    ///
    /// # Panics
    /// Panics if storage is CPU.
    pub fn slice(&self) -> &CudaSlice<f32> {
        match &self.guard.data {
            StorageData::Cuda(pooled) => pooled.slice(),
            StorageData::Cpu(_) => panic!("Storage is on CPU, not GPU"),
        }
    }
}

/// Write guard that provides mutable access to the CudaSlice.
#[cfg(feature = "cuda")]
pub struct CudaSliceWriteGuard<'a> {
    guard: parking_lot::RwLockWriteGuard<'a, StorageInner<f32>>,
}

#[cfg(feature = "cuda")]
impl<'a> CudaSliceWriteGuard<'a> {
    /// Returns a mutable reference to the CudaSlice.
    ///
    /// # Panics
    /// Panics if storage is CPU.
    pub fn slice_mut(&mut self) -> &mut CudaSlice<f32> {
        match &mut self.guard.data {
            StorageData::Cuda(pooled) => pooled.slice_mut(),
            StorageData::Cpu(_) => panic!("Storage is on CPU, not GPU"),
        }
    }
}

impl<T: Scalar> Clone for Storage<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            offset: self.offset,
            len: self.len,
        }
    }
}

// =============================================================================
// Guard Types for Safe Access
// =============================================================================

/// Read guard for storage data.
pub struct StorageReadGuard<'a, T: Scalar> {
    guard: parking_lot::RwLockReadGuard<'a, StorageInner<T>>,
    offset: usize,
    len: usize,
}

impl<T: Scalar> Deref for StorageReadGuard<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match &self.guard.data {
            StorageData::Cpu(data) => &data[self.offset..self.offset + self.len],
            #[cfg(feature = "cuda")]
            StorageData::Cuda(_) => panic!("Cannot access GPU storage as CPU slice. Use to_vec() for device-safe access."),
        }
    }
}

/// Write guard for storage data.
pub struct StorageWriteGuard<'a, T: Scalar> {
    guard: parking_lot::RwLockWriteGuard<'a, StorageInner<T>>,
    offset: usize,
    len: usize,
}

impl<T: Scalar> Deref for StorageWriteGuard<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match &self.guard.data {
            StorageData::Cpu(data) => &data[self.offset..self.offset + self.len],
            #[cfg(feature = "cuda")]
            StorageData::Cuda(_) => panic!("Cannot access GPU storage as CPU slice."),
        }
    }
}

impl<T: Scalar> DerefMut for StorageWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.guard.data {
            StorageData::Cpu(data) => &mut data[self.offset..self.offset + self.len],
            #[cfg(feature = "cuda")]
            StorageData::Cuda(_) => panic!("Cannot access GPU storage as mutable CPU slice."),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_zeros() {
        let storage = Storage::<f32>::zeros(10, Device::Cpu);
        assert_eq!(storage.len(), 10);
        assert!(!storage.is_empty());

        let data = storage.as_slice();
        for &val in data.iter() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_storage_from_vec() {
        let vec = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let storage = Storage::from_vec(vec.clone(), Device::Cpu);

        let data = storage.as_slice();
        assert_eq!(&*data, &vec[..]);
    }

    #[test]
    fn test_storage_slice() {
        let vec = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let storage = Storage::from_vec(vec, Device::Cpu);
        let slice = storage.slice(1, 3).unwrap();

        assert_eq!(slice.len(), 3);
        let data = slice.as_slice();
        assert_eq!(&*data, &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_storage_clone_shares() {
        let storage1 = Storage::<f32>::zeros(10, Device::Cpu);
        let storage2 = storage1.clone();

        assert!(!storage1.is_unique());
        assert!(!storage2.is_unique());
    }

    #[test]
    fn test_storage_deep_copy() {
        let storage1 = Storage::from_vec(vec![1.0_f32, 2.0, 3.0], Device::Cpu);
        let storage2 = storage1.deep_copy();

        assert!(storage1.is_unique());
        assert!(storage2.is_unique());

        // Modify storage2
        storage2.as_slice_mut()[0] = 99.0;

        // storage1 should be unchanged
        assert_eq!(storage1.as_slice()[0], 1.0);
    }

    #[test]
    fn test_storage_copy_from() {
        let src = Storage::from_vec(vec![1.0_f32, 2.0, 3.0], Device::Cpu);
        let dst = Storage::<f32>::zeros(3, Device::Cpu);

        dst.copy_from(&src).unwrap();

        let data = dst.as_slice();
        assert_eq!(&*data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_storage_slice_out_of_bounds() {
        let storage = Storage::<f32>::zeros(10, Device::Cpu);
        let result = storage.slice(5, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_to_vec_cpu() {
        let storage = Storage::from_vec(vec![1.0_f32, 2.0, 3.0], Device::Cpu);
        assert_eq!(storage.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_storage_is_cpu() {
        let storage = Storage::from_vec(vec![1.0_f32], Device::Cpu);
        assert!(storage.is_cpu());
        assert!(!storage.is_gpu());
    }
}
