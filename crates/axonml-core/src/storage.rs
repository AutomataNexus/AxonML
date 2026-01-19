//! Storage - Raw Memory Management for Tensors
//!
//! Provides efficient memory storage that underlies all tensor operations.
//! Storage is reference-counted for efficient sharing between tensor views.
//!
//! # Key Features
//! - Reference-counted memory for efficient views
//! - Device-agnostic storage interface
//! - Zero-copy slicing through offset/length
//! - Automatic memory cleanup
//!
//! # Example
//! ```rust
//! use axonml_core::{Storage, Device};
//!
//! // Create storage for 100 f32 values on CPU
//! let storage = Storage::<f32>::zeros(100, Device::Cpu);
//! assert_eq!(storage.len(), 100);
//! ```
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use core::ops::{Deref, DerefMut};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::device::Device;
use crate::dtype::Scalar;
use crate::error::{Error, Result};

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
    /// Raw data pointer (owned).
    data: Vec<T>,
    /// The device this storage resides on.
    device: Device,
}

impl<T: Scalar> Storage<T> {
    /// Creates new storage with the given capacity, initialized to zero.
    ///
    /// # Arguments
    /// * `len` - Number of elements to allocate
    /// * `device` - Device to allocate on
    ///
    /// # Returns
    /// New storage initialized to zeros.
    #[must_use]
    pub fn zeros(len: usize, device: Device) -> Self {
        let data = vec![T::zeroed(); len];
        Self::from_vec(data, device)
    }

    /// Creates storage from an existing vector.
    ///
    /// # Arguments
    /// * `data` - Vector of data
    /// * `device` - Device the storage is on
    ///
    /// # Returns
    /// New storage wrapping the data.
    #[must_use]
    pub fn from_vec(data: Vec<T>, device: Device) -> Self {
        let len = data.len();
        Self {
            inner: Arc::new(RwLock::new(StorageInner { data, device })),
            offset: 0,
            len,
        }
    }

    /// Creates storage from a slice by copying the data.
    ///
    /// # Arguments
    /// * `data` - Slice of data to copy
    /// * `device` - Device to allocate on
    ///
    /// # Returns
    /// New storage containing a copy of the data.
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

    /// Returns the size in bytes of this storage.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * core::mem::size_of::<T>()
    }

    /// Creates a view into a portion of this storage.
    ///
    /// # Arguments
    /// * `offset` - Starting offset relative to this view
    /// * `len` - Number of elements in the new view
    ///
    /// # Returns
    /// A new storage view, or error if bounds are invalid.
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

    /// Returns an immutable reference to the data.
    ///
    /// # Panics
    /// Panics if the lock is poisoned.
    #[must_use]
    pub fn as_slice(&self) -> StorageReadGuard<'_, T> {
        StorageReadGuard {
            guard: self.inner.read(),
            offset: self.offset,
            len: self.len,
        }
    }

    /// Returns a mutable reference to the data.
    ///
    /// # Panics
    /// Panics if the lock is poisoned.
    #[must_use]
    pub fn as_slice_mut(&self) -> StorageWriteGuard<'_, T> {
        StorageWriteGuard {
            guard: self.inner.write(),
            offset: self.offset,
            len: self.len,
        }
    }

    /// Copies data from another storage into this one.
    ///
    /// # Arguments
    /// * `other` - Source storage to copy from
    ///
    /// # Returns
    /// Ok if successful, error if lengths don't match.
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
    #[must_use]
    pub fn deep_copy(&self) -> Self {
        let data = self.as_slice().to_vec();
        Self::from_vec(data, self.device())
    }

    /// Transfers this storage to a different device.
    ///
    /// # Arguments
    /// * `device` - Target device
    ///
    /// # Returns
    /// New storage on the target device.
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }

        // For now, only CPU is supported
        if !device.is_cpu() {
            return Err(Error::DeviceNotAvailable { device });
        }

        Ok(self.deep_copy())
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
        &self.guard.data[self.offset..self.offset + self.len]
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
        &self.guard.data[self.offset..self.offset + self.len]
    }
}

impl<T: Scalar> DerefMut for StorageWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard.data[self.offset..self.offset + self.len]
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
}
