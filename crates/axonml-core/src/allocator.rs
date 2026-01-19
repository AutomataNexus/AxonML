//! Allocator - Memory Allocation Traits and Implementations
//!
//! Provides the allocator abstraction for device-specific memory management.
//! Each backend implements the Allocator trait for its memory operations.
//!
//! # Key Features
//! - Unified allocator trait for all devices
//! - Pluggable allocator implementations
//! - Memory pool support for performance
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use crate::device::Device;
use crate::dtype::Scalar;
use crate::error::Result;
use sysinfo::System;

// =============================================================================
// Default Allocator
// =============================================================================

/// Default CPU allocator using system memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultAllocator;

impl DefaultAllocator {
    /// Creates a new default allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Returns the device this allocator is for.
    #[must_use]
    pub const fn device(&self) -> Device {
        Device::Cpu
    }

    /// Allocates memory for `count` elements of type T.
    pub fn allocate<T: Scalar>(&self, count: usize) -> Result<*mut T> {
        let mut vec = Vec::<T>::with_capacity(count);
        let ptr = vec.as_mut_ptr();
        core::mem::forget(vec);
        Ok(ptr)
    }

    /// Deallocates memory previously allocated.
    ///
    /// # Safety
    /// The pointer must have been allocated by this allocator.
    pub unsafe fn deallocate<T: Scalar>(&self, ptr: *mut T, count: usize) {
        drop(Vec::from_raw_parts(ptr, 0, count));
    }

    /// Copies memory from one location to another.
    ///
    /// # Safety
    /// Both pointers must be valid for `count` elements.
    pub unsafe fn copy<T: Scalar>(&self, dst: *mut T, src: *const T, count: usize) {
        core::ptr::copy_nonoverlapping(src, dst, count);
    }

    /// Fills memory with zeros.
    ///
    /// # Safety
    /// The pointer must be valid for `count` elements.
    pub unsafe fn zero<T: Scalar>(&self, ptr: *mut T, count: usize) {
        core::ptr::write_bytes(ptr, 0, count);
    }

    /// Returns the total memory available on the device.
    #[must_use]
    pub fn total_memory(&self) -> usize {
        let sys = System::new_all();
        sys.total_memory() as usize
    }

    /// Returns the currently free memory on the device.
    #[must_use]
    pub fn free_memory(&self) -> usize {
        let sys = System::new_all();
        sys.available_memory() as usize
    }
}

// =============================================================================
// Allocator Trait (for future extensibility)
// =============================================================================

/// Marker trait for types that can act as allocators.
///
/// Note: Due to Rust's object safety rules, we use concrete types
/// instead of dynamic dispatch for allocators.
pub trait Allocator {
    /// Returns the device this allocator is for.
    fn device(&self) -> Device;

    /// Returns the total memory available.
    fn total_memory(&self) -> usize;

    /// Returns the free memory available.
    fn free_memory(&self) -> usize;
}

impl Allocator for DefaultAllocator {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn total_memory(&self) -> usize {
        self.total_memory()
    }

    fn free_memory(&self) -> usize {
        self.free_memory()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_allocator() {
        let alloc = DefaultAllocator::new();
        assert_eq!(alloc.device(), Device::Cpu);
        assert!(alloc.total_memory() > 0);
    }

    #[test]
    fn test_allocate_deallocate() {
        let alloc = DefaultAllocator::new();

        let ptr = alloc.allocate::<f32>(100).unwrap();
        assert!(!ptr.is_null());

        unsafe {
            alloc.zero(ptr, 100);
            alloc.deallocate(ptr, 100);
        }
    }

    #[test]
    fn test_copy() {
        let alloc = DefaultAllocator::new();

        let src = alloc.allocate::<f32>(10).unwrap();
        let dst = alloc.allocate::<f32>(10).unwrap();

        unsafe {
            for i in 0..10 {
                *src.add(i) = i as f32;
            }

            alloc.copy(dst, src, 10);

            for i in 0..10 {
                assert_eq!(*dst.add(i), i as f32);
            }

            alloc.deallocate(src, 10);
            alloc.deallocate(dst, 10);
        }
    }
}
