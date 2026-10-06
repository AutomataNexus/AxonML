//! Memory allocation traits and the default CPU allocator.
//!
//! `DefaultAllocator` names the host device and reports `total_memory()` /
//! `free_memory()` via sysinfo (`None` without `std`, where there is no OS to
//! ask). Host buffers are ordinary `Vec`s; there is
//! no raw-pointer allocation API.
//! The `Allocator` trait is the extension point for custom allocators
//! (arena-based, pool-based, or device-specific).
//!
//! # File
//! `crates/axonml-core/src/allocator.rs`
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

use crate::device::Device;
#[cfg(feature = "std")]
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

    /// Returns the total memory available on the device.
    #[must_use]
    pub fn total_memory(&self) -> Option<usize> {
        #[cfg(feature = "std")]
        {
            Some(System::new_all().total_memory() as usize)
        }
        #[cfg(not(feature = "std"))]
        {
            None
        }
    }

    /// Returns the currently free memory on the device.
    #[must_use]
    pub fn free_memory(&self) -> Option<usize> {
        #[cfg(feature = "std")]
        {
            Some(System::new_all().available_memory() as usize)
        }
        #[cfg(not(feature = "std"))]
        {
            None
        }
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

    /// Returns the total memory, or `None` when it cannot be determined.
    fn total_memory(&self) -> Option<usize>;

    /// Returns the free memory, or `None` when it cannot be determined.
    fn free_memory(&self) -> Option<usize>;
}

impl Allocator for DefaultAllocator {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn total_memory(&self) -> Option<usize> {
        self.total_memory()
    }

    fn free_memory(&self) -> Option<usize> {
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
        assert!(alloc.total_memory().is_some_and(|m| m > 0));
    }
}
