//! Memory allocation traits and the default CPU allocator.
//!
//! `DefaultAllocator` names the host device and reports `total_memory()` /
//! `free_memory()` via sysinfo. Host buffers are ordinary `Vec`s; there is
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

use crate::device::Device;
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
}
