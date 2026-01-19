//! Memory Profiling Module
//!
//! Tracks memory allocations, deallocations, peak usage, and memory leaks.

use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};

/// Record of a single memory allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRecord {
    /// Name/identifier for this allocation
    pub name: String,
    /// Size in bytes
    pub size: usize,
    /// Timestamp when allocated
    pub timestamp: u64,
    /// Whether this allocation has been freed
    pub freed: bool,
}

/// Summary statistics for memory usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Total bytes freed
    pub total_freed: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Per-name memory usage
    pub per_name_usage: HashMap<String, usize>,
}

/// Memory profiler for tracking allocations and usage.
#[derive(Debug)]
pub struct MemoryProfiler {
    /// Active allocations by name
    allocations: HashMap<String, Vec<AllocationRecord>>,
    /// Current total memory usage
    current_usage: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Total bytes allocated
    total_allocated: usize,
    /// Total bytes freed
    total_freed: usize,
    /// Number of allocations
    allocation_count: usize,
    /// Number of deallocations
    deallocation_count: usize,
    /// Start time for relative timestamps
    start_time: Instant,
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProfiler {
    /// Creates a new memory profiler.
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            current_usage: 0,
            peak_usage: 0,
            total_allocated: 0,
            total_freed: 0,
            allocation_count: 0,
            deallocation_count: 0,
            start_time: Instant::now(),
        }
    }

    /// Records a memory allocation.
    pub fn record_alloc(&mut self, name: &str, bytes: usize) {
        let timestamp = self.start_time.elapsed().as_micros() as u64;

        let record = AllocationRecord {
            name: name.to_string(),
            size: bytes,
            timestamp,
            freed: false,
        };

        self.allocations
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(record);

        self.current_usage += bytes;
        self.total_allocated += bytes;
        self.allocation_count += 1;

        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    /// Records a memory deallocation.
    pub fn record_free(&mut self, name: &str, bytes: usize) {
        if let Some(records) = self.allocations.get_mut(name) {
            // Find first non-freed allocation with matching size
            for record in records.iter_mut() {
                if !record.freed && record.size == bytes {
                    record.freed = true;
                    break;
                }
            }
        }

        self.current_usage = self.current_usage.saturating_sub(bytes);
        self.total_freed += bytes;
        self.deallocation_count += 1;
    }

    /// Returns current memory usage in bytes.
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Returns peak memory usage in bytes.
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Returns total bytes allocated.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Returns total bytes freed.
    pub fn total_freed(&self) -> usize {
        self.total_freed
    }

    /// Returns memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let mut per_name_usage = HashMap::new();

        for (name, records) in &self.allocations {
            let active_bytes: usize = records
                .iter()
                .filter(|r| !r.freed)
                .map(|r| r.size)
                .sum();
            if active_bytes > 0 {
                per_name_usage.insert(name.clone(), active_bytes);
            }
        }

        MemoryStats {
            current_usage: self.current_usage,
            peak_usage: self.peak_usage,
            total_allocated: self.total_allocated,
            total_freed: self.total_freed,
            allocation_count: self.allocation_count,
            deallocation_count: self.deallocation_count,
            per_name_usage,
        }
    }

    /// Returns memory leaks (allocations not freed).
    pub fn leaks(&self) -> Vec<AllocationRecord> {
        self.allocations
            .values()
            .flatten()
            .filter(|r| !r.freed)
            .cloned()
            .collect()
    }

    /// Resets the profiler.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.current_usage = 0;
        self.peak_usage = 0;
        self.total_allocated = 0;
        self.total_freed = 0;
        self.allocation_count = 0;
        self.deallocation_count = 0;
        self.start_time = Instant::now();
    }

    /// Formats bytes into human-readable string.
    pub fn format_bytes(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_tracking() {
        let mut profiler = MemoryProfiler::new();

        profiler.record_alloc("tensor_a", 1024);
        assert_eq!(profiler.current_usage(), 1024);

        profiler.record_alloc("tensor_b", 2048);
        assert_eq!(profiler.current_usage(), 3072);
        assert_eq!(profiler.peak_usage(), 3072);
    }

    #[test]
    fn test_deallocation() {
        let mut profiler = MemoryProfiler::new();

        profiler.record_alloc("tensor", 1000);
        profiler.record_free("tensor", 1000);

        assert_eq!(profiler.current_usage(), 0);
        assert_eq!(profiler.peak_usage(), 1000);
    }

    #[test]
    fn test_leak_detection() {
        let mut profiler = MemoryProfiler::new();

        profiler.record_alloc("leak1", 100);
        profiler.record_alloc("leak2", 200);
        profiler.record_alloc("freed", 300);
        profiler.record_free("freed", 300);

        let leaks = profiler.leaks();
        assert_eq!(leaks.len(), 2);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryProfiler::format_bytes(500), "500 B");
        assert_eq!(MemoryProfiler::format_bytes(1024), "1.00 KB");
        assert_eq!(MemoryProfiler::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(MemoryProfiler::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
