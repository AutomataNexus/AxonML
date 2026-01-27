//! axonml-profile - Profiling Tools for Axonml ML Framework
//!
//! Provides comprehensive profiling capabilities for neural network training
//! and inference, including memory tracking, compute profiling, and bottleneck detection.
//!
//! # Key Features
//! - Memory profiler: Track allocations, peak usage, and memory leaks
//! - Compute profiler: Measure operation times, FLOPS, and throughput
//! - Timeline profiler: Record events with timestamps for visualization
//! - Bottleneck detection: Identify performance bottlenecks automatically
//!
//! # Example
//! ```ignore
//! use axonml_profile::{Profiler, MemoryProfiler, ComputeProfiler};
//!
//! // Create a profiler
//! let profiler = Profiler::new();
//!
//! // Profile a forward pass
//! profiler.start("forward_pass");
//! let output = model.forward(&input);
//! profiler.stop("forward_pass");
//!
//! // Print summary
//! profiler.summary();
//! ```
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod bottleneck;
pub mod compute;
pub mod error;
pub mod memory;
pub mod report;
pub mod timeline;

pub use bottleneck::{Bottleneck, BottleneckAnalyzer, BottleneckType};
pub use compute::{ComputeProfiler, OperationStats, ProfiledOp};
pub use error::{ProfileError, ProfileResult};
pub use memory::{AllocationRecord, MemoryProfiler, MemoryStats};
pub use report::{ProfileReport, ReportFormat};
pub use timeline::{Event, EventType, TimelineProfiler};

use parking_lot::RwLock;
use std::sync::Arc;

// =============================================================================
// Unified Profiler
// =============================================================================

/// Unified profiler combining memory, compute, and timeline profiling.
#[derive(Debug)]
pub struct Profiler {
    /// Memory profiler instance
    pub memory: Arc<RwLock<MemoryProfiler>>,
    /// Compute profiler instance
    pub compute: Arc<RwLock<ComputeProfiler>>,
    /// Timeline profiler instance
    pub timeline: Arc<RwLock<TimelineProfiler>>,
    /// Whether profiling is enabled
    enabled: bool,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    /// Creates a new unified profiler.
    pub fn new() -> Self {
        Self {
            memory: Arc::new(RwLock::new(MemoryProfiler::new())),
            compute: Arc::new(RwLock::new(ComputeProfiler::new())),
            timeline: Arc::new(RwLock::new(TimelineProfiler::new())),
            enabled: true,
        }
    }

    /// Enables or disables profiling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Starts profiling an operation.
    pub fn start(&self, name: &str) {
        if self.enabled {
            self.compute.write().start(name);
            self.timeline.write().record(name, EventType::Start);
        }
    }

    /// Stops profiling an operation.
    pub fn stop(&self, name: &str) {
        if self.enabled {
            self.compute.write().stop(name);
            self.timeline.write().record(name, EventType::End);
        }
    }

    /// Records a memory allocation.
    pub fn record_alloc(&self, name: &str, bytes: usize) {
        if self.enabled {
            self.memory.write().record_alloc(name, bytes);
        }
    }

    /// Records a memory deallocation.
    pub fn record_free(&self, name: &str, bytes: usize) {
        if self.enabled {
            self.memory.write().record_free(name, bytes);
        }
    }

    /// Gets peak memory usage in bytes.
    pub fn peak_memory(&self) -> usize {
        self.memory.read().peak_usage()
    }

    /// Gets current memory usage in bytes.
    pub fn current_memory(&self) -> usize {
        self.memory.read().current_usage()
    }

    /// Gets the total time spent on an operation.
    pub fn total_time(&self, name: &str) -> std::time::Duration {
        self.compute.read().total_time(name)
    }

    /// Gets the average time for an operation.
    pub fn avg_time(&self, name: &str) -> std::time::Duration {
        self.compute.read().avg_time(name)
    }

    /// Resets all profiling data.
    pub fn reset(&self) {
        self.memory.write().reset();
        self.compute.write().reset();
        self.timeline.write().reset();
    }

    /// Generates a summary report.
    pub fn summary(&self) -> ProfileReport {
        ProfileReport::generate(self)
    }

    /// Prints a summary to stdout.
    pub fn print_summary(&self) {
        println!("{}", self.summary());
    }

    /// Analyzes for bottlenecks.
    pub fn analyze_bottlenecks(&self) -> Vec<Bottleneck> {
        let analyzer = BottleneckAnalyzer::new();
        let compute_stats = self.compute.read().all_stats();
        let memory_stats = self.memory.read().stats();
        analyzer.analyze(&compute_stats, &memory_stats)
    }
}

/// RAII guard for automatic profiling scope.
pub struct ProfileGuard<'a> {
    profiler: &'a Profiler,
    name: String,
}

impl<'a> ProfileGuard<'a> {
    /// Creates a new profile guard.
    pub fn new(profiler: &'a Profiler, name: &str) -> Self {
        profiler.start(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ProfileGuard<'a> {
    fn drop(&mut self) {
        self.profiler.stop(&self.name);
    }
}

/// Creates a profile guard for automatic scope-based profiling.
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr) => {
        let _guard = $crate::ProfileGuard::new($profiler, $name);
    };
}

// =============================================================================
// Global Profiler
// =============================================================================

use std::sync::OnceLock;

static GLOBAL_PROFILER: OnceLock<Profiler> = OnceLock::new();

/// Gets the global profiler instance.
pub fn global_profiler() -> &'static Profiler {
    GLOBAL_PROFILER.get_or_init(Profiler::new)
}

/// Starts profiling an operation using the global profiler.
pub fn start(name: &str) {
    global_profiler().start(name);
}

/// Stops profiling an operation using the global profiler.
pub fn stop(name: &str) {
    global_profiler().stop(name);
}

/// Records a memory allocation using the global profiler.
pub fn record_alloc(name: &str, bytes: usize) {
    global_profiler().record_alloc(name, bytes);
}

/// Records a memory deallocation using the global profiler.
pub fn record_free(name: &str, bytes: usize) {
    global_profiler().record_free(name, bytes);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_profile_operation() {
        let profiler = Profiler::new();
        profiler.start("test_op");
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.stop("test_op");

        let total = profiler.total_time("test_op");
        assert!(total.as_millis() >= 10);
    }

    #[test]
    fn test_memory_tracking() {
        let profiler = Profiler::new();
        profiler.record_alloc("tensor_a", 1024);
        profiler.record_alloc("tensor_b", 2048);

        assert_eq!(profiler.current_memory(), 3072);
        assert_eq!(profiler.peak_memory(), 3072);

        profiler.record_free("tensor_a", 1024);
        assert_eq!(profiler.current_memory(), 2048);
        assert_eq!(profiler.peak_memory(), 3072);
    }

    #[test]
    fn test_profile_guard() {
        let profiler = Profiler::new();
        {
            let _guard = ProfileGuard::new(&profiler, "scoped_op");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let total = profiler.total_time("scoped_op");
        assert!(total.as_millis() >= 5);
    }

    #[test]
    fn test_reset() {
        let profiler = Profiler::new();
        profiler.start("test");
        profiler.stop("test");
        profiler.record_alloc("mem", 1000);

        profiler.reset();

        assert_eq!(profiler.current_memory(), 0);
        assert_eq!(profiler.total_time("test"), std::time::Duration::ZERO);
    }
}
