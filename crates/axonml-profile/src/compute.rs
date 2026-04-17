//! Compute Profiling — Per-Operation Timing, FLOPs, and Bandwidth
//!
//! Profiles per-operation execution using `Instant::now()` timestamps.
//! `OperationStats` is the aggregate record (name, `call_count`,
//! `total_time_ns`, `min_time_ns`, `max_time_ns`, optional total `flops`, and
//! optional total `bytes_processed`), with derived accessors: `avg_time` /
//! `total_time` / `min_time` / `max_time` (as `Duration`), `gflops`
//! (`flops / 1e9`), `gflops_per_sec` (total FLOPs divided by elapsed
//! seconds, in GFLOPS/s), and `bandwidth_gbps` (bytes processed per second
//! in GB/s). `ProfiledOp` is the in-flight record (name, `start: Instant`,
//! optional FLOPs and bytes). `ComputeProfiler` maintains a
//! `HashMap<String, OperationStats>` plus a `HashMap<String, Vec<ProfiledOp>>`
//! stack per-name so nested / re-entrant starts and stops interleave
//! correctly. Public API: `start` / `start_with_flops` / `start_with_bytes`
//! push an active op; `stop(name)` pops the most recent one, computes elapsed,
//! and folds it into min/max/total/count and accumulates FLOPs and bytes.
//! Query helpers include `get_stats`, `all_stats` (clone), `total_time`,
//! `avg_time`, `top_by_time(n)`, `top_by_calls(n)`, `reset`, and a
//! `format_duration` pretty-printer that chooses ns/us/ms/s.
//! `TimingGuard<'a>` is the RAII wrapper that calls `start` on `new` and
//! `stop` on `Drop`. Tests cover single-op timing, five-call aggregation,
//! nested outer/inner ops, `top_by_time` ordering, and the format-duration
//! unit suffixes.
//!
//! # File
//! `crates/axonml-profile/src/compute.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// =============================================================================
// OperationStats
// =============================================================================

/// Statistics for a single profiled operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperationStats {
    /// Name of the operation
    pub name: String,
    /// Number of times this operation was called
    pub call_count: usize,
    /// Total time spent in this operation
    pub total_time_ns: u64,
    /// Minimum execution time
    pub min_time_ns: u64,
    /// Maximum execution time
    pub max_time_ns: u64,
    /// FLOPS (if computed)
    pub flops: Option<f64>,
    /// Bytes processed (for bandwidth calculation)
    pub bytes_processed: Option<usize>,
}

impl OperationStats {
    /// Returns the average execution time.
    pub fn avg_time(&self) -> Duration {
        if self.call_count == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(self.total_time_ns / self.call_count as u64)
        }
    }

    /// Returns the total execution time.
    pub fn total_time(&self) -> Duration {
        Duration::from_nanos(self.total_time_ns)
    }

    /// Returns the minimum execution time.
    pub fn min_time(&self) -> Duration {
        Duration::from_nanos(self.min_time_ns)
    }

    /// Returns the maximum execution time.
    pub fn max_time(&self) -> Duration {
        Duration::from_nanos(self.max_time_ns)
    }

    /// Returns total GFLOPS (accumulated across all calls).
    pub fn gflops(&self) -> Option<f64> {
        self.flops.map(|f| f / 1e9)
    }

    /// Returns GFLOPS/sec throughput (total GFLOPS / total time).
    pub fn gflops_per_sec(&self) -> Option<f64> {
        if let Some(flops) = self.flops {
            if self.total_time_ns > 0 {
                let seconds = self.total_time_ns as f64 / 1e9;
                Some(flops / seconds / 1e9)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Returns bandwidth in GB/s if bytes_processed is set.
    pub fn bandwidth_gbps(&self) -> Option<f64> {
        if let Some(bytes) = self.bytes_processed {
            if self.total_time_ns > 0 {
                let seconds = self.total_time_ns as f64 / 1e9;
                Some(bytes as f64 / seconds / 1e9)
            } else {
                None
            }
        } else {
            None
        }
    }
}

// =============================================================================
// ProfiledOp
// =============================================================================

/// A profiled operation with timing.
#[derive(Debug, Clone)]
pub struct ProfiledOp {
    /// Operation name
    pub name: String,
    /// Start time
    pub start: Instant,
    /// FLOPS count (optional)
    pub flops: Option<f64>,
    /// Bytes processed (optional)
    pub bytes: Option<usize>,
}

// =============================================================================
// ComputeProfiler
// =============================================================================

/// Compute profiler for tracking operation execution times.
#[derive(Debug)]
pub struct ComputeProfiler {
    /// Statistics per operation name
    stats: HashMap<String, OperationStats>,
    /// Currently active operations
    active: HashMap<String, Vec<ProfiledOp>>,
}

impl Default for ComputeProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeProfiler {
    /// Creates a new compute profiler.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            active: HashMap::new(),
        }
    }

    // -------------------------------------------------------------------------
    // Start/Stop Timing
    // -------------------------------------------------------------------------

    /// Starts profiling an operation.
    pub fn start(&mut self, name: &str) {
        let op = ProfiledOp {
            name: name.to_string(),
            start: Instant::now(),
            flops: None,
            bytes: None,
        };

        self.active.entry(name.to_string()).or_default().push(op);
    }

    /// Starts profiling an operation with FLOPS count.
    pub fn start_with_flops(&mut self, name: &str, flops: f64) {
        let op = ProfiledOp {
            name: name.to_string(),
            start: Instant::now(),
            flops: Some(flops),
            bytes: None,
        };

        self.active.entry(name.to_string()).or_default().push(op);
    }

    /// Starts profiling an operation with bytes processed.
    pub fn start_with_bytes(&mut self, name: &str, bytes: usize) {
        let op = ProfiledOp {
            name: name.to_string(),
            start: Instant::now(),
            flops: None,
            bytes: Some(bytes),
        };

        self.active.entry(name.to_string()).or_default().push(op);
    }

    /// Stops profiling an operation and records its duration.
    pub fn stop(&mut self, name: &str) {
        let _elapsed = if let Some(ops) = self.active.get_mut(name) {
            if let Some(op) = ops.pop() {
                let elapsed = op.start.elapsed();

                // Update stats
                let stats = self
                    .stats
                    .entry(name.to_string())
                    .or_insert_with(|| OperationStats {
                        name: name.to_string(),
                        min_time_ns: u64::MAX,
                        ..Default::default()
                    });

                let elapsed_ns = elapsed.as_nanos() as u64;
                stats.call_count += 1;
                stats.total_time_ns += elapsed_ns;
                stats.min_time_ns = stats.min_time_ns.min(elapsed_ns);
                stats.max_time_ns = stats.max_time_ns.max(elapsed_ns);

                if let Some(flops) = op.flops {
                    stats.flops = Some(stats.flops.unwrap_or(0.0) + flops);
                }
                if let Some(bytes) = op.bytes {
                    stats.bytes_processed = Some(stats.bytes_processed.unwrap_or(0) + bytes);
                }

                Some(elapsed)
            } else {
                None
            }
        } else {
            None
        };
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    /// Gets statistics for a specific operation.
    pub fn get_stats(&self, name: &str) -> Option<&OperationStats> {
        self.stats.get(name)
    }

    /// Gets all operation statistics.
    pub fn all_stats(&self) -> HashMap<String, OperationStats> {
        self.stats.clone()
    }

    /// Gets total time for an operation.
    pub fn total_time(&self, name: &str) -> Duration {
        self.stats
            .get(name)
            .map(|s| s.total_time())
            .unwrap_or(Duration::ZERO)
    }

    /// Gets average time for an operation.
    pub fn avg_time(&self, name: &str) -> Duration {
        self.stats
            .get(name)
            .map(|s| s.avg_time())
            .unwrap_or(Duration::ZERO)
    }

    /// Gets the top N operations by total time.
    pub fn top_by_time(&self, n: usize) -> Vec<&OperationStats> {
        let mut sorted: Vec<_> = self.stats.values().collect();
        sorted.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));
        sorted.into_iter().take(n).collect()
    }

    /// Gets the top N operations by call count.
    pub fn top_by_calls(&self, n: usize) -> Vec<&OperationStats> {
        let mut sorted: Vec<_> = self.stats.values().collect();
        sorted.sort_by(|a, b| b.call_count.cmp(&a.call_count));
        sorted.into_iter().take(n).collect()
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.stats.clear();
        self.active.clear();
    }

    // -------------------------------------------------------------------------
    // Formatting
    // -------------------------------------------------------------------------

    /// Formats a duration for display.
    pub fn format_duration(d: Duration) -> String {
        let nanos = d.as_nanos();
        if nanos >= 1_000_000_000 {
            format!("{:.3} s", d.as_secs_f64())
        } else if nanos >= 1_000_000 {
            format!("{:.3} ms", nanos as f64 / 1_000_000.0)
        } else if nanos >= 1_000 {
            format!("{:.3} us", nanos as f64 / 1_000.0)
        } else {
            format!("{} ns", nanos)
        }
    }
}

// =============================================================================
// TimingGuard (RAII)
// =============================================================================

/// RAII guard for automatic operation timing.
pub struct TimingGuard<'a> {
    profiler: &'a mut ComputeProfiler,
    name: String,
}

impl<'a> TimingGuard<'a> {
    /// Creates a new timing guard.
    pub fn new(profiler: &'a mut ComputeProfiler, name: &str) -> Self {
        profiler.start(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for TimingGuard<'a> {
    fn drop(&mut self) {
        self.profiler.stop(&self.name);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_timing() {
        let mut profiler = ComputeProfiler::new();

        profiler.start("test_op");
        std::thread::sleep(Duration::from_millis(10));
        profiler.stop("test_op");

        let stats = profiler.get_stats("test_op").unwrap();
        assert_eq!(stats.call_count, 1);
        assert!(stats.total_time() >= Duration::from_millis(10));
    }

    #[test]
    fn test_multiple_calls() {
        let mut profiler = ComputeProfiler::new();

        for _ in 0..5 {
            profiler.start("multi_op");
            std::thread::sleep(Duration::from_millis(1));
            profiler.stop("multi_op");
        }

        let stats = profiler.get_stats("multi_op").unwrap();
        assert_eq!(stats.call_count, 5);
    }

    #[test]
    fn test_nested_operations() {
        let mut profiler = ComputeProfiler::new();

        profiler.start("outer");
        profiler.start("inner");
        std::thread::sleep(Duration::from_millis(5));
        profiler.stop("inner");
        std::thread::sleep(Duration::from_millis(5));
        profiler.stop("outer");

        let outer = profiler.get_stats("outer").unwrap();
        let inner = profiler.get_stats("inner").unwrap();

        assert!(outer.total_time() >= inner.total_time());
    }

    #[test]
    fn test_top_operations() {
        let mut profiler = ComputeProfiler::new();

        profiler.start("slow");
        std::thread::sleep(Duration::from_millis(20));
        profiler.stop("slow");

        profiler.start("fast");
        std::thread::sleep(Duration::from_millis(5));
        profiler.stop("fast");

        let top = profiler.top_by_time(2);
        assert_eq!(top[0].name, "slow");
        assert_eq!(top[1].name, "fast");
    }

    #[test]
    fn test_format_duration() {
        assert!(ComputeProfiler::format_duration(Duration::from_nanos(500)).contains("ns"));
        assert!(ComputeProfiler::format_duration(Duration::from_micros(500)).contains("us"));
        assert!(ComputeProfiler::format_duration(Duration::from_millis(500)).contains("ms"));
        assert!(ComputeProfiler::format_duration(Duration::from_secs(5)).contains("s"));
    }
}
