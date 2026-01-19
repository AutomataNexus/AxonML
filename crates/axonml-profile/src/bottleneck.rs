//! Bottleneck Detection Module
//!
//! Analyzes profiling data to identify performance bottlenecks.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::compute::OperationStats;
use crate::memory::MemoryStats;

/// Type of bottleneck detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Operation taking disproportionate time
    SlowOperation,
    /// Operation called too frequently
    HighCallCount,
    /// Memory allocation hotspot
    MemoryHotspot,
    /// Memory leak suspected
    MemoryLeak,
    /// Low computational throughput
    LowThroughput,
    /// High memory bandwidth usage
    MemoryBound,
    /// Unbalanced workload distribution
    LoadImbalance,
}

/// Severity level of a bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Minor issue, optimization optional
    Low,
    /// Notable issue, should consider fixing
    Medium,
    /// Significant issue, recommend fixing
    High,
    /// Critical issue, strongly recommend fixing
    Critical,
}

/// A detected bottleneck with details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Severity level
    pub severity: Severity,
    /// Name of the affected operation/resource
    pub name: String,
    /// Description of the issue
    pub description: String,
    /// Suggested fix or optimization
    pub suggestion: String,
    /// Relevant metrics
    pub metrics: HashMap<String, f64>,
}

/// Configuration for bottleneck analysis thresholds.
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Minimum percentage of total time for slow operation detection
    pub slow_op_threshold_pct: f64,
    /// Minimum call count for high call count detection
    pub high_call_threshold: usize,
    /// Minimum memory usage percentage for hotspot detection
    pub memory_hotspot_threshold_pct: f64,
    /// Minimum GFLOPS for throughput analysis
    pub min_gflops_threshold: f64,
    /// Whether to check for memory leaks
    pub check_memory_leaks: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            slow_op_threshold_pct: 20.0,  // 20% of total time
            high_call_threshold: 10000,    // 10k+ calls
            memory_hotspot_threshold_pct: 30.0, // 30% of peak memory
            min_gflops_threshold: 1.0,     // 1 GFLOPS minimum
            check_memory_leaks: true,
        }
    }
}

/// Analyzer for detecting performance bottlenecks.
#[derive(Debug)]
pub struct BottleneckAnalyzer {
    /// Configuration
    config: AnalyzerConfig,
}

impl Default for BottleneckAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl BottleneckAnalyzer {
    /// Creates a new bottleneck analyzer with default config.
    pub fn new() -> Self {
        Self {
            config: AnalyzerConfig::default(),
        }
    }

    /// Creates a new bottleneck analyzer with custom config.
    pub fn with_config(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyzes compute and memory stats for bottlenecks.
    pub fn analyze(
        &self,
        compute_stats: &HashMap<String, OperationStats>,
        memory_stats: &MemoryStats,
    ) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Analyze compute bottlenecks
        bottlenecks.extend(self.analyze_slow_operations(compute_stats));
        bottlenecks.extend(self.analyze_high_call_counts(compute_stats));
        bottlenecks.extend(self.analyze_throughput(compute_stats));

        // Analyze memory bottlenecks
        bottlenecks.extend(self.analyze_memory_hotspots(memory_stats));
        if self.config.check_memory_leaks {
            bottlenecks.extend(self.analyze_memory_leaks(memory_stats));
        }

        // Sort by severity (highest first)
        bottlenecks.sort_by(|a, b| b.severity.cmp(&a.severity));

        bottlenecks
    }

    /// Analyzes for slow operations.
    fn analyze_slow_operations(
        &self,
        stats: &HashMap<String, OperationStats>,
    ) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Calculate total time
        let total_time_ns: u64 = stats.values().map(|s| s.total_time_ns).sum();
        if total_time_ns == 0 {
            return bottlenecks;
        }

        for (name, op_stats) in stats {
            let pct = (op_stats.total_time_ns as f64 / total_time_ns as f64) * 100.0;

            if pct >= self.config.slow_op_threshold_pct {
                let severity = if pct >= 50.0 {
                    Severity::Critical
                } else if pct >= 35.0 {
                    Severity::High
                } else if pct >= 25.0 {
                    Severity::Medium
                } else {
                    Severity::Low
                };

                let mut metrics = HashMap::new();
                metrics.insert("time_percentage".to_string(), pct);
                metrics.insert("total_time_ms".to_string(), op_stats.total_time_ns as f64 / 1e6);
                metrics.insert("call_count".to_string(), op_stats.call_count as f64);

                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::SlowOperation,
                    severity,
                    name: name.clone(),
                    description: format!(
                        "Operation '{}' takes {:.1}% of total execution time ({:.2}ms across {} calls)",
                        name, pct, op_stats.total_time_ns as f64 / 1e6, op_stats.call_count
                    ),
                    suggestion: format!(
                        "Consider optimizing '{}' - look for algorithmic improvements, \
                        GPU acceleration, or caching opportunities",
                        name
                    ),
                    metrics,
                });
            }
        }

        bottlenecks
    }

    /// Analyzes for high call counts.
    fn analyze_high_call_counts(
        &self,
        stats: &HashMap<String, OperationStats>,
    ) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        for (name, op_stats) in stats {
            if op_stats.call_count >= self.config.high_call_threshold {
                let severity = if op_stats.call_count >= 100000 {
                    Severity::High
                } else if op_stats.call_count >= 50000 {
                    Severity::Medium
                } else {
                    Severity::Low
                };

                let avg_time_ns = if op_stats.call_count > 0 {
                    op_stats.total_time_ns / op_stats.call_count as u64
                } else {
                    0
                };

                let mut metrics = HashMap::new();
                metrics.insert("call_count".to_string(), op_stats.call_count as f64);
                metrics.insert("avg_time_ns".to_string(), avg_time_ns as f64);

                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::HighCallCount,
                    severity,
                    name: name.clone(),
                    description: format!(
                        "Operation '{}' called {} times (avg {:.2}us per call)",
                        name, op_stats.call_count, avg_time_ns as f64 / 1000.0
                    ),
                    suggestion: format!(
                        "Consider batching calls to '{}' or caching results if inputs repeat",
                        name
                    ),
                    metrics,
                });
            }
        }

        bottlenecks
    }

    /// Analyzes computational throughput.
    fn analyze_throughput(
        &self,
        stats: &HashMap<String, OperationStats>,
    ) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        for (name, op_stats) in stats {
            if let Some(flops) = op_stats.flops {
                let gflops = flops / 1e9;
                let seconds = op_stats.total_time_ns as f64 / 1e9;

                if seconds > 0.0 {
                    let gflops_per_sec = gflops / seconds;

                    if gflops_per_sec < self.config.min_gflops_threshold && gflops > 0.0 {
                        let mut metrics = HashMap::new();
                        metrics.insert("gflops_per_sec".to_string(), gflops_per_sec);
                        metrics.insert("total_gflops".to_string(), gflops);

                        bottlenecks.push(Bottleneck {
                            bottleneck_type: BottleneckType::LowThroughput,
                            severity: Severity::Medium,
                            name: name.clone(),
                            description: format!(
                                "Operation '{}' has low throughput: {:.2} GFLOPS/s",
                                name, gflops_per_sec
                            ),
                            suggestion: format!(
                                "Consider using optimized BLAS libraries or GPU acceleration for '{}'",
                                name
                            ),
                            metrics,
                        });
                    }
                }
            }
        }

        bottlenecks
    }

    /// Analyzes memory hotspots.
    fn analyze_memory_hotspots(&self, stats: &MemoryStats) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        if stats.peak_usage == 0 {
            return bottlenecks;
        }

        for (name, &usage) in &stats.per_name_usage {
            let pct = (usage as f64 / stats.peak_usage as f64) * 100.0;

            if pct >= self.config.memory_hotspot_threshold_pct {
                let severity = if pct >= 60.0 {
                    Severity::High
                } else if pct >= 45.0 {
                    Severity::Medium
                } else {
                    Severity::Low
                };

                let mut metrics = HashMap::new();
                metrics.insert("memory_percentage".to_string(), pct);
                metrics.insert("memory_bytes".to_string(), usage as f64);

                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::MemoryHotspot,
                    severity,
                    name: name.clone(),
                    description: format!(
                        "Memory allocation '{}' uses {:.1}% of peak memory ({} bytes)",
                        name, pct, usage
                    ),
                    suggestion: format!(
                        "Consider reducing memory for '{}' - use in-place operations, \
                        gradient checkpointing, or smaller batch sizes",
                        name
                    ),
                    metrics,
                });
            }
        }

        bottlenecks
    }

    /// Analyzes for potential memory leaks.
    fn analyze_memory_leaks(&self, stats: &MemoryStats) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Check if more allocated than freed
        if stats.total_allocated > stats.total_freed {
            let leaked = stats.total_allocated - stats.total_freed;
            let leak_pct = (leaked as f64 / stats.total_allocated as f64) * 100.0;

            if leak_pct > 5.0 {
                let severity = if leak_pct >= 50.0 {
                    Severity::Critical
                } else if leak_pct >= 25.0 {
                    Severity::High
                } else if leak_pct >= 10.0 {
                    Severity::Medium
                } else {
                    Severity::Low
                };

                let mut metrics = HashMap::new();
                metrics.insert("leak_percentage".to_string(), leak_pct);
                metrics.insert("leaked_bytes".to_string(), leaked as f64);

                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::MemoryLeak,
                    severity,
                    name: "memory_leak".to_string(),
                    description: format!(
                        "Potential memory leak detected: {} bytes ({:.1}%) not freed",
                        leaked, leak_pct
                    ),
                    suggestion: "Review memory management - ensure all allocations are properly freed. \
                        Consider using RAII patterns or explicit deallocation tracking.".to_string(),
                    metrics,
                });
            }
        }

        bottlenecks
    }

    /// Generates a summary report of bottlenecks.
    pub fn summary(bottlenecks: &[Bottleneck]) -> String {
        if bottlenecks.is_empty() {
            return "No bottlenecks detected.".to_string();
        }

        let mut output = String::new();
        output.push_str(&format!("Found {} bottleneck(s):\n\n", bottlenecks.len()));

        for (i, b) in bottlenecks.iter().enumerate() {
            let severity_str = match b.severity {
                Severity::Critical => "[CRITICAL]",
                Severity::High => "[HIGH]",
                Severity::Medium => "[MEDIUM]",
                Severity::Low => "[LOW]",
            };

            output.push_str(&format!(
                "{}. {} {:?}: {}\n",
                i + 1, severity_str, b.bottleneck_type, b.name
            ));
            output.push_str(&format!("   Description: {}\n", b.description));
            output.push_str(&format!("   Suggestion: {}\n\n", b.suggestion));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_compute_stats() -> HashMap<String, OperationStats> {
        let mut stats = HashMap::new();

        stats.insert("slow_op".to_string(), OperationStats {
            name: "slow_op".to_string(),
            call_count: 100,
            total_time_ns: 8_000_000_000, // 8 seconds (80%)
            min_time_ns: 70_000_000,
            max_time_ns: 90_000_000,
            flops: None,
            bytes_processed: None,
        });

        stats.insert("fast_op".to_string(), OperationStats {
            name: "fast_op".to_string(),
            call_count: 1000,
            total_time_ns: 2_000_000_000, // 2 seconds (20%)
            min_time_ns: 1_000_000,
            max_time_ns: 3_000_000,
            flops: None,
            bytes_processed: None,
        });

        stats
    }

    fn create_test_memory_stats() -> MemoryStats {
        let mut per_name = HashMap::new();
        per_name.insert("big_tensor".to_string(), 800_000_000); // 800 MB

        MemoryStats {
            current_usage: 500_000_000,
            peak_usage: 1_000_000_000, // 1 GB
            total_allocated: 2_000_000_000,
            total_freed: 1_500_000_000,
            allocation_count: 1000,
            deallocation_count: 800,
            per_name_usage: per_name,
        }
    }

    #[test]
    fn test_slow_operation_detection() {
        let analyzer = BottleneckAnalyzer::new();
        let compute_stats = create_test_compute_stats();
        let memory_stats = MemoryStats::default();

        let bottlenecks = analyzer.analyze(&compute_stats, &memory_stats);

        let slow_ops: Vec<_> = bottlenecks
            .iter()
            .filter(|b| b.bottleneck_type == BottleneckType::SlowOperation)
            .collect();

        assert!(!slow_ops.is_empty());
        assert_eq!(slow_ops[0].name, "slow_op");
    }

    #[test]
    fn test_memory_hotspot_detection() {
        let analyzer = BottleneckAnalyzer::new();
        let compute_stats = HashMap::new();
        let memory_stats = create_test_memory_stats();

        let bottlenecks = analyzer.analyze(&compute_stats, &memory_stats);

        let hotspots: Vec<_> = bottlenecks
            .iter()
            .filter(|b| b.bottleneck_type == BottleneckType::MemoryHotspot)
            .collect();

        assert!(!hotspots.is_empty());
        assert_eq!(hotspots[0].name, "big_tensor");
    }

    #[test]
    fn test_memory_leak_detection() {
        let analyzer = BottleneckAnalyzer::new();
        let compute_stats = HashMap::new();
        let memory_stats = create_test_memory_stats();

        let bottlenecks = analyzer.analyze(&compute_stats, &memory_stats);

        let leaks: Vec<_> = bottlenecks
            .iter()
            .filter(|b| b.bottleneck_type == BottleneckType::MemoryLeak)
            .collect();

        assert!(!leaks.is_empty());
    }

    #[test]
    fn test_summary_generation() {
        let analyzer = BottleneckAnalyzer::new();
        let compute_stats = create_test_compute_stats();
        let memory_stats = create_test_memory_stats();

        let bottlenecks = analyzer.analyze(&compute_stats, &memory_stats);
        let summary = BottleneckAnalyzer::summary(&bottlenecks);

        assert!(summary.contains("bottleneck"));
        assert!(summary.contains("slow_op") || summary.contains("big_tensor"));
    }
}
