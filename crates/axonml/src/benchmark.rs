//! Model Benchmarking Utilities
//!
//! Provides utilities for benchmarking model inference and comparing performance.
//!
//! # Example
//! ```rust,ignore
//! use axonml::benchmark::{benchmark_model, warmup_model};
//! use axonml::nn::Linear;
//!
//! let model = Linear::new(784, 10);
//! let input = Tensor::randn(&[32, 784]);
//!
//! // Warmup
//! warmup_model(&model, &input, 5);
//!
//! // Benchmark
//! let result = benchmark_model(&model, &input, 100);
//! result.print_summary();
//! ```
//!
//! @version 0.1.0

use std::time::Instant;

#[cfg(all(feature = "core", feature = "nn"))]
use axonml_tensor::Tensor;

#[cfg(all(feature = "core", feature = "nn"))]
use axonml_autograd::Variable;

#[cfg(feature = "nn")]
use axonml_nn::Module;

use crate::hub::BenchmarkResult;

// =============================================================================
// Benchmarking Functions
// =============================================================================

/// Warm up a model by running a few forward passes.
///
/// This helps stabilize timing measurements by ensuring any lazy initialization
/// is complete and caches are populated.
#[cfg(all(feature = "core", feature = "nn"))]
pub fn warmup_model<M: Module>(model: &M, input: &Variable, iterations: usize) {
    for _ in 0..iterations {
        let _ = model.forward(input);
    }
}

/// Benchmark model inference.
///
/// Runs the model forward pass multiple times and collects timing statistics.
#[cfg(all(feature = "core", feature = "nn"))]
pub fn benchmark_model<M: Module>(
    model: &M,
    input: &Variable,
    iterations: usize,
) -> BenchmarkResult {
    let mut latencies = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.forward(input);
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1000.0);
    }

    // Estimate memory usage from input/output sizes
    let input_elements: usize = input.data().shape().iter().product();
    let peak_memory = (input_elements * 4 * 3) as u64; // Rough estimate: input + output + intermediate

    BenchmarkResult::new("model", &latencies, peak_memory)
}

/// Benchmark model with custom name.
#[cfg(all(feature = "core", feature = "nn"))]
pub fn benchmark_model_named<M: Module>(
    model: &M,
    input: &Variable,
    iterations: usize,
    name: &str,
) -> BenchmarkResult {
    let mut result = benchmark_model(model, input, iterations);
    result.model_name = name.to_string();
    result
}

/// Compare multiple models on the same input.
#[cfg(all(feature = "core", feature = "nn"))]
pub fn compare_models<M: Module>(
    models: &[(&str, &M)],
    input: &Variable,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for (name, model) in models {
        // Warmup
        warmup_model(*model, input, 5);

        // Benchmark
        let result = benchmark_model_named(*model, input, iterations, name);
        results.push(result);
    }

    results
}

// =============================================================================
// Throughput Testing
// =============================================================================

/// Configuration for throughput testing.
#[derive(Debug, Clone)]
pub struct ThroughputConfig {
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Number of iterations per batch size
    pub iterations: usize,
    /// Warmup iterations
    pub warmup: usize,
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            batch_sizes: vec![1, 4, 8, 16, 32, 64],
            iterations: 50,
            warmup: 5,
        }
    }
}

/// Throughput test result for a single batch size.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// Batch size
    pub batch_size: usize,
    /// Samples per second
    pub throughput: f64,
    /// Average latency in ms
    pub latency_ms: f64,
    /// Latency per sample in ms
    pub latency_per_sample_ms: f64,
}

/// Run throughput tests across different batch sizes.
#[cfg(all(feature = "core", feature = "nn"))]
pub fn throughput_test<M, F>(
    model: &M,
    input_fn: F,
    config: &ThroughputConfig,
) -> Vec<ThroughputResult>
where
    M: Module,
    F: Fn(usize) -> Variable,
{
    let mut results = Vec::new();

    for &batch_size in &config.batch_sizes {
        let input = input_fn(batch_size);

        // Warmup
        warmup_model(model, &input, config.warmup);

        // Benchmark
        let bench = benchmark_model(model, &input, config.iterations);

        results.push(ThroughputResult {
            batch_size,
            throughput: bench.throughput * batch_size as f64,
            latency_ms: bench.avg_latency_ms,
            latency_per_sample_ms: bench.avg_latency_ms / batch_size as f64,
        });
    }

    results
}

/// Print throughput results in a table format.
pub fn print_throughput_results(results: &[ThroughputResult]) {
    println!(
        "\n{:<12} {:>14} {:>14} {:>18}",
        "Batch Size", "Throughput", "Latency (ms)", "Per Sample (ms)"
    );
    println!("{}", "-".repeat(60));

    for result in results {
        println!(
            "{:<12} {:>12.1}/s {:>14.2} {:>18.3}",
            result.batch_size, result.throughput, result.latency_ms, result.latency_per_sample_ms
        );
    }

    // Find optimal batch size (highest throughput)
    if let Some(best) = results.iter().max_by(|a, b| {
        a.throughput
            .partial_cmp(&b.throughput)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "\nOptimal batch size: {} ({:.1} samples/sec)",
            best.batch_size, best.throughput
        );
    }
}

// =============================================================================
// Memory Profiling
// =============================================================================

/// Memory usage snapshot.
#[derive(Debug, Clone, Default)]
pub struct MemorySnapshot {
    /// Tensor allocations count
    pub tensor_count: usize,
    /// Total tensor memory in bytes
    pub tensor_bytes: u64,
    /// Parameter count
    pub param_count: usize,
    /// Parameter memory in bytes
    pub param_bytes: u64,
}

impl MemorySnapshot {
    /// Total memory in MB.
    pub fn total_mb(&self) -> f64 {
        (self.tensor_bytes + self.param_bytes) as f64 / 1_000_000.0
    }
}

/// Profile memory usage of a model.
#[cfg(feature = "nn")]
pub fn profile_model_memory<M: Module>(model: &M) -> MemorySnapshot {
    let params = model.parameters();
    let param_count = params.len();

    let param_bytes: u64 = params
        .iter()
        .map(|p| (p.numel() * 4) as u64) // 4 bytes per f32
        .sum();

    MemorySnapshot {
        tensor_count: 0,
        tensor_bytes: 0,
        param_count,
        param_bytes,
    }
}

/// Print memory profile.
pub fn print_memory_profile(snapshot: &MemorySnapshot, name: &str) {
    println!("\nMemory Profile: {}", name);
    println!(
        "  Parameters: {} ({:.2} MB)",
        snapshot.param_count,
        snapshot.param_bytes as f64 / 1_000_000.0
    );
    println!(
        "  Tensors: {} ({:.2} MB)",
        snapshot.tensor_count,
        snapshot.tensor_bytes as f64 / 1_000_000.0
    );
    println!("  Total: {:.2} MB", snapshot.total_mb());
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throughput_config_default() {
        let config = ThroughputConfig::default();
        assert!(!config.batch_sizes.is_empty());
        assert!(config.iterations > 0);
    }

    #[test]
    fn test_memory_snapshot_total() {
        let snapshot = MemorySnapshot {
            tensor_count: 10,
            tensor_bytes: 1_000_000,
            param_count: 5,
            param_bytes: 500_000,
        };
        assert!((snapshot.total_mb() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_throughput_result() {
        let result = ThroughputResult {
            batch_size: 32,
            throughput: 1000.0,
            latency_ms: 32.0,
            latency_per_sample_ms: 1.0,
        };
        assert_eq!(result.batch_size, 32);
        assert!((result.latency_per_sample_ms - 1.0).abs() < 0.01);
    }

    #[cfg(all(feature = "core", feature = "nn"))]
    #[test]
    fn test_benchmark_model() {
        use axonml_nn::Linear;

        let model = Linear::new(10, 5);
        let input = Variable::new(Tensor::randn(&[4, 10]), false);

        warmup_model(&model, &input, 2);
        let result = benchmark_model(&model, &input, 10);

        assert_eq!(result.iterations, 10);
        assert!(result.avg_latency_ms >= 0.0);
        assert!(result.throughput > 0.0);
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_profile_model_memory() {
        use axonml_nn::Linear;

        let model = Linear::new(100, 50);
        let snapshot = profile_model_memory(&model);

        assert!(snapshot.param_count > 0);
        assert!(snapshot.param_bytes > 0);
    }
}
