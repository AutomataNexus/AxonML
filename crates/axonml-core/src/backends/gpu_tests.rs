//! GPU Backend Testing Infrastructure
//!
//! Comprehensive tests for all GPU backends (CUDA, Vulkan, Metal, WebGPU).
//! These tests verify correctness by comparing GPU results against CPU reference.
//!
//! # Running Tests
//!
//! ```bash
//! # Run all GPU tests (requires hardware)
//! cargo test -p axonml-core --features cuda gpu_tests
//!
//! # Run specific backend tests
//! cargo test -p axonml-core --features cuda test_cuda
//! cargo test -p axonml-core --features vulkan test_vulkan
//! cargo test -p axonml-core --features wgpu test_wgpu
//! ```
//!
//! @version 0.1.0

use crate::device::DeviceCapabilities;

// =============================================================================
// Test Configuration
// =============================================================================

/// Configuration for GPU tests.
#[derive(Debug, Clone)]
pub struct GpuTestConfig {
    /// Tolerance for floating point comparisons
    pub atol: f32,
    /// Relative tolerance
    pub rtol: f32,
    /// Test sizes for correctness tests
    pub test_sizes: Vec<usize>,
    /// Benchmark sizes
    pub benchmark_sizes: Vec<usize>,
    /// Number of warmup iterations for benchmarks
    pub warmup_iters: usize,
    /// Number of benchmark iterations
    pub bench_iters: usize,
}

impl Default for GpuTestConfig {
    fn default() -> Self {
        Self {
            atol: 1e-5,
            rtol: 1e-4,
            test_sizes: vec![1, 7, 16, 64, 256, 1024, 4096],
            benchmark_sizes: vec![1024, 4096, 16384, 65536, 262144, 1048576],
            warmup_iters: 5,
            bench_iters: 100,
        }
    }
}

// =============================================================================
// Test Results
// =============================================================================

/// Result of a GPU test.
#[derive(Debug, Clone)]
pub struct GpuTestResult {
    /// Test name
    pub name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Maximum absolute error (for correctness tests)
    pub max_abs_error: Option<f32>,
    /// Throughput in GB/s (for benchmarks)
    pub throughput_gbps: Option<f64>,
    /// Latency in microseconds
    pub latency_us: Option<f64>,
}

impl GpuTestResult {
    /// Create a passed result.
    pub fn pass(name: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            error: None,
            max_abs_error: None,
            throughput_gbps: None,
            latency_us: None,
        }
    }

    /// Create a failed result.
    pub fn fail(name: &str, error: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            error: Some(error.to_string()),
            max_abs_error: None,
            throughput_gbps: None,
            latency_us: None,
        }
    }

    /// Add correctness metrics.
    pub fn with_error(mut self, max_abs_error: f32) -> Self {
        self.max_abs_error = Some(max_abs_error);
        self
    }

    /// Add performance metrics.
    pub fn with_perf(mut self, throughput_gbps: f64, latency_us: f64) -> Self {
        self.throughput_gbps = Some(throughput_gbps);
        self.latency_us = Some(latency_us);
        self
    }
}

/// Collection of test results.
#[derive(Debug, Default)]
pub struct GpuTestReport {
    /// Backend name
    pub backend: String,
    /// Device capabilities
    pub capabilities: Option<DeviceCapabilities>,
    /// Individual test results
    pub results: Vec<GpuTestResult>,
}

impl GpuTestReport {
    /// Create a new report for a backend.
    pub fn new(backend: &str) -> Self {
        Self {
            backend: backend.to_string(),
            capabilities: None,
            results: Vec::new(),
        }
    }

    /// Set device capabilities.
    pub fn with_capabilities(mut self, caps: DeviceCapabilities) -> Self {
        self.capabilities = Some(caps);
        self
    }

    /// Add a test result.
    pub fn add_result(&mut self, result: GpuTestResult) {
        self.results.push(result);
    }

    /// Get number of passed tests.
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Get number of failed tests.
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Print a summary of the report.
    pub fn print_summary(&self) {
        println!("\n========================================");
        println!("GPU Test Report: {}", self.backend);
        println!("========================================");

        if let Some(caps) = &self.capabilities {
            println!("Device: {}", caps.name);
            println!(
                "Memory: {:.1} GB total, {:.1} GB available",
                caps.total_memory as f64 / 1e9,
                caps.available_memory as f64 / 1e9
            );
            if let Some(cc) = &caps.compute_capability {
                println!("Compute Capability: {}.{}", cc.0, cc.1);
            }
            println!();
        }

        println!(
            "Results: {} passed, {} failed",
            self.passed_count(),
            self.failed_count()
        );
        println!();

        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            print!("[{}] {}", status, result.name);

            if let Some(err) = &result.error {
                print!(" - {}", err);
            }
            if let Some(mae) = result.max_abs_error {
                print!(" (max_err: {:.2e})", mae);
            }
            if let Some(tp) = result.throughput_gbps {
                print!(" [{:.2} GB/s]", tp);
            }
            if let Some(lat) = result.latency_us {
                print!(" [{:.1} us]", lat);
            }
            println!();
        }

        if self.failed_count() > 0 {
            println!("\nFailed tests:");
            for result in self.results.iter().filter(|r| !r.passed) {
                println!("  - {}: {}", result.name, result.error.as_deref().unwrap_or("Unknown"));
            }
        }
    }
}

// =============================================================================
// Test Utilities
// =============================================================================

/// Compare two float slices for approximate equality.
pub fn assert_close(expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> Result<f32, String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "Length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        ));
    }

    let mut max_abs_error = 0.0f32;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let abs_err = (e - a).abs();
        let rel_tol = rtol * e.abs().max(a.abs());
        max_abs_error = max_abs_error.max(abs_err);

        if abs_err > atol + rel_tol {
            return Err(format!(
                "Mismatch at index {}: expected {}, got {} (abs_err: {:.2e}, tol: {:.2e})",
                i, e, a, abs_err, atol + rel_tol
            ));
        }
    }

    Ok(max_abs_error)
}

/// Generate random test data.
pub fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducibility
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            // Map to [-1, 1]
            ((state >> 16) & 0x7FFF) as f32 / 16384.0 - 1.0
        })
        .collect()
}

/// CPU reference implementation for element-wise addition.
pub fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// CPU reference implementation for element-wise multiplication.
pub fn cpu_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// CPU reference implementation for scalar multiplication.
pub fn cpu_scale(a: &[f32], alpha: f32) -> Vec<f32> {
    a.iter().map(|x| x * alpha).collect()
}

/// CPU reference implementation for ReLU.
pub fn cpu_relu(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.max(0.0)).collect()
}

/// CPU reference implementation for sigmoid.
pub fn cpu_sigmoid(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
}

/// CPU reference implementation for tanh.
pub fn cpu_tanh(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.tanh()).collect()
}

/// CPU reference implementation for matrix multiplication.
/// A is m x k, B is k x n, C is m x n (row-major).
pub fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// =============================================================================
// CUDA Tests
// =============================================================================

#[cfg(feature = "cuda")]
pub mod cuda_tests {
    use super::*;
    use crate::backends::cuda::{is_available, device_count, CudaBackend};

    /// Run all CUDA tests.
    pub fn run_all_tests(config: &GpuTestConfig) -> GpuTestReport {
        let mut report = GpuTestReport::new("CUDA");

        if !is_available() {
            report.add_result(GpuTestResult::fail(
                "cuda_availability",
                "CUDA not available on this system",
            ));
            return report;
        }

        let backend = match CudaBackend::new(0) {
            Some(b) => b,
            None => {
                report.add_result(GpuTestResult::fail(
                    "backend_creation",
                    "Failed to create CUDA backend",
                ));
                return report;
            }
        };

        report = report.with_capabilities(backend.capabilities());

        // Memory operations
        report.add_result(test_memory_roundtrip(&backend, config));

        // Element-wise operations
        for &size in &config.test_sizes {
            report.add_result(test_add(&backend, size, config));
            report.add_result(test_mul(&backend, size, config));
            report.add_result(test_scale(&backend, size, config));
        }

        // Activation functions
        for &size in &config.test_sizes {
            report.add_result(test_relu(&backend, size, config));
            report.add_result(test_sigmoid(&backend, size, config));
            report.add_result(test_tanh(&backend, size, config));
        }

        // Matrix multiplication
        report.add_result(test_gemm_square(&backend, 64, config));
        report.add_result(test_gemm_square(&backend, 256, config));
        report.add_result(test_gemm_rectangular(&backend, 128, 64, 96, config));

        report
    }

    fn test_memory_roundtrip(backend: &CudaBackend, _config: &GpuTestConfig) -> GpuTestResult {
        let name = "memory_roundtrip";
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();

        match backend.htod_copy(&data) {
            Ok(gpu_data) => match backend.dtoh_copy(&gpu_data) {
                Ok(result) => {
                    if result == data {
                        GpuTestResult::pass(name)
                    } else {
                        GpuTestResult::fail(name, "Data mismatch after roundtrip")
                    }
                }
                Err(e) => GpuTestResult::fail(name, &format!("dtoh_copy failed: {}", e)),
            },
            Err(e) => GpuTestResult::fail(name, &format!("htod_copy failed: {}", e)),
        }
    }

    fn test_add(backend: &CudaBackend, size: usize, config: &GpuTestConfig) -> GpuTestResult {
        let name = format!("add_f32_{}", size);
        let a = random_vec(size, 42);
        let b = random_vec(size, 123);
        let expected = cpu_add(&a, &b);

        let gpu_a = match backend.htod_copy(&a) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy(a): {}", e)),
        };
        let gpu_b = match backend.htod_copy(&b) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy(b): {}", e)),
        };
        let mut gpu_c = match backend.alloc::<f32>(size) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("alloc: {}", e)),
        };

        if let Err(e) = backend.add_f32(&mut gpu_c, &gpu_a, &gpu_b, size) {
            return GpuTestResult::fail(&name, &format!("add_f32: {}", e));
        }

        backend.synchronize();

        match backend.dtoh_copy(&gpu_c) {
            Ok(result) => match assert_close(&expected, &result, config.atol, config.rtol) {
                Ok(max_err) => GpuTestResult::pass(&name).with_error(max_err),
                Err(e) => GpuTestResult::fail(&name, &e),
            },
            Err(e) => GpuTestResult::fail(&name, &format!("dtoh_copy: {}", e)),
        }
    }

    fn test_mul(backend: &CudaBackend, size: usize, config: &GpuTestConfig) -> GpuTestResult {
        let name = format!("mul_f32_{}", size);
        let a = random_vec(size, 42);
        let b = random_vec(size, 123);
        let expected = cpu_mul(&a, &b);

        let gpu_a = match backend.htod_copy(&a) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy(a): {}", e)),
        };
        let gpu_b = match backend.htod_copy(&b) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy(b): {}", e)),
        };
        let mut gpu_c = match backend.alloc::<f32>(size) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("alloc: {}", e)),
        };

        if let Err(e) = backend.mul_f32(&mut gpu_c, &gpu_a, &gpu_b, size) {
            return GpuTestResult::fail(&name, &format!("mul_f32: {}", e));
        }

        backend.synchronize();

        match backend.dtoh_copy(&gpu_c) {
            Ok(result) => match assert_close(&expected, &result, config.atol, config.rtol) {
                Ok(max_err) => GpuTestResult::pass(&name).with_error(max_err),
                Err(e) => GpuTestResult::fail(&name, &e),
            },
            Err(e) => GpuTestResult::fail(&name, &format!("dtoh_copy: {}", e)),
        }
    }

    fn test_scale(backend: &CudaBackend, size: usize, config: &GpuTestConfig) -> GpuTestResult {
        let name = format!("scale_f32_{}", size);
        let a = random_vec(size, 42);
        let alpha = 2.5f32;
        let expected = cpu_scale(&a, alpha);

        let mut gpu_a = match backend.htod_copy(&a) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy: {}", e)),
        };

        if let Err(e) = backend.scale_f32(&mut gpu_a, alpha, size) {
            return GpuTestResult::fail(&name, &format!("scale_f32: {}", e));
        }

        backend.synchronize();

        match backend.dtoh_copy(&gpu_a) {
            Ok(result) => match assert_close(&expected, &result, config.atol, config.rtol) {
                Ok(max_err) => GpuTestResult::pass(&name).with_error(max_err),
                Err(e) => GpuTestResult::fail(&name, &e),
            },
            Err(e) => GpuTestResult::fail(&name, &format!("dtoh_copy: {}", e)),
        }
    }

    fn test_relu(backend: &CudaBackend, size: usize, config: &GpuTestConfig) -> GpuTestResult {
        let name = format!("relu_f32_{}", size);
        let a = random_vec(size, 42);
        let expected = cpu_relu(&a);

        let gpu_a = match backend.htod_copy(&a) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy: {}", e)),
        };
        let mut gpu_b = match backend.alloc::<f32>(size) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("alloc: {}", e)),
        };

        if let Err(e) = backend.relu_f32(&mut gpu_b, &gpu_a, size) {
            return GpuTestResult::fail(&name, &format!("relu_f32: {}", e));
        }

        backend.synchronize();

        match backend.dtoh_copy(&gpu_b) {
            Ok(result) => match assert_close(&expected, &result, config.atol, config.rtol) {
                Ok(max_err) => GpuTestResult::pass(&name).with_error(max_err),
                Err(e) => GpuTestResult::fail(&name, &e),
            },
            Err(e) => GpuTestResult::fail(&name, &format!("dtoh_copy: {}", e)),
        }
    }

    fn test_sigmoid(backend: &CudaBackend, size: usize, config: &GpuTestConfig) -> GpuTestResult {
        let name = format!("sigmoid_f32_{}", size);
        let a = random_vec(size, 42);
        let expected = cpu_sigmoid(&a);

        let gpu_a = match backend.htod_copy(&a) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy: {}", e)),
        };
        let mut gpu_b = match backend.alloc::<f32>(size) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("alloc: {}", e)),
        };

        if let Err(e) = backend.sigmoid_f32(&mut gpu_b, &gpu_a, size) {
            return GpuTestResult::fail(&name, &format!("sigmoid_f32: {}", e));
        }

        backend.synchronize();

        // Sigmoid uses fast approximations, so allow higher tolerance
        let sigmoid_atol = 1e-3;
        let sigmoid_rtol = 1e-2;

        match backend.dtoh_copy(&gpu_b) {
            Ok(result) => match assert_close(&expected, &result, sigmoid_atol, sigmoid_rtol) {
                Ok(max_err) => GpuTestResult::pass(&name).with_error(max_err),
                Err(e) => GpuTestResult::fail(&name, &e),
            },
            Err(e) => GpuTestResult::fail(&name, &format!("dtoh_copy: {}", e)),
        }
    }

    fn test_tanh(backend: &CudaBackend, size: usize, config: &GpuTestConfig) -> GpuTestResult {
        let name = format!("tanh_f32_{}", size);
        let a = random_vec(size, 42);
        let expected = cpu_tanh(&a);

        let gpu_a = match backend.htod_copy(&a) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy: {}", e)),
        };
        let mut gpu_b = match backend.alloc::<f32>(size) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("alloc: {}", e)),
        };

        if let Err(e) = backend.tanh_f32(&mut gpu_b, &gpu_a, size) {
            return GpuTestResult::fail(&name, &format!("tanh_f32: {}", e));
        }

        backend.synchronize();

        // Tanh uses fast approximations
        let tanh_atol = 1e-3;
        let tanh_rtol = 1e-2;

        match backend.dtoh_copy(&gpu_b) {
            Ok(result) => match assert_close(&expected, &result, tanh_atol, tanh_rtol) {
                Ok(max_err) => GpuTestResult::pass(&name).with_error(max_err),
                Err(e) => GpuTestResult::fail(&name, &e),
            },
            Err(e) => GpuTestResult::fail(&name, &format!("dtoh_copy: {}", e)),
        }
    }

    fn test_gemm_square(
        backend: &CudaBackend,
        n: usize,
        config: &GpuTestConfig,
    ) -> GpuTestResult {
        test_gemm_rectangular(backend, n, n, n, config)
    }

    fn test_gemm_rectangular(
        backend: &CudaBackend,
        m: usize,
        n: usize,
        k: usize,
        config: &GpuTestConfig,
    ) -> GpuTestResult {
        let name = format!("gemm_f32_{}x{}x{}", m, n, k);

        // Generate test data
        let a = random_vec(m * k, 42);
        let b = random_vec(k * n, 123);
        let expected = cpu_gemm(&a, &b, m, n, k);

        // Convert to column-major for cuBLAS
        let a_col = row_to_col_major(&a, m, k);
        let b_col = row_to_col_major(&b, k, n);

        let gpu_a = match backend.htod_copy(&a_col) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy(a): {}", e)),
        };
        let gpu_b = match backend.htod_copy(&b_col) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("htod_copy(b): {}", e)),
        };
        let mut gpu_c = match backend.alloc::<f32>(m * n) {
            Ok(d) => d,
            Err(e) => return GpuTestResult::fail(&name, &format!("alloc: {}", e)),
        };

        // cuBLAS GEMM: C = alpha * A @ B + beta * C
        if let Err(e) = backend.gemm_f32(
            false, false, // no transpose
            m, n, k,
            1.0,          // alpha
            &gpu_a, m,    // A, lda
            &gpu_b, k,    // B, ldb
            0.0,          // beta
            &mut gpu_c, m, // C, ldc
        ) {
            return GpuTestResult::fail(&name, &format!("gemm_f32: {}", e));
        }

        backend.synchronize();

        match backend.dtoh_copy(&gpu_c) {
            Ok(result_col) => {
                // Convert back from column-major
                let result = col_to_row_major(&result_col, m, n);

                // GEMM can have larger numerical errors
                let gemm_atol = 1e-3;
                let gemm_rtol = 1e-2;

                match assert_close(&expected, &result, gemm_atol, gemm_rtol) {
                    Ok(max_err) => GpuTestResult::pass(&name).with_error(max_err),
                    Err(e) => GpuTestResult::fail(&name, &e),
                }
            }
            Err(e) => GpuTestResult::fail(&name, &format!("dtoh_copy: {}", e)),
        }
    }

    // Helper: row-major to column-major conversion
    fn row_to_col_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = data[i * cols + j];
            }
        }
        result
    }

    // Helper: column-major to row-major conversion
    fn col_to_row_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                result[i * cols + j] = data[j * rows + i];
            }
        }
        result
    }
}

// =============================================================================
// Hardware Detection
// =============================================================================

/// Detect available GPU backends.
pub fn detect_gpu_backends() -> Vec<String> {
    let mut backends = Vec::new();

    #[cfg(feature = "cuda")]
    {
        if crate::backends::cuda::is_available() {
            backends.push(format!(
                "CUDA ({} device(s))",
                crate::backends::cuda::device_count()
            ));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        backends.push("Vulkan".to_string());
    }

    #[cfg(feature = "metal")]
    {
        #[cfg(target_os = "macos")]
        backends.push("Metal".to_string());
    }

    #[cfg(feature = "wgpu")]
    {
        backends.push("WebGPU".to_string());
    }

    if backends.is_empty() {
        backends.push("None (CPU only)".to_string());
    }

    backends
}

/// Print GPU detection information.
pub fn print_gpu_info() {
    println!("GPU Backend Detection");
    println!("=====================");

    let backends = detect_gpu_backends();
    for backend in &backends {
        println!("  - {}", backend);
    }

    #[cfg(feature = "cuda")]
    {
        if crate::backends::cuda::is_available() {
            println!("\nCUDA Devices:");
            for i in 0..crate::backends::cuda::device_count() {
                let caps = crate::backends::cuda::get_capabilities(i);
                println!("  [{}] {}", i, caps.name);
                println!(
                    "      Memory: {:.1} GB",
                    caps.total_memory as f64 / 1e9
                );
                if let Some(cc) = caps.compute_capability {
                    println!("      Compute: {}.{}", cc.0, cc.1);
                }
            }
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
    fn test_random_vec_reproducibility() {
        let a = random_vec(100, 42);
        let b = random_vec(100, 42);
        assert_eq!(a, b, "Same seed should produce same output");
    }

    #[test]
    fn test_cpu_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = cpu_add(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cpu_mul() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = cpu_mul(&a, &b);
        assert_eq!(c, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_cpu_relu() {
        let a = vec![-1.0, 0.0, 1.0, 2.0];
        let b = cpu_relu(&a);
        assert_eq!(b, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_cpu_gemm() {
        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = cpu_gemm(&a, &b, 2, 2, 3);
        // Expected:
        // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_assert_close_pass() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.00001, 2.00001, 3.00001];
        assert!(assert_close(&a, &b, 1e-4, 1e-4).is_ok());
    }

    #[test]
    fn test_assert_close_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.0, 3.0];
        assert!(assert_close(&a, &b, 1e-4, 1e-4).is_err());
    }

    #[test]
    fn test_detect_backends() {
        let backends = detect_gpu_backends();
        assert!(!backends.is_empty());
    }

    #[test]
    fn test_gpu_test_result() {
        let pass = GpuTestResult::pass("test").with_error(0.0001);
        assert!(pass.passed);
        assert!(pass.max_abs_error.is_some());

        let fail = GpuTestResult::fail("test", "error");
        assert!(!fail.passed);
        assert_eq!(fail.error, Some("error".to_string()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_all() {
        let config = GpuTestConfig::default();
        let report = cuda_tests::run_all_tests(&config);
        report.print_summary();

        // If CUDA is available, all tests should pass
        if crate::backends::cuda::is_available() {
            assert_eq!(report.failed_count(), 0, "Some CUDA tests failed");
        }
    }
}
