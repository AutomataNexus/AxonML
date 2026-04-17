//! GPU Backend Test Runner — Detect and Validate GPU Backends
//!
//! Standalone binary that detects available GPU backends, prints hardware info
//! via `print_gpu_info`, configures tolerance and benchmark parameters through
//! `GpuTestConfig`, conditionally runs CUDA tests when the `cuda` feature is
//! enabled, and prints an overall pass/fail summary with `GpuTestReport`
//! aggregation. Exits with code 1 if any test fails, or prints a hint to
//! enable GPU features if no tests were run.
//!
//! # File
//! `crates/axonml-core/examples/gpu_test.rs`
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

use axonml_core::backends::gpu_tests::{
    GpuTestConfig, GpuTestReport, detect_gpu_backends, print_gpu_info,
};

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    println!("AxonML GPU Backend Test Suite");
    println!("==============================\n");

    // Print GPU detection info
    print_gpu_info();
    println!();

    // Create test configuration
    let _config = GpuTestConfig {
        atol: 1e-5,
        rtol: 1e-4,
        test_sizes: vec![1, 16, 64, 256, 1024, 4096],
        benchmark_sizes: vec![1024, 4096, 16384, 65536],
        warmup_iters: 5,
        bench_iters: 50,
    };

    let reports: Vec<GpuTestReport> = Vec::new();

    // -------------------------------------------------------------------------
    // CUDA Tests (feature-gated)
    // -------------------------------------------------------------------------

    // Run CUDA tests if available
    #[cfg(feature = "cuda")]
    {
        println!("\nRunning CUDA tests...");
        let report = axonml_core::backends::gpu_tests::cuda_tests::run_all_tests(&config);
        report.print_summary();
        reports.push(report);
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------

    println!("\n========================================");
    println!("Overall Summary");
    println!("========================================");

    let backends = detect_gpu_backends();
    println!("Detected backends: {}", backends.join(", "));

    let total_passed: usize = reports.iter().map(|r| r.passed_count()).sum();
    let total_failed: usize = reports.iter().map(|r| r.failed_count()).sum();

    println!("Total: {} passed, {} failed", total_passed, total_failed);

    if total_failed > 0 {
        println!("\nSome tests failed. Check the output above for details.");
        std::process::exit(1);
    } else if total_passed > 0 {
        println!("\nAll GPU tests passed!");
    } else {
        println!("\nNo GPU tests were run. Enable a GPU feature (--features cuda) to test.");
    }
}
