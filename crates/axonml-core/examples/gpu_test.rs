//! GPU Backend Test Runner
//!
//! # File
//! `crates/axonml-core/examples/gpu_test.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_core::backends::gpu_tests::{
    detect_gpu_backends, print_gpu_info, GpuTestConfig, GpuTestReport,
};

fn main() {
    println!("AxonML GPU Backend Test Suite");
    println!("==============================\n");

    // Print GPU detection info
    print_gpu_info();
    println!();

    // Create test configuration
    let config = GpuTestConfig {
        atol: 1e-5,
        rtol: 1e-4,
        test_sizes: vec![1, 16, 64, 256, 1024, 4096],
        benchmark_sizes: vec![1024, 4096, 16384, 65536],
        warmup_iters: 5,
        bench_iters: 50,
    };

    let mut reports: Vec<GpuTestReport> = Vec::new();

    // Run CUDA tests if available
    #[cfg(feature = "cuda")]
    {
        println!("\nRunning CUDA tests...");
        let report = axonml_core::backends::gpu_tests::cuda_tests::run_all_tests(&config);
        report.print_summary();
        reports.push(report);
    }

    // Summary
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
