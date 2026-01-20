# axonml-profile

<p align="center">
  <!-- Logo placeholder -->
  <img src="../../assets/logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-profile"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version: 0.1.0"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-profile` provides comprehensive profiling capabilities for neural network training and inference. It includes memory tracking, compute profiling, timeline recording, and automatic bottleneck detection to help identify and resolve performance issues.

## Features

- **Memory Profiler**: Track allocations, deallocations, peak usage, and detect memory leaks
- **Compute Profiler**: Measure operation times, FLOPS, throughput, and bandwidth
- **Timeline Profiler**: Record timestamped events for visualization and analysis
- **Bottleneck Detection**: Automatically identify slow operations, memory hotspots, and throughput issues
- **Report Generation**: Export reports in Text, JSON, Markdown, and HTML formats
- **RAII Profiling**: Scope-based profiling with automatic start/stop via guards
- **Global Profiler**: Singleton profiler instance for convenient access throughout codebase
- **Thread-Safe**: All profilers are thread-safe with parking_lot locks

## Modules

| Module | Description |
|--------|-------------|
| `memory` | Memory profiler for tracking allocations, peak usage, and leak detection |
| `compute` | Compute profiler for measuring operation times, FLOPS, and bandwidth |
| `timeline` | Timeline profiler for recording events with timestamps and metadata |
| `bottleneck` | Bottleneck analyzer for detecting performance issues with severity ratings |
| `report` | Report generation in multiple formats (Text, JSON, Markdown, HTML) |
| `error` | Error types and Result alias for profiling operations |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-profile = "0.1.0"
```

### Basic Profiling

```rust
use axonml_profile::Profiler;

// Create a profiler
let profiler = Profiler::new();

// Profile an operation
profiler.start("forward_pass");
let output = model.forward(&input);
profiler.stop("forward_pass");

// Check timing
let total = profiler.total_time("forward_pass");
let avg = profiler.avg_time("forward_pass");
println!("Total: {:?}, Average: {:?}", total, avg);
```

### Memory Tracking

```rust
use axonml_profile::Profiler;

let profiler = Profiler::new();

// Track allocations
profiler.record_alloc("weights", 4 * 1024 * 1024);  // 4 MB
profiler.record_alloc("activations", 2 * 1024 * 1024);  // 2 MB

// Check memory usage
println!("Current: {} bytes", profiler.current_memory());
println!("Peak: {} bytes", profiler.peak_memory());

// Record deallocation
profiler.record_free("activations", 2 * 1024 * 1024);
```

### RAII Scope Profiling

```rust
use axonml_profile::{Profiler, ProfileGuard, profile_scope};

let profiler = Profiler::new();

// Using ProfileGuard directly
{
    let _guard = ProfileGuard::new(&profiler, "expensive_operation");
    // Operation automatically timed from guard creation to drop
    perform_computation();
}

// Or using the macro
{
    profile_scope!(&profiler, "another_operation");
    perform_another_computation();
}
```

### Bottleneck Analysis

```rust
use axonml_profile::Profiler;

let profiler = Profiler::new();

// Profile various operations
profiler.start("matmul");
// ... matrix multiplication
profiler.stop("matmul");

profiler.start("relu");
// ... activation
profiler.stop("relu");

// Analyze for bottlenecks
let bottlenecks = profiler.analyze_bottlenecks();
for b in &bottlenecks {
    println!("[{:?}] {}: {}", b.severity, b.name, b.description);
    println!("  Suggestion: {}", b.suggestion);
}
```

### Report Generation

```rust
use axonml_profile::{Profiler, ReportFormat};
use std::path::Path;

let profiler = Profiler::new();
// ... profiling operations ...

// Generate and print summary
let report = profiler.summary();
println!("{}", report);

// Export to file
report.export(Path::new("profile.html"), ReportFormat::Html)?;
report.export(Path::new("profile.json"), ReportFormat::Json)?;
report.export(Path::new("profile.md"), ReportFormat::Markdown)?;
```

### Global Profiler

```rust
use axonml_profile::{global_profiler, start, stop, record_alloc};

// Use global profiler functions
start("global_operation");
// ... operation ...
stop("global_operation");

record_alloc("global_tensor", 1024);

// Access the global profiler instance
let profiler = global_profiler();
profiler.print_summary();
```

### Compute Statistics

```rust
use axonml_profile::ComputeProfiler;

let mut profiler = ComputeProfiler::new();

// Profile with FLOPS tracking
profiler.start_with_flops("matmul", 2.0 * 1024.0 * 1024.0 * 1024.0);  // 2 GFLOPS
// ... operation ...
profiler.stop("matmul");

// Get top operations
for op in profiler.top_by_time(5) {
    println!("{}: {} calls, {:?} total", op.name, op.call_count, op.total_time());
}
```

## Bottleneck Types

| Type | Description | Detection Threshold |
|------|-------------|---------------------|
| `SlowOperation` | Operation taking disproportionate time | >20% of total time |
| `HighCallCount` | Operation called too frequently | >10,000 calls |
| `MemoryHotspot` | Large memory allocation | >30% of peak memory |
| `MemoryLeak` | Memory not freed | >5% of allocations |
| `LowThroughput` | Low computational throughput | <1 GFLOPS |

## Report Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| Text | Plain text with ASCII tables | Console output |
| JSON | Structured JSON | Programmatic analysis |
| Markdown | GitHub-flavored Markdown | Documentation |
| HTML | Styled HTML page | Browser viewing |

## Tests

Run the test suite:

```bash
cargo test -p axonml-profile
```

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
