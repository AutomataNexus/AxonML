# axonml-profile

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-profile"><img src="https://img.shields.io/badge/crates.io-0.6.5-green.svg" alt="Version: 0.6.5"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-profile` provides performance profiling for AxonML training and
inference. It unifies memory, compute, and timeline profiling under one
`Profiler`, exposes a global singleton, and includes a `BottleneckAnalyzer`
that inspects collected stats to surface hot ops, high call counts, memory
hotspots, suspected leaks, and low-throughput operations. Reports can be
emitted as plain text, JSON, Markdown, or HTML.

## Features

- **Memory Profiler** — tracks allocations / deallocations, peak and current usage, per-tensor records, leak detection (`leaks()`), `AllocationRecord`, `MemoryStats`, `format_bytes` helper
- **Compute Profiler** — per-op timing (`start`, `stop`, `start_with_flops`, `start_with_bytes`), `OperationStats` with avg/total/min/max time, GFLOPS, GFLOPS/sec, and bandwidth in GB/s; top-k by time / calls; `TimingGuard` RAII
- **Timeline Profiler** — event-based recording (`EventType::{Start, End, Instant, Custom(String)}`), optional metadata, capacity-bounded ring, `duration(name)`, `events_by_name`, `events_by_type`, Chrome-trace export (`to_chrome_trace`) and JSON export
- **Unified Profiler** — `Profiler` with memory + compute + timeline under one facade, enable/disable at runtime, `analyze_bottlenecks`, `summary`, `print_summary`, `reset`
- **Bottleneck Analysis** — `BottleneckAnalyzer` with `AnalyzerConfig` thresholds, `Severity::{Low, Medium, High, Critical}`, sorted-by-severity output, per-bottleneck description + suggestion + numeric metrics
- **Report Generation** — `ProfileReport::export(path, format)` to `Text`, `Json`, `Markdown`, or `Html`
- **RAII Profiling** — `ProfileGuard` and `profile_scope!` macro
- **Global Profiler** — `global_profiler()` singleton plus `start`, `stop`, `record_alloc`, `record_free` shortcuts
- **Thread-Safe** — `parking_lot::RwLock` on each sub-profiler, `AtomicBool` enable flag

## Modules

| Module | Description |
|--------|-------------|
| `memory` | `MemoryProfiler`, `AllocationRecord`, `MemoryStats`, `format_bytes` |
| `compute` | `ComputeProfiler`, `OperationStats`, `ProfiledOp`, `TimingGuard`, `format_duration` |
| `timeline` | `TimelineProfiler`, `Event`, `EventType`, Chrome-trace / JSON export |
| `bottleneck` | `BottleneckAnalyzer`, `AnalyzerConfig`, `Bottleneck`, `BottleneckType`, `Severity` |
| `report` | `ProfileReport`, `ReportFormat`, `MemorySummary`, `ComputeSummary`, `OperationSummary` |
| `error` | `ProfileError`, `ProfileResult` |

## Feature Flags

| Flag | Effect |
|------|--------|
| `chrome-trace` | Reserved for Chrome-trace output helpers (the `to_chrome_trace` method is always available on `TimelineProfiler`) |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-profile = "0.6.5"
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
let avg   = profiler.avg_time("forward_pass");
println!("Total: {:?}, Average: {:?}", total, avg);
```

### Memory Tracking

```rust
use axonml_profile::Profiler;

let profiler = Profiler::new();

// Track allocations
profiler.record_alloc("weights",     4 * 1024 * 1024);  // 4 MB
profiler.record_alloc("activations", 2 * 1024 * 1024);  // 2 MB

// Check memory usage
println!("Current: {} bytes", profiler.current_memory());
println!("Peak:    {} bytes", profiler.peak_memory());

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

profiler.start("matmul"); /* ... */ profiler.stop("matmul");
profiler.start("relu");   /* ... */ profiler.stop("relu");

let bottlenecks = profiler.analyze_bottlenecks();
for b in &bottlenecks {
    println!("[{:?}] {} ({:?}): {}", b.severity, b.name, b.bottleneck_type, b.description);
    println!("  Suggestion: {}", b.suggestion);
}
```

### Custom Bottleneck Thresholds

```rust
use axonml_profile::{BottleneckAnalyzer, bottleneck::AnalyzerConfig};

let config = AnalyzerConfig {
    slow_op_threshold_pct: 15.0,
    high_call_threshold:   5_000,
    memory_hotspot_threshold_pct: 25.0,
    min_gflops_threshold:  2.0,
    check_memory_leaks:    true,
};
let analyzer = BottleneckAnalyzer::with_config(config);
```

### Report Generation

```rust
use axonml_profile::{Profiler, ReportFormat};
use std::path::Path;

let profiler = Profiler::new();
// ... profiling operations ...

let report = profiler.summary();
println!("{}", report);

// Export to file
report.export(Path::new("profile.html"), ReportFormat::Html)?;
report.export(Path::new("profile.json"), ReportFormat::Json)?;
report.export(Path::new("profile.md"),   ReportFormat::Markdown)?;
report.export(Path::new("profile.txt"),  ReportFormat::Text)?;
```

### Timeline / Chrome Trace

```rust
use axonml_profile::{Profiler, EventType};

let profiler = Profiler::new();
profiler.start("step");
profiler.stop("step");

// Access timeline directly
let tl = profiler.timeline.read();
let trace_json = tl.to_chrome_trace(); // drop into chrome://tracing
```

### Global Profiler

```rust
use axonml_profile::{global_profiler, start, stop, record_alloc};

start("global_operation");
// ... operation ...
stop("global_operation");

record_alloc("global_tensor", 1024);

let profiler = global_profiler();
profiler.print_summary();
```

### Compute Statistics With FLOPs

```rust
use axonml_profile::ComputeProfiler;

let mut profiler = ComputeProfiler::new();

// Profile with FLOPS tracking (work = 2 GFLOPs)
profiler.start_with_flops("matmul", 2.0 * 1024.0 * 1024.0 * 1024.0);
// ... operation ...
profiler.stop("matmul");

for op in profiler.top_by_time(5) {
    println!(
        "{}: {} calls, total {:?}, throughput {:?} GFLOPS/sec",
        op.name, op.call_count, op.total_time(), op.gflops_per_sec()
    );
}

// Or track bandwidth
let mut mem = ComputeProfiler::new();
mem.start_with_bytes("copy", 16 * 1024 * 1024);
mem.stop("copy");
if let Some(stats) = mem.get_stats("copy") {
    println!("copy: {:?} GB/s", stats.bandwidth_gbps());
}
```

## Bottleneck Types

Variants of `BottleneckType`:

| Type | Description |
|------|-------------|
| `SlowOperation` | Operation taking disproportionate share of total time |
| `HighCallCount` | Operation called a very large number of times |
| `MemoryHotspot` | Single allocation / tensor holding a large share of peak memory |
| `MemoryLeak` | Allocations that were never paired with a free |
| `LowThroughput` | Measured GFLOPS below the configured threshold |
| `MemoryBound` | Op dominated by memory bandwidth rather than compute |
| `LoadImbalance` | Uneven workload distribution |

Default thresholds (from `AnalyzerConfig::default`): 20% of total time for
slow ops, 10,000 calls for high-call-count, 30% of peak memory for
hotspots, 1 GFLOPS minimum throughput, memory-leak detection on.

## Severity Levels

`Severity::{Low, Medium, High, Critical}` — bottlenecks are sorted highest
first by the analyzer.

## Report Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `Text` | Plain text with ASCII tables | Console output |
| `Json` | Structured JSON (serde) | Programmatic analysis |
| `Markdown` | GitHub-flavored Markdown | Documentation |
| `Html` | Styled HTML page | Browser viewing |

## Tests

```bash
cargo test -p axonml-profile
```

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
