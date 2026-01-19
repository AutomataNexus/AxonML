# axonml-profile

[![Crates.io](https://img.shields.io/crates/v/axonml-profile.svg)](https://crates.io/crates/axonml-profile)
[![Docs.rs](https://docs.rs/axonml-profile/badge.svg)](https://docs.rs/axonml-profile)
[![Downloads](https://img.shields.io/crates/d/axonml-profile.svg)](https://crates.io/crates/axonml-profile)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Performance profiling for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-profile` provides comprehensive profiling tools for analyzing and optimizing model performance. Track execution time, memory usage, GPU utilization, and identify bottlenecks in your training and inference pipelines.

## Features

### Timing
- **Operation profiling** - Time individual ops
- **Layer profiling** - Time each layer/module
- **End-to-end timing** - Full forward/backward pass
- **Statistical analysis** - Min, max, mean, percentiles

### Memory
- **Allocation tracking** - Track tensor allocations
- **Peak memory** - Find maximum memory usage
- **Memory timeline** - Visualize memory over time
- **Leak detection** - Find memory leaks

### GPU Profiling
- **CUDA events** - Accurate GPU timing
- **Kernel analysis** - Identify slow kernels
- **Memory transfers** - Track host-device transfers
- **Utilization** - GPU usage statistics

### Visualization
- **Chrome trace format** - View in chrome://tracing
- **Flamegraphs** - Stack-based visualization
- **TensorBoard** - Integration with TB profiler
- **HTML reports** - Self-contained reports

## Installation

```toml
[dependencies]
axonml-profile = "0.1"
```

## Usage

### Basic Profiling

```rust
use axonml_profile::{profile, Profiler};

let profiler = Profiler::new();

// Profile a code block
let output = profile!("forward_pass", {
    model.forward(&input)
});

// Print timing summary
profiler.summary();
```

### Layer-by-Layer Profiling

```rust
use axonml_profile::ProfiledModule;

// Wrap model for automatic layer profiling
let profiled_model = ProfiledModule::new(model);

// Run inference
let output = profiled_model.forward(&input);

// Get timing breakdown
for (name, timing) in profiled_model.layer_timings() {
    println!("{}: {:.2}ms", name, timing.mean_ms());
}
```

### Memory Profiling

```rust
use axonml_profile::{MemoryProfiler, memory_stats};

let profiler = MemoryProfiler::new();

profiler.start();
let output = model.forward(&input);
profiler.stop();

// Get memory statistics
let stats = profiler.stats();
println!("Peak memory: {} MB", stats.peak_mb());
println!("Allocations: {}", stats.num_allocations());
println!("Total allocated: {} MB", stats.total_allocated_mb());
```

### Training Loop Profiling

```rust
use axonml_profile::{Profiler, ProfileScope};

let profiler = Profiler::new()
    .record_shapes(true)
    .with_stack_traces(true);

for epoch in 0..num_epochs {
    for batch in dataloader.iter() {
        profiler.step();  // Mark iteration boundary

        let _forward = profiler.scope("forward");
        let output = model.forward(&batch.data);
        drop(_forward);

        let _loss = profiler.scope("loss");
        let loss = compute_loss(&output, &batch.targets);
        drop(_loss);

        let _backward = profiler.scope("backward");
        optimizer.zero_grad();
        loss.backward();
        drop(_backward);

        let _step = profiler.scope("optimizer_step");
        optimizer.step();
        drop(_step);
    }
}

// Export to Chrome trace format
profiler.export_chrome_trace("trace.json")?;
```

### GPU Profiling

```rust
use axonml_profile::{CudaProfiler, cuda_sync};

let profiler = CudaProfiler::new();

profiler.start();

// Run GPU operations
let output = model.forward(&input.to_device(Device::CUDA(0)));

// Ensure GPU operations complete
cuda_sync();

profiler.stop();

// Analyze GPU performance
println!("GPU time: {:.2}ms", profiler.gpu_time_ms());
println!("Memory transfers: {} MB", profiler.transfer_mb());
println!("GPU utilization: {:.1}%", profiler.utilization());
```

### Generate HTML Report

```rust
use axonml_profile::{Profiler, ReportFormat};

let profiler = Profiler::new();
// ... run profiling ...

// Generate comprehensive HTML report
profiler.export_report("profile_report.html", ReportFormat::Html)?;

// Or flamegraph
profiler.export_report("profile.svg", ReportFormat::Flamegraph)?;
```

## API Reference

### Profiler Methods

| Method | Description |
|--------|-------------|
| `new()` | Create new profiler |
| `start()` | Start profiling |
| `stop()` | Stop profiling |
| `scope(name)` | Create named scope |
| `step()` | Mark iteration boundary |
| `summary()` | Print timing summary |
| `export_chrome_trace(path)` | Export to Chrome format |

### MemoryProfiler Methods

| Method | Description |
|--------|-------------|
| `new()` | Create memory profiler |
| `start()` | Start tracking |
| `stop()` | Stop tracking |
| `stats()` | Get memory statistics |
| `timeline()` | Get memory timeline |

### MemoryStats

| Method | Description |
|--------|-------------|
| `peak_mb()` | Peak memory usage |
| `current_mb()` | Current usage |
| `num_allocations()` | Number of allocations |
| `total_allocated_mb()` | Total allocated |

## CLI Usage

```bash
# Profile model inference
axonml profile run model.axonml --input sample.tensor

# Profile with memory tracking
axonml profile run model.axonml --memory --input sample.tensor

# Generate HTML report
axonml profile run model.axonml --report profile.html

# Export Chrome trace
axonml profile run model.axonml --chrome-trace trace.json
```

## Part of Axonml

```toml
[dependencies]
axonml = { version = "0.1", features = ["profile"] }
```

## License

MIT OR Apache-2.0
