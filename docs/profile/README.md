# axonml-profile Documentation

> Performance profiling and bottleneck analysis for Axonml.

## Overview

`axonml-profile` provides comprehensive profiling tools for understanding and optimizing ML workloads. It includes memory profiling, compute profiling, timeline analysis, and automatic bottleneck detection.

## Modules

### profiler.rs

Core profiling infrastructure.

**Types:**

```rust
pub struct Profiler {
    // Central profiler collecting all metrics
}

pub struct ProfileGuard {
    // RAII guard for timed sections
}

pub struct ProfileReport {
    pub total_time: Duration,
    pub operations: Vec<OperationProfile>,
    pub memory_peak: usize,
    pub memory_timeline: Vec<MemorySnapshot>,
}

pub struct OperationProfile {
    pub name: String,
    pub duration: Duration,
    pub memory_used: usize,
    pub call_count: usize,
}
```

**Key methods:**

```rust
impl Profiler {
    pub fn new() -> Self;
    pub fn start(&mut self);
    pub fn stop(&mut self) -> ProfileReport;
    pub fn scope(&self, name: &str) -> ProfileGuard;
    pub fn record_op(&mut self, name: &str, duration: Duration);
}
```

### memory.rs

Memory profiling and tracking.

```rust
pub struct MemoryProfiler {
    // Tracks memory allocations and usage
}

pub struct MemorySnapshot {
    pub timestamp: Instant,
    pub allocated: usize,
    pub peak: usize,
    pub device: Device,
}

pub struct AllocationInfo {
    pub size: usize,
    pub device: Device,
    pub tensor_id: Option<usize>,
    pub timestamp: Instant,
}

impl MemoryProfiler {
    pub fn new() -> Self;
    pub fn track_allocation(&mut self, size: usize, device: Device);
    pub fn track_deallocation(&mut self, size: usize, device: Device);
    pub fn current_usage(&self) -> usize;
    pub fn peak_usage(&self) -> usize;
    pub fn snapshot(&self) -> MemorySnapshot;
    pub fn timeline(&self) -> Vec<MemorySnapshot>;
}
```

### compute.rs

Compute operation profiling.

```rust
pub struct ComputeProfiler {
    // Tracks compute operations
}

pub struct ComputeStats {
    pub operation: String,
    pub total_time: Duration,
    pub call_count: usize,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub flops: Option<u64>,
}

impl ComputeProfiler {
    pub fn new() -> Self;
    pub fn start_op(&mut self, name: &str) -> OpHandle;
    pub fn end_op(&mut self, handle: OpHandle);
    pub fn time_op<F, R>(&mut self, name: &str, f: F) -> R
        where F: FnOnce() -> R;
    pub fn stats(&self) -> Vec<ComputeStats>;
    pub fn hotspots(&self, top_n: usize) -> Vec<ComputeStats>;
}
```

### timeline.rs

Timeline-based profiling with events.

```rust
pub struct TimelineProfiler {
    // Records events on a timeline
}

pub struct TimelineEvent {
    pub name: String,
    pub category: EventCategory,
    pub start: Instant,
    pub duration: Duration,
    pub metadata: HashMap<String, String>,
}

pub enum EventCategory {
    Compute,
    Memory,
    DataLoad,
    Sync,
    Custom(String),
}

impl TimelineProfiler {
    pub fn new() -> Self;
    pub fn begin_event(&mut self, name: &str, category: EventCategory) -> EventId;
    pub fn end_event(&mut self, id: EventId);
    pub fn add_instant_event(&mut self, name: &str, category: EventCategory);
    pub fn events(&self) -> &[TimelineEvent];
    pub fn export_chrome_trace(&self) -> String;  // Chrome trace format
}
```

### bottleneck.rs

Automatic bottleneck detection and analysis.

```rust
pub struct BottleneckAnalyzer {
    // Analyzes profiles for bottlenecks
}

pub struct Bottleneck {
    pub kind: BottleneckKind,
    pub severity: Severity,
    pub description: String,
    pub suggestion: String,
    pub affected_ops: Vec<String>,
}

pub enum BottleneckKind {
    MemoryBound,
    ComputeBound,
    DataLoadBound,
    SyncBound,
    SmallBatches,
    FragmentedMemory,
    UnoptimizedOp,
}

pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl BottleneckAnalyzer {
    pub fn new() -> Self;
    pub fn analyze(&self, report: &ProfileReport) -> Vec<Bottleneck>;
    pub fn suggest_optimizations(&self, bottlenecks: &[Bottleneck]) -> Vec<String>;
}
```

## Usage

### Basic Profiling

```rust
use axonml_profile::{Profiler, ProfileGuard};

let mut profiler = Profiler::new();
profiler.start();

// Your ML code here
{
    let _guard = profiler.scope("forward_pass");
    // ... forward pass
}

{
    let _guard = profiler.scope("backward_pass");
    // ... backward pass
}

let report = profiler.stop();
println!("Total time: {:?}", report.total_time);
println!("Peak memory: {} bytes", report.memory_peak);
```

### Memory Profiling

```rust
use axonml_profile::MemoryProfiler;

let mut mem_profiler = MemoryProfiler::new();

// Track allocations
mem_profiler.track_allocation(1024 * 1024, Device::Cpu);  // 1MB

println!("Current: {} bytes", mem_profiler.current_usage());
println!("Peak: {} bytes", mem_profiler.peak_usage());

// Get timeline
for snapshot in mem_profiler.timeline() {
    println!("{:?}: {} bytes", snapshot.timestamp, snapshot.allocated);
}
```

### Compute Profiling

```rust
use axonml_profile::ComputeProfiler;

let mut compute_profiler = ComputeProfiler::new();

// Time operations
let result = compute_profiler.time_op("matmul", || {
    matrix_a.matmul(&matrix_b)
});

// Get hotspots
for stat in compute_profiler.hotspots(5) {
    println!("{}: {:?} ({} calls)",
        stat.operation, stat.avg_time, stat.call_count);
}
```

### Timeline Export

```rust
use axonml_profile::{TimelineProfiler, EventCategory};

let mut timeline = TimelineProfiler::new();

let id = timeline.begin_event("batch_process", EventCategory::Compute);
// ... process batch
timeline.end_event(id);

// Export for Chrome trace viewer
let trace_json = timeline.export_chrome_trace();
std::fs::write("trace.json", trace_json).unwrap();
// Open chrome://tracing and load trace.json
```

### Bottleneck Analysis

```rust
use axonml_profile::{Profiler, BottleneckAnalyzer};

let mut profiler = Profiler::new();
// ... run profiled code
let report = profiler.stop();

let analyzer = BottleneckAnalyzer::new();
let bottlenecks = analyzer.analyze(&report);

for bottleneck in &bottlenecks {
    println!("[{:?}] {}: {}",
        bottleneck.severity,
        bottleneck.kind,
        bottleneck.description);
    println!("  Suggestion: {}", bottleneck.suggestion);
}
```

## Integration with Training

```rust
use axonml_profile::Profiler;

let mut profiler = Profiler::new();

for epoch in 0..num_epochs {
    profiler.start();

    for batch in dataloader.iter() {
        let _data = profiler.scope("data_load");
        // ... load data

        let _forward = profiler.scope("forward");
        let output = model.forward(&batch);

        let _loss = profiler.scope("loss");
        let loss = criterion.forward(&output, &targets);

        let _backward = profiler.scope("backward");
        loss.backward();

        let _optim = profiler.scope("optimizer");
        optimizer.step();
    }

    let report = profiler.stop();
    println!("Epoch {} - Time: {:?}, Peak Memory: {}MB",
        epoch, report.total_time, report.memory_peak / 1024 / 1024);
}
```

## Best Practices

1. **Profile representative workloads**: Use realistic batch sizes and data
2. **Warm up before profiling**: Run a few iterations to stabilize JIT, caches
3. **Profile incrementally**: Start with coarse granularity, then drill down
4. **Compare configurations**: Profile different batch sizes, model sizes
5. **Use bottleneck analyzer**: Let it identify issues automatically

## Feature Flags

- Default: Basic profiling
- `detailed` - Enable detailed per-operation tracking
- `timeline` - Enable timeline export
