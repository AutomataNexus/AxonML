# axonml-profile Documentation

> Performance profiling and bottleneck analysis for AxonML.

## Overview

`axonml-profile` is a unified profiler combining memory, compute, and
timeline tracking, with automatic bottleneck analysis and multi-format
report export. A RAII `ProfileGuard` and the `profile_scope!` macro give
zero-overhead-when-disabled instrumentation. A global singleton
(`global_profiler()`) is available for cross-module use.

## Architecture

```
Profiler
+-- memory:  MemoryProfiler   (AllocationRecord, MemoryStats)
+-- compute: ComputeProfiler  (ProfiledOp, OperationStats)
+-- timeline: TimelineProfiler (Event, EventType)

ProfileReport  -> Text / JSON / Markdown / HTML
BottleneckAnalyzer -> Bottleneck (BottleneckType, severity)
```

## Modules

### `lib` — unified `Profiler`

```rust
pub struct Profiler {
    pub memory:   Arc<RwLock<MemoryProfiler>>,
    pub compute:  Arc<RwLock<ComputeProfiler>>,
    pub timeline: Arc<RwLock<TimelineProfiler>>,
    // + enabled flag (atomic)
}
```

Key methods:

- `new()`, `Default::default()`
- `set_enabled(bool)`, `is_enabled()`
- `start(name)`, `stop(name)` — wraps compute + timeline in one call
- `record_alloc(name, bytes)`, `record_free(name, bytes)`
- `peak_memory()`, `current_memory()`
- `total_time(name)`, `avg_time(name)`
- `reset()`, `summary() -> ProfileReport`, `print_summary()`
- `analyze_bottlenecks() -> Vec<Bottleneck>`

### `memory` — `MemoryProfiler`

Tracks allocations and deallocations per name. Exposes `record_alloc`,
`record_free`, `current_usage`, `peak_usage`, `reset`, `stats`, and
per-name `AllocationRecord`. `MemoryStats` summarizes current / peak /
per-name statistics.

### `compute` — `ComputeProfiler`

Per-op timing. `start(name)`, `stop(name)`, `total_time(name)`,
`avg_time(name)`, `all_stats() -> Vec<OperationStats>`, `reset`.
`ProfiledOp` captures a single timed op; `OperationStats` aggregates.

### `timeline` — `TimelineProfiler`

Event-based recording.

```rust
pub enum EventType { Start, End, Instant }
pub struct Event { name, event_type, timestamp, /* ... */ }
```

`record(name, event_type)`, `events() -> &[Event]`, `reset`. Events can be
exported via `ProfileReport` to timeline-compatible formats.

### `report` — `ProfileReport` + `ReportFormat`

```rust
pub enum ReportFormat { Text, Json, Markdown, Html }
pub struct ProfileReport { /* aggregated memory + compute + timeline */ }

impl ProfileReport {
    pub fn generate(profiler: &Profiler) -> Self;
    pub fn render(&self, format: ReportFormat) -> String;
}
```

### `bottleneck` — `BottleneckAnalyzer`

```rust
pub enum BottleneckType {
    MemoryBound,
    ComputeBound,
    IOBound,
    LaunchOverhead,
    Synchronization,
    // ...
}

pub struct Bottleneck {
    pub kind: BottleneckType,
    pub description: String,
    pub suggestion: String,
    // severity + affected ops
}

impl BottleneckAnalyzer {
    pub fn new() -> Self;
    pub fn analyze(&self, compute: &[OperationStats], memory: &MemoryStats) -> Vec<Bottleneck>;
}
```

### `error` — `ProfileError`, `ProfileResult<T>`

### `ProfileGuard` and `profile_scope!`

RAII scope-based profiling:

```rust
use axonml_profile::{Profiler, ProfileGuard, profile_scope};

let profiler = Profiler::new();

{
    let _guard = ProfileGuard::new(&profiler, "forward");
    // ... forward pass ...
} // auto stop on drop

profile_scope!(&profiler, "backward");
// ... backward pass ...
```

### Global profiler

```rust
use axonml_profile::{global_profiler, start, stop, record_alloc, record_free};

start("op");
// ... work ...
stop("op");

record_alloc("tensor_a", 1024);
record_free("tensor_a", 1024);

global_profiler().print_summary();
```

## Usage

### Basic profiling

```rust
use axonml_profile::{Profiler, ProfileGuard};

let profiler = Profiler::new();

{
    let _g = ProfileGuard::new(&profiler, "forward");
    // forward pass
}

{
    let _g = ProfileGuard::new(&profiler, "backward");
    // backward pass
}

profiler.print_summary();
println!("peak memory: {} bytes", profiler.peak_memory());
```

### Memory tracking

```rust
use axonml_profile::Profiler;

let profiler = Profiler::new();
profiler.record_alloc("tensor_a", 1024 * 1024);
profiler.record_alloc("tensor_b", 2 * 1024 * 1024);

assert_eq!(profiler.current_memory(), 3 * 1024 * 1024);
profiler.record_free("tensor_a", 1024 * 1024);
```

### Bottleneck analysis

```rust
use axonml_profile::Profiler;

let profiler = Profiler::new();
// ... run workload ...

for bottleneck in profiler.analyze_bottlenecks() {
    println!("[{:?}] {}", bottleneck.kind, bottleneck.description);
    println!("  -> {}", bottleneck.suggestion);
}
```

### Report export

```rust
use axonml_profile::{Profiler, ReportFormat};

let profiler = Profiler::new();
// ... run workload ...

let report = profiler.summary();
std::fs::write("profile.md", report.render(ReportFormat::Markdown)).unwrap();
std::fs::write("profile.html", report.render(ReportFormat::Html)).unwrap();
std::fs::write("profile.json", report.render(ReportFormat::Json)).unwrap();
```

### Integration with training

```rust
use axonml_profile::{Profiler, ProfileGuard};

let profiler = Profiler::new();

for _ in 0..num_epochs {
    profiler.reset();

    for batch in dataloader.iter() {
        { let _g = ProfileGuard::new(&profiler, "data_load"); /* ... */ }
        let out = { let _g = ProfileGuard::new(&profiler, "forward");  model.forward(&batch) };
        let loss = { let _g = ProfileGuard::new(&profiler, "loss");    criterion.forward(&out, &targets) };
        {            let _g = ProfileGuard::new(&profiler, "backward"); loss.backward() };
        {            let _g = ProfileGuard::new(&profiler, "optimizer"); optimizer.step() };
    }

    profiler.print_summary();
}
```

## Best Practices

1. Profile representative workloads (realistic batch sizes and data).
2. Warm up before profiling — first iterations hit cold JIT / GPU caches.
3. Profile coarse first, then drill down into hotspots.
4. Compare configurations side-by-side (batch size / AMP on/off / FSDP on/off).
5. Let `BottleneckAnalyzer` flag memory/compute/IO issues automatically.

## Last updated

0.6.5 (2026-06-06)
