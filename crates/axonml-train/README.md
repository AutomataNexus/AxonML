# axonml-train

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.85%2B-orange.svg" alt="Rust: 1.85+"></a>
  <img src="https://img.shields.io/badge/version-0.6.5-green.svg" alt="Version 0.6.5"/>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part%20of-AxonML-teal.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

**axonml-train** is the high-level training infrastructure sub-crate for the AxonML framework. It provides reusable training utilities, a unified model hub / registry, benchmarking helpers, and adversarial-training support — all factored out of the `axonml` umbrella crate in 0.6.1 (April 2026) to keep the umbrella focused on pure re-exports.

The **live browser training dashboard** (`TrainingMonitor`) stays in the umbrella crate at `axonml::monitor::TrainingMonitor` since it has no heavy dependencies of its own.

Last updated: 2026-06-06 — version 0.6.5.

---

## Modules

| Module | Purpose |
|--------|---------|
| `trainer` | `TrainingConfig`, `TrainingState`, `TrainingMetrics`, `TrainingHistory`, `Callback`, `EarlyStopping`, `ProgressLogger`, `clip_grad_norm`, `compute_accuracy` |
| `hub` | Unified model hub / registry — `UnifiedModelInfo`, `BenchmarkResult`, `ModelCategory`; with `full` feature adds `search_models`, `list_all_models`, `models_by_category`, `models_by_max_params`, `models_by_max_size_mb`, `recommended_models`, `compare_benchmarks` |
| `benchmark` | `benchmark_model`, `benchmark_model_named`, `throughput_test`, `warmup_model`, `profile_model_memory`, `print_memory_profile`, `print_throughput_results`, `compare_models`, `MemorySnapshot`, `ThroughputConfig`, `ThroughputResult` |
| `adversarial` | `AdversarialTrainer`, `fgsm_attack`, `pgd_attack`, `adversarial_training_step` |

## Public Re-exports

`lib.rs` re-exports the full surface directly at the crate root:

```rust
pub use trainer::{
    Callback, EarlyStopping, ProgressLogger, TrainingConfig, TrainingHistory,
    TrainingMetrics, TrainingState, clip_grad_norm, compute_accuracy,
};

pub use hub::{BenchmarkResult, ModelCategory, UnifiedModelInfo};

#[cfg(all(feature = "vision", feature = "llm"))]
pub use hub::{
    compare_benchmarks, list_all_models, models_by_category, models_by_max_params,
    models_by_max_size_mb, recommended_models, search_models,
};

pub use benchmark::{
    MemorySnapshot, ThroughputConfig, ThroughputResult, benchmark_model, benchmark_model_named,
    compare_models, print_memory_profile, print_throughput_results, profile_model_memory,
    throughput_test, warmup_model,
};

pub use adversarial::{AdversarialTrainer, adversarial_training_step, fgsm_attack, pgd_attack};
```

## Usage

```rust
use axonml_train::{
    TrainingConfig, EarlyStopping, ProgressLogger,
    AdversarialTrainer, fgsm_attack,
    benchmark_model, throughput_test,
};

let config = TrainingConfig::new()
    .epochs(50)
    .batch_size(32)
    .learning_rate(1e-3);

let early_stop = EarlyStopping::new(5);  // patience = 5 epochs
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `vision` | Include `axonml-vision` models in the hub registry |
| `llm` | Include `axonml-llm` models in the hub registry |
| `full` | Both `vision` + `llm` — required for `compare_benchmarks`, `list_all_models`, `search_models`, `recommended_models`, and the `models_by_*` helpers |
| `cuda` | GPU acceleration (forwarded to `axonml-core`, `axonml-tensor`, `axonml-nn`, `axonml-autograd`) |

```toml
[dependencies]
axonml-train = { version = "0.6.5", features = ["full"] }
```

## Dependencies

| Crate | Always | Notes |
|-------|--------|-------|
| `axonml-core` | yes | Error types, Device, DType |
| `axonml-tensor` | yes | Tensor operations |
| `axonml-autograd` | yes | Automatic differentiation |
| `axonml-nn` | yes | Model / Module trait |
| `axonml-optim` | yes | Optimizer trait for gradient clipping |
| `axonml-vision` | optional | `vision` feature — hub entries for vision models |
| `axonml-llm` | optional | `llm` feature — hub entries for LLM models |

## License

Licensed under either of Apache License 2.0 or MIT at your option.

---

*Part of [AxonML](https://github.com/AutomataNexus/AxonML) — a complete ML/AI framework in pure Rust.*
