//! Axonml Train — High-Level Training Infrastructure
//!
//! # File
//! `crates/axonml-train/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Overview
//!
//! Training utilities split out of the `axonml` umbrella crate:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`trainer`]     | `TrainingConfig`, `Callback`, `EarlyStopping`, `ProgressLogger`, `TrainingHistory`, `TrainingMetrics`, `clip_grad_norm` |
//! | [`hub`]         | Unified model hub / registry (`UnifiedModelInfo`, `BenchmarkResult`, `ModelCategory`) — requires `vision` + `llm` features for full catalog |
//! | [`benchmark`]   | `benchmark_model`, `throughput_test`, `profile_model_memory`, `compare_models` |
//! | [`adversarial`] | `AdversarialTrainer`, `fgsm_attack`, `pgd_attack` |
//!
//! The **live browser training monitor** stays in the `axonml` umbrella crate
//! (see `axonml::monitor::TrainingMonitor`).
//!
//! # Feature Flags
//!
//! - `vision` — include `axonml-vision` models in the hub registry
//! - `llm` — include `axonml-llm` models in the hub registry
//! - `full` — both `vision` + `llm`
//! - `cuda` — GPU acceleration
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#![warn(clippy::all)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::doc_markdown)]

// =============================================================================
// Training Utilities
// =============================================================================

pub mod trainer;
pub use trainer::{
    Callback, EarlyStopping, ProgressLogger, TrainingConfig, TrainingHistory, TrainingMetrics,
    TrainingState, clip_grad_norm, compute_accuracy,
};

// =============================================================================
// Model Hub
// =============================================================================

pub mod hub;
pub use hub::{BenchmarkResult, ModelCategory, UnifiedModelInfo};

#[cfg(all(feature = "vision", feature = "llm"))]
pub use hub::{
    compare_benchmarks, list_all_models, models_by_category, models_by_max_params,
    models_by_max_size_mb, recommended_models, search_models,
};

// =============================================================================
// Benchmarking
// =============================================================================

pub mod benchmark;
pub use benchmark::{
    MemorySnapshot, ThroughputConfig, ThroughputResult, benchmark_model, benchmark_model_named,
    compare_models, print_memory_profile, print_throughput_results, profile_model_memory,
    throughput_test, warmup_model,
};

// =============================================================================
// Adversarial Training
// =============================================================================

pub mod adversarial;
pub use adversarial::{AdversarialTrainer, adversarial_training_step, fgsm_attack, pgd_attack};
