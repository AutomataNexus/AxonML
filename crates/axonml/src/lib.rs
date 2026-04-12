//! Axonml — Umbrella Crate
//!
//! # File
//! `crates/axonml/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Overview
//!
//! `axonml` is a thin umbrella crate that re-exports the full AxonML deep
//! learning framework under a single unified namespace. It also hosts the
//! **live browser training monitor** (`TrainingMonitor`), which is small,
//! dependency-light, and used by essentially every training script in the
//! workspace.
//!
//! Domain-specific models (e.g. HVAC diagnostics) and training infrastructure
//! (trainer, hub, benchmark, adversarial) live in dedicated sibling crates:
//!
//! - `axonml-hvac`  — HVAC fault-detection models (Apollo, Panoptes, etc.)
//! - `axonml-train` — `TrainingConfig`, `EarlyStopping`, `AdversarialTrainer`,
//!                    `benchmark_model`, unified model hub
//!
//! This separation was made in April 2026 to keep the umbrella crate focused
//! on re-exports and the live training dashboard.
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#![warn(clippy::all)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]

// =============================================================================
// Core Re-exports
// =============================================================================

#[cfg(feature = "core")]
pub use axonml_core as core;

#[cfg(feature = "core")]
pub use axonml_tensor as tensor;

#[cfg(feature = "core")]
pub use axonml_autograd as autograd;

// =============================================================================
// Neural Network Re-exports
// =============================================================================

#[cfg(feature = "nn")]
pub use axonml_nn as nn;

#[cfg(feature = "nn")]
pub use axonml_optim as optim;

// =============================================================================
// Data Re-exports
// =============================================================================

#[cfg(feature = "data")]
pub use axonml_data as data;

// =============================================================================
// Domain-Specific Re-exports
// =============================================================================

#[cfg(feature = "vision")]
pub use axonml_vision as vision;

#[cfg(feature = "text")]
pub use axonml_text as text;

#[cfg(feature = "audio")]
pub use axonml_audio as audio;

#[cfg(feature = "distributed")]
pub use axonml_distributed as distributed;

#[cfg(feature = "profile")]
pub use axonml_profile as profile;

#[cfg(feature = "llm")]
pub use axonml_llm as llm;

#[cfg(feature = "jit")]
pub use axonml_jit as jit;

#[cfg(feature = "onnx")]
pub use axonml_onnx as onnx;

#[cfg(feature = "serialize")]
pub use axonml_serialize as serialize;

#[cfg(feature = "quant")]
pub use axonml_quant as quant;

#[cfg(feature = "fusion")]
pub use axonml_fusion as fusion;

#[cfg(feature = "hvac")]
pub use axonml_hvac as hvac;

#[cfg(feature = "train")]
pub use axonml_train as train;

// =============================================================================
// Training Monitor — stays in the umbrella crate
// =============================================================================

/// Live browser-based training monitor — opens Chromium with real-time charts.
pub mod monitor;
pub use monitor::TrainingMonitor;

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for machine learning tasks.
///
/// ```ignore
/// use axonml::prelude::*;
/// ```
pub mod prelude {
    // Core types
    #[cfg(feature = "core")]
    pub use axonml_core::{DType, Device, Error, Result};

    // Tensor operations
    #[cfg(feature = "core")]
    pub use axonml_tensor::Tensor;

    // Autograd
    #[cfg(feature = "core")]
    pub use axonml_autograd::{Variable, no_grad};

    // Neural network modules
    #[cfg(feature = "nn")]
    pub use axonml_nn::{
        AvgPool2d, BCELoss, BatchNorm1d, BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Embedding,
        GELU, GRU, L1Loss, LSTM, LayerNorm, LeakyReLU, Linear, MSELoss, MaxPool2d, Module,
        MultiHeadAttention, Parameter, RNN, ReLU, Sequential, SiLU, Sigmoid, Softmax, Tanh,
    };

    // Optimizers
    #[cfg(feature = "nn")]
    pub use axonml_optim::{
        Adam, AdamW, CosineAnnealingLR, ExponentialLR, LRScheduler, Optimizer, RMSprop, SGD, StepLR,
    };

    // Data loading
    #[cfg(feature = "data")]
    pub use axonml_data::{DataLoader, Dataset, RandomSampler, SequentialSampler, Transform};

    // Vision
    #[cfg(feature = "vision")]
    pub use axonml_vision::{
        CenterCrop, ImageNormalize, LeNet, RandomHorizontalFlip, Resize, SimpleCNN, SyntheticCIFAR,
        SyntheticMNIST,
    };

    // Text
    #[cfg(feature = "text")]
    pub use axonml_text::{
        BasicBPETokenizer, CharTokenizer, LanguageModelDataset, SyntheticSentimentDataset,
        TextDataset, Tokenizer, Vocab, WhitespaceTokenizer,
    };

    // Audio
    #[cfg(feature = "audio")]
    pub use axonml_audio::{
        AddNoise, MFCC, MelSpectrogram, NormalizeAudio, Resample, SyntheticCommandDataset,
        SyntheticMusicDataset,
    };

    // Distributed
    #[cfg(feature = "distributed")]
    pub use axonml_distributed::{
        DDP, DistributedDataParallel, ProcessGroup, World, all_reduce_mean, all_reduce_sum,
        barrier, broadcast,
    };

    // Profiling
    #[cfg(feature = "profile")]
    pub use axonml_profile::{
        Bottleneck, BottleneckAnalyzer, ComputeProfiler, MemoryProfiler, ProfileGuard,
        ProfileReport, Profiler, TimelineProfiler,
    };

    // LLM architectures — all nine models
    #[cfg(feature = "llm")]
    pub use axonml_llm::{
        Bert, BertConfig, BertForMaskedLM, BertForSequenceClassification, GPT2, GPT2Config,
        GPT2LMHead, GenerationConfig, TextGenerator,
    };

    // Training infrastructure
    #[cfg(feature = "train")]
    pub use axonml_train::{
        AdversarialTrainer, Callback, EarlyStopping, ProgressLogger, TrainingConfig,
        TrainingHistory, TrainingMetrics,
    };

    // JIT compilation
    #[cfg(feature = "jit")]
    pub use axonml_jit::{
        CompiledFunction, Graph, JitCompiler, Optimizer as JitOptimizer, TracedValue, trace,
    };
}

// =============================================================================
// Version Information
// =============================================================================

/// Returns the version of the Axonml framework.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns a string describing the enabled features.
#[must_use]
pub fn features() -> String {
    let mut features = Vec::new();

    #[cfg(feature = "core")]
    features.push("core");

    #[cfg(feature = "nn")]
    features.push("nn");

    #[cfg(feature = "data")]
    features.push("data");

    #[cfg(feature = "vision")]
    features.push("vision");

    #[cfg(feature = "text")]
    features.push("text");

    #[cfg(feature = "audio")]
    features.push("audio");

    #[cfg(feature = "distributed")]
    features.push("distributed");

    #[cfg(feature = "profile")]
    features.push("profile");

    #[cfg(feature = "llm")]
    features.push("llm");

    #[cfg(feature = "jit")]
    features.push("jit");

    #[cfg(feature = "onnx")]
    features.push("onnx");

    #[cfg(feature = "serialize")]
    features.push("serialize");

    #[cfg(feature = "quant")]
    features.push("quant");

    #[cfg(feature = "fusion")]
    features.push("fusion");

    #[cfg(feature = "hvac")]
    features.push("hvac");

    #[cfg(feature = "train")]
    features.push("train");

    if features.is_empty() {
        "none".to_string()
    } else {
        features.join(", ")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_features() {
        let f = features();
        assert!(f.contains("core"));
    }

    #[cfg(feature = "core")]
    #[test]
    fn test_tensor_creation() {
        use tensor::Tensor;
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[cfg(feature = "core")]
    #[test]
    fn test_variable_creation() {
        use autograd::Variable;
        use tensor::Tensor;
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let v = Variable::new(t, true);
        assert_eq!(v.data().shape(), &[3]);
    }

    #[cfg(feature = "nn")]
    #[test]
    fn test_linear_layer() {
        use autograd::Variable;
        use nn::Linear;
        use nn::Module;
        use tensor::Tensor;

        let layer = Linear::new(4, 2);
        let input = Variable::new(Tensor::from_vec(vec![1.0; 4], &[1, 4]).unwrap(), false);
        let output = layer.forward(&input);
        assert_eq!(output.data().shape(), &[1, 2]);
    }
}
