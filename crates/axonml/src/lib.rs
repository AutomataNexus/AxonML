//! # Axonml - A Complete ML/AI Framework in Pure Rust
//!
//! Axonml is a comprehensive machine learning framework that provides ~92-95% PyTorch-equivalent
//! functionality in pure Rust. It includes 1076+ passing tests and production-ready features:
//!
//! ## Core Features
//!
//! - **Tensors**: N-dimensional arrays with broadcasting, views, BLAS operations, sparse tensors
//! - **70+ Tensor Operations**: arithmetic, reduction, sorting (topk, sort, argsort), indexing (gather, scatter, nonzero, unique)
//! - **Autograd**: Automatic differentiation with computational graph, AMP (autocast), gradient checkpointing
//! - **Neural Networks**: Linear, Conv1d/2d, BatchNorm, LayerNorm, GroupNorm, InstanceNorm, Attention, RNN/LSTM/GRU
//! - **Optimizers**: SGD, Adam, `AdamW`, `RMSprop`, LAMB with LR schedulers and GradScaler
//! - **Data Loading**: Dataset trait, `DataLoader`, samplers, transforms, parallel loading
//! - **Vision**: Image transforms, MNIST/CIFAR datasets, ResNet/VGG/ViT architectures, pretrained hub
//! - **Text**: Tokenizers (BPE, `WordPiece`), vocabularies, text datasets
//! - **Audio**: Spectrograms, MFCC, audio transforms, audio datasets
//! - **Distributed**: DDP, FSDP (ZeRO-2/3), Pipeline Parallelism, Tensor Parallelism
//! - **LLM**: BERT, GPT-2 architectures with pretrained model hub (LLaMA, Mistral, Phi, Qwen)
//! - **GPU Backends**: CUDA, Vulkan, Metal, WebGPU with comprehensive test suite
//!
//! ## Model Hub & Benchmarking
//!
//! - Unified model registry across vision and LLM domains
//! - Model search, filtering, and recommendations
//! - Throughput testing and memory profiling utilities
//!
//! # Quick Start
//!
//! ```ignore
//! use axonml::prelude::*;
//!
//! // Create a tensor
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
//!
//! // Create a variable for autograd
//! let var = Variable::new(x, true);
//!
//! // Build a simple neural network
//! let model = Sequential::new()
//!     .add(Linear::new(784, 128))
//!     .add(ReLU)
//!     .add(Linear::new(128, 10));
//!
//! // Create an optimizer
//! let optimizer = Adam::new(model.parameters(), 0.001);
//!
//! // Load a dataset
//! let dataset = SyntheticMNIST::new(1000);
//! let loader = DataLoader::new(dataset, 32);
//!
//! // Training loop
//! for batch in loader.iter() {
//!     // Forward pass
//!     let output = model.forward(&batch.data);
//!
//!     // Compute loss
//!     let loss = cross_entropy_loss(&output, &batch.labels);
//!
//!     // Backward pass
//!     loss.backward();
//!
//!     // Update weights
//!     optimizer.step();
//!     optimizer.zero_grad();
//! }
//! ```
//!
//! # Mixed Precision Training
//!
//! ```ignore
//! use axonml_autograd::amp::autocast;
//! use axonml_optim::GradScaler;
//!
//! let mut scaler = GradScaler::new();
//!
//! // Forward with autocast
//! let loss = autocast(DType::F16, || {
//!     model.forward(&input)
//! });
//!
//! // Scale loss for backward
//! let scaled_loss = scaler.scale_loss(loss);
//! scaled_loss.backward();
//!
//! // Unscale and step
//! scaler.unscale_grads(&mut gradients);
//! optimizer.step();
//! scaler.update();
//! ```
//!
//! # Feature Flags
//!
//! - `full` (default): All features enabled
//! - `core`: Core tensor and autograd functionality
//! - `nn`: Neural network layers and optimizers
//! - `data`: Data loading utilities
//! - `vision`: Image processing and vision datasets
//! - `text`: Text processing and NLP utilities
//! - `audio`: Audio processing utilities
//! - `distributed`: Distributed training utilities (DDP, FSDP, Pipeline)
//! - `profile`: Performance profiling and bottleneck detection
//! - `llm`: LLM architectures (BERT, GPT-2) with pretrained hub
//! - `jit`: JIT compilation and tracing
//! - `onnx`: ONNX model import and export
//! - `cuda`: NVIDIA CUDA GPU backend
//! - `vulkan`: Vulkan GPU backend
//! - `metal`: Apple Metal GPU backend
//! - `wgpu`: WebGPU backend
//!
//! @version 0.2.6
//! @author `AutomataNexus` Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// ML/tensor-specific allowances
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unused_self)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::single_match_else)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::format_push_string)]
#![allow(clippy::erasing_op)]
#![allow(clippy::type_repetition_in_bounds)]
#![allow(clippy::iter_without_into_iter)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::use_debug)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::panic)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::ref_option)]
#![allow(clippy::multiple_bound_locations)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::manual_assert)]
#![allow(clippy::unnecessary_debug_formatting)]

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

// =============================================================================
// Training Utilities
// =============================================================================

pub mod trainer;
pub use trainer::{
    Callback, EarlyStopping, ProgressLogger, TrainingConfig, TrainingHistory,
    TrainingMetrics, TrainingState,
};

#[cfg(feature = "nn")]
pub use trainer::clip_grad_norm;

#[cfg(feature = "core")]
pub use trainer::compute_accuracy;

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
    print_throughput_results, MemorySnapshot, ThroughputConfig, ThroughputResult,
};

#[cfg(all(feature = "core", feature = "nn"))]
pub use benchmark::{
    benchmark_model, benchmark_model_named, compare_models, throughput_test, warmup_model,
};

#[cfg(feature = "nn")]
pub use benchmark::{print_memory_profile, profile_model_memory};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for machine learning tasks.
///
/// This module re-exports the most commonly used types and traits from all
/// Axonml subcrates, allowing you to get started quickly with:
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
    pub use axonml_autograd::{no_grad, Variable};

    // Neural network modules
    #[cfg(feature = "nn")]
    pub use axonml_nn::{
        AvgPool2d, BCELoss, BatchNorm1d, BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Embedding,
        L1Loss, LayerNorm, LeakyReLU, Linear, MSELoss, MaxPool2d, Module, MultiHeadAttention,
        Parameter, ReLU, Sequential, SiLU, Sigmoid, Softmax, Tanh, GELU, GRU, LSTM, RNN,
    };

    // Optimizers
    #[cfg(feature = "nn")]
    pub use axonml_optim::{
        Adam, AdamW, CosineAnnealingLR, ExponentialLR, LRScheduler, Optimizer, RMSprop, StepLR, SGD,
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
        AddNoise, MelSpectrogram, NormalizeAudio, Resample, SyntheticCommandDataset,
        SyntheticMusicDataset, MFCC,
    };

    // Distributed
    #[cfg(feature = "distributed")]
    pub use axonml_distributed::{
        all_reduce_mean, all_reduce_sum, barrier, broadcast, DistributedDataParallel, ProcessGroup,
        World, DDP,
    };

    // Profiling
    #[cfg(feature = "profile")]
    pub use axonml_profile::{
        Profiler, ProfileGuard, ProfileReport, MemoryProfiler, ComputeProfiler,
        TimelineProfiler, BottleneckAnalyzer, Bottleneck,
    };

    // LLM architectures
    #[cfg(feature = "llm")]
    pub use axonml_llm::{
        BertConfig, GPT2Config, Bert, BertForSequenceClassification, BertForMaskedLM,
        GPT2, GPT2LMHead, GenerationConfig, TextGenerator,
    };

    // JIT compilation
    #[cfg(feature = "jit")]
    pub use axonml_jit::{
        trace, Graph, JitCompiler, CompiledFunction, TracedValue, Optimizer as JitOptimizer,
    };
}

// =============================================================================
// Version Information
// =============================================================================

/// Returns the version of the Axonml framework.
#[must_use] pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns a string describing the enabled features.
#[must_use] pub fn features() -> String {
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
        // With default features, should have all
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

    #[cfg(feature = "nn")]
    #[test]
    fn test_optimizer() {
        use nn::Linear;
        use nn::Module;
        use optim::{Adam, Optimizer};

        let model = Linear::new(4, 2);
        let mut optimizer = Adam::new(model.parameters(), 0.001);

        // Should be able to zero gradients
        optimizer.zero_grad();
    }

    #[cfg(feature = "data")]
    #[test]
    fn test_dataloader() {
        use data::{DataLoader, Dataset};
        use tensor::Tensor;

        struct DummyDataset;

        impl Dataset for DummyDataset {
            type Item = (Tensor<f32>, Tensor<f32>);

            fn len(&self) -> usize {
                100
            }

            fn get(&self, _index: usize) -> Option<Self::Item> {
                let x = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
                let y = Tensor::from_vec(vec![1.0], &[1]).unwrap();
                Some((x, y))
            }
        }

        let dataset = DummyDataset;
        let loader = DataLoader::new(dataset, 10);

        assert_eq!(loader.len(), 10); // 100 / 10
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_vision_dataset() {
        use data::Dataset;
        use vision::SyntheticMNIST;

        let dataset = SyntheticMNIST::new(100);
        assert_eq!(dataset.len(), 100);
    }

    #[cfg(feature = "text")]
    #[test]
    fn test_tokenizer() {
        use text::{Tokenizer, WhitespaceTokenizer};

        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("hello world");

        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[cfg(feature = "audio")]
    #[test]
    fn test_audio_transform() {
        use audio::MelSpectrogram;
        use data::Transform;
        use std::f32::consts::PI;
        use tensor::Tensor;

        // Create a simple sine wave
        let data: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let audio = Tensor::from_vec(data, &[4096]).unwrap();

        let mel = MelSpectrogram::with_params(16000, 512, 256, 40);
        let spec = mel.apply(&audio);

        assert_eq!(spec.shape()[0], 40); // 40 mel bins
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_distributed_world() {
        use distributed::World;

        let world = World::mock();
        assert_eq!(world.rank(), 0);
        assert_eq!(world.world_size(), 1);
    }

    #[test]
    fn test_prelude_imports() {
        // This test just ensures the prelude compiles correctly
        use crate::prelude::*;

        #[cfg(feature = "core")]
        {
            let _ = Device::Cpu;
        }
    }
}
