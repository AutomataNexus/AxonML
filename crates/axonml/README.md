<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<h1 align="center">axonml</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.85%2B-orange.svg" alt="Rust: 1.85+"></a>
  <a href="https://crates.io/crates/axonml"><img src="https://img.shields.io/badge/crates.io-0.6.5-green.svg" alt="Crates.io: 0.6.5"></a>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part%20of-AxonML-teal.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

**axonml** is the umbrella crate for the AxonML machine-learning framework — a PyTorch-equivalent ML/AI toolkit written in pure Rust. It re-exports every sub-crate under a unified namespace, so you can pull in the whole framework with a single dependency.

The crate is intentionally thin. After the 0.6.1 split, it contains only:

1. Feature-gated re-exports of every sub-crate (`axonml-core` .. `axonml-train`)
2. A [`prelude`](#prelude) module with the most-used types
3. The live browser training monitor ([`TrainingMonitor`](#training-monitor))
4. `version()` / `features()` introspection helpers

Everything else — layers, optimizers, models, data loaders, training infrastructure, adversarial training — lives in dedicated sibling crates that can also be used standalone.

Last updated: 2026-06-06 — version 0.6.5.

---

## Sub-Crate Architecture

The umbrella crate re-exports the framework sub-crates under short module names:

### Core

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `core` | `axonml-core` | `axonml::core` | Error types, `Device`, `DType`, backend selection |
| `core` | `axonml-tensor` | `axonml::tensor` | N-dim tensor, BLAS, broadcasting, 80+ ops |
| `core` | `axonml-autograd` | `axonml::autograd` | Reverse-mode autograd, `Variable`, `no_grad` |

### Neural Networks

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `nn` | `axonml-nn` | `axonml::nn` | Layers — Linear, Conv1d/2d, Attention, LSTM/GRU, Transformer, etc. |
| `nn` | `axonml-optim` | `axonml::optim` | SGD, Adam, AdamW, RMSprop, schedulers |

### Data & I/O

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `data` | `axonml-data` | `axonml::data` | `Dataset` trait, `DataLoader`, samplers, transforms |
| `serialize` | `axonml-serialize` | `axonml::serialize` | `StateDict`, `Checkpoint`, safetensors, JSON/bincode |
| `onnx` | `axonml-onnx` | `axonml::onnx` | ONNX import / export |

### Domain-Specific

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `vision` | `axonml-vision` | `axonml::vision` | CNNs (LeNet, ResNet, VGG, ViT), detection (BlazeFace, RetinaFace, DETR, NanoDet), depth / anomaly / VQA, MNIST/CIFAR/COCO/WIDER FACE |
| `text` | `axonml-text` | `axonml::text` | BPE, WordPiece, Whitespace/Char tokenizers, text datasets |
| `audio` | `axonml-audio` | `axonml::audio` | MelSpectrogram, MFCC, resample, augmentation transforms |
| `llm` | `axonml-llm` | `axonml::llm` | Nine LLM architectures — see table below |

### Training, Optimization, Deployment

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `train` | `axonml-train` | `axonml::train` | `TrainingConfig`, `EarlyStopping`, `AdversarialTrainer`, unified model hub, benchmarking **— extracted in 0.6.1** |
| `distributed` | `axonml-distributed` | `axonml::distributed` | DDP, all-reduce, NCCL, process groups |
| `profile` | `axonml-profile` | `axonml::profile` | Memory / compute profilers, timeline, bottleneck detection |
| `quant` | `axonml-quant` | `axonml::quant` | INT8 / INT4 / FP16 quantization |
| `fusion` | `axonml-fusion` | `axonml::fusion` | Kernel fusion optimization |
| `jit` | `axonml-jit` | `axonml::jit` | Graph tracing + JIT compilation |

### Tooling (workspace crates, not re-exported)

| Sub-crate | Purpose |
|-----------|---------|
| `axonml-cli` | `axonml` command-line tool |
| `axonml-tui` | Ratatui terminal user interface |
| `axonml-server` | Axum REST/WebSocket API server |
| `axonml-dashboard` | Leptos/WASM web dashboard |

---

## LLM Architectures (`feature = "llm"`)

| Model | Novel Features | Purpose |
|-------|---------------|---------|
| **GPT-2** | Decoder-only transformer | Baseline causal LM |
| **LLaMA** | RoPE, GQA, SwiGLU | Modern efficient decoder LM |
| **Mistral** | Sliding-window attention, GQA | Long-context decoder LM |
| **Phi** | Partial RoPE, compact design | Small efficient LM |
| **BERT** | Bidirectional masked LM | Encoder for classification / masked LM |
| **SSM / Mamba** | Selective S6 scan, depthwise conv | Linear-complexity sequence model |

Plus:
- **BPE / WordPiece / Whitespace / Character tokenizers** (in `axonml-text`)
- **Generation utilities** — `TextGenerator`, `GenerationConfig`
- **Training scripts** for every model in `crates/axonml-llm/examples/`

The `axonml::prelude` re-exports only a small subset of LLM types (`Bert`, `BertConfig`, `BertForMaskedLM`, `BertForSequenceClassification`, `GPT2`, `GPT2Config`, `GPT2LMHead`, `GenerationConfig`, `TextGenerator`). For the other architectures pull from `axonml::llm::*` directly.

---

## Installation

```toml
[dependencies]
axonml = "0.6.5"                     # default = full feature set
```

Select only what you need:

```toml
[dependencies]
# Core tensors + autograd
axonml = { version = "0.6.5", default-features = false, features = ["core"] }

# Neural networks without domain-specific modules
axonml = { version = "0.6.5", default-features = false, features = ["nn", "data"] }

# Vision pipeline
axonml = { version = "0.6.5", default-features = false, features = ["vision"] }

# NLP + LLM pipeline
axonml = { version = "0.6.5", default-features = false, features = ["text", "llm"] }

# With GPU acceleration
axonml = { version = "0.6.5", features = ["full", "cuda"] }
```

---

## Usage

### Prelude

The `prelude` module exports the most-used types so you can get started fast:

```rust
use axonml::prelude::*;

fn main() -> axonml::core::Result<()> {
    // Tensor + autograd
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let var = Variable::new(x, true);

    // Build a model
    let model = Sequential::new()
        .add(Linear::new(784, 128))
        .add(ReLU)
        .add(Linear::new(128, 10));

    // Optimizer
    let mut optimizer = AdamW::new(model.parameters(), 1e-3);
    optimizer.zero_grad();

    Ok(())
}
```

Contents of the `prelude`, feature-gated:

- `core`: `DType`, `Device`, `Error`, `Result`, `Tensor`, `Variable`, `no_grad`
- `nn`: layers (`Linear`, `Conv2d`, `LSTM`, `GRU`, `RNN`, `MultiHeadAttention`, `BatchNorm1d/2d`, `LayerNorm`, `Dropout`, `Embedding`, `MaxPool2d`, `AvgPool2d`, activations, losses, `Parameter`, `Sequential`, `Module`) and optimizers (`SGD`, `Adam`, `AdamW`, `RMSprop`, `Optimizer`, `LRScheduler`, `CosineAnnealingLR`, `ExponentialLR`, `StepLR`)
- `data`: `DataLoader`, `Dataset`, `RandomSampler`, `SequentialSampler`, `Transform`
- `vision`: `LeNet`, `SimpleCNN`, `SyntheticMNIST`, `SyntheticCIFAR`, `CenterCrop`, `ImageNormalize`, `RandomHorizontalFlip`, `Resize`
- `text`: tokenizers, `Vocab`, `TextDataset`, `LanguageModelDataset`, `SyntheticSentimentDataset`
- `audio`: `MelSpectrogram`, `MFCC`, `Resample`, `NormalizeAudio`, `AddNoise`, synthetic datasets
- `distributed`: `DDP`, `DistributedDataParallel`, `ProcessGroup`, `World`, `all_reduce_{mean,sum}`, `barrier`, `broadcast`
- `profile`: `Profiler`, `ComputeProfiler`, `MemoryProfiler`, `TimelineProfiler`, `Bottleneck`, `BottleneckAnalyzer`, `ProfileGuard`, `ProfileReport`
- `llm`: `GPT2`, `GPT2Config`, `GPT2LMHead`, `Bert`, `BertConfig`, `BertForMaskedLM`, `BertForSequenceClassification`, `GenerationConfig`, `TextGenerator`
- `train`: `TrainingConfig`, `TrainingHistory`, `TrainingMetrics`, `EarlyStopping`, `ProgressLogger`, `Callback`, `AdversarialTrainer`
- `jit`: `CompiledFunction`, `Graph`, `JitCompiler`, `Optimizer as JitOptimizer`, `TracedValue`, `trace`

### Training Loop with Live Monitor

```rust
use axonml::prelude::*;
use axonml::TrainingMonitor;

fn train() -> axonml::core::Result<()> {
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Dropout::new(0.2))
        .add(Linear::new(256, 10));

    let mut optimizer = AdamW::new(model.parameters(), 1e-3);
    let param_count = model.parameters().iter().map(|p| p.numel()).sum::<usize>();

    // Live browser dashboard — opens Chromium automatically
    let monitor = TrainingMonitor::new("MNIST Classifier", param_count)
        .total_epochs(10)
        .batch_size(32)
        .launch();

    let dataset = SyntheticMNIST::new(1000);
    let loader = DataLoader::new(dataset, 32);

    for epoch in 0..10 {
        let mut epoch_loss = 0.0f32;
        let mut batches = 0;
        for batch in loader.iter() {
            let output = model.forward(&batch.data);
            let loss = CrossEntropyLoss::new().compute(&output, &batch.targets);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            epoch_loss += loss.data().to_vec()[0];
            batches += 1;
        }
        monitor.log_epoch(epoch + 1, epoch_loss / batches as f32, None, vec![]);
    }

    monitor.set_status("complete");
    Ok(())
}
```

---

## Training Monitor

`axonml::monitor::TrainingMonitor` (re-exported as `axonml::TrainingMonitor`) is a zero-dependency, pure-Rust HTTP server that serves a real-time training dashboard to your browser. The dashboard HTML lives next to the module at `crates/axonml/src/monitor_dashboard.html`. It is intentionally kept in the umbrella crate so every training script across the workspace can use it with a single import.

```rust
use axonml::TrainingMonitor;

let monitor = TrainingMonitor::new("MyModel", param_count)
    .total_epochs(50)
    .batch_size(32)
    .launch();  // opens http://127.0.0.1:<auto-port> in Chromium

// Each epoch
monitor.log_epoch(epoch + 1, train_loss, Some(val_loss), vec![
    ("accuracy", acc),
    ("lr", lr),
]);

// When done
monitor.set_status("complete");
```

The dashboard shows real-time training loss, optional validation loss, custom metric overlays, current epoch / total epochs, and best loss so far.

---

## Feature Flag Reference

| Feature | Implies | Description |
|---------|---------|-------------|
| `default` | `full` | Complete framework |
| `full` | everything below (except `nccl`/`cuda`/`cudnn`/`wgpu`) | Complete framework |
| `core` | — | `axonml-core` + `axonml-tensor` + `axonml-autograd` |
| `nn` | `core` | `axonml-nn` + `axonml-optim` |
| `data` | `core` | `axonml-data` |
| `vision` | `nn`, `data` | `axonml-vision` |
| `text` | `nn`, `data` | `axonml-text` |
| `audio` | `nn`, `data` | `axonml-audio` |
| `llm` | `nn` | `axonml-llm` (all 9 architectures) |
| `train` | `nn` | `axonml-train` (trainer, hub, benchmark, adversarial) |
| `distributed` | `nn` | `axonml-distributed` |
| `profile` | `core` | `axonml-profile` |
| `serialize` | `core` | `axonml-serialize` |
| `onnx` | `core` | `axonml-onnx` |
| `quant` | `nn` | `axonml-quant` |
| `fusion` | `core` | `axonml-fusion` |
| `jit` | `core` | `axonml-jit` |
| `nccl` | `distributed` | NCCL distributed communication (requires CUDA + `libnccl.so.2`) |
| `cuda` | — | Forwards CUDA to `axonml-core`, `axonml-tensor`, `axonml-nn` |
| `cudnn` | `cuda` | Forwards cuDNN to core/tensor/nn |
| `wgpu` | — | Forwards WebGPU/Vulkan to `axonml-core` |

---

## Examples

The crate ships three generic examples; domain-specific examples live in the per-domain crates.

```bash
# Simple training loop
cargo run -p axonml --example simple_training

# MNIST digit classification
cargo run -p axonml --example mnist_training

# NLP + audio transform smoke test
cargo run -p axonml --example nlp_audio_test
```

For model-specific training scripts, see the per-crate `examples/` directories:

```bash
# LLM training
cargo run -p axonml-llm --example train_gpt2 --release

# Vision training
cargo run -p axonml-vision --example train_resnet --release
```

---

## Introspection

```rust
use axonml::{version, features};

fn main() {
    println!("AxonML version: {}", version());
    println!("Enabled features: {}", features());
}
```

---

## Version Information

- **Crate version:** 0.6.5
- **Rust edition:** 2024
- **MSRV:** Rust 1.85+
- **0.6.1 split:** `train` (trainer / hub / benchmark / adversarial) was extracted from this umbrella into a standalone crate to keep the umbrella a thin re-export layer. The live browser `TrainingMonitor` stayed here.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../LICENSE-MIT))

at your option.

---

*Part of [AxonML](https://github.com/AutomataNexus/AxonML) — a complete ML/AI framework in pure Rust, by AutomataNexus LLC.*
