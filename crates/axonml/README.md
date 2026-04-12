<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<h1 align="center">axonml</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.85%2B-orange.svg" alt="Rust: 1.85+"></a>
  <a href="https://crates.io/crates/axonml"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Crates.io: 0.6.1"></a>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part%20of-AxonML-teal.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

**axonml** is the umbrella crate for the AxonML machine learning framework — a complete, PyTorch-equivalent ML/AI toolkit written in pure Rust. It re-exports every sub-crate under a unified API, so you can build and train deep learning models with a single dependency.

The crate is intentionally thin. It contains:

1. Feature-gated re-exports of every sub-crate
2. A [`prelude`](#prelude) module with the most-used types
3. The live browser training monitor ([`TrainingMonitor`](#training-monitor))
4. `version()` / `features()` introspection helpers

Everything else — layers, optimizers, models, data loaders, training infrastructure, HVAC diagnostics — lives in dedicated sibling crates that can also be used standalone.

---

## Sub-Crate Architecture

AxonML is a workspace of 24 sub-crates. The umbrella crate re-exports them under short names:

### Core

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `core` | `axonml-core` | `axonml::core` | Error types, `Device`, `DType`, backend selection |
| `core` | `axonml-tensor` | `axonml::tensor` | N-dim tensor, BLAS, broadcasting, 80+ ops |
| `core` | `axonml-autograd` | `axonml::autograd` | Reverse-mode autograd, `Variable`, `no_grad` |

### Neural Networks

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `nn` | `axonml-nn` | `axonml::nn` | 41 layers — Linear, Conv1d/2d, Attention, LSTM/GRU, Transformer, etc. |
| `nn` | `axonml-optim` | `axonml::optim` | SGD, Adam, AdamW, RMSprop, LAMB + schedulers |

### Data & I/O

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `data` | `axonml-data` | `axonml::data` | `Dataset` trait, `DataLoader`, samplers, transforms |
| `serialize` | `axonml-serialize` | `axonml::serialize` | `StateDict`, `Checkpoint`, safetensors, JSON/bincode |
| `onnx` | `axonml-onnx` | `axonml::onnx` | ONNX import / export |

### Domain-Specific

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `vision` | `axonml-vision` | `axonml::vision` | CNNs (LeNet, ResNet, VGG, ViT), MNIST/CIFAR/COCO/WIDER FACE, **Aegis biometric suite** (Mnemosyne, Argus, Echo, Ariadne, Themis) |
| `text` | `axonml-text` | `axonml::text` | BPE, WordPiece, Whitespace/Char tokenizers, text datasets |
| `audio` | `axonml-audio` | `axonml::audio` | MelSpectrogram, MFCC, resample, augmentation transforms |
| `llm` | `axonml-llm` | `axonml::llm` | **9 LLM architectures** — see table below |
| `hvac` | `axonml-hvac` | `axonml::hvac` | HVAC diagnostic models (Apollo, Panoptes, Vulcan, etc.) |

### Training, Optimization, Deployment

| Feature | Sub-crate | Namespace | Purpose |
|---------|-----------|-----------|---------|
| `train` | `axonml-train` | `axonml::train` | `TrainingConfig`, `EarlyStopping`, `AdversarialTrainer`, unified model hub, benchmarking |
| `distributed` | `axonml-distributed` | `axonml::distributed` | DDP, all-reduce, NCCL, process groups |
| `profile` | `axonml-profile` | `axonml::profile` | Memory / compute profilers, timeline, bottleneck detection |
| `quant` | `axonml-quant` | `axonml::quant` | INT8 / INT4 / FP16 quantization |
| `fusion` | `axonml-fusion` | `axonml::fusion` | Kernel fusion optimization |
| `jit` | `axonml-jit` | `axonml::jit` | Graph tracing + JIT compilation |

### Tooling (workspace crates, not re-exported)

| Sub-crate | Purpose |
|-----------|---------|
| `axonml-cli` | `axonml` command-line tool for scaffolding projects |
| `axonml-tui` | Terminal user interface |
| `axonml-server` | HTTP / gRPC inference server |
| `axonml-dashboard` | Web dashboard for training monitoring |

---

## Nine LLM Architectures (`feature = "llm"`)

| Model | Novel Features | Purpose |
|-------|---------------|---------|
| **GPT-2** | Decoder-only transformer | Baseline causal LM |
| **LLaMA** | RoPE, GQA, SwiGLU | Modern efficient decoder LM |
| **Mistral** | Sliding-window attention, GQA | Long-context decoder LM |
| **Phi** | Partial RoPE, compact design | Small efficient LM |
| **BERT** | Bidirectional masked LM | Encoder for classification / masked LM |
| **SSM / Mamba** | Selective S6 scan, depthwise conv | Linear-complexity sequence model |
| **Hydra** | Hybrid SSM + windowed attention | Best-of-both-worlds architecture |
| **Trident** | 1.58-bit ternary weights, 16× compression | Published paper reference implementation |
| **Chimera** | Sparse MoE (8 experts, top-2) + Differential Attention | Large-capacity conditional compute |

Plus:
- **BPE / WordPiece / Whitespace / Character tokenizers** (in `axonml-text`)
- **Generation utilities** — `TextGenerator`, `GenerationConfig`
- **Training scripts** for every model in `crates/axonml-llm/examples/`

---

## Installation

```toml
[dependencies]
axonml = "0.6.1"                     # default = full feature set
```

Select only what you need:

```toml
[dependencies]
# Core tensors + autograd
axonml = { version = "0.6.1", default-features = false, features = ["core"] }

# Neural networks without domain-specific modules
axonml = { version = "0.6.1", default-features = false, features = ["nn", "data"] }

# Vision pipeline
axonml = { version = "0.6.1", default-features = false, features = ["vision"] }

# NLP + LLM pipeline
axonml = { version = "0.6.1", default-features = false, features = ["text", "llm"] }

# With GPU acceleration
axonml = { version = "0.6.1", features = ["full", "cuda"] }
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

`axonml::monitor::TrainingMonitor` is a zero-dependency, pure-Rust HTTP server that serves a real-time training dashboard to your browser. It's intentionally kept in the umbrella crate so every training script across the workspace can use it with a single import.

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

The dashboard shows:
- Real-time training loss curve
- Validation loss (if provided)
- Custom metric overlays
- Current epoch / total epochs
- Best loss so far

---

## Feature Flag Reference

| Feature | Includes | Description |
|---------|----------|-------------|
| `full` | All features | Complete framework (default) |
| `core` | tensor, autograd | Core tensor operations and autodiff |
| `nn` | core + nn, optim | Neural network layers and optimizers |
| `data` | core + data | DataLoader and dataset utilities |
| `vision` | nn, data + vision | Image processing + Aegis biometric suite |
| `text` | nn, data + text | Tokenizers and text processing |
| `audio` | nn, data + audio | Audio transforms and datasets |
| `llm` | nn + llm | All 9 LLM architectures |
| `hvac` | nn + hvac | HVAC diagnostic models |
| `train` | nn + train | High-level training, adversarial, hub, benchmark |
| `distributed` | nn + distributed | Distributed training (DDP + NCCL) |
| `profile` | core + profile | Profiling and bottleneck analysis |
| `serialize` | core + serialize | Model checkpoint save/load |
| `onnx` | core + onnx | ONNX import/export |
| `quant` | nn + quant | INT8 / INT4 / FP16 quantization |
| `fusion` | core + fusion | Kernel fusion optimization |
| `jit` | core + jit | JIT compilation and tracing |
| `cuda` | — | CUDA GPU acceleration |
| `cudnn` | cuda + cudnn | cuDNN acceleration |
| `wgpu` | — | WebGPU/Vulkan GPU acceleration |
| `nccl` | distributed + nccl | NCCL distributed communication |

---

## Examples

The crate includes three generic examples (HVAC-specific examples moved to `axonml-hvac`):

```bash
# Simple training loop
cargo run -p axonml --example simple_training

# MNIST digit classification
cargo run -p axonml --example mnist_training

# NLP + audio transform test
cargo run -p axonml --example nlp_audio_test
```

For model-specific training scripts, see the per-crate `examples/` directories:

```bash
# LLM training (all 9 architectures)
cargo run -p axonml-llm --example train_gpt2 --release
cargo run -p axonml-llm --example train_trident --release
cargo run -p axonml-llm --example train_hydra --release

# Vision training
cargo run -p axonml-vision --example train_resnet --release
cargo run -p axonml-vision --example train_mnemosyne --release  # Aegis biometric

# HVAC training
cargo run -p axonml-hvac --example train_panoptes --release
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

- **Crate version:** 0.6.1
- **Rust edition:** 2024
- **MSRV:** Rust 1.85+
- **Workspace members:** 24 sub-crates

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../LICENSE-MIT))

at your option.

---

*Part of [AxonML](https://github.com/AutomataNexus/AxonML) — a complete ML/AI framework in pure Rust, by AutomataNexus LLC.*
