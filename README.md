<div align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="400"/>

  <h1>AxonML</h1>

  <p><strong>A complete, PyTorch-equivalent machine learning framework written in pure Rust.</strong></p>

  [![CI](https://github.com/AutomataNexus/AxonML/actions/workflows/ci.yml/badge.svg)](https://github.com/AutomataNexus/AxonML/actions/workflows/ci.yml)
  [![Crates.io](https://img.shields.io/crates/v/axonml.svg)](https://crates.io/crates/axonml)
  [![Docs.rs](https://docs.rs/axonml/badge.svg)](https://docs.rs/axonml)
  [![Downloads](https://img.shields.io/crates/d/axonml.svg)](https://crates.io/crates/axonml)
  [![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

</div>

## Overview

Axonml (named after axons - the nerve fibers that transmit signals between neurons) is an ambitious open-source project to create a complete machine learning framework in Rust. Our goal is to provide the same comprehensive functionality as PyTorch while leveraging Rust's performance, safety, and concurrency guarantees.

## PyTorch Parity: ~92-95% (and beyond)

AxonML provides comprehensive PyTorch-equivalent functionality across 24 crates with **1,700+ tests** (1,773 test functions; the workspace suite passes on CPU and on CUDA). Several features go **beyond PyTorch** with novel capabilities not available in any other framework.

## What's new in 0.7.0

The full audit trail is in [CHANGELOG.md](CHANGELOG.md). Highlights:

- **A GPU training engine you can trust.** A series of silent GPU-only bugs, each found by checking the GPU against the CPU as an oracle, are fixed:
  - `Adam`/`AdamW` applied strided gradients as if they were contiguous, which scrambled every dense-layer update.
  - CUDA `pow` computed `|x|^n`.
  - `conv2d` forward/backward raced the compute stream once the memory pool was dirty.
  - `RMSNorm`'s scale never trained.
  - BCE detached its gradient.

  Grouped/depthwise conv, pooling, GroupNorm/InstanceNorm, ConvTranspose2d and interpolate now have GPU backward passes. Batched conv is one im2col plus one strided-batched GEMM. Shape uploads are cached, cutting host-to-device copies per training step from 762 to 18.
- **JIT that captures your real model.** `JitFn::trace_forward` records a model's actual `forward`, with no tracer DSL to re-declare. Elementwise chains fuse and compile at runtime through NVRTC, and dispatch on-device.
- **Graph-carrying model files.** `.axonml` bundles store the model's exact compute graph, skip connections included. `Adam::export_state` / `import_state` give lossless pause and resume.
- **LLM fine-tuning.**
  - LoRA on `Linear`, LLaMA and Qwen3: attach, freeze base, merge.
  - Per-layer activation checkpointing (`AXONML_CKPT_LAYERS`).
  - A tied Qwen3 LM head.
  - Q8_0 GGUF export (`AXONML_GGUF_QUANT=q8_0`).
- **`axonml-subbit` (new crate): sub-bit vector-quantized GPU primitives.**
  - Fused VQ matmul runs projections straight off packed codebook indices.
  - Grouped MoE experts with device-side router top-k.
  - Offload tiles for training a block wider than VRAM.
  - Batched multi-sequence decode for Mamba2 + MoE hybrids.
  - An NVFP4 training path for Blackwell (sm_120a).

  Measured on a 12 GB laptop GPU:
  - a 30B-A3B Mamba2-MoE hybrid packs to **3.85 GiB at ~1 bit per weight** (119.9 GiB in fp32) and serves **16 concurrent streams at 170+ tok/s**;
  - a d=16384 block wider than VRAM trains through the offload path in under 10 GB.
- **`no_std` core.** `axonml-core` and `axonml-tensor` build for `no_std` + `alloc`, checked in CI on `thumbv7em-none-eabihf`. The default `std` build is unchanged.
- **Soundness.**
  - Zero undocumented `unsafe`, enforced in CI.
  - Every CUDA launcher asserts its host-side preconditions, and the device copies are bounds-checked.
  - `tools/check_launches.py` verifies all 198 kernel launches, 153 in core and 45 in subbit.
- **wgpu 29** WebGPU backend, gated in CI.
- **Device-native CPU parallelism.** Every matmul layout and the full GradFn backward family are rayon-threaded, threshold-gated so small ops stay serial with identical results.
- **CLI / TUI.**
  - `axonml train` multi-modal fusion and ensembles (`--data-b`, `--branches`, `--strategy concat|gated|moe|late-ensemble`).
  - `axonml tui --log metrics.jsonl` opens the training view on a live log.

### Core

- **Tensor Operations** (`axonml-tensor`)
  - N-dimensional tensors with arbitrary shapes
  - Automatic broadcasting following NumPy rules
  - Efficient views and slicing (zero-copy where possible)
  - Arithmetic operations (+, -, *, /, matmul)
  - Reduction operations (sum, mean, max, min, prod)
  - Sorting operations (sort, argsort, topk)
  - Indexing operations (gather, scatter, nonzero, unique)
  - Shape operations (flip, roll, squeeze, unsqueeze, permute)
  - Activation functions (ReLU, Sigmoid, Tanh, Softmax, GELU, SiLU, ELU, LeakyReLU)
  - Sparse tensor support (COO format)
  - **Lazy Tensor Computation** *(novel)* - Deferred execution with algebraic optimization (constant folding, identity elimination, inverse cancellation, scalar folding) — built into the tensor type, no external JIT needed

- **Automatic Differentiation** (`axonml-autograd`)
  - Dynamic computational graph
  - Reverse-mode autodiff (backpropagation)
  - Gradient functions for all operations
  - `no_grad` context manager
  - **Automatic Mixed Precision (AMP)** - autocast context for F16 training
  - **Gradient Checkpointing** - trade compute for memory
  - **Graph Inspection API** *(novel)* - Native computation graph visualization and analysis (trace_backward, DOT export, node/depth/leaf counting, gradient flow summary) — no external tools needed (unlike PyTorch's torchviz)

- **Neural Networks** (`axonml-nn`)
  - Module trait with train/eval modes
  - Linear, Conv1d/2d, MaxPool, AvgPool, AdaptiveAvgPool
  - **BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d**
  - Dropout
  - RNN, LSTM, GRU (with cell variants)
  - MultiHeadAttention, Embedding
  - Loss functions (MSE, CrossEntropy, BCE, BCEWithLogits, L1, SmoothL1, NLL)
  - Parameter initialization (Xavier, Kaiming, Orthogonal, etc.)
  - **Differentiable Structured Sparsity** *(novel)* - `SparseLinear` with learnable pruning masks via soft thresholding, `GroupSparsity` regularization, and `LotteryTicket` hypothesis implementation — the pruning mask is differentiable, enabling end-to-end learning of which weights to prune

- **Optimizers** (`axonml-optim`)
  - SGD with momentum and Nesterov
  - Adam, AdamW, RMSprop
  - **LAMB** - Layer-wise Adaptive Moments for large batch training
  - **GradScaler** - Gradient scaling for mixed precision
  - LR Schedulers (Step, Cosine, OneCycle, Warmup, ReduceLROnPlateau, MultiStep, Exponential)
  - **Training Health Monitor** *(novel)* - Real-time training diagnostics: NaN/gradient explosion/vanishing detection, loss trend analysis (decreasing/stable/increasing/oscillating), dead neuron tracking, convergence detection, automatic learning rate suggestions — the optimizer monitors its own health

- **Data Loading** (`axonml-data`)
  - Dataset trait and DataLoader
  - Batching and shuffling
  - Sequential and random samplers

- **Computer Vision** (`axonml-vision`)
  - Image transforms (Resize, Crop, Flip, Normalize)
  - MNIST/CIFAR loaders (real data + synthetic), COCO + WIDER FACE
  - LeNet, SimpleCNN, ResNet-18/34, VGG-11/13/16/19, Vision Transformer
  - **Object Detection** — `BlazeFace` and `RetinaFace` (face detection), `DETR` (transformer-based), `NanoDet` (mobile-class), plus the shared `FPN` feature-pyramid neck
  - **Dense prediction / anomaly / VQA** — `DPT` + `FastDepth` (monocular depth), `PatchCore` + `StudentTeacher` (anomaly detection), `VQAModel` (visual question answering)
  - **Object Detection Training Infrastructure** *(novel)*
    - Image I/O: `load_image`, `load_image_resized`, `rgb_bytes_to_tensor` (CHW, [0,1] normalized)
    - Dataset loaders: `CocoDataset` (COCO JSON, category remapping), `WiderFaceDataset` (WIDER FACE annotations)
    - Detection losses: `FocalLoss`, `GIoULoss`, `UncertaintyLoss`, `compute_centerness`
    - Anchor-free target assignment: `assign_fcos_targets` (multi-scale FCOS) and `assign_single_scale_targets` (single-scale)
    - Evaluation: `compute_ap`, `compute_map`, `compute_coco_map` (AP/mAP at IoU thresholds)

- **Audio Processing** (`axonml-audio`)
  - MelSpectrogram, MFCC transforms
  - Resample, Normalize, AddNoise
  - SyntheticCommandDataset, SyntheticMusicDataset

- **NLP Utilities** (`axonml-text`)
  - Tokenizers (Whitespace, Char, BPE)
  - Vocabulary management
  - SyntheticSentimentDataset

- **Distributed Training** (`axonml-distributed`)
  - **DistributedDataParallel (DDP)** - Data parallelism across GPUs
  - **Fully Sharded Data Parallel (FSDP)** - ZeRO-2/ZeRO-3 memory optimization
  - **Pipeline Parallelism** - Model sharding across devices with microbatching
  - **Tensor Parallelism** - Layer-wise model parallelism
  - All-reduce, broadcast, barrier, send/recv collective operations
  - Process group management with multiple backends

- **Model Serialization** (`axonml-serialize`)
  - Save/load models in multiple formats
  - Checkpoint management for training
  - StateDict (PyTorch-compatible concept)
  - SafeTensors format support
  - `.axonml` bundles carry the model's exact compute graph (DAG forward tracer, skip connections included)
  - Optimizer state export/import (`Adam::export_state` / `import_state`) for lossless pause and resume

- **ONNX Import/Export** (`axonml-onnx`)
  - Load ONNX models for inference
  - Export Axonml models to ONNX format
  - 40+ ONNX operators supported
  - ONNX opset version 17

- **Model Quantization** (`axonml-quant`)
  - INT8 (Q8_0), INT4 (Q4_0, Q4_1, Q4_K), INT5 (Q5_0, Q5_1), INT6 (Q6_K) formats
  - Half-precision (F16) support
  - Block-based quantization with calibration
  - **BitNet I2_S 1.58-bit ternary** — `matmul_i2s_i8` with AVX-VNNI fused dequant, int8 activation quantizer, 128-weight blocks; ~16x compression for sub-2-bit LLM weights
  - ~8x model size reduction with Q4, ~16x with I2_S

- **Kernel Fusion** (`axonml-fusion`)
  - Automatic fusion pattern detection
  - FusedLinear (MatMul + Bias + Activation)
  - FusedElementwise operation chains
  - Up to 2x speedup for memory-bound operations

- **Command Line Interface** (`axonml-cli`)
  - Complete CLI for ML workflows
  - Real training with axonml components
  - Weights & Biases integration for experiment tracking
  - Model conversion and export

- **Terminal User Interface** (`axonml-tui`)
  - Interactive terminal-based dashboard
  - Model architecture visualization
  - Real-time training progress monitoring
  - Dataset statistics and graphs
  - File browser for models and datasets

- **Web Dashboard** (`axonml-dashboard`)
  - Modern Leptos/WASM web frontend
  - Real-time training monitoring with WebSocket
  - Model registry and version management
  - Inference endpoint deployment
  - Multi-factor authentication (TOTP, WebAuthn)

- **API Server** (`axonml-server`)
  - Axum-based REST API backend
  - JWT authentication with refresh tokens
  - Training run management
  - Model registry and deployment
  - WebSocket terminal (PTY) for in-browser shell access
  - Prometheus metrics export

### Axonml CLI

The Axonml CLI provides a unified command-line interface for the entire ML workflow:

```bash
# Server Sync (CLI ↔ Webapp Integration)
axonml login                           # Login to AxonML server
axonml login --server http://server:3021  # Login to custom server
axonml logout                          # Logout and clear credentials
axonml sync                            # Check sync status with server
axonml sync --full                     # Full sync of training runs, models, datasets

# Project Management
axonml new my-model                    # Scaffold new project
axonml init                            # Initialize in existing directory
axonml scaffold my-project             # Generate Rust training project

# Training (with real axonml integration)
axonml train config.toml               # Train from config file
axonml train --model mlp --epochs 10   # Quick training
axonml resume checkpoint.axonml       # Resume from checkpoint

# Evaluation & Inference
axonml eval model.axonml --data test/ # Evaluate model metrics
axonml predict model.axonml input.json # Run inference

# Model Management
axonml convert pytorch.pth             # Convert PyTorch models
axonml export model.axonml --onnx     # Export to ONNX
axonml inspect model.axonml           # Inspect architecture
axonml rename model.axonml new-name   # Rename model files

# Quantization
axonml quant convert model.axonml --type q8_0   # Quantize to Q8
axonml quant convert model.pth --type q4_0       # PyTorch → Quantized Axonml
axonml quant info model.axonml                  # Show quantization info
axonml quant benchmark model.axonml             # Benchmark quantized model
axonml quant list                                # List supported formats

# Workspace Management
axonml load model model.axonml        # Load model into workspace
axonml load data ./dataset             # Load dataset into workspace
axonml load both --model m.f --data d/ # Load both
axonml load status                     # Show workspace status
axonml load clear                      # Clear workspace

# Analysis & Reports
axonml analyze model                   # Analyze loaded model
axonml analyze data                    # Analyze loaded dataset
axonml analyze both                    # Analyze both
axonml analyze report --format html    # Generate analysis report

# Data Management
axonml data info ./dataset             # Dataset information
axonml data validate ./dataset         # Validate dataset format
axonml data split ./data --train 0.8   # Split dataset

# Bundling & Deployment
axonml zip create -o bundle.zip --model m.f --data d/  # Create bundle
axonml zip extract bundle.zip -o ./output              # Extract bundle
axonml zip list bundle.zip                             # List bundle contents
axonml upload model.axonml --hub myrepo               # Upload to model hub
axonml serve model.axonml --port 8080                 # Start inference server

# Benchmarking
axonml bench model model.axonml             # Benchmark model performance
axonml bench inference model.axonml         # Test batch size scaling
axonml bench compare model1.f,model2.f       # Compare multiple models
axonml bench hardware                        # CPU/memory benchmarks

# GPU Management
axonml gpu list                              # List available GPUs
axonml gpu info                              # Detailed GPU information
axonml gpu select 0                          # Select GPU for training
axonml gpu bench                             # GPU compute benchmarks
axonml gpu memory                            # Show GPU memory usage
axonml gpu status                            # Current GPU status

# Pretrained Model Hub
axonml hub list                              # List available pretrained models
axonml hub info resnet50                     # Show model details
axonml hub download resnet50                 # Download pretrained weights
axonml hub cached                            # Show cached models
axonml hub clear                             # Clear all cached weights

# Kaggle Integration
axonml kaggle login <username> <key>         # Save Kaggle API credentials
axonml kaggle status                         # Check authentication status
axonml kaggle search "image classification"  # Search datasets
axonml kaggle download owner/dataset         # Download dataset
axonml kaggle list                           # List downloaded datasets

# Dataset Management
axonml dataset list                          # List available datasets
axonml dataset list --source kaggle          # List from specific source
axonml dataset info mnist                    # Show dataset details
axonml dataset search "classification"       # Search datasets
axonml dataset download cifar-10             # Download dataset
axonml dataset sources                       # List data sources

# Dashboard & Server Management
axon start                                   # Start dashboard + API server
axon start --server                          # Start only API server on :3000
axon start --dashboard                       # Start only dashboard on :8080
axon stop                                    # Stop all services
axon status                                  # Check service status
axon logs -f                                 # Follow logs in real-time
```

### Weights & Biases Integration

Built-in experiment tracking with W&B:

```bash
# Configure W&B
axonml wandb login
axonml wandb init --project my-project

# Training automatically logs to W&B
axonml train config.toml --wandb
```

Features:
- Automatic metric logging (loss, accuracy, learning rate)
- Hyperparameter tracking
- Model checkpointing with W&B artifacts
- Real-time training visualization

### Axonml TUI

The Axonml TUI provides an interactive terminal-based dashboard for ML development:

```bash
# Launch the TUI
axonml tui

# Load a model on startup
axonml tui --model path/to/model.axonml

# Load a dataset on startup
axonml tui --data path/to/dataset/

# Load both
axonml tui --model model.axonml --data ./data/
```

**Views:**
- **Model** - Neural network architecture visualization (layers, shapes, parameters)
- **Data** - Dataset statistics, class distributions, sample preview
- **Training** - Real-time epoch/batch progress, loss/accuracy metrics
- **Graphs** - Loss curves, accuracy curves, learning rate schedule
- **Files** - File browser for models and datasets
- **Help** - Keyboard shortcuts reference

**Keyboard Navigation:**
| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Switch between tabs |
| `1-6` | Jump directly to tab |
| `↑/k`, `↓/j` | Navigate up/down in lists |
| `←/h`, `→/l` | Navigate between panels |
| `Enter` | Select / Open |
| `?` | Show help overlay |
| `q` | Quit |

### Web Dashboard

The AxonML Web Dashboard provides a modern browser-based interface for ML operations:

```bash
# Start the full stack (dashboard + API server)
axon start

# Start only the API server
axon start --server --port 3000

# Start only the dashboard
axon start --dashboard --dashboard-port 8080

# Check status
axon status

# View logs
axon logs -f
```

**Features:**
- **Dashboard Overview** - Real-time stats on training runs, models, and endpoints
- **Training Runs** - Start, monitor, and manage training with live metrics
- **Model Registry** - Upload, version, and manage trained models
- **Inference Endpoints** - Deploy models for serving predictions
- **In-App Terminal** - Slide-out terminal with WebSocket PTY for server-side commands
- **Settings** - User profile, security settings, MFA configuration

**Authentication:**
- JWT-based authentication with refresh tokens
- Multi-factor authentication (TOTP authenticator apps)
- WebAuthn support for hardware security keys
- Recovery codes for account recovery

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    axonml-dashboard                          │
│              Leptos/WASM Frontend (CSR)                      │
├─────────────────────────────────────────────────────────────┤
│  Dashboard │ Training │ Models │ Inference │ Settings       │
└─────────────────────────────────────────────────────────────┘
                              │
                         HTTP/WebSocket
                              │
┌─────────────────────────────────────────────────────────────┐
│                      axonml-server                           │
│                    Axum REST + WS API                        │
├─────────────────────────────────────────────────────────────┤
│  Auth  │  Training  │  Models  │  Inference  │  Metrics     │
└─────────────────────────────────────────────────────────────┘
```

- **Pretrained Model Hub** (`axonml-vision/hub`)
  - Download pretrained weights (ResNet, VGG)
  - Local caching in ~/.cache/axonml/hub/
  - StateDict for named tensor storage
  - CLI: `axonml hub list/info/download/cached/clear`

- **Kaggle Integration** (`axonml-cli`)
  - Kaggle API authentication
  - Dataset search and download
  - CLI: `axonml kaggle login/status/search/download/list`

- **Dataset Management** (`axonml-cli`)
  - Dataset bridge API integration
  - Built-in datasets (MNIST, CIFAR, Iris, Wine, etc.)
  - Multiple data sources (Kaggle, UCI, data.gov)
  - CLI: `axonml dataset list/info/search/download/sources`

- **JIT Compilation** (`axonml-jit`)
  - Automatic forward capture (`JitFn::trace_forward`): records a model's real `forward`, no tracer DSL
  - Elementwise fusion pass, with fused chains (including two-input chains) compiled at runtime through NVRTC and dispatched on-device
  - Intermediate representation for computation graphs
  - Graph optimization (constant folding, DCE, CSE)
  - Function caching for compiled graphs
  - Cranelift foundation for native codegen

- **Sub-bit GPU Primitives** (`axonml-subbit`, `cuda` feature)
  - Fused VQ matmul straight off packed codebook indices (dim-2/4/8 codes, f16 or f32 codebooks), with row- and column-blocked GEMV for decode
  - Grouped MoE experts (`VqGroupedExperts`) with device-side router top-k (`RouterSel`) and combine
  - Resident per-tile index/scale assignment (`ResidentAssign`) and offload tiles for training a block wider than VRAM
  - Batched multi-sequence decode for Mamba2 + MoE hybrids with per-slot recurrent state
  - NVFP4 training path for Blackwell (sm_120a)
  - Built only on the core crates' public APIs (`SubbitTensorExt`, `SubbitBackendExt`, `SubbitEmbeddingExt`); loads its own PTX
  - Switches: `AXONML_TF32`, `AXONML_FP4`, `AXONML_FP4_BWD`, `AXONML_VQ_WARP_ROWS`

- **Profiling Tools** (`axonml-profile`)
  - Core Profiler with ProfileGuard and ProfileReport
  - MemoryProfiler for allocation tracking
  - ComputeProfiler for operation timing
  - TimelineProfiler with Chrome trace export
  - BottleneckAnalyzer for automatic issue detection

- **LLM Architectures** (`axonml-llm`) — 7 full architectures
  - **BERT** encoder + `BertForSequenceClassification` / `BertForMaskedLM`
  - **GPT-2** decoder + `GPT2LMHead` for language modeling
  - **LLaMA** (2-7B, 2-13B, 3-8B configs) with GQA + RoPE + RMSNorm
  - **Mistral** (7B, Mixtral 8×7B configs) with sliding-window attention
  - **Phi** (1/2/3-mini configs); full-RoPE workaround for partial-RoPE framework bug documented in `train_phi`
  - **SSM / Mamba** + `SSMForCausalLM` wrapper
  - **Qwen3** (trainable) with QK-norm, tied LM head (`tie_lm_head`); teacher/student + distillation
  - **Fine-tuning:** LoRA (`attach_lora` / `freeze_lora_base` / `merge_lora`, upper-layer-only `attach_lora_from`) and per-layer activation checkpointing (`AXONML_CKPT_LAYERS`) for LLaMA and Qwen3
  - **GGUF export:** F16 by default, or Q8_0 body weights with `AXONML_GGUF_QUANT=q8_0`
  - Shared infra: `FlashAttention`, `KVCache` / `LayerKVCache`, HuggingFace loader, state-dict mapping, `HFTokenizer`
  - Text generation with top-k, top-p, temperature sampling

- **GPU Backends** (`axonml-core`)
  - **CUDA**: full NVIDIA GPU support with cuBLAS and PTX kernels. Opt-in TF32 tensor cores (`AXONML_TF32`), a stream-ordered memory pool with completion fences, and CUDA-graph capture/replay (`ReplayGraph`)
  - **Vulkan**: cross-platform GPU compute
  - **Metal**: Apple Silicon optimization
  - **WebGPU** (wgpu 29): browser-based GPU acceleration
  - **GPU test suite**: correctness testing against a CPU reference
  - **`no_std`**: `axonml-core` and `axonml-tensor` build for `no_std` + `alloc` (bare-metal targets)

- **Model Hub & Benchmarking** (`axonml`)
  - **Unified Model Hub** - Combined vision/LLM model registry
  - **Model Benchmarking** - Throughput testing, memory profiling
  - **Pretrained Weights** - ResNet, VGG, MobileNet, EfficientNet, BERT, GPT-2

### Planned

- Real-time model serving with batched inference
- Self-hosted pretrained weight hosting

## Quick Start

Add Axonml to your `Cargo.toml`:

```toml
[dependencies]
axonml = "0.7"
```

If 0.7 has not reached crates.io yet, use the repository directly:

```toml
[dependencies]
axonml = { git = "https://github.com/AutomataNexus/AxonML", tag = "v0.7.0" }
```

### Basic Usage

```rust
use axonml::prelude::*;

fn main() {
    // Create tensors
    let a = zeros::<f32>(&[2, 3]);
    let b = ones::<f32>(&[2, 3]);

    // Arithmetic operations with broadcasting
    let c = &a + &b;
    let d = &c * 2.0;

    // Matrix operations
    let e = randn::<f32>(&[3, 4]);
    let f = randn::<f32>(&[4, 5]);
    let g = e.matmul(&f).unwrap();

    // Reductions
    let sum = d.sum();
    let mean = d.mean().unwrap();

    // Activations
    let h = randn::<f32>(&[10]);
    let activated = h.relu();

    println!("Result shape: {:?}", g.shape());
}
```

### Training Example

```rust
use axonml::prelude::*;
use axonml_nn::{Sequential, Linear, ReLU, CrossEntropyLoss, Module};
use axonml_optim::{Adam, Optimizer};
use axonml_data::{DataLoader, Dataset};

fn main() {
    // Build model
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));

    // Setup optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    // Training loop
    for epoch in 0..10 {
        for batch in dataloader.iter() {
            let output = model.forward(&batch.data);
            let loss = CrossEntropyLoss::new().compute(&output, &batch.targets);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }
}
```

### Tensor Creation

```rust
use axonml::prelude::*;

// Zeros and ones
let z = zeros::<f32>(&[2, 3, 4]);
let o = ones::<f64>(&[5, 5]);

// Random tensors
let r = rand::<f32>(&[10, 10]);      // Uniform [0, 1)
let n = randn::<f32>(&[10, 10]);     // Normal(0, 1)
let u = uniform::<f32>(&[5], -1.0, 1.0);

// Ranges
let a = arange::<f32>(0.0, 10.0, 1.0);
let l = linspace::<f32>(0.0, 1.0, 100);

// From data
let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

// Special matrices
let eye = eye::<f32>(4);
let diag = diag(&[1.0, 2.0, 3.0]);
```

### Shape Operations

```rust
use axonml::prelude::*;

let t = randn::<f32>(&[2, 3, 4]);

// Reshape
let r = t.reshape(&[6, 4]).unwrap();
let f = t.flatten();

// Transpose
let p = t.permute(&[2, 0, 1]).unwrap();

// Squeeze/Unsqueeze
let s = t.unsqueeze(0).unwrap();  // Add dimension
let u = s.squeeze(Some(0)).unwrap();  // Remove dimension

// Views
let v = t.slice_dim0(0, 1).unwrap();
let n = t.narrow(1, 0, 2).unwrap();
```

## Production Edge Deployment

AxonML powers real-time predictive maintenance on HVAC systems across commercial buildings. Per-site model pairs — an LSTM autoencoder for anomaly detection plus a GRU failure predictor — run live inference on Raspberry Pi edge controllers, processing sensor data at 1 Hz. Each pair is compact (≈105K–416K params, ~2–3 MB RSS), so a single Pi hosts an entire mechanical room or air-handler bank.

| Unit type | Anomaly Detector | Failure Predictor | Params | RSS |
|-----------|------------------|-------------------|--------|-----|
| Mechanical room | LSTM-AE | GRU-FDD | ~416K | ~2.5 MB |
| Air handler (AHU) | LSTM-AE | GRU-FDD | ~105–233K | ~2.1–2.4 MB |

**Stack:** AxonML training (CPU) → `.axonml` model files → cross-compiled ARM inference daemons (`armv7-unknown-linux-musleabihf`) → managed services on Raspberry Pi → local REST API

Each daemon runs pure-tensor inference (no autograd overhead), polls the local edge controller for sensor data, maintains rolling time-series buffers, and exposes anomaly scores + failure predictions via HTTP.

## Architecture

```
+------------------------------------------------------------------+
|                        axonml (main crate)                       |
+------------------------------------------------------------------+
|  axonml-train  |  axonml-vision  |  axonml-audio  |
+----------------+----------------+-----------------+----------------+
|  axonml-text   | axonml-distributed |  axonml-llm  |  axonml-jit  |
+----------------+--------------------+--------------+---------------+
|         axonml-profile         |         axonml-serialize         |
+--------------------------------+-----------------------------------+
|         axonml-onnx            |         axonml-quant             |
+--------------------------------+-----------------------------------+
|          axonml-fusion         |          axonml-subbit           |
+--------------------------------+-----------------------------------+
|                           axonml-data                             |
+--------------------------------------------------------------------+
|          axonml-optim          |           axonml-nn              |
+--------------------------------+-----------------------------------+
|                          axonml-autograd                          |
+--------------------------------------------------------------------+
|                          axonml-tensor                            |
+--------------------------------------------------------------------+
|                           axonml-core                             |
+--------------+--------------+--------------+--------------+---------+
|   CPU/BLAS   |    CUDA      |   Vulkan     |    Metal     | WebGPU  |
+--------------+--------------+--------------+--------------+---------+

+--------------------------------------------------------------------+
|                           axonml-cli                              |
|     Project scaffolding, Training, Evaluation, W&B integration     |
+--------------------------------------------------------------------+
|                           axonml-tui                              |
|  Interactive terminal dashboard for models, data, training graphs  |
+--------------------------------------------------------------------+
|                        axonml-dashboard                            |
|  Leptos/WASM Web UI: Training, Models, Inference, Settings         |
+--------------------------------------------------------------------+
|                         axonml-server                              |
|  Axum REST API: Auth, Training Runs, Model Registry, Metrics       |
+--------------------------------------------------------------------+

+--------------------------------------------------------------------+
|                           llm-training                             |
|     LM training binaries + lifecycle control (pause/resume/stop)   |
+--------------------------------------------------------------------+
```

## Building from Source

### Requirements

- Rust 1.85 or later
- Cargo
- Node.js (for PM2 process management)
- A document-store database backend

### Build

```bash
git clone https://github.com/automatanexus/axonml
cd axonml
cargo build --release
```

### Install CLI

```bash
cargo install --path crates/axonml-cli
```

### Run Tests

```bash
cargo test
```

### Run Benchmarks

```bash
cargo bench
```

## Server Deployment

### PM2 Process Management

AxonML server is managed via PM2 for automatic restarts and boot persistence.

```bash
# First-time setup
cargo build --release -p axonml-server    # Build release binary
sudo mkdir -p /var/log/axonml             # Create log directory
sudo chown $USER:$USER /var/log/axonml

# Initialize database (creates collections + users)
./AxonML_DB_Init.sh --with-user

# Start with PM2
pm2 start ecosystem.config.js
pm2 save                                   # Save process list
pm2 startup                                # Enable boot persistence

# Management
pm2 status                                 # Check status
pm2 logs axonml-server                     # View logs
pm2 restart axonml-server                  # Restart server
pm2 stop axonml-server                     # Stop server
```

### Database Initialization

AxonML uses a document-store database backend.

```bash
# Initialize database (run once or to reinitialize)
./AxonML_DB_Init.sh                        # Basic setup with admin user
./AxonML_DB_Init.sh --with-user            # Also creates DevOps admin user

# Default Users (0.6.1+)
# Admin:  admin@axonml.local  — password is a 24-char cryptographic random,
#         written to $TMPDIR/axonml-admin-password.txt at first boot.
#         Check the server log for the exact path and rotate it ASAP.
# DevOps: DevOps@automatanexus.com — only created if AXONML_DEVOPS_PASSWORD is set.
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |
| `BACKEND_URL` | `http://127.0.0.1:7001` | Database backend connection URL |
| `RESEND_API_KEY` | - | Email service API key |
| `AXONML_JWT_SECRET` | - | JWT signing secret (**required**, must be ≥32 chars; server refuses to boot otherwise) |
| `AXONML_DEVOPS_PASSWORD` | - | If set, seeds the DevOps@automatanexus.com admin on first boot |
| `VAULT_ADDR` | - | If set, enables HashiCorp Vault secrets backend (takes precedence over env) |

## Project Structure

```
Axonml/
├── Cargo.toml              # Workspace configuration
├── README.md               # This file
├── LICENSE-MIT             # MIT license
├── LICENSE-APACHE          # Apache 2.0 license
├── CONTRIBUTING.md         # Contribution guidelines
├── CHANGELOG.md            # Version history
├── COMMERCIAL.md           # Commercial licensing info
├── crates/
│   ├── axonml-core/       # Device, storage, dtypes, GPU backends (CPU/CUDA/Vulkan/Metal/WebGPU)
│   ├── axonml-tensor/     # Tensor ops (+ lazy tensor, sparse COO, CUDA ops)
│   ├── axonml-autograd/   # Automatic differentiation (+ AMP, checkpointing, graph inspection)
│   ├── axonml-nn/         # Neural network modules (+ ternary BitNet b1.58, MoE, sparse, graph, FFT)
│   ├── axonml-optim/      # Optimizers & schedulers (+ LAMB, GradScaler, health monitor)
│   ├── axonml-data/       # Data loading
│   ├── axonml-vision/     # Computer vision (ResNet/VGG/ViT/BlazeFace/RetinaFace/DETR/NanoDet/depth/anomaly/VQA)
│   ├── axonml-audio/      # Audio processing (MelSpectrogram, MFCC, pitch/time stretch)
│   ├── axonml-text/       # NLP utilities (BPE / WordPiece / Unigram tokenizers)
│   ├── axonml-distributed/# Distributed training (DDP, FSDP, tensor + pipeline parallel, NCCL)
│   ├── axonml-serialize/  # Model serialization (+ safetensors)
│   ├── axonml-onnx/       # ONNX import/export
│   ├── axonml-quant/      # Model quantization (+ BitNet I2_S 1.58-bit)
│   ├── axonml-fusion/     # Kernel fusion optimization
│   ├── axonml-subbit/     # Sub-bit VQ GPU primitives (fused VQ matmul, MoE experts, offload tiles, NVFP4)
│   ├── axonml-jit/        # JIT compilation (forward capture, NVRTC fusion, Cranelift)
│   ├── axonml-profile/    # Profiling tools
│   ├── axonml-llm/        # 7 LLM architectures (BERT, GPT-2, LLaMA, Mistral, Phi, SSM, Qwen3)
│   ├── axonml-train/      # Training glue (Trainer, callbacks, benchmarks)
│   ├── axonml-cli/        # Command line interface
│   ├── axonml-tui/        # Terminal user interface
│   ├── axonml-dashboard/  # Leptos/WASM web dashboard
│   ├── axonml-server/     # Axum API server
│   └── axonml/            # Main umbrella crate
├── llm-training/          # LM training binaries + train_ctl + TrainingLifecycle (pause/resume/stop)
├── tools/                 # check_launches.py (CUDA launch audit), ONNX model converter
├── docs/                  # Per-module documentation (Jekyll-rendered)
└── crates/axonml/examples/# Working examples
    ├── simple_training.rs # XOR with MLP
    ├── mnist_training.rs  # CNN on MNIST
    └── nlp_audio_test.rs  # Text & audio demo
```

## Documentation

- [API Documentation](docs/): per-module documentation
- [Object Detection Training](docs/detection.md): detection training guide (RetinaFace, BlazeFace, COCO, WIDER FACE)
- [Examples](crates/axonml/examples/): working code examples
- [Changelog](CHANGELOG.md): version history

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Test Suite

The framework includes **1,773 test functions** across all crates:

```bash
cargo test --workspace                                  # CPU
cargo test --workspace --features axonml-core/cuda,axonml-tensor/cuda   # CUDA
python3 tools/check_launches.py                         # audit every CUDA kernel launch
```

| Crate | Test functions |
|-------|-------|
| axonml-nn | 243 |
| axonml-vision | 224 |
| axonml-server | 152 |
| axonml-autograd | 151 |
| axonml-tensor | 122 |
| axonml-cli | 114 |
| axonml-optim | 100 |
| axonml-core | 97 |
| axonml-llm | 97 |
| axonml-distributed | 90 |
| axonml-data | 63 |
| axonml-serialize | 44 |
| axonml-quant | 43 |
| axonml-text | 39 |
| axonml-jit | 35 |
| axonml-fusion | 30 |
| axonml-audio | 28 |
| axonml-profile | 27 |
| axonml-train | 25 |
| axonml-onnx | 23 |
| axonml (umbrella) | 17 |
| axonml-tui | 9 |

Test-function counts per crate (`#[test]`), as of 0.7.0. Parameterised and doc tests add to the runtime total.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- Inspired by [PyTorch](https://pytorch.org/)
- Built with learnings from [Burn](https://github.com/tracel-ai/burn), [Candle](https://github.com/huggingface/candle), and [dfdx](https://github.com/coreylowman/dfdx)
- W&B integration inspired by [Weights & Biases](https://wandb.ai/)

---

**Axonml** - Forging the future of ML in Rust.

_Last updated: 2026-10-10 (v0.7.0)_
