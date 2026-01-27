---
layout: default
title: Crate Documentation
nav_order: 10
has_children: true
description: "Rust crate architecture and documentation"
---

# Crate Documentation
{: .no_toc }

AxonML is built as a Rust workspace with **22 specialized crates**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                             │
├─────────────┬─────────────┬─────────────┬───────────────────────────┤
│ axonml-cli  │axonml-server│ axonml-tui  │    axonml-dashboard       │
│  (Binary)   │  (REST API) │ (Terminal)  │      (WASM Web UI)        │
├─────────────┴─────────────┴─────────────┴───────────────────────────┤
│                           axonml                                     │
│                    (Umbrella Crate - Feature Flags)                  │
├─────────────────────────────────────────────────────────────────────┤
│                         Domain Layer                                 │
├─────────────┬─────────────┬─────────────┬───────────────────────────┤
│axonml-vision│ axonml-audio│ axonml-text │      axonml-llm           │
│  (CV/CNN)   │   (MFCC)    │(Tokenizers) │    (BERT, GPT-2)          │
├─────────────┴─────────────┴─────────────┴───────────────────────────┤
│                       Training Layer                                 │
├─────────────┬─────────────┬─────────────┬───────────────────────────┤
│  axonml-nn  │axonml-optim │ axonml-data │   axonml-distributed      │
│  (Modules)  │(Adam, LAMB) │(DataLoader) │    (DDP, FSDP)            │
├─────────────┴─────────────┴─────────────┴───────────────────────────┤
│                      Optimization Layer                              │
├─────────────┬─────────────┬─────────────┬───────────────────────────┤
│axonml-quant │axonml-fusion│axonml-jit   │    axonml-profile         │
│ (INT8/INT4) │(Kernel Fuse)│(Cranelift)  │     (Profiling)           │
├─────────────┴─────────────┴─────────────┴───────────────────────────┤
│                     Serialization Layer                              │
├─────────────────────────────┬───────────────────────────────────────┤
│      axonml-serialize       │           axonml-onnx                  │
│    (SafeTensors, Bincode)   │       (ONNX Import/Export)            │
├─────────────────────────────┴───────────────────────────────────────┤
│                      Computation Layer                               │
├─────────────────────────────────────────────────────────────────────┤
│                        axonml-autograd                               │
│           (Dynamic Computational Graph, AMP, Checkpointing)          │
├─────────────────────────────────────────────────────────────────────┤
│                        axonml-tensor                                 │
│        (N-dimensional Arrays, Broadcasting, BLAS Operations)         │
├─────────────────────────────────────────────────────────────────────┤
│                         axonml-core                                  │
│         (Device Abstraction, DType, Storage, Memory Management)      │
│              CPU │ CUDA │ Vulkan │ Metal │ WebGPU                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Crate Summary

### Foundation Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-core](https://docs.rs/axonml-core) | Device abstraction, data types, storage | `Device`, `DType`, `Storage` |
| [axonml-tensor](https://docs.rs/axonml-tensor) | N-dimensional tensor operations | `Tensor`, `TensorView` |

### Computation Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-autograd](https://docs.rs/axonml-autograd) | Automatic differentiation engine | `Variable`, `GradFn`, `ComputationGraph` |

### Training Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-nn](https://docs.rs/axonml-nn) | Neural network modules | `Module`, `Linear`, `Conv2d`, `Attention` |
| [axonml-optim](https://docs.rs/axonml-optim) | Optimizers and LR schedulers | `Adam`, `SGD`, `LAMB`, `CosineAnnealingLR` |
| [axonml-data](https://docs.rs/axonml-data) | Data loading and batching | `DataLoader`, `Dataset`, `Sampler` |
| [axonml-distributed](https://docs.rs/axonml-distributed) | Distributed training | `DDP`, `FSDP`, `ProcessGroup` |

### Domain Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-vision](https://docs.rs/axonml-vision) | Computer vision utilities | `MNIST`, `CIFAR10`, `ResNet`, `transforms` |
| [axonml-audio](https://docs.rs/axonml-audio) | Audio processing | `MelSpectrogram`, `MFCC` |
| [axonml-text](https://docs.rs/axonml-text) | Text processing | `Tokenizer`, `BPE`, `Vocabulary` |
| [axonml-llm](https://docs.rs/axonml-llm) | Large language models | `BERT`, `GPT2`, `Transformer` |

### Serialization Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-serialize](https://docs.rs/axonml-serialize) | Model serialization | `StateDict`, `SafeTensors` |
| [axonml-onnx](https://docs.rs/axonml-onnx) | ONNX import/export | `OnnxModel`, `OnnxGraph` |

### Optimization Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-quant](https://docs.rs/axonml-quant) | Model quantization | `QuantizedTensor`, `INT8`, `INT4` |
| [axonml-fusion](https://docs.rs/axonml-fusion) | Kernel fusion optimization | `FusedOp`, `FusionPass` |
| [axonml-jit](https://docs.rs/axonml-jit) | JIT compilation | `JitContext`, `CompiledKernel` |
| [axonml-profile](https://docs.rs/axonml-profile) | Profiling tools | `Profiler`, `MemoryStats` |

### Application Layer

| Crate | Description | Binary |
|:------|:------------|:-------|
| [axonml](https://docs.rs/axonml) | Umbrella crate (re-exports) | - |
| [axonml-cli](https://docs.rs/axonml-cli) | Command-line interface | `axonml` |
| [axonml-server](https://docs.rs/axonml-server) | REST API server | `axonml-server` |
| [axonml-tui](https://docs.rs/axonml-tui) | Terminal UI dashboard | - |
| axonml-dashboard | Web dashboard (WASM) | - |

## Dependency Graph

```
axonml (umbrella)
├── axonml-core
├── axonml-tensor ─────────────────┬── axonml-core
├── axonml-autograd ───────────────┼── axonml-tensor
├── axonml-nn ─────────────────────┼── axonml-autograd
├── axonml-optim ──────────────────┼── axonml-nn
├── axonml-data ───────────────────┼── axonml-tensor
├── axonml-vision ─────────────────┼── axonml-nn, axonml-data
├── axonml-audio ──────────────────┼── axonml-data
├── axonml-text ───────────────────┼── axonml-nn, axonml-data
├── axonml-distributed ────────────┼── axonml-nn
├── axonml-serialize ──────────────┼── axonml-nn
├── axonml-onnx ───────────────────┼── axonml-nn, axonml-serialize
├── axonml-llm ────────────────────┼── axonml-nn
├── axonml-jit ────────────────────┼── axonml-tensor
├── axonml-quant ──────────────────┼── axonml-tensor
├── axonml-fusion ─────────────────┼── axonml-tensor
└── axonml-profile ────────────────┴── axonml-tensor
```

## Building Individual Crates

```bash
# Build a specific crate
cargo build -p axonml-nn

# Test a specific crate
cargo test -p axonml-nn

# Generate docs for a crate
cargo doc -p axonml-nn --open

# Build with features
cargo build -p axonml-core --features "cuda"
```

## Feature Flags by Crate

### axonml-core

| Feature | Description |
|:--------|:------------|
| `std` | Standard library (default) |
| `cuda` | NVIDIA CUDA backend |
| `vulkan` | Vulkan backend |
| `metal` | Apple Metal backend |
| `wgpu` | WebGPU backend |

### axonml (umbrella)

| Feature | Description |
|:--------|:------------|
| `full` | All features (default) |
| `core` | Core tensor operations |
| `nn` | Neural networks |
| `vision` | Computer vision |
| `audio` | Audio processing |
| `text` | Text processing |
| `llm` | Large language models |
| `distributed` | Distributed training |
| `onnx` | ONNX import/export |
| `jit` | JIT compilation |
| `cuda` | CUDA acceleration |
| `wgpu` | WebGPU acceleration |

## API Documentation

Full API documentation is available on docs.rs:

- [axonml](https://docs.rs/axonml) - Main crate
- [axonml-tensor](https://docs.rs/axonml-tensor) - Tensor operations
- [axonml-nn](https://docs.rs/axonml-nn) - Neural networks
- [All crates](https://crates.io/search?q=axonml) - Search on crates.io
