---
layout: default
title: Crate Documentation
nav_order: 10
has_children: true
description: "Rust crate architecture and documentation"
---

# Crate Documentation
{: .no_toc }

AxonML is built as a Rust workspace with **24 specialized crates**.
{: .fs-6 .fw-300 }

---

## Architecture Overview

```
+---------------------------------------------------------------------+
|                        Application Layer                            |
+-------------+--------------+--------------+------------------------+
| axonml-cli  | axonml-server| axonml-tui   | axonml-dashboard        |
|  (Binary)   | (REST API)   | (Terminal)   | (WASM Web UI)           |
+-------------+--------------+--------------+------------------------+
|                              axonml                                 |
|               (Umbrella Crate — Feature Flags + Monitor)            |
+---------------------------------------------------------------------+
|                           Domain Layer                              |
+-------------+-------------+-------------+--------------+------------+
|axonml-vision|axonml-audio |axonml-text  | axonml-llm   |axonml-hvac |
|   (CV/CNN)  |(MFCC/Mel)   |(Tokenizers) |(BERT/GPT/    |(Panoptes,  |
|             |             |             | LLaMA/Trident|  Apollo,   |
|             |             |             | /Phi/Mistral)|  etc.)     |
+-------------+-------------+-------------+--------------+------------+
|                         Training Layer                              |
+-------------+-------------+-------------+--------------+------------+
|  axonml-nn  |axonml-optim |axonml-data  |axonml-train  |axonml-dist |
|  (Modules)  |(Adam/LAMB)  |(DataLoader) |(Trainer/     |(DDP, FSDP) |
|             |             |             |Benchmark/Adv)|            |
+-------------+-------------+-------------+--------------+------------+
|                       Optimization Layer                            |
+-------------+-------------+-------------+--------------+
|axonml-quant |axonml-fusion|axonml-jit   |axonml-profile|
| (INT8/INT4/ |(Kernel Fuse)|(Cranelift / |(Profiler/    |
|  Q4_K/Q6_K) |             | IR + trace) | Timeline)    |
+-------------+-------------+-------------+--------------+
|                      Serialization Layer                            |
+------------------------------+--------------------------------------+
|     axonml-serialize         |            axonml-onnx               |
|  (SafeTensors, Bincode,      |  (ONNX Import/Export, Opset 17,      |
|   StateDict, Checkpoint)     |   40+ operators)                     |
+------------------------------+--------------------------------------+
|                       Computation Layer                             |
+---------------------------------------------------------------------+
|                        axonml-autograd                              |
|        (Dynamic Graph, AMP, Checkpointing, Graph Inspect)           |
+---------------------------------------------------------------------+
|                         axonml-tensor                               |
|     (N-D Arrays, Broadcasting, Matmul, Lazy Tensors, Sparse)        |
+---------------------------------------------------------------------+
|                         axonml-core                                 |
|         (Device, DType, Storage, Memory, Backends)                  |
|              CPU | CUDA | Vulkan | Metal | WebGPU                   |
+---------------------------------------------------------------------+
```

## Crate Summary

### Foundation Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-core](https://docs.rs/axonml-core) | Device abstraction, dtypes, storage, backends | `Device`, `DType`, `Error`, `Scalar`, `Numeric`, `Float` |
| [axonml-tensor](https://docs.rs/axonml-tensor) | N-dimensional tensor operations | `Tensor<T>`, `LazyTensor`, `Shape`, `Strides` |

### Computation Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-autograd](https://docs.rs/axonml-autograd) | Automatic differentiation | `Variable`, `GradFn`, `no_grad`, `autocast`, `checkpoint` |

### Training Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-nn](https://docs.rs/axonml-nn) | Neural network modules | `Module`, `Linear`, `Conv2d`, `MultiHeadAttention`, `LSTM`, `TernaryLinear`, `SparseLinear`, `MoELayer` |
| [axonml-optim](https://docs.rs/axonml-optim) | Optimizers + LR schedulers + health monitor | `SGD`, `Adam`, `AdamW`, `LAMB`, `RMSprop`, `CosineAnnealingLR`, `OneCycleLR`, `GradScaler`, `TrainingMonitor` |
| [axonml-data](https://docs.rs/axonml-data) | Data loading and batching | `DataLoader`, `Dataset`, `RandomSampler`, `SequentialSampler`, `Transform` |
| [axonml-train](https://docs.rs/axonml-train) | High-level training glue | `TrainingConfig`, `EarlyStopping`, `ProgressLogger`, `benchmark_model`, `AdversarialTrainer` |
| [axonml-distributed](https://docs.rs/axonml-distributed) | Distributed training | `DDP`, `FSDP`, `Pipeline`, `ProcessGroup`, `World`, `ColumnParallelLinear`, `NcclBackend` |

### Domain Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-vision](https://docs.rs/axonml-vision) | Computer vision, detection, biometrics | `LeNet`, `ResNet`, `ViT`, `DETR`, `NanoDet`, `BlazeFace`, `RetinaFace`, `Nexus`, `Phantom`, `NightVision`, `AegisIdentity`, `Argus*`, `Echo*`, `Mnemosyne*`, `Aegis3D`, `CocoDataset`, `WiderFaceDataset`, `FocalLoss`, `GIoULoss` |
| [axonml-audio](https://docs.rs/axonml-audio) | Audio processing | `MelSpectrogram`, `MFCC`, `Resample`, `AddNoise`, `SyntheticCommandDataset` |
| [axonml-text](https://docs.rs/axonml-text) | NLP utilities | `WhitespaceTokenizer`, `CharTokenizer`, `BasicBPETokenizer`, `Vocab`, `TextDataset` |
| [axonml-llm](https://docs.rs/axonml-llm) | Large language models | `Bert`, `GPT2`, `LLaMA`, `Mistral`, `Phi`, `ChimeraModel`, `HydraModel`, `SSMBlock`, `TridentModel`, `RDTForCausalLM`, `Qwen3ForCausalLM`, `TextGenerator`, `HFLoader`, GGUF I2_S / `rdt` exporters |
| [axonml-hvac](https://docs.rs/axonml-hvac) | HVAC fault-detection models | `Panoptes`, `Apollo`, `Aquilo`, `Boreas`, `Colossus`, `Gaia`, `Naiad`, `Vulcan`, `Zephyrus` |

### Serialization Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-serialize](https://docs.rs/axonml-serialize) | Model serialization | `StateDict`, `Checkpoint`, `ModelBundle`, `BundleGraph` (embedded computation graph), `TensorData`, SafeTensors, Bincode |
| [axonml-onnx](https://docs.rs/axonml-onnx) | ONNX import/export (opset 17) | `OnnxModel`, `OnnxExporter`, `import_onnx`, `export_onnx` |

### Optimization Layer

| Crate | Description | Key Types |
|:------|:------------|:----------|
| [axonml-quant](https://docs.rs/axonml-quant) | Model quantization | INT8 (Q8_0), INT4 (Q4_0 / Q4_1), INT5 (Q5_0 / Q5_1), Q4_K / Q6_K GGUF blocks, F16 |
| [axonml-fusion](https://docs.rs/axonml-fusion) | Kernel fusion | `FusedLinear` (MatMul + Bias + Act), `FusedElementwise` |
| [axonml-jit](https://docs.rs/axonml-jit) | JIT compilation | `JitCompiler`, `Graph`, `CompiledFunction`, `trace`, Cranelift codegen |
| [axonml-profile](https://docs.rs/axonml-profile) | Profiling | `Profiler`, `MemoryProfiler`, `ComputeProfiler`, `TimelineProfiler`, `BottleneckAnalyzer` |

### Application Layer

| Crate | Description | Notes |
|:------|:------------|:------|
| [axonml](https://docs.rs/axonml) | Umbrella crate | Feature-gated re-exports, `TrainingMonitor` (live browser monitor) |
| [axonml-cli](https://docs.rs/axonml-cli) | Command-line interface | Binary `axonml` — scaffolding, training, eval, hub, kaggle, W&B, dataset mgmt |
| [axonml-server](https://docs.rs/axonml-server) | REST API server | Axum + JWT + PTY WebSocket; Binary `axonml-server` |
| [axonml-tui](https://docs.rs/axonml-tui) | Terminal UI | Model/data/training/graph views |
| axonml-dashboard | Leptos/WASM web UI | Dashboard served by `axon start` |

## Workspace

All 24 crates (from `Cargo.toml`):

```
axonml-core          axonml-llm
axonml-tensor        axonml-hvac
axonml-autograd      axonml-train
axonml-nn            axonml-distributed
axonml-optim         axonml-serialize
axonml-data          axonml-onnx
axonml-vision        axonml-quant
axonml-audio         axonml-fusion
axonml-text          axonml-jit
axonml-profile       axonml-cli
axonml-tui           axonml-server
axonml-dashboard     axonml               (umbrella)
```

## Dependency Graph (simplified)

```
axonml (umbrella)
├── axonml-core
├── axonml-tensor         — axonml-core
├── axonml-autograd       — axonml-tensor
├── axonml-nn             — axonml-autograd
├── axonml-optim          — axonml-nn
├── axonml-data           — axonml-tensor
├── axonml-vision         — axonml-nn, axonml-data
├── axonml-audio          — axonml-data
├── axonml-text           — axonml-nn, axonml-data
├── axonml-llm            — axonml-nn
├── axonml-hvac           — axonml-nn
├── axonml-train          — axonml-nn (+ axonml-vision, axonml-llm via features)
├── axonml-distributed    — axonml-nn
├── axonml-serialize      — axonml-nn
├── axonml-onnx           — axonml-nn, axonml-serialize
├── axonml-jit            — axonml-tensor
├── axonml-quant          — axonml-tensor
├── axonml-fusion         — axonml-tensor
└── axonml-profile        — axonml-tensor
```

## Building Individual Crates

```bash
# Build a specific crate
cargo build -p axonml-nn

# Test a specific crate
cargo test -p axonml-nn

# Generate docs
cargo doc -p axonml-nn --open

# Build with features
cargo build -p axonml-core --features "cuda"
cargo build -p axonml --features "cuda,nccl"
```

## Feature Flags by Crate

### axonml-core

| Feature | Description |
|:--------|:------------|
| `std` | Standard library (default) |
| `cuda` | NVIDIA CUDA backend (cuBLAS + PTX kernels) |
| `cudnn` | cuDNN dispatch (requires `cuda`) |
| `vulkan` | Vulkan compute shaders |
| `metal` | Apple Metal backend |
| `wgpu` | WebGPU (WGSL via `wgpu` crate) |

### axonml (umbrella)

| Feature | Pulls in |
|:--------|:---------|
| `full` (default) | Everything listed below |
| `core` | core + tensor + autograd |
| `nn` | core + nn + optim |
| `data` | core + data |
| `vision` | nn + data + vision |
| `text` | nn + data + text |
| `audio` | nn + data + audio |
| `llm` | nn + llm |
| `hvac` | nn + hvac |
| `train` | nn + train |
| `distributed` | nn + distributed |
| `nccl` | distributed + NCCL backend |
| `profile`, `serialize`, `quant`, `fusion`, `jit`, `onnx` | matching sub-crate |
| `cuda`, `cudnn`, `wgpu` | GPU backend passthrough |

## API Documentation

Per-crate rustdoc is published to docs.rs:

- [axonml](https://docs.rs/axonml) — umbrella re-exports
- [axonml-tensor](https://docs.rs/axonml-tensor) — tensor ops
- [axonml-nn](https://docs.rs/axonml-nn) — modules
- [All crates](https://crates.io/search?q=axonml)

---

*Last updated: 2026-06-06 (v0.6.5)*
