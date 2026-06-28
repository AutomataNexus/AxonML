---
layout: default
title: Home
nav_order: 1
description: "AxonML - A complete, PyTorch-equivalent machine learning framework written in pure Rust"
permalink: /
---

# AxonML Documentation
{: .fs-9 }

A complete, PyTorch-equivalent machine learning framework written in pure Rust.
{: .fs-6 .fw-300 }

[Get Started]({% link getting-started.md %}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/AutomataNexus/AxonML){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Overview

AxonML (named after axons — the nerve fibers that transmit signals between neurons) is a complete machine learning framework written in pure Rust. The goal is PyTorch-equivalent functionality while leveraging Rust's performance, safety, and concurrency guarantees.

### PyTorch Parity: ~92–95%

AxonML provides comprehensive PyTorch-equivalent functionality with **2,350+ passing tests** across 23 workspace crates.

### Key Features

| Category | Features |
|:---------|:---------|
| **Tensor Operations** | N-dimensional tensors, NumPy-style broadcasting, zero-copy views, matmul (with cuBLAS + Q4_K/Q6_K in-shader dequant GEMV), reductions, lazy tensors with algebraic optimization |
| **Automatic Differentiation** | Dynamic computational graph, reverse-mode autodiff, AMP autocast (F16), gradient checkpointing, graph inspection/DOT export |
| **Neural Networks** | Linear, Conv1d/2d, BatchNorm1d/2d, LayerNorm, GroupNorm, RMSNorm, MultiHead/Cross/Differential attention, LSTM/GRU/RNN, Transformer encoder/decoder, MoE, GCN/GAT, 1.58-bit ternary quantized linear (BitNet b1.58), differentiable structured sparsity |
| **Optimizers** | SGD (+ momentum, Nesterov), Adam, AdamW, RMSprop, LAMB; schedulers (Step, MultiStep, Cosine, OneCycle, Warmup, ReduceLROnPlateau, Exponential); `GradScaler`; training health monitor |
| **Distributed Training** | DDP, FSDP (ZeRO-2/ZeRO-3 + HybridShard + CPU offload), Pipeline (GPipe / 1F1B), column/row tensor parallel, optional NCCL backend |
| **Model Formats** | ONNX import/export (opset 17, 40+ ops), SafeTensors, StateDict, and `.axonml` bundles with an **embedded computation graph** for source-free recompilation (→ Hailo NPU HEF) |
| **Vision Models** | LeNet, ResNet, VGG, ViT, DETR, NanoDet, BlazeFace, RetinaFace, FPN, DPT/FastDepth (depth), PatchCore/StudentTeacher (anomaly), VQA |
| **LLM Architectures** | BERT, GPT-2, LLaMA, Mistral, Phi, SSM (Mamba), Qwen3 (trainable) |
| **Inference Stack** | the AxonML inference server — pure-Rust LLM inference with Anthropic Messages API, SSE streaming, Q4_K/Q5_K/Q6_K/Q8_0 CUDA GEMV, fused prefill + flash-decode attention, `--mlock` / `--no-mmap` / `--n-gpu-layers` / `--n-cpu-moe`, and TurboQuant KV-cache quantization (`--kv-quant q8`/`turbo`) |
| **Device-native execution** | GPU stays resident on-device; the **CPU backend is rayon-parallel** across matmul (all layouts), the full GradFn backward family, reductions, and the SwiGLU/RMSNorm/RoPE math — single-node CPU inference and training use every core |
| **GPU Backends** | CUDA (cuBLAS + 15+ PTX kernel modules), Vulkan, Metal, WebGPU |

## Architecture

```
+-------------------------------------------------------------------------+
|                        Application Layer                                |
+------------+--------------+-------------+-----------------+---------+
| axonml-cli | axonml-server| axonml-tui  | axonml-dashboard|  serve   |
|   (CLI)    | (REST API)   | (Terminal)  | (WASM Web UI)   |(inference)|
+------------+--------------+-------------+-----------------+---------+
|                             axonml                                      |
|                   (Umbrella Crate / Feature Flags)                     |
+-------------------------------------------------------------------------+
|                            Domain Layer                                 |
+-------------+-------------+-------------+----------------+--------------+
|axonml-vision|axonml-audio |axonml-text  | axonml-llm     |
+-------------+-------------+-------------+----------------+
|                          Training Layer                                 |
+-------------+-------------+-------------+----------------+
|  axonml-nn  |axonml-optim |axonml-data  |axonml-train    |axonml-distributed|
+-------------+-------------+-------------+----------------+-------------+
|                         Optimization Layer                              |
+-------------+-------------+-------------+----------------+
| axonml-quant|axonml-fusion| axonml-jit  | axonml-profile |
+-------------+-------------+-------------+----------------+
|                        Serialization Layer                              |
+-------------------------+-----------------------------------------------+
|    axonml-serialize     |                axonml-onnx                    |
+-------------------------+-----------------------------------------------+
|                        Computation Layer                                |
+-------------------------------------------------------------------------+
|                          axonml-autograd                                |
+-------------------------------------------------------------------------+
|                           axonml-tensor                                 |
+-------------------------------------------------------------------------+
|                           axonml-core                                   |
|               CPU | CUDA | Vulkan | Metal | WebGPU                      |
+-------------------------------------------------------------------------+
```

## Quick Links

| Section | Description |
|:--------|:------------|
| [Getting Started]({% link getting-started.md %}) | Installation and first model |
| [Tensor Operations]({% link tensors.md %}) | Working with tensors |
| [Neural Networks]({% link neural-networks.md %}) | Building models |
| [Training]({% link training.md %}) | Training loops and optimization |
| [Distributed]({% link distributed.md %}) | Multi-GPU and distributed training |
| [Detection]({% link detection.md %}) | Object and face detection training |
| [ONNX]({% link onnx.md %}) | ONNX import and export |
| [Crate Documentation]({% link crates/index.md %}) | All 23 crates |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml = "0.6"
```

Or with specific features:

```toml
[dependencies]
axonml = { version = "0.6", features = ["cuda", "vision", "llm"] }
```

## Quick Example

```rust
use axonml::prelude::*;
use axonml_nn::{Linear, ReLU, Sequential, CrossEntropyLoss, Module};
use axonml_optim::{Adam, Optimizer};

fn main() {
    // Build a simple MLP
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));

    // Optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::new();

    // Training step (assuming `inputs: Variable`, `targets: Variable`)
    let output = model.forward(&inputs);
    let loss = loss_fn.compute(&output, &targets);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    println!("Loss = {:.4}", loss.data().to_vec()[0]);
}
```

For a complete end-to-end runnable example, see
[`crates/axonml/examples/simple_training.rs`](https://github.com/AutomataNexus/AxonML/blob/main/crates/axonml/examples/simple_training.rs)
which trains a 2-layer MLP on the XOR problem with Adam.

## Production Deployment

AxonML powers real-time predictive maintenance on HVAC systems across commercial buildings. Site-specific models (LSTM autoencoders for anomaly detection + GRU failure predictors, ~105K–416K params per site) run live inference on Raspberry Pi edge controllers, cross-compiled to `armv7-unknown-linux-musleabihf`, polling sensor data at 1 Hz. Vision models are additionally compiled to Hailo-8/10H NPU silicon via the Hailo NPU compiler toolchain using the `.axonml` embedded-graph bundle format.

The AxonML pure-Rust LLM inference server reaches ~33 tok/s steady-state decode on a quantized 7B model (DeepSeek-R1-Distill-Qwen-7B, Q4_K_M) on an RTX 5070 Ti Laptop, via GPU-resident activations, custom Q4_K/Q5_K/Q6_K/Q8_0 dequant-in-shader GEMV, fused QKV / gate-up-SwiGLU kernels, GPU-native flash-decode attention, and TurboQuant KV-cache quantization.

## License

AxonML is dual-licensed under [MIT](https://github.com/AutomataNexus/AxonML/blob/main/LICENSE-MIT) and [Apache 2.0](https://github.com/AutomataNexus/AxonML/blob/main/LICENSE-APACHE).

---

*Last updated: 2026-06-06 (v0.6.5)*
