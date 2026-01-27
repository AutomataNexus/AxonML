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

AxonML (named after axons - the nerve fibers that transmit signals between neurons) is an ambitious open-source project to create a complete machine learning framework in Rust. Our goal is to provide the same comprehensive functionality as PyTorch while leveraging Rust's performance, safety, and concurrency guarantees.

### PyTorch Parity: ~92-95%

AxonML provides comprehensive PyTorch-equivalent functionality with **1076+ passing tests**.

### Key Features

| Category | Features |
|:---------|:---------|
| **Tensor Operations** | N-dimensional tensors, broadcasting, views, slicing, matmul, reductions |
| **Automatic Differentiation** | Dynamic computational graph, reverse-mode autodiff, AMP, checkpointing |
| **Neural Networks** | Linear, Conv1d/2d, BatchNorm, LayerNorm, GroupNorm, Attention, LSTM/GRU |
| **Optimizers** | SGD, Adam, AdamW, RMSprop, LAMB with LR schedulers |
| **Distributed Training** | DDP, FSDP (ZeRO-2/3), Pipeline Parallelism |
| **Model Formats** | ONNX import/export (40+ operators), SafeTensors |
| **GPU Backends** | CUDA, Vulkan, Metal, WebGPU |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AxonML Suite                               │
├─────────────────────────────────────────────────────────────────────┤
│  axonml-cli    │  axonml-server  │  axonml-tui  │  axonml-dashboard │
├────────────────┴─────────────────┴──────────────┴───────────────────┤
│                              axonml                                  │
│                        (Umbrella Crate)                              │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│ axonml-vision│ axonml-audio │ axonml-text  │ axonml-llm             │
├──────────────┴──────────────┴──────────────┴────────────────────────┤
│ axonml-nn   │ axonml-optim │ axonml-data  │ axonml-distributed     │
├─────────────┴──────────────┴──────────────┴─────────────────────────┤
│           axonml-autograd          │         axonml-serialize       │
├────────────────────────────────────┴────────────────────────────────┤
│                          axonml-tensor                               │
├─────────────────────────────────────────────────────────────────────┤
│                           axonml-core                                │
│              (Device, DType, Storage, Memory)                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Links

| Section | Description |
|:--------|:------------|
| [Getting Started]({% link getting-started.md %}) | Installation and first model |
| [Tensor Operations]({% link tensors.md %}) | Working with tensors |
| [Neural Networks]({% link neural-networks.md %}) | Building models |
| [Training]({% link training.md %}) | Training loops and optimization |
| [Distributed]({% link distributed.md %}) | Multi-GPU and distributed training |
| [Crate Documentation]({% link crates/index.md %}) | All 22 crates |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml = "0.2.8"
```

Or with specific features:

```toml
[dependencies]
axonml = { version = "0.2.8", features = ["cuda", "vision", "llm"] }
```

## Quick Example

```rust
use axonml::prelude::*;

fn main() {
    // Create tensors
    let x = Tensor::randn(&[32, 784]);
    let y = Tensor::randn(&[32, 10]);

    // Build a simple MLP
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));

    // Create optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    // Training loop
    for epoch in 0..100 {
        let output = model.forward(&Variable::new(x.clone(), false));
        let loss = output.mse_loss(&Variable::new(y.clone(), false));

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        println!("Epoch {}: Loss = {:.4}", epoch, loss.data().item());
    }
}
```

## Benchmarks

| Operation | AxonML | PyTorch | Ratio |
|:----------|:-------|:--------|:------|
| MatMul (1024x1024) | 2.1ms | 1.8ms | 1.17x |
| Conv2d (224x224) | 4.3ms | 3.9ms | 1.10x |
| LSTM (seq=128) | 8.2ms | 7.1ms | 1.15x |
| Adam step | 0.8ms | 0.7ms | 1.14x |

*Benchmarks on AMD Ryzen 9 5900X, single-threaded CPU*

## License

AxonML is dual-licensed under [MIT](https://github.com/AutomataNexus/AxonML/blob/main/LICENSE-MIT) and [Apache 2.0](https://github.com/AutomataNexus/AxonML/blob/main/LICENSE-APACHE).
