---
layout: default
title: Getting Started
nav_order: 2
description: "Installation and setup guide for AxonML"
---

# Getting Started
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before using AxonML, ensure you have:

- **Rust** 1.85+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Cargo** (included with Rust)

Optional for GPU acceleration:
- **CUDA Toolkit** 12.0+ (for NVIDIA GPUs)
- **Vulkan SDK** (for cross-platform GPU)
- **Metal** (macOS, built into the OS)

## Installation

### As a Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml = "0.6"
```

### Feature Flags

The `axonml` umbrella crate is a thin re-export layer plus the live browser training monitor. Each sub-crate is gated behind a feature flag:

| Feature | Pulls in | Default |
|:--------|:---------|:--------|
| `full` | Everything below | Yes |
| `core` | `axonml-core`, `axonml-tensor`, `axonml-autograd` | Yes |
| `nn`   | `core` + `axonml-nn`, `axonml-optim` | Yes |
| `data` | `core` + `axonml-data` | Yes |
| `vision` | `nn` + `data` + `axonml-vision` | Yes |
| `text` | `nn` + `data` + `axonml-text` | Yes |
| `audio` | `nn` + `data` + `axonml-audio` | Yes |
| `llm` | `nn` + `axonml-llm` | Yes |
| `hvac` | `nn` + `axonml-hvac` | Yes |
| `train` | `nn` + `axonml-train` (`TrainingConfig`, `EarlyStopping`, benchmarking, adversarial) | Yes |
| `distributed` | `nn` + `axonml-distributed` | Yes |
| `serialize` | `core` + `axonml-serialize` | Yes |
| `onnx` | `core` + `axonml-onnx` | Yes |
| `quant` | `nn` + `axonml-quant` | Yes |
| `fusion` | `core` + `axonml-fusion` | Yes |
| `jit` | `core` + `axonml-jit` | Yes |
| `profile` | `core` + `axonml-profile` | Yes |
| `cuda` | NVIDIA CUDA backend (cuBLAS + PTX kernels) | No |
| `cudnn` | `cuda` + cuDNN dispatch | No |
| `wgpu` | WebGPU / Vulkan via wgpu | No |
| `nccl` | `distributed` + NCCL backend | No |

Example with specific features:

```toml
[dependencies]
axonml = { version = "0.6", default-features = false, features = ["core", "nn", "cuda"] }
```

### CLI Installation

Install the AxonML CLI:

```bash
cargo install --path crates/axonml-cli
```

## Your First Model

The canonical runnable introduction is `crates/axonml/examples/simple_training.rs` — a two-layer MLP (`Linear(2,4)` → sigmoid → `Linear(4,1)` → sigmoid) learning XOR with Adam (lr=0.1) over 1000 epochs, manual MSE loss.

### 1. Create a New Project

```bash
cargo new my_ml_project
cd my_ml_project
```

### 2. Add Dependencies

Edit `Cargo.toml`:

```toml
[dependencies]
axonml = "0.6"
```

### 3. Write Your Model

Edit `src/main.rs`. This mirrors the shipped `simple_training.rs` example:

```rust
use axonml::prelude::*;
use axonml_nn::{Linear, Module};
use axonml_optim::{Adam, Optimizer};

fn main() {
    println!("Version: {}", axonml::version());
    println!("Features: {}\n", axonml::features());

    // XOR dataset
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![0.0, 1.0, 1.0, 0.0];

    // Model: 2 -> 4 -> 1
    let linear1 = Linear::new(2, 4);
    let linear2 = Linear::new(4, 1);

    // Optimizer
    let params = [linear1.parameters(), linear2.parameters()].concat();
    let mut optimizer = Adam::new(params, 0.1);

    // Train
    for epoch in 0..1000 {
        let mut total_loss = 0.0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let x = Variable::new(
                Tensor::from_vec(input.clone(), &[1, 2]).unwrap(), true,
            );
            let h = linear1.forward(&x).sigmoid();
            let output = linear2.forward(&h).sigmoid();

            let y = Variable::new(
                Tensor::from_vec(vec![target], &[1, 1]).unwrap(), false,
            );

            // Manual MSE: (output - target)^2
            let diff = output.sub_var(&y);
            let loss = diff.mul_var(&diff);
            total_loss += loss.data().to_vec()[0];

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }

        if epoch % 200 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, total_loss / 4.0);
        }
    }
}
```

### 4. Run Your Model

```bash
cargo run --release
```

You will see the loss decrease over epochs until the MLP learns the XOR function.

## GPU Acceleration

### CUDA

Enable CUDA in `Cargo.toml`:

```toml
axonml = { version = "0.6", features = ["cuda"] }
```

Use GPU in code. `Tensor::to_device(...)` returns a `Result<Self>` and transfers data across backends:

```rust
use axonml::prelude::*;
use axonml_tensor::Tensor;

// Create on CPU, then transfer to GPU
let x: Tensor<f32> = Tensor::randn(&[1000, 1000]);
let x_gpu = x.to_device(Device::Cuda(0)).unwrap();

// Matmul dispatches to cuBLAS for GPU tensors
let y_gpu = x_gpu.matmul(&x_gpu).unwrap();

// Move back to CPU
let y_cpu = y_gpu.cpu().unwrap();
```

Note: When training on GPU, move **both** model parameters and input tensors to the same device — forgetting to move inputs is the single most common cause of `Error::DeviceMismatch`.

### WebGPU / Vulkan

```toml
axonml = { version = "0.6", features = ["wgpu"] }
```

```rust
let device = Device::Wgpu(0);
let x = Tensor::<f32>::randn(&[1000, 1000]).to_device(device).unwrap();
```

Vulkan, Metal, and WebGPU each have their own feature flag in `axonml-core` (`vulkan`, `metal`, `wgpu`). All four GPU backends are full implementations — not stubs — with 975/769/1710 lines of kernel code respectively.

## Next Steps

- [Tensor Operations]({% link tensors.md %}) — tensor API reference
- [Neural Networks]({% link neural-networks.md %}) — building models
- [Training]({% link training.md %}) — optimizers, schedulers, AMP, checkpoint
- [Detection]({% link detection.md %}) — Nexus, Phantom, NightVision detection training
- [Distributed]({% link distributed.md %}) — DDP, FSDP, Pipeline parallelism
- [ONNX]({% link onnx.md %}) — import / export

---

*Last updated: 2026-04-16 (v0.6.1)*
