---
layout: default
title: Training
nav_order: 5
description: "Training neural networks with AxonML"
---

# Training
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Basic Training Loop

```rust
use axonml::prelude::*;
use axonml_nn::{Module, CrossEntropyLoss};
use axonml_optim::{Adam, Optimizer};

fn train<M: Module>(
    model: &mut M,
    batches: &[(Variable, Variable)],
    epochs: usize,
) {
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::new();

    for epoch in 0..epochs {
        model.train();
        let mut total_loss = 0.0;

        for (inputs, targets) in batches {
            let output = model.forward(inputs);
            let loss = loss_fn.compute(&output, targets);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data().to_vec()[0];
        }

        println!("Epoch {}: Loss = {:.4}", epoch, total_loss / batches.len() as f32);
    }
}
```

## Optimizers

All optimizers take `Vec<Parameter>` and a learning rate. Options are set via a `with_options(...)` constructor or chainable builder methods.

### SGD

```rust
use axonml_optim::SGD;

let opt = SGD::new(model.parameters(), 0.01);

// Builder chain
let opt = SGD::new(model.parameters(), 0.01)
    .momentum(0.9)
    .nesterov(true)
    .weight_decay(1e-4);

// Or all-at-once:
let opt = SGD::with_options(params, /*lr=*/0.01, /*momentum=*/0.9, /*weight_decay=*/1e-4, /*nesterov=*/true);
```

### Adam / AdamW

```rust
use axonml_optim::{Adam, AdamW};

let opt = Adam::new(model.parameters(), 0.001);

// `betas` takes a tuple
let opt = Adam::new(model.parameters(), 0.001)
    .betas((0.9, 0.999))
    .eps(1e-8)
    .weight_decay(0.0);

// AdamW (decoupled weight decay)
let opt = AdamW::new(model.parameters(), 0.001)
    .betas((0.9, 0.999))
    .weight_decay(0.01);
```

### LAMB

```rust
use axonml_optim::LAMB;

// LAMB's builder takes two f32s, not a tuple
let opt = LAMB::new(model.parameters(), 0.001)
    .betas(0.9, 0.999)
    .weight_decay(0.01);
```

### RMSprop

```rust
use axonml_optim::RMSprop;

let opt = RMSprop::new(model.parameters(), 0.01)
    .alpha(0.99)
    .eps(1e-8)
    .momentum(0.0);
```

## Learning Rate Schedulers

All schedulers take an `&impl Optimizer` and implement the `LRScheduler` trait. Call `scheduler.step(&mut optimizer)` once per epoch (or step, depending on the scheduler).

```rust
use axonml_optim::{Adam, Optimizer, StepLR, MultiStepLR, ExponentialLR,
                   CosineAnnealingLR, OneCycleLR, WarmupLR, ReduceLROnPlateau,
                   LRScheduler};

let mut opt = Adam::new(model.parameters(), 0.1);

// Step: decay by gamma every step_size epochs
let mut sch = StepLR::new(&opt, /*step_size=*/10, /*gamma=*/0.1);

// MultiStep: decay at specific milestones
let mut sch = MultiStepLR::new(&opt, vec![30, 60, 90], 0.1);

// Exponential
let mut sch = ExponentialLR::new(&opt, 0.95);

// Cosine annealing — takes t_max (period in steps)
let mut sch = CosineAnnealingLR::new(&opt, 100);

// OneCycle — max_lr + total_steps
let mut sch = OneCycleLR::new(&opt, /*max_lr=*/0.1, /*total_steps=*/1000);

// Linear warmup
let mut sch = WarmupLR::new(&opt, /*warmup_steps=*/1000);

// Reduce on plateau — takes no threshold at construction
let mut sch = ReduceLROnPlateau::new(&opt);
// scheduler.step() takes a metric for ReduceLROnPlateau
```

## Mixed Precision Training (AMP)

### Autocast

Located in `axonml_autograd::amp`:

```rust
use axonml_autograd::amp::{autocast, AutocastGuard, is_autocast_enabled, autocast_dtype};
use axonml_core::DType;

// Function-scoped autocast
let output = autocast(DType::F16, || {
    model.forward(&input)
});

// RAII guard
{
    let _guard = AutocastGuard::new(DType::F16);
    let output = model.forward(&input);
    // guard dropped -> autocast disabled
}

if is_autocast_enabled() {
    println!("Autocast is enabled at {:?}", autocast_dtype());
}
```

### GradScaler

Located in `axonml_optim::grad_scaler`:

```rust
use axonml_optim::GradScaler;

let mut scaler = GradScaler::new();
// or: let mut scaler = GradScaler::with_scale(2_f32.powi(16));

// Inside the loop:
let scaled_loss_value = scaler.scale_loss(loss_value);
loss.backward();

// Unscale + inf/nan check — skips the optimizer step if non-finite grads
let mut grads: Vec<f32> = /* collect flat grads */ Vec::new();
if scaler.unscale_grads(&mut grads) {
    optimizer.step();
}
```

## Gradient Checkpointing

Located in `axonml_autograd::checkpoint`:

```rust
use axonml_autograd::checkpoint::{checkpoint, checkpoint_sequential};

// Single-function checkpoint
let output = checkpoint(|x: &Variable| heavy_layer.forward(x), &input);

// Sequential layers, split into segments (trades recompute for peak memory)
let output = checkpoint_sequential(
    /*num_layers=*/24, /*segments=*/4, &input,
    |layer_idx, x| layers[layer_idx].forward(x),
);
```

## Gradient Clipping

Clipping utilities live in `axonml_train::trainer`:

```rust
use axonml_train::clip_grad_norm;

let total_norm = clip_grad_norm(&model.parameters(), /*max_norm=*/1.0);
```

## Evaluation

`no_grad` is a context manager in `axonml_autograd`:

```rust
use axonml_autograd::no_grad;

fn evaluate<M: Module>(model: &mut M, batches: &[(Variable, Variable)]) -> f32 {
    model.eval();

    no_grad(|| {
        let mut correct = 0;
        let mut total = 0;

        for (inputs, targets) in batches {
            let output = model.forward(inputs);
            // Compare argmax with targets; add your own logic here.
            total += output.shape()[0];
            // correct += ...
        }

        100.0 * correct as f32 / total.max(1) as f32
    })
}
```

## Training Infrastructure (`axonml-train`)

The higher-level training glue lives in the dedicated `axonml-train` crate (split out of the umbrella in April 2026):

```rust
use axonml_train::{
    TrainingConfig, TrainingHistory, TrainingMetrics, Callback, EarlyStopping,
    ProgressLogger, clip_grad_norm, compute_accuracy,
};

// Model benchmarking
use axonml_train::{benchmark_model, throughput_test, profile_model_memory, ThroughputConfig};

// Adversarial training
use axonml_train::{AdversarialTrainer, fgsm_attack, pgd_attack};
```

Every training binary in `llm-training` uses a shared `lifecycle.rs` with pause/resume/stop/checkpoint signal handlers so weeks-long runs survive process restart, plus a `train_ctl` control binary. Per project policy, every training binary ships with the live training monitor (`axonml::TrainingMonitor`) — it is not opt-out.

## Object Detection Training

Detection has a dedicated guide — see [Object Detection Training]({% link detection.md %}) for the detector models and training utilities.

Quick reference:

```rust
use axonml_vision::losses::{FocalLoss, GIoULoss, UncertaintyLoss, compute_centerness};
use axonml_nn::{BCEWithLogitsLoss, SmoothL1Loss};

let focal = FocalLoss::new();                      // alpha=0.25, gamma=2.0
let focal2 = FocalLoss::with_params(0.25, 2.0);

let smooth = SmoothL1Loss::new();
let bce = BCEWithLogitsLoss::new();

// GIoU is a bare compute function
let gl = GIoULoss::compute(&pred_boxes, &target_boxes);

// Anchor-free target assignment:
use axonml_vision::training::{assign_fcos_targets, assign_single_scale_targets};
// Compose forward → target assignment → loss → backward → optimizer step.
```

Evaluation:

```rust
use axonml_vision::training::{compute_ap, compute_map, compute_coco_map};

let ap = compute_ap(&detections, &ground_truths, 0.5);
let m  = compute_map(&all_dets, &all_gts, num_classes, 0.5);
let cm = compute_coco_map(&all_dets, &all_gts, num_classes);
```

## GPU Device Placement

**Both** model parameters and input tensors must live on the same device — `Error::DeviceMismatch` is the most common GPU training error:

```rust
use axonml_core::Device;

let device = Device::Cuda(0);

// Move model (walks Module::parameters and calls Parameter::to_device)
// Exact API lives on the Module trait's default `to_device` impl.

// Each batch:
for (inputs, targets) in batches {
    let x = Variable::new(inputs.to_device(device).unwrap(), false);
    let y = Variable::new(targets.to_device(device).unwrap(), false);

    let output = model.forward(&x);
    // loss, backward, step ...
}
```

## Device-Native Execution (CPU parallelism)

AxonML follows a device-native execution model: GPU tensors stay resident
on-device through the whole forward/backward pass, and the **CPU backend is
rayon-parallel**. As of 0.6.5 the CPU path threads:

- **matmul** in every layout — `matmul_f32` (m>1 prefill/training),
  `matmul_f32_bt` (the GGUF `[out,in]` inference layout), `f64`, the generic
  tiled fallback, and 3D/4D batched matmul;
- the **full GradFn backward family** — `MatMulBackward`, `SwigluBackward`,
  `Softmax`/`LogSoftmaxBackward`, `NarrowBackward`, `SumDimBackward`,
  `VarDimBackward`/`MeanDimBackward`, `CrossEntropyLossBackward`,
  `FusedAttentionBackward`, `reduce_grad_for_broadcast`, and the LSTM/GRU/Conv2d
  fallbacks;
- reductions, SwiGLU, RMSNorm (+heads/batched), RoPE, `layer_norm`, `gelu`, and
  residual adds, with contiguous/offset-0 fast-paths that skip the copy before
  the parallel backend.

All paths are threshold-gated, so tiny ops stay serial and numerical semantics
are unchanged. Single-node CPU training and inference now scale across cores
instead of pegging one thread.

To measure a single CPU training step:

```bash
# Repeatable pure-CPU LLM-step proxy (no GPU required)
STEPS=5 cargo run --release -p axonml --example cpu_bench_llm_step

# Per-GradFn backward timing on any training binary
AXONML_PROFILE_BACKWARD=1 cargo run --release ...
```

On a CUDA build, `profile_train_step` (in `llm-training`) buckets a step into
forward / loss / backward / optimizer and prints the per-GradFn breakdown.

---

*Last updated: 2026-06-06 (v0.6.5)*
