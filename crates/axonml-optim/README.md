# axonml-optim

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.85%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-optim"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-optim** provides optimization algorithms for training neural networks in the AxonML framework: five optimizers (SGD, Adam, AdamW, RMSprop, LAMB), seven learning-rate schedulers, a dynamic `GradScaler` for mixed-precision training, and a training health monitor that watches the run for NaNs, explosion/vanishing gradients, and stalled convergence.

## Features

- **SGD** - Stochastic Gradient Descent with optional momentum, Nesterov acceleration, weight decay, and dampening.
- **Adam** - Adaptive Moment Estimation with bias correction and optional AMSGrad variant.
- **AdamW** - Adam with decoupled weight decay regularization for improved generalization.
- **RMSprop** - Root Mean Square Propagation with optional momentum and centered gradient normalization.
- **LAMB** - Layer-wise Adaptive Moments for large-batch training (32k+ batches); Adam plus a per-layer trust ratio.
- **Learning Rate Schedulers** - `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `OneCycleLR`, `WarmupLR`, and `ReduceLROnPlateau`.
- **GradScaler** - Dynamic loss scaling for AMP; doubles on a healthy growth interval, halves on inf/NaN. Pairs with `autocast` / `AutocastGuard` from `axonml-autograd::amp`.
- **Builder Pattern** - Fluent API (`Adam::new(...).betas(...).eps(...).weight_decay(...).amsgrad(true)`) for configuring optimizer hyperparameters.
- **Unified Interface** - Common `Optimizer` trait (`step`, `zero_grad`, `get_lr`, `set_lr`) for interoperability.
- **Fused Optimizer Loops** - Adam, SGD, and RMSprop apply momentum, weight decay, and parameter updates in a single pass per tensor, reducing memory traffic.
- **GPU-Resident State** - Optimizer state (e.g. LAMB's `exp_avg` / `exp_avg_sq`) is allocated on the same device as the parameter; no CPU round-trips.
- **Training Health Monitor** - `TrainingMonitor` records per-step loss, gradient norm, and LR; emits `TrainingAlert`s (NaN, exploding/vanishing grad, stalled loss) with `AlertSeverity`; exports `HealthReport` including `LossTrend` and a convergence score with suggested LR.

## Modules

| Module | Description |
|--------|-------------|
| `optimizer` | Core `Optimizer` trait |
| `sgd` | `SGD` with momentum, Nesterov, dampening, weight decay |
| `adam` | `Adam` and `AdamW` |
| `rmsprop` | `RMSprop` with optional centering and momentum |
| `lamb` | `LAMB` layer-wise adaptive moments |
| `lr_scheduler` | `LRScheduler` trait + seven concrete schedulers |
| `grad_scaler` | `GradScaler` / `GradScalerState` for AMP loss scaling |
| `health` | `TrainingMonitor`, `MonitorConfig`, `HealthReport`, `TrainingAlert`, `AlertKind`, `AlertSeverity`, `LossTrend` |

## Cargo Features

| Feature | Purpose |
|---------|---------|
| `cuda` | Forwards CUDA support to `axonml-core` / `axonml-tensor` so optimizer state stays GPU-resident |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-optim = "0.6.1"
```

### Basic Training Loop

```rust
use axonml_optim::prelude::*;
use axonml_nn::{Linear, Module, Sequential, MSELoss};
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// Create model
let model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(Linear::new(128, 10));

// Create optimizer
let mut optimizer = Adam::new(model.parameters(), 0.001);
let loss_fn = MSELoss::new();

// Training loop
for epoch in 0..100 {
    let output = model.forward(&input);
    let loss = loss_fn.compute(&output, &target);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

### SGD with Momentum

```rust
use axonml_optim::{SGD, Optimizer};

// Basic SGD
let mut optimizer = SGD::new(model.parameters(), 0.01);

// SGD with momentum
let mut optimizer = SGD::new(model.parameters(), 0.01)
    .momentum(0.9)
    .weight_decay(0.0001)
    .nesterov(true);
```

### Adam with Custom Configuration

```rust
use axonml_optim::{Adam, AdamW, Optimizer};

// Adam with custom betas
let mut optimizer = Adam::new(model.parameters(), 0.001)
    .betas((0.9, 0.999))
    .eps(1e-8)
    .weight_decay(0.01)
    .amsgrad(true);

// AdamW for decoupled weight decay
let mut optimizer = AdamW::new(model.parameters(), 0.001)
    .weight_decay(0.01);
```

### LAMB for Large-Batch Training

```rust
use axonml_optim::{LAMB, Optimizer};

let mut optimizer = LAMB::new(model.parameters(), 0.001);
// LAMB scales each parameter's update by a per-layer trust ratio,
// enabling stable training at batch sizes of 32k+.
```

### Learning Rate Scheduling

```rust
use axonml_optim::{SGD, StepLR, CosineAnnealingLR, OneCycleLR, LRScheduler};

let mut optimizer = SGD::new(model.parameters(), 0.1);

// Step decay every 10 epochs
let mut scheduler = StepLR::new(&optimizer, 10, 0.1);

// Cosine annealing
let mut scheduler = CosineAnnealingLR::new(&optimizer, 100);

// One-cycle policy for super-convergence
let mut scheduler = OneCycleLR::new(&optimizer, 0.1, 1000);

// In training loop
for epoch in 0..epochs {
    // ... training ...
    scheduler.step(&mut optimizer);
}
```

### ReduceLROnPlateau

```rust
use axonml_optim::{SGD, ReduceLROnPlateau};

let mut optimizer = SGD::new(model.parameters(), 0.1);
let mut scheduler = ReduceLROnPlateau::with_options(
    &optimizer,
    "min",    // mode: minimize metric
    0.1,      // factor: reduce LR by 10x
    10,       // patience: wait 10 epochs
    1e-4,     // threshold
    0,        // cooldown
    1e-6,     // min_lr
);

// Step with validation loss
scheduler.step_with_metric(&mut optimizer, val_loss);
```

### Mixed Precision (AMP) with GradScaler

```rust
use axonml_optim::{Adam, GradScaler, Optimizer};
use axonml_autograd::autocast;
use axonml_core::DType;

let mut optimizer = Adam::new(model.parameters(), 1e-3);
let mut scaler = GradScaler::new();

for batch in batches {
    optimizer.zero_grad();

    // Forward in F16
    let loss = autocast(DType::F16, || loss_fn.compute(&model.forward(&batch.x), &batch.y));

    // Scale the loss before backward to avoid F16 underflow
    let scaled = scaler.scale(&loss);
    scaled.backward();

    // Unscale and step if no inf/NaN; update scale factor
    scaler.step(&mut optimizer);
    scaler.update();
}
```

### Training Health Monitor

The optimizer monitors its own training health — detects problems before they ruin the run.

```rust
use axonml_optim::{TrainingMonitor, MonitorConfig, AlertSeverity};

let mut monitor = TrainingMonitor::new(MonitorConfig::default());

// Record metrics each training step
for step in 0..1000 {
    let loss = train_step(&model, &batch);
    let grad_norm = compute_grad_norm(&model);

    monitor.record_step(loss, grad_norm, optimizer.get_lr());

    // Check for alerts
    for alert in monitor.alerts_since_last_check() {
        match alert.severity {
            AlertSeverity::Critical => eprintln!("CRITICAL: {}", alert.message),
            AlertSeverity::Warning => eprintln!("WARNING: {}", alert.message),
            _ => {}
        }
    }
}

// Analyze training health
let report = monitor.health_report();
println!("Loss trend: {:?}", report.loss_trend);
println!("Convergence: {:.2}", monitor.convergence_score());
println!("Suggested LR: {:?}", monitor.suggest_lr());
println!("{}", monitor.summary());
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-optim
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

_Last updated: 2026-04-16 (v0.6.1)_
