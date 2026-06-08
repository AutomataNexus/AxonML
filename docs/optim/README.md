# axonml-optim Documentation

> Optimization algorithms for the AxonML ML framework.

## Overview

`axonml-optim` ships the gradient-based optimizers and learning-rate
schedulers used to train AxonML models, plus a `GradScaler` for mixed
precision and a built-in training-health monitor.

## Core Concepts

### `Optimizer` trait

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn parameters(&self) -> &[Parameter];
    fn set_lr(&mut self, lr: f32);
    fn get_lr(&self) -> f32;
}
```

### Training loop

```rust
let mut optimizer = Adam::new(model.parameters(), lr);

for epoch in 0..num_epochs {
    for batch in dataloader.iter() {
        let output = model.forward(&batch.data);
        let loss = loss_fn.compute(&output, &batch.targets);

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

## Optimizers

### `SGD` — `sgd.rs`

Classic SGD with optional momentum, weight decay, dampening, and Nesterov.

```rust
let opt = SGD::new(params, lr);
let opt = SGD::with_momentum(params, lr, momentum);
```

Update rule:

```
v_t = momentum * v_{t-1} + grad
param = param - lr * v_t
```

Nesterov variant: `param = param - lr * (grad + momentum * v_t)`.

### `Adam` — `adam.rs`

Adaptive moment estimation with bias correction.

```rust
let opt = Adam::new(params, lr);
```

Update rule:

```
m_t = b1*m_{t-1} + (1-b1)*g
v_t = b2*v_{t-1} + (1-b2)*g^2
m_hat = m_t / (1 - b1^t)
v_hat = v_t / (1 - b2^t)
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

### `AdamW` — `adam.rs`

Adam with decoupled weight decay — preferred for transformers.

```rust
let opt = AdamW::new(params, lr);
```

### `RMSprop` — `rmsprop.rs`

```rust
let opt = RMSprop::new(params, lr);
```

Supports `alpha` smoothing constant, `eps`, `weight_decay`, `momentum`,
`centered` variant.

### `LAMB` — `lamb.rs`

Layer-wise adaptive moments optimizer for large-batch training (BERT / ViT
scale).

```rust
let opt = LAMB::new(params, lr);
```

### `GradScaler` — `grad_scaler.rs`

Loss scaling for AMP training with F16 autocast (pairs with `axonml-autograd`
`amp`). Exposes `scale`, `unscale`, `step`, `update`, and `GradScalerState`
for serialization.

## Learning Rate Schedulers (`lr_scheduler.rs`)

Shared `LRScheduler` trait. All schedulers take `&mut optimizer` in
`step(...)`.

| Scheduler            | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `StepLR`             | Decay by gamma every `step_size` epochs                              |
| `MultiStepLR`        | Decay by gamma at explicit milestone epochs                          |
| `ExponentialLR`      | `lr_t = lr_0 * gamma^t`                                              |
| `CosineAnnealingLR`  | `lr_t = eta_min + 0.5*(lr_0 - eta_min)*(1 + cos(pi*t/T_max))`        |
| `OneCycleLR`         | One-cycle super-convergence schedule                                 |
| `WarmupLR`           | Linear or polynomial warmup from zero                                |
| `ReduceLROnPlateau`  | Reduce when a monitored metric stops improving                       |

Example:

```rust
let mut opt = SGD::with_momentum(model.parameters(), 0.1, 0.9);
let mut sched = StepLR::new(&opt, 30, 0.1);

for epoch in 0..100 {
    train_one_epoch(&model, &mut opt);
    sched.step(&mut opt);
}
```

`ReduceLROnPlateau`:

```rust
let mut sched = ReduceLROnPlateau::new(&opt)
    .mode("min")
    .factor(0.1)
    .patience(10)
    .threshold(1e-4);

for epoch in 0..100 {
    let val_loss = validate();
    sched.step(val_loss, &mut opt);
}
```

## Training Health Monitor — `health.rs`

Built-in real-time training monitor. No external tool required.

```rust
use axonml_optim::health::{TrainingMonitor, MonitorConfig};

let mut monitor = TrainingMonitor::new(MonitorConfig::default());

for step in 0..1000 {
    let loss = train_step(&model, &batch);
    let grad_norm = compute_grad_norm(&model);
    monitor.record_step(loss, grad_norm, optimizer.get_lr());

    for alert in monitor.alerts_since_last_check() {
        eprintln!("[{:?}] {}", alert.severity, alert.message);
    }
}

let report = monitor.health_report();
println!("Loss trend: {:?}", report.loss_trend);
println!("Convergence: {:.2}", monitor.convergence_score());
println!("Suggested LR: {:?}", monitor.suggest_lr());
```

Exported types: `TrainingMonitor`, `MonitorConfig`, `TrainingAlert`,
`AlertKind`, `AlertSeverity`, `HealthReport`, `LossTrend`.

Detectors:

| Alert              | Severity  | Meaning                                          |
|--------------------|-----------|--------------------------------------------------|
| NaN Loss           | Critical  | Loss became NaN — LR likely too high             |
| Gradient Explosion | Critical  | Grad norm > threshold — clip gradients           |
| Gradient Vanishing | Warning   | Grad norm near zero — check architecture         |
| Loss Plateau       | Warning   | No improvement for N steps — reduce LR           |
| Loss Oscillation   | Warning   | Loss swinging — LR too high                      |
| Dead Neurons       | Info      | Neurons with zero gradient — check activation    |
| Divergence         | Critical  | Loss rising consistently — training failing      |

## Optimizer Selection Guide

| Optimizer      | Best For                           | Typical LR     |
|----------------|------------------------------------|----------------|
| SGD+Momentum   | CNNs, well-tuned models            | 0.01 – 0.1     |
| Adam           | General purpose, quick convergence | 0.001          |
| AdamW          | Transformers, large models         | 0.0001 – 0.001 |
| RMSprop        | RNNs, non-stationary objectives    | 0.001          |
| LAMB           | Large batch (BERT, ViT)            | 1e-3 – 1e-2    |

## Related Modules

- [Neural Networks](../nn/README.md) — models to optimize
- [Autograd](../autograd/README.md) — gradient computation (+ `amp` for AMP)

## Last updated

0.6.5 (2026-06-06)
