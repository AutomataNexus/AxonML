# axonml-optim

<p align="center">
  <img src="../../assets/logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-optim"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-optim** provides optimization algorithms for training neural networks in the AxonML framework. It includes popular gradient-based optimizers with momentum, adaptive learning rates, and comprehensive learning rate scheduling strategies.

## Features

- **SGD** - Stochastic Gradient Descent with optional momentum, Nesterov acceleration, weight decay, and dampening
- **Adam** - Adaptive Moment Estimation with bias correction and optional AMSGrad variant
- **AdamW** - Adam with decoupled weight decay regularization for improved generalization
- **RMSprop** - Root Mean Square Propagation with optional momentum and centered gradient normalization
- **Learning Rate Schedulers** - Comprehensive scheduling including StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR, WarmupLR, and ReduceLROnPlateau
- **Builder Pattern** - Fluent API for configuring optimizer hyperparameters
- **Unified Interface** - Common `Optimizer` trait for interoperability

## Modules

| Module | Description |
|--------|-------------|
| `optimizer` | Core `Optimizer` trait and `ParamState` for parameter state management |
| `sgd` | Stochastic Gradient Descent with momentum and Nesterov acceleration |
| `adam` | Adam and AdamW optimizers with adaptive learning rates |
| `rmsprop` | RMSprop optimizer with optional centering and momentum |
| `lr_scheduler` | Learning rate scheduling strategies for training dynamics |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-optim = "0.1.0"
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
