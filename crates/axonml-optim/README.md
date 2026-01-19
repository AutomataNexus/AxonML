# axonml-optim

[![Crates.io](https://img.shields.io/crates/v/axonml-optim.svg)](https://crates.io/crates/axonml-optim)
[![Docs.rs](https://docs.rs/axonml-optim/badge.svg)](https://docs.rs/axonml-optim)
[![Downloads](https://img.shields.io/crates/d/axonml-optim.svg)](https://crates.io/crates/axonml-optim)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Optimizers and learning rate schedulers for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-optim` provides gradient-based optimizers and learning rate schedulers for training neural networks. It includes all major optimization algorithms with full support for weight decay, momentum, and adaptive learning rates.

## Features

### Optimizers
- **SGD** - Stochastic Gradient Descent with momentum and Nesterov
- **Adam** - Adaptive Moment Estimation
- **AdamW** - Adam with decoupled weight decay
- **RMSprop** - Root Mean Square Propagation
- **Adagrad** - Adaptive Gradient Algorithm

### Learning Rate Schedulers
- **StepLR** - Decay by factor every N epochs
- **MultiStepLR** - Decay at specific milestones
- **ExponentialLR** - Exponential decay each epoch
- **CosineAnnealingLR** - Cosine annealing to minimum
- **OneCycleLR** - 1cycle policy for super-convergence
- **WarmupLR** - Linear warmup then constant
- **ReduceLROnPlateau** - Reduce on metric plateau

### Advanced Features
- **Parameter groups** - Different LR/weight decay per layer
- **Gradient clipping** - Prevent gradient explosion
- **Weight decay** - L2 regularization (decoupled for AdamW)

## Installation

```toml
[dependencies]
axonml-optim = "0.1"
```

## Usage

### Basic SGD

```rust
use axonml_optim::{SGD, Optimizer};
use axonml_nn::Module;

let model = create_model();
let mut optimizer = SGD::new(model.parameters(), 0.01);

for epoch in 0..100 {
    for (data, target) in &dataloader {
        let output = model.forward(&data);
        let loss = compute_loss(&output, &target);

        optimizer.zero_grad();  // Clear gradients
        loss.backward();        // Compute gradients
        optimizer.step();       // Update parameters
    }
}
```

### SGD with Momentum and Nesterov

```rust
use axonml_optim::{SGD, Optimizer};

let mut optimizer = SGD::new(model.parameters(), 0.01)
    .momentum(0.9)           // Momentum coefficient
    .nesterov(true)          // Use Nesterov momentum
    .weight_decay(1e-4);     // L2 regularization
```

### Adam Optimizer

```rust
use axonml_optim::{Adam, Optimizer};

let mut optimizer = Adam::new(model.parameters(), 0.001)
    .betas(0.9, 0.999)    // Momentum coefficients
    .eps(1e-8)            // Numerical stability
    .weight_decay(0.0);   // L2 regularization (not decoupled)
```

### AdamW (Recommended for Transformers)

```rust
use axonml_optim::{AdamW, Optimizer};

// AdamW uses decoupled weight decay (better for transformers)
let mut optimizer = AdamW::new(model.parameters(), 0.001)
    .betas(0.9, 0.999)
    .weight_decay(0.01);  // Decoupled weight decay
```

### RMSprop

```rust
use axonml_optim::{RMSprop, Optimizer};

let mut optimizer = RMSprop::new(model.parameters(), 0.01)
    .alpha(0.99)        // Smoothing constant
    .eps(1e-8)
    .momentum(0.0)
    .weight_decay(0.0);
```

### Parameter Groups

```rust
use axonml_optim::{Adam, ParamGroup, Optimizer};

// Different learning rates for different parts of the model
let encoder_params = model.encoder.parameters();
let decoder_params = model.decoder.parameters();

let param_groups = vec![
    ParamGroup::new(encoder_params).lr(1e-4),   // Lower LR for encoder
    ParamGroup::new(decoder_params).lr(1e-3),   // Higher LR for decoder
];

let mut optimizer = Adam::from_param_groups(param_groups)
    .betas(0.9, 0.999);
```

### StepLR Scheduler

```rust
use axonml_optim::{Adam, StepLR, Optimizer, Scheduler};

let mut optimizer = Adam::new(model.parameters(), 0.001);
let mut scheduler = StepLR::new(&optimizer, 30, 0.1);
// Decay LR by 0.1 every 30 epochs

for epoch in 0..100 {
    train_one_epoch(&model, &mut optimizer);
    scheduler.step();  // Update learning rate

    println!("Epoch {}: LR = {}", epoch, optimizer.get_lr());
}
```

### CosineAnnealingLR

```rust
use axonml_optim::{Adam, CosineAnnealingLR, Optimizer, Scheduler};

let mut optimizer = Adam::new(model.parameters(), 0.001);
let mut scheduler = CosineAnnealingLR::new(&optimizer, 100, 1e-6);
// Cosine decay over 100 epochs to min LR of 1e-6

for epoch in 0..100 {
    train_one_epoch(&model, &mut optimizer);
    scheduler.step();
}
```

### OneCycleLR (Super-Convergence)

```rust
use axonml_optim::{SGD, OneCycleLR, Optimizer, Scheduler};

let mut optimizer = SGD::new(model.parameters(), 0.1);
let total_steps = num_epochs * steps_per_epoch;

let mut scheduler = OneCycleLR::new(&optimizer, 0.1, total_steps)
    .pct_start(0.3)           // 30% warmup
    .div_factor(25.0)         // Initial LR = max_lr / 25
    .final_div_factor(1e4);   // Final LR = max_lr / 10000

for epoch in 0..num_epochs {
    for batch in &dataloader {
        train_step(&model, &batch, &mut optimizer);
        scheduler.step();  // Step after each batch
    }
}
```

### Warmup + Cosine Decay

```rust
use axonml_optim::{Adam, WarmupLR, CosineAnnealingLR, Optimizer, Scheduler};

let mut optimizer = Adam::new(model.parameters(), 1e-4);

// Warmup for 1000 steps
let warmup_scheduler = WarmupLR::new(&optimizer, 1000);

// Then cosine decay
let cosine_scheduler = CosineAnnealingLR::new(&optimizer, 100, 1e-6);

let mut step = 0;
for epoch in 0..100 {
    for batch in &dataloader {
        train_step(&model, &batch, &mut optimizer);

        if step < 1000 {
            warmup_scheduler.step();
        } else {
            cosine_scheduler.step();
        }
        step += 1;
    }
}
```

### ReduceLROnPlateau

```rust
use axonml_optim::{Adam, ReduceLROnPlateau, Optimizer, Scheduler};

let mut optimizer = Adam::new(model.parameters(), 0.001);
let mut scheduler = ReduceLROnPlateau::new(&optimizer)
    .mode("min")           // Minimize metric (use "max" for accuracy)
    .factor(0.1)           // Reduce LR by 10x
    .patience(10)          // Wait 10 epochs before reducing
    .threshold(1e-4)       // Minimum improvement
    .min_lr(1e-6);         // Don't go below this

for epoch in 0..100 {
    let train_loss = train_one_epoch(&model, &mut optimizer);
    let val_loss = validate(&model);

    scheduler.step(val_loss);  // Pass the metric value

    println!("Epoch {}: val_loss={}, LR={}", epoch, val_loss, optimizer.get_lr());
}
```

### Gradient Clipping

```rust
use axonml_optim::{Adam, Optimizer, clip_grad_norm, clip_grad_value};

let mut optimizer = Adam::new(model.parameters(), 0.001);

for batch in &dataloader {
    let loss = compute_loss(&model, &batch);

    optimizer.zero_grad();
    loss.backward();

    // Option 1: Clip by global norm (recommended)
    let total_norm = clip_grad_norm(model.parameters(), 1.0);

    // Option 2: Clip by value
    // clip_grad_value(model.parameters(), 0.5);

    optimizer.step();
}
```

## API Reference

### Optimizers

| Optimizer | Description | Key Parameters |
|-----------|-------------|----------------|
| `SGD` | Stochastic Gradient Descent | `lr`, `momentum`, `nesterov`, `weight_decay` |
| `Adam` | Adaptive Moment Estimation | `lr`, `betas`, `eps`, `weight_decay` |
| `AdamW` | Adam with decoupled weight decay | `lr`, `betas`, `eps`, `weight_decay` |
| `RMSprop` | Root Mean Square Propagation | `lr`, `alpha`, `eps`, `momentum` |
| `Adagrad` | Adaptive Gradient | `lr`, `eps`, `weight_decay` |

### Optimizer Trait

| Method | Description |
|--------|-------------|
| `step()` | Perform one optimization step |
| `zero_grad()` | Zero all parameter gradients |
| `get_lr()` | Get current learning rate |
| `set_lr(lr)` | Set learning rate |
| `state_dict()` | Get optimizer state |
| `load_state_dict(state)` | Load optimizer state |

### Schedulers

| Scheduler | Description |
|-----------|-------------|
| `StepLR` | Decay every N epochs |
| `MultiStepLR` | Decay at milestones |
| `ExponentialLR` | Exponential decay |
| `CosineAnnealingLR` | Cosine annealing |
| `OneCycleLR` | 1cycle policy |
| `WarmupLR` | Linear warmup |
| `ReduceLROnPlateau` | Reduce on plateau |

### Utility Functions

| Function | Description |
|----------|-------------|
| `clip_grad_norm(params, max_norm)` | Clip by global L2 norm |
| `clip_grad_value(params, clip_value)` | Clip by absolute value |

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

```toml
[dependencies]
axonml = "0.1"  # Includes axonml-optim
```

## License

MIT OR Apache-2.0
