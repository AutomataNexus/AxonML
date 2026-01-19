# axonml-optim Documentation

> Optimization algorithms for the Axonml ML framework.

## Overview

`axonml-optim` provides gradient-based optimization algorithms for training neural networks. It includes popular optimizers like SGD, Adam, and AdamW, along with learning rate schedulers.

## Core Concepts

### Optimizer Trait

All optimizers implement the `Optimizer` trait:

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn parameters(&self) -> &[Parameter];
    fn set_lr(&mut self, lr: f32);
    fn get_lr(&self) -> f32;
}
```

### Training Loop Pattern

```rust
let mut optimizer = Adam::new(model.parameters(), lr);

for epoch in 0..num_epochs {
    for batch in dataloader.iter() {
        // Forward pass
        let output = model.forward(&batch.data);
        let loss = loss_fn.forward(&output, &batch.targets);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

## Optimizers

### sgd.rs - Stochastic Gradient Descent

Classic SGD with optional momentum and weight decay.

```rust
// Basic SGD
let optimizer = SGD::new(params, lr);

// With momentum
let optimizer = SGD::with_momentum(params, lr, momentum);

// Full options
let optimizer = SGD::with_options(SGDConfig {
    lr: 0.01,
    momentum: 0.9,
    weight_decay: 1e-4,
    dampening: 0.0,
    nesterov: true,
});
```

**Update Rule:**
```
v_t = momentum * v_{t-1} + grad
param = param - lr * v_t
```

With Nesterov:
```
v_t = momentum * v_{t-1} + grad
param = param - lr * (grad + momentum * v_t)
```

### adam.rs - Adam Optimizer

Adaptive moment estimation with bias correction.

```rust
// Default Adam
let optimizer = Adam::new(params, lr);

// Custom betas
let optimizer = Adam::with_options(AdamConfig {
    lr: 0.001,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.0,
});
```

**Update Rule:**
```
m_t = β₁ * m_{t-1} + (1 - β₁) * grad
v_t = β₂ * v_{t-1} + (1 - β₂) * grad²
m̂_t = m_t / (1 - β₁^t)  // Bias correction
v̂_t = v_t / (1 - β₂^t)
param = param - lr * m̂_t / (√v̂_t + ε)
```

### adamw.rs - AdamW Optimizer

Adam with decoupled weight decay (recommended for transformers).

```rust
let optimizer = AdamW::new(params, lr);

let optimizer = AdamW::with_options(AdamWConfig {
    lr: 0.001,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,
});
```

**Difference from Adam:**
- Weight decay is applied directly to weights, not through gradient
- Better generalization in practice

### rmsprop.rs - RMSprop

Root mean square propagation.

```rust
let optimizer = RMSprop::new(params, lr);

let optimizer = RMSprop::with_options(RMSpropConfig {
    lr: 0.01,
    alpha: 0.99,  // Smoothing constant
    eps: 1e-8,
    weight_decay: 0.0,
    momentum: 0.0,
    centered: false,
});
```

## Learning Rate Schedulers

### StepLR

Decay learning rate by gamma every step_size epochs.

```rust
let scheduler = StepLR::new(&optimizer, step_size, gamma);

for epoch in 0..100 {
    train_one_epoch();
    scheduler.step();
}
```

### ExponentialLR

Decay learning rate by gamma every epoch.

```rust
let scheduler = ExponentialLR::new(&optimizer, gamma);

// lr_t = lr_0 * gamma^t
```

### CosineAnnealingLR

Cosine annealing schedule.

```rust
let scheduler = CosineAnnealingLR::new(&optimizer, T_max, eta_min);

// lr_t = eta_min + 0.5 * (lr_0 - eta_min) * (1 + cos(π * t / T_max))
```

### ReduceLROnPlateau

Reduce learning rate when a metric stops improving.

```rust
let scheduler = ReduceLROnPlateau::new(&optimizer)
    .mode("min")      // Minimize the metric
    .factor(0.1)      // Multiply lr by this on reduction
    .patience(10)     // Wait this many epochs
    .threshold(1e-4); // Minimum improvement

for epoch in 0..100 {
    let val_loss = validate();
    scheduler.step(val_loss);
}
```

## Usage Examples

### Basic Training

```rust
use axonml::prelude::*;

let model = create_model();
let mut optimizer = Adam::new(model.parameters(), 0.001);

for epoch in 0..epochs {
    for batch in train_loader.iter() {
        let output = model.forward(&batch.data);
        let loss = cross_entropy(&output, &batch.targets);

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    println!("Epoch {}: Loss = {:.4}", epoch, avg_loss);
}
```

### With Learning Rate Scheduling

```rust
use axonml::prelude::*;

let model = create_model();
let mut optimizer = SGD::with_momentum(model.parameters(), 0.1, 0.9);
let mut scheduler = StepLR::new(&optimizer, 30, 0.1);

for epoch in 0..100 {
    train_one_epoch(&model, &mut optimizer);

    // Step scheduler at end of epoch
    scheduler.step();

    println!("Epoch {}: LR = {:.6}", epoch, optimizer.get_lr());
}
```

### Gradient Clipping

```rust
use axonml::prelude::*;

let mut optimizer = Adam::new(model.parameters(), 0.001);

for batch in train_loader.iter() {
    let loss = compute_loss(&model, &batch);
    loss.backward();

    // Clip gradients before stepping
    clip_grad_norm(model.parameters(), max_norm);

    optimizer.step();
    optimizer.zero_grad();
}
```

### Parameter Groups

```rust
use axonml::prelude::*;

// Different learning rates for different parts
let optimizer = Adam::with_param_groups(vec![
    ParamGroup {
        params: encoder.parameters(),
        lr: 0.0001,
        ..Default::default()
    },
    ParamGroup {
        params: decoder.parameters(),
        lr: 0.001,
        ..Default::default()
    },
]);
```

## Optimizer Selection Guide

| Optimizer | Best For | Typical LR |
|-----------|----------|------------|
| SGD+Momentum | CNNs, well-tuned models | 0.01 - 0.1 |
| Adam | General purpose, quick convergence | 0.001 |
| AdamW | Transformers, large models | 0.0001 - 0.001 |
| RMSprop | RNNs, non-stationary objectives | 0.001 |

## Related Modules

- [Neural Networks](../nn/README.md) - Models to optimize
- [Autograd](../autograd/README.md) - Gradient computation
- [Data](../data/README.md) - Data loading for training

@version 0.1.0
@author AutomataNexus Development Team
