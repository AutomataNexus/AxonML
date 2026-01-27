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

fn train(model: &impl Module, train_loader: &DataLoader, epochs: usize) {
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::new();

    for epoch in 0..epochs {
        model.train();
        let mut total_loss = 0.0;

        for (batch_idx, (inputs, targets)) in train_loader.iter().enumerate() {
            let x = Variable::new(inputs, false);
            let y = targets;

            // Forward pass
            let output = model.forward(&x);
            let loss = loss_fn.compute(&output, &y);

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data().item();
        }

        println!("Epoch {}: Loss = {:.4}", epoch, total_loss / train_loader.len() as f32);
    }
}
```

## Optimizers

### SGD

```rust
use axonml::optim::SGD;

let optimizer = SGD::new(model.parameters(), 0.01);

// With momentum
let optimizer = SGD::new(model.parameters(), 0.01)
    .momentum(0.9);

// With Nesterov momentum
let optimizer = SGD::new(model.parameters(), 0.01)
    .momentum(0.9)
    .nesterov(true);

// With weight decay
let optimizer = SGD::new(model.parameters(), 0.01)
    .momentum(0.9)
    .weight_decay(1e-4);
```

### Adam

```rust
use axonml::optim::Adam;

let optimizer = Adam::new(model.parameters(), 0.001);

// With custom betas
let optimizer = Adam::new(model.parameters(), 0.001)
    .betas(0.9, 0.999)
    .eps(1e-8);
```

### AdamW

```rust
use axonml::optim::AdamW;

// Adam with decoupled weight decay
let optimizer = AdamW::new(model.parameters(), 0.001)
    .betas(0.9, 0.999)
    .weight_decay(0.01);
```

### LAMB

```rust
use axonml::optim::LAMB;

// For large batch training (BERT-scale)
let optimizer = LAMB::new(model.parameters(), 0.001)
    .betas(0.9, 0.999)
    .weight_decay(0.01);
```

### RMSprop

```rust
use axonml::optim::RMSprop;

let optimizer = RMSprop::new(model.parameters(), 0.01)
    .alpha(0.99)
    .eps(1e-8)
    .momentum(0.0);
```

## Learning Rate Schedulers

### StepLR

```rust
use axonml::optim::{Adam, StepLR};

let mut optimizer = Adam::new(model.parameters(), 0.1);
let mut scheduler = StepLR::new(&optimizer, 10, 0.1);

for epoch in 0..100 {
    train_one_epoch(&mut optimizer);
    scheduler.step(&mut optimizer);
    println!("LR: {}", optimizer.get_lr());
}
```

### MultiStepLR

```rust
use axonml::optim::MultiStepLR;

// Decay at epochs 30, 60, 90
let mut scheduler = MultiStepLR::new(&optimizer, &[30, 60, 90], 0.1);
```

### ExponentialLR

```rust
use axonml::optim::ExponentialLR;

// Multiply by gamma each epoch
let mut scheduler = ExponentialLR::new(&optimizer, 0.95);
```

### CosineAnnealingLR

```rust
use axonml::optim::CosineAnnealingLR;

// Cosine annealing over 100 epochs
let mut scheduler = CosineAnnealingLR::new(&optimizer, 100, 0.0);
```

### OneCycleLR

```rust
use axonml::optim::OneCycleLR;

// 1cycle policy for super-convergence
let mut scheduler = OneCycleLR::new(&optimizer, 0.1, 100, 1000);
```

### WarmupLR

```rust
use axonml::optim::WarmupLR;

// Linear warmup for 1000 steps
let mut scheduler = WarmupLR::new(&optimizer, 1000);
```

### ReduceLROnPlateau

```rust
use axonml::optim::ReduceLROnPlateau;

let mut scheduler = ReduceLROnPlateau::new(&optimizer)
    .mode("min")
    .factor(0.1)
    .patience(10);

// After validation
scheduler.step_with_metric(&mut optimizer, val_loss);
```

## Mixed Precision Training (AMP)

### GradScaler

```rust
use axonml::optim::GradScaler;
use axonml::autograd::amp::autocast;
use axonml::core::DType;

let mut optimizer = Adam::new(model.parameters(), 0.001);
let mut scaler = GradScaler::new();

for (inputs, targets) in train_loader.iter() {
    // Forward pass with autocast
    let loss = autocast(DType::F16, || {
        let output = model.forward(&inputs);
        loss_fn.compute(&output, &targets)
    });

    // Scale loss for backward
    let scaled_loss = scaler.scale_loss(loss.data().item());

    // Backward
    optimizer.zero_grad();
    loss.backward();

    // Unscale gradients and check for inf/nan
    let mut grads: Vec<f32> = model.parameters()
        .iter()
        .flat_map(|p| p.grad().unwrap().to_vec())
        .collect();

    if scaler.unscale_grads(&mut grads) {
        optimizer.step();
    }

    // Update scaler
    scaler.update();
}
```

### Autocast Context

```rust
use axonml::autograd::amp::{autocast, AutocastGuard, is_autocast_enabled};

// Function-based
let output = autocast(DType::F16, || {
    model.forward(&input)
});

// RAII guard
{
    let _guard = AutocastGuard::new(DType::F16);
    let output = model.forward(&input);
    // Autocast disabled when guard drops
}

// Check if enabled
if is_autocast_enabled() {
    println!("Autocast is on");
}
```

## Gradient Checkpointing

Trade compute for memory on large models:

```rust
use axonml::autograd::checkpoint::{checkpoint, checkpoint_sequential};

// Checkpoint a single function
let output = checkpoint(|x| heavy_layer.forward(x), &input);

// Checkpoint sequential layers in segments
let output = checkpoint_sequential(24, 4, &input, |layer_idx, x| {
    layers[layer_idx].forward(x)
});
```

## Gradient Clipping

```rust
// Clip by norm
let max_norm = 1.0;
let total_norm = clip_grad_norm(&model.parameters(), max_norm);

// Clip by value
clip_grad_value(&model.parameters(), 0.5);
```

## Evaluation

```rust
fn evaluate(model: &impl Module, test_loader: &DataLoader) -> f32 {
    model.eval();
    let mut correct = 0;
    let mut total = 0;

    // Disable gradient computation
    no_grad(|| {
        for (inputs, targets) in test_loader.iter() {
            let output = model.forward(&Variable::new(inputs, false));
            let predictions = output.data().argmax(1);

            for (pred, label) in predictions.iter().zip(targets.iter()) {
                if pred == label {
                    correct += 1;
                }
                total += 1;
            }
        }
    });

    100.0 * correct as f32 / total as f32
}
```

## Complete Training Script

```rust
use axonml::prelude::*;
use axonml::vision::MNIST;
use axonml::data::DataLoader;

fn main() {
    // Data
    let train_dataset = MNIST::new("./data", true);
    let test_dataset = MNIST::new("./data", false);
    let train_loader = DataLoader::new(train_dataset, 64, true);
    let test_loader = DataLoader::new(test_dataset, 64, false);

    // Model
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));

    // Optimizer and scheduler
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let mut scheduler = CosineAnnealingLR::new(&optimizer, 10, 1e-6);
    let loss_fn = CrossEntropyLoss::new();

    // Training
    for epoch in 0..10 {
        model.train();
        let mut train_loss = 0.0;

        for (inputs, targets) in train_loader.iter() {
            let x = Variable::new(inputs.view(&[-1, 784]), false);
            let output = model.forward(&x);
            let loss = loss_fn.compute(&output, &targets);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            train_loss += loss.data().item();
        }

        // Evaluate
        let accuracy = evaluate(&model, &test_loader);
        scheduler.step(&mut optimizer);

        println!("Epoch {}: Loss={:.4}, Acc={:.2}%, LR={:.6}",
                 epoch, train_loss / train_loader.len() as f32,
                 accuracy, optimizer.get_lr());
    }

    // Save
    save_model(&model, "mnist_model.safetensors").unwrap();
}
```
