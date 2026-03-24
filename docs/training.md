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

## Object Detection Training

AxonML includes full training infrastructure for anchor-free object detection. See the dedicated [Object Detection Training Guide](detection.md) for complete documentation.

### Detection-Specific Losses

```rust
use axonml_vision::losses::{FocalLoss, GIoULoss, UncertaintyLoss};
use axonml_nn::{BCEWithLogitsLoss, SmoothL1Loss};

// Focal Loss — essential for detection (background >> objects)
let focal = FocalLoss::new();  // alpha=0.25, gamma=2.0
let cls_loss = focal.compute(&pred_logits, &targets);

// SmoothL1 (Huber) — robust bbox regression
let smooth_l1 = SmoothL1Loss::new();
let bbox_loss = smooth_l1.compute(&pred_boxes, &target_boxes);

// GIoU — operates in IoU metric space
let giou_loss = GIoULoss::compute(&pred_boxes, &target_boxes);

// BCEWithLogits — numerically stable binary cross-entropy
let bce = BCEWithLogitsLoss::new();
let loss = bce.compute(&logits, &binary_targets);
```

### Training Loops

Built-in training step functions handle the full forward-loss-backward-step pipeline:

```rust
use axonml_vision::training::{nexus_training_step, phantom_training_step};

// Nexus (COCO object detection)
let loss = nexus_training_step(
    &mut nexus_model, &frame, &gt_boxes, &gt_classes, &mut optimizer,
);

// Phantom (WIDER FACE face detection)
let loss = phantom_training_step(
    &mut phantom_model, &frame, &gt_faces, &mut optimizer,
);
```

### Detection Evaluation

```rust
use axonml_vision::training::{compute_ap, compute_map, compute_coco_map};

let ap = compute_ap(&detections, &ground_truths, 0.5);       // AP@0.5
let map = compute_map(&all_dets, &all_gts, num_classes, 0.5); // mAP@0.5
let coco_map = compute_coco_map(&all_dets, &all_gts, num_classes); // mAP@[0.5:0.95]
```

## Biometric Model Training

AxonML includes the Aegis Biometric Suite with specialized losses and training infrastructure for identity verification models.

### Biometric Losses

```rust
use axonml_vision::models::biometric::losses::{
    ArgusLoss, EchoLoss, ContrastiveLoss, CenterLoss, AngularMarginLoss,
    CrystallizationLoss, ThemisLoss, LivenessLoss,
};

// Argus (face recognition) — ArcFace angular margin + center loss
let argus_loss = ArgusLoss::new(num_classes, embed_dim);
let loss = argus_loss.compute_var(&embeddings, &labels);

// Echo (speaker verification) — contrastive + prediction loss
let echo_loss = EchoLoss::new(margin);
let loss = echo_loss.compute_var(&embed_a, &embed_b, is_same);

// Themis (anti-spoofing) — liveness classification + trajectory regularization
let themis_loss = ThemisLoss::new();
let loss = themis_loss.compute_var(&scores, &labels);

// Mnemosyne (person re-id) — center loss + diversity regularization
let center_loss = CenterLoss::new(num_classes, embed_dim);
let loss = center_loss.compute_var(&embeddings, &labels);
```

### Biometric Training Example

```rust
use axonml_vision::models::biometric::argus::Argus;

// Create model and move to GPU
let mut model = Argus::new(num_identities, 512);
let device = Device::CUDA(0);

let mut optimizer = Adam::new(model.parameters(), 1e-4)
    .weight_decay(5e-4);

for epoch in 0..epochs {
    for (images, labels) in train_loader.iter() {
        // Move BOTH parameters AND inputs to device
        let x = Variable::new(images.to(device), false);
        let y = labels.to(device);

        let embeddings = model.forward(&x);
        let loss = argus_loss.compute_var(&embeddings, &y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
```

## NightVision Infrared Detection Training

NightVision is a multi-domain infrared object detector with domain-adaptive thermal feature extraction. It supports wildlife monitoring, human detection, and astronomical/interstellar thermal imaging.

### NightVision Configuration

```rust
use axonml_vision::models::nightvision::{NightVision, NightVisionConfig, ThermalDomain};

// Wildlife detection (thermal cameras)
let config = NightVisionConfig::wildlife(num_species);

// Human detection (search & rescue, perimeter security)
let config = NightVisionConfig::human();

// Interstellar (astronomical thermal sources)
let config = NightVisionConfig::interstellar(num_classes, bands);

// Multi-domain (all thermal domains)
let config = NightVisionConfig::multi_domain(num_classes);

// Edge deployment (lightweight)
let config = NightVisionConfig::edge(num_classes);

let model = NightVision::new(config);
```

### NightVision Architecture

```
IR Image [B, 1, H, W] or [B, 3, H, W]
  → ThermalBackbone (CSP blocks, multi-scale P3/P4/P5)
  → ThermalFPN (Feature Pyramid Network)
  → DecoupledHead (cls + bbox + objectness per scale)
  → Detections: [class, x, y, w, h, confidence, domain]
```

## GPU Device Placement

When training on GPU, you must move **both** model parameters and input tensors to the same device. Forgetting to move inputs is the most common source of device mismatch errors.

```rust
use axonml::core::Device;

let device = Device::CUDA(0);

// Move model parameters to GPU
for param in model.parameters() {
    param.to(device);
}

// In the training loop, move EACH batch to GPU
for (inputs, targets) in train_loader.iter() {
    let x = Variable::new(inputs.to(device), false);
    let y = targets.to(device);

    let output = model.forward(&x);
    // ... loss, backward, step ...
}
```

For async GPU prefetch (overlaps data transfer with compute):

```rust
// Enable prefetch — background thread pre-transfers next batch to GPU
let train_loader = DataLoader::new(dataset, batch_size, shuffle)
    .prefetch_to_gpu(device);
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
