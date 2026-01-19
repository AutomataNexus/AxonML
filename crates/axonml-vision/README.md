# axonml-vision

[![Crates.io](https://img.shields.io/crates/v/axonml-vision.svg)](https://crates.io/crates/axonml-vision)
[![Docs.rs](https://docs.rs/axonml-vision/badge.svg)](https://docs.rs/axonml-vision)
[![Downloads](https://img.shields.io/crates/d/axonml-vision.svg)](https://crates.io/crates/axonml-vision)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Computer vision utilities for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-vision` provides image processing utilities, transforms, datasets, and pre-built model architectures for computer vision tasks. Inspired by torchvision, it includes everything needed for image classification, object detection, and more.

## Features

### Image Transforms
- **Resize** - Resize images to target size
- **CenterCrop/RandomCrop** - Crop operations
- **RandomHorizontalFlip/VerticalFlip** - Flip augmentations
- **ColorJitter** - Brightness, contrast, saturation, hue
- **Normalize** - Channel-wise normalization
- **ToTensor** - Convert images to tensors

### Datasets
- **SyntheticMNIST** - 28x28 grayscale digit recognition
- **SyntheticCIFAR** - 32x32 color image classification
- **ImageFolder** - Load images from directory structure

### Model Architectures
- **LeNet** - Classic CNN for MNIST
- **SimpleCNN** - Basic CNN for learning
- **VGG** - VGG-11, VGG-13, VGG-16, VGG-19
- **ResNet** - ResNet-18, ResNet-34, ResNet-50, ResNet-101
- **ViT** - Vision Transformer

### Pretrained Models
- **Model Hub** - Download pretrained weights
- **Transfer Learning** - Fine-tune on custom data

## Installation

```toml
[dependencies]
axonml-vision = "0.1"
```

## Usage

### Image Transforms

```rust
use axonml_vision::transforms::{
    Compose, Resize, CenterCrop, ToTensor, Normalize,
    RandomHorizontalFlip, ColorJitter
};

// Training transforms with augmentation
let train_transform = Compose::new()
    .add(Resize::new(256))
    .add(RandomCrop::new(224))
    .add(RandomHorizontalFlip::new(0.5))
    .add(ColorJitter::new(0.2, 0.2, 0.2, 0.1))
    .add(ToTensor::new())
    .add(Normalize::new(
        &[0.485, 0.456, 0.406],  // ImageNet mean
        &[0.229, 0.224, 0.225]   // ImageNet std
    ));

// Validation transforms (no augmentation)
let val_transform = Compose::new()
    .add(Resize::new(256))
    .add(CenterCrop::new(224))
    .add(ToTensor::new())
    .add(Normalize::new(
        &[0.485, 0.456, 0.406],
        &[0.229, 0.224, 0.225]
    ));
```

### Loading MNIST

```rust
use axonml_vision::datasets::SyntheticMNIST;
use axonml_data::DataLoader;

// Load training and test sets
let train_dataset = SyntheticMNIST::new(true, 60000);   // train=true
let test_dataset = SyntheticMNIST::new(false, 10000);   // train=false

let train_loader = DataLoader::new(train_dataset, 64)
    .shuffle(true);

let test_loader = DataLoader::new(test_dataset, 64)
    .shuffle(false);

// Training loop
for (images, labels) in train_loader.iter() {
    // images: [64, 1, 28, 28]
    // labels: [64]
    let output = model.forward(&images);
    // ...
}
```

### Loading CIFAR-10

```rust
use axonml_vision::datasets::SyntheticCIFAR;
use axonml_data::DataLoader;

// CIFAR-10: 10 classes, 32x32 color images
let train_dataset = SyntheticCIFAR::new(true, 50000);
let test_dataset = SyntheticCIFAR::new(false, 10000);

let train_loader = DataLoader::new(train_dataset, 128)
    .shuffle(true)
    .num_workers(4);
```

### ImageFolder Dataset

```rust
use axonml_vision::datasets::ImageFolder;
use axonml_vision::transforms::{Compose, Resize, ToTensor};

// Directory structure:
// data/
//   train/
//     cat/
//       cat1.jpg
//       cat2.jpg
//     dog/
//       dog1.jpg
//       dog2.jpg

let transform = Compose::new()
    .add(Resize::new(224))
    .add(ToTensor::new());

let dataset = ImageFolder::new("data/train", transform);
println!("Classes: {:?}", dataset.classes());  // ["cat", "dog"]
println!("Samples: {}", dataset.len());
```

### Using LeNet for MNIST

```rust
use axonml_vision::models::LeNet;
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};

// LeNet for 28x28 grayscale images, 10 classes
let model = LeNet::new(1, 10);  // in_channels=1, num_classes=10

let mut optimizer = Adam::new(model.parameters(), 0.001);

for epoch in 0..10 {
    for (images, labels) in train_loader.iter() {
        let output = model.forward(&images);
        let loss = cross_entropy(&output, &labels);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
```

### ResNet for Image Classification

```rust
use axonml_vision::models::{resnet18, resnet50};
use axonml_nn::Module;

// ResNet-18 for ImageNet (1000 classes)
let model = resnet18(1000);

// ResNet-50 for custom dataset (100 classes)
let model = resnet50(100);

// Forward pass: [batch, 3, 224, 224] -> [batch, num_classes]
let images = load_batch();  // [16, 3, 224, 224]
let logits = model.forward(&images);  // [16, 1000]
```

### VGG Networks

```rust
use axonml_vision::models::{vgg11, vgg16, vgg19};

let model = vgg16(1000);  // VGG-16 for ImageNet

// With batch normalization
let model = vgg16_bn(1000);
```

### Vision Transformer (ViT)

```rust
use axonml_vision::models::ViT;

// ViT-Base: patch_size=16, embed_dim=768, depth=12, heads=12
let model = ViT::new(
    224,    // image_size
    16,     // patch_size
    1000,   // num_classes
    768,    // embed_dim
    12,     // depth
    12,     // num_heads
);

let images = load_batch();  // [16, 3, 224, 224]
let logits = model.forward(&images);  // [16, 1000]
```

### Pretrained Models (Hub)

```rust
use axonml_vision::models::{resnet50, load_pretrained};

// Load pretrained ResNet-50
let model = resnet50(1000);
load_pretrained(&model, "resnet50")?;

// Fine-tune on custom dataset
// Replace last layer for new number of classes
let model = replace_fc(model, 10);  // 10 classes

// Freeze backbone, only train new classifier
freeze_layers(&model, "layer4");
```

### Data Augmentation Pipeline

```rust
use axonml_vision::transforms::*;

// Heavy augmentation for training
let augmentation = Compose::new()
    .add(RandomResizedCrop::new(224, (0.08, 1.0), (0.75, 1.33)))
    .add(RandomHorizontalFlip::new(0.5))
    .add(ColorJitter::new(0.4, 0.4, 0.4, 0.1))
    .add(RandomGrayscale::new(0.2))
    .add(ToTensor::new())
    .add(Normalize::imagenet())
    .add(RandomErasing::new(0.25));  // Cutout augmentation
```

## API Reference

### Transforms

| Transform | Description |
|-----------|-------------|
| `Resize(size)` | Resize to size (preserves aspect) |
| `CenterCrop(size)` | Crop center region |
| `RandomCrop(size)` | Random crop |
| `RandomResizedCrop` | Random crop + resize |
| `RandomHorizontalFlip(p)` | Flip with probability p |
| `RandomVerticalFlip(p)` | Vertical flip |
| `ColorJitter` | Random color adjustments |
| `RandomGrayscale(p)` | Convert to grayscale |
| `RandomRotation(degrees)` | Random rotation |
| `ToTensor` | Convert to tensor [C, H, W] |
| `Normalize(mean, std)` | Channel normalization |
| `RandomErasing(p)` | Random rectangle erasing |

### Datasets

| Dataset | Description | Shape |
|---------|-------------|-------|
| `SyntheticMNIST` | Handwritten digits | [1, 28, 28] |
| `SyntheticCIFAR` | CIFAR-10/100 images | [3, 32, 32] |
| `ImageFolder` | Load from directories | Variable |

### Models

| Model | Parameters | Top-1 Acc* |
|-------|------------|------------|
| `LeNet` | 62K | MNIST: 99% |
| `VGG-11` | 133M | 69.0% |
| `VGG-16` | 138M | 71.6% |
| `VGG-19` | 144M | 72.4% |
| `ResNet-18` | 11.7M | 69.8% |
| `ResNet-34` | 21.8M | 73.3% |
| `ResNet-50` | 25.6M | 76.1% |
| `ResNet-101` | 44.5M | 77.4% |
| `ViT-Base` | 86M | 77.9% |

*ImageNet Top-1 accuracy (pretrained weights)

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

```toml
[dependencies]
axonml = { version = "0.1", features = ["vision"] }
```

## License

MIT OR Apache-2.0
