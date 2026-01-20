# axonml-vision

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-vision"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-vision** provides computer vision functionality for the AxonML framework. It includes image-specific transforms, loaders for common vision datasets (MNIST, CIFAR), pre-defined neural network architectures, and a model hub for pretrained weights.

## Features

- **Image Transforms** - Comprehensive augmentation including `Resize`, `CenterCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `ColorJitter`, `Grayscale`, `ImageNormalize`, and `Pad`
- **Vision Datasets** - Loaders for MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and synthetic variants for testing
- **Neural Network Models** - Pre-defined architectures including LeNet, SimpleCNN, MLP, ResNet (18/34), VGG (11/13/16/19), and Vision Transformer (ViT)
- **Model Hub** - Download, cache, and load pretrained weights with checksum verification
- **Bilinear Interpolation** - High-quality image resizing for 2D, 3D, and 4D tensors
- **ImageNet Normalization** - Built-in presets for ImageNet, MNIST, and CIFAR normalization

## Modules

| Module | Description |
|--------|-------------|
| `transforms` | Image-specific data augmentation and preprocessing transforms |
| `datasets` | Loaders for MNIST, CIFAR, and synthetic vision datasets |
| `models` | Pre-defined neural network architectures (LeNet, ResNet, VGG, ViT) |
| `hub` | Pretrained model weights management (download, cache, load) |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-vision = "0.1.0"
```

### Loading Datasets

```rust
use axonml_vision::prelude::*;

// Synthetic MNIST for testing
let train_data = SyntheticMNIST::train();
let test_data = SyntheticMNIST::test();

// Synthetic CIFAR-10
let cifar = SyntheticCIFAR::small();

// Get a sample
let (image, label) = train_data.get(0).unwrap();
assert_eq!(image.shape(), &[1, 28, 28]);  // MNIST: 1 channel, 28x28
assert_eq!(label.shape(), &[10]);          // One-hot encoded
```

### Image Transforms

```rust
use axonml_vision::{Resize, CenterCrop, RandomHorizontalFlip, ImageNormalize};
use axonml_data::{Compose, Transform};

// Build transform pipeline
let transform = Compose::empty()
    .add(Resize::new(256, 256))
    .add(CenterCrop::new(224, 224))
    .add(RandomHorizontalFlip::new())
    .add(ImageNormalize::imagenet());

let output = transform.apply(&image);
assert_eq!(output.shape(), &[3, 224, 224]);
```

### Normalization Presets

```rust
use axonml_vision::ImageNormalize;

// ImageNet normalization
let normalize = ImageNormalize::imagenet();
// mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

// MNIST normalization
let normalize = ImageNormalize::mnist();
// mean=[0.1307], std=[0.3081]

// CIFAR-10 normalization
let normalize = ImageNormalize::cifar10();
// mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
```

### Using Vision Models

```rust
use axonml_vision::{LeNet, MLP, SimpleCNN};
use axonml_vision::models::{resnet18, vgg16};
use axonml_nn::Module;
use axonml_autograd::Variable;

// LeNet for MNIST
let model = LeNet::new();
let input = Variable::new(batched_image, false);  // [N, 1, 28, 28]
let output = model.forward(&input);               // [N, 10]

// MLP for flattened images
let model = MLP::for_mnist();  // 784 -> 256 -> 128 -> 10

// ResNet18 for ImageNet
let model = resnet18(1000);
let output = model.forward(&input);  // [N, 1000]

// VGG16
let model = vgg16(1000, true);  // with batch normalization
```

### Full Training Pipeline

```rust
use axonml_vision::prelude::*;
use axonml_data::DataLoader;
use axonml_optim::{Adam, Optimizer};
use axonml_nn::{CrossEntropyLoss, Module};

// Create dataset and dataloader
let dataset = SyntheticMNIST::train();
let loader = DataLoader::new(dataset, 32).shuffle(true);

// Create model and optimizer
let model = LeNet::new();
let mut optimizer = Adam::new(model.parameters(), 0.001);
let loss_fn = CrossEntropyLoss::new();

// Training loop
for batch in loader.iter() {
    let input = Variable::new(batch.data, true);
    let target = batch.targets;

    optimizer.zero_grad();
    let output = model.forward(&input);
    let loss = loss_fn.compute(&output, &target);
    loss.backward();
    optimizer.step();
}
```

### Model Hub for Pretrained Weights

```rust
use axonml_vision::hub::{download_weights, load_state_dict, list_models, model_info};

// List available models
let models = list_models();
for model in models {
    println!("{}: {} classes, {:.1}MB", model.name, model.num_classes,
             model.size_bytes as f64 / 1_000_000.0);
}

// Get model info
if let Some(info) = model_info("resnet18") {
    println!("Accuracy: {:.2}%", info.accuracy);
}

// Download and load weights
let path = download_weights("resnet18", false)?;
let state_dict = load_state_dict(&path)?;

// Load into model
// model.load_state_dict(state_dict);
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-vision
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
