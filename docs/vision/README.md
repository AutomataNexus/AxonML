# axonml-vision Documentation

> Computer vision utilities for the Axonml ML framework.

## Overview

`axonml-vision` provides image processing capabilities including transforms, datasets, and pre-built model architectures. It's the Axonml equivalent of PyTorch's torchvision.

## Modules

### transforms/

Image preprocessing and augmentation transforms.

#### Geometric Transforms

```rust
use axonml_vision::transforms::*;

// Resize image
let resize = Resize::new(224, 224);

// Center crop
let crop = CenterCrop::new(224);

// Random crop
let crop = RandomCrop::new(224);

// Random horizontal flip (p=0.5)
let flip = RandomHorizontalFlip::new();

// Random vertical flip
let flip = RandomVerticalFlip::new();

// Random rotation
let rotate = RandomRotation::new(degrees);
```

#### Color Transforms

```rust
// Normalize with mean and std
let normalize = ImageNormalize::new(
    vec![0.485, 0.456, 0.406],  // mean (RGB)
    vec![0.229, 0.224, 0.225],  // std (RGB)
);

// Random color jitter
let jitter = ColorJitter::new(brightness, contrast, saturation, hue);

// Grayscale conversion
let gray = Grayscale::new();
```

#### Composing Transforms

```rust
use axonml_vision::transforms::Compose;

let transform = Compose::new(vec![
    Box::new(Resize::new(256, 256)),
    Box::new(CenterCrop::new(224)),
    Box::new(ImageNormalize::imagenet()),
]);

let processed = transform.apply(&image);
```

### datasets/

Built-in vision datasets.

#### SyntheticMNIST

Synthetic MNIST-like dataset for testing:

```rust
use axonml_vision::SyntheticMNIST;
use axonml_data::Dataset;

// Create dataset with 60000 samples
let train = SyntheticMNIST::new(60000);
let test = SyntheticMNIST::new(10000);

// Get a sample
let (image, label) = train.get(0).unwrap();
// image: [28, 28] tensor
// label: [10] one-hot tensor
```

#### SyntheticCIFAR

Synthetic CIFAR-like dataset:

```rust
use axonml_vision::SyntheticCIFAR;

let train = SyntheticCIFAR::new(50000);
let test = SyntheticCIFAR::new(10000);

let (image, label) = train.get(0).unwrap();
// image: [3, 32, 32] tensor (RGB)
// label: [10] one-hot tensor
```

### models/

Pre-built vision model architectures.

#### LeNet

Classic LeNet-5 architecture:

```rust
use axonml_vision::LeNet;

let model = LeNet::new(1, 10);  // 1 channel input, 10 classes

// Architecture:
// Conv2d(1, 6, 5) -> ReLU -> MaxPool2d(2)
// Conv2d(6, 16, 5) -> ReLU -> MaxPool2d(2)
// Flatten -> Linear(400, 120) -> ReLU
// Linear(120, 84) -> ReLU
// Linear(84, 10)
```

#### SimpleCNN

Flexible CNN for quick experiments:

```rust
use axonml_vision::SimpleCNN;

// For MNIST (1 channel, 10 classes)
let model = SimpleCNN::new(1, 10);

// For CIFAR (3 channels, 10 classes)
let model = SimpleCNN::new(3, 10);
```

### io.rs

Image I/O utilities (feature-gated).

```rust
use axonml_vision::io;

// Load image (requires 'image' feature)
let tensor = io::load_image("path/to/image.png")?;

// Save tensor as image
io::save_image(&tensor, "output.png")?;
```

## Usage Examples

### Training on MNIST

```rust
use axonml::prelude::*;

fn main() {
    // 1. Create dataset
    let train_data = SyntheticMNIST::new(60000);
    let test_data = SyntheticMNIST::new(10000);

    // 2. Create data loaders
    let train_loader = DataLoader::with_shuffle(train_data, 64, true);
    let test_loader = DataLoader::new(test_data, 64);

    // 3. Create model
    let model = SimpleCNN::new(1, 10);

    // 4. Create optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    // 5. Training loop
    for epoch in 0..10 {
        for batch in train_loader.iter() {
            // Reshape to [batch, 1, 28, 28]
            let input = batch.data.reshape(&[-1, 1, 28, 28]).unwrap();
            let input = Variable::new(input, true);

            let output = model.forward(&input);
            let loss = cross_entropy(&output, &batch.targets);

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}
```

### Image Classification Pipeline

```rust
use axonml::prelude::*;

// Define preprocessing
let preprocess = Compose::new(vec![
    Box::new(Resize::new(256, 256)),
    Box::new(CenterCrop::new(224)),
    Box::new(ImageNormalize::imagenet()),
]);

// Load and preprocess image
let image = io::load_image("image.jpg")?;
let processed = preprocess.apply(&image);

// Add batch dimension
let input = processed.unsqueeze(0)?;

// Inference
let output = model.forward(&Variable::new(input, false));
let prediction = output.argmax(1)?;
```

### Data Augmentation for Training

```rust
use axonml_vision::transforms::*;

// Training transforms (with augmentation)
let train_transform = Compose::new(vec![
    Box::new(RandomResizedCrop::new(224)),
    Box::new(RandomHorizontalFlip::new()),
    Box::new(ColorJitter::new(0.4, 0.4, 0.4, 0.1)),
    Box::new(ImageNormalize::imagenet()),
]);

// Validation transforms (no augmentation)
let val_transform = Compose::new(vec![
    Box::new(Resize::new(256, 256)),
    Box::new(CenterCrop::new(224)),
    Box::new(ImageNormalize::imagenet()),
]);
```

## ImageNet Normalization

Standard ImageNet normalization values:

```rust
// Mean (RGB): [0.485, 0.456, 0.406]
// Std (RGB): [0.229, 0.224, 0.225]

let normalize = ImageNormalize::imagenet();
```

## Feature Flags

- `image` - Enable image I/O using the `image` crate

## Related Modules

- [Data](../data/README.md) - DataLoader and Dataset traits
- [Neural Networks](../nn/README.md) - Conv2d, pooling layers
- [Autograd](../autograd/README.md) - Training with gradients

@version 0.1.0
@author AutomataNexus Development Team
