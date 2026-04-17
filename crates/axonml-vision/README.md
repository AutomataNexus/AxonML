# axonml-vision

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-vision"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-vision** provides the computer-vision stack for AxonML: image-specific transforms, loaders for classical vision datasets (MNIST, Fashion-MNIST, CIFAR-10/100) plus synthetic variants, and a wide catalog of neural-network architectures covering classification, detection, dense prediction, anomaly detection, VQA, 3D reconstruction, and biometrics. A pretrained-weights hub with on-disk caching rounds it out.

## Features

- **Image transforms** — `Resize`, `CenterCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `ColorJitter`, `Grayscale`, `ImageNormalize` (presets: `imagenet`, `mnist`, `cifar10`), `Pad`, `ToTensorImage`.
- **Datasets** — real-file loaders for `MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`, and synthetic variants `SyntheticMNIST` / `SyntheticCIFAR` for fast tests.
- **Classification** — `LeNet`, `MLP`, `SimpleCNN`, `ResNet` (`resnet18`, `resnet34`, `BasicBlock`, `Bottleneck`), `VGG` (`vgg11`, `vgg13`, `vgg16`, `vgg19` with optional batch-norm), `VisionTransformer` (`vit_base`, `vit_large`).
- **Detection** — `BlazeFace` (dual-scale 128×128 face detector, 896 anchors), `RetinaFace` (ResNet34 backbone + multi-level FPN head), `DETR` (transformer-based, `small` preset), `NanoDet` (mobile-class detector), `Helios` (YOLO-family detector with 5 sizes Nano/Small/Medium/Large/XLarge and loss utilities `HeliosLoss`, `CIoULoss`, `TaskAlignedAssigner`).
- **Novel detection architectures** — `Nexus` (predictive dual-pathway detector with multi-scale fusion, object-memory bank, and predictive-coding surprise gating) and `Phantom` (temporal event-driven face detection with pseudo-event encoder and GRU-based face-state tracker). `NightVision` (multi-domain infrared detector with thermal stem, CSP backbone, thermal FPN, YOLOX-style decoupled heads, `ThermalDomain` domain tagging).
- **Dense prediction** — `DPT` (depth transformer, `small`/`base` presets) and `FastDepth` (mobile depth estimator).
- **Anomaly detection** — `PatchCore` and `StudentTeacher`, both with `default_rgb()` constructors.
- **Visual Question Answering** — `VQAModel` (`small` preset).
- **3D reconstruction** — `Aegis3D`: Fourier-feature SDF networks (`LocalSDF` + `GlobalSDF`), adaptive octree spatial indexing, differentiable sphere-tracing renderer, and marching-cubes mesh extraction.
- **FPN infrastructure** — shared `FPN` (feature pyramid network) used by multiple detectors.
- **Aegis biometric identity suite** — `AegisIdentity` orchestrator with `full` / `face_only` / `edge_minimal` constructors; modality models `MnemosyneIdentity` (face), `AriadneFingerprint`, `EchoSpeaker` (voice), `ArgusIris`, plus `ThemisFusion` (uncertainty-weighted fusion). Enrollment, verification, forensic verification, liveness, secure verification, identification. Companion losses: `AngularMarginLoss`, `CenterLoss`, `ContrastiveLoss`, `CrystallizationLoss`, `DiversityRegularization`, `EchoLoss`, `ArgusLoss`, `LivenessLoss`, `ThemisLoss`.
- **Model Hub** — `download_weights`, `load_state_dict`, `list_models`, `model_info`, `is_cached`, `model_registry`, with on-disk caching.
- **CUDA feature** — optional `cuda` cargo feature propagates to core/tensor/autograd/nn.

## Modules

| Module | Description |
|--------|-------------|
| `transforms` | Image data-augmentation and preprocessing transforms |
| `datasets` | MNIST, Fashion-MNIST, CIFAR-10/100 loaders plus synthetic variants |
| `models` | All neural network architectures (see below) |
| `models::biometric` | Aegis biometric suite (Mnemosyne, Ariadne, Echo, Argus, Themis + identity orchestrator) |
| `models::helios` | YOLO-style object detector with 5 size variants |
| `models::nexus` | Predictive dual-pathway detector with object memory |
| `models::phantom` | Temporal event-driven face detection |
| `models::nightvision` | Multi-domain infrared detection |
| `models::aegis3d` | Octree-adaptive neural implicit surface reconstruction |
| `camera` | Camera I/O utilities |
| `edge` | Edge-deployment helpers |
| `hub` | Pretrained model weights management |
| `image_io` | Image load/save helpers |
| `losses` | Vision-specific loss functions |
| `ops` | Low-level vision ops |
| `training` | Training utilities |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-vision = "0.6.1"
```

### Loading Datasets

```rust
use axonml_vision::prelude::*;

// Synthetic MNIST for fast tests
let train_data = SyntheticMNIST::train();
let test_data  = SyntheticMNIST::test();

// Synthetic CIFAR-10
let cifar = SyntheticCIFAR::small();

let (image, label) = train_data.get(0).unwrap();
assert_eq!(image.shape(), &[1, 28, 28]);  // MNIST: 1 channel, 28x28
assert_eq!(label.shape(), &[10]);          // One-hot encoded
```

### Image Transforms

```rust
use axonml_vision::{Resize, CenterCrop, RandomHorizontalFlip, ImageNormalize};
use axonml_data::{Compose, Transform};

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

let imagenet = ImageNormalize::imagenet();  // mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
let mnist    = ImageNormalize::mnist();     // mean=[0.1307] std=[0.3081]
let cifar10  = ImageNormalize::cifar10();   // mean=[0.4914,0.4822,0.4465] std=[0.2470,0.2435,0.2616]
```

### Classification Models

```rust
use axonml_vision::{LeNet, MLP, SimpleCNN};
use axonml_vision::models::{resnet18, resnet34, vgg16, vit_base};
use axonml_nn::Module;
use axonml_autograd::Variable;

let lenet   = LeNet::new();                       // [N, 1, 28, 28] -> [N, 10]
let mlp     = MLP::for_mnist();                   // 784 -> 256 -> 128 -> 10
let rn18    = resnet18(1000);                     // ImageNet classes
let vgg     = vgg16(1000, /*batch_norm=*/ true);
let vit     = vit_base(1000);
```

### Detection Models

```rust
use axonml_vision::models::{BlazeFace, RetinaFace, NanoDet, DETR};
use axonml_vision::models::helios::{Helios, HeliosSize};

let blaze   = BlazeFace::new();                   // dual-scale 128x128 face detector
let retina  = RetinaFace::new();                  // ResNet34 backbone
let nanodet = NanoDet::new(/*num_classes=*/ 80);
let detr    = DETR::small(10);
let helios  = Helios::small(80);                  // also: new(config), large(num_classes)
```

### Novel Detection Architectures

```rust
use axonml_vision::models::{Nexus, Phantom, NightVision, NightVisionConfig};

let nexus   = Nexus::default(); // predictive dual-pathway + object memory
let phantom = Phantom::default(); // event-driven temporal face detector
let night   = NightVision::new(NightVisionConfig::default());
```

### Dense Prediction & Anomaly / VQA

```rust
use axonml_vision::models::{DPT, FastDepth, PatchCore, StudentTeacher, VQAModel};

let dpt       = DPT::small();                         // transformer depth
let fast      = FastDepth::new();                     // mobile depth
let patch     = PatchCore::default_rgb();             // anomaly detection, 256-d features
let st        = StudentTeacher::default_rgb();        // student-teacher anomaly
let vqa       = VQAModel::small(100, 50);             // vocab=100, answers=50
```

### Aegis3D — 3D Reconstruction

```rust
use axonml_vision::models::{Aegis3D, aegis3d::{Aegis3DConfig, AABB, SphereTracingConfig}};

let aegis3d = Aegis3D::new(); // Fourier-feature SDF + adaptive octree + sphere tracing + marching cubes
```

### Full Training Pipeline

```rust
use axonml_vision::prelude::*;
use axonml_data::DataLoader;
use axonml_optim::{Adam, Optimizer};
use axonml_nn::{CrossEntropyLoss, Module};

let dataset = SyntheticMNIST::train();
let loader  = DataLoader::new(dataset, 32).shuffle(true);

let model        = LeNet::new();
let mut optim    = Adam::new(model.parameters(), 0.001);
let loss_fn      = CrossEntropyLoss::new();

for batch in loader.iter() {
    let input  = Variable::new(batch.data, true);
    let target = batch.targets;

    optim.zero_grad();
    let output = model.forward(&input);
    let loss   = loss_fn.compute(&output, &target);
    loss.backward();
    optim.step();
}
```

### Model Hub for Pretrained Weights

```rust
use axonml_vision::hub::{
    download_weights, load_state_dict, list_models, model_info, is_cached, model_registry,
};

for model in list_models() {
    println!("{}: {} classes, {:.1} MB", model.name, model.num_classes,
             model.size_bytes as f64 / 1_000_000.0);
}

if let Some(info) = model_info("resnet18") {
    println!("Top-1 accuracy: {:.2}%", info.accuracy);
}

if !is_cached("resnet18") {
    let path = download_weights("resnet18", /*force=*/ false)?;
    let state_dict = load_state_dict(&path)?;
    // model.load_state_dict(state_dict);
}
```

### Aegis Identity — Biometric Framework

Unified biometric identity system with 5 modality-specific architectures plus `ThemisFusion` for uncertainty-weighted evidence fusion. Designed for edge deployment (sub-2 MB total in `edge_minimal` configuration).

```rust
use axonml_vision::models::biometric::{
    AegisIdentity, BiometricEvidence, BiometricModality,
};
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// Full multimodal system — face + fingerprint + voice + iris
let mut aegis = AegisIdentity::full();
// Or smaller deployments:
let face_only = AegisIdentity::face_only();
let edge      = AegisIdentity::edge_minimal();

// Enroll
let face     = Variable::new(Tensor::randn(&[1, 3, 64, 64]), false);
let evidence = BiometricEvidence::new().with_face(face);
let enrolled = aegis.enroll(1001, &evidence);

// Verify
let probe        = BiometricEvidence::new()
    .with_face(Variable::new(Tensor::randn(&[1, 3, 64, 64]), false));
let verification = aegis.verify(1001, &probe);
println!("match={}, score={:.3}, confidence={:.3}",
    verification.is_match, verification.match_score, verification.confidence);

// Forensic verification with per-modality scores and cross-modal consistency
let (result, forensic) = aegis.verify_forensic(1001, &probe);

// Anti-spoofing liveness
let liveness = aegis.assess_liveness(&evidence);

// Quality -> liveness -> verification secure pipeline
let secure = aegis.secure_verify(1001, &evidence);

// 1:N identification
let ident = aegis.identify(&probe);
```

**Modality architectures:**

| Model | Modality | Novel idea |
|-------|----------|------------|
| `MnemosyneIdentity` | Face | Identity crystallizes via GRU attractor convergence |
| `AriadneFingerprint` | Fingerprint | Ridge event fields with Gabor wavelets |
| `EchoSpeaker` | Voice | Identity = unpredictable speech residuals |
| `ArgusIris` | Iris | Polar-native radial / angular Conv1d encoding (backed by `polar::polar_unwrap`) |
| `ThemisFusion` | Fusion | Belief propagation with uncertainty gating |

## Features flags

- `default = ["download"]` — enables `reqwest` for hub downloads.
- `cuda` — propagates CUDA support to `axonml-tensor`, `axonml-nn`, `axonml-autograd`, `axonml-core`.

## Tests

```bash
cargo test -p axonml-vision
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

_Last updated: 2026-04-16 (v0.6.1)_
