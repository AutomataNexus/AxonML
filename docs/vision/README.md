# axonml-vision Documentation

> Computer vision for the AxonML ML framework.

## Overview

`axonml-vision` is the AxonML counterpart to `torchvision`: image preprocessing
transforms, vision datasets (real + synthetic), a large collection of
classification / detection / depth / anomaly / biometric / 3D / VQA model
architectures, an image IO module, training utilities for detection, and a
pretrained-model hub.

## Modules

### `transforms`

All transforms implement `axonml_data::Transform`.

| Transform                                      | Notes                                                   |
|------------------------------------------------|---------------------------------------------------------|
| `Resize(h, w)`                                 | Bilinear interpolation on 2D/3D/4D tensors              |
| `CenterCrop(h, w)`                             | Extract central region                                  |
| `RandomHorizontalFlip` / `RandomVerticalFlip`  | Stochastic mirroring                                    |
| `RandomRotation`                               | 90-degree increments                                    |
| `ColorJitter(brightness, contrast, saturation)`| Random color adjustments                                |
| `Grayscale`                                    | RGB -> 1-channel via BT.601 coefficients                |
| `ImageNormalize::new(mean, std)`               | Per-channel normalize; `::imagenet()`, `::mnist()`, `::cifar10()` presets |
| `Pad(size, value)`                             | Constant-value borders                                  |
| `ToTensorImage`                                | Rescale [0, 255] -> [0, 1]                              |

Compose via `axonml_data::Compose`:

```rust
use axonml_data::Compose;
use axonml_vision::transforms::*;

let transform = Compose::empty()
    .add(Resize::new(256, 256))
    .add(CenterCrop::new(224, 224))
    .add(ImageNormalize::imagenet());
```

### `datasets`

- `MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100` — real datasets (disk-backed)
- `SyntheticMNIST`, `SyntheticCIFAR` — synthetic variants for smoke tests
- `CocoDataset` — COCO-format object detection dataset
- `WiderFaceDataset` — WIDER FACE detection dataset

```rust
use axonml_vision::{SyntheticMNIST, SyntheticCIFAR};
use axonml_data::Dataset;

let train = SyntheticMNIST::new(60000);
let test  = SyntheticMNIST::new(10000);
let (image, label) = train.get(0).unwrap(); // image: [1, 28, 28], label: [10] one-hot

let train = SyntheticCIFAR::new(50000);
let (image, label) = train.get(0).unwrap(); // image: [3, 32, 32], label: [10]
```

```rust
use axonml_vision::datasets::CocoDataset;
let ds = CocoDataset::new(
    "data/coco/train2017",
    "data/coco/annotations/instances_train2017.json",
    (320, 320),
)?;
```

### `image_io`

Image loading with the `image` crate. All outputs are CHW f32, normalised to
[0.0, 1.0].

```rust
use axonml_vision::image_io;

let tensor = image_io::load_image("photo.jpg")?;
let tensor = image_io::load_image_resized("photo.jpg", 320, 320)?;
let (tensor, (orig_h, orig_w)) = image_io::load_image_with_info("photo.jpg")?;
let tensor = image_io::rgb_bytes_to_tensor(&rgb_data, 480, 640)?;
```

Supports JPEG / PNG / BMP via the `image` crate.

### `models`

Classification:

- `LeNet`, `MLP`, `SimpleCNN`
- `ResNet`, `resnet18`, `resnet34` (with `BasicBlock` / `Bottleneck`)
- `VGG`, `vgg11`, `vgg13`, `vgg16`, `vgg19` (`VggFeatures` + `VggClassifier`)
- `VisionTransformer`, `vit_base`, `vit_large` (+ reusable
  `Transformer*Layer` building blocks)

Detection / detection-infrastructure:

- `BlazeFace`, `RetinaFace`, `DETR`, `NanoDet`
- `Helios` + `HeliosLoss`, `CIoULoss`, `TaskAlignedAssigner`
- `FPN`
- Novel: `Nexus` (dual-pathway predictive coder), `Phantom` (event-driven
  edge face detector), `NightVision` (thermal IR multi-domain detector)

Anomaly / depth / VQA / 3D:

- `PatchCore`, `StudentTeacher` (anomaly detection)
- `DPT`, `FastDepth` (monocular depth)
- `VQAModel` (visual question answering)
- `Aegis3D` (3D reconstruction)

Biometric (Aegis Identity suite):

- `AegisIdentity` — fusion entry point
- `MnemosyneIdentity` — face (temporal GRU attractor)
- `AriadneFingerprint` — fingerprint (Gabor ridge events)
- `EchoSpeaker` — voice (predictive-coding residual identity)
- `ArgusIris` — iris (polar radial/angular conv)
- `ThemisFusion` — multimodal belief-propagation fusion
- `IdentityBank`, `BiometricEvidence`, `BiometricConfig`, `BiometricModality`
- Losses: `AngularMarginLoss`, `ArgusLoss`, `CenterLoss`, `ContrastiveLoss`,
  `CrystallizationLoss`, `DiversityRegularization`, `EchoLoss`,
  `LivenessLoss`, `ThemisLoss`
- Results: `EnrollmentResult`, `IdentificationResult`, `VerificationResult`

### `losses`

Detection-specific losses.

```rust
use axonml_vision::losses::{FocalLoss, GIoULoss, UncertaintyLoss, compute_centerness};

let focal = FocalLoss::new();
let loss = focal.compute(&pred_logits, &targets);

let loss = GIoULoss::compute(&pred_boxes, &target_boxes);
let loss = UncertaintyLoss::compute(&pred_mean, &pred_log_var, &target);
let score = compute_centerness(l, t, r, b);
```

### `training`

Training infrastructure for detection models:

```rust
use axonml_vision::training::{
    nexus_training_step, phantom_training_step,
    assign_fcos_targets, assign_phantom_targets,
    compute_ap, compute_map, compute_coco_map,
    DetectionResult, GroundTruth, TrainConfig,
};
```

See the [Object Detection Training Guide](../detection.md) for full docs.

### `ops`

Low-level vision ops (e.g. NMS, IoU, anchor generation) used by the
detection models.

### `edge`

Edge-deployment helpers (quantization-friendly pipelines, frame differencing
for Phantom, etc.).

### `camera`

Camera capture utilities and frame-buffer bridging for live inference.

### `hub`

`PretrainedModel`, `StateDict`, `cache_dir`, `download_weights`, `is_cached`,
`list_models`, `load_state_dict`, `model_info`, `model_registry`,
`HubError`, `HubResult`.

## Usage

### Training on synthetic MNIST

```rust
use axonml::prelude::*;
use axonml_vision::{SyntheticMNIST, SimpleCNN};

let train = SyntheticMNIST::new(60000);
let loader = DataLoader::with_shuffle(train, 64, true);

let model = SimpleCNN::new(1, 10);
let mut opt = Adam::new(model.parameters(), 0.001);

for epoch in 0..10 {
    for batch in loader.iter() {
        let input = batch.data.reshape(&[-1, 1, 28, 28]).unwrap();
        let input = Variable::new(input, true);
        let output = model.forward(&input);
        let loss = cross_entropy(&output, &batch.targets);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }
}
```

### Image classification pipeline

```rust
use axonml_data::Compose;
use axonml_vision::{transforms::*, image_io};

let preprocess = Compose::empty()
    .add(Resize::new(256, 256))
    .add(CenterCrop::new(224, 224))
    .add(ImageNormalize::imagenet());

let image = image_io::load_image("image.jpg")?;
let processed = preprocess.apply(&image);
let input = processed.unsqueeze(0)?;
let output = model.forward(&Variable::new(input, false));
let prediction = output.argmax(1)?;
```

### Aegis Identity — biometric verification

```rust
use axonml_vision::models::biometric::{AegisIdentity, BiometricEvidence};

let mut aegis = AegisIdentity::full(); // or ::face_only() / ::edge_minimal()

let evidence = BiometricEvidence::new()
    .with_face(face_tensor)
    .with_voice(voice_tensor);
aegis.enroll(1001, &evidence);

let result = aegis.verify(1001, &probe_evidence);
println!("match: {}, score: {:.3}", result.is_match, result.match_score);

let (result, forensic) = aegis.verify_forensic(1001, &probe);
let secure = aegis.secure_verify(1001, &evidence);
```

Five modalities, ~362K total params, <2MB — any subset is deployable
independently on edge hardware.

### NightVision thermal detection

```rust
use axonml_vision::models::nightvision::{NightVision, NightVisionConfig, ThermalDomain};

let model = NightVision::new(NightVisionConfig::wildlife(20));
let outputs = model.forward_detection(&ir_image);
let (cls, bbox, obj) = model.forward_flat(&ir_image);
```

Five thermal domains: `Wildlife`, `Human`, `Interstellar`, `Vehicle`,
`General`. Handles single-channel (1-ch) or multi-band (3-ch) IR input,
CSP backbone, thermal FPN (P3/P4/P5), decoupled YOLOX-style heads, and an
optional domain-tag head.

### Object detection training (Phantom + WIDER FACE)

```rust
use axonml_vision::models::phantom::Phantom;
use axonml_vision::datasets::WiderFaceDataset;
use axonml_vision::training::phantom_training_step;

let dataset = WiderFaceDataset::new("data/wider_face", "train", (128, 128))?;
let mut model = Phantom::new();
let mut opt = Adam::new(model.parameters(), 1e-4);

for _ in 0..50 {
    for i in 0..dataset.len() {
        let (image, faces) = dataset.get(i).unwrap();
        let frame = Variable::new(image.unsqueeze(0).unwrap(), true);
        let _loss = phantom_training_step(&mut model, &frame, &faces, &mut opt);
    }
}
```

## Biometric Training Examples

Training pipelines (`examples/train_mnemosyne.rs`, `train_argus.rs`,
`train_ariadne.rs`) for the Aegis suite ship checkpoint/resume + training
monitor + state-dict serialisation. See
`/opt/AxonML/crates/axonml-vision/examples/`.

```bash
cargo run --example train_mnemosyne --release -p axonml-vision -- \
  --data-dir /opt/datasets/lfw/processed \
  --epochs 100 --lr 0.001 --batch-size 8
```

## ImageNet Normalization

```rust
// mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
let normalize = ImageNormalize::imagenet();
```

## Feature Flags

- `image` — enable image IO via the `image` crate

## Related Modules

- [Data](../../crates/axonml-data) — `DataLoader`, `Dataset`, `Transform`,
  `Compose`
- [Neural Networks](../nn/README.md) — `Conv2d`, pooling, etc.
- [Autograd](../autograd/README.md) — training with gradients

## Last updated

0.6.1 (2026-04-16)
