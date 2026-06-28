# axonml-vision Documentation

> Computer vision for the AxonML ML framework.

## Overview

`axonml-vision` is the AxonML counterpart to `torchvision`: image preprocessing
transforms, vision datasets (real + synthetic), a collection of
classification / detection / depth / anomaly / VQA model
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
- `FPN`

Anomaly / depth / VQA:

- `PatchCore`, `StudentTeacher` (anomaly detection)
- `DPT`, `FastDepth` (monocular depth)
- `VQAModel` (visual question answering)

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
    assign_fcos_targets, assign_single_scale_targets,
    compute_ap, compute_map, compute_coco_map,
    DetectionResult, GroundTruth, TrainConfig,
};
```

See the [Object Detection Training Guide](../detection.md) for full docs.

### `ops`

Low-level vision ops (e.g. NMS, IoU, anchor generation) used by the
detection models.

### `edge`

Edge-deployment helpers (quantization-friendly pipelines, frame differencing,
etc.).

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

### Object detection training (WIDER FACE)

Load a detection dataset and compose your own forward → target-assignment →
loss → step loop with the shared detection utilities:

```rust
use axonml_vision::datasets::WiderFaceDataset;
use axonml_vision::training::assign_single_scale_targets;

let dataset = WiderFaceDataset::new("data/wider_face", "train", (128, 128))?;

for i in 0..dataset.len() {
    let (image, faces) = dataset.get(i).unwrap();
    let frame = Variable::new(image.unsqueeze(0).unwrap(), true);
    let (cls_t, bbox_t) = assign_single_scale_targets(&faces, feat_h, feat_w, stride);
    // forward → loss(cls_t, bbox_t) → backward → optimizer step
}
```

See the [Object Detection Training Guide](../detection.md) for the full loop.

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

0.6.5 (2026-06-06)
