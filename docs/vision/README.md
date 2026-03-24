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

### image_io.rs

Load images from disk as CHW tensors normalized to `[0.0, 1.0]`:

```rust
use axonml_vision::image_io;

// Load at original resolution → [3, H, W]
let tensor = image_io::load_image("photo.jpg")?;

// Load and resize → [3, target_h, target_w]
let tensor = image_io::load_image_resized("photo.jpg", 320, 320)?;

// Load with original dimensions
let (tensor, (orig_h, orig_w)) = image_io::load_image_with_info("photo.jpg")?;

// Convert raw RGB bytes → [3, H, W]
let tensor = image_io::rgb_bytes_to_tensor(&rgb_data, 480, 640)?;
```

Supports JPEG, PNG, BMP via the `image` crate. All outputs are CHW layout, float32, normalized to `[0, 1]`.

#### CocoDataset

COCO format dataset loader for general object detection:

```rust
use axonml_vision::datasets::CocoDataset;

let dataset = CocoDataset::new(
    "data/coco/train2017",
    "data/coco/annotations/instances_train2017.json",
    (320, 320),
)?;

let (image, annotations) = dataset.get(0).unwrap();
// image: [3, 320, 320]
// annotations[i].bbox: [x1, y1, x2, y2] normalized [0, 1]
// annotations[i].category_id: 0-indexed class
```

Features: COCO JSON parsing, non-contiguous category ID remapping, crowd annotation filtering, on-demand image loading/resizing.

#### WiderFaceDataset

WIDER FACE dataset loader for face detection:

```rust
use axonml_vision::datasets::WiderFaceDataset;

let dataset = WiderFaceDataset::new(
    "data/wider_face", "train", (128, 128),
)?;

let (image, face_boxes) = dataset.get(0).unwrap();
// image: [3, 128, 128]
// face_boxes: Vec<[f32; 4]> — [x1, y1, x2, y2] in pixel coords
```

Parses WIDER FACE annotation format. Supports train/val splits.

### losses.rs

Detection-specific loss functions:

```rust
use axonml_vision::losses::{FocalLoss, GIoULoss, UncertaintyLoss, compute_centerness};

// Focal Loss — down-weights easy examples (alpha=0.25, gamma=2.0)
let focal = FocalLoss::new();
let loss = focal.compute(&pred_logits, &targets);

// GIoU Loss — bbox regression in IoU metric space
let loss = GIoULoss::compute(&pred_boxes, &target_boxes);

// Uncertainty Loss — learns prediction + aleatoric uncertainty
let loss = UncertaintyLoss::compute(&pred_mean, &pred_log_var, &target);

// FCOS centerness score
let score = compute_centerness(l, t, r, b);
```

### training/

Training infrastructure for detection models:

```rust
use axonml_vision::training::{
    nexus_training_step, phantom_training_step,       // Training loops
    assign_fcos_targets, assign_phantom_targets,       // Target assignment
    compute_ap, compute_map, compute_coco_map,         // Evaluation metrics
    DetectionResult, GroundTruth, TrainConfig,         // Types
};
```

See the [Object Detection Training Guide](../detection.md) for complete documentation.

### models/nexus/

Neuroscience-inspired dual-pathway object detector (~430K params):

```rust
use axonml_vision::models::nexus::{Nexus, NexusConfig};

let mut model = Nexus::new();

// Inference — returns detections with uncertainty estimates
let detections = model.detect(&frame);

// Training — returns raw multi-scale head outputs
let output = model.forward_train(&frame);
```

Key innovations: dual-pathway processing (ventral/dorsal), predictive coding, persistent GRU object memory, uncertainty quantification, 3-scale anchor-free heads.

### models/phantom/

Event-driven face detector for edge deployment (~126K params):

```rust
use axonml_vision::models::phantom::{Phantom, PhantomConfig};

let mut model = Phantom::new();

// Inference — returns faces with tracking IDs
let faces = model.detect_frame(&frame);

// Training — returns raw classification and bbox head outputs
let output = model.forward_train(&frame);
```

Key innovations: pseudo-event generation (frame differencing), sparse processing, GRU-based face tracking, confidence accumulation. Compute drops from 100% (cold start) to ~5% (steady state).

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

## Aegis Identity — Biometric Framework *(novel)*

A unified biometric identity system with 5 novel architectures. Total ~362K params, <2MB, each modality deployable independently on Raspberry Pi.

### Architecture Overview

| Model | Modality | Novel Idea | Params |
|-------|----------|------------|--------|
| **Mnemosyne** | Face | GRU hidden state converges to identity attractor over multiple frames | ~115K |
| **Ariadne** | Fingerprint | Ridge event fields with learned Gabor wavelet bank | ~65K |
| **Echo** | Voice | Identity = unpredictable speech residuals (predictive coding) | ~68K |
| **Argus** | Iris | Polar-native radial/angular Conv1d with phase encoding | ~65K |
| **Themis** | Fusion | Belief propagation with uncertainty gating + GRU belief accumulation | ~49K |

### Quick Start

```rust
use axonml_vision::models::biometric::{AegisIdentity, BiometricEvidence};

// Create system (full, face_only, or edge_minimal)
let mut aegis = AegisIdentity::full();

// Enroll
let evidence = BiometricEvidence::new()
    .with_face(face_tensor)
    .with_voice(voice_tensor);
aegis.enroll(1001, &evidence);

// Verify
let result = aegis.verify(1001, &probe_evidence);
println!("Match: {}, Score: {:.3}", result.is_match, result.match_score);

// Forensic verification with audit trail
let (result, forensic) = aegis.verify_forensic(1001, &probe);

// Secure pipeline: quality → liveness → verification
let secure = aegis.secure_verify(1001, &evidence);
```

### Key Capabilities

- **Temporal Crystallization** — Identity emerges from GRU convergence, not single-shot embeddings
- **Predictive Residual Identity** — Voice identity is what a speech predictor cannot predict
- **Belief Propagation Fusion** — Uncertainty-aware multimodal fusion, not score averaging
- **Forensic Verification** — Per-modality breakdown, cross-modal consistency, dimension contributions
- **Identity Drift Detection** — Monitor template aging via embedding trajectory tracking
- **Liveness Detection** — Anti-spoofing via GRU trajectory + replay detection
- **Graceful Degradation** — Any subset of modalities works; missing modalities get zero weight

### Object Detection Training

```rust
use axonml_vision::models::phantom::Phantom;
use axonml_vision::datasets::WiderFaceDataset;
use axonml_vision::training::phantom_training_step;

let dataset = WiderFaceDataset::new("data/wider_face", "train", (128, 128))?;
let mut model = Phantom::new();
let mut optimizer = Adam::new(model.parameters(), 1e-4);

for epoch in 0..50 {
    for i in 0..dataset.len() {
        let (image, faces) = dataset.get(i).unwrap();
        let frame = Variable::new(image.unsqueeze(0).unwrap(), true);
        let loss = phantom_training_step(&mut model, &frame, &faces, &mut optimizer);
    }
}
```

See the [Object Detection Training Guide](../detection.md) for Nexus + COCO examples, loss details, target assignment, and mAP evaluation.

### models/nightvision/

Multi-domain infrared object detection for thermal imagery (~200K-500K params depending on config):

```rust
use axonml_vision::models::nightvision::{NightVision, NightVisionConfig, ThermalDomain};

// Wildlife detection — single-channel thermal, animal species classes
let model = NightVision::new(NightVisionConfig::wildlife(20));

// Human detection — search & rescue, perimeter security
let model = NightVision::new(NightVisionConfig::human());

// Interstellar — multi-band IR, astronomical thermal sources
let model = NightVision::new(NightVisionConfig::interstellar(3, 3));

// Vehicle detection — engine heat, tire friction
let model = NightVision::new(NightVisionConfig::multi_domain(50));

// Edge deployment — compact model
let model = NightVision::new(NightVisionConfig::edge(10));

// Forward pass — returns per-scale (cls, bbox, obj, domain) tuples
let outputs = model.forward_detection(&ir_image);

// Flattened forward — concatenates across FPN scales
let (cls, bbox, obj) = model.forward_flat(&ir_image);
```

Key innovations:
- **Thermal-adaptive stem** — handles single-channel (1-ch) or multi-band (3-ch) IR input
- **CSP backbone** — Cross-Stage Partial blocks for efficient thermal feature extraction
- **Thermal FPN** — Feature Pyramid Network with top-down + lateral connections (P3/P4/P5)
- **Decoupled heads** — YOLOX-style separate classification, bbox, and objectness branches
- **Domain tagging** — optional domain classification head (wildlife/human/interstellar/vehicle/general)

Five thermal domains: `Wildlife`, `Human`, `Interstellar`, `Vehicle`, `General`.

---

## Biometric Training Pipelines

The Aegis Biometric Suite includes GPU-accelerated training pipelines with checkpoint/resume support for all models.

### Mnemosyne — Face Verification on LFW

Train the temporal crystallization face model on Labeled Faces in the Wild:

```bash
# Training
cargo run --example train_mnemosyne --release -p axonml-vision -- \
  --data-dir /opt/datasets/lfw/processed \
  --epochs 100 --lr 0.001 --batch-size 8

# Benchmarking (verification pairs, ROC-AUC, EER)
cargo run --example bench_mnemosyne --release -p axonml-vision -- \
  --model /opt/AxonML/checkpoints/mnemosyne/best_model.axonml \
  --pairs 3000 --seq-len 5
```

The `bench_mnemosyne` example evaluates face verification accuracy on same-identity vs different-identity pairs. Reports: accuracy at multiple thresholds, ROC-AUC, EER, FAR/FRR, and F1.

### Argus — Iris Recognition on CASIA

Train the polar-native radial/angular iris model on CASIA-Iris:

```bash
cargo run --example train_argus --release -p axonml-vision -- \
  --data-dir /opt/datasets/casia-iris/processed \
  --epochs 80 --lr 0.0005
```

### Ariadne — Fingerprint Verification on FVC2000

Train the ridge event field fingerprint model on FVC2000:

```bash
cargo run --example train_ariadne --release -p axonml-vision -- \
  --data-dir /opt/datasets/fvc2000/processed \
  --epochs 60 --lr 0.001
```

All biometric training pipelines support:
- **GPU acceleration** — automatic device placement (CPU/CUDA)
- **Checkpoint/resume** — saves best model + periodic checkpoints
- **Training monitor** — live loss/accuracy tracking
- **State dict serialization** — load/save via `axonml-serialize`

---

## Feature Flags

- `image` - Enable image I/O using the `image` crate

## Related Modules

- [Data](../data/README.md) - DataLoader and Dataset traits
- [Neural Networks](../nn/README.md) - Conv2d, pooling layers
- [Autograd](../autograd/README.md) - Training with gradients

@version 0.4.1
@author AutomataNexus Development Team
