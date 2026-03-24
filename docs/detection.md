---
layout: default
title: Object Detection Training
nav_order: 6
description: "Training object detection and face detection models with AxonML"
---

# Object Detection Training
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AxonML provides end-to-end training infrastructure for anchor-free object detection models. The system includes image loading, dataset parsers (COCO, WIDER FACE), detection-specific losses (Focal, GIoU, Uncertainty), FCOS-style target assignment, complete training loops, and AP/mAP evaluation metrics.

Three built-in detector architectures are trainable out of the box:

| Model | Task | Architecture | Params | Target Size |
|:------|:-----|:-------------|:-------|:------------|
| **Nexus** | General object detection | Dual-pathway + predictive coding + object memory | ~430K | 320x320 |
| **Phantom** | Face detection | Event-driven + sparse processing + face tracking | ~126K | 128x128 |
| **NightVision** | Multi-domain IR detection | CSP backbone + Thermal FPN + decoupled heads | ~200K-500K | 320x320 |

Nexus and Phantom use FCOS-style anchor-free detection heads. NightVision uses YOLOX-style decoupled heads. All are designed for edge deployment.

---

## Image Loading

Load images from disk as CHW tensors normalized to `[0.0, 1.0]`:

```rust
use axonml_vision::image_io;

// Load image at original resolution → [3, H, W]
let tensor = image_io::load_image("photo.jpg")?;

// Load and resize → [3, target_h, target_w]
let tensor = image_io::load_image_resized("photo.jpg", 320, 320)?;

// Load with original dimensions returned
let (tensor, (orig_h, orig_w)) = image_io::load_image_with_info("photo.jpg")?;

// Convert raw RGB bytes (e.g., from a camera) → [3, H, W]
let tensor = image_io::rgb_bytes_to_tensor(&rgb_data, 480, 640)?;
```

All functions return `Tensor<f32>` in CHW layout with values in `[0.0, 1.0]`. Supports JPEG, PNG, BMP, and other formats via the `image` crate.

---

## Datasets

### COCO Dataset

For general object detection with 80 categories:

```rust
use axonml_vision::datasets::CocoDataset;

let dataset = CocoDataset::new(
    "data/coco/train2017",                      // image directory
    "data/coco/annotations/instances_train2017.json",  // annotation file
    (320, 320),                                  // target size (H, W)
)?;

println!("Images: {}", dataset.len());
println!("Classes: {}", dataset.num_classes());

// Get a sample: image tensor + annotations
let (image, annotations) = dataset.get(0).unwrap();
// image: [3, 320, 320] normalized to [0, 1]

for ann in &annotations {
    // ann.bbox: [x1, y1, x2, y2] normalized to [0, 1]
    // ann.category_id: 0-indexed class ID (remapped from COCO's non-contiguous IDs)
}
```

**Expected directory structure:**

```
data/coco/
  train2017/
    000000000001.jpg
    000000000002.jpg
    ...
  annotations/
    instances_train2017.json
```

**Features:**
- Parses standard COCO JSON format (images, annotations, categories)
- Remaps non-contiguous COCO category IDs (1-90) to contiguous 0-indexed IDs
- Filters out crowd annotations (`iscrowd=0` only)
- Normalizes bounding boxes from COCO `[x, y, w, h]` to `[x1, y1, x2, y2]` in `[0, 1]`
- Loads and resizes images on demand

### WIDER FACE Dataset

For face detection training:

```rust
use axonml_vision::datasets::WiderFaceDataset;

let dataset = WiderFaceDataset::new(
    "data/wider_face",    // root directory
    "train",              // split: "train" or "val"
    (128, 128),           // target size (H, W)
)?;

println!("Images: {}", dataset.len());

// Get a sample: image tensor + face bounding boxes
let (image, face_boxes) = dataset.get(0).unwrap();
// image: [3, 128, 128] normalized to [0, 1]

for bbox in &face_boxes {
    // bbox: [x1, y1, x2, y2] in pixel coordinates (scaled to target size)
}

// Access raw annotation data
let entry = dataset.get_annotation(0).unwrap();
println!("Original path: {:?}", entry.image_path);
```

**Expected directory structure:**

```
data/wider_face/
  WIDER_train/images/
    0--Parade/0_Parade_001.jpg
    1--Handshaking/...
    ...
  WIDER_val/images/
    ...
  wider_face_split/
    wider_face_train_bbx_gt.txt
    wider_face_val_bbx_gt.txt
```

**WIDER FACE annotation format (parsed automatically):**

```text
0--Parade/0_Parade_marchingband_1_849.jpg
1
449 330 122 149 0 0 0 0 0 0
```

Each entry: image path, number of faces, then one line per face with `x y w h blur expression illumination invalid occlusion pose`.

---

## Detection Losses

### Focal Loss

Down-weights easy examples and focuses training on hard negatives. Essential for detection where background vastly outnumbers objects:

```rust
use axonml_vision::losses::FocalLoss;

let focal = FocalLoss::new();                    // alpha=0.25, gamma=2.0
let focal = FocalLoss::with_params(0.25, 2.0);   // custom params

// pred_logits: raw logits before sigmoid [N]
// targets: binary labels {0, 1} [N]
let loss = focal.compute(&pred_logits, &targets);
```

**Formula:** `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`

- `alpha` (default 0.25): Balancing factor for positive vs. negative classes
- `gamma` (default 2.0): Focusing parameter. Higher values = more focus on hard examples

### GIoU Loss

Generalized Intersection-over-Union loss for bounding box regression. Better gradient signal than L1/L2 for non-overlapping boxes:

```rust
use axonml_vision::losses::GIoULoss;

// pred: [N, 4] as (x1, y1, x2, y2) in pixel coordinates
// target: [N, 4] as (x1, y1, x2, y2) in pixel coordinates
let loss = GIoULoss::compute(&pred_boxes, &target_boxes);
```

**Formula:** `Loss = 1 - GIoU` where `GIoU = IoU - (C - union) / C`

- `C` is the area of the smallest enclosing box
- Returns scalar loss (mean reduction)

### Uncertainty Loss

Learns both prediction and aleatoric uncertainty. The model outputs a mean and log-variance for each prediction:

```rust
use axonml_vision::losses::UncertaintyLoss;

// pred_mean, pred_log_var, target: all [N, D]
let loss = UncertaintyLoss::compute(&pred_mean, &pred_log_var, &target);
```

**Formula:** `L = 0.5 * exp(-log_var) * (pred - target)^2 + 0.5 * log_var`

This naturally balances the loss: high uncertainty reduces the penalty for inaccurate predictions, while the `log_var` term penalizes excessive uncertainty.

### Centerness

FCOS-style centerness score for weighting detection quality:

```rust
use axonml_vision::losses::compute_centerness;

// l, t, r, b = distances from location to box edges
let score = compute_centerness(l, t, r, b);
// Returns: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
```

---

## Target Assignment

### FCOS Target Assignment (Multi-Scale)

Used by **Nexus** for multi-scale detection. Assigns ground truth boxes to spatial locations on feature maps based on center-point containment and size ranges:

```rust
use axonml_vision::training::{assign_fcos_targets, fcos_targets_to_tensors};

let gt_boxes: Vec<[f32; 4]> = vec![[10.0, 20.0, 50.0, 80.0]];  // pixel coords
let gt_classes: Vec<usize> = vec![3];

// Feature map sizes at each scale
let feat_sizes = vec![(40, 40), (20, 20), (10, 10)];
let strides = vec![8.0, 16.0, 32.0];
let size_ranges = vec![(0.0, 64.0), (64.0, 128.0), (128.0, f32::INFINITY)];

let targets = assign_fcos_targets(
    &gt_boxes, &gt_classes,
    &feat_sizes, &strides, &size_ranges,
);
// targets: Vec<Vec<FcosTarget>> — one vec per scale

// Convert to tensors for loss computation
let tensor_targets = fcos_targets_to_tensors(&targets);
// Returns: Vec<(cls_tensor, bbox_tensor, centerness_tensor)>
```

**Algorithm:**
1. For each spatial location `(fx, fy)` on each scale, convert to image coordinates: `(fx + 0.5) * stride`
2. Check if the location falls inside any GT box (center-point assignment)
3. If multiple boxes match, assign the smallest-area box
4. Compute LTRB (left, top, right, bottom) distances from location to box edges
5. Check size constraint: `max(l, t, r, b)` must be within the scale's size range
6. Compute centerness score

**Default scale configuration:**

| Scale | Stride | Object Size Range |
|:------|:-------|:------------------|
| 0 | 8 | [0, 64] pixels |
| 1 | 16 | [64, 128] pixels |
| 2 | 32 | [128, infinity] pixels |

### Phantom Target Assignment (Single-Scale)

Used by **Phantom** for single-scale face detection:

```rust
use axonml_vision::training::assign_phantom_targets;

let gt_faces: Vec<[f32; 4]> = vec![[10.0, 15.0, 40.0, 50.0]];  // pixel coords
let feat_h = 32;
let feat_w = 32;
let stride = 4.0;

let (cls_target, bbox_target) = assign_phantom_targets(
    &gt_faces, feat_h, feat_w, stride,
);
// cls_target: [H, W] — 1.0 at face center cells, 0.0 elsewhere
// bbox_target: [H, W, 4] — [dx, dy, log_w, log_h] at positive cells
```

**Bbox target encoding:**
- `dx = (face_cx - cell_cx) / stride` — horizontal offset
- `dy = (face_cy - cell_cy) / stride` — vertical offset
- `log_w = ln(face_w / stride)` — log-space width
- `log_h = ln(face_h / stride)` — log-space height

---

## Training Loops

### Nexus Training (General Object Detection)

```rust
use axonml_vision::models::nexus::Nexus;
use axonml_vision::training::nexus_training_step;
use axonml_optim::Adam;

let mut model = Nexus::new();  // or Nexus::with_config(config)
let mut optimizer = Adam::new(model.parameters(), 1e-4);

// Training loop
for epoch in 0..num_epochs {
    for i in 0..dataset.len() {
        let (image, annotations) = dataset.get(i).unwrap();
        let frame = Variable::new(image.unsqueeze(0).unwrap(), true);

        // Extract boxes and class IDs
        let gt_boxes: Vec<[f32; 4]> = annotations.iter()
            .map(|a| a.bbox)
            .collect();
        let gt_classes: Vec<usize> = annotations.iter()
            .map(|a| a.category_id)
            .collect();

        let loss = nexus_training_step(
            &mut model,
            &frame,
            &gt_boxes,
            &gt_classes,
            &mut optimizer,
        );

        if i % 100 == 0 {
            println!("Epoch {} [{}/{}] Loss: {:.4}", epoch, i, dataset.len(), loss);
        }
    }
}
```

**Pipeline:** forward → FCOS target assignment (3 scales) → FocalLoss (cls) + SmoothL1Loss (bbox) → backward → optimizer step

### Phantom Training (Face Detection)

```rust
use axonml_vision::models::phantom::Phantom;
use axonml_vision::training::phantom_training_step;
use axonml_optim::Adam;

let mut model = Phantom::new();
let mut optimizer = Adam::new(model.parameters(), 1e-4);

for epoch in 0..num_epochs {
    for i in 0..dataset.len() {
        let (image, face_boxes) = dataset.get(i).unwrap();
        let frame = Variable::new(image.unsqueeze(0).unwrap(), true);

        let loss = phantom_training_step(
            &mut model,
            &frame,
            &face_boxes,
            &mut optimizer,
        );

        if i % 100 == 0 {
            println!("Epoch {} [{}/{}] Loss: {:.4}", epoch, i, dataset.len(), loss);
        }
    }
}
```

**Pipeline:** forward → single-scale target assignment → FocalLoss (cls) + SmoothL1Loss (bbox) → backward → optimizer step

---

## Evaluation Metrics

### Average Precision (AP)

Compute AP for a single class using 11-point interpolation (Pascal VOC 2007):

```rust
use axonml_vision::training::{DetectionResult, GroundTruth, compute_ap};

let detections = vec![
    DetectionResult { bbox: [10.0, 10.0, 50.0, 50.0], confidence: 0.9, class_id: 0 },
    DetectionResult { bbox: [60.0, 60.0, 100.0, 100.0], confidence: 0.7, class_id: 0 },
];

let ground_truths = vec![
    GroundTruth { bbox: [12.0, 12.0, 48.0, 48.0], class_id: 0 },
];

let ap = compute_ap(&detections, &ground_truths, 0.5);  // IoU threshold 0.5
println!("AP@0.5: {:.4}", ap);
```

### Mean Average Precision (mAP)

Compute mAP across all classes:

```rust
use axonml_vision::training::compute_map;

// all_detections[i] = detections for image i
// all_ground_truths[i] = ground truths for image i
let map = compute_map(&all_detections, &all_ground_truths, num_classes, 0.5);
println!("mAP@0.5: {:.4}", map);
```

### COCO mAP

Average mAP over IoU thresholds `[0.50, 0.55, 0.60, ..., 0.95]` (the COCO primary metric):

```rust
use axonml_vision::training::compute_coco_map;

let coco_map = compute_coco_map(&all_detections, &all_ground_truths, num_classes);
println!("COCO mAP@[0.5:0.95]: {:.4}", coco_map);
```

---

## Model Architectures

### Nexus — Dual-Pathway Object Detector

A neuroscience-inspired object detector with five key innovations:

1. **Dual-pathway processing** — Ventral ("what") and dorsal ("where") streams process features separately before cross-pathway fusion
2. **Predictive coding** — Surprise-gated processing allocates more compute to unexpected regions
3. **Persistent object memory** — GRU hidden state per tracked object maintains identity across frames
4. **Uncertainty quantification** — Every bbox prediction includes mean + log-variance
5. **Multi-scale detection** — 3 scales with FCOS-style anchor-free heads

```rust
use axonml_vision::models::nexus::{Nexus, NexusConfig};

// Default config: 320x320, 20 classes
let mut model = Nexus::new();

// Custom config
let config = NexusConfig {
    input_width: 640,
    input_height: 640,
    num_classes: 80,
    memory_hidden_size: 128,
    proposal_threshold: 0.3,
    nms_threshold: 0.5,
};
let mut model = Nexus::with_config(config);

// Inference
let detections = model.detect(&frame_variable);
for det in &detections {
    println!("Class {}: [{:.0}, {:.0}, {:.0}, {:.0}] conf={:.2}",
        det.class_id, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
        det.confidence);
}

// Training forward pass
let train_output = model.forward_train(&frame_variable);
// train_output.scales: Vec<NexusScaleOutput>
//   .cls_logits: [1, 1, H, W]
//   .bbox_pred: [1, 4, H, W]
//   .centerness: [1, 1, H, W]
```

~430K parameters, <2MB float32, <500KB INT8

### Phantom — Event-Driven Face Detector

An ultra-efficient face detector inspired by neuromorphic event cameras:

1. **Pseudo-event generation** — Frame differencing on standard cameras creates event maps
2. **Sparse processing** — Only event-active regions receive heavy compute
3. **Predictive tracking** — GRU state per face predicts next location
4. **Implicit identity** — Tracking ID from temporal continuity
5. **Confidence accumulation** — Long-tracked faces receive higher confidence

```rust
use axonml_vision::models::phantom::{Phantom, PhantomConfig};

// Default config: 128x128
let mut model = Phantom::new();

// Custom config
let config = PhantomConfig {
    input_width: 256,
    input_height: 256,
    backbone_refresh_interval: 30,
    tracker_hidden_size: 64,
    detection_threshold: 0.5,
};
let mut model = Phantom::with_config(config);

// Inference (processes temporal sequence)
let faces = model.detect_frame(&frame_variable);
for face in &faces {
    println!("Face: [{:.0}, {:.0}, {:.0}, {:.0}] conf={:.2} track_id={}",
        face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
        face.confidence, face.track_id);
}

// Training forward pass
let train_output = model.forward_train(&frame_variable);
// train_output.face_cls: [1, 1, H/4, W/4]
// train_output.face_bbox: [1, 4, H/4, W/4]
```

~126K parameters, <500KB float32, <130KB INT8

**Compute efficiency profile:**

| Frame | Compute | Reason |
|:------|:--------|:-------|
| 1 | 100% | Cold start, full backbone |
| 5 | ~30% | Sparse event processing |
| 30 | ~5% | Predictions accurate, minimal correction |
| Static | ~0% | Cached backbone, no events |

### NightVision — Multi-Domain Infrared Detector

A YOLOX-inspired detector adapted for thermal imagery across multiple domains:

1. **Thermal-adaptive stem** — handles single-channel (1-ch) or multi-band (3-ch) IR input with thermal normalization
2. **CSP backbone** — Cross-Stage Partial blocks for efficient multi-scale thermal feature extraction
3. **Thermal FPN** — Feature Pyramid Network with top-down + lateral connections (P3/P4/P5)
4. **Decoupled heads** — Separate classification, bbox regression, and objectness branches per scale
5. **Domain tagging** — Optional domain classification head for multi-domain operation

```rust
use axonml_vision::models::nightvision::{NightVision, NightVisionConfig, ThermalDomain};

// Preset configurations for each domain
let model = NightVision::new(NightVisionConfig::wildlife(20));    // 20 animal species
let model = NightVision::new(NightVisionConfig::human());         // search & rescue
let model = NightVision::new(NightVisionConfig::interstellar(3, 3)); // 3-band IR, 3 classes
let model = NightVision::new(NightVisionConfig::multi_domain(50));   // all domains, domain tags
let model = NightVision::new(NightVisionConfig::edge(10));           // compact for edge

// Detection forward pass — per-scale outputs
let outputs = model.forward_detection(&ir_image);
// outputs: Vec<(cls, bbox, obj, Option<domain>)> — one per FPN level

// Flattened forward — concatenated across scales
let (cls, bbox, obj) = model.forward_flat(&ir_image);
// cls: [B, total_anchors, num_classes]
// bbox: [B, total_anchors, 4]
// obj: [B, total_anchors, 1]
```

~200K-500K parameters (config-dependent), designed for edge/embedded thermal camera deployments.

**Thermal domains:** Wildlife (warm-blooded animals), Human (body heat / SAR), Interstellar (astronomical thermal sources), Vehicle (engine heat / friction), General (domain-agnostic).

---

## Autograd Additions

The following `Variable` operations were added to support detection training:

```rust
// Exponential and logarithm (with full gradient tracking)
let y = x.exp();       // e^x, grad: exp(x)
let y = x.log();       // ln(x), grad: 1/x

// Clamping with gradient passthrough
let y = x.clamp(0.0, 1.0);  // grad: 1.0 where min < x < max, else 0.0
```

### Loss Functions (axonml-nn)

**BCEWithLogitsLoss** — Binary cross-entropy with built-in sigmoid (numerically stable):

```rust
use axonml_nn::BCEWithLogitsLoss;

let loss_fn = BCEWithLogitsLoss::new();
let loss = loss_fn.compute(&logits, &targets);
// Formula: max(x, 0) - x*t + log(1 + exp(-|x|))
// Gradient: sigmoid(x) - target
```

**SmoothL1Loss (Huber Loss)** — Smooth transition between L1 and L2:

```rust
use axonml_nn::SmoothL1Loss;

let loss_fn = SmoothL1Loss::new();              // beta=1.0
let loss_fn = SmoothL1Loss::with_beta(0.1);     // custom beta

let loss = loss_fn.compute(&pred, &target);
// |diff| < beta: 0.5 * diff^2 / beta (L2-like, smooth at origin)
// |diff| >= beta: |diff| - 0.5 * beta (L1-like, robust to outliers)
```

---

## Complete Example: Train Phantom on WIDER FACE

```rust
use axonml_vision::models::phantom::Phantom;
use axonml_vision::datasets::WiderFaceDataset;
use axonml_vision::training::phantom_training_step;
use axonml_autograd::Variable;
use axonml_optim::Adam;

fn main() -> Result<(), String> {
    // Load dataset
    let dataset = WiderFaceDataset::new(
        "/data/wider_face", "train", (128, 128),
    )?;
    println!("Training on {} images", dataset.len());

    // Create model and optimizer
    let mut model = Phantom::new();
    let mut optimizer = Adam::new(model.parameters(), 1e-4);

    // Training loop
    for epoch in 0..50 {
        let mut epoch_loss = 0.0;
        for i in 0..dataset.len() {
            let (image, face_boxes) = dataset.get(i).unwrap();
            let frame = Variable::new(
                image.unsqueeze(0).unwrap(), true,
            );

            let loss = phantom_training_step(
                &mut model, &frame, &face_boxes, &mut optimizer,
            );
            epoch_loss += loss;
        }

        println!("Epoch {}: avg_loss = {:.4}",
            epoch, epoch_loss / dataset.len() as f32);
    }

    Ok(())
}
```

## Complete Example: Train Nexus on COCO

```rust
use axonml_vision::models::nexus::Nexus;
use axonml_vision::datasets::CocoDataset;
use axonml_vision::training::nexus_training_step;
use axonml_autograd::Variable;
use axonml_optim::Adam;

fn main() -> Result<(), String> {
    // Load dataset
    let dataset = CocoDataset::new(
        "/data/coco/train2017",
        "/data/coco/annotations/instances_train2017.json",
        (320, 320),
    )?;
    println!("Training on {} images, {} classes",
        dataset.len(), dataset.num_classes());

    // Create model and optimizer
    let mut model = Nexus::new();
    let mut optimizer = Adam::new(model.parameters(), 1e-4);

    // Training loop
    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        for i in 0..dataset.len() {
            let (image, annotations) = dataset.get(i).unwrap();
            let frame = Variable::new(
                image.unsqueeze(0).unwrap(), true,
            );

            let gt_boxes: Vec<[f32; 4]> = annotations.iter()
                .map(|a| a.bbox).collect();
            let gt_classes: Vec<usize> = annotations.iter()
                .map(|a| a.category_id).collect();

            let loss = nexus_training_step(
                &mut model, &frame, &gt_boxes, &gt_classes,
                &mut optimizer,
            );
            epoch_loss += loss;
        }

        println!("Epoch {}: avg_loss = {:.4}",
            epoch, epoch_loss / dataset.len() as f32);
    }

    Ok(())
}
```
