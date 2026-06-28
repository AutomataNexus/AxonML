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

AxonML provides end-to-end training infrastructure for object and face
detection. The system includes image loading, dataset parsers (COCO,
WIDER FACE), detection-specific losses (Focal, GIoU, Uncertainty),
FCOS-style anchor-free target assignment, and AP/mAP evaluation metrics.

The built-in detector architectures are:

| Model | Task | Architecture | Notes |
|:------|:-----|:-------------|:------|
| **BlazeFace** | Face detection | Depthwise-separable conv, dual-scale 128×128, 896 anchors | ~72K params |
| **RetinaFace** | Face detection | ResNet34 backbone + multi-level FPN, cls/bbox/landmark heads | — |
| **DETR** | General object detection | Transformer encoder-decoder, set prediction | `small` preset |
| **NanoDet** | General object detection | ShuffleNet backbone + Ghost-PAN neck, edge-ready | ~364K params |

All are designed to be trainable with the shared detection utilities below.

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

Assigns ground-truth boxes to spatial locations on multiple feature-map
scales based on center-point containment and size ranges:

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

### Single-Scale Target Assignment

`assign_single_scale_targets` is the generic single-scale variant, useful for
single-stride face/object heads:

```rust
use axonml_vision::training::assign_single_scale_targets;

let gt_faces: Vec<[f32; 4]> = vec![[10.0, 15.0, 40.0, 50.0]];  // pixel coords
let feat_h = 32;
let feat_w = 32;
let stride = 4.0;

let (cls_target, bbox_target) = assign_single_scale_targets(
    &gt_faces, feat_h, feat_w, stride,
);
// cls_target: [H, W] — 1.0 at object-center cells, 0.0 elsewhere
// bbox_target: [H, W, 4] — [dx, dy, log_w, log_h] at positive cells
```

**Bbox target encoding:**
- `dx = (cx - cell_cx) / stride` — horizontal offset
- `dy = (cy - cell_cy) / stride` — vertical offset
- `log_w = ln(w / stride)` — log-space width
- `log_h = ln(h / stride)` — log-space height

---

## Building a Training Loop

The losses and target-assignment utilities compose into a standard
forward → assign → loss → backward → step loop. For a single-scale head:

```rust
use axonml_vision::training::assign_single_scale_targets;
use axonml_nn::{BCEWithLogitsLoss, SmoothL1Loss};
use axonml_autograd::Variable;
use axonml_optim::{Adam, Optimizer};

let mut optimizer = Adam::new(model.parameters(), 1e-4);
let cls_loss_fn = BCEWithLogitsLoss::new();
let bbox_loss_fn = SmoothL1Loss::new();

for (image, gt_boxes) in dataset.iter() {
    let frame = Variable::new(image.unsqueeze(0).unwrap(), true);
    let out = model.forward(&frame);   // your model's detection head output

    let (cls_t, bbox_t) = assign_single_scale_targets(&gt_boxes, feat_h, feat_w, stride);
    let cls_loss = cls_loss_fn.compute(&out.cls_logits, &cls_t.into());
    let bbox_loss = bbox_loss_fn.compute(&out.bbox_pred, &bbox_t.into());
    let loss = &cls_loss + &bbox_loss;

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

For multi-scale FCOS heads, use `assign_fcos_targets` + `fcos_targets_to_tensors`
and sum the per-scale losses (typically `FocalLoss` for classification +
`SmoothL1Loss`/`GIoULoss` for boxes, weighted by centerness).

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

## Detector Models

```rust
use axonml_vision::models::{BlazeFace, RetinaFace, NanoDet, DETR};

let blaze   = BlazeFace::new();                   // dual-scale 128x128 face detector
let retina  = RetinaFace::new();                  // ResNet34 backbone + FPN
let nanodet = NanoDet::new(/*num_classes=*/ 80);  // mobile-class general detector
let detr    = DETR::small(10);                     // transformer set-prediction detector
```

- **BlazeFace** — depthwise-separable conv, dual-scale 128×128 head, 896 anchors (~72K params).
- **RetinaFace** — ResNet34 backbone with a multi-level FPN head producing class / bbox / 5-point landmark predictions.
- **NanoDet** — ShuffleNet backbone + Ghost-PAN neck, anchor-free, edge-ready (~364K params).
- **DETR** — transformer encoder-decoder with set-based prediction; `small` preset.

All share the `FPN` feature-pyramid neck where applicable and integrate with the losses and evaluation utilities above.

---

## Autograd Additions

The following `Variable` operations support detection training:

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

*Last updated: 2026-06-06 (v0.6.5)*
