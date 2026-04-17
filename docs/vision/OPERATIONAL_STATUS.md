---
layout: default
title: Vision Operational Status
parent: Vision
nav_order: 1
description: "Operational status of all models in the axonml-vision crate"
---

# AxonML Vision Crate — Operational Status

> Last updated: 2026-04-16 (v0.6.1)

## Overview

- **Tests**: 741 passing (from the workspace test matrix in the root README; `cargo test -p axonml-vision`)
- **Source files**: 79 `.rs` files across models, datasets, ops, losses, transforms, training infra, camera, hub, edge profiling
- **Workspace total**: 2,182 tests across 24 crates

Every model listed below compiles, has a working `forward()`, supports train/eval mode through the `Module` trait, and has unit tests. None are stubs.

> Note on prior versions of this document: the claim that some models (e.g. Bottleneck-based ResNets, RetinaFace, FPN) were "stubbed" was inaccurate. All detection and backbone models in this table have full implementations — the only gaps are the *pretrained weights hub* (infrastructure exists, but no hosted weights yet) and ResNet-50/101/152 *assembled models* (`BasicBlock` / `Bottleneck` blocks are both implemented; only resnet18 / resnet34 are wired up as ready-to-use variants).

---

## Model Inventory

### Classic CNNs

| Model | File | Source LoC | Notes |
|-------|------|-----------|-------|
| LeNet / MLP / SimpleCNN | `models/lenet.rs` | — | Three classic baselines in one file |
| ResNet (BasicBlock + Bottleneck) | `models/resnet.rs` | — | `resnet18`, `resnet34` ready-to-use. `Bottleneck` block is fully implemented; `resnet50/101/152` factory not yet wired. |
| VGG11/13/16/19 | `models/vgg.rs` | — | With/without batch norm |

### Vision Transformer

| Model | File | Notes |
|-------|------|-------|
| ViT (`vit_base`, `vit_large`) | `models/transformer.rs` | Custom `PatchExtractBackward` for full gradient flow through patch extraction |
| Transformer Encoder / Decoder | `models/transformer.rs` | Full encoder-decoder stack (also used by DETR) |

### Detection

| Model | File / Dir | Source LoC | Notes |
|-------|------------|-----------|-------|
| DETR | `models/detr.rs` | 493 | Transformer-based detection — smoke-tested in `models/mod.rs` |
| NanoDet | `models/nanodet.rs` | 804 | ShuffleNet backbone, Ghost-PAN neck, edge-ready (~364K params) |
| BlazeFace | `models/blazeface.rs` | 754 | Depthwise separable conv, mobile face detection (~72K params) |
| RetinaFace | `models/retinaface.rs` | 504 | ResNet+FPN, multi-task (cls/bbox/landmarks) |
| FPN | `models/fpn.rs` | 224 | Feature Pyramid Network (used by RetinaFace + as standalone) |
| Helios | `models/helios/` | — | Separate detector with `HeliosLoss`, `CIoULoss`, `TaskAlignedAssigner` |

### Advanced Detection (Novel)

| Model | Dir | Source LoC | Notes |
|-------|-----|-----------|-------|
| Nexus | `models/nexus/` | 2,363 | Dual ventral/dorsal pathways, predictive coding, GRU object memory, uncertainty quantification, 3-scale FCOS heads (~430K params) |
| Phantom | `models/phantom/` | 1,768 | Pseudo-event generation (frame differencing), sparse processing, predictive GRU face tracking, Pi-deployable (~126K params) |
| NightVision | `models/nightvision/` | 1,483 | Multi-domain IR: Wildlife / Human / Interstellar / Vehicle / General; thermal-adaptive stem + CSP backbone + Thermal FPN + YOLOX decoupled heads |

**Nexus components**: `backbone.rs` (484), `detector.rs` (476), `fusion.rs` (264), `heads.rs` (284), `memory.rs` (385), `predictive.rs` (373)

**Phantom components**: `backbone.rs` (446), `detector.rs` (461), `events.rs` (347), `tracker.rs` (430)

**NightVision components**: `backbone.rs` (454), `detector.rs` (488), `head.rs` (242), `neck.rs` (221)

### Biometric Framework (Novel — Aegis Identity)

| Model | File | Source LoC | Modality |
|-------|------|-----------|----------|
| Mnemosyne | `biometric/mnemosyne.rs` | 1,657 | Face — temporal crystallization via GRU attractor convergence + liveness |
| Ariadne | `biometric/ariadne.rs` | 1,402 | Fingerprint — ridge event fields (Gabor wavelet banks, singularity detection) |
| Echo | `biometric/echo.rs` | 1,320 | Voice — predictive speaker residuals (identity = what can't be predicted) |
| Argus | `biometric/argus.rs` | 1,376 | Iris — polar-native radial phase encoding, rotation-invariant matching |
| Themis | `biometric/themis.rs` | 1,500 | Multimodal fusion — belief propagation (uncertainty-aware) |
| Identity Bank | `biometric/identity.rs` | 1,829 | Enrollment / verification / identification / forensics |
| Polar | `biometric/polar.rs` | 782 | Polar-coordinate iris processing |
| Losses | `biometric/losses.rs` | 1,707 | 8+ specialized losses: `ArgusLoss`, `EchoLoss`, `ContrastiveLoss`, `CenterLoss`, `AngularMarginLoss`, `CrystallizationLoss`, `ThemisLoss`, `LivenessLoss` |

**Total biometric**: ~362K params across five modalities, <2MB f32, <400KB INT8, Pi-deployable. Every modality can run standalone or through Themis fusion.

**GPU training pipelines**: All biometric models have full GPU training with checkpoint/resume support. Example binaries: `train_mnemosyne` (LFW face pairs), `train_argus` (CASIA-Iris), `train_ariadne` (FVC2000 fingerprint); benchmark binary `bench_mnemosyne` (verification pairs → ROC-AUC, EER, FAR/FRR).

### Anomaly Detection

| Model | File | Notes |
|-------|------|-------|
| PatchCore | `models/anomaly.rs` | Memory-bank anomaly detection (`default_rgb()` builds the 256-dim RGB variant) |
| StudentTeacher | `models/anomaly.rs` | Feature-matching student-teacher anomaly detection |

### Depth Estimation

| Model | File | Notes |
|-------|------|-------|
| DPT (`small`) | `models/depth.rs` | Dense prediction transformer |
| FastDepth | `models/depth.rs` | Lightweight monocular depth |

### Other Models

| Model | File | Notes |
|-------|------|-------|
| VQA | `models/vqa.rs` | ViT image encoder + text encoder + cross-attention |
| Aegis3D | `models/aegis3d/` | Octree-adaptive neural implicit surfaces (SDF + marching cubes + sphere tracing) |

### 3D Reconstruction (Novel)

| Component | File | Notes |
|-----------|------|-------|
| Aegis3D | `models/aegis3d/mod.rs` | Main API |
| Implicit SDF | `models/aegis3d/implicit.rs` | Per-node SDF networks |
| Mesh | `models/aegis3d/mesh.rs` | Pure-Rust marching cubes, OBJ/STL export |
| Octree | `models/aegis3d/octree.rs` | Adaptive spatial index |
| Renderer | `models/aegis3d/renderer.rs` | Sphere tracing |

---

## Datasets

| Dataset | File | Notes |
|---------|------|-------|
| MNIST / SyntheticMNIST | `datasets/mnist.rs` | IDX file loader; synthetic variant for smoke tests |
| CIFAR-10/100 / SyntheticCIFAR | `datasets/cifar.rs` | Binary loader; synthetic variant for smoke tests |
| COCO | `datasets/coco.rs` | Full COCO JSON parser, non-contiguous category-ID remap, bbox `[x,y,w,h] → [x1,y1,x2,y2]` normalization to `[0,1]`, crowd filter |
| WIDER FACE | `datasets/wider_face.rs` | Full WIDER FACE annotation format parser |

All datasets require pre-downloaded files — there is no auto-download (intentional, for edge / offline use).

---

## Infrastructure

### Vision Operations (`ops.rs`)

- `box_iou`, `box_cxcywh_to_xyxy`, `box_xyxy_to_cxcywh`
- `nms` — Non-maximum suppression
- `positional_encoding_2d` — used by DETR
- `interpolate`, `interpolate_var` — bilinear + nearest with `InterpolateBackward` for gradient flow
- `roi_align` — region-of-interest pooling
- Result types: `Detection`, `FaceDetection`, `DepthMap`

### Losses (`losses.rs`)

- `FocalLoss` (α=0.25, γ=2.0 defaults, customizable via `with_params`)
- `GIoULoss::compute(&pred, &target)` — Generalized IoU
- `UncertaintyLoss::compute(&pred_mean, &pred_log_var, &target)` — aleatoric-uncertainty regression
- `compute_centerness(l, t, r, b) -> f32` — FCOS centerness

Plus the 8+ biometric-specific losses in `models/biometric/losses.rs`.

### Transforms (`transforms.rs`)

- `Resize` — bilinear (works for 2D/3D/4D tensors)
- `ImageNormalize` — mean/std, ImageNet and MNIST presets
- `RandomHorizontalFlip`, `RandomCrop`, `RandomRotation`, `CenterCrop`
- `Compose` — chain transforms

### Image I/O (`image_io.rs`)

- `load_image(path) -> Tensor<f32>` — `[3, H, W]`, `[0, 1]`
- `load_image_resized(path, h, w)` — with resize
- `load_image_with_info(path)` — returns original `(h, w)` along with the tensor
- `rgb_bytes_to_tensor(&[u8], h, w)` — raw frame → tensor
- `tensor_to_rgb_bytes`, `save_image` — reverse pipeline

Formats: JPEG, PNG, BMP, and everything else the `image` crate supports.

### Camera Pipeline (`camera/`)

- `Camera` — intrinsics / extrinsics
- `FileCamera` — frame sequence from image files
- `V4L2Camera` — Linux V4L2 live capture
- `CameraPipeline` — real-time preprocessing
- `Preprocess` — denormalization, format conversion, cropping

### Training Infrastructure (`training/`)

- `assign_fcos_targets` — FCOS-style multi-scale anchor-free target assignment
- `assign_phantom_targets` — single-scale target assignment for Phantom
- `nexus_training_step`, `phantom_training_step` — full forward→loss→backward→step
- `compute_ap`, `compute_map`, `compute_coco_map` — AP, mAP@0.5, and COCO mAP@[0.5:0.95]
- `TrainConfig` — epochs, batch size, learning rate, input size

### Edge Profiling (`edge.rs`)

- `profile_model` — param count, FLOPS estimate, deployment target
- `ModelProfile` — size (f32 / INT8), target recommendation
- `DeployTarget` — Edge (<5M params, <20MB) / Server / Both

### Model Hub (`hub.rs`)

- Pretrained model loading from remote sources
- Model registry and metadata
- Download caching under `~/.cache/axonml/hub/`
- CLI integration: `axonml hub list/info/download/cached/clear`

---

## Known Limitations

1. **Datasets require pre-downloaded files** — no auto-download (by design for edge / offline use).
2. **ResNet-50/101/152** — `Bottleneck` block is fully implemented, but the 50/101/152 model factories are not wired up as ready-to-use variants. Only `resnet18` and `resnet34` ship.
3. **Aegis3D** — `affected_nodes` tracking not yet populated for multi-view incremental updates (1 minor TODO in source).
4. **No hosted pretrained weights** — `hub` infrastructure exists, CLI commands work, but no self-hosted weights are served yet.
5. **FPN / RetinaFace edge cases** — `panic!()` on empty input rather than graceful error; tracked for future cleanup.

---

*Last updated: 2026-04-16 (v0.6.1)*
