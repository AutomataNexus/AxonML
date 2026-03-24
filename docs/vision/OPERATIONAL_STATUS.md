# AxonML Vision Crate — Operational Status

> Last updated: 2026-03-06

## Overview

- **Tests**: 637 passing, 0 failing
- **Source files**: 59 `.rs` files across models, datasets, ops, training infra
- **Test coverage**: 54/57 files have tests (94.7%)
- **TODOs remaining**: 1 minor (Aegis3D affected_nodes tracking)

---

## Model Inventory

All models compile, have `forward()` methods, support train/eval mode, and have unit tests.

### Classic CNNs

| Model | File | Params | Tests | Notes |
|-------|------|--------|-------|-------|
| LeNet | `models/lenet.rs` | ~60K | 6 | Also includes SimpleCNN, MLP |
| ResNet18/34 | `models/resnet.rs` | 11M/21M | 4 | BasicBlock; Bottleneck (50/101/152) stubbed |
| VGG11–19 | `models/vgg.rs` | 133M+ | 6 | With/without batch norm |

### Vision Transformer

| Model | File | Params | Tests | Notes |
|-------|------|--------|-------|-------|
| ViT | `models/transformer.rs` | Configurable | 8 | PatchExtractBackward for full gradient flow |
| Transformer Encoder/Decoder | `models/transformer.rs` | — | (included above) | Full encoder-decoder stack |

### Detection

| Model | File | Params | Tests | Notes |
|-------|------|--------|-------|-------|
| DETR | `models/detr.rs` | — | 5+ | Transformer-based detection |
| NanoDet | `models/nanodet.rs` | <1M | 5 | ShuffleNet backbone, Ghost PAN, edge-ready |
| BlazeFace | `models/blazeface.rs` | ~100K | 4 | Depthwise separable, mobile face detection |
| RetinaFace | `models/retinaface.rs` | — | 4 | ResNet+FPN, multi-task (cls/bbox/landmarks) |
| FPN | `models/fpn.rs` | — | 4 | Feature Pyramid Network |

### Advanced Detection (Novel Architectures)

| Model | Directory | Params | Tests | Notes |
|-------|-----------|--------|-------|-------|
| Nexus | `models/nexus/` | ~430K | 30+ | Dual ventral/dorsal pathways, predictive coding, GRU object memory, uncertainty quantification |
| Phantom | `models/phantom/` | ~126K | 24+ | Pseudo-event generation, sparse processing, predictive GRU tracking, Pi-deployable |

**Nexus components**: backbone, detector, fusion, heads, memory, predictive coding

**Phantom components**: backbone, detector, events, tracker

### Biometric Framework (Novel)

| Model | File | Params | Tests | Modality |
|-------|------|--------|-------|----------|
| Mnemosyne | `biometric/mnemosyne.rs` | ~115K | 44 | Face — temporal crystallization via GRU |
| Ariadne | `biometric/ariadne.rs` | ~65K | 37 | Fingerprint — ridge event fields |
| Echo | `biometric/echo.rs` | ~68K | 32 | Voice — predictive speaker residuals |
| Argus | `biometric/argus.rs` | ~65K | 47 | Iris — radial phase encoding |
| Themis | `biometric/themis.rs` | ~49K | 41 | Multimodal fusion — belief propagation |
| Identity Bank | `biometric/identity.rs` | — | 55 | Enrollment/verification/identification |
| Losses | `biometric/losses.rs` | — | 39 | 8+ specialized loss functions |
| Polar | `biometric/polar.rs` | — | 26 | Polar coordinate iris processing |

**Total biometric**: ~362K params, <2MB f32, <400KB INT8, Pi-deployable

### Other Models

| Model | File | Params | Tests | Notes |
|-------|------|--------|-------|-------|
| DPT / FastDepth | `models/depth.rs` | — | 5 | Monocular depth estimation |
| PatchCore / StudentTeacher | `models/anomaly.rs` | — | 4 | Anomaly detection |
| VQA | `models/vqa.rs` | — | 4 | ViT encoder + text encoder + cross-attention |

### 3D Reconstruction (Novel)

| Component | File | Tests | Notes |
|-----------|------|-------|-------|
| Aegis3D | `models/aegis3d/mod.rs` | 10 | Octree-adaptive neural implicit surfaces |
| Implicit SDF | `models/aegis3d/implicit.rs` | — | Per-node SDF networks |
| Mesh | `models/aegis3d/mesh.rs` | — | Pure Rust marching cubes, OBJ/STL export |
| Octree | `models/aegis3d/octree.rs` | 8 | Adaptive spatial index |
| Renderer | `models/aegis3d/renderer.rs` | — | Sphere tracing |

---

## Datasets

| Dataset | File | Status | Auto-download | Tests |
|---------|------|--------|---------------|-------|
| MNIST | `datasets/mnist.rs` | Complete | No (IDX file loader) | Yes + SyntheticMNIST |
| CIFAR-10/100 | `datasets/cifar.rs` | Complete | No (binary loader) | Yes + SyntheticCIFAR |
| COCO | `datasets/coco.rs` | Complete | No (JSON annotations) | Yes |
| WIDER FACE | `datasets/wider_face.rs` | Complete | No (txt annotations) | Yes |

---

## Infrastructure

### Vision Operations (`ops.rs`) — 16+ tests
- `box_iou()`, `box_cxcywh_to_xyxy()`, `box_xyxy_to_cxcywh()`
- `nms()` — Non-maximum suppression
- `positional_encoding_2d()` — For DETR
- `interpolate()`, `interpolate_var()` — Bilinear + nearest with `InterpolateBackward`
- `roi_align()` — Region of interest pooling
- `Detection`, `FaceDetection`, `DepthMap` result types

### Losses (`losses.rs`) — 39+ tests
- `FocalLoss` — Dense detection (α=0.25, γ=2.0)
- `GIoULoss` — Generalized IoU for bbox regression
- `UncertaintyNLLLoss` — Probabilistic predictions
- 8+ biometric-specific losses (angular margin, center, contrastive, crystallization, etc.)

### Transforms (`transforms.rs`) — 11+ tests
- `Resize` — Bilinear interpolation (2D/3D/4D)
- `ImageNormalize` — Mean/std (ImageNet, MNIST presets)
- `RandomHorizontalFlip`, `RandomCrop`, `RandomRotation`
- `Compose` — Transform chaining

### Image I/O (`image_io.rs`) — 7+ tests
- `load_image()` — JPEG/PNG → `[3, H, W]` f32 tensor [0, 1]
- `load_image_resized()`, `load_image_with_info()`
- `rgb_bytes_to_tensor()`, `tensor_to_rgb_bytes()`
- `save_image()` — Tensor → PNG/JPEG

### Camera Pipeline (`camera/`) — 16+ tests
- `Camera` — Intrinsics/extrinsics
- `FileCamera` — Frame sequence from image files
- `V4L2Camera` — Linux V4L2 capture
- `CameraPipeline` — Real-time preprocessing
- `Preprocess` — Denormalization, format conversion, cropping

### Training Infrastructure (`training/`) — 15+ tests
- `assign_fcos_targets()` — FCOS-style anchor-free target assignment
- `assign_phantom_targets()` — Phantom face detector targets
- `compute_ap()` / `compute_map()` — Average precision, mAP (COCO-style)
- `nexus_training_step()`, `phantom_training_step()`
- `TrainConfig` — Epochs, batch size, learning rate, input size

### Edge Profiling (`edge.rs`) — 4+ tests
- `profile_model()` — Parameter count, FLOPS estimate, deployment target
- `ModelProfile` — Size (f32/INT8), target recommendation
- `DeployTarget` — Edge (<5M params, <20MB) / Server / Both

### Model Hub (`hub.rs`)
- Pretrained model loading from remote sources
- Model registry and metadata
- Download caching

---

## Operational Roadmap

### Phase 1: End-to-End Training Convergence — COMPLETE

Prove models actually learn by training to convergence on synthetic data.

File: `training/convergence.rs` (8 tests)

| Test | Model | Dataset | Target | Status |
|------|-------|---------|--------|--------|
| `convergence_lenet_mnist` | LeNet | SyntheticMNIST | Loss decreasing, Acc > 30% | PASS |
| `convergence_mlp_mnist` | MLP | SyntheticMNIST | Loss decreasing | PASS |
| `convergence_lenet_cifar` | LeNet | SyntheticCIFAR | Loss decreasing | PASS |
| `convergence_resnet18_cifar_smoke` | ResNet18 | SyntheticCIFAR | Loss decreasing (8 steps) | PASS |
| `convergence_vit_cifar_smoke` | VisionTransformer | SyntheticCIFAR | Loss decreasing (5 steps) | PASS |
| `convergence_nanodet_forward_smoke` | NanoDet | Synthetic | Finite output at 64/128px | PASS |
| `convergence_nanodet_training_step` | Phantom | Synthetic | Finite loss (3 steps) | PASS |
| `convergence_lenet_sgd` | LeNet | SyntheticMNIST | SGD+Momentum loss decreasing | PASS |

### Phase 2: Integration Tests — COMPLETE

Full pipeline validation: dataset -> transform -> model -> loss -> backward -> optimizer step.

File: `training/integration.rs` (8 tests)

| Test | Pipeline | Status |
|------|----------|--------|
| `integration_mnist_lenet_adam` | MNIST -> Normalize -> LeNet -> CrossEntropy -> Adam | PASS |
| `integration_cifar_resnet_sgd` | CIFAR -> ResNet18 -> CrossEntropy -> SGD | PASS |
| `integration_cifar_simplecnn_adam` | CIFAR -> Normalize -> SimpleCNN -> CrossEntropy -> Adam | PASS |
| `integration_cifar_vit_adam` | CIFAR -> ViT -> CrossEntropy -> Adam | PASS |
| `integration_mnist_mlp_mse` | MNIST -> MLP -> MSELoss -> Adam | PASS |
| `integration_detection_phantom` | Synthetic -> Phantom -> FocalLoss+SmoothL1 -> Adam | PASS |
| `integration_biometric_mnemosyne` | Synthetic face -> Mnemosyne -> MSELoss -> Adam | PASS |
| `integration_gradient_flow_lenet` | Full gradient flow validation (all params get grads) | PASS |

### Phase 3: Benchmarks — COMPLETE

Throughput and latency measurements (CPU, single-thread).

File: `training/benchmarks.rs` (13 tests)

| Model | Batch | Latency | Throughput | Params | Size (f32) |
|-------|-------|---------|------------|--------|------------|
| LeNet | 1 | ~5ms | ~200 img/s | 44K | 0.17 MB |
| LeNet | 32 | ~35ms | ~900 img/s | | |
| LeNet | 128 | ~130ms | ~1000 img/s | | |
| SimpleCNN | 1 | ~33ms | ~30 img/s | 563K | 2.15 MB |
| SimpleCNN | 32 | ~100ms | ~300 img/s | | |
| MLP | 1 | ~9ms | ~110 img/s | 235K | 0.90 MB |
| MLP | 128 | ~16ms | ~8000 img/s | | |
| ResNet18 | 1 | ~80ms | ~12 img/s | 11.2M | 42.67 MB |
| ResNet18 | 8 | ~500ms | ~16 img/s | | |
| VGG16 | 1 | ~40s | <1 img/s | 133M+ | 508+ MB |
| ViT-Small | 1 | ~8ms | ~130 img/s | 147K | 0.56 MB |
| ViT-Small | 8 | ~40ms | ~200 img/s | | |
| NanoDet-64 | 1 | ~210ms | ~5 img/s | 364K | 1.39 MB |
| BlazeFace | 1 | ~230ms | ~4 img/s | 72K | 0.27 MB |
| Nexus-64 | 1 | ~410ms | ~2 img/s | 430K+ | ~1.7 MB |
| Phantom-64 | 1 | ~80ms | ~12 img/s | 126K+ | ~0.5 MB |
| Mnemosyne | 1 | ~5ms | ~180 img/s | 43K | 0.17 MB |
| Mnemosyne | 8 | ~19ms | ~420 img/s | | |
| LeNet train step | 16 | ~37ms | ~430 img/s | (fwd+bwd+step) | |

---

## Known Limitations

1. **Datasets require pre-downloaded files** — no auto-download (by design for edge/offline use)
2. **ResNet50/101/152** — Bottleneck block defined but not wired up as full models
3. **Aegis3D** — `affected_nodes` tracking not yet populated for multi-view incremental updates
4. **FPN/RetinaFace** — `panic!()` on empty input rather than graceful error
5. **No pretrained weights** — hub infrastructure exists but no hosted weights yet
