# Ariadne — Fingerprint Identity via Ridge Event Fields

A fingerprint recognition model that processes fingerprint images through learned Gabor filters to extract ridge orientation fields, then encodes these fields through depthwise separable convolutions with spatial hashing for compact identity embeddings.

**~65K parameters** | Input: [B, 1, 128, 128] | Embedding: 128-dim

---

## Table of Contents

- [Architecture](#architecture)
- [Gabor Ridge Detection](#gabor-ridge-detection)
- [Training](#training)
- [API Reference](#api-reference)
- [Loss Function](#loss-function)
- [Singularity Detection](#singularity-detection)
- [Configuration](#configuration)

---

## Architecture

```
Fingerprint Image [B, 1, 128, 128]
     |
 ┌─────────────────────────────┐
 │ Gabor Ridge Detection        │
 │  8 orientation-selective     │
 │  Gabor filters (learnable)  │
 │  → 2-channel event field:   │
 │    [orientation, magnitude]  │
 │  Output: [B, 2, 128, 128]  │
 └─────────────────────────────┘
     |
 ┌─────────────────────────────┐
 │ DWSepBlock 1                 │  DWSep(2→16, s=2) + Residual
 │                              │  [B, 16, 64, 64]
 └─────────────────────────────┘
     |
 ┌─────────────────────────────┐
 │ DWSepBlock 2                 │  DWSep(16→32, s=2) + Residual
 │                              │  [B, 32, 32, 32]
 └─────────────────────────────┘
     |
 ┌─────────────────────────────┐
 │ DWSepBlock 3                 │  DWSep(32→64, s=2) + Residual
 │                              │  [B, 64, 32, 32]
 └─────────────────────────────┘
     |
 ┌─────────────────────────────┐
 │ Spatial Hashing              │
 │  Conv2d(64→16, 1×1) + BN   │
 │  AdaptiveAvgPool2d(4, 4)   │
 │  → [B, 16, 4, 4] = 256     │
 └─────────────────────────────┘
     |
 ┌─────────────────────────────┐
 │ Projection                   │  Linear(256→64) + ReLU
 │                              │  Linear(64→128) → L2 normalize
 │                              │  → fingerprint_embedding [B, 128]
 ├─────────────────────────────┤
 │ Uncertainty                  │  Linear(256→1)
 │                              │  → log_variance [B, 1]
 └─────────────────────────────┘
```

### Parameter Breakdown

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Gabor filters (8 orientations) | 200 | Learnable, initialized from Gabor theory |
| DWSepBlock 1 (2→16) | 640 | Depthwise separable + residual |
| DWSepBlock 2 (16→32) | 2,432 | Progressive downsampling |
| DWSepBlock 3 (32→64) | 8,512 | Deepest feature extraction |
| Spatial conv (1×1) | 1,040 | Channel reduction 64→16 |
| Projection (2-layer) | 24,704 | 256→64→128 |
| Uncertainty | 257 | 256→1 |
| **Total** | **~38K** | |

## Gabor Ridge Detection

Fingerprints are characterized by ridge patterns at specific orientations. Ariadne uses **8 learned Gabor filters** (one per 22.5deg orientation) to detect ridges:

1. **Initialization**: Each filter is initialized from the theoretical Gabor kernel `G(x,y) = exp(-0.5 * (x'²/σ² + y'²/σ²)) * cos(2π * x'/λ)` at its orientation
2. **Learning**: Filters are fine-tuned during training to adapt to the specific fingerprint domain
3. **Output**: 2-channel event field per pixel — ridge **orientation** (argmax across 8 filters) and **magnitude** (max response)

```
Input [1, 128, 128]
  → apply 8 Gabor filters → [8, 128, 128] responses
  → argmax over orientations → orientation_map [1, 128, 128]
  → max response → magnitude_map [1, 128, 128]
  → stack → ridge_event_field [2, 128, 128]
```

## Training

### Dataset

- **FVC2000 DB4_B**: 10 identities, 80 images each (800 total), 160×160 grayscale
- Preprocessed to [1, 128, 128]
- Binary format: `/opt/datasets/fingerprint/processed/identity_NNNN.bin`

### Commands

```bash
# GPU training (recommended) — 50 epochs, batch=32, 100 batches/epoch
cargo run --example train_ariadne --release -p axonml-vision --features cuda

# Custom
cargo run --example train_ariadne --release -p axonml-vision --features cuda -- \
  --epochs 100 --bs 64 --pairs 5000 --lr 0.0005
```

### Training Strategy

- **Batched pair mining**: 32 pairs per batch, alternating same/different identity
- **GPU acceleration**: Model params + input tensors on `Device::Cuda(0)`
- **Contrastive loss**: Same → minimize distance; different → push apart beyond margin
- **Optimizer**: AdamW (lr=1e-3)
- **LR schedule**: Cosine annealing with 3-epoch warmup
- **Note**: FVC2000 DB4_B only has 10 identities — consider supplementing with the multimodal dataset (45 identities) at `/opt/datasets/multimodal_biometric/`

## API Reference

```rust
use axonml_vision::models::biometric::AriadneFingerprint;

let model = AriadneFingerprint::new();          // Default: embed_dim=128

// Full forward (Gabor extraction + encoding)
let (embedding, logvar) = model.forward_full(&fingerprint_var);
// [B, 1, 128, 128] → ([B, 128], [B, 1])

// Extract ridge event field
let events = model.extract_ridge_events(&fingerprint_var);
// [B, 1, 128, 128] → [B, 2, 128, 128] (orientation + magnitude)

// Extract identity
let identity = model.extract_identity(&fingerprint_var);  // Vec<f32> [128]

// Singularity detection (cores + deltas)
let singularities = model.detect_singularities(&fingerprint_var);

// Minutiae extraction (bifurcations + endings)
let minutiae = model.extract_minutiae(&fingerprint_var);

// Quality assessment
let quality = model.assess_quality(&fingerprint_var);

// Rotation estimation
let angle = model.estimate_rotation(&fingerprint_var);
```

## Loss Function

**ContrastiveLoss** (`losses.rs`):

```
Same identity:      L = ||emb_a - emb_b||²
Different identity: L = max(0, margin - ||emb_a - emb_b||)²
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `margin` | 1.0 | Minimum distance between different identities |
| `orientation_weight` | 0.05 | Regularization for ridge orientation consistency |

For L2-normalized embeddings: `||a - b||² = 2 - 2·dot(a,b)`

## Singularity Detection

Ariadne can detect fingerprint singularities (topological landmarks):

- **Core points**: Where ridges converge (ridge flow center)
- **Delta points**: Three-way ridge junctions (triangular pattern)

Detection uses Poincare index analysis on the ridge orientation field — no additional neural network required.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 128 | Output fingerprint code dimension |
| Input size | [1, 128, 128] | Grayscale fingerprint image |
| Gabor filters | 8 orientations | 0deg, 22.5deg, ..., 157.5deg |
| DWSepBlock stages | 3 | 2→16→32→64 channels |
| Spatial pool | [16, 4, 4] = 256 | Fixed spatial features |

---

*Part of the [Aegis Biometric Suite](README.md) in [AxonML](https://github.com/AutomataNexus/AxonML).*
