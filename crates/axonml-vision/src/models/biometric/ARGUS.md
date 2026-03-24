# Argus — Iris Identity via Radial Phase Encoding

A novel iris recognition model that processes iris images in polar coordinates, encoding radial and angular patterns through 1D convolutions and detecting phase transitions at ridge boundaries. Rotation-invariant matching via circular shift alignment.

**~65K parameters** | Input: [B, 1, 64, 64] raw or [B, 1, 32, 256] polar | Embedding: 128-dim

---

## Table of Contents

- [Architecture](#architecture)
- [Polar Unwrap](#polar-unwrap)
- [Phase Detection](#phase-detection)
- [Training](#training)
- [API Reference](#api-reference)
- [Loss Function](#loss-function)
- [Rotation-Invariant Matching](#rotation-invariant-matching)
- [Configuration](#configuration)

---

## Architecture

```
Raw Iris Image [B, 1, 64, 64]
     |
 [Polar Unwrap]              →  Polar strip [B, 1, 32, 256]
  Cartesian → polar coords       (32 radial bins × 256 angular bins)
     |
 ═══════════════════════════════════════════════════
 ║ RADIAL ENCODING (per angular column as 1D signal)
 ╠═══════════════════════════════════════════════════
 │ Reshape: [B, 1, R, A] → [B*A, 1, R]
 │ Conv1d(1→16, k=5) + ReLU
 │ Conv1d(16→32, k=3) + ReLU
 │ Output: [B*A, 32, R']
 ║
 ═══════════════════════════════════════════════════
 ║ ANGULAR ENCODING (per radial row, with circular padding)
 ╠═══════════════════════════════════════════════════
 │ Transpose: [B*A, 32, R'] → [B*R', 32, A]
 │ Circular pad(3) + Conv1d(32→48, k=7) + ReLU
 │ Circular pad(2) + Conv1d(48→48, k=5) + ReLU
 │ Output: [B*R', 48, A']
 ║
 ═══════════════════════════════════════════════════
 ║ PHASE DETECTION (angular gradient → threshold)
 ╠═══════════════════════════════════════════════════
 │ Central finite differences on angular axis
 │ |gradient| → magnitude of phase transitions
 │ Conv1d(48→32, k=3) + ReLU (phase feature extraction)
 │ Output: [B*R', 32, A']
 ║
 ═══════════════════════════════════════════════════
 ║ SPATIAL REDUCTION
 ╠═══════════════════════════════════════════════════
 │ Reshape → [B, 32, R', A']
 │ Conv2d(32→8, 1×1) + ReLU    (channel reduction)
 │ AdaptiveAvgPool2d(4, 8)     → [B, 8, 4, 8] = 256 features
 │ Flatten → [B, 256]
 ║
 ═══════════════════════════════════════════════════
     |
 ┌─────────────────────────┐
 │ Projection               │  Linear(256→128) → L2 normalize
 │                          │  → iris_embedding [B, 128]
 ├─────────────────────────┤
 │ Uncertainty              │  Linear(256→1)
 │                          │  → log_variance [B, 1]
 └─────────────────────────┘
```

### Parameter Breakdown

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Radial Conv1d ×2 | 1,600 | 1D convs processing radial signals |
| Angular Conv1d ×2 | 22,080 | 1D convs with circular padding for 360deg continuity |
| Phase Conv1d | 4,640 | Phase transition detection |
| Reduce Conv2d (1×1) | 264 | Channel reduction 32→8 |
| AdaptiveAvgPool2d | 0 | Spatial pooling to fixed size |
| Projection (Linear) | 32,896 | 256→128 embedding |
| Uncertainty (Linear) | 257 | 256→1 log_variance |
| **Total** | **~62K** | |

## Polar Unwrap

Iris images are fundamentally circular — the iris is a ring around the pupil. Argus transforms the Cartesian iris image to polar coordinates, making rotation → horizontal translation (much easier to handle).

```
Cartesian (x, y)     →     Polar (r, θ)
  ┌─────────┐               ┌──────────────────────┐
  │  ●───●  │               │ radial  ↓             │
  │ ● ○  ● │    unwrap     │ bins    ↓             │
  │  ●───●  │  ─────────>  │         ↓             │
  │         │               │ angular → → → → → →  │
  └─────────┘               └──────────────────────┘
  64×64 grayscale            32 radial × 256 angular
```

**Circular padding** ensures angular continuity at the 0deg/360deg boundary — Conv1d kernels seamlessly wrap around.

Configuration (`PolarUnwrapConfig`):
- `default()`: 32 radial bins, 256 angular bins (standard)
- `high_res()`: 64 radial, 512 angular (more detail)
- `low_res()`: 16 radial, 128 angular (faster, edge deployment)

## Phase Detection

Ridge patterns in the iris encode identity through their **phase transitions** — where ridges start, end, and change direction.

1. **Angular gradient**: Central finite differences along the angular axis
2. **Magnitude thresholding**: `|gradient|` emphasizes transition points
3. **Phase Conv1d**: Learned filters extract meaningful phase patterns

This is analogous to how Daugman's IrisCode uses Gabor phase, but learned end-to-end rather than hand-crafted.

## Training

### Dataset

- **CASIA-Iris-Syn**: 1,000 identities, 10 images each (10,000 total)
- Preprocessed to [1, 64, 64] grayscale, center-cropped
- Binary format: `/opt/datasets/iris/processed/identity_NNNN.bin`

### Commands

```bash
# GPU training (recommended) — 50 epochs, batch=32, 100 batches/epoch
cargo run --example train_argus --release -p axonml-vision --features cuda

# Custom
cargo run --example train_argus --release -p axonml-vision --features cuda -- \
  --epochs 100 --bs 64 --batches 200 --lr 0.0005
```

### Pre-computed Polar Cache

Argus training uses **pre-computed polar unwraps** instead of raw iris images. The polar transform (Cartesian → polar coordinates) is expensive and doesn't change between epochs, so it's cached once:

- Raw iris images: `/opt/datasets/iris/processed/` (64×64 grayscale)
- Polar cache: `/opt/datasets/iris/polar_cache/` (32×256 polar strips)
- Training loads polar strips directly via `encode_polar()` (skips `forward_full()`)
- ~80% GPU utilization during training

### Training Strategy

- **Batched triplet mining**: 32 triplets per batch with anchor + positive + negative
- **Pre-computed polar strips**: Eliminates expensive Cartesian→polar transform from training loop
- **GPU acceleration**: Model params + input tensors on `Device::Cuda(0)`
- **Phase consistency**: Same-identity codes from different images should be similar
- **Loss**: ArgusLoss (triplet + phase consistency regularization)
- **Optimizer**: AdamW (lr=1e-3)
- **LR schedule**: Cosine annealing with 3-epoch warmup
- **Observed**: Loss ~0.007 at epoch 2 with GPU, converging rapidly

## API Reference

```rust
use axonml_vision::models::biometric::ArgusIris;

let model = ArgusIris::new();                    // Default: embed_dim=128

// From raw iris image (includes polar unwrap)
let (embedding, logvar) = model.forward_full(&iris_var);  // [B, 1, 64, 64] → ([B, 128], [B, 1])

// From pre-unwrapped polar strip
let (embedding, logvar) = model.encode_polar(&polar_var); // [B, 1, 32, 256] → ([B, 128], [B, 1])

// Extract identity
let identity = model.extract_identity(&iris_var);  // Vec<f32> [128]

// Rotation-invariant matching
let score = ArgusIris::match_iris(&code_a, &code_b, 16);  // 16 circular shifts

// Multi-resolution encoding
let (coarse, medium, fine) = model.encode_multi_resolution(&iris_var);

// Quality assessment
let quality = model.assess_quality(&iris_var);

// Fragile bit masking
let mask = ArgusIris::fragile_bits(&code, threshold);

// Hamming distance matching
let dist = ArgusIris::match_hamming(&code_a, &code_b);
```

## Loss Function

**ArgusLoss** (`losses.rs`):

```
L = L_triplet + λ_phase × L_phase

L_triplet = max(0, d(anchor, positive) - d(anchor, negative) + margin)
L_phase = 1 - cos_similarity(code_original, code_rotated)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `margin` | 0.3 | Triplet separation margin |
| `phase_weight` | 0.1 | Weight for phase consistency regularization |

The phase consistency term enforces that the same iris from different viewpoints (rotations) produces similar codes.

## Rotation-Invariant Matching

Eye rotation maps to horizontal translation in polar coordinates. `match_iris()` tries multiple circular shifts and returns the best cosine similarity:

```rust
let best_score = ArgusIris::match_iris(&code_a, &code_b, n_shifts);
```

With `n_shifts=16`, this tests 16 evenly-spaced rotational alignments, finding the best match regardless of head tilt.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 128 | Output iris code dimension |
| `polar_config` | PolarUnwrapConfig::default() | 32 radial × 256 angular |
| Radial convs | k=5, k=3 | 1D kernels for radial signal processing |
| Angular convs | k=7, k=5 | 1D kernels with circular padding |
| Phase conv | k=3 | Phase transition detection |
| Pool output | [8, 4, 8] = 256 | Fixed spatial features |

---

*Part of the [Aegis Biometric Suite](README.md) in [AxonML](https://github.com/AutomataNexus/AxonML).*
