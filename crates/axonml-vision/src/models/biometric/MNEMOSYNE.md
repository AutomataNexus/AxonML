# Mnemosyne — Face Identity via Temporal Crystallization

A novel face recognition model that treats identity as a dynamical system attractor. Instead of encoding identity from a single snapshot, Mnemosyne evolves a GRU hidden state over multiple face observations until it **crystallizes** — converging to a stable identity representation.

**~115K parameters** | Input: [B, 3, 64, 64] | Embedding: 64-dim | Encoding: 96-dim

---

## Table of Contents

- [Architecture](#architecture)
- [Temporal Crystallization](#temporal-crystallization)
- [Training](#training)
- [API Reference](#api-reference)
- [Loss Function](#loss-function)
- [Liveness Detection](#liveness-detection)
- [Configuration](#configuration)

---

## Architecture

```
Face Image [B, 3, 64, 64]
     |
 ┌─────────────────────┐
 │ Stem                 │  Conv2d(3→16, 3×3, s=2) + BN + ReLU
 │                      │  [B, 16, 32, 32]
 └─────────────────────┘
     |
 ┌─────────────────────┐
 │ BlazeBlock 1         │  DWSep(16→24, s=2) + Residual
 │                      │  [B, 24, 16, 16]
 └─────────────────────┘
     |
 ┌─────────────────────┐
 │ BlazeBlock 2         │  DWSep(24→32, s=2) + Residual
 │                      │  [B, 32, 8, 8]
 └─────────────────────┘
     |
 ┌─────────────────────┐
 │ BlazeBlock 3         │  DWSep(32→48, s=2) + Residual
 │                      │  [B, 48, 4, 4]
 └─────────────────────┘
     |
 [AdaptiveAvgPool2d]    →  [B, 48, 1, 1]
     |
 [Flatten + Linear]     →  face_encoding [B, 96]
     |
 ┌─────────────────────┐
 │ Quality Gate         │  Linear(96→1) + Sigmoid → [0, 1]
 │                      │  Modulates how much this frame updates identity
 └─────────────────────┘
     |
 [Element-wise multiply] →  gated_encoding = encoding × quality
     |
 ┌─────────────────────┐
 │ Crystallization GRU  │  GRUCell(input=96, hidden=64)
 │                      │  hidden state IS the identity
 └─────────────────────┘
     |
 ┌─────────────────────┐
 │ Convergence Head     │  Linear(64→2)
 │                      │  → velocity (sigmoid) + log_variance
 └─────────────────────┘
     |
 [L2 Normalize hidden]  →  identity_embedding [64]
```

### BlazeBlock (Depthwise Separable + Residual)

Each BlazeBlock consists of:
- **Depthwise conv** (3×3, groups=in_ch) → BN → ReLU
- **Pointwise conv** (1×1) → BN
- **Residual path** (1×1 conv if dimensions change)
- **ReLU** after addition

This is the same efficient block design used in Google's BlazeFace, optimized for mobile face detection.

### Parameter Breakdown

| Component | Parameters | Shape |
|-----------|-----------|-------|
| Stem conv + BN | 480 | Conv2d(3→16) + BN(16) |
| BlazeBlock 1 | 1,256 | DWSep(16→24) + proj |
| BlazeBlock 2 | 2,176 | DWSep(24→32) + proj |
| BlazeBlock 3 | 3,408 | DWSep(32→48) + proj |
| Face projection | 4,704 | Linear(48→96) |
| Quality gate | 97 | Linear(96→1) |
| GRU cell | 31,104 | GRUCell(96, 64) |
| Convergence head | 130 | Linear(64→2) |
| **Total** | **~43.4K** | |

## Temporal Crystallization

The core insight: a single face image is noisy — lighting, angle, expression, occlusion all corrupt the identity signal. But across multiple observations, these variations cancel out while the identity signal reinforces.

**The GRU hidden state IS the identity.** Over repeated `crystallize_step()` calls:

1. Each face is encoded → 96-dim feature vector
2. Quality gate estimates frame quality [0, 1] — blurry/occluded → low weight
3. Gated encoding feeds into GRU, evolving the hidden state
4. Convergence velocity drops as the state stabilizes
5. After sufficient observations, the hidden state has "crystallized"

```
Observation 1: h₁ = GRU(encode(face₁) × quality₁, h₀)    velocity: 0.8
Observation 2: h₂ = GRU(encode(face₂) × quality₂, h₁)    velocity: 0.5
Observation 3: h₃ = GRU(encode(face₃) × quality₃, h₂)    velocity: 0.2
...
Observation N: hₙ = GRU(encode(faceₙ) × qualityₙ, hₙ₋₁)  velocity: 0.05 ← crystallized
```

**Identity = L2-normalize(hₙ)**

## Training

### Dataset

- **LFW (Labeled Faces in the Wild)**: 423 identities, 5,985 face images
- Preprocessed to [3, 64, 64] (grayscale replicated to 3 channels)
- Binary format: `/opt/datasets/lfw/processed/identity_NNNN.bin`

### Commands

```bash
# GPU training (recommended) — 50 epochs, batch=32, 100 batches/epoch, seq_len=5
cargo run --example train_mnemosyne --release -p axonml-vision --features cuda

# Custom configuration
cargo run --example train_mnemosyne --release -p axonml-vision --features cuda -- \
  --epochs 100 --bs 64 --batches 200 --seq-len 8 --lr 0.0005

# Benchmark after training — same/different face pair verification
cargo run --example bench_mnemosyne --release -p axonml-vision -- \
  --model /opt/AxonML/checkpoints/mnemosyne/best_model.axonml --pairs 1000
```

### Training Strategy

- **Batched triplet mining**: 32 triplets per batch, each with anchor + positive + negative face sequences
- **Crystallization**: Run each sequence through `crystallize_step()` × seq_len, use final hidden state as embedding
- **GPU acceleration**: Model params + input tensors moved to `Device::Cuda(0)` — Conv2d, BN, GRU all run on GPU
- **Loss**: CrystallizationLoss (triplet margin + convergence regularization)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **LR schedule**: Cosine annealing with 3-epoch linear warmup
- **Observed**: Loss ~0.017 at epoch 2 with GPU, converging well

### Training Monitor

Automatically launches a live browser dashboard showing:
- Training loss curve
- Convergence velocity (should decrease over training)
- Learning rate schedule

## API Reference

```rust
use axonml_vision::models::biometric::MnemosyneIdentity;

let model = MnemosyneIdentity::new();           // Default: encoding=96, hidden=64
let model = MnemosyneIdentity::with_dims(128, 96); // Custom dimensions

// Single-frame encoding
let encoding = model.encode_face(&face_var);     // [B, 3, 64, 64] → [B, 96]

// Quality assessment
let quality = model.compute_quality(&encoding);  // [B, 96] → [B, 1] in [0, 1]

// Crystallization step
let (hidden, velocity, logvar, quality) =
    model.crystallize_step(&face_var, prev_hidden.as_ref());

// Extract identity (L2-normalized Vec<f32>)
let identity = model.extract_identity(&hidden);  // Vec<f32> [hidden_dim]

// Graph-tracked normalization
let normed = model.normalize_identity(&hidden);  // Variable [B, hidden_dim]

// Match two identities
let score = MnemosyneIdentity::match_identities(
    &emb_a, &emb_b, logvar_a, logvar_b
);  // [-1, 1], 1.0 = perfect match

// Liveness detection
let liveness = model.assess_liveness(&face_sequence, &hidden_states);

// Quality assessment
let quality = model.detect_face_quality(&face_var);
```

## Loss Function

**CrystallizationLoss** (`losses.rs`):

```
L = L_triplet + λ_conv × L_convergence

L_triplet = max(0, d(anchor, positive) - d(anchor, negative) + margin)
L_convergence = mean(max(0, velocity - target)²)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `margin` | 0.3 | Triplet margin — identity separation distance |
| `convergence_weight` | 0.1 | Weight for convergence regularization |
| `target_velocity` | 0.1 | States should stabilize below this velocity |

The convergence term ensures the GRU actually crystallizes — without it, the hidden state might oscillate indefinitely.

## Liveness Detection

Mnemosyne includes built-in spoofing detection via temporal analysis:

- **Trajectory smoothness**: Real faces produce smooth hidden state trajectories; photos/screens produce jerky ones
- **Convergence pattern**: Live faces crystallize naturally; replayed video produces abnormal convergence curves
- **Quality variance**: Live subjects have natural quality variation; static images don't

```rust
let liveness = model.assess_liveness(&face_sequence, &hidden_states);
// liveness.liveness_score: 0.0 (spoof) to 1.0 (live)
// liveness.is_live: bool
// liveness.temporal_variance: f32
// liveness.trajectory_smoothness: f32
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoding_dim` | 96 | Face encoding dimension (GRU input) |
| `hidden_dim` | 64 | GRU hidden state = identity embedding dimension |
| Input size | [3, 64, 64] | RGB face image |
| BlazeBlock stages | 3 | 16→24→32→48 channels |
| Pool output | [48, 1, 1] | Adaptive average pooling |

---

*Part of the [Aegis Biometric Suite](README.md) in [AxonML](https://github.com/AutomataNexus/AxonML).*
