# Aegis Biometric Suite

A unified multimodal biometric identity system built entirely in Rust using the **AxonML** deep learning framework. Four specialized modality models + a belief propagation fusion engine, totaling ~362K parameters for the full system.

**Novel paradigm:** Each model uses a unique biometric encoding approach — temporal crystallization (face), radial phase encoding (iris), predictive speaker residuals (voice), and ridge event fields (fingerprint) — fused via uncertainty-aware belief propagation.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Models](#models)
- [Fusion Engine](#fusion-engine)
- [Unified API](#unified-api)
- [Training](#training)
- [Datasets](#datasets)
- [Loss Functions](#loss-functions)
- [Deployment Configurations](#deployment-configurations)
- [File Structure](#file-structure)

---

## Architecture Overview

```
                    ┌────────────────┐
    Face Image ───> │  Mnemosyne     │ ──> face_embedding [64]
    [3, 64, 64]     │  ~115K params  │     + log_variance
                    └────────────────┘
                                          ┌──────────────────┐
    Iris Image ───> │  Argus         │ ──>│                  │
    [1, 64, 64]     │  ~65K params   │    │                  │
                    └────────────────┘    │  Themis Fusion   │ ──> fused_identity [48]
                                          │  ~49K params     │     + match_probability
    Voice Mel  ───> │  Echo          │ ──>│  Uncertainty-    │     + confidence
    [40, T]         │  ~68K params   │    │  aware belief    │     + forensic_report
                    └────────────────┘    │  propagation     │
                                          │                  │
    Fingerprint ──> │  Ariadne       │ ──>│                  │
    [1, 128, 128]   │  ~65K params   │    └──────────────────┘
                    └────────────────┘
```

**Key design principles:**

- **Uncertainty quantification** — every modality produces `log_variance` alongside embeddings, enabling principled fusion
- **Graceful degradation** — missing modalities contribute zero weight via uncertainty gating (sigmoid(-logvar/T))
- **Forensic auditability** — full per-modality breakdown for every match decision
- **Temporal crystallization** — identity representations improve with repeated observations
- **Lightweight** — full system fits in ~362K params, suitable for edge deployment

## Models

| Model | File | Params | Input | Embedding | Paradigm |
|-------|------|--------|-------|-----------|----------|
| [**Mnemosyne**](MNEMOSYNE.md) | `mnemosyne.rs` | ~115K | Face [3, 64, 64] | 64-dim | Temporal crystallization via GRU |
| [**Argus**](ARGUS.md) | `argus.rs` | ~65K | Iris [1, 64, 64] | 128-dim | Radial phase encoding in polar coords |
| [**Echo**](ECHO.md) | `echo.rs` | ~68K | Voice [40, T] mel | 64-dim | Predictive speaker residuals |
| [**Ariadne**](ARIADNE.md) | `ariadne.rs` | ~65K | Fingerprint [1, 128, 128] | 128-dim | Gabor ridge event fields |
| [**Themis**](THEMIS.md) | `themis.rs` | ~49K | Multi-modal embeddings | 48-dim fused | Uncertainty-aware belief propagation |

## Fusion Engine

Themis fuses up to 4 modality embeddings with:

1. **Modality projectors** — project each embedding to a common 48-dim space
2. **Uncertainty gating** — `weight = sigmoid(-logvar / temperature)`, missing modalities → 0
3. **Consistency checking** — cross-modal agreement scoring via learned FC layers
4. **Belief GRU** — temporal belief state evolution for sequential evidence
5. **Decision head** — match probability + fused identity embedding

## Unified API

`AegisIdentity` provides the end-to-end biometric system:

```rust
use axonml_vision::models::biometric::AegisIdentity;

let system = AegisIdentity::full();  // All 4 modalities

// Enrollment
let result = system.enroll("subject_001", &evidence);

// Verification (1:1)
let result = system.verify("subject_001", &probe);

// Identification (1:N)
let candidates = system.identify(&probe);

// Liveness detection
let liveness = system.assess_liveness(&evidence);

// Quality assessment
let quality = system.assess_quality(&evidence);

// Drift detection
let drift = system.detect_drift("subject_001", &new_evidence);
```

## Training

Training examples for each modality:

```bash
# Mnemosyne — Face identity on LFW (423 identities, 5,985 images)
cargo run --example train_mnemosyne --release -p axonml-vision --features cuda

# Argus — Iris identity on CASIA-Iris-Syn (1,000 identities, 10,000 polar strips)
cargo run --example train_argus --release -p axonml-vision --features cuda

# Ariadne — Fingerprint identity on FVC2000 DB4_B (10 identities, 800 images)
cargo run --example train_ariadne --release -p axonml-vision --features cuda

# Benchmark — Verify trained Mnemosyne on same/different face pairs
cargo run --example bench_mnemosyne --release -p axonml-vision
```

All training examples feature:
- **GPU acceleration** — model params + input tensors moved to CUDA device
- Live browser training dashboard (AxonML Training Monitor)
- Checkpoint saving (periodic + best model)
- Cosine annealing LR with linear warmup
- AdamW optimizer with weight decay
- Batched forward passes (32 triplets/pairs per batch)

**Important:** Always use `--features cuda` for training. AxonML requires explicit tensor device placement — both model parameters and input tensors must be moved to GPU.

## Datasets

| Dataset | Modality | Identities | Images | Size | Location |
|---------|----------|------------|--------|------|----------|
| LFW (Labeled Faces in the Wild) | Face | 423 | 5,985 | 294 MB | `/opt/datasets/lfw/processed/` |
| CASIA-Iris-Syn | Iris | 1,000 | 10,000 | 164 MB | `/opt/datasets/iris/processed/` |
| CASIA-Iris-Syn (polar cache) | Iris | 1,000 | 10,000 | 328 MB | `/opt/datasets/iris/polar_cache/` |
| FVC2000 DB4_B | Fingerprint | 10 | 800 | 52 MB | `/opt/datasets/fingerprint/processed/` |

Preprocessed format: Binary files per identity — `[num:u32][channels:u32][height:u32][width:u32][f32 pixel data...]`

**Polar cache:** Argus training uses pre-computed polar unwraps (`/opt/datasets/iris/polar_cache/`) instead of raw iris images. This eliminates the expensive Cartesian→polar transform from the training loop. Pre-compute with `preprocess.py` in the iris dataset directory.

## Loss Functions

All losses in `losses.rs` with both raw f32 (`compute()`) and graph-tracked (`compute_var()`) variants:

| Loss | Model | Description |
|------|-------|-------------|
| **CrystallizationLoss** | Mnemosyne | Triplet + convergence regularization (penalizes unstable hidden states) |
| **ArgusLoss** | Argus | Triplet + phase consistency (rotated iris → circular-shifted code) |
| **ContrastiveLoss** | Ariadne | Margin-based contrastive (same-identity minimize, different push apart) |
| **EchoLoss** | Echo | Prediction MSE + speaker triplet (identity = what you can't predict) |
| **ThemisLoss** | Themis | BCE match + triplet on fused embeddings + calibration |
| **AngularMarginLoss** | General | ArcFace-style angular margin for closed-set classification |
| **CenterLoss** | General | Uncertainty-weighted distance to class centers |
| **LivenessLoss** | General | BCE liveness + trajectory smoothness regularization |
| **DiversityRegularization** | General | Penalizes collapsed/similar embeddings across classes |

## Deployment Configurations

| Config | Modalities | Params | Use Case |
|--------|-----------|--------|----------|
| `AegisIdentity::full()` | Face + Iris + Voice + Fingerprint | ~362K | Maximum security |
| `AegisIdentity::edge_minimal()` | Face + Voice | ~183K | Edge devices, IoT |
| `AegisIdentity::face_only()` | Face only | ~115K | Smallest deployment |
| `BiometricConfig::high_security()` | Any | — | Low FAR threshold (0.3) |
| `BiometricConfig::convenience()` | Any | — | Higher threshold (0.6) |

## File Structure

```
biometric/
├── README.md          # This file — suite overview
├── MNEMOSYNE.md       # Mnemosyne face identity documentation
├── ARGUS.md           # Argus iris identity documentation
├── ECHO.md            # Echo voice identity documentation
├── ARIADNE.md         # Ariadne fingerprint identity documentation
├── THEMIS.md          # Themis fusion engine documentation
├── mod.rs             # Core types: BiometricModality, BiometricEvidence, result types
├── mnemosyne.rs       # Face identity model (~115K params)
├── argus.rs           # Iris identity model (~65K params)
├── echo.rs            # Voice identity model (~68K params)
├── ariadne.rs         # Fingerprint identity model (~65K params)
├── themis.rs          # Fusion engine (~49K params)
├── identity.rs        # AegisIdentity unified API + IdentityBank
├── losses.rs          # All biometric loss functions
└── polar.rs           # Iris polar unwrap utilities
```

---

*Part of [AxonML](https://github.com/AutomataNexus/AxonML) — a Rust deep learning framework by AutomataNexus LLC.*
