# biometric-training

End-to-end training pipelines for the **Aegis biometric suite** (Mnemosyne / Argus / Ariadne / Echo / Themis), each trained on a real biometric dataset with GPU acceleration, live browser monitoring, checkpointing, and resume support.

**Crates used from AxonML:** `axonml-core`, `axonml-tensor`, `axonml-autograd`, `axonml-nn`, `axonml-optim`, `axonml-vision` (Aegis models + losses), `axonml-serialize`, `axonml` (`TrainingMonitor`).

This is a **standalone crate** — `[workspace]` is empty so `cargo` doesn't pull in the full AxonML workspace. It depends on the framework via `path = "/opt/AxonML/crates/..."`.

---

## The Five Modalities

| Model | Modality | Paradigm | Binary | Dataset | Status |
|-------|----------|----------|--------|---------|--------|
| **Mnemosyne** | Face | Temporal crystallization (GRU attractor) | `train_mnemosyne` | LFW | ✓ |
| **Argus** | Iris | Radial phase encoding | `train_argus` | CASIA-Iris-Syn (polar cache) | ✓ |
| **Ariadne** | Fingerprint | Gabor ridge event fields | `train_ariadne` | FVC2000 DB4_B | ✓ |
| **Echo** | Voice | Predictive speaker residuals | (deferred) | (no voice dataset yet) | — |
| **Themis** | Fusion | Uncertainty-gated belief propagation | (deferred) | Embeddings from the above | — |

## Datasets

| Dataset | Modality | Identities | Samples | Size | Location |
|---------|----------|------------|---------|------|----------|
| LFW (Labeled Faces in the Wild) | Face | 423 | 5,985 | 294 MB | `/opt/datasets/lfw/processed/` |
| CASIA-Iris-Syn | Iris (raw) | 1,000 | 10,000 | 164 MB | `/opt/datasets/iris/processed/` |
| CASIA-Iris-Syn (polar cache) | Iris (unwrapped) | 1,000 | 10,000 | 328 MB | `/opt/datasets/iris/polar_cache/` |
| FVC2000 DB4_B | Fingerprint | 10 | 800 | 52 MB | `/opt/datasets/fingerprint/processed/` |

**Binary format:** one file per identity (`identity_NNNN.bin`), `[num:u32][channels:u32][height:u32][width:u32][f32 pixel data...]`.

**Polar cache:** Argus training uses pre-computed Cartesian→polar unwraps at `/opt/datasets/iris/polar_cache/`. This eliminates the expensive unwrap transform from the training loop and lets the model use `encode_polar()` directly instead of `forward_full()`.

## Usage

### Mnemosyne (Face — LFW)

```bash
# GPU training (recommended)
cargo run --release --bin train_mnemosyne -p biometric-training --features cuda

# Custom configuration
cargo run --release --bin train_mnemosyne -p biometric-training --features cuda -- \
    --epochs 50 --bs 64 --seq-len 8 --batches 200 --lr 5e-4

# Resume from latest checkpoint
cargo run --release --bin train_mnemosyne -p biometric-training --features cuda -- \
    --resume latest

# Start fresh
cargo run --release --bin train_mnemosyne -p biometric-training --features cuda --fresh
```

### Argus (Iris — CASIA-Iris-Syn polar cache)

```bash
cargo run --release --bin train_argus -p biometric-training --features cuda

# Custom
cargo run --release --bin train_argus -p biometric-training --features cuda -- \
    --epochs 50 --bs 64 --batches 200 --lr 5e-4
```

### Ariadne (Fingerprint — FVC2000 DB4_B)

```bash
cargo run --release --bin train_ariadne -p biometric-training --features cuda

# Custom
cargo run --release --bin train_ariadne -p biometric-training --features cuda -- \
    --epochs 50 --bs 32 --batches 200 --lr 5e-4
```

**Note:** FVC2000 DB4_B only has 10 identities × 80 samples per identity. This is small — useful for smoke-testing the architecture and pipeline but not enough for a competitive model. See the Aegis paper's limitations section.

## Features Every Trainer Has

| Feature | Description |
|---------|-------------|
| **GPU acceleration** | `--features cuda` — AxonML auto-migrates tensors to `Device::Cuda(0)` on first forward |
| **Live browser monitor** | `axonml::TrainingMonitor` — opens Chromium with real-time loss/metric charts |
| **Best model tracking** | Saves `best_model.axonml` + `checkpoint_best.axonml` whenever loss improves |
| **Latest checkpoint** | Always saves `checkpoint_latest.axonml` at end of each epoch (for resume) |
| **Periodic epoch checkpoints** | Saves `checkpoint_epoch_NNNN.axonml` every `--save-every` epochs |
| **Resume support** | `--resume latest|best|<path>` or `--fresh` to ignore checkpoints |
| **Cosine LR + warmup** | Linear warmup for `--warmup` epochs then cosine decay to 1% of peak |
| **AdamW optimizer** | Decoupled weight decay for better generalization |

## Default Hyperparameters

| Parameter | Mnemosyne | Argus | Ariadne |
|-----------|-----------|-------|---------|
| Input | 3×64×64 | 1×32×256 | 1×128×128 |
| Epochs | 30 | 30 | 30 |
| Batch size | 32 | 32 | 16 |
| Sequence length | 5 | — | — |
| Batches/epoch | 100 | 150 | 150 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | — | — |
| Warmup epochs | 3 | 3 | 3 |
| Loss | CrystallizationLoss | ArgusLoss | ContrastiveLoss |

## Loss Functions

- **CrystallizationLoss** (Mnemosyne): triplet margin on L2-normalized hidden states + convergence-velocity regularization that pushes the GRU to stabilize after repeated observations.
- **ArgusLoss** (Argus): triplet margin + phase-consistency regularization (same eye seen from different angles should produce similar codes).
- **ContrastiveLoss** (Ariadne): same-identity → minimize Euclidean distance; different-identity → push apart beyond margin, max(0, margin - d)².

All three losses ship with both `compute()` (raw f32, for inference / evaluation) and `compute_var()` (graph-tracked `Variable`, for training with backprop).

## Output

Checkpoints land in `checkpoints/<model>/`:

```
biometric-training/checkpoints/mnemosyne/
├── best_model.axonml              # weights only — for inference
├── checkpoint_best.axonml         # full checkpoint (model + optim + state)
├── checkpoint_latest.axonml       # for --resume latest
└── checkpoint_epoch_NNNN.axonml   # every --save-every epochs
```

## Why a Companion Crate?

The old training scripts inside `crates/axonml-vision/examples/` are reference-style demos baked into the framework. This companion crate keeps the training pipelines **out of the framework proper** — it depends on the locked `axonml-vision` models but lives in its own crate so we can iterate on training code (new mining strategies, ArcFace, semi-hard mining, etc.) without touching the framework. Same pattern as `llm-training`.
