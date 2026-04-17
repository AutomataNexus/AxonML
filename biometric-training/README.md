# biometric-training

End-to-end training binaries for the **Aegis biometric suite** — three modality trainers (face / iris / fingerprint) built on `axonml-vision`, with GPU acceleration, a live browser training monitor, AdamW + cosine LR schedule, and checkpoint-resume support.

**Version:** 0.6.1 — updated 2026-04-16.

**Crates used from AxonML:** `axonml-core`, `axonml-tensor`, `axonml-autograd`, `axonml-nn`, `axonml-optim`, `axonml-vision` (Aegis models + losses), `axonml-serialize`, `axonml` (`TrainingMonitor`).

This is a **standalone crate** — the `Cargo.toml` declares its own empty `[workspace]` so `cargo` doesn't pull in the full AxonML workspace. Framework crates are referenced by `path = "/opt/AxonML/crates/..."`.

---

## The Modalities

Three training binaries are wired today. Echo (voice) and Themis (fusion) are listed in the Aegis suite but have no training binaries in this crate yet.

| Model | Modality | Paradigm | Binary | Dataset | Status |
|-------|----------|----------|--------|---------|--------|
| **Mnemosyne** | Face | Temporal crystallization (GRU hidden-state stabilization + convergence-velocity) | `train_mnemosyne` | LFW (64×64 RGB) | Works |
| **Argus** | Iris | Radial phase encoding, polar-unwrapped input | `train_argus` | CASIA-Iris-Syn polar cache | Works |
| **Ariadne** | Fingerprint | Gabor ridge event fields, contrastive pairs | `train_ariadne` | FVC2000 DB4_B | Works |
| Echo | Voice | Predictive speaker residuals | (deferred) | (no voice dataset yet) | Not in this crate |
| Themis | Fusion | Uncertainty-gated belief propagation | (deferred) | Embeddings from the above | Not in this crate |

## Datasets

| Dataset | Modality | Identities | Samples | Size | Location |
|---------|----------|------------|---------|------|----------|
| LFW (Labeled Faces in the Wild) | Face | ~5,985 | ~13k images | ~294 MB | `/opt/datasets/lfw/processed/` |
| CASIA-Iris-Syn | Iris (raw) | 1,000 | 10,000 | ~164 MB | `/opt/datasets/iris/processed/` |
| CASIA-Iris-Syn (polar cache) | Iris (unwrapped) | 1,000 | 10,000 | ~328 MB | `/opt/datasets/iris/polar_cache/` |
| FVC2000 DB4_B | Fingerprint | 10 | 800 | ~52 MB | `/opt/datasets/fingerprint/processed/` |

**Binary format:** one file per identity (`identity_NNNN.bin`), 16-byte little-endian header `[num:u32][channels:u32][height:u32][width:u32]` followed by `num * channels * height * width` f32 values. `IdentityDataset::load` sorts the filenames to produce stable label IDs and validates shape consistency across identities.

**Polar cache:** Argus training uses pre-computed Cartesian→polar unwraps at `/opt/datasets/iris/polar_cache/` (shape `[1, 32, 256]` — 32 radial bins × 256 angular bins). This eliminates the expensive unwrap transform from the training loop and lets the model use `encode_polar()` directly (≈5× faster than `forward_full`). The trainer asserts the cache shape at startup.

## Usage

### Mnemosyne (Face — LFW)

```bash
# GPU training (recommended)
cargo run --release --bin train_mnemosyne -p biometric-training --features cuda

# Custom configuration
cargo run --release --bin train_mnemosyne -p biometric-training --features cuda -- \
    --epochs 30 --bs 32 --seq-len 5 --batches 100 --lr 1e-3 --wd 1e-4

# Resume from latest checkpoint
cargo run --release --bin train_mnemosyne -p biometric-training --features cuda -- \
    --resume latest
```

### Argus (Iris — CASIA-Iris-Syn polar cache)

```bash
cargo run --release --bin train_argus -p biometric-training --features cuda

# Custom
cargo run --release --bin train_argus -p biometric-training --features cuda -- \
    --epochs 30 --bs 32 --batches 150 --lr 1e-3
```

### Ariadne (Fingerprint — FVC2000 DB4_B)

```bash
cargo run --release --bin train_ariadne -p biometric-training --features cuda

# Custom
cargo run --release --bin train_ariadne -p biometric-training --features cuda -- \
    --epochs 30 --bs 16 --batches 150 --lr 1e-3
```

**Note:** FVC2000 DB4_B is only 10 identities × 80 samples — useful for smoke-testing the architecture and pipeline but not enough for a competitive model. See the Aegis paper's limitations section.

## Features Every Trainer Has

| Feature | Description |
|---------|-------------|
| **GPU acceleration** | `--features cuda` — AxonML auto-migrates tensors to `Device::Cuda(0)` on first forward |
| **Live browser monitor** | `axonml::TrainingMonitor` serves real-time loss/metric charts over HTTP |
| **Best model tracking** | Saves `best_model.axonml` + `checkpoint_best.axonml` whenever the tracked loss improves |
| **Latest checkpoint** | Writes `checkpoint_latest.axonml` at the end of each epoch (used by `--resume latest`) |
| **Periodic epoch checkpoints** | Writes `checkpoint_epoch_NNNN.axonml` every `--save-every` epochs |
| **Resume support** | `--resume latest\|best\|<path>` with name-based then shape-based in-order fallback; `--fresh` to ignore checkpoints |
| **Cosine LR + warmup** | Linear warmup for `--warmup` epochs then cosine decay to 1% of the peak LR |
| **AdamW optimizer** | Decoupled weight decay (Mnemosyne uses `--wd`; Argus and Ariadne use default AdamW) |

Note: unlike `llm-training`, this crate does **not** currently wire the shared `TrainingLifecycle` Unix-socket controller. Pause/resume/stop over a socket is not available here today — use `--resume latest` after a clean exit instead.

## Default Hyperparameters

| Parameter | Mnemosyne | Argus | Ariadne |
|-----------|-----------|-------|---------|
| Input shape | 3×64×64 | 1×32×256 (polar) | 1×128×128 |
| Epochs | 30 | 30 | 30 |
| Batch size | 32 | 32 | 16 |
| Sequence length | 5 | — | — |
| Batches / epoch | 100 | 150 | 150 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | (default) | (default) |
| Warmup epochs | 3 | 3 | 3 |
| Log every (batches) | 10 | 15 | 15 |
| Save every (epochs) | 5 | 5 | 5 |
| RNG seed | 42 | 7 | (per-binary) |
| Loss | `CrystallizationLoss` | `ArgusLoss` | `ContrastiveLoss` |

## Loss Functions

All three losses (in `axonml-vision::models::biometric`) expose both `compute()` (raw f32, for inference / evaluation) and `compute_var()` (graph-tracked `Variable`, for training with backprop).

- **CrystallizationLoss** (Mnemosyne) — triplet margin in cosine space on L2-normalized final hidden states plus a convergence-velocity regularization that pushes the GRU to stabilize after repeated face observations of the same identity. The trainer runs a `seq_len`-step crystallization via `crystallize_step`, accumulates per-step velocity Variables, and feeds the mean anchor velocity into the loss.
- **ArgusLoss** (Argus) — triplet margin plus phase-consistency regularization so rotations of the same iris produce similar radial-phase codes. Anchor/positive pairs are L2-normalized and the loss operates on the `encode_polar` output.
- **ContrastiveLoss** (Ariadne) — 50/50 same-identity / different-identity pair mining via `mine_pair_batch`; same-identity pairs minimize Euclidean distance, different-identity pairs push apart beyond margin via `max(0, margin - d)^2`.

## Shared Utilities (`src/lib.rs`)

- `IdentityRecord` + `IdentityDataset` — uniform per-identity binary loader for all three modalities, with `sample_len`, `num_identities`, `total_samples`, and `count_with_at_least(k)` accessors.
- `mine_triplet_batch` — random (anchor, positive, negative) triplet sampler (Argus).
- `mine_identity_sequence_batches` — per-step triplet sequences threaded across `seq_len` time steps (Mnemosyne).
- `mine_pair_batch` — 50/50 same/different pair sampler (Ariadne).
- `l2_normalize_var` — graph-tracked L2 normalization (`mul_var` / `sum` / `sqrt` / `div_var`) that puts embeddings on the unit hypersphere.
- `lcg_range` — Numerical Recipes LCG for deterministic batch mining (no external RNG crate pulled into training).
- `format_count` — thousands-separator formatter.
- `ResumeMode` + `find_checkpoint` + `load_model_from_checkpoint` — checkpoint-resume helpers. Intentionally duplicated from `llm-training` so `biometric-training` stays a self-contained standalone crate.

## Output Layout

Checkpoints land in `checkpoints/<model>/` under `/opt/AxonML/biometric-training/checkpoints/` — the `argus/`, `ariadne/`, and `mnemosyne/` subdirs exist today.

```
biometric-training/checkpoints/mnemosyne/
├── best_model.axonml              # weights only — for inference
├── checkpoint_best.axonml         # full checkpoint (model + optim + state)
├── checkpoint_latest.axonml       # for --resume latest
└── checkpoint_epoch_NNNN.axonml   # every --save-every epochs
```

## Why a Companion Crate?

The old training scripts inside `crates/axonml-vision/examples/` are reference-style demos baked into the framework. This companion crate keeps the real training pipelines **out of the framework proper** — it depends on the locked `axonml-vision` models but lives in its own crate so we can iterate on training code (new mining strategies, ArcFace, semi-hard mining, etc.) without touching the framework. Same pattern as `llm-training`.
