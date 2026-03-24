# Themis — Multimodal Belief Propagation Fusion

An uncertainty-aware fusion engine that combines up to 4 biometric modality embeddings (face, iris, voice, fingerprint) into a single fused identity with principled confidence estimation. Uses learned consistency checking and belief propagation via GRU for sequential evidence accumulation.

**~49K parameters** | Input: Up to 4 modality embeddings | Output: 48-dim fused identity + match probability

---

## Table of Contents

- [Architecture](#architecture)
- [Uncertainty Gating](#uncertainty-gating)
- [Consistency Checking](#consistency-checking)
- [Belief Propagation](#belief-propagation)
- [API Reference](#api-reference)
- [Loss Function](#loss-function)
- [Forensic Reporting](#forensic-reporting)
- [Configuration](#configuration)

---

## Architecture

```
Face embedding [64] + logvar ──┐
                                │  ┌─────────────────────┐
Iris embedding [128] + logvar ──┼──│ Modality Projectors  │──> 4 × [48] projected
                                │  │  face:  Linear(64→48) │
Voice embedding [64] + logvar ──┤  │  iris:  Linear(128→48)│
                                │  │  voice: Linear(64→48) │
Finger embedding [128] + logvar─┘  │  finger:Linear(128→48)│
                                   └─────────────────────┘
                                            |
                              ┌─────────────┼─────────────┐
                              │             │             │
                    ┌─────────┴───────┐     │    ┌────────┴────────┐
                    │ Uncertainty Gate │     │    │ Consistency     │
                    │                 │     │    │ Checker         │
                    │ For each modality:    │    │                 │
                    │ w = σ(-logvar/T)│     │    │ FC(192→64→4)   │
                    │                 │     │    │ → per-modality  │
                    │ Missing → w ≈ 0 │     │    │   consistency  │
                    │ Confident → w ≈ 1    │    │   weights       │
                    └─────────┬───────┘     │    └────────┬────────┘
                              │             │             │
                              └──────┬──────┘─────┬───────┘
                                     │            │
                              ┌──────┴────────────┴──────┐
                              │  Weighted Combination     │
                              │  fused = Σ(w_i × c_i × p_i)│
                              │  / Σ(w_i × c_i)          │
                              └───────────┬───────────────┘
                                          │
                              ┌───────────┴───────────────┐
                              │  Belief GRU               │
                              │  GRUCell(48, 48)          │
                              │  Temporal belief evolution │
                              └───────────┬───────────────┘
                                          │
                              ┌───────────┴───────┐
                              │ Decision Head      │  Linear(48→1) → σ
                              │ → match_prob [0,1] │
                              ├───────────────────┤
                              │ Identity Head      │  Linear(48→48) → L2
                              │ → fused_identity   │  [48-dim]
                              └───────────────────┘
```

### Parameter Breakdown

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Face projector | 3,120 | Linear(64→48) |
| Iris projector | 6,192 | Linear(128→48) |
| Voice projector | 3,120 | Linear(64→48) |
| Finger projector | 6,192 | Linear(128→48) |
| Consistency FC1 | 12,352 | Linear(192→64) |
| Consistency FC2 | 260 | Linear(64→4) |
| Belief GRU | 14,112 | GRUCell(48, 48) |
| Decision head | 49 | Linear(48→1) |
| Identity head | 2,352 | Linear(48→48) |
| **Total** | **~48K** | |

## Uncertainty Gating

Each modality produces a `log_variance` alongside its embedding. Themis converts this to a confidence weight:

```
weight = sigmoid(-log_variance / temperature)
```

| Scenario | log_variance | weight | Effect |
|----------|-------------|--------|--------|
| High confidence | -2.0 | ~0.88 | Strong contribution |
| Medium confidence | 0.0 | ~0.50 | Moderate contribution |
| Low confidence | 2.0 | ~0.12 | Weak contribution |
| Missing modality | +inf | ~0.00 | Zero contribution |

**Graceful degradation**: If a modality is missing (no iris scanner, microphone broken), its log_variance is set to +inf, making its weight effectively zero. The system operates on whatever modalities are available.

## Consistency Checking

After projection to the common 48-dim space, Themis checks cross-modal consistency:

1. Concatenate all 4 projected embeddings → [192]
2. FC layers → 4 per-modality consistency scores
3. Softmax → consistency weights

**Why**: If face says "person A" but voice says "person B", the consistency checker down-weights the conflicting modality. This catches spoofing attempts where one modality is faked but others aren't.

## Belief Propagation

For sequential evidence (multiple observations over time), Themis maintains a **belief state** via GRU:

```
Observation 1: belief₁ = GRU(fused_evidence₁, belief₀)
Observation 2: belief₂ = GRU(fused_evidence₂, belief₁)
...
```

This allows the system to accumulate evidence — e.g., first seeing a face, then hearing a voice, then scanning a fingerprint — with the belief state growing more confident with each corroborating modality.

**Temporal decay**: `fuse_with_decay()` applies exponential decay to stale evidence, ensuring that old observations gradually lose influence.

## API Reference

```rust
use axonml_vision::models::biometric::ThemisFusion;

let fusion = ThemisFusion::new();

// Fuse available modalities (any can be None)
let (fused_identity, match_prob, confidence, belief_state) = fusion.fuse(
    Some((&face_emb, &face_logvar)),    // Face
    Some((&iris_emb, &iris_logvar)),     // Iris
    None,                                 // Voice (not available)
    Some((&finger_emb, &finger_logvar)), // Fingerprint
    prev_belief.as_ref(),                // Previous belief state
);

// Fuse with temporal decay
let result = fusion.fuse_with_decay(
    face, iris, voice, finger,
    prev_belief, decay_rate
);

// Forensic fusion (detailed audit trail)
let report = fusion.fuse_forensic(
    face, iris, voice, finger, prev_belief
);
// report.modality_scores: per-modality match scores
// report.modality_weights: per-modality contribution weights
// report.cross_consistency: cross-modal agreement score
// report.dominant_modality: which modality contributed most
// report.weakest_modality: which modality contributed least
// report.contributing_dims: top dimensions driving the decision
// report.conflicts: modality disagreements (if any)

// Update reliability tracking
fusion.update_reliability(modality, was_correct);
```

## Loss Function

**ThemisLoss** (`losses.rs`):

```
L = λ_bce × BCE(match_prob, is_match)
  + λ_triplet × triplet(fused_anchor, fused_pos, fused_neg)
  + λ_calibration × (confidence - accuracy)²
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bce_weight` | 1.0 | Weight for match prediction BCE |
| `triplet_weight` | 0.5 | Weight for fused embedding triplet |
| `calibration_weight` | 0.1 | Confidence calibration (Brier-like) |
| `margin` | 0.3 | Triplet margin |

## Forensic Reporting

Every match decision can produce a `ForensicReport`:

```rust
pub struct ForensicReport {
    pub modality_scores: Vec<(BiometricModality, f32)>,   // Per-modality match score
    pub modality_weights: Vec<(BiometricModality, f32)>,   // Per-modality contribution
    pub modality_agreement: f32,                            // Cross-modal consistency
    pub cross_consistency: f32,                             // Overall agreement [0,1]
    pub dominant_modality: BiometricModality,               // Strongest contributor
    pub weakest_modality: BiometricModality,                // Weakest contributor
    pub contributing_dims: Vec<(usize, f32)>,               // Top embedding dimensions
    pub conflicts: Vec<(BiometricModality, BiometricModality, f32)>,  // Disagreements
}
```

This is critical for:
- **Audit trails** — explain every match/reject decision
- **Spoofing detection** — identify which modality was likely faked
- **System tuning** — understand which modalities are most reliable

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Common dim | 48 | Shared projection space |
| Belief GRU | 48 hidden | Temporal belief state |
| Temperature | 1.0 | Uncertainty gating sensitivity |
| Face input | 64-dim | From Mnemosyne |
| Iris input | 128-dim | From Argus |
| Voice input | 64-dim | From Echo |
| Finger input | 128-dim | From Ariadne |

---

*Part of the [Aegis Biometric Suite](README.md) in [AxonML](https://github.com/AutomataNexus/AxonML).*
