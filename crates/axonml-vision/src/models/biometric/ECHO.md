# Echo — Voice Identity via Predictive Speaker Residuals

A novel speaker verification model based on the insight that **identity is what you can't predict**. Echo learns to predict generic speech patterns, then extracts speaker identity from the prediction residuals — the signal that remains after removing what's common to all speakers.

**~68K parameters** | Input: Mel spectrogram [B, 40, T] | Embedding: 64-dim

---

## Table of Contents

- [Architecture](#architecture)
- [Predictive Residual Paradigm](#predictive-residual-paradigm)
- [Training](#training)
- [API Reference](#api-reference)
- [Loss Function](#loss-function)
- [Replay/Spoofing Detection](#replayspoofing-detection)
- [Configuration](#configuration)

---

## Architecture

```
Mel Spectrogram [B, 40, T]
     |
 ═══════════════════════════════════════════════════
 ║ PREDICTIVE PATH (learns generic speech patterns)
 ╠═══════════════════════════════════════════════════
 │ pred_conv_in: Conv1d(40→64, k=3) + ReLU
 │    ↓
 │ pred_gru: GRUCell(64, 64)
 │    Frame-by-frame autoregressive prediction
 │    h_t = GRU(frame_t, h_{t-1})
 │    ↓
 │ pred_conv_out: Conv1d(64→40, k=1)
 │    → predicted_mel [B, 40, T]
 ║
 ═══════════════════════════════════════════════════
     |
 [Compute Residuals]
   residuals = actual_mel - predicted_mel
   (This is the SPEAKER SIGNAL — what the predictor couldn't model)
     |
 ═══════════════════════════════════════════════════
 ║ RESIDUAL ENCODING (extracts speaker identity)
 ╠═══════════════════════════════════════════════════
 │ res_conv1: Conv1d(40→32, k=5, pad=2) + ReLU
 │ res_conv2: Conv1d(32→32, k=3, pad=1) + ReLU
 │    → [B, 32, T]
 │    ↓
 │ Statistics Pooling:
 │    mean across time → [B, 32]
 │    std across time  → [B, 32]
 │    concat → [B, 64]
 ║
 ═══════════════════════════════════════════════════
     |
 ┌─────────────────────────────┐
 │ Speaker Head                 │  Linear(64→64) → L2 normalize
 │                              │  → speaker_embedding [B, 64]
 ├─────────────────────────────┤
 │ Uncertainty Head             │  Linear(64→1)
 │                              │  → log_variance [B, 1]
 └─────────────────────────────┘
```

### Parameter Breakdown

| Component | Parameters | Description |
|-----------|-----------|-------------|
| pred_conv_in | 7,744 | Conv1d(40→64, k=3) + bias |
| pred_gru | 24,960 | GRUCell(64, 64) — autoregressive predictor |
| pred_conv_out | 2,600 | Conv1d(64→40, k=1) — reconstruct mel |
| res_conv1 | 6,432 | Conv1d(40→32, k=5) — residual encoder |
| res_conv2 | 3,104 | Conv1d(32→32, k=3) |
| speaker_head | 4,160 | Linear(64→64) |
| uncertainty_head | 65 | Linear(64→1) |
| **Total** | **~49K** | |

## Predictive Residual Paradigm

The key insight: all human speech shares common patterns (phonemes, prosody, formant transitions). A good predictor can model these generic patterns. What's left — the **residual** — is the speaker-specific signal.

```
Actual mel:    [shared speech patterns] + [speaker-specific coloring]
Predicted mel: [shared speech patterns]   (learned by GRU predictor)
─────────────────────────────────────────────────────────────
Residual:      [speaker-specific coloring] ← THIS IS THE IDENTITY
```

**Why this works:**
- The GRU learns to predict the next mel frame from previous frames
- It can only model patterns common across all speakers in training
- What it **can't predict** (the residual) is unique to each speaker
- This is analogous to how novelty detection works in neuroscience

**Advantages:**
- Naturally separates content from identity
- Works with any speech content (language-independent)
- The prediction error itself serves as a novelty detector for unknown speakers

## Training

### Dataset

Echo trains on speaker recognition datasets with multiple utterances per speaker. The model needs mel spectrograms (40 bins) from audio files.

### Training Strategy

- **Combined loss**: Prediction MSE (forces predictor to learn speech) + Speaker triplet (forces residuals to be discriminative)
- **Triplets**: Anchor + positive from same speaker, negative from different speaker
- **The predictor must be good enough** — if it can't predict speech at all, residuals are just noise

### Commands

```bash
# Train with speaker dataset
cargo run --example train_echo --release -p axonml-vision

# GPU accelerated
cargo run --example train_echo --release -p axonml-vision --features cuda
```

## API Reference

```rust
use axonml_vision::models::biometric::EchoSpeaker;

let model = EchoSpeaker::new();                // Default: mel_bins=40, embed_dim=64

// Full forward
let (predicted_mel, embedding, logvar) = model.forward_full(&mel_var);
// [B, 40, T] → ([B, 40, T], [B, 64], [B, 1])

// Predict + compute residuals
let (predicted, residuals) = model.predict_and_residual(&mel_var);

// Encode residuals → speaker embedding
let (embedding, logvar) = model.encode_residuals(&residuals);

// Extract identity
let identity = model.extract_identity(&mel_var);  // Vec<f32> [64]

// Prediction error (novelty score — high = unknown speaker)
let error = model.prediction_error(&mel_var);

// Replay/spoofing detection
let is_replay = model.detect_replay(&mel_var);

// Voice activity detection
let vad = model.voice_activity(&mel_var);

// Temporal consistency (speaker stability across frames)
let consistency = model.temporal_consistency(&mel_var);

// Speaking rate estimation
let rate = model.speaking_rate(&mel_var);
```

## Loss Function

**EchoLoss** (`losses.rs`):

```
L = λ_pred × L_prediction + λ_speaker × L_speaker

L_prediction = MSE(predicted_mel, actual_mel)
L_speaker = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prediction_weight` | 1.0 | Weight for mel prediction MSE |
| `speaker_weight` | 0.5 | Weight for speaker triplet loss |
| `margin` | 0.3 | Triplet margin |

**Critical balance:** The prediction loss must be high enough that the GRU actually learns speech patterns, but the speaker loss ensures the residuals are discriminative.

## Replay/Spoofing Detection

Echo includes built-in anti-spoofing via **spectral flatness variance analysis**:

- **Live speech**: Natural variation in spectral flatness across frames (breathing, emphasis, pauses)
- **Replay attack**: Recorded/played-back audio has unnaturally uniform spectral flatness (compressed, equalized)

```rust
let is_replay = model.detect_replay(&mel_var);
// true = likely replay attack, false = likely live speech
```

Additional liveness signals:
- **Prediction error patterns**: Live speech has natural prediction error variance; replayed audio is more predictable
- **Temporal consistency**: Real speakers maintain consistent identity embedding across an utterance

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mel_bins` | 40 | Number of mel frequency bins (input channels) |
| `hidden_dim` | 64 | GRU predictor hidden dimension |
| `embed_dim` | 64 | Speaker embedding dimension |
| `pred_kernel` | 3 | Prediction conv kernel size |
| `res_kernels` | 5, 3 | Residual encoding conv kernels |

---

*Part of the [Aegis Biometric Suite](README.md) in [AxonML](https://github.com/AutomataNexus/AxonML).*
