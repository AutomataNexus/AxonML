# axonml-audio

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-audio"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-audio** provides audio signal-processing and dataset utilities for the AxonML framework: waveform transforms (spectrograms, MFCC, resampling, time stretch, pitch shift, augmentation), synthetic datasets, and an `AudioSeq2SeqDataset` for noise-reduction-style source/target pairs. FFT-based transforms use `rustfft` (O(n log n)).

## Features

- **Resampling** — `Resample` with linear interpolation between arbitrary sample rates.
- **Mel spectrogram** — `MelSpectrogram::new(sample_rate)` (defaults: n_fft=2048, hop=512, n_mels=128) or `with_params`; rustfft-backed.
- **MFCC** — `MFCC::new(sample_rate, n_mfcc)` for cepstral features.
- **Time stretching** — `TimeStretch::new(rate)` changes duration; preserves shape when `rate = 1.0`.
- **Pitch shifting** — `PitchShift::new(semitones)`.
- **Noise augmentation** — `AddNoise::new(snr_db)` with a configurable signal-to-noise ratio.
- **Audio normalization** — `NormalizeAudio::new()` peak-normalizes to max-|amplitude| = 1.
- **Silence trimming** — `TrimSilence::new(threshold_db)` or `TrimSilence::default_threshold()`.
- **Classification datasets** — generic `AudioClassificationDataset` plus synthetic `SyntheticCommandDataset` (small/medium/large), `SyntheticMusicDataset` (small/medium), `SyntheticSpeakerDataset` (small/medium). Labels are class-index tensors of shape `[1]` (CrossEntropyLoss-compatible).
- **Sequence-to-sequence datasets** — `AudioSeq2SeqDataset` for source/target waveform pairs, with `noise_reduction_task` constructor.

## Modules

| Module | Description |
|--------|-------------|
| `transforms` | `Resample`, `MelSpectrogram`, `MFCC`, `TimeStretch`, `PitchShift`, `AddNoise`, `NormalizeAudio`, `TrimSilence` (all implement `axonml_data::Transform`) |
| `datasets`   | `AudioClassificationDataset`, `SyntheticCommandDataset`, `SyntheticMusicDataset`, `SyntheticSpeakerDataset`, `AudioSeq2SeqDataset` |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-audio = "0.6.1"
```

### Loading Audio Datasets

```rust
use axonml_audio::prelude::*;

// Synthetic command dataset (e.g., "yes"/"no"/"stop")
let dataset = SyntheticCommandDataset::small();   // 100 samples,  10 classes, 16 kHz, 0.5 s
let dataset = SyntheticCommandDataset::medium();  // 1000 samples, 10 classes
let dataset = SyntheticCommandDataset::large();   // 10000 samples, 35 classes

// Music genre / speaker presets
let music   = SyntheticMusicDataset::small();     // multiple genres
let speakers = SyntheticSpeakerDataset::small();

let (waveform, label) = dataset.get(0).unwrap();
// waveform: [n_samples] float; label: [1] class index
```

### Mel Spectrogram

```rust
use axonml_audio::{MelSpectrogram};
use axonml_data::Transform;

let mel = MelSpectrogram::new(16000);                       // defaults: n_fft=2048, hop=512, n_mels=128
let mel = MelSpectrogram::with_params(16000, 512, 256, 40); // custom

let spectrogram = mel.apply(&waveform);
assert_eq!(spectrogram.shape()[0], 40); // n_mels
```

### MFCC Feature Extraction

```rust
use axonml_audio::MFCC;
use axonml_data::Transform;

let mfcc = MFCC::new(16000, 13);
let coefficients = mfcc.apply(&waveform);
assert_eq!(coefficients.shape()[0], 13);
```

### Audio Resampling

```rust
use axonml_audio::Resample;
use axonml_data::Transform;

let resample = Resample::new(22050, 16000);
let resampled = resample.apply(&waveform);
// Output length = floor(input_len * new_freq / orig_freq)
```

### Audio Augmentation

```rust
use axonml_audio::{AddNoise, TimeStretch, PitchShift};
use axonml_data::Transform;

let noisy     = AddNoise::new(20.0).apply(&waveform);  // SNR in dB
let stretched = TimeStretch::new(1.5).apply(&waveform);
let shifted   = PitchShift::new(2.0).apply(&waveform); // semitones
```

### Normalization and Trimming

```rust
use axonml_audio::{NormalizeAudio, TrimSilence};
use axonml_data::Transform;

let normalized = NormalizeAudio::new().apply(&waveform);      // peak to 1.0
let trimmed    = TrimSilence::new(-60.0).apply(&waveform);    // threshold in dB
let trimmed    = TrimSilence::default_threshold().apply(&waveform);
```

### Full Audio Processing Pipeline

```rust
use axonml_audio::prelude::*;
use axonml_data::DataLoader;

let dataset = SyntheticCommandDataset::medium();
let loader  = DataLoader::new(dataset, 32).shuffle(true);

let resample  = Resample::new(16000, 8000);
let normalize = NormalizeAudio::new();
let mel       = MelSpectrogram::with_params(8000, 256, 128, 40);

for batch in loader.iter() {
    // batch.data: [B, n_samples]   batch.targets: [B, 1]
    // Apply per-sample transforms inside the training loop, or pre-apply with MapDataset.
}
```

### Sequence-to-Sequence Audio Tasks

```rust
use axonml_audio::AudioSeq2SeqDataset;

// Noise reduction (noisy -> clean pairs)
let dataset = AudioSeq2SeqDataset::noise_reduction_task(
    100,    // num_samples
    16000,  // sample_rate
    0.5,    // duration (seconds)
);

let (noisy, clean) = dataset.get(0).unwrap();
assert_eq!(noisy.shape(), clean.shape());

// Or bring your own source/target waveforms:
let ds = AudioSeq2SeqDataset::new(sources, targets);
```

## Tests

```bash
cargo test -p axonml-audio
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

_Last updated: 2026-04-16 (v0.6.1)_
