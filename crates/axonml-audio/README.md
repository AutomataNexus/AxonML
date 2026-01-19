# axonml-audio

[![Crates.io](https://img.shields.io/crates/v/axonml-audio.svg)](https://crates.io/crates/axonml-audio)
[![Docs.rs](https://docs.rs/axonml-audio/badge.svg)](https://docs.rs/axonml-audio)
[![Downloads](https://img.shields.io/crates/d/axonml-audio.svg)](https://crates.io/crates/axonml-audio)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Audio processing utilities for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-audio` provides audio processing transforms, datasets, and utilities for speech recognition, audio classification, and music information retrieval tasks. Inspired by torchaudio.

## Features

### Audio Transforms
- **MelSpectrogram** - Mel-frequency spectrograms
- **MFCC** - Mel-frequency cepstral coefficients
- **Spectrogram** - Short-time Fourier transform
- **Resample** - Sample rate conversion
- **Normalize** - Audio normalization

### Augmentations
- **AddNoise** - Add background noise
- **TimeStretch** - Change speed without pitch
- **PitchShift** - Change pitch without speed
- **TimeMask/FreqMask** - SpecAugment masking

### Datasets
- **SyntheticCommandDataset** - Speech command recognition
- **SyntheticMusicDataset** - Music genre classification

## Installation

```toml
[dependencies]
axonml-audio = "0.1"
```

## Usage

### Mel Spectrogram

```rust
use axonml_audio::transforms::MelSpectrogram;

let mel_transform = MelSpectrogram::new(16000)  // sample_rate
    .n_fft(400)
    .hop_length(160)
    .n_mels(80);

// waveform: [batch, time] -> mel: [batch, n_mels, frames]
let waveform = load_audio("speech.wav");
let mel = mel_transform.forward(&waveform);
```

### MFCC Features

```rust
use axonml_audio::transforms::MFCC;

let mfcc_transform = MFCC::new(16000)
    .n_mfcc(13)
    .n_mels(40);

let mfcc = mfcc_transform.forward(&waveform);  // [batch, n_mfcc, frames]
```

### Audio Augmentation

```rust
use axonml_audio::transforms::{Compose, AddNoise, TimeStretch, TimeMask};

let augment = Compose::new()
    .add(AddNoise::new(0.005))      // SNR noise
    .add(TimeStretch::new(0.8, 1.2)) // Random speed 0.8x-1.2x
    .add(TimeMask::new(80, 2));      // SpecAugment time mask

let augmented = augment.forward(&mel_spectrogram);
```

### Speech Command Dataset

```rust
use axonml_audio::datasets::SyntheticCommandDataset;
use axonml_data::DataLoader;

let dataset = SyntheticCommandDataset::new(true, 8000);  // train, num_samples
let dataloader = DataLoader::new(dataset, 32).shuffle(true);

for (audio, label) in dataloader.iter() {
    // audio: [32, 16000] (1 second at 16kHz)
    // label: [32] (command index)
}
```

## API Reference

### Transforms

| Transform | Description |
|-----------|-------------|
| `Spectrogram` | STFT magnitude spectrogram |
| `MelSpectrogram` | Mel-scale spectrogram |
| `MFCC` | Mel-frequency cepstral coefficients |
| `Resample` | Change sample rate |
| `Normalize` | Normalize amplitude |
| `AddNoise` | Add random noise |
| `TimeStretch` | Speed change |
| `PitchShift` | Pitch change |
| `TimeMask` | Random time masking |
| `FreqMask` | Random frequency masking |

## Part of Axonml

```toml
[dependencies]
axonml = { version = "0.1", features = ["audio"] }
```

## License

MIT OR Apache-2.0
