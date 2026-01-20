# axonml-audio

<p align="center">
  <img src="../../assets/logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-audio"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-audio** provides audio processing functionality for the AxonML framework. It includes signal processing transforms for spectrograms and feature extraction, audio augmentation techniques, and datasets for audio classification, speech recognition, and music genre tasks.

## Features

- **Resampling** - Sample rate conversion using linear interpolation
- **Mel Spectrogram** - Compute mel-scaled spectrograms with configurable FFT size, hop length, and mel bins
- **MFCC** - Mel-frequency cepstral coefficients for speech and audio feature extraction
- **Time Stretching** - Speed up or slow down audio without changing pitch
- **Pitch Shifting** - Change pitch without altering duration
- **Noise Augmentation** - Add Gaussian noise with configurable SNR for data augmentation
- **Audio Normalization** - Peak normalization to maximum amplitude
- **Silence Trimming** - Remove silence from beginning and end of audio
- **Synthetic Datasets** - Command recognition, music genre, and speaker identification datasets

## Modules

| Module | Description |
|--------|-------------|
| `transforms` | Audio signal processing transforms (`Resample`, `MelSpectrogram`, `MFCC`, `TimeStretch`, `PitchShift`, `AddNoise`, `NormalizeAudio`, `TrimSilence`) |
| `datasets` | Audio dataset implementations (`SyntheticCommandDataset`, `SyntheticMusicDataset`, `SyntheticSpeakerDataset`, `AudioSeq2SeqDataset`) |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-audio = "0.1.0"
```

### Loading Audio Datasets

```rust
use axonml_audio::prelude::*;

// Synthetic command dataset (like "yes", "no", "stop")
let dataset = SyntheticCommandDataset::small();  // 100 samples, 10 classes
let dataset = SyntheticCommandDataset::medium(); // 1000 samples
let dataset = SyntheticCommandDataset::large();  // 10000 samples, 35 classes

// Music genre dataset
let music = SyntheticMusicDataset::small();  // 5 genres

// Speaker identification dataset
let speakers = SyntheticSpeakerDataset::small();  // 5 speakers

// Get a sample
let (waveform, label) = dataset.get(0).unwrap();
println!("Sample rate: {}", dataset.sample_rate());
println!("Waveform shape: {:?}", waveform.shape());
```

### Mel Spectrogram

```rust
use axonml_audio::{MelSpectrogram, Transform};

// Default parameters
let mel = MelSpectrogram::new(16000);  // 16kHz sample rate

// Custom parameters
let mel = MelSpectrogram::with_params(
    16000,  // sample rate
    512,    // n_fft
    256,    // hop_length
    40,     // n_mels
);

let spectrogram = mel.apply(&waveform);
assert_eq!(spectrogram.shape()[0], 40);  // n_mels
```

### MFCC Feature Extraction

```rust
use axonml_audio::{MFCC, Transform};

let mfcc = MFCC::new(16000, 13);  // 16kHz, 13 coefficients

let (waveform, _) = dataset.get(0).unwrap();
let coefficients = mfcc.apply(&waveform);
assert_eq!(coefficients.shape()[0], 13);  // n_mfcc
```

### Audio Resampling

```rust
use axonml_audio::{Resample, Transform};

// Resample from 22050Hz to 16000Hz
let resample = Resample::new(22050, 16000);
let resampled = resample.apply(&waveform);

// New length proportional to sample rate ratio
```

### Audio Augmentation

```rust
use axonml_audio::{AddNoise, TimeStretch, PitchShift, Transform};

// Add Gaussian noise with 20dB SNR
let add_noise = AddNoise::new(20.0);
let noisy = add_noise.apply(&waveform);

// Time stretch (speed up 1.5x)
let stretch = TimeStretch::new(1.5);
let stretched = stretch.apply(&waveform);

// Pitch shift up 2 semitones
let shift = PitchShift::new(2.0);
let shifted = shift.apply(&waveform);
```

### Audio Normalization and Trimming

```rust
use axonml_audio::{NormalizeAudio, TrimSilence, Transform};

// Normalize to peak amplitude of 1.0
let normalize = NormalizeAudio::new();
let normalized = normalize.apply(&waveform);

// Trim silence below -60dB
let trim = TrimSilence::new(-60.0);
let trimmed = trim.apply(&waveform);
```

### Full Audio Processing Pipeline

```rust
use axonml_audio::prelude::*;
use axonml_data::DataLoader;

// Create dataset and dataloader
let dataset = SyntheticCommandDataset::medium();
let loader = DataLoader::new(dataset, 32).shuffle(true);

// Define transforms
let resample = Resample::new(16000, 8000);
let normalize = NormalizeAudio::new();
let mel = MelSpectrogram::with_params(8000, 256, 128, 40);

// Process batches
for batch in loader.iter() {
    let waveforms = batch.data;
    let labels = batch.targets;

    // Apply pipeline to each sample
    // ... training code ...
}
```

### Sequence-to-Sequence Audio Tasks

```rust
use axonml_audio::AudioSeq2SeqDataset;

// Noise reduction dataset (noisy -> clean pairs)
let dataset = AudioSeq2SeqDataset::noise_reduction_task(
    100,    // num_samples
    16000,  // sample_rate
    0.5,    // duration in seconds
);

let (noisy, clean) = dataset.get(0).unwrap();
assert_eq!(noisy.shape(), clean.shape());
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-audio
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
