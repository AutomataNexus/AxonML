# axonml-audio Documentation

> Audio processing utilities for the Axonml ML framework.

## Overview

`axonml-audio` provides audio processing capabilities including transforms for feature extraction (spectrograms, MFCCs), audio augmentation, and synthetic audio datasets. It's the Axonml equivalent of PyTorch's torchaudio.

## Modules

### transforms/

Audio processing transforms implementing the `Transform` trait.

#### Resample

Change the sample rate of audio:

```rust
use axonml_audio::Resample;

// Resample from 44100 Hz to 16000 Hz
let resample = Resample::new(44100, 16000);
let resampled = resample.apply(&audio);
```

#### MelSpectrogram

Convert waveform to mel-scaled spectrogram:

```rust
use axonml_audio::MelSpectrogram;

// Default parameters
let mel = MelSpectrogram::new(sample_rate);

// Custom parameters
let mel = MelSpectrogram::with_params(
    sample_rate,    // e.g., 16000
    n_fft,          // FFT size, e.g., 512
    hop_length,     // Hop between windows, e.g., 256
    n_mels,         // Number of mel bins, e.g., 40
);

let spectrogram = mel.apply(&waveform);
// Output shape: [n_mels, time_frames]
```

#### MFCC

Mel-frequency cepstral coefficients:

```rust
use axonml_audio::MFCC;

// Create MFCC transform
let mfcc = MFCC::new(sample_rate, n_mfcc);

// With custom mel spectrogram parameters
let mfcc = MFCC::with_params(sample_rate, n_mfcc, n_fft, hop_length, n_mels);

let features = mfcc.apply(&waveform);
// Output shape: [n_mfcc, time_frames]
```

#### NormalizeAudio

Normalize audio to [-1, 1] range:

```rust
use axonml_audio::NormalizeAudio;

let normalize = NormalizeAudio::new();
let normalized = normalize.apply(&waveform);
```

#### AddNoise

Add random noise for data augmentation:

```rust
use axonml_audio::AddNoise;

// Add noise with 20dB SNR
let add_noise = AddNoise::new(20.0);
let noisy = add_noise.apply(&waveform);
```

### datasets/

Synthetic audio datasets for testing.

#### SyntheticCommandDataset

Synthetic speech command dataset:

```rust
use axonml_audio::SyntheticCommandDataset;
use axonml_data::Dataset;

// Create small dataset
let dataset = SyntheticCommandDataset::small();

// Create with specific size
let dataset = SyntheticCommandDataset::new(1000, 10);

println!("Samples: {}", dataset.len());
println!("Classes: {}", dataset.num_classes());

let (waveform, label) = dataset.get(0).unwrap();
// waveform: [16000] tensor (1 second at 16kHz)
// label: [num_classes] one-hot tensor
```

#### SyntheticMusicDataset

Synthetic music genre classification dataset:

```rust
use axonml_audio::SyntheticMusicDataset;

let dataset = SyntheticMusicDataset::small();
println!("Samples: {}", dataset.len());
println!("Genres: {}", dataset.num_genres());

let (waveform, genre) = dataset.get(0).unwrap();
```

## Usage Examples

### Audio Classification Pipeline

```rust
use axonml::prelude::*;

fn main() {
    // 1. Create dataset
    let train_data = SyntheticCommandDataset::new(8000, 10);
    let test_data = SyntheticCommandDataset::new(2000, 10);

    // 2. Create data loader
    let train_loader = DataLoader::with_shuffle(train_data, 32, true);

    // 3. Define preprocessing
    let mel = MelSpectrogram::with_params(16000, 512, 256, 40);

    // 4. Create model (CNN for spectrograms)
    let model = create_audio_cnn();
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    // 5. Training loop
    for epoch in 0..10 {
        for batch in train_loader.iter() {
            // Apply mel spectrogram transform
            let specs = batch_transform(&batch.data, &mel);

            // Forward pass
            let output = model.forward(&specs);
            let loss = cross_entropy(&output, &batch.targets);

            // Backward pass
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}
```

### Feature Extraction

```rust
use axonml::prelude::*;

// Load or create audio
let sample_rate = 16000;
let duration = 2.0;
let n_samples = (sample_rate as f32 * duration) as usize;

// Generate sine wave
let frequency = 440.0;
let audio: Vec<f32> = (0..n_samples)
    .map(|i| {
        let t = i as f32 / sample_rate as f32;
        (2.0 * std::f32::consts::PI * frequency * t).sin()
    })
    .collect();

let waveform = Tensor::from_vec(audio, &[n_samples]).unwrap();

// Extract features
let mel = MelSpectrogram::with_params(16000, 512, 256, 40);
let spectrogram = mel.apply(&waveform);
println!("Spectrogram shape: {:?}", spectrogram.shape());

let mfcc = MFCC::new(16000, 13);
let mfcc_features = mfcc.apply(&waveform);
println!("MFCC shape: {:?}", mfcc_features.shape());
```

### Audio Augmentation

```rust
use axonml_audio::*;

// Original audio
let audio = load_audio("speech.wav")?;

// Augmentation pipeline
let normalize = NormalizeAudio::new();
let add_noise = AddNoise::new(20.0);  // 20dB SNR

// Apply augmentations
let normalized = normalize.apply(&audio);
let augmented = add_noise.apply(&normalized);
```

### Resampling

```rust
use axonml_audio::Resample;

// Audio at 44.1kHz
let audio_44k = load_audio("music.wav")?;

// Resample to 16kHz for speech processing
let resample = Resample::new(44100, 16000);
let audio_16k = resample.apply(&audio_44k);

println!("Original samples: {}", audio_44k.shape()[0]);
println!("Resampled samples: {}", audio_16k.shape()[0]);
```

## Audio Feature Reference

### Mel Spectrogram Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `sample_rate` | Audio sample rate | 16000, 22050, 44100 |
| `n_fft` | FFT window size | 256, 512, 1024, 2048 |
| `hop_length` | Samples between frames | n_fft / 4 |
| `n_mels` | Number of mel bands | 40, 80, 128 |

### MFCC Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n_mfcc` | Number of coefficients | 13, 20, 40 |

## Related Modules

- [Data](../data/README.md) - DataLoader and Dataset traits
- [Neural Networks](../nn/README.md) - Models for audio classification
- [Transforms](../data/README.md#transforms) - Transform trait

@version 0.1.0
@author AutomataNexus Development Team
