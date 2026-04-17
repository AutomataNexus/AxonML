# axonml-audio Documentation

> Audio processing for the AxonML ML framework.

## Overview

`axonml-audio` provides audio feature-extraction transforms (mel
spectrogram, MFCC, resample), augmentation ops (noise, time-stretch,
pitch-shift, silence-trim), and synthetic datasets for command recognition,
music genre classification, and speaker identification. Equivalent to
PyTorch's `torchaudio` at a smaller scope. Transforms use `rustfft` for
O(n log n) FFT; everything else is pure Rust.

## Modules

### `transforms`

All transforms implement `axonml_data::Transform` (`apply(&Tensor<f32>) ->
Tensor<f32>`).

#### `Resample`

Linear-interpolation resampling between sample rates.

```rust
use axonml_audio::Resample;

let resample = Resample::new(44100, 16000);
let resampled = resample.apply(&audio);
```

#### `MelSpectrogram`

STFT + mel-filterbank projection.

```rust
use axonml_audio::MelSpectrogram;

let mel = MelSpectrogram::new(sample_rate);                      // sensible defaults
let mel = MelSpectrogram::with_params(16000, 512, 256, 40);      // sr, n_fft, hop, n_mels

let spec = mel.apply(&waveform); // shape: [n_mels, time_frames]
```

#### `MFCC`

Mel-frequency cepstral coefficients (mel-spec -> log -> DCT).

```rust
use axonml_audio::MFCC;
let mfcc = MFCC::new(16000, 13);
let features = mfcc.apply(&waveform); // [n_mfcc, time_frames]
```

#### `TimeStretch`

Phase-vocoder-style time stretching by a rate factor.

```rust
let stretch = TimeStretch::new(1.2);
let longer = stretch.apply(&waveform);
```

#### `PitchShift`

Pitch shift in semitones.

```rust
let shift = PitchShift::new(2.0); // +2 semitones
let shifted = shift.apply(&waveform);
```

#### `AddNoise`

Additive Gaussian noise at a target SNR (dB).

```rust
let add_noise = AddNoise::new(20.0);
let noisy = add_noise.apply(&waveform);
```

#### `NormalizeAudio`

Normalize peak to 1.0.

```rust
let normalize = NormalizeAudio::new();
let normalized = normalize.apply(&waveform);
```

#### `TrimSilence`

Trim leading/trailing silence below a dB threshold.

```rust
let trim = TrimSilence::new(-40.0);
let trimmed = trim.apply(&waveform);
```

### `datasets`

Synthetic datasets for testing and benchmarking. All implement
`axonml_data::Dataset`.

#### `AudioClassificationDataset`

Generic container: `Vec<Tensor<f32>> waveforms`, `Vec<usize> labels`,
`sample_rate`, `num_classes`. `Dataset::Item = (Tensor<f32>, Tensor<f32>)`
where the label is a single-element class-index tensor (compatible with
`CrossEntropyLoss`).

```rust
use axonml_audio::AudioClassificationDataset;
let ds = AudioClassificationDataset::new(waveforms, labels, 16000, 10);
```

#### `AudioSeq2SeqDataset`

Pairs of source/target tensors for sequence-to-sequence audio tasks.

```rust
use axonml_audio::AudioSeq2SeqDataset;
let ds = AudioSeq2SeqDataset::new(sources, targets);
```

#### Synthetic generators

```rust
use axonml_audio::{SyntheticCommandDataset, SyntheticMusicDataset, SyntheticSpeakerDataset};

// num_samples, sample_rate, duration_sec, num_classes
let cmd = SyntheticCommandDataset::new(8000, 16000, 1.0, 10);
let cmd_small = SyntheticCommandDataset::small();

let music = SyntheticMusicDataset::new(2000, 22050, 3.0, 4);
let music_small = SyntheticMusicDataset::small();

let spk = SyntheticSpeakerDataset::new(1000, 16000, 2.0, 8);
let spk_small = SyntheticSpeakerDataset::small();
```

## Usage Examples

### Feature extraction

```rust
use axonml::prelude::*;

let sample_rate = 16000;
let duration = 2.0;
let n = (sample_rate as f32 * duration) as usize;

let freq = 440.0;
let audio: Vec<f32> = (0..n)
    .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32 / sample_rate as f32)).sin())
    .collect();
let waveform = Tensor::from_vec(audio, &[n]).unwrap();

let mel = MelSpectrogram::with_params(16000, 512, 256, 40);
let spec = mel.apply(&waveform);
println!("spec shape: {:?}", spec.shape());

let mfcc = MFCC::new(16000, 13);
let coefs = mfcc.apply(&waveform);
println!("mfcc shape: {:?}", coefs.shape());
```

### Command classification pipeline

```rust
use axonml::prelude::*;

let train = SyntheticCommandDataset::new(8000, 16000, 1.0, 10);
let test  = SyntheticCommandDataset::new(2000, 16000, 1.0, 10);
let loader = DataLoader::with_shuffle(train, 32, true);

let mel = MelSpectrogram::with_params(16000, 512, 256, 40);
let model = create_audio_cnn();
let mut opt = Adam::new(model.parameters(), 0.001);

for epoch in 0..10 {
    for batch in loader.iter() {
        let specs = batch_transform(&batch.data, &mel);
        let out = model.forward(&specs);
        let loss = cross_entropy(&out, &batch.targets);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }
}
```

### Augmentation pipeline

```rust
use axonml_audio::*;

let audio = load_audio("speech.wav")?;

let normalize = NormalizeAudio::new();
let add_noise = AddNoise::new(20.0);

let normalized = normalize.apply(&audio);
let augmented  = add_noise.apply(&normalized);
```

## Parameter Reference

### Mel spectrogram

| Parameter     | Description                | Typical               |
|---------------|----------------------------|-----------------------|
| `sample_rate` | Audio sample rate          | 16000 / 22050 / 44100 |
| `n_fft`       | FFT window size            | 256 / 512 / 1024      |
| `hop_length`  | Samples between frames     | n_fft / 4             |
| `n_mels`      | Number of mel bands        | 40 / 80 / 128         |

### MFCC

| Parameter     | Description              | Typical     |
|---------------|--------------------------|-------------|
| `n_mfcc`      | Number of coefficients   | 13 / 20 / 40 |

## Related Modules

- [Data](../../crates/axonml-data) — `DataLoader`, `Dataset`, `Transform`
- [Neural Networks](../nn/README.md) — models for audio classification

## Last updated

0.6.1 (2026-04-16)
