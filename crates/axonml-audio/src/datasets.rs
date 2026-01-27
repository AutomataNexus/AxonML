//! Audio Datasets - Dataset implementations for audio processing tasks
//!
//! Provides datasets for audio classification, speech recognition, and music tasks.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_data::Dataset;
use axonml_tensor::Tensor;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;

// =============================================================================
// Audio Classification Dataset
// =============================================================================

/// A dataset for audio classification tasks.
pub struct AudioClassificationDataset {
    waveforms: Vec<Tensor<f32>>,
    labels: Vec<usize>,
    sample_rate: usize,
    num_classes: usize,
}

impl AudioClassificationDataset {
    /// Creates a new audio classification dataset from waveforms and labels.
    #[must_use]
    pub fn new(
        waveforms: Vec<Tensor<f32>>,
        labels: Vec<usize>,
        sample_rate: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            waveforms,
            labels,
            sample_rate,
            num_classes,
        }
    }

    /// Returns the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Returns the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

impl Dataset for AudioClassificationDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.waveforms.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.len() {
            return None;
        }

        let waveform = self.waveforms[index].clone();

        // One-hot encode label
        let mut label_vec = vec![0.0f32; self.num_classes];
        label_vec[self.labels[index]] = 1.0;
        let label = Tensor::from_vec(label_vec, &[self.num_classes]).unwrap();

        Some((waveform, label))
    }
}

// =============================================================================
// Synthetic Audio Command Dataset
// =============================================================================

/// A synthetic dataset simulating audio commands (like "yes", "no", "stop", etc.).
/// Uses different frequency patterns to represent different commands.
pub struct SyntheticCommandDataset {
    num_samples: usize,
    sample_rate: usize,
    duration: f32,
    num_classes: usize,
}

impl SyntheticCommandDataset {
    /// Creates a new synthetic command dataset.
    #[must_use]
    pub fn new(num_samples: usize, sample_rate: usize, duration: f32, num_classes: usize) -> Self {
        Self {
            num_samples,
            sample_rate,
            duration,
            num_classes: num_classes.max(2),
        }
    }

    /// Creates a small dataset with 100 samples.
    #[must_use]
    pub fn small() -> Self {
        Self::new(100, 16000, 0.5, 10)
    }

    /// Creates a medium dataset with 1000 samples.
    #[must_use]
    pub fn medium() -> Self {
        Self::new(1000, 16000, 0.5, 10)
    }

    /// Creates a large dataset with 10000 samples.
    #[must_use]
    pub fn large() -> Self {
        Self::new(10000, 16000, 0.5, 35)
    }

    /// Generates a synthetic waveform for a given class.
    fn generate_waveform(&self, class: usize, seed: u64) -> Tensor<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (self.sample_rate as f32 * self.duration) as usize;

        // Different classes have different frequency patterns
        let base_freq = 200.0 + (class as f32 * 100.0);
        let freq_variation = rng.gen_range(0.9..1.1);
        let freq = base_freq * freq_variation;

        // Add some harmonics based on class
        let harmonic_weight = 0.3 + (class as f32 * 0.05);

        let data: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / self.sample_rate as f32;
                let fundamental = (2.0 * PI * freq * t).sin();
                let harmonic1 = harmonic_weight * (2.0 * PI * freq * 2.0 * t).sin();
                let harmonic2 = harmonic_weight * 0.5 * (2.0 * PI * freq * 3.0 * t).sin();

                // Add envelope
                let envelope = if t < 0.05 {
                    t / 0.05
                } else if t > self.duration - 0.1 {
                    (self.duration - t) / 0.1
                } else {
                    1.0
                };

                // Add small noise
                let noise: f32 = rng.gen_range(-0.05..0.05);

                (fundamental + harmonic1 + harmonic2 + noise) * envelope * 0.5
            })
            .collect();

        Tensor::from_vec(data, &[n_samples]).unwrap()
    }

    /// Returns the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Returns the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

impl Dataset for SyntheticCommandDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.num_samples {
            return None;
        }

        let class = index % self.num_classes;
        let waveform = self.generate_waveform(class, index as u64);

        // One-hot encode label
        let mut label_vec = vec![0.0f32; self.num_classes];
        label_vec[class] = 1.0;
        let label = Tensor::from_vec(label_vec, &[self.num_classes]).unwrap();

        Some((waveform, label))
    }
}

// =============================================================================
// Synthetic Music Genre Dataset
// =============================================================================

/// A synthetic dataset for music genre classification.
/// Simulates different genres with distinct rhythm and frequency patterns.
pub struct SyntheticMusicDataset {
    num_samples: usize,
    sample_rate: usize,
    duration: f32,
    num_genres: usize,
}

impl SyntheticMusicDataset {
    /// Creates a new synthetic music dataset.
    #[must_use]
    pub fn new(num_samples: usize, sample_rate: usize, duration: f32, num_genres: usize) -> Self {
        Self {
            num_samples,
            sample_rate,
            duration,
            num_genres: num_genres.max(2),
        }
    }

    /// Creates a small dataset.
    #[must_use]
    pub fn small() -> Self {
        Self::new(100, 22050, 1.0, 5)
    }

    /// Creates a medium dataset.
    #[must_use]
    pub fn medium() -> Self {
        Self::new(500, 22050, 2.0, 10)
    }

    /// Generates a synthetic waveform for a music genre.
    fn generate_waveform(&self, genre: usize, seed: u64) -> Tensor<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (self.sample_rate as f32 * self.duration) as usize;

        // Different genres have different characteristics
        let bpm = match genre % 5 {
            0 => 60.0 + rng.gen_range(-5.0..5.0),    // Classical - slow
            1 => 90.0 + rng.gen_range(-10.0..10.0),  // Jazz
            2 => 120.0 + rng.gen_range(-10.0..10.0), // Pop
            3 => 140.0 + rng.gen_range(-15.0..15.0), // Electronic
            _ => 180.0 + rng.gen_range(-20.0..20.0), // Metal
        };

        let beat_duration = 60.0 / bpm;
        let base_freq = 220.0 + (genre as f32 * 50.0);

        let data: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / self.sample_rate as f32;
                let beat_phase = (t / beat_duration).fract();

                // Create rhythm pattern
                let rhythm = if beat_phase < 0.1 {
                    1.0 - beat_phase / 0.1
                } else {
                    0.0
                };

                // Melodic content
                let melody_freq = base_freq * (1.0 + 0.2 * (t * 2.0 * PI / beat_duration).sin());
                let melody = (2.0 * PI * melody_freq * t).sin();

                // Bass
                let bass = 0.5 * (2.0 * PI * base_freq * 0.5 * t).sin();

                // Combine with genre-specific mixing
                let mix = match genre % 5 {
                    0 => melody * 0.8 + bass * 0.2,
                    1 => melody * 0.6 + bass * 0.3 + rhythm * 0.1,
                    2 => melody * 0.5 + bass * 0.3 + rhythm * 0.2,
                    3 => melody * 0.3 + bass * 0.4 + rhythm * 0.3,
                    _ => melody * 0.4 + bass * 0.5 + rhythm * 0.3,
                };

                // Add noise for texture
                let noise: f32 = rng.gen_range(-0.02..0.02);

                (mix + noise) * 0.5
            })
            .collect();

        Tensor::from_vec(data, &[n_samples]).unwrap()
    }

    /// Returns the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Returns the number of genres.
    #[must_use]
    pub fn num_genres(&self) -> usize {
        self.num_genres
    }
}

impl Dataset for SyntheticMusicDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.num_samples {
            return None;
        }

        let genre = index % self.num_genres;
        let waveform = self.generate_waveform(genre, index as u64);

        // One-hot encode label
        let mut label_vec = vec![0.0f32; self.num_genres];
        label_vec[genre] = 1.0;
        let label = Tensor::from_vec(label_vec, &[self.num_genres]).unwrap();

        Some((waveform, label))
    }
}

// =============================================================================
// Synthetic Speech Dataset
// =============================================================================

/// A synthetic dataset for speaker identification.
/// Simulates different speakers with distinct vocal characteristics.
pub struct SyntheticSpeakerDataset {
    num_samples: usize,
    sample_rate: usize,
    duration: f32,
    num_speakers: usize,
}

impl SyntheticSpeakerDataset {
    /// Creates a new synthetic speaker dataset.
    #[must_use]
    pub fn new(num_samples: usize, sample_rate: usize, duration: f32, num_speakers: usize) -> Self {
        Self {
            num_samples,
            sample_rate,
            duration,
            num_speakers: num_speakers.max(2),
        }
    }

    /// Creates a small dataset.
    #[must_use]
    pub fn small() -> Self {
        Self::new(100, 16000, 0.5, 5)
    }

    /// Creates a medium dataset.
    #[must_use]
    pub fn medium() -> Self {
        Self::new(500, 16000, 1.0, 20)
    }

    /// Generates a synthetic waveform for a speaker.
    fn generate_waveform(&self, speaker: usize, seed: u64) -> Tensor<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (self.sample_rate as f32 * self.duration) as usize;

        // Different speakers have different fundamental frequencies
        let f0 = 80.0 + (speaker as f32 * 15.0) + rng.gen_range(-10.0..10.0);

        // Formant frequencies (simplified vocal tract model)
        let formants = [
            f0 * 5.0 + (speaker as f32 * 20.0),
            f0 * 10.0 + (speaker as f32 * 30.0),
            f0 * 25.0 + (speaker as f32 * 10.0),
        ];

        let data: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / self.sample_rate as f32;

                // Glottal pulse train
                let pulse_phase = (t * f0).fract();
                let glottal = if pulse_phase < 0.3 {
                    (pulse_phase * PI / 0.3).sin()
                } else {
                    0.0
                };

                // Add formants
                let mut signal = glottal;
                for &formant in &formants {
                    signal += 0.2 * glottal * (2.0 * PI * formant * t).sin();
                }

                // Add some variation
                let variation = 1.0 + 0.1 * (t * 5.0 * PI).sin();

                // Add noise for breathiness
                let noise: f32 = rng.gen_range(-0.03..0.03);

                signal * variation * 0.3 + noise
            })
            .collect();

        Tensor::from_vec(data, &[n_samples]).unwrap()
    }

    /// Returns the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Returns the number of speakers.
    #[must_use]
    pub fn num_speakers(&self) -> usize {
        self.num_speakers
    }
}

impl Dataset for SyntheticSpeakerDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.num_samples {
            return None;
        }

        let speaker = index % self.num_speakers;
        let waveform = self.generate_waveform(speaker, index as u64);

        // One-hot encode label
        let mut label_vec = vec![0.0f32; self.num_speakers];
        label_vec[speaker] = 1.0;
        let label = Tensor::from_vec(label_vec, &[self.num_speakers]).unwrap();

        Some((waveform, label))
    }
}

// =============================================================================
// Sequence-to-Sequence Audio Dataset
// =============================================================================

/// A dataset for audio sequence-to-sequence tasks.
pub struct AudioSeq2SeqDataset {
    sources: Vec<Tensor<f32>>,
    targets: Vec<Tensor<f32>>,
}

impl AudioSeq2SeqDataset {
    /// Creates a new audio seq2seq dataset.
    #[must_use]
    pub fn new(sources: Vec<Tensor<f32>>, targets: Vec<Tensor<f32>>) -> Self {
        Self { sources, targets }
    }

    /// Creates a synthetic noise reduction dataset.
    #[must_use]
    pub fn noise_reduction_task(num_samples: usize, sample_rate: usize, duration: f32) -> Self {
        let n_samples_per = (sample_rate as f32 * duration) as usize;
        let mut sources = Vec::with_capacity(num_samples);
        let mut targets = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let mut rng = rand::rngs::StdRng::seed_from_u64(i as u64);
            let freq = 200.0 + (i as f32 * 50.0) % 800.0;

            // Clean signal
            let clean: Vec<f32> = (0..n_samples_per)
                .map(|j| {
                    let t = j as f32 / sample_rate as f32;
                    (2.0 * PI * freq * t).sin() * 0.5
                })
                .collect();

            // Noisy signal
            let noisy: Vec<f32> = clean
                .iter()
                .map(|&x| x + rng.gen_range(-0.2..0.2))
                .collect();

            sources.push(Tensor::from_vec(noisy, &[n_samples_per]).unwrap());
            targets.push(Tensor::from_vec(clean, &[n_samples_per]).unwrap());
        }

        Self { sources, targets }
    }
}

impl Dataset for AudioSeq2SeqDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.sources.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.len() {
            return None;
        }

        Some((self.sources[index].clone(), self.targets[index].clone()))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_classification_dataset() {
        let waveforms = vec![
            Tensor::from_vec(vec![0.0; 16000], &[16000]).unwrap(),
            Tensor::from_vec(vec![0.0; 16000], &[16000]).unwrap(),
        ];
        let labels = vec![0, 1];

        let dataset = AudioClassificationDataset::new(waveforms, labels, 16000, 2);

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.sample_rate(), 16000);
        assert_eq!(dataset.num_classes(), 2);

        let (wave, label) = dataset.get(0).unwrap();
        assert_eq!(wave.shape(), &[16000]);
        assert_eq!(label.shape(), &[2]);
    }

    #[test]
    fn test_synthetic_command_dataset() {
        let dataset = SyntheticCommandDataset::small();

        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_classes(), 10);
        assert_eq!(dataset.sample_rate(), 16000);

        let (wave, label) = dataset.get(0).unwrap();
        assert_eq!(wave.shape()[0], 8000); // 0.5s at 16000Hz
        assert_eq!(label.shape(), &[10]);

        // Check label is one-hot
        let label_sum: f32 = label.to_vec().iter().sum();
        assert!((label_sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_synthetic_command_dataset_different_classes() {
        let dataset = SyntheticCommandDataset::small();

        // Different indices should produce different class labels
        let (_, label0) = dataset.get(0).unwrap();
        let (_, label1) = dataset.get(1).unwrap();

        let label0_vec = label0.to_vec();
        let label1_vec = label1.to_vec();

        let class0 = label0_vec.iter().position(|&x| x > 0.5).unwrap();
        let class1 = label1_vec.iter().position(|&x| x > 0.5).unwrap();

        assert_eq!(class0, 0);
        assert_eq!(class1, 1);
    }

    #[test]
    fn test_synthetic_music_dataset() {
        let dataset = SyntheticMusicDataset::small();

        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_genres(), 5);
        assert_eq!(dataset.sample_rate(), 22050);

        let (wave, label) = dataset.get(0).unwrap();
        assert_eq!(wave.shape()[0], 22050); // 1.0s at 22050Hz
        assert_eq!(label.shape(), &[5]);
    }

    #[test]
    fn test_synthetic_speaker_dataset() {
        let dataset = SyntheticSpeakerDataset::small();

        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_speakers(), 5);
        assert_eq!(dataset.sample_rate(), 16000);

        let (wave, label) = dataset.get(0).unwrap();
        assert_eq!(wave.shape()[0], 8000); // 0.5s at 16000Hz
        assert_eq!(label.shape(), &[5]);
    }

    #[test]
    fn test_audio_seq2seq_dataset() {
        let dataset = AudioSeq2SeqDataset::noise_reduction_task(10, 16000, 0.1);

        assert_eq!(dataset.len(), 10);

        let (source, target) = dataset.get(0).unwrap();
        assert_eq!(source.shape(), target.shape());
    }

    #[test]
    fn test_dataset_bounds() {
        let dataset = SyntheticCommandDataset::small();

        assert!(dataset.get(99).is_some());
        assert!(dataset.get(100).is_none());
    }

    #[test]
    fn test_waveform_values_in_range() {
        let dataset = SyntheticCommandDataset::small();

        let (wave, _) = dataset.get(0).unwrap();
        let data = wave.to_vec();

        // All values should be in reasonable range
        for &val in &data {
            assert!(val.abs() <= 1.0, "Waveform value {val} out of range");
        }
    }

    #[test]
    fn test_music_dataset_different_genres() {
        let dataset = SyntheticMusicDataset::small();

        // Get waveforms from two different genres
        let (wave0, _) = dataset.get(0).unwrap();
        let (wave1, _) = dataset.get(1).unwrap();

        // The waveforms should be different
        assert_ne!(wave0.to_vec(), wave1.to_vec());
    }
}
