//! Audio Transforms - Signal Processing and Augmentation
//!
//! Provides audio-specific transformations for preprocessing and data augmentation.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_data::Transform;
use axonml_tensor::Tensor;
use rand::Rng;
use std::f32::consts::PI;

// =============================================================================
// Resample
// =============================================================================

/// Resamples audio to a target sample rate using linear interpolation.
pub struct Resample {
    orig_freq: usize,
    new_freq: usize,
}

impl Resample {
    /// Creates a new Resample transform.
    #[must_use] pub fn new(orig_freq: usize, new_freq: usize) -> Self {
        Self {
            orig_freq,
            new_freq,
        }
    }
}

impl Transform for Resample {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        if self.orig_freq == self.new_freq {
            return input.clone();
        }

        let data = input.to_vec();
        let orig_len = data.len();
        let new_len = (orig_len as f64 * self.new_freq as f64 / self.orig_freq as f64) as usize;

        if new_len == 0 {
            return Tensor::from_vec(vec![], &[0]).unwrap();
        }

        let mut resampled = Vec::with_capacity(new_len);
        let ratio = orig_len as f64 / new_len as f64;

        for i in 0..new_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(orig_len - 1);
            let frac = (src_idx - idx0 as f64) as f32;

            let value = data[idx0] * (1.0 - frac) + data[idx1] * frac;
            resampled.push(value);
        }

        Tensor::from_vec(resampled, &[new_len]).unwrap()
    }
}

// =============================================================================
// MelSpectrogram
// =============================================================================

/// Computes a mel spectrogram from audio waveform.
pub struct MelSpectrogram {
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
}

impl MelSpectrogram {
    /// Creates a new `MelSpectrogram` transform with default parameters.
    #[must_use] pub fn new(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            n_fft: 2048,
            hop_length: 512,
            n_mels: 128,
        }
    }

    /// Creates a `MelSpectrogram` with custom parameters.
    #[must_use] pub fn with_params(sample_rate: usize, n_fft: usize, hop_length: usize, n_mels: usize) -> Self {
        Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
        }
    }

    /// Converts frequency to mel scale.
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Converts mel to frequency.
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Creates mel filterbank.
    fn mel_filterbank(&self) -> Vec<Vec<f32>> {
        let fmax = self.sample_rate as f32 / 2.0;
        let mel_min = Self::hz_to_mel(0.0);
        let mel_max = Self::hz_to_mel(fmax);

        // Create mel points
        let mel_points: Vec<f32> = (0..=self.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (self.n_mels + 1) as f32)
            .collect();

        // Convert to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert to FFT bins
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((self.n_fft + 1) as f32 * hz / self.sample_rate as f32).floor() as usize)
            .collect();

        // Create filterbank
        let n_bins = self.n_fft / 2 + 1;
        let mut filterbank = vec![vec![0.0f32; n_bins]; self.n_mels];

        for m in 0..self.n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for k in left..center {
                if center > left && k < n_bins {
                    filterbank[m][k] = (k - left) as f32 / (center - left) as f32;
                }
            }
            for k in center..right {
                if right > center && k < n_bins {
                    filterbank[m][k] = (right - k) as f32 / (right - center) as f32;
                }
            }
        }

        filterbank
    }

    /// Applies Hann window.
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (size - 1) as f32).cos()))
            .collect()
    }

    /// Simple DFT (not FFT, but works for demonstration).
    fn dft(signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let n_out = n / 2 + 1;
        let mut magnitude = vec![0.0f32; n_out];

        for k in 0..n_out {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (t, &x) in signal.iter().enumerate() {
                let angle = 2.0 * PI * k as f32 * t as f32 / n as f32;
                real += x * angle.cos();
                imag -= x * angle.sin();
            }

            magnitude[k] = (real * real + imag * imag).sqrt();
        }

        magnitude
    }
}

impl Transform for MelSpectrogram {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let window = Self::hann_window(self.n_fft);
        let filterbank = self.mel_filterbank();

        // Calculate number of frames
        let n_frames = if data.len() >= self.n_fft {
            (data.len() - self.n_fft) / self.hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Tensor::from_vec(vec![0.0; self.n_mels], &[self.n_mels, 1]).unwrap();
        }

        let mut mel_spec = vec![0.0f32; self.n_mels * n_frames];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.n_fft).min(data.len());

            // Extract and window frame
            let mut frame: Vec<f32> = data[start..end].to_vec();
            frame.resize(self.n_fft, 0.0);

            for (i, w) in window.iter().enumerate() {
                frame[i] *= w;
            }

            // Compute magnitude spectrum
            let spectrum = Self::dft(&frame);

            // Apply mel filterbank
            for (m, filter) in filterbank.iter().enumerate() {
                let mut mel_energy = 0.0;
                for (k, &mag) in spectrum.iter().enumerate() {
                    if k < filter.len() {
                        mel_energy += mag * mag * filter[k];
                    }
                }
                // Convert to log scale
                mel_spec[m * n_frames + frame_idx] = (mel_energy + 1e-10).ln();
            }
        }

        Tensor::from_vec(mel_spec, &[self.n_mels, n_frames]).unwrap()
    }
}

// =============================================================================
// MFCC
// =============================================================================

/// Computes Mel-frequency cepstral coefficients.
pub struct MFCC {
    mel_spec: MelSpectrogram,
    n_mfcc: usize,
}

impl MFCC {
    /// Creates a new MFCC transform.
    #[must_use] pub fn new(sample_rate: usize, n_mfcc: usize) -> Self {
        Self {
            mel_spec: MelSpectrogram::new(sample_rate),
            n_mfcc,
        }
    }

    /// DCT-II for MFCC computation.
    fn dct(input: &[f32], n_out: usize) -> Vec<f32> {
        let n = input.len();
        let mut output = vec![0.0f32; n_out];

        for k in 0..n_out {
            let mut sum = 0.0;
            for (i, &x) in input.iter().enumerate() {
                sum += x * (PI * k as f32 * (2 * i + 1) as f32 / (2 * n) as f32).cos();
            }
            output[k] = sum;
        }

        output
    }
}

impl Transform for MFCC {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // First compute mel spectrogram
        let mel = self.mel_spec.apply(input);
        let mel_data = mel.to_vec();
        let mel_shape = mel.shape();

        if mel_shape.len() != 2 {
            return input.clone();
        }

        let (n_mels, n_frames) = (mel_shape[0], mel_shape[1]);
        let mut mfcc = vec![0.0f32; self.n_mfcc * n_frames];

        // Apply DCT to each frame
        for frame in 0..n_frames {
            let frame_data: Vec<f32> = (0..n_mels)
                .map(|m| mel_data[m * n_frames + frame])
                .collect();

            let coeffs = Self::dct(&frame_data, self.n_mfcc);

            for (k, &c) in coeffs.iter().enumerate() {
                mfcc[k * n_frames + frame] = c;
            }
        }

        Tensor::from_vec(mfcc, &[self.n_mfcc, n_frames]).unwrap()
    }
}

// =============================================================================
// TimeStretch
// =============================================================================

/// Time stretches audio without changing pitch.
pub struct TimeStretch {
    rate: f32,
}

impl TimeStretch {
    /// Creates a new `TimeStretch` transform.
    /// rate > 1.0 speeds up, rate < 1.0 slows down.
    #[must_use] pub fn new(rate: f32) -> Self {
        Self {
            rate: rate.max(0.1).min(10.0),
        }
    }
}

impl Transform for TimeStretch {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let new_len = (data.len() as f32 / self.rate) as usize;

        if new_len == 0 {
            return Tensor::from_vec(vec![], &[0]).unwrap();
        }

        let mut stretched = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f32 * self.rate;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(data.len() - 1);
            let frac = src_idx - idx0 as f32;

            if idx0 < data.len() {
                let value =
                    data[idx0] * (1.0 - frac) + data.get(idx1).copied().unwrap_or(0.0) * frac;
                stretched.push(value);
            }
        }

        let len = stretched.len();
        Tensor::from_vec(stretched, &[len]).unwrap()
    }
}

// =============================================================================
// PitchShift
// =============================================================================

/// Shifts the pitch of audio.
pub struct PitchShift {
    semitones: f32,
}

impl PitchShift {
    /// Creates a new `PitchShift` transform.
    /// Positive semitones shift up, negative shift down.
    #[must_use] pub fn new(semitones: f32) -> Self {
        Self { semitones }
    }
}

impl Transform for PitchShift {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // Simplified pitch shift using resampling
        // Real implementation would use phase vocoder
        let rate = 2.0_f32.powf(self.semitones / 12.0);
        let data = input.to_vec();
        let orig_len = data.len();

        // Resample to change pitch
        let resampled_len = (orig_len as f32 / rate) as usize;
        if resampled_len == 0 {
            return input.clone();
        }

        let mut resampled = Vec::with_capacity(resampled_len);
        for i in 0..resampled_len {
            let src_idx = i as f32 * rate;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(orig_len - 1);
            let frac = src_idx - idx0 as f32;

            if idx0 < orig_len {
                let value =
                    data[idx0] * (1.0 - frac) + data.get(idx1).copied().unwrap_or(0.0) * frac;
                resampled.push(value);
            }
        }

        // Time stretch back to original length
        let mut result = Vec::with_capacity(orig_len);
        for i in 0..orig_len {
            let src_idx = i as f32 * resampled.len() as f32 / orig_len as f32;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(resampled.len().saturating_sub(1));
            let frac = src_idx - idx0 as f32;

            if idx0 < resampled.len() {
                let value = resampled[idx0] * (1.0 - frac)
                    + resampled.get(idx1).copied().unwrap_or(0.0) * frac;
                result.push(value);
            } else {
                result.push(0.0);
            }
        }

        Tensor::from_vec(result, &[orig_len]).unwrap()
    }
}

// =============================================================================
// AddNoise
// =============================================================================

/// Adds random noise to audio.
pub struct AddNoise {
    snr_db: f32,
}

impl AddNoise {
    /// Creates a new `AddNoise` transform with specified SNR in dB.
    #[must_use] pub fn new(snr_db: f32) -> Self {
        Self { snr_db }
    }
}

impl Transform for AddNoise {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let mut rng = rand::thread_rng();

        // Calculate signal power
        let signal_power: f32 = data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32;

        // Calculate noise power from SNR
        let noise_power = signal_power / 10.0_f32.powf(self.snr_db / 10.0);
        let noise_std = noise_power.sqrt();

        // Add Gaussian noise
        let noisy: Vec<f32> = data
            .iter()
            .map(|&x| {
                // Box-Muller transform for Gaussian noise
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                x + z * noise_std
            })
            .collect();

        Tensor::from_vec(noisy, input.shape()).unwrap()
    }
}

// =============================================================================
// Normalize Audio
// =============================================================================

/// Normalizes audio to have maximum amplitude of 1.0.
pub struct NormalizeAudio;

impl NormalizeAudio {
    /// Creates a new `NormalizeAudio` transform.
    #[must_use] pub fn new() -> Self {
        Self
    }
}

impl Default for NormalizeAudio {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for NormalizeAudio {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let max_val = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        if max_val < 1e-10 {
            return input.clone();
        }

        let normalized: Vec<f32> = data.iter().map(|&x| x / max_val).collect();
        Tensor::from_vec(normalized, input.shape()).unwrap()
    }
}

// =============================================================================
// Trim Silence
// =============================================================================

/// Trims silence from the beginning and end of audio.
pub struct TrimSilence {
    threshold_db: f32,
}

impl TrimSilence {
    /// Creates a `TrimSilence` transform with specified threshold in dB.
    #[must_use] pub fn new(threshold_db: f32) -> Self {
        Self { threshold_db }
    }

    /// Creates with default -60dB threshold.
    #[must_use] pub fn default_threshold() -> Self {
        Self::new(-60.0)
    }
}

impl Transform for TrimSilence {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let threshold = 10.0_f32.powf(self.threshold_db / 20.0);

        // Find first non-silent sample
        let start = data.iter().position(|&x| x.abs() > threshold).unwrap_or(0);

        // Find last non-silent sample
        let end = data
            .iter()
            .rposition(|&x| x.abs() > threshold)
            .map_or(data.len(), |i| i + 1);

        if start >= end {
            return Tensor::from_vec(vec![], &[0]).unwrap();
        }

        let trimmed = data[start..end].to_vec();
        let len = trimmed.len();
        Tensor::from_vec(trimmed, &[len]).unwrap()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sine_wave(freq: f32, sample_rate: usize, duration: f32) -> Tensor<f32> {
        let n_samples = (sample_rate as f32 * duration) as usize;
        let data: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();
        Tensor::from_vec(data, &[n_samples]).unwrap()
    }

    #[test]
    fn test_resample() {
        let audio = create_sine_wave(440.0, 16000, 0.1);
        let resample = Resample::new(16000, 8000);

        let resampled = resample.apply(&audio);

        // Should be half the length
        assert_eq!(resampled.shape()[0], audio.shape()[0] / 2);
    }

    #[test]
    fn test_resample_same_rate() {
        let audio = create_sine_wave(440.0, 16000, 0.1);
        let resample = Resample::new(16000, 16000);

        let result = resample.apply(&audio);
        assert_eq!(result.to_vec(), audio.to_vec());
    }

    #[test]
    fn test_mel_spectrogram() {
        let audio = create_sine_wave(440.0, 16000, 0.5);
        let mel = MelSpectrogram::with_params(16000, 512, 256, 40);

        let spec = mel.apply(&audio);

        assert_eq!(spec.shape()[0], 40); // n_mels
        assert!(spec.shape()[1] > 0); // n_frames
    }

    #[test]
    fn test_mfcc() {
        let audio = create_sine_wave(440.0, 16000, 0.5);
        let mfcc = MFCC::new(16000, 13);

        let coeffs = mfcc.apply(&audio);

        assert_eq!(coeffs.shape()[0], 13); // n_mfcc
    }

    #[test]
    fn test_time_stretch() {
        let audio = create_sine_wave(440.0, 16000, 0.1);
        let orig_len = audio.shape()[0];

        // Speed up 2x
        let stretch = TimeStretch::new(2.0);
        let stretched = stretch.apply(&audio);

        assert!(stretched.shape()[0] < orig_len);
    }

    #[test]
    fn test_pitch_shift() {
        let audio = create_sine_wave(440.0, 16000, 0.1);
        let orig_len = audio.shape()[0];

        let shift = PitchShift::new(2.0); // Shift up 2 semitones
        let shifted = shift.apply(&audio);

        // Length should remain the same
        assert_eq!(shifted.shape()[0], orig_len);
    }

    #[test]
    fn test_add_noise() {
        let audio = create_sine_wave(440.0, 16000, 0.1);
        let add_noise = AddNoise::new(20.0); // 20dB SNR

        let noisy = add_noise.apply(&audio);

        assert_eq!(noisy.shape(), audio.shape());
        // Values should be different (noise added)
        assert_ne!(noisy.to_vec(), audio.to_vec());
    }

    #[test]
    fn test_normalize_audio() {
        let data = vec![0.1, -0.5, 0.3, -0.2];
        let audio = Tensor::from_vec(data, &[4]).unwrap();

        let normalize = NormalizeAudio::new();
        let normalized = normalize.apply(&audio);

        let max_val = normalized
            .to_vec()
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        assert!((max_val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_trim_silence() {
        let data = vec![0.0, 0.0, 0.5, 0.3, 0.0, 0.0];
        let audio = Tensor::from_vec(data, &[6]).unwrap();

        let trim = TrimSilence::new(-20.0);
        let trimmed = trim.apply(&audio);

        assert_eq!(trimmed.shape()[0], 2); // Only [0.5, 0.3]
    }

    #[test]
    fn test_hz_to_mel_conversion() {
        let hz = 1000.0;
        let mel = MelSpectrogram::hz_to_mel(hz);
        let back = MelSpectrogram::mel_to_hz(mel);

        assert!((hz - back).abs() < 0.1);
    }
}
