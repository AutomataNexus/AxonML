//! FFT Layers - Fast Fourier Transform for Neural Networks
//!
//! Provides FFT1d and STFT as neural network layers for spectral
//! feature extraction in signal processing and time series models.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// FFT1d
// =============================================================================

/// 1D Fast Fourier Transform layer.
///
/// Computes the magnitude spectrum of the input signal using an efficient
/// FFT algorithm. Returns the first `n_fft/2 + 1` frequency bins (real FFT).
///
/// # Input Shape
/// `(batch, channels, time)` or `(batch, time)`
///
/// # Output Shape
/// `(batch, channels, n_fft/2+1)` or `(batch, n_fft/2+1)`
///
/// # Example
/// ```ignore
/// use axonml_nn::layers::FFT1d;
///
/// let fft = FFT1d::new(256);
/// let signal = Variable::new(Tensor::randn(&[2, 1, 256]), true);
/// let spectrum = fft.forward(&signal);
/// // spectrum shape: (2, 1, 129)
/// ```
pub struct FFT1d {
    n_fft: usize,
    normalized: bool,
}

impl FFT1d {
    /// Creates a new FFT1d layer.
    ///
    /// # Arguments
    /// * `n_fft` - FFT size. Input will be zero-padded or truncated to this length.
    pub fn new(n_fft: usize) -> Self {
        Self {
            n_fft,
            normalized: false,
        }
    }

    /// Creates an FFT1d layer with optional normalization.
    ///
    /// When normalized, the output is divided by `sqrt(n_fft)`.
    pub fn with_normalization(n_fft: usize, normalized: bool) -> Self {
        Self { n_fft, normalized }
    }

    /// Returns the number of output frequency bins.
    pub fn output_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Compute FFT magnitude for a single 1D signal.
    fn fft_magnitude(&self, signal: &[f32]) -> Vec<f32> {
        let n = self.n_fft;
        let n_out = n / 2 + 1;

        // Prepare complex input (zero-pad or truncate)
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n];
        let copy_len = signal.len().min(n);
        for i in 0..copy_len {
            buffer[i] = Complex::new(signal[i], 0.0);
        }

        // Compute FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Extract magnitude for positive frequencies
        let norm_factor = if self.normalized {
            1.0 / (n as f32).sqrt()
        } else {
            1.0
        };

        let mut magnitude = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let mag = (buffer[i].re * buffer[i].re + buffer[i].im * buffer[i].im).sqrt();
            magnitude.push(mag * norm_factor);
        }

        magnitude
    }
}

impl Module for FFT1d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let data = input.data().to_vec();
        let n_out = self.output_bins();

        match shape.len() {
            2 => {
                // (batch, time) → (batch, n_fft/2+1)
                let batch = shape[0];
                let time = shape[1];
                let mut output = Vec::with_capacity(batch * n_out);

                for b in 0..batch {
                    let start = b * time;
                    let end = start + time;
                    let signal = &data[start..end];
                    output.extend_from_slice(&self.fft_magnitude(signal));
                }

                Variable::new(
                    Tensor::from_vec(output, &[batch, n_out]).unwrap(),
                    input.requires_grad(),
                )
            }
            3 => {
                // (batch, channels, time) → (batch, channels, n_fft/2+1)
                let batch = shape[0];
                let channels = shape[1];
                let time = shape[2];
                let mut output = Vec::with_capacity(batch * channels * n_out);

                for b in 0..batch {
                    for c in 0..channels {
                        let start = (b * channels + c) * time;
                        let end = start + time;
                        let signal = &data[start..end];
                        output.extend_from_slice(&self.fft_magnitude(signal));
                    }
                }

                Variable::new(
                    Tensor::from_vec(output, &[batch, channels, n_out]).unwrap(),
                    input.requires_grad(),
                )
            }
            _ => panic!(
                "FFT1d expects input of shape (batch, time) or (batch, channels, time), got {:?}",
                shape
            ),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new() // FFT has no learnable parameters
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn name(&self) -> &'static str {
        "FFT1d"
    }
}

// =============================================================================
// STFT
// =============================================================================

/// Short-Time Fourier Transform layer.
///
/// Applies a sliding window FFT to compute time-frequency representations.
/// Uses a Hann window by default.
///
/// # Input Shape
/// `(batch, channels, time)` or `(batch, time)`
///
/// # Output Shape
/// `(batch, channels, n_frames, n_fft/2+1)` or `(batch, n_frames, n_fft/2+1)`
///
/// # Example
/// ```ignore
/// use axonml_nn::layers::STFT;
///
/// let stft = STFT::new(256, 128); // n_fft=256, hop=128
/// let signal = Variable::new(Tensor::randn(&[2, 1, 1024]), false);
/// let spec = stft.forward(&signal);
/// // spec shape: (2, 1, 7, 129)
/// ```
pub struct STFT {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    normalized: bool,
}

impl STFT {
    /// Creates a new STFT layer with a Hann window.
    ///
    /// # Arguments
    /// * `n_fft` - FFT window size
    /// * `hop_length` - Number of samples between successive frames
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        let window = hann_window(n_fft);
        Self {
            n_fft,
            hop_length,
            window,
            normalized: false,
        }
    }

    /// Creates an STFT layer with normalization.
    pub fn with_normalization(n_fft: usize, hop_length: usize, normalized: bool) -> Self {
        let window = hann_window(n_fft);
        Self {
            n_fft,
            hop_length,
            window,
            normalized,
        }
    }

    /// Returns the number of output frequency bins.
    pub fn output_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Computes the number of frames for a given signal length.
    pub fn n_frames(&self, signal_length: usize) -> usize {
        if signal_length < self.n_fft {
            1
        } else {
            (signal_length - self.n_fft) / self.hop_length + 1
        }
    }

    /// Compute STFT magnitude for a single 1D signal.
    fn stft_magnitude(&self, signal: &[f32]) -> Vec<f32> {
        let n = self.n_fft;
        let n_out = n / 2 + 1;
        let n_frames = self.n_frames(signal.len());

        let norm_factor = if self.normalized {
            1.0 / (n as f32).sqrt()
        } else {
            1.0
        };

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut output = Vec::with_capacity(n_frames * n_out);

        for frame in 0..n_frames {
            let start = frame * self.hop_length;

            // Apply window and create complex buffer
            let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n];
            for i in 0..n {
                let idx = start + i;
                let sample = if idx < signal.len() { signal[idx] } else { 0.0 };
                buffer[i] = Complex::new(sample * self.window[i], 0.0);
            }

            fft.process(&mut buffer);

            for i in 0..n_out {
                let mag = (buffer[i].re * buffer[i].re + buffer[i].im * buffer[i].im).sqrt();
                output.push(mag * norm_factor);
            }
        }

        output
    }
}

impl Module for STFT {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let data = input.data().to_vec();
        let n_out = self.output_bins();

        match shape.len() {
            2 => {
                // (batch, time) → (batch, n_frames, n_fft/2+1)
                let batch = shape[0];
                let time = shape[1];
                let n_frames = self.n_frames(time);
                let mut output = Vec::with_capacity(batch * n_frames * n_out);

                for b in 0..batch {
                    let start = b * time;
                    let end = start + time;
                    let signal = &data[start..end];
                    output.extend_from_slice(&self.stft_magnitude(signal));
                }

                Variable::new(
                    Tensor::from_vec(output, &[batch, n_frames, n_out]).unwrap(),
                    input.requires_grad(),
                )
            }
            3 => {
                // (batch, channels, time) → (batch, channels, n_frames, n_fft/2+1)
                let batch = shape[0];
                let channels = shape[1];
                let time = shape[2];
                let n_frames = self.n_frames(time);
                let mut output = Vec::with_capacity(batch * channels * n_frames * n_out);

                for b in 0..batch {
                    for c in 0..channels {
                        let start = (b * channels + c) * time;
                        let end = start + time;
                        let signal = &data[start..end];
                        output.extend_from_slice(&self.stft_magnitude(signal));
                    }
                }

                Variable::new(
                    Tensor::from_vec(output, &[batch, channels, n_frames, n_out]).unwrap(),
                    input.requires_grad(),
                )
            }
            _ => panic!(
                "STFT expects input of shape (batch, time) or (batch, channels, time), got {:?}",
                shape
            ),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn name(&self) -> &'static str {
        "STFT"
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Generates a Hann window of the given size.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft1d_shape_2d() {
        let fft = FFT1d::new(64);
        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 128], &[2, 64]).unwrap(),
            false,
        );
        let output = fft.forward(&input);
        assert_eq!(output.shape(), vec![2, 33]); // n_fft/2+1 = 33
    }

    #[test]
    fn test_fft1d_shape_3d() {
        let fft = FFT1d::new(128);
        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 2 * 3 * 128], &[2, 3, 128]).unwrap(),
            false,
        );
        let output = fft.forward(&input);
        assert_eq!(output.shape(), vec![2, 3, 65]); // n_fft/2+1 = 65
    }

    #[test]
    fn test_fft1d_known_sinusoid() {
        // Create a pure 10 Hz sinusoid sampled at 64 Hz
        let n = 64;
        let freq = 10.0;
        let sample_rate = 64.0;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();

        let fft = FFT1d::new(n);
        let input = Variable::new(
            Tensor::from_vec(signal, &[1, n]).unwrap(),
            false,
        );
        let output = fft.forward(&input);
        let spectrum = output.data().to_vec();

        // The peak should be at bin 10 (freq * n / sample_rate = 10)
        let peak_bin = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_bin, 10);
    }

    #[test]
    fn test_fft1d_zero_padding() {
        // Input shorter than n_fft gets zero-padded
        let fft = FFT1d::new(128);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 32], &[1, 32]).unwrap(),
            false,
        );
        let output = fft.forward(&input);
        assert_eq!(output.shape(), vec![1, 65]);
    }

    #[test]
    fn test_fft1d_normalized() {
        let fft_norm = FFT1d::with_normalization(64, true);
        let fft_raw = FFT1d::new(64);

        let signal = vec![1.0; 64];
        let input = Variable::new(
            Tensor::from_vec(signal, &[1, 64]).unwrap(),
            false,
        );

        let out_norm = fft_norm.forward(&input).data().to_vec();
        let out_raw = fft_raw.forward(&input).data().to_vec();

        // Normalized should be raw / sqrt(64) = raw / 8
        let ratio = out_raw[0] / out_norm[0];
        assert!((ratio - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_stft_shape() {
        let stft = STFT::new(256, 128);
        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 2 * 1024], &[2, 1024]).unwrap(),
            false,
        );
        let output = stft.forward(&input);

        let n_frames = stft.n_frames(1024); // (1024 - 256) / 128 + 1 = 7
        assert_eq!(output.shape(), vec![2, n_frames, 129]);
        assert_eq!(n_frames, 7);
    }

    #[test]
    fn test_stft_shape_3d() {
        let stft = STFT::new(64, 32);
        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 2 * 3 * 256], &[2, 3, 256]).unwrap(),
            false,
        );
        let output = stft.forward(&input);

        let n_frames = stft.n_frames(256); // (256 - 64) / 32 + 1 = 7
        assert_eq!(output.shape(), vec![2, 3, n_frames, 33]);
    }

    #[test]
    fn test_stft_no_parameters() {
        let stft = STFT::new(256, 128);
        assert_eq!(stft.parameters().len(), 0);
    }

    #[test]
    fn test_fft1d_output_bins() {
        assert_eq!(FFT1d::new(64).output_bins(), 33);
        assert_eq!(FFT1d::new(256).output_bins(), 129);
        assert_eq!(FFT1d::new(512).output_bins(), 257);
    }

    #[test]
    fn test_hann_window() {
        let w = hann_window(4);
        // Hann window for size 4: [0, 0.75, 0.75, 0]
        assert!((w[0]).abs() < 1e-6);
        assert!((w[1] - 0.75).abs() < 0.01);
        assert!((w[2] - 0.75).abs() < 0.01);
        assert!((w[3]).abs() < 1e-6);
    }
}
