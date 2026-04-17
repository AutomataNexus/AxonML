//! Echo — Voice Identity via Predictive Speaker Residuals (~68K params)
//!
//! Implements `EchoSpeaker`, a speaker recognition model that trains a speech
//! predictor (Conv1d + frame-wise GRU + Conv1d decoder) to learn generic temporal
//! speech structure, then extracts prediction residuals as the speaker-specific
//! signal. Residuals are encoded through two Conv1d layers with statistics pooling
//! (mean + std) and projected to an L2-normalized embedding with uncertainty.
//! Provides prediction error as a novelty/impostor score, spectral-flatness-based
//! replay/spoofing detection, energy-based voice activity detection with adaptive
//! thresholding, temporal consistency measurement across overlapping segments, and
//! syllable-rate estimation via envelope autocorrelation.
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/echo.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::{Conv1d, GRUCell, Linear, Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// EchoSpeaker
// =============================================================================

/// Voice identity via predictive speaker residuals.
///
/// # Usage
///
/// ```ignore
/// use axonml_vision::models::biometric::EchoSpeaker;
/// use axonml_autograd::Variable;
/// use axonml_tensor::Tensor;
///
/// let model = EchoSpeaker::new();
///
/// // Mel spectrogram input [B, 40, T]
/// let mel = Variable::new(Tensor::zeros(&[1, 40, 100]), false);
///
/// // Full forward: prediction + identity
/// let (predicted_mel, speaker_embedding, log_variance) = model.forward_full(&mel);
///
/// // Prediction error as novelty score (low for enrolled speakers)
/// let error = model.prediction_error(&mel);
///
/// // Replay/spoofing detection
/// let spoof_score = model.detect_replay(&mel);
///
/// // Voice activity detection
/// let vad_mask = model.voice_activity(&mel);
///
/// // Temporal consistency of speaker identity
/// let consistency = model.temporal_consistency(&mel);
///
/// // Speaking rate estimation
/// let rate = model.speaking_rate(&mel);
/// ```
pub struct EchoSpeaker {
    // Speech predictor (learns generic speech temporal structure)
    pred_conv_in: Conv1d,
    pred_gru: GRUCell,
    pred_conv_out: Conv1d,

    // Residual encoder (extracts speaker-specific signal from prediction errors)
    res_conv1: Conv1d,
    res_conv2: Conv1d,

    // Speaker embedding head
    speaker_head: Linear,

    // Uncertainty estimation head
    uncertainty_head: Linear,

    /// Number of mel frequency bins
    n_mels: usize,
    /// Hidden dimension for predictor GRU
    pred_hidden: usize,
    /// Speaker embedding dimension
    embed_dim: usize,
}

impl Default for EchoSpeaker {
    fn default() -> Self {
        Self::new()
    }
}

impl EchoSpeaker {
    /// Create a new Echo speaker model with default configuration.
    ///
    /// Default: n_mels=40, pred_hidden=64, embed_dim=64
    pub fn new() -> Self {
        Self::with_config(40, 64, 64)
    }

    /// Create with custom configuration.
    ///
    /// # Arguments
    /// * `n_mels` - Number of mel frequency bins (matches mel spectrogram)
    /// * `pred_hidden` - Hidden dimension for the predictive GRU
    /// * `embed_dim` - Dimension of the final speaker embedding
    pub fn with_config(n_mels: usize, pred_hidden: usize, embed_dim: usize) -> Self {
        let pred_conv_in = Conv1d::with_options(n_mels, pred_hidden, 5, 1, 2, true);
        let pred_gru = GRUCell::new(pred_hidden, pred_hidden);
        let pred_conv_out = Conv1d::with_options(pred_hidden, n_mels, 3, 1, 1, true);

        let res_conv1 = Conv1d::with_options(n_mels, 48, 3, 1, 1, true);
        let res_conv2 = Conv1d::with_options(48, pred_hidden, 3, 2, 1, true);

        // mean + std pooling → 2 * pred_hidden = 128
        let pool_dim = pred_hidden * 2;
        let speaker_head = Linear::new(pool_dim, embed_dim);
        let uncertainty_head = Linear::new(pool_dim, 1);

        Self {
            pred_conv_in,
            pred_gru,
            pred_conv_out,
            res_conv1,
            res_conv2,
            speaker_head,
            uncertainty_head,
            n_mels,
            pred_hidden,
            embed_dim,
        }
    }

    /// Predict next mel frames and compute residuals.
    ///
    /// The speech predictor learns what's predictable about speech. The
    /// prediction residuals — what's NOT predictable — carry the speaker
    /// identity signal.
    ///
    /// Input: mel spectrogram [B, n_mels, T]
    /// Returns: (predicted_mel [B, n_mels, T], residuals [B, n_mels, T])
    pub fn predict_and_residual(&self, mel: &Variable) -> (Variable, Variable) {
        let shape = mel.shape();
        let (batch, _mels, time) = (shape[0], shape[1], shape[2]);

        // Encode mel through conv
        let encoded = self.pred_conv_in.forward(mel).relu();

        // Run GRU frame by frame for autoregressive prediction
        // Zero-init hidden state (requires_grad=false is correct for initial state)
        let mut h = Variable::new(
            Tensor::from_vec(
                vec![0.0f32; batch * self.pred_hidden],
                &[batch, self.pred_hidden],
            )
            .unwrap(),
            false,
        );

        let mut gru_outputs: Vec<Variable> = Vec::with_capacity(time);

        for t in 0..time {
            // Extract frame [B, pred_hidden] at time t via narrow (graph-tracked)
            let frame = encoded.narrow(2, t, 1).reshape(&[batch, self.pred_hidden]);

            let new_h = self.pred_gru.forward_step(&frame, &h);
            // Unsqueeze to [B, pred_hidden, 1] for later concatenation
            gru_outputs.push(new_h.reshape(&[batch, self.pred_hidden, 1]));
            h = new_h;
        }

        // Concatenate all GRU outputs along time dim: [B, pred_hidden, T]
        let gru_refs: Vec<&Variable> = gru_outputs.iter().collect();
        let gru_var = Variable::cat(&gru_refs, 2);
        let predicted = self.pred_conv_out.forward(&gru_var);

        // Compute residuals using sub_var: actual - predicted (graph-tracked)
        let residuals = mel.sub_var(&predicted);

        (predicted, residuals)
    }

    /// Encode prediction residuals into speaker embedding.
    ///
    /// Processes the prediction errors through a convolutional encoder,
    /// applies statistics pooling (mean + std), and projects to a unit
    /// speaker embedding.
    ///
    /// Input: residuals [B, n_mels, T]
    /// Returns: (L2-normalized embedding [B, embed_dim], log_variance [B, 1])
    pub fn encode_residuals(&self, residuals: &Variable) -> (Variable, Variable) {
        // Conv encode residuals
        let x = self.res_conv1.forward(residuals).relu();
        let x = self.res_conv2.forward(&x).relu();
        // x: [B, pred_hidden, T/2]

        // Statistics pooling: mean + std across time (graph-tracked)
        // mean_dim and var_dim backward are now fixed
        let x_mean = x.mean_dim(2, false); // [B, C]
        let x_std = x.var_dim(2, false).add_scalar(1e-8).sqrt(); // [B, C]
        let pooled_var = Variable::cat(&[&x_mean, &x_std], 1); // [B, 2*C]

        // Speaker embedding
        let embedding = self.speaker_head.forward(&pooled_var);

        // L2 normalize via scalar division (graph-tracked on embedding)
        let emb_data = embedding.data().to_vec();
        let norm_val: f32 = emb_data.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let norm_embedding = embedding.mul_scalar(1.0 / norm_val);

        // Uncertainty
        let uncertainty = self.uncertainty_head.forward(&pooled_var);

        (norm_embedding, uncertainty)
    }

    /// Full forward: mel → (predicted_mel, speaker_embedding, log_variance).
    ///
    /// Runs the complete pipeline: speech prediction, residual extraction,
    /// residual encoding, and speaker embedding extraction.
    pub fn forward_full(&self, mel: &Variable) -> (Variable, Variable, Variable) {
        let (predicted, residuals) = self.predict_and_residual(mel);
        let (embedding, logvar) = self.encode_residuals(&residuals);
        (predicted, embedding, logvar)
    }

    /// Extract speaker identity embedding from mel spectrogram.
    ///
    /// Convenience method that returns just the L2-normalized embedding
    /// as a Vec<f32>.
    pub fn extract_identity(&self, mel: &Variable) -> Vec<f32> {
        let (_pred, embedding, _logvar) = self.forward_full(mel);
        embedding.data().to_vec()
    }

    /// Compute prediction error magnitude (surprise signal).
    ///
    /// For enrolled speakers: prediction errors are LOW (the model has learned
    /// their speech patterns). For unknown voices: errors are HIGH.
    /// This serves as a novelty/impostor detector.
    ///
    /// Returns: mean squared prediction error (scalar)
    pub fn prediction_error(&self, mel: &Variable) -> f32 {
        let (_predicted, residuals) = self.predict_and_residual(mel);
        let res_data = residuals.data().to_vec();
        let n = res_data.len() as f32;
        res_data.iter().map(|r| r * r).sum::<f32>() / n
    }

    /// Detect replay/spoofing attacks via spectral flatness analysis.
    ///
    /// Live speech has variable spectral flatness across time because natural
    /// vocalization produces harmonics with uneven energy distribution that
    /// shifts dynamically. Replayed or synthesized audio passes through
    /// loudspeakers and recording channels that compress the spectral dynamic
    /// range, producing more uniform (flatter) spectral profiles.
    ///
    /// The method computes per-frame spectral flatness (ratio of geometric
    /// mean to arithmetic mean of mel bin magnitudes), then measures how
    /// uniform that flatness is across time. Live speech shows high variance
    /// in per-frame flatness; spoofed audio shows low variance.
    ///
    /// Returns: spoofing score in [0, 1] where 1 = likely spoofed.
    pub fn detect_replay(&self, mel: &Variable) -> f32 {
        let shape = mel.shape();
        let (batch, n_mels, time) = (shape[0], shape[1], shape[2]);
        let mel_data = mel.data().to_vec();

        // We analyze the first sample in the batch (batch index 0).
        // For batch processing, callers should iterate.
        let _ = batch; // single-sample path: always index the first element
        let b = 0usize;

        if time < 2 || n_mels < 2 {
            return 0.5; // Insufficient data for reliable detection
        }

        // Compute per-frame spectral flatness
        let mut flatness_values = Vec::with_capacity(time);

        for t in 0..time {
            // Collect mel bin magnitudes for this frame (use absolute values
            // since mel values can be log-scaled and negative)
            let mut log_sum = 0.0f64;
            let mut arith_sum = 0.0f64;
            let mut valid_bins = 0usize;

            for m in 0..n_mels {
                let idx = b * n_mels * time + m * time + t;
                let val = (mel_data[idx].abs() + 1e-10) as f64;
                log_sum += val.ln();
                arith_sum += val;
                valid_bins += 1;
            }

            if valid_bins > 0 {
                let n = valid_bins as f64;
                let geometric_mean = (log_sum / n).exp();
                let arithmetic_mean = arith_sum / n;
                // Spectral flatness: ratio in [0, 1], 1 = perfectly flat (white noise)
                let flatness = (geometric_mean / arithmetic_mean.max(1e-10)).min(1.0);
                flatness_values.push(flatness as f32);
            }
        }

        if flatness_values.len() < 2 {
            return 0.5;
        }

        // Compute variance of per-frame spectral flatness
        let n = flatness_values.len() as f32;
        let mean_flatness: f32 = flatness_values.iter().sum::<f32>() / n;
        let var_flatness: f32 = flatness_values
            .iter()
            .map(|f| (f - mean_flatness) * (f - mean_flatness))
            .sum::<f32>()
            / n;

        // Live speech: high variance in flatness (voiced/unvoiced transitions,
        // consonants vs vowels). Spoofed: low variance (compressed dynamics).
        //
        // Map variance to spoofing score using a sigmoid-like transform.
        // Empirically, live speech flatness variance > 0.01; spoofed < 0.005.
        // We use a soft threshold centered at 0.005.
        let sensitivity = 400.0f32; // Controls steepness of the decision boundary
        let threshold = 0.005f32;
        let score = 1.0 / (1.0 + (sensitivity * (var_flatness - threshold)).exp());

        score.clamp(0.0, 1.0)
    }

    /// Energy-based voice activity detection.
    ///
    /// Computes per-frame energy (sum of squared mel bin values) and applies
    /// an adaptive threshold to distinguish speech frames from silence.
    /// The threshold is set relative to the energy distribution of the
    /// utterance itself, making it robust to varying recording levels.
    ///
    /// Returns: per-frame boolean mask where `true` = speech activity detected.
    /// The length equals the number of time frames T in the input mel [B, n_mels, T].
    pub fn voice_activity(&self, mel: &Variable) -> Vec<bool> {
        let shape = mel.shape();
        let (batch, n_mels, time) = (shape[0], shape[1], shape[2]);
        let mel_data = mel.data().to_vec();

        let _ = batch; // single-sample path: always index the first element
        let b = 0usize;

        if time == 0 {
            return Vec::new();
        }

        // Compute per-frame energy
        let mut energies = Vec::with_capacity(time);
        for t in 0..time {
            let mut energy = 0.0f32;
            for m in 0..n_mels {
                let idx = b * n_mels * time + m * time + t;
                let val = mel_data[idx];
                energy += val * val;
            }
            energies.push(energy);
        }

        // Adaptive threshold: use a percentile-based approach.
        // Sort energies to find the distribution, then set threshold at
        // the 25th percentile plus a fraction of the dynamic range.
        // This adapts to both quiet and loud recordings.
        let mut sorted_energies = energies.clone();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p25_idx = (sorted_energies.len() as f32 * 0.25) as usize;
        let p75_idx =
            ((sorted_energies.len() as f32 * 0.75) as usize).min(sorted_energies.len() - 1);

        let p25 = sorted_energies[p25_idx];
        let p75 = sorted_energies[p75_idx];

        // Threshold = 25th percentile + 30% of the interquartile range
        // This biases toward detecting speech (lower threshold)
        let iqr = p75 - p25;
        let threshold = p25 + 0.30 * iqr;

        energies.iter().map(|&e| e > threshold).collect()
    }

    /// Measure temporal consistency of the speaker embedding across segments.
    ///
    /// Splits the mel spectrogram into overlapping segments, computes the
    /// speaker embedding for each segment independently, and measures the
    /// pairwise cosine similarity between all segment embeddings.
    ///
    /// High consistency (close to 1.0) indicates a reliable, single-speaker
    /// utterance. Low consistency suggests noise, multiple speakers, or
    /// unreliable audio that should not be used for enrollment/verification.
    ///
    /// Returns: mean pairwise cosine similarity in [-1, 1], where values
    /// close to 1.0 indicate high temporal consistency.
    pub fn temporal_consistency(&self, mel: &Variable) -> f32 {
        let shape = mel.shape();
        let (_batch, n_mels, time) = (shape[0], shape[1], shape[2]);
        let mel_data = mel.data().to_vec();

        // Minimum segment size: need enough frames for the conv + GRU pipeline
        let min_segment = 4;
        // Target ~4 segments with 50% overlap
        let segment_len = (time / 3).max(min_segment);

        if time < min_segment * 2 {
            // Too short to split — return neutral consistency
            return 1.0;
        }

        // Compute hop size for overlap; ensure at least 2 segments
        let hop = (segment_len / 2).max(1);
        let mut segments_start = Vec::new();
        let mut start = 0;
        while start + segment_len <= time {
            segments_start.push(start);
            start += hop;
        }
        // If we only got one segment, add a second one at the end
        if segments_start.len() < 2 && time >= min_segment {
            let last_start = time - segment_len;
            if segments_start.is_empty() || *segments_start.last().unwrap() != last_start {
                segments_start.push(last_start);
            }
        }

        if segments_start.len() < 2 {
            return 1.0;
        }

        // Extract embeddings for each segment
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        for &seg_start in &segments_start {
            let seg_end = (seg_start + segment_len).min(time);
            let seg_time = seg_end - seg_start;

            // Extract segment data [1, n_mels, seg_time]
            let mut seg_data = vec![0.0f32; n_mels * seg_time];
            for m in 0..n_mels {
                for t in 0..seg_time {
                    // Source: batch 0, mel m, time (seg_start + t)
                    let src_idx = 0 * n_mels * time + m * time + (seg_start + t);
                    let dst_idx = m * seg_time + t;
                    seg_data[dst_idx] = mel_data[src_idx];
                }
            }

            let seg_mel = Variable::new(
                Tensor::from_vec(seg_data, &[1, n_mels, seg_time]).unwrap(),
                false,
            );
            let emb = self.extract_identity(&seg_mel);
            embeddings.push(emb);
        }

        // Compute mean pairwise cosine similarity
        let n_emb = embeddings.len();
        let mut total_sim = 0.0f32;
        let mut n_pairs = 0;

        for i in 0..n_emb {
            for j in (i + 1)..n_emb {
                let sim = cosine_similarity_slice(&embeddings[i], &embeddings[j]);
                total_sim += sim;
                n_pairs += 1;
            }
        }

        if n_pairs == 0 {
            return 1.0;
        }

        total_sim / n_pairs as f32
    }

    /// Estimate speaking rate from mel energy envelope modulation.
    ///
    /// Speech exhibits quasi-periodic energy modulation at the syllable rate
    /// (typically 3-8 syllables/second for English). This method computes the
    /// dominant modulation frequency of the broadband energy envelope by
    /// finding peaks in the envelope autocorrelation.
    ///
    /// Returns: estimated syllable rate in syllables per second (Hz).
    /// Assumes standard 10ms frame shift (100 frames/second). Returns 0.0
    /// if the input is too short or has no detectable modulation.
    pub fn speaking_rate(&self, mel: &Variable) -> f32 {
        let shape = mel.shape();
        let (batch, n_mels, time) = (shape[0], shape[1], shape[2]);
        let mel_data = mel.data().to_vec();

        let _ = batch; // single-sample path: always index the first element
        let b = 0usize;

        // Standard assumption: 10ms frame shift → 100 frames/second
        let frames_per_second = 100.0f32;

        // Need at least ~0.5 seconds (50 frames) for reliable rate estimation
        if time < 10 {
            return 0.0;
        }

        // Compute broadband energy envelope: sum of squared mel values per frame
        let mut envelope = Vec::with_capacity(time);
        for t in 0..time {
            let mut energy = 0.0f32;
            for m in 0..n_mels {
                let idx = b * n_mels * time + m * time + t;
                let val = mel_data[idx];
                energy += val * val;
            }
            envelope.push(energy.sqrt()); // Use amplitude envelope
        }

        // Remove DC component (subtract mean)
        let mean_env: f32 = envelope.iter().sum::<f32>() / envelope.len() as f32;
        for v in envelope.iter_mut() {
            *v -= mean_env;
        }

        // Compute normalized autocorrelation for lags corresponding to
        // syllable rates between 2 Hz and 10 Hz
        let min_lag = (frames_per_second / 10.0) as usize; // ~10 frames (10 Hz)
        let max_lag = (frames_per_second / 2.0) as usize; // ~50 frames (2 Hz)
        let max_lag = max_lag.min(time / 2);

        if min_lag >= max_lag || max_lag >= time {
            return 0.0;
        }

        // Autocorrelation at lag 0 for normalization
        let acf_0: f32 = envelope.iter().map(|v| v * v).sum::<f32>();
        if acf_0 < 1e-10 {
            return 0.0; // Silent signal
        }

        // Find lag with maximum autocorrelation in the syllable-rate range
        let mut best_lag = min_lag;
        let mut best_acf = f32::NEG_INFINITY;

        for lag in min_lag..=max_lag {
            let mut acf = 0.0f32;
            for t in 0..(time - lag) {
                acf += envelope[t] * envelope[t + lag];
            }
            let acf_norm = acf / acf_0;

            if acf_norm > best_acf {
                best_acf = acf_norm;
                best_lag = lag;
            }
        }

        // Convert lag to frequency (syllable rate)
        if best_lag == 0 || best_acf < 0.0 {
            return 0.0;
        }

        frames_per_second / best_lag as f32
    }

    /// Collect all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.pred_conv_in.parameters());
        p.extend(self.pred_gru.parameters());
        p.extend(self.pred_conv_out.parameters());
        p.extend(self.res_conv1.parameters());
        p.extend(self.res_conv2.parameters());
        p.extend(self.speaker_head.parameters());
        p.extend(self.uncertainty_head.parameters());
        p
    }

    /// Get the speaker embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get the number of mel frequency bins.
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }
}

impl Module for EchoSpeaker {
    /// Forward pass: mel → speaker embedding [B, embed_dim].
    ///
    /// For the full prediction + embedding output, use [`forward_full`].
    fn forward(&self, input: &Variable) -> Variable {
        let (_pred, embedding, _logvar) = self.forward_full(input);
        embedding
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.parameters()
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Cosine similarity between two f32 slices.
fn cosine_similarity_slice(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
    dot / denom
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Existing tests (preserved from original)
    // -------------------------------------------------------------------------

    #[test]
    fn test_echo_creation() {
        let model = EchoSpeaker::new();
        assert_eq!(model.embed_dim(), 64);
        assert_eq!(model.n_mels(), 40);
    }

    #[test]
    fn test_echo_param_count() {
        let model = EchoSpeaker::new();
        let total: usize = model
            .parameters()
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        assert!(total < 100_000, "Params {} exceeds 100K budget", total);
        assert!(total > 30_000, "Params {} seems too low", total);
    }

    #[test]
    fn test_echo_forward_shape() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 50], &[1, 40, 50]).unwrap(),
            false,
        );
        let output = model.forward(&mel);
        assert_eq!(output.shape(), &[1, 64]);

        // Output should have non-zero values
        let data = output.data().to_vec();
        let nonzero = data.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(nonzero > 0, "All outputs are zero — dead network");
    }

    #[test]
    fn test_echo_full_forward() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 50], &[1, 40, 50]).unwrap(),
            false,
        );
        let (predicted, embedding, logvar) = model.forward_full(&mel);
        assert_eq!(predicted.shape(), &[1, 40, 50]);
        assert_eq!(embedding.shape(), &[1, 64]);
        assert_eq!(logvar.shape(), &[1, 1]);
    }

    #[test]
    fn test_echo_embedding_normalized() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.2f32; 40 * 30], &[1, 40, 30]).unwrap(),
            false,
        );
        let identity = model.extract_identity(&mel);
        let norm: f32 = identity.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding not unit norm: {}",
            norm
        );
    }

    #[test]
    fn test_echo_prediction_error_nonneg() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 50], &[1, 40, 50]).unwrap(),
            false,
        );
        let error = model.prediction_error(&mel);
        assert!(
            error >= 0.0,
            "Prediction error should be non-negative: {}",
            error
        );
        assert!(
            error.is_finite(),
            "Prediction error should be finite: {}",
            error
        );
    }

    #[test]
    fn test_echo_residuals_shape() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 50], &[1, 40, 50]).unwrap(),
            false,
        );
        let (predicted, residuals) = model.predict_and_residual(&mel);
        assert_eq!(predicted.shape(), mel.shape());
        assert_eq!(residuals.shape(), mel.shape());

        // Residuals = mel - predicted, so predicted + residuals ≈ mel
        let mel_data = mel.data().to_vec();
        let pred_data = predicted.data().to_vec();
        let res_data = residuals.data().to_vec();
        for i in 0..mel_data.len() {
            let reconstructed = pred_data[i] + res_data[i];
            assert!(
                (reconstructed - mel_data[i]).abs() < 1e-4,
                "Residual reconstruction error at {}: {} vs {}",
                i,
                reconstructed,
                mel_data[i]
            );
        }
    }

    #[test]
    fn test_echo_variable_length_input() {
        let model = EchoSpeaker::new();

        // Short utterance
        let mel_short = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 20], &[1, 40, 20]).unwrap(),
            false,
        );
        let out_short = model.forward(&mel_short);
        assert_eq!(out_short.shape(), &[1, 64]);

        // Long utterance
        let mel_long = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 200], &[1, 40, 200]).unwrap(),
            false,
        );
        let out_long = model.forward(&mel_long);
        assert_eq!(out_long.shape(), &[1, 64]);
    }

    // -------------------------------------------------------------------------
    // Replay/Spoofing Detection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_replay_detection_output_range() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 100], &[1, 40, 100]).unwrap(),
            false,
        );
        let score = model.detect_replay(&mel);
        assert!(
            (0.0..=1.0).contains(&score),
            "Spoofing score should be in [0,1], got {}",
            score
        );
        assert!(score.is_finite(), "Spoofing score should be finite");
    }

    #[test]
    fn test_replay_detection_uniform_spectral_flatness() {
        // Uniform input (same value across all mel bins and time frames) should
        // have very low spectral flatness variance — characteristic of spoofed audio.
        let model = EchoSpeaker::new();
        let mel_uniform = Variable::new(
            Tensor::from_vec(vec![0.5f32; 40 * 100], &[1, 40, 100]).unwrap(),
            false,
        );
        let score_uniform = model.detect_replay(&mel_uniform);

        // Create varied input (different patterns per frame to simulate speech)
        let mut varied_data = vec![0.0f32; 40 * 100];
        for t in 0..100 {
            for m in 0..40 {
                // Simulate voiced/unvoiced variation: some frames have harmonic
                // structure (low flatness), others noise-like (high flatness)
                let idx = m * 100 + t;
                if t % 5 < 3 {
                    // Voiced frame: energy concentrated in low frequencies
                    varied_data[idx] = if m < 10 {
                        1.0 + 0.3 * ((t as f32 * 0.1).sin())
                    } else {
                        0.01
                    };
                } else {
                    // Unvoiced frame: flatter spectrum
                    varied_data[idx] = 0.2 + 0.1 * ((m as f32 * 0.5).sin());
                }
            }
        }
        let mel_varied =
            Variable::new(Tensor::from_vec(varied_data, &[1, 40, 100]).unwrap(), false);
        let score_varied = model.detect_replay(&mel_varied);

        // The uniform input should have a HIGHER spoofing score (more likely spoofed)
        // than the varied input (more like real speech)
        assert!(
            score_uniform > score_varied,
            "Uniform mel (score={}) should be more spoofed than varied mel (score={})",
            score_uniform,
            score_varied
        );
    }

    #[test]
    fn test_replay_detection_finite_short_input() {
        let model = EchoSpeaker::new();
        // Very short input — should still produce a valid score
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 4], &[1, 40, 4]).unwrap(),
            false,
        );
        let score = model.detect_replay(&mel);
        assert!(
            score.is_finite(),
            "Replay score should be finite for short input"
        );
        assert!((0.0..=1.0).contains(&score));
    }

    // -------------------------------------------------------------------------
    // Voice Activity Detection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vad_output_length() {
        let model = EchoSpeaker::new();
        let time = 80;
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * time], &[1, 40, time]).unwrap(),
            false,
        );
        let mask = model.voice_activity(&mel);
        assert_eq!(
            mask.len(),
            time,
            "VAD mask length should equal number of time frames"
        );
    }

    #[test]
    fn test_vad_silence_vs_speech() {
        let model = EchoSpeaker::new();

        // Create input with clear silence (near-zero) and speech (high energy) regions
        let time = 100;
        let n_mels = 40;
        let mut data = vec![0.0f32; n_mels * time];

        // Frames 0-29: silence (very low energy)
        for t in 0..30 {
            for m in 0..n_mels {
                data[m * time + t] = 0.001;
            }
        }
        // Frames 30-69: speech (high energy)
        for t in 30..70 {
            for m in 0..n_mels {
                data[m * time + t] = 1.0 + 0.5 * ((m as f32 * 0.3).sin());
            }
        }
        // Frames 70-99: silence again
        for t in 70..100 {
            for m in 0..n_mels {
                data[m * time + t] = 0.001;
            }
        }

        let mel = Variable::new(Tensor::from_vec(data, &[1, n_mels, time]).unwrap(), false);
        let mask = model.voice_activity(&mel);

        // Speech region (30-69) should mostly be active
        let speech_active: usize = mask[30..70].iter().filter(|&&v| v).count();
        assert!(
            speech_active > 30,
            "At least 75% of speech frames should be active, got {}/40",
            speech_active
        );

        // Silence regions should mostly be inactive
        let silence_active: usize = mask[0..30].iter().filter(|&&v| v).count()
            + mask[70..100].iter().filter(|&&v| v).count();
        assert!(
            silence_active < 20,
            "Most silence frames should be inactive, got {}/60 active",
            silence_active
        );
    }

    #[test]
    fn test_vad_all_silence() {
        let model = EchoSpeaker::new();
        // Near-zero energy everywhere — threshold adapts, but with constant
        // energy the IQR is 0 so threshold equals p25
        let mel = Variable::new(
            Tensor::from_vec(vec![0.0001f32; 40 * 50], &[1, 40, 50]).unwrap(),
            false,
        );
        let mask = model.voice_activity(&mel);
        assert_eq!(mask.len(), 50);
        // With constant energy, all frames have identical energy;
        // none should be above the threshold (they equal it exactly)
    }

    #[test]
    fn test_vad_single_frame() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.5f32; 40], &[1, 40, 1]).unwrap(),
            false,
        );
        let mask = model.voice_activity(&mel);
        assert_eq!(mask.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Temporal Consistency tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_temporal_consistency_output_range() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 100], &[1, 40, 100]).unwrap(),
            false,
        );
        let consistency = model.temporal_consistency(&mel);
        assert!(
            consistency.is_finite(),
            "Temporal consistency should be finite, got {}",
            consistency
        );
        // Cosine similarity range is [-1, 1]
        assert!(
            (-1.01..=1.01).contains(&consistency),
            "Temporal consistency should be in [-1, 1], got {}",
            consistency
        );
    }

    #[test]
    fn test_temporal_consistency_same_speaker_segments() {
        // For a constant-valued mel, all segments produce similar embeddings,
        // so consistency should be high.
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.3f32; 40 * 120], &[1, 40, 120]).unwrap(),
            false,
        );
        let consistency = model.temporal_consistency(&mel);
        // Same input across all segments should yield high consistency
        assert!(
            consistency > 0.5,
            "Constant mel should have high temporal consistency, got {}",
            consistency
        );
    }

    #[test]
    fn test_temporal_consistency_short_input() {
        // Very short input — cannot split into multiple segments
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 4], &[1, 40, 4]).unwrap(),
            false,
        );
        let consistency = model.temporal_consistency(&mel);
        // Too short to meaningfully segment; should return 1.0 (neutral)
        assert!(
            (consistency - 1.0).abs() < 0.01,
            "Very short input should return ~1.0 consistency, got {}",
            consistency
        );
    }

    // -------------------------------------------------------------------------
    // Speaking Rate tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_speaking_rate_finite_nonneg() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 100], &[1, 40, 100]).unwrap(),
            false,
        );
        let rate = model.speaking_rate(&mel);
        assert!(
            rate >= 0.0,
            "Speaking rate should be non-negative, got {}",
            rate
        );
        assert!(
            rate.is_finite(),
            "Speaking rate should be finite, got {}",
            rate
        );
    }

    #[test]
    fn test_speaking_rate_short_input() {
        let model = EchoSpeaker::new();
        // Very short input — not enough for rate estimation
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 5], &[1, 40, 5]).unwrap(),
            false,
        );
        let rate = model.speaking_rate(&mel);
        // Should return 0 or a valid rate, but must be finite and non-negative
        assert!(
            rate >= 0.0 && rate.is_finite(),
            "Short input speaking rate should be finite and >= 0, got {}",
            rate
        );
    }

    #[test]
    fn test_speaking_rate_modulated_envelope() {
        // Create a mel spectrogram with a clear ~5 Hz energy modulation
        // (5 syllables/second, typical English speech rate)
        let model = EchoSpeaker::new();
        let time = 200; // 2 seconds at 100 fps
        let n_mels = 40;
        let mut data = vec![0.0f32; n_mels * time];

        // 5 Hz modulation: period = 20 frames at 100 fps
        for t in 0..time {
            let modulation =
                0.5 + 0.5 * (2.0 * std::f32::consts::PI * 5.0 * t as f32 / 100.0).sin();
            for m in 0..n_mels {
                data[m * time + t] = modulation * (0.5 + 0.1 * ((m as f32 * 0.2).sin()));
            }
        }

        let mel = Variable::new(Tensor::from_vec(data, &[1, n_mels, time]).unwrap(), false);
        let rate = model.speaking_rate(&mel);
        assert!(
            rate > 0.0,
            "Modulated signal should produce non-zero rate, got {}",
            rate
        );
        assert!(rate.is_finite(), "Rate should be finite");
        // Expect rate near 5 Hz (allow generous tolerance)
        assert!(
            rate > 2.0 && rate < 10.0,
            "Expected rate near 5 Hz for 5 Hz modulation, got {} Hz",
            rate
        );
    }

    #[test]
    fn test_speaking_rate_silence() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.0f32; 40 * 100], &[1, 40, 100]).unwrap(),
            false,
        );
        let rate = model.speaking_rate(&mel);
        assert_eq!(
            rate, 0.0,
            "Silent input should produce 0 speaking rate, got {}",
            rate
        );
    }

    // -------------------------------------------------------------------------
    // Batch processing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_echo_batch_forward() {
        let model = EchoSpeaker::new();
        // Batch of 3, same time length
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 3 * 40 * 50], &[3, 40, 50]).unwrap(),
            false,
        );
        let output = model.forward(&mel);
        assert_eq!(output.shape(), &[3, 64]);
    }

    #[test]
    fn test_echo_batch_full_forward() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.2f32; 2 * 40 * 60], &[2, 40, 60]).unwrap(),
            false,
        );
        let (predicted, embedding, logvar) = model.forward_full(&mel);
        assert_eq!(predicted.shape(), &[2, 40, 60]);
        assert_eq!(embedding.shape(), &[2, 64]);
        assert_eq!(logvar.shape(), &[2, 1]);
    }

    // -------------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_echo_very_short_input_4_frames() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 40 * 4], &[1, 40, 4]).unwrap(),
            false,
        );
        let output = model.forward(&mel);
        assert_eq!(output.shape(), &[1, 64]);
        // All values should be finite
        let data = output.data().to_vec();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "All output values should be finite for minimum-length input"
        );
    }

    #[test]
    fn test_echo_single_frame_input() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.5f32; 40], &[1, 40, 1]).unwrap(),
            false,
        );
        // Single frame should still produce valid output (conv padding handles it)
        let output = model.forward(&mel);
        assert_eq!(output.shape(), &[1, 64]);
        let data = output.data().to_vec();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "Single-frame output should be finite"
        );
    }

    // -------------------------------------------------------------------------
    // Numerical properties
    // -------------------------------------------------------------------------

    #[test]
    fn test_echo_outputs_all_finite() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.3f32; 40 * 80], &[1, 40, 80]).unwrap(),
            false,
        );
        let (predicted, embedding, logvar) = model.forward_full(&mel);

        let pred_data = predicted.data().to_vec();
        assert!(
            pred_data.iter().all(|v| v.is_finite()),
            "All predicted values should be finite"
        );

        let emb_data = embedding.data().to_vec();
        assert!(
            emb_data.iter().all(|v| v.is_finite()),
            "All embedding values should be finite"
        );

        let logvar_data = logvar.data().to_vec();
        assert!(
            logvar_data.iter().all(|v| v.is_finite()),
            "All logvar values should be finite"
        );
    }

    #[test]
    fn test_echo_embedding_unit_norm_various_inputs() {
        let model = EchoSpeaker::new();

        // Test normalization for several different inputs
        for val in [0.01f32, 0.1, 0.5, 1.0, 2.0] {
            let mel = Variable::new(
                Tensor::from_vec(vec![val; 40 * 40], &[1, 40, 40]).unwrap(),
                false,
            );
            let identity = model.extract_identity(&mel);
            let norm: f32 = identity.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.02,
                "Embedding should be unit norm for input val={}, got norm={}",
                val,
                norm
            );
        }
    }

    // -------------------------------------------------------------------------
    // Prediction quality: reconstruction identity
    // -------------------------------------------------------------------------

    #[test]
    fn test_prediction_residual_reconstruction_identity() {
        // Verify that predicted + residuals = original mel for various inputs
        let model = EchoSpeaker::new();

        let mut input_data = vec![0.0f32; 40 * 60];
        for i in 0..input_data.len() {
            input_data[i] = (i as f32 * 0.01).sin() * 0.5;
        }
        let mel = Variable::new(
            Tensor::from_vec(input_data.clone(), &[1, 40, 60]).unwrap(),
            false,
        );
        let (predicted, residuals) = model.predict_and_residual(&mel);

        let mel_data = mel.data().to_vec();
        let pred_data = predicted.data().to_vec();
        let res_data = residuals.data().to_vec();

        assert_eq!(pred_data.len(), mel_data.len());
        assert_eq!(res_data.len(), mel_data.len());

        let mut max_error = 0.0f32;
        for i in 0..mel_data.len() {
            let reconstructed = pred_data[i] + res_data[i];
            let error = (reconstructed - mel_data[i]).abs();
            max_error = max_error.max(error);
        }
        assert!(
            max_error < 1e-4,
            "Max reconstruction error should be < 1e-4, got {}",
            max_error
        );
    }

    #[test]
    fn test_prediction_error_finite_for_varied_input() {
        let model = EchoSpeaker::new();
        let mut data = vec![0.0f32; 40 * 50];
        for i in 0..data.len() {
            data[i] = ((i as f32) * 0.1).cos() * 0.8;
        }
        let mel = Variable::new(Tensor::from_vec(data, &[1, 40, 50]).unwrap(), false);
        let error = model.prediction_error(&mel);
        assert!(
            error.is_finite(),
            "Prediction error should be finite for varied input"
        );
        assert!(error >= 0.0, "Prediction error should be non-negative");
    }

    // -------------------------------------------------------------------------
    // Custom config test
    // -------------------------------------------------------------------------

    #[test]
    fn test_echo_custom_config() {
        let model = EchoSpeaker::with_config(32, 48, 32);
        assert_eq!(model.n_mels(), 32);
        assert_eq!(model.embed_dim(), 32);

        let mel = Variable::new(
            Tensor::from_vec(vec![0.1f32; 32 * 40], &[1, 32, 40]).unwrap(),
            false,
        );
        let output = model.forward(&mel);
        assert_eq!(output.shape(), &[1, 32]);
    }

    // -------------------------------------------------------------------------
    // Integration: replay + VAD + consistency + rate on same input
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_analysis_methods_on_same_input() {
        let model = EchoSpeaker::new();
        let mel = Variable::new(
            Tensor::from_vec(vec![0.2f32; 40 * 100], &[1, 40, 100]).unwrap(),
            false,
        );

        let spoof = model.detect_replay(&mel);
        let vad = model.voice_activity(&mel);
        let consistency = model.temporal_consistency(&mel);
        let rate = model.speaking_rate(&mel);
        let error = model.prediction_error(&mel);
        let identity = model.extract_identity(&mel);

        // All should produce valid outputs
        assert!((0.0..=1.0).contains(&spoof));
        assert_eq!(vad.len(), 100);
        assert!(consistency.is_finite());
        assert!(rate >= 0.0 && rate.is_finite());
        assert!(error >= 0.0 && error.is_finite());
        assert_eq!(identity.len(), 64);
        let norm: f32 = identity.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.02);
    }
}
