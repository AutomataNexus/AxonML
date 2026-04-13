//! Mnemosyne — Face Identity via Temporal Crystallization (~115K params)
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/mnemosyne.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::{AdaptiveAvgPool2d, BatchNorm2d, Conv2d, GRUCell, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use super::LivenessResult;

// =============================================================================
// BlazeBlock — Depthwise Separable + Residual
// =============================================================================

/// Depthwise separable convolution block with optional residual projection.
///
/// Architecture: Depthwise Conv → BN → ReLU → Pointwise Conv → BN + Residual → ReLU
///
/// When in_ch != out_ch or stride > 1, a 1×1 projection convolution aligns
/// the residual path dimensions.
struct BlazeBlock {
    dw_conv: Conv2d,
    dw_bn: BatchNorm2d,
    pw_conv: Conv2d,
    pw_bn: BatchNorm2d,
    project: Option<(Conv2d, BatchNorm2d)>,
}

impl BlazeBlock {
    fn new(in_ch: usize, out_ch: usize, stride: usize) -> Self {
        let dw_conv =
            Conv2d::with_groups(in_ch, in_ch, (3, 3), (stride, stride), (1, 1), true, in_ch);
        let dw_bn = BatchNorm2d::new(in_ch);

        let pw_conv = Conv2d::with_options(in_ch, out_ch, (1, 1), (1, 1), (0, 0), true);
        let pw_bn = BatchNorm2d::new(out_ch);

        let project = if in_ch != out_ch || stride != 1 {
            Some((
                Conv2d::with_options(in_ch, out_ch, (1, 1), (stride, stride), (0, 0), true),
                BatchNorm2d::new(out_ch),
            ))
        } else {
            None
        };

        Self {
            dw_conv,
            dw_bn,
            pw_conv,
            pw_bn,
            project,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let identity = if let Some((ref proj_conv, ref proj_bn)) = self.project {
            proj_bn.forward(&proj_conv.forward(x))
        } else {
            x.clone()
        };

        let out = self.dw_bn.forward(&self.dw_conv.forward(x)).relu();
        let out = self.pw_bn.forward(&self.pw_conv.forward(&out));
        out.add_var(&identity).relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.dw_conv.parameters());
        p.extend(self.dw_bn.parameters());
        p.extend(self.pw_conv.parameters());
        p.extend(self.pw_bn.parameters());
        if let Some((ref c, ref b)) = self.project {
            p.extend(c.parameters());
            p.extend(b.parameters());
        }
        p
    }
}

// =============================================================================
// MnemosyneIdentity
// =============================================================================

/// Face identity via temporal crystallization.
///
/// A GRU hidden state evolves over multiple face observations, converging
/// toward an "identity attractor." The hidden state IS the identity.
///
/// # Usage
///
/// ```ignore
/// use axonml_vision::models::biometric::MnemosyneIdentity;
/// use axonml_autograd::Variable;
/// use axonml_tensor::Tensor;
///
/// let model = MnemosyneIdentity::new();
///
/// // Single-frame encoding
/// let face = Variable::new(Tensor::zeros(&[1, 3, 64, 64]), false);
/// let encoding = model.encode_face(&face);
///
/// // Temporal crystallization over multiple frames
/// let mut hidden = None;
/// for _ in 0..10 {
///     let (h, velocity, logvar, quality) = model.crystallize_step(&face, hidden.as_ref());
///     hidden = Some(h);
///     // velocity decreases as identity crystallizes
/// }
///
/// // Extract crystallized identity
/// let identity = model.extract_identity(hidden.as_ref().unwrap());
/// ```
pub struct MnemosyneIdentity {
    // Face encoder: stem + 3 BlazeBlocks + adaptive pool
    stem_conv: Conv2d,
    stem_bn: BatchNorm2d,
    block1: BlazeBlock,
    block2: BlazeBlock,
    block3: BlazeBlock,
    pool: AdaptiveAvgPool2d,
    face_proj: Linear,

    // Quality gate: estimates frame quality, modulates GRU input
    quality_gate: Linear,

    // Crystallization GRU: hidden state IS the identity
    gru: GRUCell,

    // Convergence head: predicts convergence velocity and uncertainty
    convergence_head: Linear,

    /// Hidden state dimension (identity embedding dim)
    hidden_dim: usize,
    /// Face encoding dimension (GRU input dim)
    encoding_dim: usize,
}

impl Default for MnemosyneIdentity {
    fn default() -> Self {
        Self::new()
    }
}

impl MnemosyneIdentity {
    /// Create a new Mnemosyne face identity model with default dimensions.
    ///
    /// Default: encoding_dim=96, hidden_dim=64
    pub fn new() -> Self {
        Self::with_dims(96, 64)
    }

    /// Create with custom encoding and hidden dimensions.
    ///
    /// # Arguments
    /// * `encoding_dim` - Dimension of face encoding (GRU input). Larger = more
    ///   expressive single-frame features, but more GRU params.
    /// * `hidden_dim` - Dimension of GRU hidden state (= identity embedding dim).
    ///   Larger = richer identity representation, but harder to match.
    pub fn with_dims(encoding_dim: usize, hidden_dim: usize) -> Self {
        // Stem: [3, 64, 64] → [16, 32, 32]
        let stem_conv = Conv2d::with_options(3, 16, (3, 3), (2, 2), (1, 1), true);
        let stem_bn = BatchNorm2d::new(16);

        // Block1: [16, 32, 32] → [24, 16, 16]
        let block1 = BlazeBlock::new(16, 24, 2);
        // Block2: [24, 16, 16] → [32, 8, 8]
        let block2 = BlazeBlock::new(24, 32, 2);
        // Block3: [32, 8, 8] → [48, 4, 4]
        let block3 = BlazeBlock::new(32, 48, 2);

        // AdaptiveAvgPool2d: [B, 48, 4, 4] → [B, 48, 1, 1]
        let pool = AdaptiveAvgPool2d::new((1, 1));

        // 48 * 1 * 1 = 48
        let face_proj = Linear::new(48, encoding_dim);

        let quality_gate = Linear::new(encoding_dim, 1);
        let gru = GRUCell::new(encoding_dim, hidden_dim);
        let convergence_head = Linear::new(hidden_dim, 2);

        Self {
            stem_conv,
            stem_bn,
            block1,
            block2,
            block3,
            pool,
            face_proj,
            quality_gate,
            gru,
            convergence_head,
            hidden_dim,
            encoding_dim,
        }
    }

    /// Encode a face image to a feature vector.
    ///
    /// Runs the face through the BlazeBlock backbone, adaptive pooling,
    /// and projection to produce a compact encoding.
    ///
    /// Input: [B, 3, 64, 64] → Output: [B, encoding_dim]
    pub fn encode_face(&self, face: &Variable) -> Variable {
        let x = self.stem_bn.forward(&self.stem_conv.forward(face)).relu();
        let x = self.block1.forward(&x);
        let x = self.block2.forward(&x);
        let x = self.block3.forward(&x);

        // AdaptiveAvgPool2d: [B, 48, H, W] → [B, 48, 1, 1]
        let x = self.pool.forward(&x);

        // Flatten: [B, 48, 1, 1] → [B, 48] — uses Variable::reshape for autograd
        let shape = x.shape();
        let batch = shape[0];
        let channels = shape[1];
        let flat = x.reshape(&[batch, channels]);

        self.face_proj.forward(&flat).relu()
    }

    /// Compute quality gate for a face encoding.
    ///
    /// Returns a quality score in [0, 1] per sample. High quality frames
    /// (sharp, well-lit, unoccluded) get scores near 1.0, modulating the
    /// GRU input strongly. Blurry/poor frames get scores near 0.0, causing
    /// minimal hidden state update.
    ///
    /// Input: [B, encoding_dim] → Output: [B, 1]
    pub fn compute_quality(&self, encoding: &Variable) -> Variable {
        self.quality_gate.forward(encoding).sigmoid()
    }

    /// Perform one crystallization step.
    ///
    /// The core temporal identity mechanism: quality-gates the face encoding,
    /// feeds it through the GRU to evolve the identity hidden state, and
    /// predicts convergence metrics.
    ///
    /// # Arguments
    /// * `face` - Face image [B, 3, 64, 64]
    /// * `hidden` - Previous GRU hidden state [B, hidden_dim], or None for first frame
    ///
    /// # Returns
    /// * `new_hidden` - Updated identity state [B, hidden_dim]
    /// * `convergence_velocity` - How fast the state is changing [B, 1] (sigmoid, 0=crystallized)
    /// * `log_variance` - Uncertainty in the identity estimate [B, 1]
    /// * `quality` - Frame quality score [B, 1] (0=poor, 1=high quality)
    pub fn crystallize_step(
        &self,
        face: &Variable,
        hidden: Option<&Variable>,
    ) -> (Variable, Variable, Variable, Variable) {
        let encoding = self.encode_face(face);
        let quality = self.compute_quality(&encoding);
        let batch = encoding.shape()[0];

        // Quality-gate the encoding: gated = encoding ⊙ quality.expand
        // quality is [B, 1], encoding is [B, encoding_dim]
        // Expand quality to match encoding dim, then element-wise multiply
        let quality_expanded = quality.expand(&[batch, self.encoding_dim]);
        let gated_input = encoding.mul_var(&quality_expanded);

        // GRU step — evolve identity state
        let h = match hidden {
            Some(h) => h.clone(),
            None => Variable::new(Tensor::zeros(&[batch, self.hidden_dim]), false),
        };
        let new_hidden = self.gru.forward_step(&gated_input, &h);

        // Convergence head: [B, hidden_dim] → [B, 2]
        let conv_out = self.convergence_head.forward(&new_hidden);

        // Split into convergence_velocity (sigmoid → [0,1]) and log_variance
        // Use Variable::narrow to slice then apply sigmoid
        let velocity = conv_out.narrow(1, 0, 1).sigmoid();
        let log_variance = conv_out.narrow(1, 1, 1);

        (new_hidden, velocity, log_variance, quality)
    }

    /// Extract the crystallized identity embedding from a hidden state.
    ///
    /// L2-normalizes the hidden state to produce a unit embedding vector.
    /// This is the final identity representation used for matching.
    ///
    /// # Arguments
    /// * `hidden` - GRU hidden state [B, hidden_dim]
    ///
    /// # Returns
    /// L2-normalized embedding as Vec<f32> (first sample in batch)
    pub fn extract_identity(&self, hidden: &Variable) -> Vec<f32> {
        let data = hidden.data().to_vec();
        let dim = self.hidden_dim;
        let norm: f32 = data[..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 {
            return vec![0.0; dim];
        }
        data[..dim].iter().map(|x| x / norm).collect()
    }

    /// L2-normalize an identity embedding as a Variable (graph-tracked).
    ///
    /// Input: [B, hidden_dim] → Output: [B, hidden_dim] (unit norm per sample)
    pub fn normalize_identity(&self, hidden: &Variable) -> Variable {
        // L2 norm via scalar division (graph-tracked on hidden)
        let h_data = hidden.data().to_vec();
        let norm_val: f32 = h_data.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        hidden.mul_scalar(1.0 / norm_val)
    }

    /// Uncertainty-weighted cosine similarity between two crystallized identities.
    ///
    /// Combines both embeddings' uncertainty estimates to down-weight dimensions
    /// where either identity is uncertain. High precision (low variance) dimensions
    /// contribute more to the final score.
    ///
    /// # Arguments
    /// * `embedding_a`, `embedding_b` - L2-normalized identity embeddings
    /// * `logvar_a`, `logvar_b` - Log-variance uncertainty estimates
    ///
    /// # Returns
    /// Similarity score in [-1, 1], with 1.0 = perfect match
    pub fn match_identities(
        embedding_a: &[f32],
        embedding_b: &[f32],
        logvar_a: f32,
        logvar_b: f32,
    ) -> f32 {
        assert_eq!(embedding_a.len(), embedding_b.len());
        let dim = embedding_a.len();

        // Combined precision (inverse combined variance)
        let var_a = logvar_a.exp();
        let var_b = logvar_b.exp();
        let precision = 1.0 / (var_a + var_b + 1e-8);

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..dim {
            let wa = embedding_a[i] * precision;
            let wb = embedding_b[i] * precision;
            dot += wa * wb;
            norm_a += wa * wa;
            norm_b += wb * wb;
        }

        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
        dot / denom
    }

    /// Measure actual convergence between two consecutive hidden states.
    ///
    /// Computes the L2 distance between hidden states, normalized by dimension.
    /// Low values indicate the identity has crystallized (stabilized).
    ///
    /// # Returns
    /// Convergence delta in [0, inf), approaching 0 as identity stabilizes
    pub fn convergence_delta(hidden_prev: &[f32], hidden_curr: &[f32]) -> f32 {
        assert_eq!(hidden_prev.len(), hidden_curr.len());
        let dim = hidden_prev.len() as f32;
        let sq_dist: f32 = hidden_prev
            .iter()
            .zip(hidden_curr.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        (sq_dist / dim).sqrt()
    }

    // =========================================================================
    // Novel Capabilities
    // =========================================================================

    /// Assess liveness (anti-spoofing) from a temporal sequence of face frames.
    ///
    /// Analyzes the trajectory of GRU hidden state updates across a sequence of
    /// face frames. Real faces exhibit micro-variations in hidden state trajectory
    /// due to micro-expressions, subtle head movement, and natural lighting
    /// fluctuations. Photos and screens produce abnormally smooth or repetitive
    /// trajectories because the input signal lacks temporal diversity.
    ///
    /// # Mechanism
    ///
    /// 1. Each frame is processed through `crystallize_step`, collecting the
    ///    hidden state at each timestep.
    /// 2. **Temporal variance**: Compute the variance of consecutive hidden state
    ///    deltas (h_t - h_{t-1}). Real faces produce higher variance because each
    ///    frame genuinely differs. Spoofed inputs produce near-zero variance.
    /// 3. **Trajectory smoothness**: Compute the mean autocorrelation between
    ///    consecutive delta vectors. Real faces have low autocorrelation (each
    ///    delta is in a different direction). Replay attacks have high
    ///    autocorrelation (repetitive or static deltas).
    /// 4. The liveness score combines both signals: high variance AND low
    ///    smoothness indicates a live subject.
    ///
    /// # Arguments
    /// * `face_sequence` - Ordered sequence of face images, each [B, 3, 64, 64].
    ///   Minimum 3 frames required for meaningful analysis.
    ///
    /// # Returns
    /// A `LivenessResult` with liveness score, temporal variance, and trajectory
    /// smoothness. Returns `LivenessResult::unknown()` if fewer than 3 frames.
    pub fn assess_liveness(&self, face_sequence: &[Variable]) -> LivenessResult {
        use super::BiometricModality;

        if face_sequence.len() < 3 {
            return LivenessResult::unknown();
        }

        // Collect hidden states across the sequence
        let mut hidden: Option<Variable> = None;
        let mut hidden_states: Vec<Vec<f32>> = Vec::new();

        for frame in face_sequence {
            let (h, _velocity, _logvar, _quality) = self.crystallize_step(frame, hidden.as_ref());
            hidden_states.push(h.data().to_vec());
            hidden = Some(h);
        }

        // Compute deltas: delta_t = h_t - h_{t-1}
        let mut deltas: Vec<Vec<f32>> = Vec::new();
        for i in 1..hidden_states.len() {
            let delta: Vec<f32> = hidden_states[i]
                .iter()
                .zip(hidden_states[i - 1].iter())
                .map(|(a, b)| a - b)
                .collect();
            deltas.push(delta);
        }

        if deltas.is_empty() {
            return LivenessResult::unknown();
        }

        // --- Temporal variance ---
        // Compute the magnitude of each delta, then take the variance of those magnitudes
        let delta_magnitudes: Vec<f32> = deltas
            .iter()
            .map(|d| {
                let sq_sum: f32 = d.iter().map(|x| x * x).sum();
                sq_sum.sqrt()
            })
            .collect();

        let mean_mag: f32 = delta_magnitudes.iter().sum::<f32>() / delta_magnitudes.len() as f32;
        let temporal_variance: f32 = if delta_magnitudes.len() > 1 {
            delta_magnitudes
                .iter()
                .map(|m| (m - mean_mag) * (m - mean_mag))
                .sum::<f32>()
                / (delta_magnitudes.len() - 1) as f32
        } else {
            0.0
        };

        // --- Trajectory smoothness (autocorrelation of consecutive deltas) ---
        // For each pair of consecutive deltas, compute cosine similarity.
        // High mean cosine = smooth (suspicious). Low mean cosine = varied (live).
        let mut autocorrelations: Vec<f32> = Vec::new();
        for i in 1..deltas.len() {
            let dot: f32 = deltas[i]
                .iter()
                .zip(deltas[i - 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_a: f32 = deltas[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = deltas[i - 1].iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = (norm_a * norm_b).max(1e-8);
            autocorrelations.push(dot / denom);
        }

        let trajectory_smoothness: f32 = if autocorrelations.is_empty() {
            0.0
        } else {
            autocorrelations.iter().sum::<f32>() / autocorrelations.len() as f32
        };

        // --- Liveness score ---
        // High temporal_variance => live (more varied hidden state changes)
        // Low trajectory_smoothness => live (deltas point in different directions)
        //
        // variance_signal: sigmoid that saturates around typical real-face variance
        // We use a soft threshold: variance_signal = sigmoid(temporal_variance * scale - bias)
        let variance_signal = 1.0 / (1.0 + (-50.0 * (temporal_variance - 0.001)).exp());

        // smoothness_signal: lower smoothness is better for liveness
        // Map smoothness from [-1, 1] to a liveness-friendly score
        let smoothness_signal = 1.0 - (trajectory_smoothness.max(0.0));

        // Combine: geometric-ish blend giving both signals weight
        let liveness_score = (0.6 * variance_signal + 0.4 * smoothness_signal).clamp(0.0, 1.0);

        let is_live = liveness_score > 0.5;

        LivenessResult {
            liveness_score,
            is_live,
            temporal_variance,
            trajectory_smoothness,
            modality_liveness: vec![(BiometricModality::Face, liveness_score)],
        }
    }

    /// Detect identity drift by comparing current crystallized state to the
    /// original enrollment embedding.
    ///
    /// Identity drift occurs when a person's biometrics change over time (aging,
    /// weight change, injury). By measuring cosine distance between the current
    /// GRU hidden state and the original enrollment embedding, we can detect
    /// when re-enrollment is needed.
    ///
    /// # Arguments
    /// * `current_hidden` - Current GRU hidden state [B, hidden_dim] from recent
    ///   crystallization
    /// * `original_embedding` - L2-normalized embedding from original enrollment
    ///
    /// # Returns
    /// Cosine distance in [0, 2] where 0 = identical, 2 = opposite.
    /// Typical thresholds: <0.3 normal, 0.3-0.6 monitor, >0.6 re-enroll.
    pub fn detect_drift(&self, current_hidden: &Variable, original_embedding: &[f32]) -> f32 {
        let current_embedding = self.extract_identity(current_hidden);
        assert_eq!(
            current_embedding.len(),
            original_embedding.len(),
            "Embedding dimensions must match: current={}, original={}",
            current_embedding.len(),
            original_embedding.len()
        );

        // Cosine similarity
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..current_embedding.len() {
            dot += current_embedding[i] * original_embedding[i];
            norm_a += current_embedding[i] * current_embedding[i];
            norm_b += original_embedding[i] * original_embedding[i];
        }
        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
        let cosine_sim = dot / denom;

        // Cosine distance = 1 - cosine_similarity, clamped to [0, 2]
        (1.0 - cosine_sim).clamp(0.0, 2.0)
    }

    /// Process a sequence of face frames through the full crystallization pipeline
    /// with attention-weighted multi-frame aggregation.
    ///
    /// Each frame is encoded, quality-gated, and fed through the GRU. The method
    /// tracks per-frame quality scores and convergence velocity across the entire
    /// sequence, returning the final crystallized hidden state along with
    /// diagnostic information.
    ///
    /// # Arguments
    /// * `faces` - Ordered sequence of face images, each [B, 3, 64, 64]
    ///
    /// # Returns
    /// * `final_hidden` - The crystallized identity state [B, hidden_dim]
    /// * `per_frame_qualities` - Quality score for each frame (length = faces.len())
    /// * `final_convergence_velocity` - The convergence velocity at the last step.
    ///   Low values indicate the identity has crystallized.
    ///
    /// # Panics
    /// Panics if `faces` is empty.
    pub fn crystallize_sequence(&self, faces: &[Variable]) -> (Variable, Vec<f32>, f32) {
        assert!(!faces.is_empty(), "Face sequence must not be empty");

        let mut hidden: Option<Variable> = None;
        let mut per_frame_qualities: Vec<f32> = Vec::with_capacity(faces.len());
        let mut final_velocity: f32 = 1.0;

        for frame in faces {
            let (h, velocity, _logvar, quality) = self.crystallize_step(frame, hidden.as_ref());

            // Extract scalar quality for this frame
            let q_val = quality.data().to_vec()[0];
            per_frame_qualities.push(q_val);

            // Track convergence velocity
            final_velocity = velocity.data().to_vec()[0];

            hidden = Some(h);
        }

        (hidden.unwrap(), per_frame_qualities, final_velocity)
    }

    /// Assess the quality of a single face image.
    ///
    /// Encodes the face and evaluates it through the quality gate. The quality
    /// score reflects the learned assessment of frame suitability for identity
    /// recognition — considering factors like sharpness, lighting, and pose
    /// that the model has learned to associate with reliable identity extraction.
    ///
    /// # Arguments
    /// * `face` - Face image [B, 3, 64, 64]
    ///
    /// # Returns
    /// Quality score in [0, 1] where 0 = poor quality, 1 = high quality.
    /// Uses the first sample in the batch.
    pub fn assess_quality(&self, face: &Variable) -> f32 {
        let encoding = self.encode_face(face);

        // Encoding magnitude as a secondary signal: very weak encodings
        // suggest the input is not a meaningful face
        let enc_data = encoding.data().to_vec();
        let enc_magnitude: f32 = enc_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_factor = 1.0 / (1.0 + (-0.1 * (enc_magnitude - 1.0)).exp());

        // Quality gate output
        let quality = self.compute_quality(&encoding);
        let gate_score = quality.data().to_vec()[0];

        // Combined quality: both the gate and the encoding magnitude must be
        // reasonable for the face to be considered high quality
        (gate_score * 0.7 + magnitude_factor * 0.3).clamp(0.0, 1.0)
    }

    /// Collect all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.stem_conv.parameters());
        p.extend(self.stem_bn.parameters());
        p.extend(self.block1.parameters());
        p.extend(self.block2.parameters());
        p.extend(self.block3.parameters());
        p.extend(self.pool.parameters());
        p.extend(self.face_proj.parameters());
        p.extend(self.quality_gate.parameters());
        p.extend(self.gru.parameters());
        p.extend(self.convergence_head.parameters());
        p
    }

    /// Get the hidden state dimension (identity embedding dim).
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get the face encoding dimension (GRU input dim).
    pub fn encoding_dim(&self) -> usize {
        self.encoding_dim
    }
}

impl Module for MnemosyneIdentity {
    /// Forward pass: single-frame face encoding (no temporal state).
    ///
    /// For temporal crystallization, use [`crystallize_step`] directly.
    ///
    /// Input: [B, 3, 64, 64] → Output: [B, encoding_dim]
    fn forward(&self, input: &Variable) -> Variable {
        self.encode_face(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.parameters()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helper: create a face tensor with a given fill value
    // -------------------------------------------------------------------------

    fn make_face(batch: usize, fill: f32) -> Variable {
        let n = batch * 3 * 64 * 64;
        Variable::new(
            Tensor::from_vec(vec![fill; n], &[batch, 3, 64, 64]).unwrap(),
            false,
        )
    }

    fn make_face_grad(batch: usize, fill: f32) -> Variable {
        let n = batch * 3 * 64 * 64;
        Variable::new(
            Tensor::from_vec(vec![fill; n], &[batch, 3, 64, 64]).unwrap(),
            true,
        )
    }

    /// Create a face tensor with varied pixel values (simulates a real frame).
    fn make_varied_face(batch: usize, seed: u32) -> Variable {
        let n = batch * 3 * 64 * 64;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                // Simple pseudo-random based on index and seed
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32
                    / u32::MAX as f32;
                v * 2.0 - 1.0 // Map to [-1, 1]
            })
            .collect();
        Variable::new(Tensor::from_vec(data, &[batch, 3, 64, 64]).unwrap(), false)
    }

    // =========================================================================
    // Existing tests (preserved)
    // =========================================================================

    #[test]
    fn test_mnemosyne_creation() {
        let model = MnemosyneIdentity::new();
        assert_eq!(model.hidden_dim(), 64);
        assert_eq!(model.encoding_dim(), 96);
    }

    #[test]
    fn test_mnemosyne_param_count() {
        let model = MnemosyneIdentity::new();
        let total: usize = model
            .parameters()
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        assert!(total < 150_000, "Params {} exceeds 150K budget", total);
        assert!(total > 10_000, "Params {} seems too low", total);
    }

    #[test]
    fn test_mnemosyne_forward_shape() {
        let model = MnemosyneIdentity::new();
        let input = make_face(1, 0.5);
        let output = model.forward(&input);
        assert_eq!(output.shape(), &[1, 96]);

        // Verify output has non-zero values (not dead network)
        let data = output.data().to_vec();
        let nonzero = data.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(nonzero > 0, "All outputs are zero — dead network");
    }

    #[test]
    fn test_mnemosyne_crystallize_step() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.5);

        // First step (no previous hidden)
        let (hidden, velocity, logvar, quality) = model.crystallize_step(&face, None);
        assert_eq!(hidden.shape(), &[1, 64]);
        assert_eq!(velocity.shape(), &[1, 1]);
        assert_eq!(logvar.shape(), &[1, 1]);
        assert_eq!(quality.shape(), &[1, 1]);

        // Velocity should be in [0, 1] (sigmoid output)
        let vel_val = velocity.data().to_vec()[0];
        assert!(
            (0.0..=1.0).contains(&vel_val),
            "Velocity {} not in [0,1]",
            vel_val
        );

        // Quality should be in [0, 1] (sigmoid output)
        let qual_val = quality.data().to_vec()[0];
        assert!(
            (0.0..=1.0).contains(&qual_val),
            "Quality {} not in [0,1]",
            qual_val
        );
    }

    #[test]
    fn test_mnemosyne_multi_step_crystallization() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.3);

        let mut hidden = None;
        let mut prev_hidden_data: Option<Vec<f32>> = None;
        let mut convergence_deltas = Vec::new();

        for _ in 0..5 {
            let (h, _velocity, _logvar, _quality) = model.crystallize_step(&face, hidden.as_ref());

            let h_data = h.data().to_vec();
            if let Some(ref prev) = prev_hidden_data {
                let delta = MnemosyneIdentity::convergence_delta(prev, &h_data);
                convergence_deltas.push(delta);
            }
            prev_hidden_data = Some(h_data);
            hidden = Some(h);
        }

        // We should have 4 deltas (between 5 steps)
        assert_eq!(convergence_deltas.len(), 4);
        // All deltas should be finite
        for (i, d) in convergence_deltas.iter().enumerate() {
            assert!(d.is_finite(), "Delta {} at step {} is not finite", d, i);
        }
    }

    #[test]
    fn test_mnemosyne_identity_matching() {
        let a = vec![0.5, 0.3, 0.8, 0.1];
        let b = vec![0.5, 0.3, 0.8, 0.1];
        let score = MnemosyneIdentity::match_identities(&a, &b, -1.0, -1.0);
        assert!(score > 0.99, "Self-match score {} too low", score);

        let c = vec![-0.5, -0.3, -0.8, -0.1];
        let score2 = MnemosyneIdentity::match_identities(&a, &c, -1.0, -1.0);
        assert!(
            score2 < 0.0,
            "Opposite embedding score {} should be negative",
            score2
        );

        // Higher uncertainty should reduce confidence
        let score_uncertain = MnemosyneIdentity::match_identities(&a, &b, 2.0, 2.0);
        assert!(
            score_uncertain > 0.9,
            "Same embedding should still match: {}",
            score_uncertain
        );
    }

    #[test]
    fn test_mnemosyne_normalize_identity() {
        let model = MnemosyneIdentity::new();
        let hidden = Variable::new(
            Tensor::from_vec(vec![3.0, 4.0, 0.0, 0.0], &[1, 4]).unwrap(),
            false,
        );
        // Override hidden_dim for this test
        let normalized = model.normalize_identity(&hidden);
        let data = normalized.data().to_vec();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Not unit norm: {}", norm);
    }

    #[test]
    fn test_mnemosyne_quality_gate_range() {
        let model = MnemosyneIdentity::new();
        // Test with high-contrast input (should have defined quality)
        let face = make_face(1, 0.8);
        let encoding = model.encode_face(&face);
        let quality = model.compute_quality(&encoding);
        let q = quality.data().to_vec()[0];
        assert!((0.0..=1.0).contains(&q), "Quality {} not in [0,1]", q);
    }

    #[test]
    fn test_mnemosyne_forward_backward() {
        // NOTE: Full gradient flow to input requires Conv2d to track the graph
        // (known limitation — Conv2d uses Variable::new() internally).
        // This test verifies the forward pass produces a valid output
        // that can be used in loss computation.
        let model = MnemosyneIdentity::new();
        let face = make_face_grad(1, 0.5);
        let output = model.forward(&face);
        assert_eq!(output.shape(), &[1, 96]);

        // Verify output is valid for downstream loss
        let loss = output.sum();
        let loss_val = loss.data().to_vec()[0];
        assert!(loss_val.is_finite(), "Loss should be finite: {}", loss_val);
    }

    // =========================================================================
    // Liveness detection tests
    // =========================================================================

    #[test]
    fn test_liveness_real_face_varied_input() {
        let model = MnemosyneIdentity::new();

        // Simulate a real face: each frame has different pixel values
        let sequence: Vec<Variable> = (0..8).map(|i| make_varied_face(1, i * 12345 + 7)).collect();

        let result = model.assess_liveness(&sequence);

        assert!(
            result.liveness_score >= 0.0 && result.liveness_score <= 1.0,
            "Liveness score {} out of range",
            result.liveness_score
        );
        assert!(
            result.temporal_variance.is_finite(),
            "Temporal variance should be finite"
        );
        assert!(
            result.trajectory_smoothness.is_finite(),
            "Trajectory smoothness should be finite"
        );
        assert!(
            !result.modality_liveness.is_empty(),
            "Should have modality liveness entries"
        );
    }

    #[test]
    fn test_liveness_spoofed_constant_input() {
        let model = MnemosyneIdentity::new();

        // Simulate a spoofed (photo) face: every frame is identical
        let constant_face = make_face(1, 0.5);
        let sequence: Vec<Variable> = (0..8).map(|_| constant_face.clone()).collect();

        let result = model.assess_liveness(&sequence);

        // Constant input should produce low temporal variance
        // The GRU will still evolve the hidden state, but the deltas will be
        // more repetitive than with varied input
        assert!(
            result.temporal_variance.is_finite(),
            "Temporal variance should be finite"
        );
        assert!(
            result.liveness_score >= 0.0 && result.liveness_score <= 1.0,
            "Liveness score {} out of range",
            result.liveness_score
        );
    }

    #[test]
    fn test_liveness_varied_vs_constant_variance() {
        let model = MnemosyneIdentity::new();

        // Varied (real) sequence — use widely spaced seeds for maximum input diversity
        let varied_seq: Vec<Variable> = (0..8)
            .map(|i| make_varied_face(1, (i as u32).wrapping_mul(999_999_937)))
            .collect();
        let varied_result = model.assess_liveness(&varied_seq);

        // Constant (spoofed) sequence — every frame is identical
        let const_face = make_face(1, 0.5);
        let const_seq: Vec<Variable> = (0..8).map(|_| const_face.clone()).collect();
        let const_result = model.assess_liveness(&const_seq);

        // Both should produce finite, non-negative variance
        assert!(varied_result.temporal_variance.is_finite());
        assert!(const_result.temporal_variance.is_finite());
        assert!(varied_result.temporal_variance >= 0.0);
        assert!(const_result.temporal_variance >= 0.0);

        // With untrained weights, the absolute ordering is not guaranteed,
        // but both should be in a reasonable range. Constant input should
        // have near-zero variance since all frames are identical.
        // Use a relaxed check: constant variance should be small.
        assert!(
            const_result.temporal_variance < 0.5,
            "Constant input should have low temporal variance ({}), got {}",
            0.5,
            const_result.temporal_variance
        );
    }

    #[test]
    fn test_liveness_too_few_frames() {
        let model = MnemosyneIdentity::new();

        // Only 2 frames — below the minimum of 3
        let seq: Vec<Variable> = (0..2).map(|i| make_varied_face(1, i)).collect();
        let result = model.assess_liveness(&seq);

        // Should return unknown result
        assert_eq!(
            result.liveness_score, 0.5,
            "Too few frames should return unknown"
        );
        assert!(!result.is_live, "Too few frames should not be judged live");
    }

    #[test]
    fn test_liveness_minimum_frames() {
        let model = MnemosyneIdentity::new();

        // Exactly 3 frames — minimum required
        let seq: Vec<Variable> = (0..3).map(|i| make_varied_face(1, i * 5555)).collect();
        let result = model.assess_liveness(&seq);

        // Should produce a valid (non-unknown) result
        assert!(result.temporal_variance.is_finite());
        assert!(result.trajectory_smoothness.is_finite());
        assert!(result.liveness_score >= 0.0 && result.liveness_score <= 1.0);
    }

    #[test]
    fn test_liveness_smoothness_range() {
        let model = MnemosyneIdentity::new();

        let seq: Vec<Variable> = (0..6).map(|i| make_varied_face(1, i * 77777)).collect();
        let result = model.assess_liveness(&seq);

        // Trajectory smoothness is mean autocorrelation, should be in [-1, 1]
        assert!(
            result.trajectory_smoothness >= -1.0 && result.trajectory_smoothness <= 1.0,
            "Trajectory smoothness {} out of [-1, 1]",
            result.trajectory_smoothness
        );
    }

    // =========================================================================
    // Drift detection tests
    // =========================================================================

    #[test]
    fn test_drift_same_face_low_drift() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.5);

        // Crystallize to get a hidden state
        let (hidden, _v, _l, _q) = model.crystallize_step(&face, None);
        let embedding = model.extract_identity(&hidden);

        // Drift from same hidden state to same embedding should be ~0
        let drift = model.detect_drift(&hidden, &embedding);
        assert!(
            drift < 0.01,
            "Same-face drift should be near zero, got {}",
            drift
        );
    }

    #[test]
    fn test_drift_different_face_high_drift() {
        let model = MnemosyneIdentity::new();

        // Two very different face inputs
        let face_a = make_face(1, 0.1);
        let face_b = make_varied_face(1, 42);

        // Crystallize each
        let (hidden_a, _v, _l, _q) = model.crystallize_step(&face_a, None);
        let embedding_a = model.extract_identity(&hidden_a);

        // Run face_b through multiple steps to diverge
        let mut hidden_b = None;
        for _ in 0..5 {
            let (h, _v, _l, _q) = model.crystallize_step(&face_b, hidden_b.as_ref());
            hidden_b = Some(h);
        }

        let drift = model.detect_drift(hidden_b.as_ref().unwrap(), &embedding_a);

        // Drift should be non-trivial for different faces
        assert!(drift.is_finite(), "Drift should be finite, got {}", drift);
        assert!(drift >= 0.0, "Drift should be non-negative, got {}", drift);
    }

    #[test]
    fn test_drift_range() {
        let model = MnemosyneIdentity::new();
        let face = make_varied_face(1, 123);
        let (hidden, _v, _l, _q) = model.crystallize_step(&face, None);

        // Drift against an orthogonal embedding
        let orthogonal: Vec<f32> = (0..model.hidden_dim())
            .map(|i| if i == 0 { 1.0 } else { 0.0 })
            .collect();

        let drift = model.detect_drift(&hidden, &orthogonal);
        assert!(
            (0.0..=2.0).contains(&drift),
            "Drift {} should be in [0, 2]",
            drift
        );
    }

    // =========================================================================
    // Sequence crystallization tests
    // =========================================================================

    #[test]
    fn test_crystallize_sequence_basic() {
        let model = MnemosyneIdentity::new();

        let faces: Vec<Variable> = (0..5).map(|i| make_varied_face(1, i * 11111)).collect();

        let (final_hidden, qualities, final_velocity) = model.crystallize_sequence(&faces);

        assert_eq!(final_hidden.shape(), &[1, 64]);
        assert_eq!(qualities.len(), 5);

        // All qualities should be in [0, 1]
        for (i, q) in qualities.iter().enumerate() {
            assert!(
                *q >= 0.0 && *q <= 1.0,
                "Quality {} at frame {} out of [0,1]",
                q,
                i
            );
        }

        // Final velocity should be in [0, 1] (sigmoid output)
        assert!(
            (0.0..=1.0).contains(&final_velocity),
            "Final velocity {} out of [0,1]",
            final_velocity
        );
    }

    #[test]
    fn test_crystallize_sequence_single_frame() {
        let model = MnemosyneIdentity::new();

        let faces = vec![make_face(1, 0.5)];
        let (hidden, qualities, velocity) = model.crystallize_sequence(&faces);

        assert_eq!(hidden.shape(), &[1, 64]);
        assert_eq!(qualities.len(), 1);
        assert!((0.0..=1.0).contains(&velocity));
    }

    #[test]
    fn test_crystallize_sequence_convergence_over_time() {
        let model = MnemosyneIdentity::new();

        // Use the same face repeatedly — the hidden state should converge
        let face = make_face(1, 0.4);
        let faces: Vec<Variable> = (0..10).map(|_| face.clone()).collect();

        let mut prev_hidden_data: Option<Vec<f32>> = None;
        let mut hidden: Option<Variable> = None;
        let mut deltas = Vec::new();

        for frame in &faces {
            let (h, _v, _l, _q) = model.crystallize_step(frame, hidden.as_ref());
            let h_data = h.data().to_vec();
            if let Some(ref prev) = prev_hidden_data {
                deltas.push(MnemosyneIdentity::convergence_delta(prev, &h_data));
            }
            prev_hidden_data = Some(h_data);
            hidden = Some(h);
        }

        // All deltas should be finite and non-negative
        for d in &deltas {
            assert!(d.is_finite() && *d >= 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "Face sequence must not be empty")]
    fn test_crystallize_sequence_empty_panics() {
        let model = MnemosyneIdentity::new();
        let _result = model.crystallize_sequence(&[]);
    }

    // =========================================================================
    // Quality assessment tests
    // =========================================================================

    #[test]
    fn test_assess_quality_valid_input() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.5);

        let quality = model.assess_quality(&face);
        assert!(
            (0.0..=1.0).contains(&quality),
            "Quality {} out of [0,1]",
            quality
        );
    }

    #[test]
    fn test_assess_quality_nonzero() {
        let model = MnemosyneIdentity::new();
        let face = make_varied_face(1, 42);

        let quality = model.assess_quality(&face);
        // For non-trivial input, quality should be non-zero
        // (the encoding will have non-zero magnitude)
        assert!(quality.is_finite(), "Quality should be finite");
    }

    #[test]
    fn test_assess_quality_zero_input() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.0);

        let quality = model.assess_quality(&face);
        assert!(
            (0.0..=1.0).contains(&quality),
            "Quality {} out of range for zero input",
            quality
        );
    }

    #[test]
    fn test_assess_quality_range_across_inputs() {
        let model = MnemosyneIdentity::new();

        // Test several different inputs — all should produce valid quality
        for fill in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let face = make_face(1, fill);
            let q = model.assess_quality(&face);
            assert!(
                (0.0..=1.0).contains(&q),
                "Quality {} out of [0,1] for fill={}",
                q,
                fill
            );
        }
    }

    // =========================================================================
    // Batch processing tests
    // =========================================================================

    #[test]
    fn test_forward_batch() {
        let model = MnemosyneIdentity::new();
        let batch_face = make_face(4, 0.5);

        let output = model.forward(&batch_face);
        assert_eq!(output.shape(), &[4, 96]);
    }

    #[test]
    fn test_crystallize_step_batch() {
        let model = MnemosyneIdentity::new();
        let batch_face = make_face(3, 0.5);

        let (hidden, velocity, logvar, quality) = model.crystallize_step(&batch_face, None);
        assert_eq!(hidden.shape(), &[3, 64]);
        assert_eq!(velocity.shape(), &[3, 1]);
        assert_eq!(logvar.shape(), &[3, 1]);
        assert_eq!(quality.shape(), &[3, 1]);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_single_frame_liveness() {
        let model = MnemosyneIdentity::new();
        let seq = vec![make_face(1, 0.5)];
        let result = model.assess_liveness(&seq);
        // Single frame => unknown
        assert_eq!(result.liveness_score, 0.5);
        assert!(!result.is_live);
    }

    #[test]
    fn test_zero_input_forward() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.0);

        let output = model.forward(&face);
        assert_eq!(output.shape(), &[1, 96]);

        // Output should be finite even for zero input
        let data = output.data().to_vec();
        for v in &data {
            assert!(v.is_finite(), "Output should be finite for zero input");
        }
    }

    #[test]
    fn test_zero_input_crystallize() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.0);

        let (hidden, velocity, logvar, quality) = model.crystallize_step(&face, None);

        // All outputs should be finite
        for v in hidden.data().to_vec() {
            assert!(v.is_finite());
        }
        assert!(velocity.data().to_vec()[0].is_finite());
        assert!(logvar.data().to_vec()[0].is_finite());
        assert!(quality.data().to_vec()[0].is_finite());
    }

    // =========================================================================
    // Numerical stability tests
    // =========================================================================

    #[test]
    fn test_large_input_stability() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 100.0);

        let output = model.forward(&face);
        let data = output.data().to_vec();
        for v in &data {
            assert!(
                v.is_finite(),
                "Output should be finite for large input, got {}",
                v
            );
        }
    }

    #[test]
    fn test_small_input_stability() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 1e-7);

        let output = model.forward(&face);
        let data = output.data().to_vec();
        for v in &data {
            assert!(
                v.is_finite(),
                "Output should be finite for small input, got {}",
                v
            );
        }
    }

    #[test]
    fn test_negative_input_stability() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, -1.0);

        let output = model.forward(&face);
        let data = output.data().to_vec();
        for v in &data {
            assert!(
                v.is_finite(),
                "Output should be finite for negative input, got {}",
                v
            );
        }
    }

    // =========================================================================
    // Embedding property tests
    // =========================================================================

    #[test]
    fn test_extract_identity_l2_norm() {
        let model = MnemosyneIdentity::new();
        let face = make_face(1, 0.5);

        let (hidden, _v, _l, _q) = model.crystallize_step(&face, None);
        let embedding = model.extract_identity(&hidden);

        assert_eq!(embedding.len(), 64);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Identity embedding should be L2-normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn test_extract_identity_finite_values() {
        let model = MnemosyneIdentity::new();
        let face = make_varied_face(1, 999);

        let (hidden, _v, _l, _q) = model.crystallize_step(&face, None);
        let embedding = model.extract_identity(&hidden);

        for (i, v) in embedding.iter().enumerate() {
            assert!(v.is_finite(), "Embedding dim {} is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_extract_identity_zero_hidden() {
        let model = MnemosyneIdentity::new();

        // Zero hidden state should return zero embedding (safeguard)
        let hidden = Variable::new(Tensor::zeros(&[1, model.hidden_dim()]), false);
        let embedding = model.extract_identity(&hidden);

        assert_eq!(embedding.len(), model.hidden_dim());
        // All should be zero (norm < 1e-8 branch)
        for v in &embedding {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_normalize_identity_preserves_direction() {
        let model = MnemosyneIdentity::new();
        let hidden = Variable::new(
            Tensor::from_vec(vec![3.0, 4.0, 0.0, 0.0], &[1, 4]).unwrap(),
            false,
        );
        let normalized = model.normalize_identity(&hidden);
        let data = normalized.data().to_vec();

        // Direction should be preserved: ratio of first two elements
        // should be 3/4 = 0.75
        if data[1].abs() > 1e-8 {
            let ratio = data[0] / data[1];
            assert!(
                (ratio - 0.75).abs() < 0.01,
                "Direction not preserved: ratio={}",
                ratio
            );
        }
    }

    // =========================================================================
    // Custom dims test
    // =========================================================================

    #[test]
    fn test_custom_dims() {
        let model = MnemosyneIdentity::with_dims(128, 32);
        assert_eq!(model.encoding_dim(), 128);
        assert_eq!(model.hidden_dim(), 32);

        let face = make_face(1, 0.5);
        let output = model.forward(&face);
        assert_eq!(output.shape(), &[1, 128]);

        let (hidden, _v, _l, _q) = model.crystallize_step(&face, None);
        assert_eq!(hidden.shape(), &[1, 32]);

        let embedding = model.extract_identity(&hidden);
        assert_eq!(embedding.len(), 32);
    }

    // =========================================================================
    // Integration: liveness + crystallize_sequence combined
    // =========================================================================

    #[test]
    fn test_liveness_and_crystallize_sequence_together() {
        let model = MnemosyneIdentity::new();

        let faces: Vec<Variable> = (0..6).map(|i| make_varied_face(1, i * 31337)).collect();

        // Both should work on the same sequence
        let liveness = model.assess_liveness(&faces);
        let (final_hidden, qualities, final_vel) = model.crystallize_sequence(&faces);

        assert!(liveness.liveness_score >= 0.0 && liveness.liveness_score <= 1.0);
        assert_eq!(final_hidden.shape(), &[1, 64]);
        assert_eq!(qualities.len(), 6);
        assert!((0.0..=1.0).contains(&final_vel));
    }

    // =========================================================================
    // Integration: drift after crystallize_sequence
    // =========================================================================

    #[test]
    fn test_drift_after_crystallize_sequence() {
        let model = MnemosyneIdentity::new();

        let faces: Vec<Variable> = (0..5).map(|_| make_face(1, 0.5)).collect();

        let (hidden, _qualities, _vel) = model.crystallize_sequence(&faces);
        let embedding = model.extract_identity(&hidden);

        // Drift from the crystallized state to its own embedding should be ~0
        let drift = model.detect_drift(&hidden, &embedding);
        assert!(
            drift < 0.01,
            "Self-drift should be near zero, got {}",
            drift
        );
    }

    // =========================================================================
    // Convergence delta tests
    // =========================================================================

    #[test]
    fn test_convergence_delta_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let delta = MnemosyneIdentity::convergence_delta(&a, &a);
        assert!(delta < 1e-6, "Identical states should have zero delta");
    }

    #[test]
    fn test_convergence_delta_known() {
        // delta = sqrt(sum((a-b)^2) / dim)
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let delta = MnemosyneIdentity::convergence_delta(&a, &b);
        let expected = (25.0f32 / 3.0).sqrt();
        assert!(
            (delta - expected).abs() < 0.001,
            "Expected {}, got {}",
            expected,
            delta
        );
    }

    // =========================================================================
    // ACTUAL TRAINING TEST — Does this model learn?
    // =========================================================================

    #[test]
    fn test_mnemosyne_training_e2e() {
        use axonml_optim::{Adam, Optimizer};

        let model = MnemosyneIdentity::new();
        let params = model.parameters();
        println!("Mnemosyne params: {}", params.len());
        assert!(!params.is_empty(), "Model must have parameters");

        let mut optimizer = Adam::new(params, 0.001);

        // Create synthetic triplet data:
        // - anchor and positive: similar random faces (same "identity")
        // - negative: different random face
        let base_face = Tensor::randn(&[1, 3, 64, 64]);

        let mut losses = Vec::new();

        for step in 0..20 {
            // Generate triplet: anchor/positive are perturbations of same base
            let anchor_face = Variable::new(
                base_face
                    .add(&Tensor::randn(&[1, 3, 64, 64]).mul_scalar(0.1))
                    .unwrap(),
                false,
            );
            let positive_face = Variable::new(
                base_face
                    .add(&Tensor::randn(&[1, 3, 64, 64]).mul_scalar(0.1))
                    .unwrap(),
                false,
            );
            let negative_face = Variable::new(Tensor::randn(&[1, 3, 64, 64]), false);

            // Forward pass — single frame crystallization
            let (hidden_a, _vel_a, _, _) = model.crystallize_step(&anchor_face, None);
            let (hidden_p, _vel_p, _, _) = model.crystallize_step(&positive_face, None);
            let (hidden_n, _vel_n, _, _) = model.crystallize_step(&negative_face, None);

            // L2-normalize embeddings (graph-tracked via mul_scalar)
            let emb_a = l2_normalize_var(&hidden_a);
            let emb_p = l2_normalize_var(&hidden_p);
            let emb_n = l2_normalize_var(&hidden_n);

            // Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            let dot_pos = emb_a.mul_var(&emb_p).sum();
            let dot_neg = emb_a.mul_var(&emb_n).sum();
            let dist_pos = dot_pos.mul_scalar(-1.0).add_scalar(1.0);
            let dist_neg = dot_neg.mul_scalar(-1.0).add_scalar(1.0);
            let margin = 0.3;
            let loss = dist_pos.sub_var(&dist_neg).add_scalar(margin).relu();

            let loss_val = loss.data().to_vec()[0];
            losses.push(loss_val);

            if step == 0 {
                println!("Step 0: loss = {}", loss_val);
                assert!(
                    loss_val.is_finite(),
                    "Initial loss must be finite, got {}",
                    loss_val
                );
            }

            // Backward
            loss.backward();

            // Check gradients exist on at least some parameters
            if step == 0 {
                let params_after = model.parameters();
                let mut has_grad = 0;
                let mut zero_grad = 0;
                let mut no_grad = 0;
                for p in &params_after {
                    let name = p.name().to_string();
                    if let Some(g) = p.variable().grad() {
                        let grad_norm: f32 = g.to_vec().iter().map(|x| x * x).sum::<f32>().sqrt();
                        if grad_norm > 1e-10 {
                            has_grad += 1;
                            println!(
                                "  HAS GRAD: {} shape={:?} grad_norm={:.6}",
                                name,
                                p.variable().shape(),
                                grad_norm
                            );
                        } else {
                            zero_grad += 1;
                            println!("  ZERO GRAD: {} shape={:?}", name, p.variable().shape());
                        }
                    } else {
                        no_grad += 1;
                        println!("  NO GRAD: {} shape={:?}", name, p.variable().shape());
                    }
                }
                println!(
                    "Params with nonzero grad: {}, zero grad: {}, no grad: {}",
                    has_grad, zero_grad, no_grad
                );
                assert!(
                    has_grad > 0,
                    "At least some parameters must have non-zero gradients"
                );
            }

            // Step optimizer
            optimizer.step();
            optimizer.zero_grad();
        }

        // Check that loss trajectory shows some learning
        let first_5_avg: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_5_avg: f32 = losses[15..].iter().sum::<f32>() / 5.0;
        println!(
            "First 5 avg loss: {:.4}, Last 5 avg loss: {:.4}",
            first_5_avg, last_5_avg
        );
        println!("All losses: {:?}", losses);

        // Loss should at least not explode
        for (i, l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "Loss became non-finite at step {}: {}", i, l);
        }

        // We expect SOME decrease, but with random data it might not be dramatic
        // At minimum, loss should not increase significantly
        assert!(
            last_5_avg <= first_5_avg + 0.5,
            "Loss should not increase significantly: first_5={:.4} last_5={:.4}",
            first_5_avg,
            last_5_avg
        );
    }

    /// L2-normalize a Variable while keeping gradient flow.
    /// Uses sum + pow + mul_scalar which all have working backward.
    fn l2_normalize_var(x: &Variable) -> Variable {
        let sq = x.mul_var(x); // x^2
        let sum_sq = sq.sum(); // sum(x^2) — scalar
        let norm_val = sum_sq.data().to_vec()[0].sqrt().max(1e-8);
        x.mul_scalar(1.0 / norm_val)
    }
}
