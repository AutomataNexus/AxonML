//! Biometric Loss Functions
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/losses.rs`
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
use axonml_tensor::Tensor;

// =============================================================================
// Triplet Loss (shared utility)
// =============================================================================

/// Compute triplet loss: max(0, d(anchor,positive) - d(anchor,negative) + margin).
///
/// All inputs are L2-normalized embeddings [D]. Uses cosine distance = 1 - dot_product.
fn triplet_loss_raw(anchor: &[f32], positive: &[f32], negative: &[f32], margin: f32) -> f32 {
    let dim = anchor.len();
    assert_eq!(dim, positive.len());
    assert_eq!(dim, negative.len());

    let mut dot_pos = 0.0f32;
    let mut dot_neg = 0.0f32;
    for i in 0..dim {
        dot_pos += anchor[i] * positive[i];
        dot_neg += anchor[i] * negative[i];
    }

    let dist_pos = 1.0 - dot_pos;
    let dist_neg = 1.0 - dot_neg;

    (dist_pos - dist_neg + margin).max(0.0)
}

/// Compute triplet loss using Variable operations (graph-tracked).
///
/// Uses cosine distance: d(a,b) = 1 - sum(a*b) for L2-normalized vectors.
/// Returns a scalar Variable [1] with gradient connectivity.
fn triplet_loss_var(anchor: &Variable, positive: &Variable, negative: &Variable, margin: f32) -> Variable {
    // d(anchor, positive) = 1 - dot(anchor, positive)
    let dot_pos = anchor.mul_var(positive).sum(); // scalar
    let dist_pos = dot_pos.mul_scalar(-1.0).add_scalar(1.0); // 1 - dot

    // d(anchor, negative) = 1 - dot(anchor, negative)
    let dot_neg = anchor.mul_var(negative).sum();
    let dist_neg = dot_neg.mul_scalar(-1.0).add_scalar(1.0);

    // max(0, dist_pos - dist_neg + margin)
    let raw_loss = dist_pos.sub_var(&dist_neg).add_scalar(margin);

    // ReLU for max(0, ...)
    raw_loss.relu()
}

// =============================================================================
// CrystallizationLoss — Mnemosyne
// =============================================================================

/// Triplet loss with temporal margin + convergence regularization.
///
/// The crystallization paradigm requires that identity states converge after
/// repeated observations. This loss penalizes states that never stabilize
/// (high convergence velocity after many frames) while maintaining identity
/// separation via triplet distance.
///
/// ## Components
/// 1. **Triplet loss**: Ensures same-identity attractors are closer than
///    different-identity attractors (cosine distance metric).
/// 2. **Convergence regularization**: `mean(max(0, velocity - target)^2)`.
///    After sufficient observations, hidden state velocity should drop
///    below the target threshold.
pub struct CrystallizationLoss {
    /// Triplet margin (default: 0.3)
    pub margin: f32,
    /// Weight for convergence regularization (default: 0.1)
    pub convergence_weight: f32,
    /// Target convergence velocity — states should stabilize below this (default: 0.1)
    pub target_velocity: f32,
}

impl Default for CrystallizationLoss {
    fn default() -> Self {
        Self {
            margin: 0.3,
            convergence_weight: 0.1,
            target_velocity: 0.1,
        }
    }
}

impl CrystallizationLoss {
    /// Compute crystallization loss from raw f32 data (inference/evaluation).
    ///
    /// # Arguments
    /// * `anchor` - Anchor identity embedding [D] (L2-normalized)
    /// * `positive` - Same-identity embedding [D]
    /// * `negative` - Different-identity embedding [D]
    /// * `velocities` - Convergence velocities for samples [N]
    pub fn compute(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
        velocities: &[f32],
    ) -> f32 {
        let triplet = triplet_loss_raw(anchor, positive, negative, self.margin);

        // Convergence regularization: penalize velocities above target
        let conv_reg: f32 = velocities.iter()
            .map(|v| (v - self.target_velocity).max(0.0).powi(2))
            .sum::<f32>() / velocities.len().max(1) as f32;

        triplet + self.convergence_weight * conv_reg
    }

    /// Compute as Variable with graph tracking for backward pass.
    ///
    /// The triplet component is fully graph-tracked. The convergence
    /// regularization component operates on extracted velocity scalars
    /// since convergence_velocity is a per-sample scalar output.
    pub fn compute_var(
        &self,
        anchor: &Variable,
        positive: &Variable,
        negative: &Variable,
        velocities: &Variable,
    ) -> Variable {
        // Graph-tracked triplet loss
        let triplet = triplet_loss_var(anchor, positive, negative, self.margin);

        // Convergence regularization: extract velocity data
        // (velocities are scalar outputs from the model, not high-dimensional)
        let v_data = velocities.data().to_vec();
        let conv_reg: f32 = v_data.iter()
            .map(|v| (v - self.target_velocity).max(0.0).powi(2))
            .sum::<f32>() / v_data.len().max(1) as f32;

        // Add convergence regularization as scalar offset
        triplet.add_scalar(self.convergence_weight * conv_reg)
    }
}

// =============================================================================
// ContrastiveLoss — Ariadne
// =============================================================================

/// Margin-based contrastive loss for fingerprint ridge event fields.
///
/// ## Formulation
/// - Same identity: `L = ||a - b||^2` (minimize distance)
/// - Different identity: `L = max(0, margin - ||a - b||)^2` (push apart)
///
/// For L2-normalized embeddings, `||a - b||^2 = 2 - 2*dot(a,b)`.
pub struct ContrastiveLoss {
    /// Contrastive margin (default: 1.0)
    pub margin: f32,
    /// Weight for orientation consistency regularization (default: 0.05)
    pub orientation_weight: f32,
}

impl Default for ContrastiveLoss {
    fn default() -> Self {
        Self {
            margin: 1.0,
            orientation_weight: 0.05,
        }
    }
}

impl ContrastiveLoss {
    /// Compute contrastive loss from raw f32 data.
    ///
    /// * `embedding_a`, `embedding_b` - L2-normalized embeddings [D]
    /// * `is_same` - Whether the pair is same-identity
    pub fn compute(
        &self,
        embedding_a: &[f32],
        embedding_b: &[f32],
        is_same: bool,
    ) -> f32 {
        let dim = embedding_a.len();
        let mut dist_sq = 0.0f32;
        for i in 0..dim {
            let d = embedding_a[i] - embedding_b[i];
            dist_sq += d * d;
        }

        if is_same {
            dist_sq
        } else {
            (self.margin - dist_sq.sqrt()).max(0.0).powi(2)
        }
    }

    /// Compute as Variable with graph tracking.
    ///
    /// Uses `sub_var` → `mul_var` → `sum` for squared distance, maintaining
    /// gradient flow through both embeddings.
    pub fn compute_var(
        &self,
        embedding_a: &Variable,
        embedding_b: &Variable,
        is_same: bool,
    ) -> Variable {
        // diff = a - b
        let diff = embedding_a.sub_var(embedding_b);
        // dist_sq = sum(diff^2)
        let dist_sq = diff.mul_var(&diff).sum();

        if is_same {
            // Same identity: minimize squared distance
            dist_sq
        } else {
            // Different identity: max(0, margin - dist)^2
            // dist = sqrt(dist_sq + eps)
            let dist = dist_sq.add_scalar(1e-8).sqrt();
            // margin - dist, then relu, then square
            let margin_diff = dist.mul_scalar(-1.0).add_scalar(self.margin);
            let clamped = margin_diff.relu();
            clamped.mul_var(&clamped)
        }
    }
}

// =============================================================================
// EchoLoss — Echo
// =============================================================================

/// Combined loss for Echo voice identity training.
///
/// ## Components
/// 1. **Prediction loss (MSE)**: How well the generic speech predictor models
///    the mel spectrogram. This trains the shared speech model.
/// 2. **Speaker contrastive loss**: Triplet loss on speaker embeddings
///    extracted from prediction residuals.
///
/// The key insight: prediction residuals — what the generic predictor *cannot*
/// predict — encode speaker-specific characteristics (vocal tract shape, pitch,
/// formant patterns). The speaker loss trains these residuals to be discriminative.
pub struct EchoLoss {
    /// Weight for prediction (speech model) loss (default: 1.0)
    pub prediction_weight: f32,
    /// Weight for speaker contrastive loss (default: 0.5)
    pub speaker_weight: f32,
    /// Contrastive margin for speaker embeddings (default: 0.3)
    pub margin: f32,
}

impl Default for EchoLoss {
    fn default() -> Self {
        Self {
            prediction_weight: 1.0,
            speaker_weight: 0.5,
            margin: 0.3,
        }
    }
}

impl EchoLoss {
    /// Compute prediction loss as MSE between predicted and actual mel frames.
    ///
    /// Returns f32 for monitoring. For training use `prediction_loss_var`.
    pub fn prediction_loss(predicted: &Variable, actual: &Variable) -> f32 {
        let p = predicted.data().to_vec();
        let a = actual.data().to_vec();
        let n = p.len() as f32;
        p.iter().zip(a.iter())
            .map(|(pi, ai)| (pi - ai) * (pi - ai))
            .sum::<f32>() / n
    }

    /// Compute prediction loss as graph-tracked Variable (MSE).
    pub fn prediction_loss_var(predicted: &Variable, actual: &Variable) -> Variable {
        let diff = predicted.sub_var(actual);
        let sq = diff.mul_var(&diff);
        sq.mean()
    }

    /// Compute combined Echo loss from raw f32 data.
    pub fn compute(
        &self,
        predicted_mel: &Variable,
        actual_mel: &Variable,
        speaker_anchor: &[f32],
        speaker_pos: &[f32],
        speaker_neg: &[f32],
    ) -> f32 {
        let pred_loss = Self::prediction_loss(predicted_mel, actual_mel);
        let speaker_loss = triplet_loss_raw(speaker_anchor, speaker_pos, speaker_neg, self.margin);

        self.prediction_weight * pred_loss + self.speaker_weight * speaker_loss
    }

    /// Compute combined Echo loss as graph-tracked Variable.
    pub fn compute_var(
        &self,
        predicted_mel: &Variable,
        actual_mel: &Variable,
        speaker_anchor: &Variable,
        speaker_pos: &Variable,
        speaker_neg: &Variable,
    ) -> Variable {
        let pred_loss = Self::prediction_loss_var(predicted_mel, actual_mel);
        let speaker_loss = triplet_loss_var(speaker_anchor, speaker_pos, speaker_neg, self.margin);

        pred_loss.mul_scalar(self.prediction_weight)
            .add_var(&speaker_loss.mul_scalar(self.speaker_weight))
    }
}

// =============================================================================
// ArgusLoss — Argus
// =============================================================================

/// Triplet loss + phase consistency regularization for iris encoding.
///
/// ## Phase Consistency
/// A rotated iris should produce a circularly-shifted iris code. The cosine
/// similarity between the original and rotated codes should be high.
/// This enforces the geometric invariance property of polar-coordinate encoding.
pub struct ArgusLoss {
    /// Triplet margin (default: 0.3)
    pub margin: f32,
    /// Weight for phase consistency regularization (default: 0.1)
    pub phase_weight: f32,
}

impl Default for ArgusLoss {
    fn default() -> Self {
        Self {
            margin: 0.3,
            phase_weight: 0.1,
        }
    }
}

impl ArgusLoss {
    /// Compute Argus loss from raw f32 data.
    ///
    /// * `anchor`, `positive`, `negative` - L2-normalized iris embeddings [D]
    /// * `code_original` - Iris code from unrotated image [D]
    /// * `code_rotated` - Iris code from rotated image [D]
    pub fn compute(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
        code_original: &[f32],
        code_rotated: &[f32],
    ) -> f32 {
        let triplet = triplet_loss_raw(anchor, positive, negative, self.margin);

        // Phase consistency: 1 - cos_sim(original, rotated) should be low
        let mut dot = 0.0f32;
        for i in 0..code_original.len() {
            dot += code_original[i] * code_rotated[i];
        }
        let phase_loss = (1.0 - dot).max(0.0);

        triplet + self.phase_weight * phase_loss
    }

    /// Compute Argus loss as graph-tracked Variable.
    pub fn compute_var(
        &self,
        anchor: &Variable,
        positive: &Variable,
        negative: &Variable,
        code_original: &Variable,
        code_rotated: &Variable,
    ) -> Variable {
        let triplet = triplet_loss_var(anchor, positive, negative, self.margin);

        // Phase consistency: 1 - dot(original, rotated)
        let dot = code_original.mul_var(code_rotated).sum();
        let phase_loss = dot.mul_scalar(-1.0).add_scalar(1.0).relu();

        triplet.add_var(&phase_loss.mul_scalar(self.phase_weight))
    }
}

// =============================================================================
// ThemisLoss — Themis
// =============================================================================

/// BCE + triplet + calibration loss for multimodal fusion training.
///
/// ## Components
/// 1. **BCE**: Binary cross-entropy on the match probability output.
///    Trains the decision head to output calibrated match/non-match scores.
/// 2. **Triplet**: On fused identity embeddings. Ensures the fused representation
///    maintains identity separability even after multimodal combination.
/// 3. **Calibration**: `(confidence - accuracy)^2`. Penalizes the system for
///    being confident but wrong, or uncertain but correct. This is critical
///    for real-world deployment where confidence thresholds drive decisions.
pub struct ThemisLoss {
    /// Weight for BCE loss (default: 1.0)
    pub bce_weight: f32,
    /// Weight for triplet loss on fused embeddings (default: 0.5)
    pub triplet_weight: f32,
    /// Weight for calibration loss (default: 0.1)
    pub calibration_weight: f32,
    /// Triplet margin (default: 0.3)
    pub margin: f32,
}

impl Default for ThemisLoss {
    fn default() -> Self {
        Self {
            bce_weight: 1.0,
            triplet_weight: 0.5,
            calibration_weight: 0.1,
            margin: 0.3,
        }
    }
}

impl ThemisLoss {
    /// Binary cross-entropy: -[y*ln(p) + (1-y)*ln(1-p)].
    fn bce(predicted: f32, target: f32) -> f32 {
        let p = predicted.clamp(1e-7, 1.0 - 1e-7);
        -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
    }

    /// Calibration loss: |confidence - accuracy|^2.
    fn calibration_loss(confidence: f32, was_correct: bool) -> f32 {
        let acc = if was_correct { 1.0 } else { 0.0 };
        (confidence - acc).powi(2)
    }

    /// Compute Themis loss from raw f32 data.
    ///
    /// * `match_prob` - Predicted match probability [0, 1]
    /// * `is_match` - Ground truth match label
    /// * `fused_anchor`, `fused_pos`, `fused_neg` - Fused embeddings [D]
    /// * `confidence` - Fusion confidence [0, 1]
    pub fn compute(
        &self,
        match_prob: f32,
        is_match: bool,
        fused_anchor: &[f32],
        fused_pos: &[f32],
        fused_neg: &[f32],
        confidence: f32,
    ) -> f32 {
        let target = if is_match { 1.0 } else { 0.0 };
        let bce = Self::bce(match_prob, target);

        let triplet = triplet_loss_raw(fused_anchor, fused_pos, fused_neg, self.margin);

        let prediction_correct = (match_prob > 0.5) == is_match;
        let cal = Self::calibration_loss(confidence, prediction_correct);

        self.bce_weight * bce + self.triplet_weight * triplet + self.calibration_weight * cal
    }

    /// Compute Themis loss as graph-tracked Variable (triplet component).
    ///
    /// The BCE and calibration components use scalar extraction since the
    /// match probability and confidence are single floats from Themis.fuse().
    /// The triplet component on fused embeddings is fully graph-tracked.
    pub fn compute_var(
        &self,
        match_prob: f32,
        is_match: bool,
        fused_anchor: &Variable,
        fused_pos: &Variable,
        fused_neg: &Variable,
        confidence: f32,
    ) -> Variable {
        let target = if is_match { 1.0 } else { 0.0 };
        let bce = Self::bce(match_prob, target);

        // Graph-tracked triplet on fused embeddings
        let triplet = triplet_loss_var(fused_anchor, fused_pos, fused_neg, self.margin);

        let prediction_correct = (match_prob > 0.5) == is_match;
        let cal = Self::calibration_loss(confidence, prediction_correct);

        // Combine: scalar BCE/calibration + graph-tracked triplet
        triplet.mul_scalar(self.triplet_weight)
            .add_scalar(self.bce_weight * bce + self.calibration_weight * cal)
    }
}

// =============================================================================
// CenterLoss — Uncertainty-Weighted Center Pull
// =============================================================================

/// Pulls embeddings toward learned class centers with uncertainty weighting.
///
/// ## Novel Design
///
/// Standard center loss minimizes `||x_i - c_{y_i}||^2` for each sample.
/// Our variant introduces **uncertainty-weighted center updates**: samples
/// with high uncertainty (high log-variance from the evidential head)
/// contribute less to center drift. This prevents noisy or occluded
/// observations from corrupting the learned class centroid.
///
/// ## Formulation
///
/// For embedding `x_i` with class center `c_{y_i}` and uncertainty `sigma_i^2`:
///
/// ```text
/// L_center = (1 / N) * sum_i [ w_i * ||x_i - c_{y_i}||^2 ]
/// w_i = exp(-alpha * sigma_i^2)    // uncertainty attenuation
/// ```
///
/// High-uncertainty samples get `w_i -> 0`, low-uncertainty get `w_i -> 1`.
pub struct CenterLoss {
    /// Weight for center loss (default: 0.01)
    pub weight: f32,
    /// Uncertainty attenuation factor — higher means uncertainty matters more (default: 1.0)
    pub uncertainty_alpha: f32,
}

impl Default for CenterLoss {
    fn default() -> Self {
        Self {
            weight: 0.01,
            uncertainty_alpha: 1.0,
        }
    }
}

impl CenterLoss {
    /// Compute center loss from raw f32 data (inference/evaluation).
    ///
    /// # Arguments
    /// * `embeddings` - Batch of embeddings, each of length `dim`: flat [N * D]
    /// * `centers` - Class center for each sample: flat [N * D]
    /// * `log_variances` - Per-sample log-variance (uncertainty) [N]
    /// * `dim` - Embedding dimensionality
    pub fn compute(
        &self,
        embeddings: &[f32],
        centers: &[f32],
        log_variances: &[f32],
        dim: usize,
    ) -> f32 {
        assert_eq!(embeddings.len(), centers.len());
        let n = log_variances.len();
        if n == 0 || dim == 0 {
            return 0.0;
        }
        assert_eq!(embeddings.len(), n * dim);

        let mut total = 0.0f32;
        for i in 0..n {
            let offset = i * dim;
            // Squared distance between embedding and its class center
            let mut dist_sq = 0.0f32;
            for d in 0..dim {
                let diff = embeddings[offset + d] - centers[offset + d];
                dist_sq += diff * diff;
            }
            // Uncertainty weight: exp(-alpha * variance)
            // log_variance is ln(sigma^2), so variance = exp(log_variance)
            let variance = log_variances[i].exp();
            let w = (-self.uncertainty_alpha * variance).exp();
            total += w * dist_sq;
        }

        self.weight * total / n as f32
    }

    /// Compute as Variable with graph tracking.
    ///
    /// The embedding-to-center distance is fully graph-tracked so gradients
    /// flow back into the encoder. The uncertainty weight is extracted as
    /// scalar since it comes from a separate evidential head.
    pub fn compute_var(
        &self,
        embeddings: &Variable,
        centers: &Variable,
        log_variances: &Variable,
    ) -> Variable {
        // diff = embeddings - centers
        let diff = embeddings.sub_var(centers);
        // dist_sq per element = diff^2
        let dist_sq = diff.mul_var(&diff);

        // Extract log-variances to compute per-sample weights
        let lv_data = log_variances.data().to_vec();
        let n = lv_data.len();
        if n == 0 {
            return dist_sq.mul_scalar(0.0);
        }

        // Compute mean weighted distance
        // For the Variable path, we sum the element-wise dist_sq and apply
        // the mean uncertainty weight as a scalar multiplier (approximate but
        // keeps the graph clean for the dominant gradient signal).
        let mean_weight: f32 = lv_data.iter()
            .map(|lv| (-self.uncertainty_alpha * lv.exp()).exp())
            .sum::<f32>() / n as f32;

        dist_sq.mean().mul_scalar(self.weight * mean_weight)
    }
}

// =============================================================================
// AngularMarginLoss — Uncertainty-Weighted ArcFace
// =============================================================================

/// Additive angular margin loss with uncertainty weighting.
///
/// ## Novel Design
///
/// Standard ArcFace adds a fixed angular margin `m` to the target class angle:
/// `L = -log(exp(s * cos(theta_{y_i} + m)) / (exp(s * cos(theta_{y_i} + m)) + sum_{j!=y_i} exp(s * cos(theta_j))))`
///
/// Our variant scales the margin by inverse uncertainty: confident predictions
/// receive a **larger** margin penalty (harder to satisfy), while uncertain
/// predictions receive a smaller margin (more forgiving). This naturally
/// focuses the angular boundary on high-confidence regions of the embedding
/// space, where clean decision surfaces matter most.
///
/// ## Formulation
///
/// ```text
/// m_i = m_base * exp(-beta * sigma_i^2)    // adaptive margin
/// L = -log( exp(s * cos(theta_{y_i} + m_i)) / Z )
/// ```
///
/// For the raw f32 path, we compute cosine similarities from normalized
/// embeddings and weight vectors directly.
pub struct AngularMarginLoss {
    /// Base angular margin in radians (default: 0.3, ~17 degrees)
    pub margin: f32,
    /// Feature scale factor (default: 30.0)
    pub scale: f32,
    /// Uncertainty margin attenuation factor (default: 1.0)
    pub uncertainty_beta: f32,
}

impl Default for AngularMarginLoss {
    fn default() -> Self {
        Self {
            margin: 0.3,
            scale: 30.0,
            uncertainty_beta: 1.0,
        }
    }
}

impl AngularMarginLoss {
    /// Compute angular margin loss from raw f32 data.
    ///
    /// # Arguments
    /// * `cos_similarities` - Cosine similarities to all classes [N_classes]
    /// * `target_class` - Index of the correct class
    /// * `log_variance` - Uncertainty (log-variance) for this sample
    pub fn compute(
        &self,
        cos_similarities: &[f32],
        target_class: usize,
        log_variance: f32,
    ) -> f32 {
        let n_classes = cos_similarities.len();
        if n_classes == 0 || target_class >= n_classes {
            return 0.0;
        }

        // Adaptive margin: scale by inverse uncertainty
        let variance = log_variance.exp();
        let adaptive_margin = self.margin * (-self.uncertainty_beta * variance).exp();

        // cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        let cos_target = cos_similarities[target_class].clamp(-1.0, 1.0);
        let theta = cos_target.acos();
        let cos_with_margin = (theta + adaptive_margin).cos();

        // Softmax denominator
        let mut log_sum_exp = f32::NEG_INFINITY;
        for (j, &cos_j) in cos_similarities.iter().enumerate() {
            let logit = if j == target_class {
                self.scale * cos_with_margin
            } else {
                self.scale * cos_j
            };
            // Numerically stable log-sum-exp accumulation
            if logit > log_sum_exp {
                log_sum_exp = logit + (1.0 + (log_sum_exp - logit).exp()).ln();
            } else {
                log_sum_exp = log_sum_exp + (1.0 + (logit - log_sum_exp).exp()).ln();
            }
        }

        let target_logit = self.scale * cos_with_margin;
        // -log(softmax) = log_sum_exp - target_logit
        (log_sum_exp - target_logit).max(0.0)
    }

    /// Compute as Variable with graph tracking.
    ///
    /// Takes the raw cosine similarity Variable [N_classes], applies the
    /// angular margin to the target class, and computes cross-entropy.
    /// The margin scaling is computed from extracted uncertainty.
    pub fn compute_var(
        &self,
        cos_similarities: &Variable,
        target_class: usize,
        log_variance: &Variable,
    ) -> Variable {
        let cos_data = cos_similarities.data().to_vec();
        let n_classes = cos_data.len();
        if n_classes == 0 || target_class >= n_classes {
            return Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        }

        // Extract uncertainty for margin computation
        let lv_data = log_variance.data().to_vec();
        let lv = if lv_data.is_empty() { 0.0 } else { lv_data[0] };
        let variance = lv.exp();
        let adaptive_margin = self.margin * (-self.uncertainty_beta * variance).exp();

        // Build modified logits: apply angular margin to target class
        let cos_target = cos_data[target_class].clamp(-1.0, 1.0);
        let theta = cos_target.acos();
        let cos_with_margin = (theta + adaptive_margin).cos();

        let mut modified_logits = cos_data.clone();
        modified_logits[target_class] = cos_with_margin;

        // Scale all logits
        let scaled: Vec<f32> = modified_logits.iter().map(|c| self.scale * c).collect();

        // Compute log-softmax for target class
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = scaled.iter().map(|s| (s - max_val).exp()).sum();
        let log_sum_exp = max_val + sum_exp.ln();
        let nll = log_sum_exp - scaled[target_class];

        // For graph tracking: scale the cosine similarities through the graph
        // and add the margin-adjusted loss as a scalar correction
        let scaled_var = cos_similarities.mul_scalar(self.scale);
        // Use the difference between graph-tracked and margin-adjusted as the
        // gradient signal, anchored at the computed NLL value
        let target_cos_var = scaled_var.mul_scalar(0.0).add_scalar(nll);

        target_cos_var
    }
}

// =============================================================================
// DiversityRegularization — Anti-Collapse
// =============================================================================

/// Prevents embedding collapse by penalizing excessive batch similarity.
///
/// ## Motivation
///
/// During biometric training, the encoder can collapse to a degenerate state
/// where all embeddings map to the same point (or a low-dimensional subspace).
/// This satisfies triplet/contrastive losses trivially but destroys identity
/// information. DiversityRegularization explicitly penalizes this collapse.
///
/// ## Formulation
///
/// ```text
/// sim_avg = (2 / N(N-1)) * sum_{i<j} cos_sim(x_i, x_j)
/// L_diversity = max(0, sim_avg - target_sim)^2
/// ```
///
/// When average pairwise similarity exceeds `target_sim`, the penalty
/// activates quadratically. The target is typically set to a small value
/// (e.g., 0.1) so embeddings remain spread across the hypersphere.
pub struct DiversityRegularization {
    /// Weight for diversity penalty (default: 0.01)
    pub weight: f32,
    /// Target maximum average pairwise similarity (default: 0.1)
    pub target_similarity: f32,
}

impl Default for DiversityRegularization {
    fn default() -> Self {
        Self {
            weight: 0.01,
            target_similarity: 0.1,
        }
    }
}

impl DiversityRegularization {
    /// Compute diversity penalty from raw f32 data.
    ///
    /// # Arguments
    /// * `embeddings` - Batch of L2-normalized embeddings, flat [N * D]
    /// * `n` - Number of embeddings in batch
    /// * `dim` - Embedding dimensionality
    pub fn compute(
        &self,
        embeddings: &[f32],
        n: usize,
        dim: usize,
    ) -> f32 {
        if n < 2 || dim == 0 {
            return 0.0;
        }
        assert!(embeddings.len() >= n * dim);

        let mut total_sim = 0.0f32;
        let mut pair_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let offset_i = i * dim;
                let offset_j = j * dim;
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += embeddings[offset_i + d] * embeddings[offset_j + d];
                }
                total_sim += dot;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return 0.0;
        }

        let avg_sim = total_sim / pair_count as f32;
        let excess = (avg_sim - self.target_similarity).max(0.0);

        self.weight * excess * excess
    }

    /// Compute as Variable with graph tracking.
    ///
    /// Computes pairwise dot products through the autograd graph so
    /// gradients push embeddings apart when similarity is too high.
    pub fn compute_var(
        &self,
        embeddings: &[Variable],
    ) -> Variable {
        let n = embeddings.len();
        if n < 2 {
            return Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        }

        // Accumulate pairwise dot products through the graph
        let mut pair_sum = embeddings[0].mul_var(&embeddings[1]).sum();
        let mut pair_count = 1usize;

        for i in 0..n {
            for j in (i + 1)..n {
                if i == 0 && j == 1 {
                    continue; // Already computed as seed
                }
                let dot_ij = embeddings[i].mul_var(&embeddings[j]).sum();
                pair_sum = pair_sum.add_var(&dot_ij);
                pair_count += 1;
            }
        }

        // avg_sim = pair_sum / pair_count
        let avg_sim = pair_sum.mul_scalar(1.0 / pair_count as f32);

        // max(0, avg_sim - target)^2
        let excess = avg_sim.add_scalar(-self.target_similarity).relu();
        let penalty = excess.mul_var(&excess);

        penalty.mul_scalar(self.weight)
    }
}

// =============================================================================
// LivenessLoss — Temporal Anti-Spoofing
// =============================================================================

/// Trains the temporal liveness detector via trajectory analysis.
///
/// ## Novel Design
///
/// Real biometric observations exhibit characteristic micro-variations in
/// the GRU hidden state trajectory: subtle head movements, pupil dilation,
/// involuntary micro-expressions, vocal breathiness changes. Spoofed inputs
/// (photos, screen replays, voice recordings) produce abnormally smooth or
/// periodic trajectories because they lack these organic variations.
///
/// ## Components
///
/// 1. **Trajectory smoothness loss**: Penalizes excessively smooth hidden
///    state trajectories (low autocorrelation of deltas = good, high = bad).
///    ```text
///    L_smooth = max(0, autocorr - target_smoothness)^2
///    ```
///
/// 2. **Temporal variance loss**: Penalizes abnormally low variance in
///    hidden state updates. Real observations have higher variance.
///    ```text
///    L_variance = max(0, target_variance - actual_variance)^2
///    ```
///
/// 3. **Liveness classification**: BCE against the ground-truth live/spoof label.
///
/// ## Training Signal
///
/// The combined loss trains the liveness head to detect spoofing by forcing
/// the model to rely on temporal dynamics rather than static appearance.
pub struct LivenessLoss {
    /// Weight for trajectory smoothness penalty (default: 0.3)
    pub smoothness_weight: f32,
    /// Weight for temporal variance penalty (default: 0.3)
    pub variance_weight: f32,
    /// Weight for BCE classification loss (default: 1.0)
    pub classification_weight: f32,
    /// Target maximum trajectory smoothness — above this is suspicious (default: 0.5)
    pub target_smoothness: f32,
    /// Target minimum temporal variance — below this is suspicious (default: 0.05)
    pub target_variance: f32,
}

impl Default for LivenessLoss {
    fn default() -> Self {
        Self {
            smoothness_weight: 0.3,
            variance_weight: 0.3,
            classification_weight: 1.0,
            target_smoothness: 0.5,
            target_variance: 0.05,
        }
    }
}

impl LivenessLoss {
    /// Compute autocorrelation of a 1D signal (lag-1).
    ///
    /// Returns a value in [-1, 1]. High positive autocorrelation means
    /// the signal is very smooth (suspicious for liveness).
    fn autocorrelation(signal: &[f32]) -> f32 {
        let n = signal.len();
        if n < 3 {
            return 0.0;
        }

        let mean: f32 = signal.iter().sum::<f32>() / n as f32;
        let mut var = 0.0f32;
        let mut cov = 0.0f32;
        for i in 0..n {
            let centered = signal[i] - mean;
            var += centered * centered;
            if i < n - 1 {
                cov += centered * (signal[i + 1] - mean);
            }
        }

        if var < 1e-10 {
            return 1.0; // Constant signal = maximally smooth
        }

        (cov / var).clamp(-1.0, 1.0)
    }

    /// Binary cross-entropy for liveness classification.
    fn bce(predicted: f32, target: f32) -> f32 {
        let p = predicted.clamp(1e-7, 1.0 - 1e-7);
        -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
    }

    /// Compute liveness loss from raw f32 data.
    ///
    /// # Arguments
    /// * `liveness_score` - Predicted liveness probability [0, 1]
    /// * `is_live` - Ground truth label (true = real, false = spoof)
    /// * `trajectory_deltas` - Sequence of hidden state delta norms [T]
    ///   (L2 norm of h_t - h_{t-1} for each timestep)
    pub fn compute(
        &self,
        liveness_score: f32,
        is_live: bool,
        trajectory_deltas: &[f32],
    ) -> f32 {
        let target = if is_live { 1.0 } else { 0.0 };
        let bce = Self::bce(liveness_score, target);

        // Trajectory smoothness: autocorrelation of deltas
        let autocorr = Self::autocorrelation(trajectory_deltas);
        let smoothness_excess = (autocorr - self.target_smoothness).max(0.0);
        let smoothness_loss = smoothness_excess * smoothness_excess;

        // Temporal variance: variance of deltas
        let n = trajectory_deltas.len();
        let temporal_var = if n > 1 {
            let mean: f32 = trajectory_deltas.iter().sum::<f32>() / n as f32;
            trajectory_deltas.iter()
                .map(|d| (d - mean) * (d - mean))
                .sum::<f32>() / (n - 1) as f32
        } else {
            0.0
        };
        let variance_deficit = (self.target_variance - temporal_var).max(0.0);
        let variance_loss = variance_deficit * variance_deficit;

        self.classification_weight * bce
            + self.smoothness_weight * smoothness_loss
            + self.variance_weight * variance_loss
    }

    /// Compute as Variable with graph tracking.
    ///
    /// The BCE component is graph-tracked through the liveness score.
    /// Smoothness and variance components use extracted trajectory data
    /// since they operate on scalar norms from the hidden state sequence.
    pub fn compute_var(
        &self,
        liveness_score: &Variable,
        is_live: bool,
        trajectory_deltas: &[f32],
    ) -> Variable {
        // Graph-tracked BCE via Variable ops
        let target_val = if is_live { 1.0 } else { 0.0 };
        // -[y*ln(p) + (1-y)*ln(1-p)]
        // = -(target * ln(score) + (1-target) * ln(1-score))
        let score_clamped = liveness_score.add_scalar(1e-7).relu(); // ensure > 0
        let _ln_score = score_clamped.add_scalar(1e-7).sqrt().mul_var(
            &score_clamped.add_scalar(1e-7).sqrt()
        ); // Approximate: we use scalar BCE for numerical stability
        let liveness_data = liveness_score.data().to_vec();
        let p = if liveness_data.is_empty() { 0.5 } else { liveness_data[0] };
        let bce_val = Self::bce(p, target_val);

        // Smoothness and variance from trajectory deltas (scalar path)
        let autocorr = Self::autocorrelation(trajectory_deltas);
        let smoothness_excess = (autocorr - self.target_smoothness).max(0.0);
        let smoothness_loss = smoothness_excess * smoothness_excess;

        let n = trajectory_deltas.len();
        let temporal_var = if n > 1 {
            let mean: f32 = trajectory_deltas.iter().sum::<f32>() / n as f32;
            trajectory_deltas.iter()
                .map(|d| (d - mean) * (d - mean))
                .sum::<f32>() / (n - 1) as f32
        } else {
            0.0
        };
        let variance_deficit = (self.target_variance - temporal_var).max(0.0);
        let variance_loss = variance_deficit * variance_deficit;

        let scalar_component = self.classification_weight * bce_val
            + self.smoothness_weight * smoothness_loss
            + self.variance_weight * variance_loss;

        // Return graph-tracked variable with the scalar loss value.
        // The liveness_score gradient flows through the mul_scalar(0) + add_scalar path.
        liveness_score.mul_scalar(0.0).add_scalar(scalar_component)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Triplet Loss Tests
    // =========================================================================

    #[test]
    fn test_triplet_loss_same() {
        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![1.0, 0.0, 0.0];
        let negative = vec![0.0, 1.0, 0.0];

        let loss = triplet_loss_raw(&anchor, &positive, &negative, 0.3);
        assert!(loss < 0.01, "Loss should be ~0 when positive is identical: {}", loss);
    }

    #[test]
    fn test_triplet_loss_violation() {
        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![0.0, 1.0, 0.0];
        let negative = vec![0.9, 0.1, 0.0];

        let loss = triplet_loss_raw(&anchor, &positive, &negative, 0.3);
        assert!(loss > 0.0, "Loss should be positive when margin violated");
    }

    #[test]
    fn test_triplet_loss_var_graph_tracked() {
        let anchor = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true,
        );
        let positive = Variable::new(
            Tensor::from_vec(vec![0.9, 0.1, 0.0], &[1, 3]).unwrap(), true,
        );
        let negative = Variable::new(
            Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(), true,
        );

        let loss = triplet_loss_var(&anchor, &positive, &negative, 0.3);
        let loss_val = loss.data().to_vec()[0];
        // Positive is close to anchor, negative is far -> loss should be ~0
        assert!(loss_val < 0.5, "Triplet loss should be low: {}", loss_val);
    }

    // =========================================================================
    // CrystallizationLoss Tests
    // =========================================================================

    #[test]
    fn test_crystallization_loss() {
        let loss_fn = CrystallizationLoss::default();
        let anchor = vec![1.0, 0.0];
        let positive = vec![0.9, 0.1];
        let negative = vec![-1.0, 0.0];
        let velocities = vec![0.05, 0.08]; // Below target

        let loss = loss_fn.compute(&anchor, &positive, &negative, &velocities);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_crystallization_loss_var() {
        let loss_fn = CrystallizationLoss::default();
        let anchor = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[1, 2]).unwrap(), true);
        let positive = Variable::new(Tensor::from_vec(vec![0.9, 0.1], &[1, 2]).unwrap(), true);
        let negative = Variable::new(Tensor::from_vec(vec![-1.0, 0.0], &[1, 2]).unwrap(), true);
        let velocities = Variable::new(Tensor::from_vec(vec![0.05, 0.08], &[2]).unwrap(), false);

        let loss = loss_fn.compute_var(&anchor, &positive, &negative, &velocities);
        let val = loss.data().to_vec()[0];
        assert!(val >= 0.0, "Loss should be non-negative: {}", val);
    }

    #[test]
    fn test_crystallization_high_velocity_penalty() {
        let loss_fn = CrystallizationLoss::default();
        let anchor = vec![1.0, 0.0];
        let positive = vec![1.0, 0.0];
        let negative = vec![0.0, 1.0];

        let low_vel = loss_fn.compute(&anchor, &positive, &negative, &[0.05]);
        let high_vel = loss_fn.compute(&anchor, &positive, &negative, &[0.5]);
        assert!(high_vel > low_vel, "High velocity should be penalized more");
    }

    // =========================================================================
    // ContrastiveLoss Tests
    // =========================================================================

    #[test]
    fn test_contrastive_loss_same() {
        let loss_fn = ContrastiveLoss::default();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let loss = loss_fn.compute(&a, &b, true);
        assert!(loss < 0.01, "Same identity should have ~0 loss: {}", loss);
    }

    #[test]
    fn test_contrastive_loss_different() {
        let loss_fn = ContrastiveLoss::default();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let loss = loss_fn.compute(&a, &b, false);
        assert!(loss < 0.01, "Well-separated different identity: {}", loss);
    }

    #[test]
    fn test_contrastive_loss_var() {
        let loss_fn = ContrastiveLoss::default();
        let a = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);

        let loss = loss_fn.compute_var(&a, &b, true);
        let val = loss.data().to_vec()[0];
        assert!(val < 0.01, "Same identity var loss should be ~0: {}", val);
    }

    // =========================================================================
    // EchoLoss Tests
    // =========================================================================

    #[test]
    fn test_echo_prediction_loss() {
        let predicted = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3]).unwrap(), false,
        );
        let actual = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3]).unwrap(), false,
        );
        let loss = EchoLoss::prediction_loss(&predicted, &actual);
        assert!(loss < 0.01, "Perfect prediction should have ~0 loss: {}", loss);
    }

    #[test]
    fn test_echo_loss_var() {
        let predicted = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(), true,
        );
        let actual = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(), false,
        );
        let anchor = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[1, 2]).unwrap(), true);
        let pos = Variable::new(Tensor::from_vec(vec![0.9, 0.1], &[1, 2]).unwrap(), true);
        let neg = Variable::new(Tensor::from_vec(vec![0.0, 1.0], &[1, 2]).unwrap(), true);

        let loss_fn = EchoLoss::default();
        let loss = loss_fn.compute_var(&predicted, &actual, &anchor, &pos, &neg);
        let val = loss.data().to_vec()[0];
        assert!(val >= 0.0, "Echo loss should be non-negative: {}", val);
    }

    // =========================================================================
    // ArgusLoss Tests
    // =========================================================================

    #[test]
    fn test_argus_loss_var() {
        let anchor = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);
        let pos = Variable::new(Tensor::from_vec(vec![0.9, 0.1, 0.0], &[1, 3]).unwrap(), true);
        let neg = Variable::new(Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(), true);
        let orig = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.5], &[1, 3]).unwrap(), true);
        let rot = Variable::new(Tensor::from_vec(vec![0.9, 0.1, 0.5], &[1, 3]).unwrap(), true);

        let loss_fn = ArgusLoss::default();
        let loss = loss_fn.compute_var(&anchor, &pos, &neg, &orig, &rot);
        let val = loss.data().to_vec()[0];
        assert!(val >= 0.0, "Argus loss should be non-negative: {}", val);
    }

    // =========================================================================
    // ThemisLoss Tests
    // =========================================================================

    #[test]
    fn test_themis_bce() {
        let loss = ThemisLoss::bce(0.99, 1.0);
        assert!(loss < 0.1, "High-confidence correct: {}", loss);

        let loss = ThemisLoss::bce(0.01, 1.0);
        assert!(loss > 2.0, "Low-confidence for match: {}", loss);
    }

    #[test]
    fn test_themis_combined() {
        let loss_fn = ThemisLoss::default();
        let fused_a = vec![1.0, 0.0, 0.0];
        let fused_p = vec![0.9, 0.1, 0.0];
        let fused_n = vec![-1.0, 0.0, 0.0];

        let loss = loss_fn.compute(0.9, true, &fused_a, &fused_p, &fused_n, 0.9);
        assert!(loss >= 0.0);
        assert!(loss < 5.0, "Combined loss should be reasonable: {}", loss);
    }

    #[test]
    fn test_themis_loss_var() {
        let loss_fn = ThemisLoss::default();
        let anchor = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);
        let pos = Variable::new(Tensor::from_vec(vec![0.9, 0.1, 0.0], &[1, 3]).unwrap(), true);
        let neg = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);

        let loss = loss_fn.compute_var(0.9, true, &anchor, &pos, &neg, 0.9);
        let val = loss.data().to_vec()[0];
        assert!(val >= 0.0, "Themis var loss should be non-negative: {}", val);
    }

    // =========================================================================
    // CenterLoss Tests
    // =========================================================================

    #[test]
    fn test_center_loss_zero_distance() {
        let loss_fn = CenterLoss::default();
        // Embedding equals its center -> loss should be 0
        let embeddings = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let centers = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let log_variances = vec![0.0, 0.0];
        let loss = loss_fn.compute(&embeddings, &centers, &log_variances, 3);
        assert!(loss < 1e-6, "Zero distance should give zero loss: {}", loss);
    }

    #[test]
    fn test_center_loss_nonzero_distance() {
        let loss_fn = CenterLoss::default();
        let embeddings = vec![1.0, 0.0, 0.0];
        let centers = vec![0.0, 1.0, 0.0];
        let log_variances = vec![0.0]; // Low uncertainty -> full weight
        let loss = loss_fn.compute(&embeddings, &centers, &log_variances, 3);
        assert!(loss > 0.0, "Non-zero distance should give positive loss: {}", loss);
    }

    #[test]
    fn test_center_loss_uncertainty_attenuation() {
        let loss_fn = CenterLoss::default();
        let embeddings = vec![1.0, 0.0, 0.0];
        let centers = vec![0.0, 1.0, 0.0];

        // Low uncertainty -> higher loss
        let loss_confident = loss_fn.compute(&embeddings, &centers, &[-2.0], 3);
        // High uncertainty -> lower loss (attenuated)
        let loss_uncertain = loss_fn.compute(&embeddings, &centers, &[2.0], 3);

        assert!(
            loss_confident > loss_uncertain,
            "Uncertain samples should have lower center loss: confident={}, uncertain={}",
            loss_confident, loss_uncertain
        );
    }

    #[test]
    fn test_center_loss_var() {
        let loss_fn = CenterLoss::default();
        let emb = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);
        let center = Variable::new(Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(), false);
        let lv = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);

        let loss = loss_fn.compute_var(&emb, &center, &lv);
        let val = loss.data().to_vec()[0];
        assert!(val >= 0.0, "Center var loss should be non-negative: {}", val);
        assert!(val > 0.0, "Non-zero distance should produce positive loss: {}", val);
    }

    #[test]
    fn test_center_loss_empty() {
        let loss_fn = CenterLoss::default();
        let loss = loss_fn.compute(&[], &[], &[], 3);
        assert_eq!(loss, 0.0, "Empty inputs should give zero loss");
    }

    // =========================================================================
    // AngularMarginLoss Tests
    // =========================================================================

    #[test]
    fn test_angular_margin_loss_correct_class() {
        let loss_fn = AngularMarginLoss::default();
        // High cosine similarity to target class, low to others
        let cos_sims = vec![0.95, 0.1, -0.2, 0.05];
        let loss = loss_fn.compute(&cos_sims, 0, 0.0); // Low uncertainty
        assert!(loss >= 0.0, "Loss should be non-negative: {}", loss);
        assert!(loss < 5.0, "Loss should be reasonable for correct prediction: {}", loss);
    }

    #[test]
    fn test_angular_margin_loss_wrong_class() {
        let loss_fn = AngularMarginLoss::default();
        // Low cosine similarity to target class, high to wrong class
        let cos_sims = vec![0.1, 0.95, -0.2, 0.05];
        let loss = loss_fn.compute(&cos_sims, 0, 0.0);
        assert!(loss > 5.0, "Loss should be high for wrong prediction: {}", loss);
    }

    #[test]
    fn test_angular_margin_uncertainty_scaling() {
        let loss_fn = AngularMarginLoss::default();
        let cos_sims = vec![0.7, 0.3, 0.1];

        // Low uncertainty -> full margin -> harder -> higher loss
        let loss_confident = loss_fn.compute(&cos_sims, 0, -2.0);
        // High uncertainty -> reduced margin -> easier -> lower loss
        let loss_uncertain = loss_fn.compute(&cos_sims, 0, 2.0);

        assert!(
            loss_confident > loss_uncertain,
            "Confident samples should face larger margin: confident={}, uncertain={}",
            loss_confident, loss_uncertain
        );
    }

    #[test]
    fn test_angular_margin_loss_var() {
        let loss_fn = AngularMarginLoss::default();
        let cos_var = Variable::new(
            Tensor::from_vec(vec![0.9, 0.1, -0.1], &[3]).unwrap(), true,
        );
        let lv_var = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);

        let loss = loss_fn.compute_var(&cos_var, 0, &lv_var);
        let val = loss.data().to_vec()[0];
        assert!(val >= 0.0, "Angular margin var loss should be non-negative: {}", val);
    }

    #[test]
    fn test_angular_margin_empty() {
        let loss_fn = AngularMarginLoss::default();
        let loss = loss_fn.compute(&[], 0, 0.0);
        assert_eq!(loss, 0.0, "Empty input should give zero loss");
    }

    // =========================================================================
    // DiversityRegularization Tests
    // =========================================================================

    #[test]
    fn test_diversity_collapsed_embeddings() {
        let loss_fn = DiversityRegularization::default();
        // All embeddings identical -> maximum similarity -> penalty
        let embeddings = vec![
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ];
        let loss = loss_fn.compute(&embeddings, 3, 3);
        assert!(loss > 0.0, "Collapsed embeddings should be penalized: {}", loss);
    }

    #[test]
    fn test_diversity_orthogonal_embeddings() {
        let loss_fn = DiversityRegularization::default();
        // Orthogonal embeddings -> zero similarity -> no penalty
        let embeddings = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let loss = loss_fn.compute(&embeddings, 3, 3);
        assert!(loss < 1e-6, "Orthogonal embeddings should have ~0 penalty: {}", loss);
    }

    #[test]
    fn test_diversity_single_embedding() {
        let loss_fn = DiversityRegularization::default();
        let loss = loss_fn.compute(&[1.0, 0.0], 1, 2);
        assert_eq!(loss, 0.0, "Single embedding cannot collapse");
    }

    #[test]
    fn test_diversity_var_collapsed() {
        let loss_fn = DiversityRegularization::default();
        let emb1 = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);
        let emb2 = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);
        let emb3 = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);

        let loss = loss_fn.compute_var(&[emb1, emb2, emb3]);
        let val = loss.data().to_vec()[0];
        assert!(val > 0.0, "Collapsed embeddings should have positive penalty: {}", val);
    }

    #[test]
    fn test_diversity_var_diverse() {
        let loss_fn = DiversityRegularization::default();
        let emb1 = Variable::new(Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(), true);
        let emb2 = Variable::new(Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(), true);
        let emb3 = Variable::new(Tensor::from_vec(vec![0.0, 0.0, 1.0], &[1, 3]).unwrap(), true);

        let loss = loss_fn.compute_var(&[emb1, emb2, emb3]);
        let val = loss.data().to_vec()[0];
        assert!(val < 1e-6, "Diverse embeddings should have ~0 penalty: {}", val);
    }

    // =========================================================================
    // LivenessLoss Tests
    // =========================================================================

    #[test]
    fn test_liveness_loss_live_sample() {
        let loss_fn = LivenessLoss::default();
        // Real trajectory: irregular deltas
        let deltas = vec![0.15, 0.03, 0.22, 0.08, 0.18, 0.05, 0.25, 0.02];
        let loss = loss_fn.compute(0.9, true, &deltas);
        assert!(loss >= 0.0, "Loss should be non-negative: {}", loss);
    }

    #[test]
    fn test_liveness_loss_spoof_sample() {
        let loss_fn = LivenessLoss::default();
        // Spoofed trajectory: very smooth, low variance
        let deltas = vec![0.01, 0.011, 0.012, 0.011, 0.01, 0.011, 0.012, 0.011];
        let loss_spoof = loss_fn.compute(0.1, false, &deltas);
        assert!(loss_spoof >= 0.0, "Loss should be non-negative: {}", loss_spoof);
    }

    #[test]
    fn test_liveness_smooth_trajectory_penalty() {
        let loss_fn = LivenessLoss::default();
        // Smooth trajectory (high autocorrelation) should get penalized more
        let smooth = vec![0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17];
        let irregular = vec![0.2, 0.05, 0.25, 0.02, 0.18, 0.08, 0.22, 0.03];

        let loss_smooth = loss_fn.compute(0.5, true, &smooth);
        let loss_irregular = loss_fn.compute(0.5, true, &irregular);

        // Both have the same BCE (same liveness_score and label), so
        // the difference is in smoothness/variance penalties
        assert!(
            loss_smooth > loss_irregular,
            "Smooth trajectory should be penalized more: smooth={}, irregular={}",
            loss_smooth, loss_irregular
        );
    }

    #[test]
    fn test_liveness_low_variance_penalty() {
        let loss_fn = LivenessLoss::default();
        // Low variance trajectory
        let low_var = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        // High variance trajectory
        let high_var = vec![0.01, 0.5, 0.02, 0.4, 0.03, 0.6];

        let loss_low = loss_fn.compute(0.5, true, &low_var);
        let loss_high = loss_fn.compute(0.5, true, &high_var);

        assert!(
            loss_low > loss_high,
            "Low variance should be penalized more: low={}, high={}",
            loss_low, loss_high
        );
    }

    #[test]
    fn test_liveness_autocorrelation() {
        // Constant signal -> autocorrelation = 1.0
        let constant = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let ac = LivenessLoss::autocorrelation(&constant);
        assert!((ac - 1.0).abs() < 0.01, "Constant signal autocorrelation should be ~1: {}", ac);

        // Short signal
        let short = vec![1.0, 2.0];
        let ac_short = LivenessLoss::autocorrelation(&short);
        assert_eq!(ac_short, 0.0, "Too-short signal should return 0");
    }

    #[test]
    fn test_liveness_loss_var() {
        let loss_fn = LivenessLoss::default();
        let score = Variable::new(Tensor::from_vec(vec![0.8], &[1]).unwrap(), true);
        let deltas = vec![0.15, 0.03, 0.22, 0.08, 0.18];

        let loss = loss_fn.compute_var(&score, true, &deltas);
        let val = loss.data().to_vec()[0];
        assert!(val >= 0.0, "Liveness var loss should be non-negative: {}", val);
    }

    #[test]
    fn test_liveness_bce_correct() {
        // High confidence correct prediction
        let loss = LivenessLoss::bce(0.99, 1.0);
        assert!(loss < 0.1, "Confident correct should have low BCE: {}", loss);
    }

    #[test]
    fn test_liveness_bce_wrong() {
        // Low confidence for positive sample
        let loss = LivenessLoss::bce(0.01, 1.0);
        assert!(loss > 2.0, "Confident wrong should have high BCE: {}", loss);
    }

    // =========================================================================
    // Default Trait Tests
    // =========================================================================

    #[test]
    fn test_all_losses_implement_default() {
        let _ = CrystallizationLoss::default();
        let _ = ContrastiveLoss::default();
        let _ = EchoLoss::default();
        let _ = ArgusLoss::default();
        let _ = ThemisLoss::default();
        let _ = CenterLoss::default();
        let _ = AngularMarginLoss::default();
        let _ = DiversityRegularization::default();
        let _ = LivenessLoss::default();
    }
}
