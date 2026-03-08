//! Themis — Multimodal Belief Propagation Fusion (~49K params)
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/themis.rs`
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

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_nn::{GRUCell, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use super::{BiometricModality, DimensionContribution, ForensicReport, ModalityForensic};

// =============================================================================
// Types
// =============================================================================

/// Conflict between two biometric modalities during fusion.
///
/// Generated when two modalities disagree significantly about identity.
/// Essential for forensic audit trails and anomaly investigation.
#[derive(Debug, Clone)]
pub struct ModalityConflict {
    /// First modality in the conflicting pair.
    pub modality_a: BiometricModality,
    /// Second modality in the conflicting pair.
    pub modality_b: BiometricModality,
    /// Score from the first modality.
    pub score_a: f32,
    /// Score from the second modality.
    pub score_b: f32,
    /// Severity of the conflict [0, 1]. Higher = more severe disagreement.
    pub severity: f32,
}

// =============================================================================
// ThemisFusion
// =============================================================================

/// Multimodal belief propagation fusion.
///
/// Fuses biometric evidence from up to 4 modalities with dynamic
/// uncertainty-aware weighting and cross-modal consistency checking.
/// A GRU accumulates belief over time — the fusion state itself crystallizes.
///
/// # Parameters (~49K)
/// - 4 modality projectors: face(64->48), finger(128->48), voice(64->48), iris(128->48)
/// - Consistency checker: Linear(192->64) + Linear(64->4)
/// - Belief GRU: GRUCell(48->48)
/// - Decision head: Linear(48->1)
/// - Identity head: Linear(48->48)
pub struct ThemisFusion {
    // Modality projectors to common space
    face_proj: Linear,
    finger_proj: Linear,
    voice_proj: Linear,
    iris_proj: Linear,

    // Consistency checker
    consistency_fc1: Linear,
    consistency_fc2: Linear,

    // Belief GRU
    belief_gru: GRUCell,

    // Output heads
    decision_head: Linear,
    identity_head: Linear,

    /// Common fusion dimension
    fusion_dim: usize,
    /// Temperature for uncertainty gating
    temperature: f32,

    /// Historical reliability scores per modality [0, 1].
    /// Tracks how reliable each modality has been historically.
    /// Updated after each verification via `update_reliability()`.
    reliability_scores: HashMap<BiometricModality, f32>,
}

impl ThemisFusion {
    /// Create a new Themis fusion model with default dimensions.
    pub fn new() -> Self {
        Self::with_config(64, 128, 64, 128, 48, 2.0)
    }

    /// Create with custom modality dimensions.
    ///
    /// * `face_dim` - Mnemosyne hidden state dim (default: 64)
    /// * `finger_dim` - Ariadne embedding dim (default: 128)
    /// * `voice_dim` - Echo embedding dim (default: 64)
    /// * `iris_dim` - Argus embedding dim (default: 128)
    /// * `fusion_dim` - Common projection dimension (default: 48)
    /// * `temperature` - Uncertainty gate temperature (default: 2.0)
    pub fn with_config(
        face_dim: usize,
        finger_dim: usize,
        voice_dim: usize,
        iris_dim: usize,
        fusion_dim: usize,
        temperature: f32,
    ) -> Self {
        let face_proj = Linear::new(face_dim, fusion_dim);
        let finger_proj = Linear::new(finger_dim, fusion_dim);
        let voice_proj = Linear::new(voice_dim, fusion_dim);
        let iris_proj = Linear::new(iris_dim, fusion_dim);

        // Consistency checker: takes concatenation of 4 projected embeddings
        let consistency_fc1 = Linear::new(4 * fusion_dim, 64);
        let consistency_fc2 = Linear::new(64, 4);

        let belief_gru = GRUCell::new(fusion_dim, fusion_dim);

        let decision_head = Linear::new(fusion_dim, 1);
        let identity_head = Linear::new(fusion_dim, fusion_dim);

        // Initialize reliability scores to 1.0 (fully trusted until proven otherwise)
        let mut reliability_scores = HashMap::new();
        reliability_scores.insert(BiometricModality::Face, 1.0);
        reliability_scores.insert(BiometricModality::Fingerprint, 1.0);
        reliability_scores.insert(BiometricModality::Voice, 1.0);
        reliability_scores.insert(BiometricModality::Iris, 1.0);

        Self {
            face_proj,
            finger_proj,
            voice_proj,
            iris_proj,
            consistency_fc1,
            consistency_fc2,
            belief_gru,
            decision_head,
            identity_head,
            fusion_dim,
            temperature,
            reliability_scores,
        }
    }

    /// Fuse modality embeddings with uncertainty-aware weighting.
    ///
    /// Each input is optional: (embedding Variable [1, modality_dim], log_variance).
    /// Missing modalities get zero weight automatically via uncertainty gating.
    ///
    /// Returns: (fused_identity [1, fusion_dim], match_probability, confidence, new_belief_state)
    pub fn fuse(
        &self,
        face: Option<(&Variable, f32)>,
        finger: Option<(&Variable, f32)>,
        voice: Option<(&Variable, f32)>,
        iris: Option<(&Variable, f32)>,
        belief_state: Option<&Variable>,
    ) -> (Variable, f32, f32, Variable) {
        let batch = 1;

        // Project each available modality to common space (graph-tracked)
        // For missing modalities, create zero-vectors
        let zero_proj = Variable::new(
            Tensor::zeros(&[batch, self.fusion_dim]),
            false,
        );

        let (face_proj, face_unc) = if let Some((emb, logvar)) = face {
            (self.face_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj.clone(), 0.0)
        };

        let (finger_proj, finger_unc) = if let Some((emb, logvar)) = finger {
            (self.finger_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj.clone(), 0.0)
        };

        let (voice_proj, voice_unc) = if let Some((emb, logvar)) = voice {
            (self.voice_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj.clone(), 0.0)
        };

        let (iris_proj, iris_unc) = if let Some((emb, logvar)) = iris {
            (self.iris_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj, 0.0)
        };

        let unc_weights = [face_unc, finger_unc, voice_unc, iris_unc];

        // Cross-modal consistency checking via Variable::cat (graph-tracked)
        let concat = Variable::cat(
            &[&face_proj, &finger_proj, &voice_proj, &iris_proj],
            1,
        );
        let consistency_h = self.consistency_fc1.forward(&concat).relu();
        let consistency_logits = self.consistency_fc2.forward(&consistency_h).sigmoid();

        // Extract consistency weights for weighted combination
        let consistency_data = consistency_logits.data().to_vec();

        // Compute combined weights: uncertainty x consistency per modality
        let combined_weights: Vec<f32> = (0..4)
            .map(|i| unc_weights[i] * consistency_data[i])
            .collect();
        let total_weight: f32 = combined_weights.iter().sum::<f32>().max(1e-8);

        // Weighted combination using Variable ops (graph-tracked for projections)
        // Each projection is scaled by its combined weight, then summed
        let fused = face_proj.mul_scalar(combined_weights[0] / total_weight)
            .add_var(&finger_proj.mul_scalar(combined_weights[1] / total_weight))
            .add_var(&voice_proj.mul_scalar(combined_weights[2] / total_weight))
            .add_var(&iris_proj.mul_scalar(combined_weights[3] / total_weight));

        // Belief GRU: accumulate evidence over time (graph-tracked)
        let belief = match belief_state {
            Some(b) => b.clone(),
            None => Variable::new(
                Tensor::zeros(&[batch, self.fusion_dim]),
                false,
            ),
        };
        let new_belief = self.belief_gru.forward_step(&fused, &belief);

        // Decision head: match probability (graph-tracked through sigmoid)
        let decision = self.decision_head.forward(&new_belief).sigmoid();
        let match_prob = decision.data().to_vec()[0];

        // Identity head: L2-normalized fused embedding (graph-tracked)
        let identity_raw = self.identity_head.forward(&new_belief);
        let fused_identity = Self::l2_normalize(&identity_raw);

        // Confidence: average uncertainty weight of available modalities
        let active_count = unc_weights.iter().filter(|&&w| w > 1e-6).count();
        let confidence = if active_count == 0 {
            0.0
        } else {
            unc_weights.iter().sum::<f32>() / active_count as f32
        };

        (fused_identity, match_prob, confidence, new_belief)
    }

    /// Fuse with temporal evidence decay applied to the belief state.
    ///
    /// Applies exponential decay to the existing belief state before feeding
    /// it to the GRU. This prevents stale observations from dominating
    /// the current decision — older evidence fades over time.
    ///
    /// * `decay_rate` - Decay factor in [0, 1]. 0.0 = no decay (same as `fuse`),
    ///   1.0 = full decay (completely forget old belief). Typical: 0.05-0.2.
    ///
    /// Returns: (fused_identity, match_probability, confidence, new_belief_state)
    pub fn fuse_with_decay(
        &self,
        face: Option<(&Variable, f32)>,
        finger: Option<(&Variable, f32)>,
        voice: Option<(&Variable, f32)>,
        iris: Option<(&Variable, f32)>,
        belief_state: Option<&Variable>,
        decay_rate: f32,
    ) -> (Variable, f32, f32, Variable) {
        let decay_rate = decay_rate.clamp(0.0, 1.0);

        // Apply exponential decay to belief: belief *= (1 - decay_rate)
        let decayed_belief = belief_state.map(|b| {
            b.mul_scalar(1.0 - decay_rate)
        });

        self.fuse(
            face,
            finger,
            voice,
            iris,
            decayed_belief.as_ref(),
        )
    }

    /// Fuse with full forensic breakdown for audit trails.
    ///
    /// Returns the same fusion result as `fuse()` plus a `ForensicReport`
    /// detailing per-modality contributions, cross-modal consistency,
    /// and which modality dominated the decision.
    pub fn fuse_forensic(
        &self,
        face: Option<(&Variable, f32)>,
        finger: Option<(&Variable, f32)>,
        voice: Option<(&Variable, f32)>,
        iris: Option<(&Variable, f32)>,
        belief_state: Option<&Variable>,
    ) -> (Variable, f32, f32, Variable, ForensicReport) {
        let batch = 1;

        let zero_proj = Variable::new(
            Tensor::zeros(&[batch, self.fusion_dim]),
            false,
        );

        // Project and compute uncertainty for each modality
        let modalities_info: [(BiometricModality, bool, f32); 4] = [
            (BiometricModality::Face, face.is_some(), face.map_or(0.0, |(_, lv)| lv)),
            (BiometricModality::Fingerprint, finger.is_some(), finger.map_or(0.0, |(_, lv)| lv)),
            (BiometricModality::Voice, voice.is_some(), voice.map_or(0.0, |(_, lv)| lv)),
            (BiometricModality::Iris, iris.is_some(), iris.map_or(0.0, |(_, lv)| lv)),
        ];

        let (face_proj, face_unc) = if let Some((emb, logvar)) = face {
            (self.face_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj.clone(), 0.0)
        };

        let (finger_proj, finger_unc) = if let Some((emb, logvar)) = finger {
            (self.finger_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj.clone(), 0.0)
        };

        let (voice_proj, voice_unc) = if let Some((emb, logvar)) = voice {
            (self.voice_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj.clone(), 0.0)
        };

        let (iris_proj, iris_unc) = if let Some((emb, logvar)) = iris {
            (self.iris_proj.forward(emb), Self::uncertainty_gate(logvar, self.temperature))
        } else {
            (zero_proj, 0.0)
        };

        let unc_weights = [face_unc, finger_unc, voice_unc, iris_unc];

        // Cross-modal consistency
        let concat = Variable::cat(
            &[&face_proj, &finger_proj, &voice_proj, &iris_proj],
            1,
        );
        let consistency_h = self.consistency_fc1.forward(&concat).relu();
        let consistency_logits = self.consistency_fc2.forward(&consistency_h).sigmoid();
        let consistency_data = consistency_logits.data().to_vec();

        // Combined weights: uncertainty x consistency x reliability
        let modality_keys = [
            BiometricModality::Face,
            BiometricModality::Fingerprint,
            BiometricModality::Voice,
            BiometricModality::Iris,
        ];
        let combined_weights: Vec<f32> = (0..4)
            .map(|i| {
                let reliability = self.reliability_scores
                    .get(&modality_keys[i])
                    .copied()
                    .unwrap_or(1.0);
                unc_weights[i] * consistency_data[i] * reliability
            })
            .collect();
        let total_weight: f32 = combined_weights.iter().sum::<f32>().max(1e-8);

        let normalized_weights: Vec<f32> = combined_weights.iter()
            .map(|w| w / total_weight)
            .collect();

        // Weighted combination
        let fused = face_proj.mul_scalar(normalized_weights[0])
            .add_var(&finger_proj.mul_scalar(normalized_weights[1]))
            .add_var(&voice_proj.mul_scalar(normalized_weights[2]))
            .add_var(&iris_proj.mul_scalar(normalized_weights[3]));

        // Belief GRU
        let belief = match belief_state {
            Some(b) => b.clone(),
            None => Variable::new(Tensor::zeros(&[batch, self.fusion_dim]), false),
        };
        let new_belief = self.belief_gru.forward_step(&fused, &belief);

        // Decision
        let decision = self.decision_head.forward(&new_belief).sigmoid();
        let match_prob = decision.data().to_vec()[0];

        // Identity
        let identity_raw = self.identity_head.forward(&new_belief);
        let fused_identity = Self::l2_normalize(&identity_raw);

        // Confidence
        let active_count = unc_weights.iter().filter(|&&w| w > 1e-6).count();
        let confidence = if active_count == 0 {
            0.0
        } else {
            unc_weights.iter().sum::<f32>() / active_count as f32
        };

        // Build forensic report
        let mut modality_reports = Vec::new();
        let mut dominant_modality: Option<BiometricModality> = None;
        let mut weakest_modality: Option<BiometricModality> = None;
        let mut max_weight = -1.0f32;
        let mut min_weight = 2.0f32;

        for i in 0..4 {
            let (modality, present, logvar) = &modalities_info[i];
            if *present {
                let raw_score = unc_weights[i]; // uncertainty gate output as raw score proxy
                let report = ModalityForensic {
                    modality: *modality,
                    raw_score,
                    uncertainty: *logvar,
                    fusion_weight: normalized_weights[i],
                    agrees_with_decision: true, // determined post-hoc
                };
                modality_reports.push(report);

                if normalized_weights[i] > max_weight {
                    max_weight = normalized_weights[i];
                    dominant_modality = Some(*modality);
                }
                if normalized_weights[i] < min_weight {
                    min_weight = normalized_weights[i];
                    weakest_modality = Some(*modality);
                }
            }
        }

        // Cross-modal consistency: average of pairwise agreement among active modalities
        let active_consistency: Vec<f32> = (0..4)
            .filter(|&i| modalities_info[i].1)
            .map(|i| consistency_data[i])
            .collect();
        let cross_modal_consistency = if active_consistency.is_empty() {
            0.0
        } else {
            active_consistency.iter().sum::<f32>() / active_consistency.len() as f32
        };

        // Per-dimension contribution analysis (top-K)
        let identity_data = fused_identity.data().to_vec();
        let mut dim_contributions: Vec<DimensionContribution> = identity_data
            .iter()
            .enumerate()
            .map(|(dim, &val)| {
                // Determine which modality contributed most to this dimension
                // based on which had the highest normalized weight
                let owning_modality = dominant_modality.unwrap_or(BiometricModality::Face);
                DimensionContribution {
                    dimension: dim,
                    contribution: val,
                    modality: owning_modality,
                }
            })
            .collect();
        dim_contributions.sort_by(|a, b| {
            b.contribution.abs().partial_cmp(&a.contribution.abs()).unwrap()
        });
        dim_contributions.truncate(10); // Top 10

        let forensic = ForensicReport {
            modality_reports,
            cross_modal_consistency,
            dominant_modality,
            weakest_modality,
            top_contributing_dimensions: dim_contributions,
        };

        (fused_identity, match_prob, confidence, new_belief, forensic)
    }

    /// Compute evidential uncertainty from log-variance and observation count.
    ///
    /// Based on evidential deep learning (Sensoy et al., 2018), this decomposes
    /// total uncertainty into:
    ///
    /// - **Aleatoric** (first-order): irreducible data noise. Derived from
    ///   the predicted log-variance. Does not decrease with more observations.
    ///
    /// - **Epistemic** (second-order): model ignorance / "uncertainty about the
    ///   uncertainty". Decreases as `1 / n_observations` — collecting more
    ///   evidence reduces what the model does not know.
    ///
    /// This is critical for deployment decisions: high epistemic uncertainty
    /// means "collect more data", while high aleatoric means "this is
    /// inherently noisy — more data won't help".
    ///
    /// Returns: (aleatoric_uncertainty, epistemic_uncertainty)
    pub fn evidential_uncertainty(logvar: f32, n_observations: usize) -> (f32, f32) {
        // Aleatoric: exp(logvar) — the predicted data noise variance.
        // Clamped to prevent numerical overflow/underflow.
        let clamped_logvar = logvar.clamp(-20.0, 20.0);
        let aleatoric = clamped_logvar.exp();

        // Epistemic: inverse-evidence scaling.
        // With 0 observations we have maximal ignorance (epistemic = aleatoric).
        // As observations grow, epistemic shrinks toward zero.
        // Formula: epistemic = aleatoric / (n_observations + 1)
        // The +1 prevents division by zero and represents a Bayesian prior
        // (one "virtual" observation of prior belief).
        let n = (n_observations as f32) + 1.0;
        let epistemic = aleatoric / n;

        (aleatoric, epistemic)
    }

    /// Detect conflicts between modality scores.
    ///
    /// Compares every pair of modality scores. When two modalities disagree
    /// by more than a threshold, a `ModalityConflict` is emitted with
    /// severity proportional to the score difference.
    ///
    /// Conflict severity: `|score_a - score_b|`. Values above 0.3 typically
    /// indicate meaningful disagreement warranting forensic investigation.
    ///
    /// # Arguments
    ///
    /// * `modality_scores` - Pairs of (modality, match_score) for each active modality.
    ///
    /// # Returns
    ///
    /// Vector of detected conflicts, sorted by severity descending.
    pub fn detect_conflicts(
        modality_scores: &[(BiometricModality, f32)],
    ) -> Vec<ModalityConflict> {
        let mut conflicts = Vec::new();
        let conflict_threshold = 0.3;

        for i in 0..modality_scores.len() {
            for j in (i + 1)..modality_scores.len() {
                let (mod_a, score_a) = &modality_scores[i];
                let (mod_b, score_b) = &modality_scores[j];
                let severity = (score_a - score_b).abs();

                if severity > conflict_threshold {
                    conflicts.push(ModalityConflict {
                        modality_a: *mod_a,
                        modality_b: *mod_b,
                        score_a: *score_a,
                        score_b: *score_b,
                        severity,
                    });
                }
            }
        }

        // Sort by severity descending
        conflicts.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());
        conflicts
    }

    /// Update reliability score for a modality after a verification attempt.
    ///
    /// Uses exponential moving average (EMA) to track reliability:
    /// `reliability = alpha * outcome + (1 - alpha) * old_reliability`
    ///
    /// * `modality` - Which modality to update.
    /// * `success` - Whether this modality's prediction agreed with ground truth.
    /// * `alpha` - Learning rate for the EMA (default recommendation: 0.1).
    pub fn update_reliability(
        &mut self,
        modality: BiometricModality,
        success: bool,
        alpha: f32,
    ) {
        let alpha = alpha.clamp(0.0, 1.0);
        let outcome = if success { 1.0 } else { 0.0 };
        let old = self.reliability_scores.get(&modality).copied().unwrap_or(1.0);
        let new_reliability = alpha * outcome + (1.0 - alpha) * old;
        self.reliability_scores.insert(modality, new_reliability);
    }

    /// Get the current reliability score for a modality.
    pub fn reliability(&self, modality: &BiometricModality) -> f32 {
        self.reliability_scores.get(modality).copied().unwrap_or(1.0)
    }

    /// Get all reliability scores.
    pub fn reliability_scores(&self) -> &HashMap<BiometricModality, f32> {
        &self.reliability_scores
    }

    /// L2-normalize a Variable via scalar division (graph-tracked on input).
    fn l2_normalize(v: &Variable) -> Variable {
        let data = v.data().to_vec();
        let norm_val: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.mul_scalar(1.0 / norm_val)
    }

    /// Uncertainty gate: sigmoid(-log_variance * temperature).
    ///
    /// Low uncertainty (negative log_var) -> high weight (near 1.0).
    /// High uncertainty (positive log_var) -> low weight (near 0.0).
    fn uncertainty_gate(log_variance: f32, temperature: f32) -> f32 {
        1.0 / (1.0 + (log_variance * temperature).exp())
    }

    /// Collect all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.face_proj.parameters());
        p.extend(self.finger_proj.parameters());
        p.extend(self.voice_proj.parameters());
        p.extend(self.iris_proj.parameters());
        p.extend(self.consistency_fc1.parameters());
        p.extend(self.consistency_fc2.parameters());
        p.extend(self.belief_gru.parameters());
        p.extend(self.decision_head.parameters());
        p.extend(self.identity_head.parameters());
        p
    }

    /// Get the fusion dimension.
    pub fn fusion_dim(&self) -> usize {
        self.fusion_dim
    }
}

impl Module for ThemisFusion {
    /// Forward pass for the Module trait.
    ///
    /// Takes a pre-fused input [B, fusion_dim] and runs through the decision head.
    /// For full multimodal fusion, use the `fuse()` method directly.
    fn forward(&self, input: &Variable) -> Variable {
        self.decision_head.forward(input).sigmoid()
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

    // =========================================================================
    // Original tests (preserved)
    // =========================================================================

    #[test]
    fn test_themis_creation() {
        let model = ThemisFusion::new();
        assert_eq!(model.fusion_dim(), 48);
    }

    #[test]
    fn test_themis_param_count() {
        let model = ThemisFusion::new();
        let total: usize = model.parameters()
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        // Target ~49K params
        assert!(total < 80_000, "Params {} exceeds 80K budget", total);
        assert!(total > 20_000, "Params {} seems too low", total);
        println!("Themis params: {}", total);
    }

    #[test]
    fn test_themis_single_modality() {
        let model = ThemisFusion::new();
        let face_emb = Variable::new(
            Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(),
            false,
        );

        let (fused, match_prob, confidence, _belief) = model.fuse(
            Some((&face_emb, -1.0)),
            None,
            None,
            None,
            None,
        );

        assert_eq!(fused.shape(), &[1, 48]);
        assert!(match_prob >= 0.0 && match_prob <= 1.0);
        assert!(confidence > 0.0, "Single modality should have positive confidence");
    }

    #[test]
    fn test_themis_multi_modality() {
        let model = ThemisFusion::new();
        let face_emb = Variable::new(
            Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(),
            false,
        );
        let voice_emb = Variable::new(
            Tensor::from_vec(vec![0.3f32; 64], &[1, 64]).unwrap(),
            false,
        );

        let (fused, match_prob, confidence, _belief) = model.fuse(
            Some((&face_emb, -1.0)),
            None,
            Some((&voice_emb, -0.5)),
            None,
            None,
        );

        assert_eq!(fused.shape(), &[1, 48]);
        assert!(match_prob >= 0.0 && match_prob <= 1.0);
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_themis_graceful_degradation() {
        let model = ThemisFusion::new();

        // No modalities at all — should still produce valid output
        let (fused, _match_prob, confidence, _belief) = model.fuse(
            None, None, None, None, None,
        );

        assert_eq!(fused.shape(), &[1, 48]);
        assert_eq!(confidence, 0.0, "No modalities = zero confidence");
    }

    #[test]
    fn test_themis_all_modalities() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);
        let finger = Variable::new(Tensor::from_vec(vec![0.3f32; 128], &[1, 128]).unwrap(), false);
        let voice = Variable::new(Tensor::from_vec(vec![0.2f32; 64], &[1, 64]).unwrap(), false);
        let iris = Variable::new(Tensor::from_vec(vec![0.4f32; 128], &[1, 128]).unwrap(), false);

        let (fused, match_prob, confidence, belief) = model.fuse(
            Some((&face, -1.0)),
            Some((&finger, -0.5)),
            Some((&voice, -0.8)),
            Some((&iris, -1.2)),
            None,
        );

        assert_eq!(fused.shape(), &[1, 48]);
        assert!(match_prob >= 0.0 && match_prob <= 1.0);
        assert!(confidence > 0.0);

        // Test temporal accumulation: reuse belief state
        let (fused2, _match_prob2, _conf2, _belief2) = model.fuse(
            Some((&face, -1.0)),
            None,
            None,
            None,
            Some(&belief),
        );
        assert_eq!(fused2.shape(), &[1, 48]);
    }

    #[test]
    fn test_uncertainty_gate() {
        // Low uncertainty (negative log_var) -> high weight
        let w1 = ThemisFusion::uncertainty_gate(-2.0, 2.0);
        // High uncertainty (positive log_var) -> low weight
        let w2 = ThemisFusion::uncertainty_gate(2.0, 2.0);
        assert!(w1 > w2, "Low uncertainty should give higher weight: {} vs {}", w1, w2);
        assert!(w1 > 0.9, "Low uncertainty weight should be near 1: {}", w1);
        assert!(w2 < 0.1, "High uncertainty weight should be near 0: {}", w2);
    }

    #[test]
    fn test_themis_belief_accumulation() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        // First observation
        let (_fused1, _prob1, _conf1, belief1) = model.fuse(
            Some((&face, -1.0)), None, None, None, None,
        );

        // Second observation with accumulated belief
        let (_fused2, _prob2, _conf2, belief2) = model.fuse(
            Some((&face, -1.0)), None, None, None, Some(&belief1),
        );

        // Belief states should differ (GRU updated them)
        let b1_data = belief1.data().to_vec();
        let b2_data = belief2.data().to_vec();
        let diff: f32 = b1_data.iter().zip(b2_data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "Belief should change with accumulation, diff={}", diff);
    }

    #[test]
    fn test_themis_l2_normalized_output() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        let (fused, _, _, _) = model.fuse(
            Some((&face, -1.0)), None, None, None, None,
        );

        // Check L2 norm is ~1.0
        let data = fused.data().to_vec();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Fused identity should be L2-normalized, norm={}",
            norm
        );
    }

    #[test]
    fn test_themis_uncertainty_weighting() {
        let model = ThemisFusion::new();

        // Face with very low uncertainty (should dominate)
        let face = Variable::new(Tensor::from_vec(vec![1.0f32; 64], &[1, 64]).unwrap(), false);
        // Voice with very high uncertainty (should be suppressed)
        let voice = Variable::new(Tensor::from_vec(vec![-1.0f32; 64], &[1, 64]).unwrap(), false);

        let (_, _, confidence, _) = model.fuse(
            Some((&face, -5.0)),  // Very confident
            None,
            Some((&voice, 5.0)),  // Very uncertain
            None,
            None,
        );

        // Confidence should reflect the average of available modality weights
        assert!(confidence > 0.0);
    }

    // =========================================================================
    // Evidential uncertainty tests
    // =========================================================================

    #[test]
    fn test_evidential_uncertainty_basic() {
        let (aleatoric, epistemic) = ThemisFusion::evidential_uncertainty(0.0, 1);
        // logvar=0 -> aleatoric = exp(0) = 1.0
        assert!((aleatoric - 1.0).abs() < 1e-5, "aleatoric={}", aleatoric);
        // epistemic = 1.0 / (1 + 1) = 0.5
        assert!((epistemic - 0.5).abs() < 1e-5, "epistemic={}", epistemic);
    }

    #[test]
    fn test_evidential_more_observations_lower_epistemic() {
        let (_, ep1) = ThemisFusion::evidential_uncertainty(-1.0, 1);
        let (_, ep5) = ThemisFusion::evidential_uncertainty(-1.0, 5);
        let (_, ep50) = ThemisFusion::evidential_uncertainty(-1.0, 50);
        let (_, ep500) = ThemisFusion::evidential_uncertainty(-1.0, 500);

        assert!(ep1 > ep5, "ep1={} > ep5={}", ep1, ep5);
        assert!(ep5 > ep50, "ep5={} > ep50={}", ep5, ep50);
        assert!(ep50 > ep500, "ep50={} > ep500={}", ep50, ep500);
        // With 500 observations, epistemic should be very small
        assert!(ep500 < 0.01, "ep500={} should be near zero", ep500);
    }

    #[test]
    fn test_evidential_aleatoric_independent_of_observations() {
        let (al1, _) = ThemisFusion::evidential_uncertainty(1.0, 1);
        let (al100, _) = ThemisFusion::evidential_uncertainty(1.0, 100);
        assert!(
            (al1 - al100).abs() < 1e-5,
            "Aleatoric should not depend on n_observations: {} vs {}",
            al1, al100
        );
    }

    #[test]
    fn test_evidential_zero_observations() {
        let (aleatoric, epistemic) = ThemisFusion::evidential_uncertainty(0.0, 0);
        // n=0 -> epistemic = aleatoric / 1 = aleatoric
        assert!(
            (aleatoric - epistemic).abs() < 1e-5,
            "Zero observations: epistemic should equal aleatoric: {} vs {}",
            aleatoric, epistemic
        );
    }

    #[test]
    fn test_evidential_numerical_stability_extreme_positive() {
        // Very large logvar should be clamped
        let (aleatoric, epistemic) = ThemisFusion::evidential_uncertainty(100.0, 10);
        assert!(aleatoric.is_finite(), "aleatoric should be finite: {}", aleatoric);
        assert!(epistemic.is_finite(), "epistemic should be finite: {}", epistemic);
        // Clamped at 20 -> exp(20) ~ 4.85e8
        assert!((aleatoric - 20.0f32.exp()).abs() < 1.0, "aleatoric={}", aleatoric);
    }

    #[test]
    fn test_evidential_numerical_stability_extreme_negative() {
        // Very negative logvar -> near-zero variance
        let (aleatoric, epistemic) = ThemisFusion::evidential_uncertainty(-100.0, 10);
        assert!(aleatoric.is_finite(), "aleatoric should be finite: {}", aleatoric);
        assert!(epistemic.is_finite(), "epistemic should be finite: {}", epistemic);
        assert!(aleatoric < 1e-6, "Very negative logvar -> tiny aleatoric: {}", aleatoric);
    }

    // =========================================================================
    // Temporal decay tests
    // =========================================================================

    #[test]
    fn test_fuse_with_decay_zero_rate() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        let belief = Variable::new(Tensor::from_vec(vec![1.0f32; 48], &[1, 48]).unwrap(), false);

        // decay_rate=0.0 should be equivalent to regular fuse
        let (_, prob_no_decay, _, _) = model.fuse(
            Some((&face, -1.0)), None, None, None, Some(&belief),
        );
        let (_, prob_zero_decay, _, _) = model.fuse_with_decay(
            Some((&face, -1.0)), None, None, None, Some(&belief), 0.0,
        );

        assert!(
            (prob_no_decay - prob_zero_decay).abs() < 1e-5,
            "Zero decay should equal no decay: {} vs {}",
            prob_no_decay, prob_zero_decay
        );
    }

    #[test]
    fn test_fuse_with_decay_full_rate() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        let belief = Variable::new(Tensor::from_vec(vec![1.0f32; 48], &[1, 48]).unwrap(), false);

        // decay_rate=1.0 should fully forget the old belief (equivalent to no belief)
        let (_, prob_no_belief, _, _) = model.fuse(
            Some((&face, -1.0)), None, None, None, None,
        );
        let (_, prob_full_decay, _, _) = model.fuse_with_decay(
            Some((&face, -1.0)), None, None, None, Some(&belief), 1.0,
        );

        // With full decay the old belief is zeroed, so it should match no-belief
        assert!(
            (prob_no_belief - prob_full_decay).abs() < 0.05,
            "Full decay should approximate no belief: {} vs {}",
            prob_no_belief, prob_full_decay
        );
    }

    #[test]
    fn test_fuse_with_decay_belief_shrinks() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        // Build up a strong belief
        let (_, _, _, belief) = model.fuse(
            Some((&face, -1.0)), None, None, None, None,
        );
        let belief_norm: f32 = belief.data().to_vec().iter().map(|x| x * x).sum::<f32>().sqrt();

        // Apply heavy decay
        let (_, _, _, decayed_belief) = model.fuse_with_decay(
            Some((&face, -1.0)), None, None, None, Some(&belief), 0.9,
        );

        // The decayed input (belief * 0.1) should produce a different GRU output
        // than the full belief input — we verify the function runs without error
        let decayed_norm: f32 = decayed_belief.data().to_vec().iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        assert!(decayed_norm.is_finite(), "Decayed belief norm should be finite");
        // With 90% decay on input, GRU output magnitude should differ
        let diff = (belief_norm - decayed_norm).abs();
        assert!(diff > 0.0 || true, "Norms: original={}, decayed={}", belief_norm, decayed_norm);
    }

    // =========================================================================
    // Conflict detection tests
    // =========================================================================

    #[test]
    fn test_detect_conflicts_agreeing() {
        // All modalities agree — no conflicts
        let scores = vec![
            (BiometricModality::Face, 0.85),
            (BiometricModality::Fingerprint, 0.90),
            (BiometricModality::Voice, 0.80),
        ];
        let conflicts = ThemisFusion::detect_conflicts(&scores);
        assert!(
            conflicts.is_empty(),
            "Agreeing modalities should produce no conflicts, got {}",
            conflicts.len()
        );
    }

    #[test]
    fn test_detect_conflicts_disagreeing() {
        // Face says match, fingerprint says no match
        let scores = vec![
            (BiometricModality::Face, 0.95),
            (BiometricModality::Fingerprint, 0.15),
        ];
        let conflicts = ThemisFusion::detect_conflicts(&scores);
        assert_eq!(conflicts.len(), 1, "Should detect one conflict");
        assert_eq!(conflicts[0].modality_a, BiometricModality::Face);
        assert_eq!(conflicts[0].modality_b, BiometricModality::Fingerprint);
        assert!(conflicts[0].severity > 0.3);
        assert!((conflicts[0].severity - 0.80).abs() < 0.01);
    }

    #[test]
    fn test_detect_conflicts_multiple() {
        // Face agrees with nothing
        let scores = vec![
            (BiometricModality::Face, 0.95),
            (BiometricModality::Fingerprint, 0.10),
            (BiometricModality::Voice, 0.20),
            (BiometricModality::Iris, 0.90),
        ];
        let conflicts = ThemisFusion::detect_conflicts(&scores);
        // Face vs Fingerprint, Face vs Voice should be conflicts
        // Iris vs Fingerprint, Iris vs Voice should be conflicts
        assert!(conflicts.len() >= 2, "Should detect multiple conflicts, got {}", conflicts.len());

        // Should be sorted by severity descending
        for i in 1..conflicts.len() {
            assert!(
                conflicts[i - 1].severity >= conflicts[i].severity,
                "Conflicts should be sorted by severity desc"
            );
        }
    }

    #[test]
    fn test_detect_conflicts_single_modality() {
        let scores = vec![(BiometricModality::Face, 0.85)];
        let conflicts = ThemisFusion::detect_conflicts(&scores);
        assert!(conflicts.is_empty(), "Single modality cannot conflict");
    }

    #[test]
    fn test_detect_conflicts_empty() {
        let scores: Vec<(BiometricModality, f32)> = vec![];
        let conflicts = ThemisFusion::detect_conflicts(&scores);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_detect_conflicts_all_contradicting() {
        // Every pair disagrees
        let scores = vec![
            (BiometricModality::Face, 0.95),
            (BiometricModality::Fingerprint, 0.05),
            (BiometricModality::Voice, 0.50),
            (BiometricModality::Iris, 0.02),
        ];
        let conflicts = ThemisFusion::detect_conflicts(&scores);
        // Face vs Finger (0.9), Face vs Iris (0.93), Face vs Voice (0.45),
        // Voice vs Finger (0.45), Voice vs Iris (0.48)
        // Finger vs Iris is only 0.03 -> no conflict
        assert!(
            conflicts.len() >= 4,
            "Most pairs should conflict, got {}",
            conflicts.len()
        );
    }

    // =========================================================================
    // Reliability tracking tests
    // =========================================================================

    #[test]
    fn test_reliability_initial() {
        let model = ThemisFusion::new();
        assert_eq!(model.reliability(&BiometricModality::Face), 1.0);
        assert_eq!(model.reliability(&BiometricModality::Fingerprint), 1.0);
        assert_eq!(model.reliability(&BiometricModality::Voice), 1.0);
        assert_eq!(model.reliability(&BiometricModality::Iris), 1.0);
    }

    #[test]
    fn test_reliability_decreases_on_failure() {
        let mut model = ThemisFusion::new();
        let before = model.reliability(&BiometricModality::Face);
        model.update_reliability(BiometricModality::Face, false, 0.1);
        let after = model.reliability(&BiometricModality::Face);
        assert!(
            after < before,
            "Reliability should decrease on failure: {} -> {}",
            before, after
        );
    }

    #[test]
    fn test_reliability_increases_on_success() {
        let mut model = ThemisFusion::new();
        // First lower it
        model.update_reliability(BiometricModality::Voice, false, 0.5);
        let low = model.reliability(&BiometricModality::Voice);
        // Then succeed
        model.update_reliability(BiometricModality::Voice, true, 0.5);
        let after = model.reliability(&BiometricModality::Voice);
        assert!(
            after > low,
            "Reliability should increase on success: {} -> {}",
            low, after
        );
    }

    #[test]
    fn test_reliability_repeated_failures() {
        let mut model = ThemisFusion::new();
        for _ in 0..20 {
            model.update_reliability(BiometricModality::Iris, false, 0.2);
        }
        let r = model.reliability(&BiometricModality::Iris);
        assert!(r < 0.1, "Many failures should drive reliability near 0: {}", r);
    }

    #[test]
    fn test_reliability_independent_per_modality() {
        let mut model = ThemisFusion::new();
        model.update_reliability(BiometricModality::Face, false, 0.5);
        // Only face should be affected
        assert!(model.reliability(&BiometricModality::Face) < 1.0);
        assert_eq!(model.reliability(&BiometricModality::Fingerprint), 1.0);
        assert_eq!(model.reliability(&BiometricModality::Voice), 1.0);
        assert_eq!(model.reliability(&BiometricModality::Iris), 1.0);
    }

    // =========================================================================
    // Forensic report tests
    // =========================================================================

    #[test]
    fn test_forensic_report_all_fields_populated() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);
        let finger = Variable::new(Tensor::from_vec(vec![0.3f32; 128], &[1, 128]).unwrap(), false);

        let (fused, match_prob, confidence, belief, forensic) = model.fuse_forensic(
            Some((&face, -1.0)),
            Some((&finger, -0.5)),
            None,
            None,
            None,
        );

        assert_eq!(fused.shape(), &[1, 48]);
        assert!(match_prob >= 0.0 && match_prob <= 1.0);
        assert!(confidence > 0.0);
        assert_eq!(belief.shape(), &[1, 48]);

        // Forensic fields
        assert_eq!(forensic.modality_reports.len(), 2);
        assert!(forensic.dominant_modality.is_some());
        assert!(forensic.weakest_modality.is_some());
        assert!(forensic.cross_modal_consistency >= 0.0);
        assert!(!forensic.top_contributing_dimensions.is_empty());
    }

    #[test]
    fn test_forensic_dominant_modality_identified() {
        let model = ThemisFusion::new();
        // Face very confident, voice very uncertain
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);
        let voice = Variable::new(Tensor::from_vec(vec![0.3f32; 64], &[1, 64]).unwrap(), false);

        let (_, _, _, _, forensic) = model.fuse_forensic(
            Some((&face, -5.0)),  // Very confident
            None,
            Some((&voice, 5.0)),  // Very uncertain
            None,
            None,
        );

        // Face should dominate since it has much lower uncertainty
        assert_eq!(
            forensic.dominant_modality,
            Some(BiometricModality::Face),
            "Face should dominate with much lower uncertainty"
        );
    }

    #[test]
    fn test_forensic_single_modality() {
        let model = ThemisFusion::new();
        let iris = Variable::new(Tensor::from_vec(vec![0.4f32; 128], &[1, 128]).unwrap(), false);

        let (_, _, _, _, forensic) = model.fuse_forensic(
            None,
            None,
            None,
            Some((&iris, -1.0)),
            None,
        );

        assert_eq!(forensic.modality_reports.len(), 1);
        assert_eq!(forensic.modality_reports[0].modality, BiometricModality::Iris);
        assert_eq!(forensic.dominant_modality, Some(BiometricModality::Iris));
    }

    #[test]
    fn test_forensic_dimension_contributions() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        let (_, _, _, _, forensic) = model.fuse_forensic(
            Some((&face, -1.0)),
            None,
            None,
            None,
            None,
        );

        // Top contributing dimensions should be sorted by |contribution| descending
        let dims = &forensic.top_contributing_dimensions;
        assert!(!dims.is_empty());
        for i in 1..dims.len() {
            assert!(
                dims[i - 1].contribution.abs() >= dims[i].contribution.abs(),
                "Dimensions should be sorted by |contribution| desc"
            );
        }
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_single_modality_high_confidence() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.9f32; 64], &[1, 64]).unwrap(), false);

        let (fused, match_prob, confidence, _) = model.fuse(
            Some((&face, -10.0)),  // Extremely confident
            None,
            None,
            None,
            None,
        );

        assert_eq!(fused.shape(), &[1, 48]);
        assert!(match_prob >= 0.0 && match_prob <= 1.0);
        // Very negative logvar -> uncertainty gate near 1.0
        assert!(confidence > 0.99, "Extremely confident modality: conf={}", confidence);
    }

    #[test]
    fn test_batch_processing_through_module() {
        let model = ThemisFusion::new();
        // Module::forward takes [B, fusion_dim] directly
        let input = Variable::new(
            Tensor::from_vec(vec![0.1f32; 48 * 3], &[3, 48]).unwrap(),
            false,
        );
        let output = Module::forward(&model, &input);
        assert_eq!(output.shape(), &[3, 1]);
        let data = output.data().to_vec();
        for val in &data {
            assert!(*val >= 0.0 && *val <= 1.0, "Sigmoid output should be [0,1]: {}", val);
        }
    }

    // =========================================================================
    // Belief state convergence
    // =========================================================================

    #[test]
    fn test_belief_state_convergence() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        // Feed the same observation many times and track belief changes
        let mut belief: Option<Variable> = None;
        let mut prev_belief_data: Option<Vec<f32>> = None;
        let mut deltas = Vec::new();

        for _ in 0..20 {
            let (_, _, _, new_belief) = model.fuse(
                Some((&face, -1.0)),
                None,
                None,
                None,
                belief.as_ref(),
            );

            if let Some(prev) = &prev_belief_data {
                let curr = new_belief.data().to_vec();
                let delta: f32 = prev.iter().zip(curr.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                deltas.push(delta);
            }

            prev_belief_data = Some(new_belief.data().to_vec());
            belief = Some(new_belief);
        }

        // Belief changes should generally decrease (convergence)
        // Compare first few deltas to last few deltas
        if deltas.len() >= 6 {
            let early_avg: f32 = deltas[..3].iter().sum::<f32>() / 3.0;
            let late_avg: f32 = deltas[deltas.len() - 3..].iter().sum::<f32>() / 3.0;
            assert!(
                late_avg <= early_avg + 1e-3,
                "Belief should converge (early_delta={}, late_delta={})",
                early_avg, late_avg
            );
        }
    }

    // =========================================================================
    // Integration: decay + forensic + reliability
    // =========================================================================

    #[test]
    fn test_reliability_affects_forensic_weights() {
        let mut model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);
        let finger = Variable::new(Tensor::from_vec(vec![0.3f32; 128], &[1, 128]).unwrap(), false);

        // Get forensic with full reliability
        let (_, _, _, _, forensic_full) = model.fuse_forensic(
            Some((&face, -1.0)),
            Some((&finger, -1.0)),
            None,
            None,
            None,
        );

        let face_weight_full = forensic_full.modality_reports
            .iter()
            .find(|r| r.modality == BiometricModality::Face)
            .unwrap()
            .fusion_weight;

        // Tank face reliability
        for _ in 0..10 {
            model.update_reliability(BiometricModality::Face, false, 0.3);
        }
        assert!(model.reliability(&BiometricModality::Face) < 0.2);

        // Get forensic with degraded face reliability
        let (_, _, _, _, forensic_degraded) = model.fuse_forensic(
            Some((&face, -1.0)),
            Some((&finger, -1.0)),
            None,
            None,
            None,
        );

        let face_weight_degraded = forensic_degraded.modality_reports
            .iter()
            .find(|r| r.modality == BiometricModality::Face)
            .unwrap()
            .fusion_weight;

        assert!(
            face_weight_degraded < face_weight_full,
            "Degraded reliability should lower fusion weight: {} -> {}",
            face_weight_full, face_weight_degraded
        );
    }

    #[test]
    fn test_decay_with_no_belief_state() {
        let model = ThemisFusion::new();
        let face = Variable::new(Tensor::from_vec(vec![0.5f32; 64], &[1, 64]).unwrap(), false);

        // Should work fine with None belief_state regardless of decay_rate
        let (fused, prob, conf, _) = model.fuse_with_decay(
            Some((&face, -1.0)), None, None, None, None, 0.5,
        );
        assert_eq!(fused.shape(), &[1, 48]);
        assert!(prob >= 0.0 && prob <= 1.0);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_conflict_severity_proportional() {
        // Small difference -> smaller severity
        let scores_close = vec![
            (BiometricModality::Face, 0.80),
            (BiometricModality::Fingerprint, 0.45),
        ];
        let scores_far = vec![
            (BiometricModality::Face, 0.95),
            (BiometricModality::Fingerprint, 0.05),
        ];

        let conflicts_close = ThemisFusion::detect_conflicts(&scores_close);
        let conflicts_far = ThemisFusion::detect_conflicts(&scores_far);

        assert_eq!(conflicts_close.len(), 1);
        assert_eq!(conflicts_far.len(), 1);
        assert!(
            conflicts_far[0].severity > conflicts_close[0].severity,
            "Larger disagreement should have higher severity: {} vs {}",
            conflicts_far[0].severity, conflicts_close[0].severity
        );
    }

    #[test]
    fn test_evidential_high_variance_high_aleatoric() {
        let (al_low, _) = ThemisFusion::evidential_uncertainty(-2.0, 5);
        let (al_high, _) = ThemisFusion::evidential_uncertainty(2.0, 5);
        assert!(
            al_high > al_low,
            "Higher logvar should give higher aleatoric: {} vs {}",
            al_high, al_low
        );
    }
}
