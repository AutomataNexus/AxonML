//! Aegis Identity — Novel Unified Biometric Framework
//!
//! Top-level module for the Aegis biometric suite. Re-exports all four modality
//! models (`MnemosyneIdentity` for face, `AriadneFingerprint` for fingerprint,
//! `EchoSpeaker` for voice, `ArgusIris` for iris) plus the `ThemisFusion`
//! uncertainty-aware fusion layer and `AegisIdentity`/`IdentityBank` for
//! enrollment, verification, and identification. Defines core types including
//! `BiometricModality`, `BiometricEvidence` (optional per-modality tensors with
//! metadata), `BiometricConfig` (embedding dims, thresholds, crystallization
//! steps), result structs for enrollment/verification/identification, liveness
//! detection, quality assessment, forensic analysis, identity drift tracking,
//! and FAR/FRR operating curves. Also provides utility functions for cosine
//! similarity, L2 normalization, Euclidean distance, weighted cosine similarity,
//! and distribution entropy.
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/mod.rs`
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

pub mod argus;
pub mod ariadne;
pub mod echo;
pub mod identity;
pub mod losses;
pub mod mnemosyne;
pub mod polar;
pub mod themis;

// =============================================================================
// Re-exports
// =============================================================================

pub use argus::ArgusIris;
pub use ariadne::AriadneFingerprint;
pub use echo::EchoSpeaker;
pub use identity::{AegisIdentity, IdentityBank};
pub use losses::{
    AngularMarginLoss, ArgusLoss, CenterLoss, ContrastiveLoss, CrystallizationLoss,
    DiversityRegularization, EchoLoss, LivenessLoss, ThemisLoss,
};
pub use mnemosyne::MnemosyneIdentity;
pub use themis::ThemisFusion;

// =============================================================================
// Types — Modality & Evidence
// =============================================================================

/// Biometric modality identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiometricModality {
    /// Face — temporal crystallization (Mnemosyne)
    Face,
    /// Fingerprint — ridge event fields (Ariadne)
    Fingerprint,
    /// Voice — predictive speaker residuals (Echo)
    Voice,
    /// Iris — radial phase encoding (Argus)
    Iris,
}

impl std::fmt::Display for BiometricModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BiometricModality::Face => write!(f, "Face (Mnemosyne)"),
            BiometricModality::Fingerprint => write!(f, "Fingerprint (Ariadne)"),
            BiometricModality::Voice => write!(f, "Voice (Echo)"),
            BiometricModality::Iris => write!(f, "Iris (Argus)"),
        }
    }
}

impl BiometricModality {
    /// All available modalities.
    pub fn all() -> Vec<Self> {
        vec![Self::Face, Self::Fingerprint, Self::Voice, Self::Iris]
    }

    /// Expected input tensor shape description.
    pub fn input_description(&self) -> &'static str {
        match self {
            Self::Face => "[B, 3, 64, 64] RGB face image",
            Self::Fingerprint => "[B, 1, 128, 128] grayscale fingerprint",
            Self::Voice => "[B, 40, T] mel spectrogram (variable length)",
            Self::Iris => "[B, 1, 32, 256] polar iris strip or [B, 1, H, W] raw",
        }
    }

    /// Approximate model parameter count for this modality.
    pub fn approx_params(&self) -> usize {
        match self {
            Self::Face => 43_000,
            Self::Fingerprint => 65_000,
            Self::Voice => 68_000,
            Self::Iris => 65_000,
        }
    }
}

/// Biometric evidence from one or more modalities.
///
/// Each field is optional — the system gracefully degrades when modalities
/// are missing (their uncertainty gates go to zero in Themis).
#[derive(Debug, Clone)]
pub struct BiometricEvidence {
    /// Face image tensor [B, 3, 64, 64]
    pub face: Option<axonml_autograd::Variable>,
    /// Fingerprint image tensor [B, 1, 128, 128]
    pub fingerprint: Option<axonml_autograd::Variable>,
    /// Voice mel spectrogram tensor [B, 40, T] (variable length)
    pub voice: Option<axonml_autograd::Variable>,
    /// Iris polar strip tensor [B, 1, 32, 256]
    pub iris: Option<axonml_autograd::Variable>,
    /// Optional sequence of face frames for temporal crystallization [N × [B, 3, 64, 64]]
    pub face_sequence: Option<Vec<axonml_autograd::Variable>>,
    /// Metadata: capture timestamp (seconds since epoch)
    pub timestamp: Option<f64>,
    /// Metadata: capture device identifier
    pub device_id: Option<String>,
}

impl BiometricEvidence {
    /// Create empty evidence (no modalities).
    pub fn empty() -> Self {
        Self {
            face: None,
            fingerprint: None,
            voice: None,
            iris: None,
            face_sequence: None,
            timestamp: None,
            device_id: None,
        }
    }

    /// Create face-only evidence.
    pub fn face(tensor: axonml_autograd::Variable) -> Self {
        Self {
            face: Some(tensor),
            ..Self::empty()
        }
    }

    /// Create face evidence with temporal sequence for crystallization.
    pub fn face_sequence(frames: Vec<axonml_autograd::Variable>) -> Self {
        let first = frames.first().cloned();
        Self {
            face: first,
            face_sequence: Some(frames),
            ..Self::empty()
        }
    }

    /// Create fingerprint-only evidence.
    pub fn fingerprint(tensor: axonml_autograd::Variable) -> Self {
        Self {
            fingerprint: Some(tensor),
            ..Self::empty()
        }
    }

    /// Create voice-only evidence.
    pub fn voice(tensor: axonml_autograd::Variable) -> Self {
        Self {
            voice: Some(tensor),
            ..Self::empty()
        }
    }

    /// Create iris-only evidence.
    pub fn iris(tensor: axonml_autograd::Variable) -> Self {
        Self {
            iris: Some(tensor),
            ..Self::empty()
        }
    }

    /// Create multi-modal evidence.
    pub fn multi(
        face: Option<axonml_autograd::Variable>,
        fingerprint: Option<axonml_autograd::Variable>,
        voice: Option<axonml_autograd::Variable>,
        iris: Option<axonml_autograd::Variable>,
    ) -> Self {
        Self {
            face,
            fingerprint,
            voice,
            iris,
            face_sequence: None,
            timestamp: None,
            device_id: None,
        }
    }

    /// Attach timestamp to evidence.
    pub fn with_timestamp(mut self, ts: f64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Attach device identifier.
    pub fn with_device(mut self, device: String) -> Self {
        self.device_id = Some(device);
        self
    }

    /// Which modalities are present.
    pub fn available_modalities(&self) -> Vec<BiometricModality> {
        let mut mods = Vec::new();
        if self.face.is_some() {
            mods.push(BiometricModality::Face);
        }
        if self.fingerprint.is_some() {
            mods.push(BiometricModality::Fingerprint);
        }
        if self.voice.is_some() {
            mods.push(BiometricModality::Voice);
        }
        if self.iris.is_some() {
            mods.push(BiometricModality::Iris);
        }
        mods
    }

    /// Number of modalities present.
    pub fn modality_count(&self) -> usize {
        self.available_modalities().len()
    }

    /// Whether this evidence has temporal face data.
    pub fn has_face_sequence(&self) -> bool {
        self.face_sequence.as_ref().is_some_and(|s| s.len() > 1)
    }
}

// =============================================================================
// Types — Results & Outputs
// =============================================================================

/// Result from modality-specific processing: embedding + uncertainty.
#[derive(Debug, Clone)]
pub struct ModalityOutput {
    /// L2-normalized embedding vector [embed_dim]
    pub embedding: Vec<f32>,
    /// Uncertainty estimate (log-variance). Lower = more confident.
    pub log_variance: f32,
    /// Which modality produced this.
    pub modality: BiometricModality,
}

/// Result of an enrollment operation.
#[derive(Debug, Clone)]
pub struct EnrollmentResult {
    /// Whether enrollment succeeded.
    pub success: bool,
    /// Subject ID enrolled.
    pub subject_id: u64,
    /// Per-modality embeddings stored.
    pub modalities_enrolled: Vec<BiometricModality>,
    /// Number of observations accumulated (for crystallization).
    pub observation_count: usize,
    /// Quality score for this enrollment [0, 1] — higher is better.
    pub quality_score: f32,
}

/// Result of a verification (1:1) operation.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Fused match probability [0, 1].
    pub match_score: f32,
    /// Whether the match exceeds the threshold.
    pub is_match: bool,
    /// Per-modality similarity scores.
    pub modality_scores: Vec<(BiometricModality, f32)>,
    /// Fused confidence (from Themis uncertainty).
    pub confidence: f32,
    /// Decision threshold used.
    pub threshold: f32,
}

/// Result of an identification (1:N) operation.
#[derive(Debug, Clone)]
pub struct IdentificationResult {
    /// Top candidate matches, sorted by score descending.
    pub candidates: Vec<IdentificationCandidate>,
    /// Fused confidence.
    pub confidence: f32,
}

/// A single identification candidate.
#[derive(Debug, Clone)]
pub struct IdentificationCandidate {
    /// Subject ID.
    pub subject_id: u64,
    /// Match score.
    pub score: f32,
    /// Per-modality scores.
    pub modality_scores: Vec<(BiometricModality, f32)>,
}

// =============================================================================
// Types — Liveness & Anti-Spoofing
// =============================================================================

/// Result of liveness (anti-spoofing) analysis.
///
/// Temporal liveness detection is a novel paradigm unique to AxonML's
/// crystallization architecture. Real biometrics exhibit micro-variations
/// in the GRU hidden state trajectory; spoofed inputs (photos, recordings)
/// produce abnormally smooth or repetitive trajectories.
#[derive(Debug, Clone)]
pub struct LivenessResult {
    /// Liveness probability [0, 1]. Above threshold = live.
    pub liveness_score: f32,
    /// Whether the input is judged as live (not spoofed).
    pub is_live: bool,
    /// Temporal variance of hidden state updates.
    /// Real biometrics: high variance. Spoofed: low variance.
    pub temporal_variance: f32,
    /// Trajectory smoothness (autocorrelation of hidden state deltas).
    /// Real: low autocorrelation. Replay: high autocorrelation.
    pub trajectory_smoothness: f32,
    /// Per-modality liveness indicators.
    pub modality_liveness: Vec<(BiometricModality, f32)>,
}

impl LivenessResult {
    /// Create a default "unknown" liveness result.
    pub fn unknown() -> Self {
        Self {
            liveness_score: 0.5,
            is_live: false,
            temporal_variance: 0.0,
            trajectory_smoothness: 0.0,
            modality_liveness: Vec::new(),
        }
    }
}

// =============================================================================
// Types — Quality Assessment
// =============================================================================

/// Quality assessment for biometric evidence.
///
/// Evaluates whether input data is suitable for reliable biometric
/// recognition before processing. Poor quality inputs should trigger
/// re-capture rather than producing unreliable results.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Overall quality score [0, 1]. Below 0.3 = reject.
    pub overall_score: f32,
    /// Per-modality quality scores.
    pub modality_scores: Vec<(BiometricModality, f32)>,
    /// Whether the evidence meets minimum quality requirements.
    pub meets_threshold: bool,
    /// Human-readable quality issues detected.
    pub issues: Vec<QualityIssue>,
}

/// A specific quality issue detected in biometric evidence.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Which modality has the issue.
    pub modality: BiometricModality,
    /// Issue severity [0, 1]. Higher = more severe.
    pub severity: f32,
    /// Description of the issue.
    pub description: String,
}

impl QualityReport {
    /// Create a report indicating all modalities passed quality checks.
    pub fn all_pass(modalities: &[BiometricModality]) -> Self {
        Self {
            overall_score: 1.0,
            modality_scores: modalities.iter().map(|m| (*m, 1.0)).collect(),
            meets_threshold: true,
            issues: Vec::new(),
        }
    }
}

// =============================================================================
// Types — Forensic Analysis
// =============================================================================

/// Forensic analysis of a biometric match/non-match decision.
///
/// Provides explainability for biometric decisions — critical for
/// audit trails, legal compliance, and debugging false matches.
#[derive(Debug, Clone)]
pub struct ForensicReport {
    /// Per-modality detailed breakdown.
    pub modality_reports: Vec<ModalityForensic>,
    /// Cross-modal consistency score [0, 1].
    /// High = all modalities agree. Low = conflicting evidence.
    pub cross_modal_consistency: f32,
    /// Which modality most influenced the final decision.
    pub dominant_modality: Option<BiometricModality>,
    /// Which modality is most uncertain.
    pub weakest_modality: Option<BiometricModality>,
    /// Per-dimension contribution to the match score (top-K most influential).
    pub top_contributing_dimensions: Vec<DimensionContribution>,
}

/// Forensic breakdown for a single modality.
#[derive(Debug, Clone)]
pub struct ModalityForensic {
    /// Which modality.
    pub modality: BiometricModality,
    /// Raw similarity score before fusion.
    pub raw_score: f32,
    /// Uncertainty of the modality output.
    pub uncertainty: f32,
    /// Weight assigned by Themis fusion.
    pub fusion_weight: f32,
    /// Whether this modality agreed with the final decision.
    pub agrees_with_decision: bool,
}

/// Contribution of a single embedding dimension to the match decision.
#[derive(Debug, Clone)]
pub struct DimensionContribution {
    /// Embedding dimension index.
    pub dimension: usize,
    /// Contribution magnitude (positive = toward match, negative = against).
    pub contribution: f32,
    /// Which modality owns this dimension.
    pub modality: BiometricModality,
}

// =============================================================================
// Types — Identity Drift
// =============================================================================

/// Alert for identity drift — gradual change in biometric template.
///
/// Novel to AxonML: by tracking the trajectory of crystallized embeddings
/// over time, we can detect when a person's biometrics are drifting
/// (aging, injury, weight change) and trigger re-enrollment.
#[derive(Debug, Clone)]
pub struct DriftAlert {
    /// Subject ID affected.
    pub subject_id: u64,
    /// How much the template has drifted since enrollment.
    /// Cosine distance between current and original embedding.
    pub drift_magnitude: f32,
    /// Rate of drift (distance per observation).
    pub drift_rate: f32,
    /// Which modalities show drift.
    pub affected_modalities: Vec<(BiometricModality, f32)>,
    /// Recommended action.
    pub recommendation: DriftRecommendation,
}

/// Recommendation for handling identity drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftRecommendation {
    /// No action needed — drift within normal bounds.
    None,
    /// Monitor: drift is approaching threshold.
    Monitor,
    /// Re-enroll: significant drift detected.
    ReEnroll,
    /// Investigate: abnormal drift pattern (possible impostor).
    Investigate,
}

impl std::fmt::Display for DriftRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "No action"),
            Self::Monitor => write!(f, "Monitor"),
            Self::ReEnroll => write!(f, "Re-enroll"),
            Self::Investigate => write!(f, "Investigate"),
        }
    }
}

// =============================================================================
// Types — Operating Point Analysis
// =============================================================================

/// Operating point on a FAR/FRR curve.
///
/// For real-world deployment, you need to choose a threshold that balances
/// False Accept Rate (letting impostors in) vs False Reject Rate (locking
/// out legitimate users).
#[derive(Debug, Clone)]
pub struct OperatingPoint {
    /// Decision threshold at this point.
    pub threshold: f32,
    /// False Accept Rate at this threshold.
    pub far: f32,
    /// False Reject Rate at this threshold.
    pub frr: f32,
    /// Equal Error Rate point (where FAR = FRR).
    pub is_eer: bool,
}

/// Full operating curve for threshold selection.
#[derive(Debug, Clone)]
pub struct OperatingCurve {
    /// Points on the FAR/FRR curve.
    pub points: Vec<OperatingPoint>,
    /// Equal Error Rate (EER) — where FAR = FRR.
    pub eer: f32,
    /// Threshold at EER.
    pub eer_threshold: f32,
}

impl OperatingCurve {
    /// Compute operating curve from genuine and impostor score distributions.
    ///
    /// * `genuine_scores` - Match scores for same-identity comparisons
    /// * `impostor_scores` - Match scores for different-identity comparisons
    /// * `n_thresholds` - Number of threshold points to evaluate
    pub fn compute(genuine_scores: &[f32], impostor_scores: &[f32], n_thresholds: usize) -> Self {
        if genuine_scores.is_empty() || impostor_scores.is_empty() {
            return Self {
                points: Vec::new(),
                eer: 1.0,
                eer_threshold: 0.5,
            };
        }

        let mut points = Vec::new();
        let mut best_eer_diff = f32::MAX;
        let mut eer = 0.5;
        let mut eer_threshold = 0.5;

        for i in 0..=n_thresholds {
            let threshold = i as f32 / n_thresholds as f32;

            // FAR = fraction of impostor scores above threshold
            let false_accepts = impostor_scores.iter().filter(|&&s| s > threshold).count();
            let far = false_accepts as f32 / impostor_scores.len() as f32;

            // FRR = fraction of genuine scores below threshold
            let false_rejects = genuine_scores.iter().filter(|&&s| s <= threshold).count();
            let frr = false_rejects as f32 / genuine_scores.len() as f32;

            let eer_diff = (far - frr).abs();
            if eer_diff < best_eer_diff {
                best_eer_diff = eer_diff;
                eer = (far + frr) * 0.5;
                eer_threshold = threshold;
            }

            points.push(OperatingPoint {
                threshold,
                far,
                frr,
                is_eer: false,
            });
        }

        // Mark EER point
        for p in &mut points {
            if (p.threshold - eer_threshold).abs() < 1e-6 {
                p.is_eer = true;
            }
        }

        Self {
            points,
            eer,
            eer_threshold,
        }
    }

    /// Find threshold for a target FAR.
    pub fn threshold_at_far(&self, target_far: f32) -> Option<f32> {
        // Find the highest threshold where FAR <= target_far
        self.points
            .iter()
            .filter(|p| p.far <= target_far)
            .min_by(|a, b| a.threshold.partial_cmp(&b.threshold).unwrap())
            .map(|p| p.threshold)
    }

    /// Find threshold for a target FRR.
    pub fn threshold_at_frr(&self, target_frr: f32) -> Option<f32> {
        self.points
            .iter()
            .filter(|p| p.frr <= target_frr)
            .max_by(|a, b| a.threshold.partial_cmp(&b.threshold).unwrap())
            .map(|p| p.threshold)
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the biometric system.
#[derive(Debug, Clone)]
pub struct BiometricConfig {
    /// Face embedding dimension.
    pub face_embed_dim: usize,
    /// Fingerprint embedding dimension.
    pub fingerprint_embed_dim: usize,
    /// Voice embedding dimension.
    pub voice_embed_dim: usize,
    /// Iris embedding dimension.
    pub iris_embed_dim: usize,
    /// Fusion common dimension.
    pub fusion_dim: usize,
    /// Verification threshold.
    pub verify_threshold: f32,
    /// Number of top-K candidates for identification.
    pub identify_top_k: usize,
    /// Liveness detection threshold (above = live).
    pub liveness_threshold: f32,
    /// Minimum quality score to proceed with recognition.
    pub quality_threshold: f32,
    /// Identity drift threshold before triggering re-enrollment.
    pub drift_threshold: f32,
    /// Number of crystallization steps for temporal face processing.
    pub crystallization_steps: usize,
}

impl Default for BiometricConfig {
    fn default() -> Self {
        Self {
            face_embed_dim: 64,
            fingerprint_embed_dim: 128,
            voice_embed_dim: 64,
            iris_embed_dim: 128,
            fusion_dim: 48,
            verify_threshold: 0.5,
            identify_top_k: 5,
            liveness_threshold: 0.6,
            quality_threshold: 0.3,
            drift_threshold: 0.4,
            crystallization_steps: 5,
        }
    }
}

impl BiometricConfig {
    /// Security-hardened configuration (higher thresholds, stricter checks).
    pub fn high_security() -> Self {
        Self {
            verify_threshold: 0.7,
            liveness_threshold: 0.8,
            quality_threshold: 0.5,
            drift_threshold: 0.3,
            crystallization_steps: 10,
            ..Default::default()
        }
    }

    /// Convenience-optimized configuration (lower thresholds, faster).
    pub fn convenience() -> Self {
        Self {
            verify_threshold: 0.35,
            liveness_threshold: 0.4,
            quality_threshold: 0.2,
            drift_threshold: 0.6,
            crystallization_steps: 3,
            ..Default::default()
        }
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Cosine similarity between two f32 vectors.
///
/// Returns 0.0 for mismatched or empty vectors. For L2-normalized inputs,
/// this equals the dot product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// L2-normalize a vector in-place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum::<f32>()
        .sqrt()
}

/// Weighted cosine similarity with per-dimension precision weighting.
pub fn weighted_cosine_similarity(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), weights.len());

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        let w = weights[i].max(0.0);
        dot += a[i] * b[i] * w;
        norm_a += a[i] * a[i] * w;
        norm_b += b[i] * b[i] * w;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
    dot / denom
}

/// Compute the entropy of a probability distribution.
/// Higher entropy = more uncertain.
pub fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 1e-10 {
            h -= p * p.ln();
        }
    }
    h
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_autograd::Variable;
    use axonml_tensor::Tensor;

    // --- BiometricEvidence ---

    #[test]
    fn test_biometric_evidence_empty() {
        let ev = BiometricEvidence::empty();
        assert!(ev.available_modalities().is_empty());
        assert_eq!(ev.modality_count(), 0);
        assert!(!ev.has_face_sequence());
    }

    #[test]
    fn test_biometric_evidence_face_only() {
        let face = Variable::new(Tensor::zeros(&[1, 3, 64, 64]), false);
        let ev = BiometricEvidence::face(face);
        let mods = ev.available_modalities();
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0], BiometricModality::Face);
    }

    #[test]
    fn test_biometric_evidence_face_sequence() {
        let frames: Vec<_> = (0..5)
            .map(|_| Variable::new(Tensor::zeros(&[1, 3, 64, 64]), false))
            .collect();
        let ev = BiometricEvidence::face_sequence(frames);
        assert!(ev.has_face_sequence());
        assert_eq!(ev.face_sequence.as_ref().unwrap().len(), 5);
        assert!(ev.face.is_some()); // First frame copied to face
    }

    #[test]
    fn test_biometric_evidence_multi_modal() {
        let face = Variable::new(Tensor::zeros(&[1, 3, 64, 64]), false);
        let voice = Variable::new(Tensor::zeros(&[1, 40, 100]), false);
        let ev = BiometricEvidence::multi(Some(face), None, Some(voice), None);
        assert_eq!(ev.modality_count(), 2);
        assert!(ev.available_modalities().contains(&BiometricModality::Face));
        assert!(
            ev.available_modalities()
                .contains(&BiometricModality::Voice)
        );
    }

    #[test]
    fn test_biometric_evidence_all_modalities() {
        let face = Variable::new(Tensor::zeros(&[1, 3, 64, 64]), false);
        let finger = Variable::new(Tensor::zeros(&[1, 1, 128, 128]), false);
        let voice = Variable::new(Tensor::zeros(&[1, 40, 100]), false);
        let iris = Variable::new(Tensor::zeros(&[1, 1, 32, 256]), false);
        let ev = BiometricEvidence::multi(Some(face), Some(finger), Some(voice), Some(iris));
        assert_eq!(ev.modality_count(), 4);
    }

    #[test]
    fn test_biometric_evidence_metadata() {
        let face = Variable::new(Tensor::zeros(&[1, 3, 64, 64]), false);
        let ev = BiometricEvidence::face(face)
            .with_timestamp(1709500000.0)
            .with_device("cam-001".to_string());
        assert_eq!(ev.timestamp, Some(1709500000.0));
        assert_eq!(ev.device_id.as_deref(), Some("cam-001"));
    }

    // --- BiometricConfig ---

    #[test]
    fn test_biometric_config_default() {
        let config = BiometricConfig::default();
        assert_eq!(config.face_embed_dim, 64);
        assert_eq!(config.fusion_dim, 48);
        assert_eq!(config.verify_threshold, 0.5);
        assert_eq!(config.crystallization_steps, 5);
    }

    #[test]
    fn test_biometric_config_high_security() {
        let config = BiometricConfig::high_security();
        assert!(config.verify_threshold > BiometricConfig::default().verify_threshold);
        assert!(config.liveness_threshold > BiometricConfig::default().liveness_threshold);
        assert!(config.quality_threshold > BiometricConfig::default().quality_threshold);
    }

    #[test]
    fn test_biometric_config_convenience() {
        let config = BiometricConfig::convenience();
        assert!(config.verify_threshold < BiometricConfig::default().verify_threshold);
        assert!(config.crystallization_steps < BiometricConfig::default().crystallization_steps);
    }

    // --- BiometricModality ---

    #[test]
    fn test_modality_display() {
        assert_eq!(format!("{}", BiometricModality::Face), "Face (Mnemosyne)");
        assert_eq!(format!("{}", BiometricModality::Voice), "Voice (Echo)");
        assert_eq!(
            format!("{}", BiometricModality::Fingerprint),
            "Fingerprint (Ariadne)"
        );
        assert_eq!(format!("{}", BiometricModality::Iris), "Iris (Argus)");
    }

    #[test]
    fn test_modality_all() {
        let all = BiometricModality::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_modality_input_description() {
        let desc = BiometricModality::Face.input_description();
        assert!(desc.contains("64"));
    }

    #[test]
    fn test_modality_approx_params() {
        assert!(BiometricModality::Face.approx_params() > 10_000);
        assert!(BiometricModality::Face.approx_params() < 200_000);
    }

    // --- LivenessResult ---

    #[test]
    fn test_liveness_result_unknown() {
        let lr = LivenessResult::unknown();
        assert_eq!(lr.liveness_score, 0.5);
        assert!(!lr.is_live);
    }

    // --- QualityReport ---

    #[test]
    fn test_quality_report_all_pass() {
        let report = QualityReport::all_pass(&[BiometricModality::Face, BiometricModality::Voice]);
        assert_eq!(report.overall_score, 1.0);
        assert!(report.meets_threshold);
        assert!(report.issues.is_empty());
        assert_eq!(report.modality_scores.len(), 2);
    }

    // --- DriftRecommendation ---

    #[test]
    fn test_drift_recommendation_display() {
        assert_eq!(format!("{}", DriftRecommendation::None), "No action");
        assert_eq!(format!("{}", DriftRecommendation::ReEnroll), "Re-enroll");
        assert_eq!(
            format!("{}", DriftRecommendation::Investigate),
            "Investigate"
        );
    }

    // --- OperatingCurve ---

    #[test]
    fn test_operating_curve_perfect_separation() {
        // Genuine scores all above 0.8, impostor scores all below 0.2
        let genuine: Vec<f32> = (0..100).map(|i| 0.8 + 0.2 * (i as f32 / 100.0)).collect();
        let impostor: Vec<f32> = (0..100).map(|i| 0.0 + 0.2 * (i as f32 / 100.0)).collect();
        let curve = OperatingCurve::compute(&genuine, &impostor, 100);

        assert!(!curve.points.is_empty());
        assert!(curve.eer < 0.1, "EER should be very low: {}", curve.eer);
    }

    #[test]
    fn test_operating_curve_overlapping() {
        let genuine: Vec<f32> = (0..100).map(|i| 0.3 + 0.4 * (i as f32 / 100.0)).collect();
        let impostor: Vec<f32> = (0..100).map(|i| 0.2 + 0.4 * (i as f32 / 100.0)).collect();
        let curve = OperatingCurve::compute(&genuine, &impostor, 100);

        assert!(curve.eer > 0.0);
        assert!(curve.eer < 1.0);
    }

    #[test]
    fn test_operating_curve_empty() {
        let curve = OperatingCurve::compute(&[], &[], 100);
        assert!(curve.points.is_empty());
        assert_eq!(curve.eer, 1.0);
    }

    #[test]
    fn test_operating_curve_threshold_at_far() {
        let genuine: Vec<f32> = (0..100).map(|i| 0.7 + 0.3 * (i as f32 / 100.0)).collect();
        let impostor: Vec<f32> = (0..100).map(|i| 0.0 + 0.3 * (i as f32 / 100.0)).collect();
        let curve = OperatingCurve::compute(&genuine, &impostor, 100);

        let threshold = curve.threshold_at_far(0.01);
        assert!(threshold.is_some());
    }

    // --- Utility functions ---

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
        assert!((v[0] - 0.6).abs() < 0.001);
        assert!((v[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v); // Should not panic
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_euclidean_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(euclidean_distance(&a, &a) < 0.001);
    }

    #[test]
    fn test_euclidean_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_weighted_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let w = vec![1.0, 1.0, 1.0];
        assert!((weighted_cosine_similarity(&a, &b, &w) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weighted_cosine_zero_weight() {
        // If weight on differing dimension is zero, similarity should be high
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let w = vec![1.0, 0.0]; // Ignore second dimension
        let sim = weighted_cosine_similarity(&a, &b, &w);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Should ignore zero-weighted dim: {}",
            sim
        );
    }

    #[test]
    fn test_entropy_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy(&probs);
        let expected = -(4.0 * 0.25 * 0.25f32.ln());
        assert!((h - expected).abs() < 0.001);
    }

    #[test]
    fn test_entropy_certain() {
        let probs = vec![1.0, 0.0, 0.0];
        let h = entropy(&probs);
        assert!(
            h < 0.001,
            "Certain distribution should have ~0 entropy: {}",
            h
        );
    }
}
