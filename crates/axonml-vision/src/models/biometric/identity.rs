//! AegisIdentity — Unified Biometric API + Identity Bank
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/identity.rs`
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
use axonml_nn::Parameter;
use axonml_tensor::Tensor;

use super::{
    AriadneFingerprint, ArgusIris, BiometricConfig, BiometricEvidence, BiometricModality,
    DriftAlert, DriftRecommendation, EchoSpeaker, EnrollmentResult, ForensicReport,
    IdentificationCandidate, IdentificationResult, LivenessResult, ModalityOutput,
    MnemosyneIdentity, QualityIssue, QualityReport, ThemisFusion, VerificationResult,
};

// =============================================================================
// IdentityBank
// =============================================================================

/// Stored identity record for a single enrolled subject.
///
/// Holds per-modality embeddings with uncertainty estimates, plus
/// Mnemosyne's crystallization hidden state for incremental updates.
#[derive(Clone)]
struct IdentityRecord {
    /// Face crystallized embedding + log_variance
    face: Option<(Vec<f32>, f32)>,
    /// Fingerprint embedding + log_variance
    fingerprint: Option<(Vec<f32>, f32)>,
    /// Voice embedding + log_variance
    voice: Option<(Vec<f32>, f32)>,
    /// Iris embedding + log_variance
    iris: Option<(Vec<f32>, f32)>,
    /// Number of enrollment observations accumulated
    observation_count: usize,
    /// Mnemosyne GRU hidden state (for incremental crystallization)
    face_hidden: Option<Vec<f32>>,
    /// Original enrollment embeddings (for drift detection)
    original_face: Option<Vec<f32>>,
    original_fingerprint: Option<Vec<f32>>,
    original_voice: Option<Vec<f32>>,
    original_iris: Option<Vec<f32>>,
    /// Enrollment timestamp
    enrolled_at: Option<f64>,
    /// Last verification timestamp
    _last_verified_at: Option<f64>,
}

impl IdentityRecord {
    fn new() -> Self {
        Self {
            face: None,
            fingerprint: None,
            voice: None,
            iris: None,
            observation_count: 0,
            face_hidden: None,
            original_face: None,
            original_fingerprint: None,
            original_voice: None,
            original_iris: None,
            enrolled_at: None,
            _last_verified_at: None,
        }
    }

    /// Get the stored embedding + log_variance for a modality.
    fn get_modality(&self, modality: BiometricModality) -> Option<&(Vec<f32>, f32)> {
        match modality {
            BiometricModality::Face => self.face.as_ref(),
            BiometricModality::Fingerprint => self.fingerprint.as_ref(),
            BiometricModality::Voice => self.voice.as_ref(),
            BiometricModality::Iris => self.iris.as_ref(),
        }
    }

    /// Get the original enrollment embedding for drift comparison.
    fn get_original(&self, modality: BiometricModality) -> Option<&Vec<f32>> {
        match modality {
            BiometricModality::Face => self.original_face.as_ref(),
            BiometricModality::Fingerprint => self.original_fingerprint.as_ref(),
            BiometricModality::Voice => self.original_voice.as_ref(),
            BiometricModality::Iris => self.original_iris.as_ref(),
        }
    }

    /// Store the original embedding on first enrollment.
    fn set_original(&mut self, modality: BiometricModality, embedding: &[f32]) {
        let target = match modality {
            BiometricModality::Face => &mut self.original_face,
            BiometricModality::Fingerprint => &mut self.original_fingerprint,
            BiometricModality::Voice => &mut self.original_voice,
            BiometricModality::Iris => &mut self.original_iris,
        };
        if target.is_none() {
            *target = Some(embedding.to_vec());
        }
    }

    /// Which modalities have been enrolled.
    fn enrolled_modalities(&self) -> Vec<BiometricModality> {
        let mut mods = Vec::new();
        if self.face.is_some() { mods.push(BiometricModality::Face); }
        if self.fingerprint.is_some() { mods.push(BiometricModality::Fingerprint); }
        if self.voice.is_some() { mods.push(BiometricModality::Voice); }
        if self.iris.is_some() { mods.push(BiometricModality::Iris); }
        mods
    }
}

/// In-memory identity bank storing enrolled subjects.
///
/// Each subject has per-modality embeddings with uncertainty estimates.
/// Supports incremental enrollment (multiple observations improve accuracy).
pub struct IdentityBank {
    records: HashMap<u64, IdentityRecord>,
}

impl IdentityBank {
    /// Create a new empty identity bank.
    pub fn new() -> Self {
        Self { records: HashMap::new() }
    }

    /// Number of enrolled subjects.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Check if a subject is enrolled.
    pub fn contains(&self, subject_id: u64) -> bool {
        self.records.contains_key(&subject_id)
    }

    /// Remove a subject from the bank.
    pub fn remove(&mut self, subject_id: u64) -> bool {
        self.records.remove(&subject_id).is_some()
    }

    /// List all enrolled subject IDs.
    pub fn subjects(&self) -> Vec<u64> {
        self.records.keys().copied().collect()
    }

    /// Number of observations for a subject.
    pub fn observation_count(&self, subject_id: u64) -> usize {
        self.records.get(&subject_id)
            .map_or(0, |r| r.observation_count)
    }

    /// Which modalities are enrolled for a subject.
    pub fn enrolled_modalities(&self, subject_id: u64) -> Vec<BiometricModality> {
        self.records.get(&subject_id)
            .map_or_else(Vec::new, |r| r.enrolled_modalities())
    }
}

// =============================================================================
// AegisIdentity
// =============================================================================

/// Unified biometric identity system.
///
/// Wraps all modalities and provides enroll/verify/identify API.
/// Missing modalities gracefully degrade through Themis uncertainty gating.
///
/// # Example
///
/// ```ignore
/// let mut aegis = AegisIdentity::face_only();
///
/// // Enroll a subject
/// let face_tensor = Variable::new(Tensor::rand(&[1, 3, 64, 64]), false);
/// let evidence = BiometricEvidence::face(face_tensor);
/// aegis.enroll(42, &evidence);
///
/// // Verify identity
/// let probe = BiometricEvidence::face(probe_tensor);
/// let result = aegis.verify(42, &probe);
/// println!("Match: {} (score: {:.3})", result.is_match, result.match_score);
/// ```
pub struct AegisIdentity {
    face: Option<MnemosyneIdentity>,
    finger: Option<AriadneFingerprint>,
    voice: Option<EchoSpeaker>,
    iris: Option<ArgusIris>,
    fusion: ThemisFusion,
    /// Identity bank for enrolled subjects
    pub bank: IdentityBank,
    config: BiometricConfig,
}

impl AegisIdentity {
    /// Create with all modalities enabled (~362K params).
    pub fn full() -> Self {
        Self {
            face: Some(MnemosyneIdentity::new()),
            finger: Some(AriadneFingerprint::new()),
            voice: Some(EchoSpeaker::new()),
            iris: Some(ArgusIris::new()),
            fusion: ThemisFusion::new(),
            bank: IdentityBank::new(),
            config: BiometricConfig::default(),
        }
    }

    /// Create with face-only (~115K params, smallest single modality).
    pub fn face_only() -> Self {
        Self {
            face: Some(MnemosyneIdentity::new()),
            finger: None,
            voice: None,
            iris: None,
            fusion: ThemisFusion::new(),
            bank: IdentityBank::new(),
            config: BiometricConfig::default(),
        }
    }

    /// Create with face + voice (~183K params, minimal edge deployment).
    pub fn edge_minimal() -> Self {
        Self {
            face: Some(MnemosyneIdentity::new()),
            finger: None,
            voice: Some(EchoSpeaker::new()),
            iris: None,
            fusion: ThemisFusion::new(),
            bank: IdentityBank::new(),
            config: BiometricConfig::default(),
        }
    }

    /// Create with custom modality selection.
    pub fn with_modalities(
        face: bool,
        finger: bool,
        voice: bool,
        iris: bool,
    ) -> Self {
        Self {
            face: if face { Some(MnemosyneIdentity::new()) } else { None },
            finger: if finger { Some(AriadneFingerprint::new()) } else { None },
            voice: if voice { Some(EchoSpeaker::new()) } else { None },
            iris: if iris { Some(ArgusIris::new()) } else { None },
            fusion: ThemisFusion::new(),
            bank: IdentityBank::new(),
            config: BiometricConfig::default(),
        }
    }

    /// Create with a specific configuration profile.
    pub fn with_config(config: BiometricConfig) -> Self {
        Self {
            face: Some(MnemosyneIdentity::new()),
            finger: Some(AriadneFingerprint::new()),
            voice: Some(EchoSpeaker::new()),
            iris: Some(ArgusIris::new()),
            fusion: ThemisFusion::new(),
            bank: IdentityBank::new(),
            config,
        }
    }

    /// Set the verification threshold (default: 0.5).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.config.verify_threshold = threshold;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &BiometricConfig {
        &self.config
    }

    // =========================================================================
    // Core operations
    // =========================================================================

    /// Process biometric evidence through all available modalities.
    ///
    /// Returns per-modality outputs (embedding + uncertainty) for each
    /// modality that has both a model and input data.
    fn process_evidence(&self, evidence: &BiometricEvidence) -> Vec<ModalityOutput> {
        let mut outputs = Vec::new();

        if let (Some(ref model), Some(ref face_var)) = (&self.face, &evidence.face) {
            let encoding = model.encode_face(face_var);
            let identity = model.extract_identity(&encoding);
            outputs.push(ModalityOutput {
                embedding: identity,
                log_variance: 0.0, // Single-frame: default uncertainty
                modality: BiometricModality::Face,
            });
        }

        if let (Some(ref model), Some(ref finger_var)) = (&self.finger, &evidence.fingerprint) {
            let (embedding, logvar) = model.forward_full(finger_var);
            outputs.push(ModalityOutput {
                embedding: embedding.data().to_vec(),
                log_variance: logvar.data().to_vec()[0],
                modality: BiometricModality::Fingerprint,
            });
        }

        if let (Some(ref model), Some(ref voice_var)) = (&self.voice, &evidence.voice) {
            let (_pred, embedding, logvar) = model.forward_full(voice_var);
            outputs.push(ModalityOutput {
                embedding: embedding.data().to_vec(),
                log_variance: logvar.data().to_vec()[0],
                modality: BiometricModality::Voice,
            });
        }

        if let (Some(ref model), Some(ref iris_var)) = (&self.iris, &evidence.iris) {
            let (embedding, logvar) = model.forward_full(iris_var);
            outputs.push(ModalityOutput {
                embedding: embedding.data().to_vec(),
                log_variance: logvar.data().to_vec()[0],
                modality: BiometricModality::Iris,
            });
        }

        outputs
    }

    /// Enroll a subject with biometric evidence.
    ///
    /// Can be called multiple times to:
    /// - Add additional modalities
    /// - Refine face crystallization with more observations
    /// - Update existing modality embeddings
    pub fn enroll(&mut self, subject_id: u64, evidence: &BiometricEvidence) -> EnrollmentResult {
        let outputs = self.process_evidence(evidence);

        let record = self.bank.records.entry(subject_id).or_insert_with(IdentityRecord::new);
        let mut enrolled_modalities = Vec::new();
        let mut quality_sum = 0.0f32;
        let mut quality_count = 0;

        for output in &outputs {
            match output.modality {
                BiometricModality::Face => {
                    // Face uses crystallization: GRU hidden state evolves with each observation
                    if let (Some(ref model), Some(ref face_var)) = (&self.face, &evidence.face) {
                        let hidden = record.face_hidden.as_ref().map(|h| {
                            Variable::new(
                                Tensor::from_vec(h.clone(), &[1, model.hidden_dim()]).unwrap(),
                                false,
                            )
                        });
                        let (new_hidden, _vel, logvar, qual) = model.crystallize_step(
                            face_var,
                            hidden.as_ref(),
                        );
                        let identity = model.extract_identity(&new_hidden);
                        let lv = logvar.data().to_vec()[0];
                        record.set_original(BiometricModality::Face, &identity);
                        record.face = Some((identity, lv));
                        record.face_hidden = Some(new_hidden.data().to_vec());
                        quality_sum += qual.data().to_vec()[0].clamp(0.0, 1.0);
                        quality_count += 1;
                    }
                    enrolled_modalities.push(BiometricModality::Face);
                }
                BiometricModality::Fingerprint => {
                    record.set_original(BiometricModality::Fingerprint, &output.embedding);
                    record.fingerprint = Some((output.embedding.clone(), output.log_variance));
                    // Quality from uncertainty: lower logvar = higher quality
                    quality_sum += (-output.log_variance).clamp(0.0, 1.0);
                    quality_count += 1;
                    enrolled_modalities.push(BiometricModality::Fingerprint);
                }
                BiometricModality::Voice => {
                    record.set_original(BiometricModality::Voice, &output.embedding);
                    record.voice = Some((output.embedding.clone(), output.log_variance));
                    quality_sum += (-output.log_variance).clamp(0.0, 1.0);
                    quality_count += 1;
                    enrolled_modalities.push(BiometricModality::Voice);
                }
                BiometricModality::Iris => {
                    record.set_original(BiometricModality::Iris, &output.embedding);
                    record.iris = Some((output.embedding.clone(), output.log_variance));
                    quality_sum += (-output.log_variance).clamp(0.0, 1.0);
                    quality_count += 1;
                    enrolled_modalities.push(BiometricModality::Iris);
                }
            }
        }

        record.observation_count += 1;
        if record.enrolled_at.is_none() {
            record.enrolled_at = evidence.timestamp;
        }

        let quality_score = if quality_count > 0 {
            (quality_sum / quality_count as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        EnrollmentResult {
            success: !enrolled_modalities.is_empty(),
            subject_id,
            modalities_enrolled: enrolled_modalities,
            observation_count: record.observation_count,
            quality_score,
        }
    }

    /// Verify a claimed identity (1:1 matching).
    ///
    /// Computes per-modality similarity scores between the probe evidence
    /// and the enrolled record, then fuses through Themis for a combined
    /// match probability.
    pub fn verify(&self, claimed_id: u64, evidence: &BiometricEvidence) -> VerificationResult {
        let record = match self.bank.records.get(&claimed_id) {
            Some(r) => r,
            None => return VerificationResult {
                match_score: 0.0,
                is_match: false,
                modality_scores: Vec::new(),
                confidence: 0.0,
                threshold: self.config.verify_threshold,
            },
        };

        let outputs = self.process_evidence(evidence);
        let mut modality_scores = Vec::new();

        // Per-modality cosine similarity between probe and enrolled
        for output in &outputs {
            if let Some((enrolled_emb, _)) = record.get_modality(output.modality) {
                let score = cosine_similarity(&output.embedding, enrolled_emb);
                modality_scores.push((output.modality, score));
            }
        }

        // Fuse via Themis for combined decision
        let face_input = self.make_fusion_input(&outputs, record, BiometricModality::Face);
        let finger_input = self.make_fusion_input(&outputs, record, BiometricModality::Fingerprint);
        let voice_input = self.make_fusion_input(&outputs, record, BiometricModality::Voice);
        let iris_input = self.make_fusion_input(&outputs, record, BiometricModality::Iris);

        let face_ref = face_input.as_ref().map(|(v, lv)| (v, *lv));
        let finger_ref = finger_input.as_ref().map(|(v, lv)| (v, *lv));
        let voice_ref = voice_input.as_ref().map(|(v, lv)| (v, *lv));
        let iris_ref = iris_input.as_ref().map(|(v, lv)| (v, *lv));

        let (_fused_identity, match_prob, confidence, _belief) = self.fusion.fuse(
            face_ref,
            finger_ref,
            voice_ref,
            iris_ref,
            None,
        );

        // Score selection: use Themis match_prob when confidence is available,
        // otherwise fall back to raw modality score average
        let match_score = if modality_scores.is_empty() {
            0.0
        } else if confidence < 0.01 {
            // No modalities had meaningful confidence → raw average
            modality_scores.iter().map(|(_, s)| s).sum::<f32>()
                / modality_scores.len() as f32
        } else {
            // Confidence-weighted blend: Themis decision + raw scores
            // Higher confidence → trust Themis more
            let raw_avg = modality_scores.iter().map(|(_, s)| s).sum::<f32>()
                / modality_scores.len() as f32;
            let themis_weight = confidence.min(1.0);
            themis_weight * match_prob + (1.0 - themis_weight) * raw_avg
        };

        VerificationResult {
            match_score,
            is_match: match_score > self.config.verify_threshold,
            modality_scores,
            confidence,
            threshold: self.config.verify_threshold,
        }
    }

    /// Identify against all enrolled subjects (1:N matching).
    ///
    /// Returns candidates ranked by score with per-modality breakdowns.
    /// The number of candidates is limited by `config.identify_top_k`.
    pub fn identify(&self, evidence: &BiometricEvidence) -> IdentificationResult {
        let outputs = self.process_evidence(evidence);

        if outputs.is_empty() {
            return IdentificationResult {
                candidates: Vec::new(),
                confidence: 0.0,
            };
        }

        let mut candidates: Vec<IdentificationCandidate> = Vec::new();

        for (&subject_id, record) in &self.bank.records {
            let mut modality_scores = Vec::new();

            for output in &outputs {
                if let Some((enrolled_emb, _)) = record.get_modality(output.modality) {
                    let score = cosine_similarity(&output.embedding, enrolled_emb);
                    modality_scores.push((output.modality, score));
                }
            }

            if !modality_scores.is_empty() {
                let avg_score = modality_scores.iter().map(|(_, s)| s).sum::<f32>()
                    / modality_scores.len() as f32;

                candidates.push(IdentificationCandidate {
                    subject_id,
                    score: avg_score,
                    modality_scores,
                });
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.identify_top_k);

        // Confidence: gap between top-1 and top-2 (discriminability)
        let confidence = if candidates.is_empty() {
            0.0
        } else if candidates.len() == 1 {
            candidates[0].score
        } else {
            (candidates[0].score - candidates[1].score).max(0.0)
        };

        IdentificationResult { candidates, confidence }
    }

    // =========================================================================
    // Forensic Verification
    // =========================================================================

    /// Verify with full forensic audit trail.
    ///
    /// Returns both the standard VerificationResult and a ForensicReport
    /// detailing per-modality contributions, cross-modal consistency,
    /// and per-dimension influence on the decision.
    pub fn verify_forensic(
        &self,
        claimed_id: u64,
        evidence: &BiometricEvidence,
    ) -> (VerificationResult, ForensicReport) {
        let record = match self.bank.records.get(&claimed_id) {
            Some(r) => r,
            None => {
                let vr = VerificationResult {
                    match_score: 0.0,
                    is_match: false,
                    modality_scores: Vec::new(),
                    confidence: 0.0,
                    threshold: self.config.verify_threshold,
                };
                let fr = ForensicReport {
                    modality_reports: Vec::new(),
                    cross_modal_consistency: 0.0,
                    dominant_modality: None,
                    weakest_modality: None,
                    top_contributing_dimensions: Vec::new(),
                };
                return (vr, fr);
            }
        };

        let outputs = self.process_evidence(evidence);
        let mut modality_scores = Vec::new();
        let mut forensic_reports = Vec::new();
        let mut all_contributions = Vec::new();

        for output in &outputs {
            if let Some((enrolled_emb, enrolled_lv)) = record.get_modality(output.modality) {
                let score = cosine_similarity(&output.embedding, enrolled_emb);
                modality_scores.push((output.modality, score));

                // Uncertainty gate weight
                let combined_lv = (output.log_variance + enrolled_lv) * 0.5;
                let uncertainty = combined_lv.exp();
                let gate_weight = 1.0 / (1.0 + uncertainty);

                forensic_reports.push(super::ModalityForensic {
                    modality: output.modality,
                    raw_score: score,
                    uncertainty,
                    fusion_weight: gate_weight,
                    agrees_with_decision: true, // Updated below
                });

                // Per-dimension contribution (top contributing dimensions)
                let n = output.embedding.len().min(enrolled_emb.len());
                for dim in 0..n {
                    let contrib = output.embedding[dim] * enrolled_emb[dim];
                    all_contributions.push(super::DimensionContribution {
                        dimension: dim,
                        contribution: contrib,
                        modality: output.modality,
                    });
                }
            }
        }

        // Sort contributions by absolute magnitude, take top 10
        all_contributions.sort_by(|a, b| {
            b.contribution.abs().partial_cmp(&a.contribution.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_contributions.truncate(10);

        // Cross-modal consistency: variance of raw scores
        let cross_modal_consistency = if modality_scores.len() < 2 {
            1.0
        } else {
            let mean = modality_scores.iter().map(|(_, s)| s).sum::<f32>()
                / modality_scores.len() as f32;
            let var = modality_scores.iter()
                .map(|(_, s)| (s - mean) * (s - mean))
                .sum::<f32>() / modality_scores.len() as f32;
            // Low variance = high consistency. Map var to [0, 1].
            (1.0 - var.sqrt() * 4.0).max(0.0)
        };

        // Dominant and weakest modality
        let dominant = forensic_reports.iter()
            .max_by(|a, b| (a.raw_score * a.fusion_weight)
                .partial_cmp(&(b.raw_score * b.fusion_weight))
                .unwrap_or(std::cmp::Ordering::Equal))
            .map(|r| r.modality);

        let weakest = forensic_reports.iter()
            .min_by(|a, b| a.fusion_weight
                .partial_cmp(&b.fusion_weight)
                .unwrap_or(std::cmp::Ordering::Equal))
            .map(|r| r.modality);

        // Compute the verification result
        let verification = self.verify(claimed_id, evidence);

        // Update agrees_with_decision based on the result
        for report in &mut forensic_reports {
            let agrees = if verification.is_match {
                report.raw_score > self.config.verify_threshold * 0.5
            } else {
                report.raw_score < self.config.verify_threshold * 1.5
            };
            report.agrees_with_decision = agrees;
        }

        let forensic = ForensicReport {
            modality_reports: forensic_reports,
            cross_modal_consistency,
            dominant_modality: dominant,
            weakest_modality: weakest,
            top_contributing_dimensions: all_contributions,
        };

        (verification, forensic)
    }

    // =========================================================================
    // Batch Operations
    // =========================================================================

    /// Enroll multiple subjects in batch.
    ///
    /// Processes each (subject_id, evidence) pair sequentially but shares
    /// model state for efficiency.
    pub fn batch_enroll(
        &mut self,
        subjects: &[(u64, BiometricEvidence)],
    ) -> Vec<EnrollmentResult> {
        subjects.iter()
            .map(|(id, evidence)| self.enroll(*id, evidence))
            .collect()
    }

    /// Verify multiple subjects in batch.
    ///
    /// Returns verification results in the same order as input.
    pub fn batch_verify(
        &self,
        claims: &[(u64, BiometricEvidence)],
    ) -> Vec<VerificationResult> {
        claims.iter()
            .map(|(id, evidence)| self.verify(*id, evidence))
            .collect()
    }

    /// Identify multiple probes against the identity bank.
    pub fn batch_identify(
        &self,
        probes: &[BiometricEvidence],
    ) -> Vec<IdentificationResult> {
        probes.iter()
            .map(|evidence| self.identify(evidence))
            .collect()
    }

    // =========================================================================
    // Identity Drift Detection
    // =========================================================================

    /// Check for identity drift for a specific subject.
    ///
    /// Compares current enrolled embeddings against original enrollment
    /// to detect gradual biometric changes (aging, injury, etc.).
    pub fn detect_drift(&self, subject_id: u64) -> Option<DriftAlert> {
        let record = self.bank.records.get(&subject_id)?;

        if record.observation_count < 2 {
            return None; // Need at least 2 observations to detect drift
        }

        let mut affected_modalities = Vec::new();
        let mut total_drift = 0.0f32;
        let mut drift_count = 0;

        for modality in BiometricModality::all() {
            if let (Some((current_emb, _)), Some(original_emb)) = (
                record.get_modality(modality),
                record.get_original(modality),
            ) {
                let sim = cosine_similarity(current_emb, original_emb);
                let drift = 1.0 - sim; // cosine distance
                if drift > 0.01 { // Only report meaningful drift
                    affected_modalities.push((modality, drift));
                    total_drift += drift;
                    drift_count += 1;
                }
            }
        }

        if drift_count == 0 {
            return None;
        }

        let avg_drift = total_drift / drift_count as f32;
        let drift_rate = avg_drift / record.observation_count as f32;

        let recommendation = if avg_drift > self.config.drift_threshold * 1.5 {
            DriftRecommendation::Investigate
        } else if avg_drift > self.config.drift_threshold {
            DriftRecommendation::ReEnroll
        } else if avg_drift > self.config.drift_threshold * 0.5 {
            DriftRecommendation::Monitor
        } else {
            DriftRecommendation::None
        };

        Some(DriftAlert {
            subject_id,
            drift_magnitude: avg_drift,
            drift_rate,
            affected_modalities,
            recommendation,
        })
    }

    /// Check drift for all enrolled subjects.
    ///
    /// Returns alerts only for subjects with detectable drift.
    pub fn detect_all_drift(&self) -> Vec<DriftAlert> {
        self.bank.subjects().iter()
            .filter_map(|&id| self.detect_drift(id))
            .collect()
    }

    // =========================================================================
    // Quality Assessment
    // =========================================================================

    /// Assess the quality of biometric evidence before processing.
    ///
    /// Returns a quality report with per-modality scores and issues.
    /// Evidence below quality_threshold should trigger re-capture.
    pub fn assess_quality(&self, evidence: &BiometricEvidence) -> QualityReport {
        let mut modality_scores = Vec::new();
        let mut issues = Vec::new();

        // Face quality: check via Mnemosyne's quality gate
        if let (Some(ref model), Some(ref face_var)) = (&self.face, &evidence.face) {
            let quality = model.assess_quality(face_var);
            modality_scores.push((BiometricModality::Face, quality));
            if quality < self.config.quality_threshold {
                issues.push(QualityIssue {
                    modality: BiometricModality::Face,
                    severity: 1.0 - quality,
                    description: format!(
                        "Face quality {:.2} below threshold {:.2}",
                        quality, self.config.quality_threshold
                    ),
                });
            }
        }

        // Fingerprint quality: encoding magnitude as proxy
        if let (Some(ref model), Some(ref finger_var)) = (&self.finger, &evidence.fingerprint) {
            let (emb, logvar) = model.forward_full(finger_var);
            let mag = emb.data().to_vec().iter().map(|x| x * x).sum::<f32>().sqrt();
            let lv = logvar.data().to_vec()[0];
            // Quality from norm (should be ~1.0 for normalized) and low uncertainty
            let quality = (mag.min(1.5) / 1.5 * 0.5 + (-lv).clamp(0.0, 1.0) * 0.5).clamp(0.0, 1.0);
            modality_scores.push((BiometricModality::Fingerprint, quality));
            if quality < self.config.quality_threshold {
                issues.push(QualityIssue {
                    modality: BiometricModality::Fingerprint,
                    severity: 1.0 - quality,
                    description: format!(
                        "Fingerprint quality {:.2} below threshold {:.2}",
                        quality, self.config.quality_threshold
                    ),
                });
            }
        }

        // Voice quality: via prediction error variance
        if let (Some(ref model), Some(ref voice_var)) = (&self.voice, &evidence.voice) {
            let (_pred, _emb, logvar) = model.forward_full(voice_var);
            let lv = logvar.data().to_vec()[0];
            let quality = (-lv).clamp(0.0, 1.0);
            modality_scores.push((BiometricModality::Voice, quality));
            if quality < self.config.quality_threshold {
                issues.push(QualityIssue {
                    modality: BiometricModality::Voice,
                    severity: 1.0 - quality,
                    description: format!(
                        "Voice quality {:.2} below threshold {:.2}",
                        quality, self.config.quality_threshold
                    ),
                });
            }
        }

        // Iris quality: via Argus quality assessment
        if let (Some(ref model), Some(ref iris_var)) = (&self.iris, &evidence.iris) {
            let quality = model.assess_quality(iris_var);
            modality_scores.push((BiometricModality::Iris, quality));
            if quality < self.config.quality_threshold {
                issues.push(QualityIssue {
                    modality: BiometricModality::Iris,
                    severity: 1.0 - quality,
                    description: format!(
                        "Iris quality {:.2} below threshold {:.2}",
                        quality, self.config.quality_threshold
                    ),
                });
            }
        }

        let overall = if modality_scores.is_empty() {
            0.0
        } else {
            modality_scores.iter().map(|(_, s)| s).sum::<f32>()
                / modality_scores.len() as f32
        };

        QualityReport {
            overall_score: overall,
            modality_scores,
            meets_threshold: overall >= self.config.quality_threshold,
            issues,
        }
    }

    // =========================================================================
    // Liveness Detection
    // =========================================================================

    /// Assess liveness (anti-spoofing) from temporal face evidence.
    ///
    /// Requires `face_sequence` in the evidence (multiple frames).
    /// Uses Mnemosyne's GRU hidden state trajectory analysis to detect
    /// photos, videos, and deepfakes.
    pub fn assess_liveness(&self, evidence: &BiometricEvidence) -> LivenessResult {
        if !evidence.has_face_sequence() {
            return LivenessResult::unknown();
        }

        let model = match &self.face {
            Some(m) => m,
            None => return LivenessResult::unknown(),
        };

        let frames = evidence.face_sequence.as_ref().unwrap();
        let liveness = model.assess_liveness(frames);

        let mut modality_liveness = vec![(BiometricModality::Face, liveness.liveness_score)];

        // Voice replay detection if available
        if let (Some(ref voice_model), Some(ref voice_var)) = (&self.voice, &evidence.voice) {
            let replay_score = voice_model.detect_replay(voice_var);
            // replay_score near 1.0 = likely spoofed, near 0.0 = live
            let voice_liveness = 1.0 - replay_score;
            modality_liveness.push((BiometricModality::Voice, voice_liveness));
        }

        // Combined liveness from all available modalities
        let combined = modality_liveness.iter().map(|(_, s)| s).sum::<f32>()
            / modality_liveness.len() as f32;

        LivenessResult {
            liveness_score: combined,
            is_live: combined > self.config.liveness_threshold,
            temporal_variance: liveness.temporal_variance,
            trajectory_smoothness: liveness.trajectory_smoothness,
            modality_liveness,
        }
    }

    // =========================================================================
    // Secure Verification (quality + liveness + verify)
    // =========================================================================

    /// Full security pipeline: quality check → liveness → verification.
    ///
    /// Returns the verification result only if quality and liveness pass.
    /// Otherwise returns a failed result with the reason.
    pub fn secure_verify(
        &self,
        claimed_id: u64,
        evidence: &BiometricEvidence,
    ) -> (VerificationResult, Option<QualityReport>, Option<LivenessResult>) {
        // Step 1: Quality assessment
        let quality = self.assess_quality(evidence);
        if !quality.meets_threshold {
            let vr = VerificationResult {
                match_score: 0.0,
                is_match: false,
                modality_scores: Vec::new(),
                confidence: 0.0,
                threshold: self.config.verify_threshold,
            };
            return (vr, Some(quality), None);
        }

        // Step 2: Liveness (only if face sequence available)
        let liveness = if evidence.has_face_sequence() {
            let lr = self.assess_liveness(evidence);
            if !lr.is_live {
                let vr = VerificationResult {
                    match_score: 0.0,
                    is_match: false,
                    modality_scores: Vec::new(),
                    confidence: 0.0,
                    threshold: self.config.verify_threshold,
                };
                return (vr, Some(quality), Some(lr));
            }
            Some(lr)
        } else {
            None
        };

        // Step 3: Verification
        let vr = self.verify(claimed_id, evidence);
        (vr, Some(quality), liveness)
    }

    // =========================================================================
    // Statistics and Diagnostics
    // =========================================================================

    /// Compute operating curve from the identity bank.
    ///
    /// Generates genuine/impostor score distributions by comparing all
    /// enrolled subjects pairwise, then computes FAR/FRR curve.
    pub fn compute_operating_curve(&self, n_thresholds: usize) -> super::OperatingCurve {
        let subjects: Vec<u64> = self.bank.subjects();
        let mut genuine_scores = Vec::new();
        let mut impostor_scores = Vec::new();

        for i in 0..subjects.len() {
            let record_i = match self.bank.records.get(&subjects[i]) {
                Some(r) => r,
                None => continue,
            };

            for j in (i + 1)..subjects.len() {
                let record_j = match self.bank.records.get(&subjects[j]) {
                    Some(r) => r,
                    None => continue,
                };

                // Compare all shared modalities
                for modality in BiometricModality::all() {
                    if let (Some((emb_i, _)), Some((emb_j, _))) = (
                        record_i.get_modality(modality),
                        record_j.get_modality(modality),
                    ) {
                        let score = cosine_similarity(emb_i, emb_j);
                        impostor_scores.push(score);
                    }
                }
            }

            // Self-comparison for genuine scores (enrollment vs enrollment)
            // Use original vs current as a proxy
            for modality in BiometricModality::all() {
                if let (Some((current, _)), Some(original)) = (
                    record_i.get_modality(modality),
                    record_i.get_original(modality),
                ) {
                    let score = cosine_similarity(current, original);
                    genuine_scores.push(score);
                }
            }
        }

        super::OperatingCurve::compute(&genuine_scores, &impostor_scores, n_thresholds)
    }

    /// Create a fusion input Variable from probe output and enrolled record.
    fn make_fusion_input(
        &self,
        outputs: &[ModalityOutput],
        record: &IdentityRecord,
        modality: BiometricModality,
    ) -> Option<(Variable, f32)> {
        let probe = outputs.iter().find(|o| o.modality == modality)?;
        let (_, enrolled_lv) = record.get_modality(modality)?;

        let var = Variable::new(
            Tensor::from_vec(probe.embedding.clone(), &[1, probe.embedding.len()]).unwrap(),
            false,
        );

        // Average log_variance from probe and enrolled
        let combined_lv = (probe.log_variance + enrolled_lv) * 0.5;

        Some((var, combined_lv))
    }

    /// Collect all learnable parameters from all enabled models.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        if let Some(ref m) = self.face {
            p.extend(m.parameters());
        }
        if let Some(ref m) = self.finger {
            p.extend(m.parameters());
        }
        if let Some(ref m) = self.voice {
            p.extend(m.parameters());
        }
        if let Some(ref m) = self.iris {
            p.extend(m.parameters());
        }
        p.extend(self.fusion.parameters());
        p
    }

    /// Total parameter count across all enabled models.
    pub fn total_params(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum()
    }

    /// Which modalities are enabled in this configuration.
    pub fn enabled_modalities(&self) -> Vec<BiometricModality> {
        let mut mods = Vec::new();
        if self.face.is_some() { mods.push(BiometricModality::Face); }
        if self.finger.is_some() { mods.push(BiometricModality::Fingerprint); }
        if self.voice.is_some() { mods.push(BiometricModality::Voice); }
        if self.iris.is_some() { mods.push(BiometricModality::Iris); }
        mods
    }

    /// Summary of the system configuration for diagnostics.
    pub fn summary(&self) -> String {
        let mods = self.enabled_modalities();
        let total = self.total_params();
        format!(
            "AegisIdentity: {} modalities ({}) | {} params ({:.1}KB f32) | {} enrolled | threshold={:.2}",
            mods.len(),
            mods.iter().map(|m| format!("{}", m)).collect::<Vec<_>>().join(", "),
            total,
            total as f32 * 4.0 / 1024.0,
            self.bank.len(),
            self.config.verify_threshold,
        )
    }
}

// =============================================================================
// Utilities
// =============================================================================

/// Cosine similarity between two f32 vectors.
///
/// Returns 0.0 for mismatched or empty vectors. For L2-normalized inputs,
/// this equals the dot product.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

    fn make_face(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 1 * 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        )
    }

    fn make_face_evidence(val: f32) -> BiometricEvidence {
        BiometricEvidence::face(make_face(val))
    }

    fn make_fingerprint(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        )
    }

    fn make_voice(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 1 * 40 * 50], &[1, 40, 50]).unwrap(),
            false,
        )
    }

    fn make_iris(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 1 * 1 * 64 * 64], &[1, 1, 64, 64]).unwrap(),
            false,
        )
    }

    // ---- Creation tests ----

    #[test]
    fn test_aegis_full_creation() {
        let system = AegisIdentity::full();
        assert_eq!(system.enabled_modalities().len(), 4);
    }

    #[test]
    fn test_aegis_face_only_creation() {
        let system = AegisIdentity::face_only();
        assert_eq!(system.enabled_modalities().len(), 1);
        assert_eq!(system.enabled_modalities()[0], BiometricModality::Face);
    }

    #[test]
    fn test_aegis_edge_minimal_creation() {
        let system = AegisIdentity::edge_minimal();
        assert_eq!(system.enabled_modalities().len(), 2);
    }

    #[test]
    fn test_aegis_custom_modalities() {
        let system = AegisIdentity::with_modalities(true, false, true, false);
        let mods = system.enabled_modalities();
        assert_eq!(mods.len(), 2);
        assert!(mods.contains(&BiometricModality::Face));
        assert!(mods.contains(&BiometricModality::Voice));
    }

    #[test]
    fn test_aegis_with_config() {
        let config = BiometricConfig::high_security();
        let system = AegisIdentity::with_config(config);
        assert_eq!(system.config().verify_threshold, 0.7);
    }

    #[test]
    fn test_aegis_total_params() {
        let system = AegisIdentity::full();
        let total = system.total_params();
        assert!(total < 400_000, "Total params {} exceeds 400K budget", total);
        assert!(total > 100_000, "Total params {} seems too low", total);
    }

    // ---- Identity Bank tests ----

    #[test]
    fn test_identity_bank() {
        let mut bank = IdentityBank::new();
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);
        assert!(!bank.contains(1));

        bank.records.insert(1, IdentityRecord::new());
        assert!(bank.contains(1));
        assert_eq!(bank.len(), 1);
        assert!(bank.remove(1));
        assert!(!bank.contains(1));
    }

    #[test]
    fn test_bank_subjects_list() {
        let mut system = AegisIdentity::face_only();
        let face = make_face(0.5);
        system.enroll(10, &BiometricEvidence::face(face.clone()));
        system.enroll(20, &BiometricEvidence::face(face));

        let subjects = system.bank.subjects();
        assert_eq!(subjects.len(), 2);
        assert!(subjects.contains(&10));
        assert!(subjects.contains(&20));
    }

    #[test]
    fn test_bank_observation_count() {
        let mut system = AegisIdentity::face_only();
        assert_eq!(system.bank.observation_count(1), 0);

        system.enroll(1, &make_face_evidence(0.3));
        assert_eq!(system.bank.observation_count(1), 1);

        system.enroll(1, &make_face_evidence(0.3));
        assert_eq!(system.bank.observation_count(1), 2);
    }

    #[test]
    fn test_bank_enrolled_modalities() {
        let mut system = AegisIdentity::full();
        assert!(system.bank.enrolled_modalities(1).is_empty());

        system.enroll(1, &make_face_evidence(0.3));
        let mods = system.bank.enrolled_modalities(1);
        assert!(mods.contains(&BiometricModality::Face));
    }

    // ---- Enrollment tests ----

    #[test]
    fn test_aegis_enroll_face() {
        let mut system = AegisIdentity::face_only();
        let evidence = make_face_evidence(0.5);
        let result = system.enroll(1, &evidence);

        assert!(result.success);
        assert_eq!(result.subject_id, 1);
        assert!(result.modalities_enrolled.contains(&BiometricModality::Face));
        assert_eq!(result.observation_count, 1);
        assert!(system.bank.contains(1));
    }

    #[test]
    fn test_aegis_enroll_quality_score() {
        let mut system = AegisIdentity::face_only();
        let result = system.enroll(1, &make_face_evidence(0.5));
        assert!(result.quality_score >= 0.0 && result.quality_score <= 1.0);
    }

    #[test]
    fn test_aegis_multi_enrollment() {
        let mut system = AegisIdentity::face_only();
        for _ in 0..3 {
            let result = system.enroll(1, &make_face_evidence(0.3));
            assert!(result.success);
        }
        assert_eq!(system.bank.records.get(&1).unwrap().observation_count, 3);
    }

    // ---- Verification tests ----

    #[test]
    fn test_aegis_enroll_verify_round_trip() {
        let mut system = AegisIdentity::face_only();
        system.enroll(42, &make_face_evidence(0.3));

        let result = system.verify(42, &make_face_evidence(0.3));
        assert!(result.match_score > 0.0, "Self-match should be positive: {}", result.match_score);
        assert!(!result.modality_scores.is_empty());
    }

    #[test]
    fn test_aegis_verify_unknown_subject() {
        let system = AegisIdentity::face_only();
        let result = system.verify(999, &make_face_evidence(0.5));
        assert!(!result.is_match);
        assert_eq!(result.match_score, 0.0);
    }

    #[test]
    fn test_aegis_verify_threshold() {
        let mut system = AegisIdentity::face_only();
        system.set_threshold(0.99); // Very high threshold
        system.enroll(1, &make_face_evidence(0.3));
        let result = system.verify(1, &make_face_evidence(0.3));
        assert_eq!(result.threshold, 0.99);
    }

    // ---- Identification tests ----

    #[test]
    fn test_aegis_identify() {
        let mut system = AegisIdentity::face_only();

        system.enroll(1, &make_face_evidence(0.3));
        system.enroll(2, &make_face_evidence(0.7));

        let result = system.identify(&make_face_evidence(0.3));
        assert!(!result.candidates.is_empty());
        assert!(result.candidates.len() <= 5);
    }

    #[test]
    fn test_aegis_identify_empty_bank() {
        let system = AegisIdentity::face_only();
        let result = system.identify(&make_face_evidence(0.5));
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn test_aegis_identify_no_evidence() {
        let system = AegisIdentity::full();
        let result = system.identify(&BiometricEvidence::empty());
        assert!(result.candidates.is_empty());
    }

    // ---- Forensic Verification tests ----

    #[test]
    fn test_forensic_verify_unknown() {
        let system = AegisIdentity::face_only();
        let (vr, fr) = system.verify_forensic(999, &make_face_evidence(0.5));
        assert!(!vr.is_match);
        assert!(fr.modality_reports.is_empty());
    }

    #[test]
    fn test_forensic_verify_enrolled() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));

        let (vr, fr) = system.verify_forensic(1, &make_face_evidence(0.3));
        assert!(!fr.modality_reports.is_empty());
        assert!(fr.cross_modal_consistency >= 0.0 && fr.cross_modal_consistency <= 1.0);
        assert!(fr.dominant_modality.is_some());
        assert!(vr.match_score.is_finite());
    }

    #[test]
    fn test_forensic_per_modality_breakdown() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));

        let (_vr, fr) = system.verify_forensic(1, &make_face_evidence(0.3));
        for report in &fr.modality_reports {
            assert!(report.raw_score.is_finite());
            assert!(report.uncertainty.is_finite());
            assert!(report.fusion_weight.is_finite());
        }
    }

    #[test]
    fn test_forensic_dimension_contributions() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));

        let (_vr, fr) = system.verify_forensic(1, &make_face_evidence(0.3));
        assert!(fr.top_contributing_dimensions.len() <= 10);
        // Should be sorted by absolute contribution descending
        for window in fr.top_contributing_dimensions.windows(2) {
            assert!(window[0].contribution.abs() >= window[1].contribution.abs());
        }
    }

    #[test]
    fn test_forensic_cross_modal_single_modality() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        let (_vr, fr) = system.verify_forensic(1, &make_face_evidence(0.3));
        // Single modality = perfect consistency
        assert_eq!(fr.cross_modal_consistency, 1.0);
    }

    // ---- Batch Operations tests ----

    #[test]
    fn test_batch_enroll() {
        let mut system = AegisIdentity::face_only();
        let subjects = vec![
            (1, make_face_evidence(0.3)),
            (2, make_face_evidence(0.5)),
            (3, make_face_evidence(0.7)),
        ];
        let results = system.batch_enroll(&subjects);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
        assert_eq!(system.bank.len(), 3);
    }

    #[test]
    fn test_batch_verify() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        system.enroll(2, &make_face_evidence(0.7));

        let claims = vec![
            (1, make_face_evidence(0.3)),
            (2, make_face_evidence(0.7)),
            (999, make_face_evidence(0.5)), // Unknown
        ];
        let results = system.batch_verify(&claims);
        assert_eq!(results.len(), 3);
        assert!(!results[2].is_match); // Unknown should fail
    }

    #[test]
    fn test_batch_identify() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));

        let probes = vec![
            make_face_evidence(0.3),
            make_face_evidence(0.7),
        ];
        let results = system.batch_identify(&probes);
        assert_eq!(results.len(), 2);
    }

    // ---- Drift Detection tests ----

    #[test]
    fn test_drift_no_observations() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        // Only 1 observation — no drift detectable
        let drift = system.detect_drift(1);
        assert!(drift.is_none());
    }

    #[test]
    fn test_drift_same_face() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        system.enroll(1, &make_face_evidence(0.3));

        let drift = system.detect_drift(1);
        // Same face data → minimal or no drift
        if let Some(alert) = drift {
            assert!(alert.drift_magnitude < 0.5, "Same face should have low drift: {}", alert.drift_magnitude);
        }
    }

    #[test]
    fn test_drift_different_face() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        system.enroll(1, &make_face_evidence(0.9));

        let drift = system.detect_drift(1);
        // Different face data should show drift
        if let Some(alert) = drift {
            assert!(alert.drift_magnitude > 0.0);
            assert!(!alert.affected_modalities.is_empty());
            assert!(alert.drift_rate > 0.0);
        }
    }

    #[test]
    fn test_drift_unknown_subject() {
        let system = AegisIdentity::face_only();
        assert!(system.detect_drift(999).is_none());
    }

    #[test]
    fn test_detect_all_drift() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        system.enroll(1, &make_face_evidence(0.3));
        system.enroll(2, &make_face_evidence(0.5));
        system.enroll(2, &make_face_evidence(0.5));

        let alerts = system.detect_all_drift();
        // May or may not have alerts depending on embedding similarity
        for alert in &alerts {
            assert!(alert.drift_magnitude >= 0.0);
        }
    }

    // ---- Quality Assessment tests ----

    #[test]
    fn test_quality_face() {
        let system = AegisIdentity::face_only();
        let quality = system.assess_quality(&make_face_evidence(0.5));
        assert!(quality.overall_score >= 0.0 && quality.overall_score <= 1.0);
        assert!(!quality.modality_scores.is_empty());
    }

    #[test]
    fn test_quality_empty_evidence() {
        let system = AegisIdentity::full();
        let quality = system.assess_quality(&BiometricEvidence::empty());
        assert_eq!(quality.overall_score, 0.0);
        assert!(quality.modality_scores.is_empty());
        assert!(!quality.meets_threshold);
    }

    #[test]
    fn test_quality_with_issues() {
        let system = AegisIdentity::face_only();
        // Zero-valued face should have low quality
        let quality = system.assess_quality(&make_face_evidence(0.0));
        // Quality should be computed regardless
        assert!(quality.overall_score.is_finite());
    }

    // ---- Liveness Detection tests ----

    #[test]
    fn test_liveness_no_sequence() {
        let system = AegisIdentity::face_only();
        let liveness = system.assess_liveness(&make_face_evidence(0.3));
        // No sequence = unknown
        assert_eq!(liveness.liveness_score, 0.5);
    }

    #[test]
    fn test_liveness_with_sequence() {
        let system = AegisIdentity::face_only();
        let frames: Vec<Variable> = (0..5)
            .map(|i| make_face(0.3 + i as f32 * 0.01))
            .collect();
        let evidence = BiometricEvidence::face_sequence(frames);
        let liveness = system.assess_liveness(&evidence);

        assert!(liveness.liveness_score.is_finite());
        assert!(!liveness.modality_liveness.is_empty());
        assert!(liveness.modality_liveness[0].0 == BiometricModality::Face);
    }

    #[test]
    fn test_liveness_no_face_model() {
        let system = AegisIdentity::with_modalities(false, true, false, false);
        let frames = vec![make_face(0.3); 5];
        let evidence = BiometricEvidence::face_sequence(frames);
        let liveness = system.assess_liveness(&evidence);
        assert_eq!(liveness.liveness_score, 0.5); // Unknown
    }

    // ---- Secure Verification tests ----

    #[test]
    fn test_secure_verify_enrolled() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));

        let (vr, quality, liveness) = system.secure_verify(1, &make_face_evidence(0.3));
        assert!(quality.is_some());
        assert!(liveness.is_none()); // No face sequence
        assert!(vr.match_score.is_finite());
    }

    #[test]
    fn test_secure_verify_with_sequence() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));

        let frames: Vec<Variable> = (0..5)
            .map(|i| make_face(0.3 + i as f32 * 0.01))
            .collect();
        let mut evidence = BiometricEvidence::face_sequence(frames);
        evidence.face = Some(make_face(0.3));

        let (_vr, quality, liveness) = system.secure_verify(1, &evidence);
        assert!(quality.is_some());
        assert!(liveness.is_some());
    }

    // ---- Operating Curve tests ----

    #[test]
    fn test_operating_curve_empty() {
        let system = AegisIdentity::face_only();
        let curve = system.compute_operating_curve(10);
        // Empty bank = no data
        assert!(curve.points.is_empty() || curve.eer <= 1.0);
    }

    #[test]
    fn test_operating_curve_with_subjects() {
        let mut system = AegisIdentity::face_only();
        // Enroll several subjects with different data
        for i in 0..5 {
            let val = 0.1 + i as f32 * 0.15;
            system.enroll(i, &make_face_evidence(val));
            // Re-enroll to create originals vs current
            system.enroll(i, &make_face_evidence(val));
        }

        let curve = system.compute_operating_curve(20);
        assert!(curve.eer >= 0.0 && curve.eer <= 1.0);
        assert!(curve.eer_threshold >= 0.0 && curve.eer_threshold <= 1.0);
    }

    // ---- Summary and diagnostics tests ----

    #[test]
    fn test_summary() {
        let system = AegisIdentity::full();
        let summary = system.summary();
        assert!(summary.contains("AegisIdentity"));
        assert!(summary.contains("4 modalities"));
        assert!(summary.contains("params"));
    }

    #[test]
    fn test_summary_after_enrollment() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        let summary = system.summary();
        assert!(summary.contains("1 enrolled"));
    }

    // ---- Cosine similarity tests ----

    #[test]
    fn test_cosine_similarity_self() {
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
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    // ---- Graceful degradation tests ----

    #[test]
    fn test_aegis_graceful_no_evidence() {
        let system = AegisIdentity::full();
        let empty = BiometricEvidence::empty();

        let result = system.verify(1, &empty);
        assert!(!result.is_match);

        let id_result = system.identify(&empty);
        assert!(id_result.candidates.is_empty());
    }

    #[test]
    fn test_aegis_mismatched_modality() {
        // System only has face, but evidence only has voice
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));

        let voice_evidence = BiometricEvidence::voice(make_voice(0.5));
        let result = system.verify(1, &voice_evidence);
        // No matching modality → score 0
        assert_eq!(result.match_score, 0.0);
    }

    // ---- Config preset tests ----

    #[test]
    fn test_high_security_config() {
        let system = AegisIdentity::with_config(BiometricConfig::high_security());
        assert_eq!(system.config().verify_threshold, 0.7);
        assert_eq!(system.config().liveness_threshold, 0.8);
    }

    #[test]
    fn test_convenience_config() {
        let system = AegisIdentity::with_config(BiometricConfig::convenience());
        assert_eq!(system.config().verify_threshold, 0.35);
    }

    // ---- Edge case tests ----

    #[test]
    fn test_remove_enrolled_then_verify() {
        let mut system = AegisIdentity::face_only();
        system.enroll(1, &make_face_evidence(0.3));
        system.bank.remove(1);
        let result = system.verify(1, &make_face_evidence(0.3));
        assert!(!result.is_match);
        assert_eq!(result.match_score, 0.0);
    }

    #[test]
    fn test_enroll_empty_evidence() {
        let mut system = AegisIdentity::face_only();
        let result = system.enroll(1, &BiometricEvidence::empty());
        assert!(!result.success);
        assert!(result.modalities_enrolled.is_empty());
    }
}
