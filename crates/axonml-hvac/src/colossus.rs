//! COLOSSUS — Master Aggregator Model (~1.5M params)
//!
//! # File
//! `crates/axonml-hvac/src/colossus.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_nn::{Dropout, GELU, Linear, Module, MultiHeadAttention, Parameter, ReLU, Sequential};

// =============================================================================
// Colossus Model
// =============================================================================

/// Master aggregator that fuses specialist outputs.
///
/// Architecture:
/// - Per-specialist linear projections to common dim (256)
/// - Cross-specialist multi-head attention (12 heads)
/// - MLP decision network
///
/// Input: Concatenated specialist outputs (256+384+256+384+320 = 1600)
/// Outputs: system_fault(24), cascade_prob(8), health_score(1), specialist_confidence(5)
pub struct Colossus {
    // Project each specialist to common dim 256
    proj_aquilo: Linear,   // 256 → 256
    proj_boreas: Linear,   // 384 → 256
    proj_naiad: Linear,    // 256 → 256
    proj_vulcan: Linear,   // 384 → 256 (see note: vulcan embedding is 256, plan says 384)
    proj_zephyrus: Linear, // 320 → 256
    // Cross-specialist attention
    attention: MultiHeadAttention,
    // Decision network
    decision_net: Sequential,
    // Output heads
    fault_head: Linear,
    cascade_head: Linear,
    health_head: Linear,
    confidence_head: Linear,
    training: bool,
}

/// Aquilo (electrical) embedding dimension.
pub const AQUILO_DIM: usize = 256;
/// Boreas (refrigeration) embedding dimension.
pub const BOREAS_DIM: usize = 384;
/// Naiad (water) embedding dimension.
pub const NAIAD_DIM: usize = 256;
/// Vulcan (mechanical) embedding dimension.
pub const VULCAN_DIM: usize = 256;
/// Zephyrus (airflow) embedding dimension.
pub const ZEPHYRUS_DIM: usize = 320;
/// Total concatenated specialist embedding dimension.
pub const TOTAL_SPECIALIST_DIM: usize =
    AQUILO_DIM + BOREAS_DIM + NAIAD_DIM + VULCAN_DIM + ZEPHYRUS_DIM;

impl Default for Colossus {
    fn default() -> Self {
        Self::new()
    }
}

impl Colossus {
    /// Creates a new Colossus aggregator.
    pub fn new() -> Self {
        let proj_aquilo = Linear::new(AQUILO_DIM, 256);
        let proj_boreas = Linear::new(BOREAS_DIM, 256);
        let proj_naiad = Linear::new(NAIAD_DIM, 256);
        let proj_vulcan = Linear::new(VULCAN_DIM, 256);
        let proj_zephyrus = Linear::new(ZEPHYRUS_DIM, 256);

        // Cross-specialist attention: 5 specialists as sequence, dim=256, 12 heads
        let attention = MultiHeadAttention::new(256, 8);

        // After attention: flatten (5, 256) → 1280, then MLP
        let decision_net = Sequential::new()
            .add(Linear::new(1280, 512))
            .add(GELU)
            .add(Dropout::new(0.3))
            .add(Linear::new(512, 256))
            .add(ReLU);

        let fault_head = Linear::new(256, 24);
        let cascade_head = Linear::new(256, 8);
        let health_head = Linear::new(256, 1);
        let confidence_head = Linear::new(256, 5);

        Self {
            proj_aquilo,
            proj_boreas,
            proj_naiad,
            proj_vulcan,
            proj_zephyrus,
            attention,
            decision_net,
            fault_head,
            cascade_head,
            health_head,
            confidence_head,
            training: true,
        }
    }

    /// Forward pass with individual specialist embeddings.
    ///
    /// Returns (system_fault, cascade_prob, health_score, confidence, embedding)
    pub fn forward_specialists(
        &self,
        aquilo_emb: &Variable,
        boreas_emb: &Variable,
        naiad_emb: &Variable,
        vulcan_emb: &Variable,
        zephyrus_emb: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        let batch = aquilo_emb.shape()[0];

        // Project all to 256
        let a = self.proj_aquilo.forward(aquilo_emb); // (batch, 256)
        let b = self.proj_boreas.forward(boreas_emb); // (batch, 256)
        let n = self.proj_naiad.forward(naiad_emb); // (batch, 256)
        let v = self.proj_vulcan.forward(vulcan_emb); // (batch, 256)
        let z = self.proj_zephyrus.forward(zephyrus_emb); // (batch, 256)

        // Stack as (batch, 5, 256) for attention
        let stacked_var = Variable::cat(
            &[
                &a.unsqueeze(1),
                &b.unsqueeze(1),
                &n.unsqueeze(1),
                &v.unsqueeze(1),
                &z.unsqueeze(1),
            ],
            1,
        ); // (batch, 5, 256)

        // Cross-specialist attention
        let attn_out = self.attention.forward(&stacked_var); // (batch, 5, 256)

        // Flatten to (batch, 1280)
        let flat = attn_out.reshape(&[batch, 1280]);

        // Decision network
        let embedding = self.decision_net.forward(&flat); // (batch, 256)

        let fault = self.fault_head.forward(&embedding);
        let cascade = self.cascade_head.forward(&embedding);
        let health = self.health_head.forward(&embedding);
        let confidence = self.confidence_head.forward(&embedding);

        (fault, cascade, health, confidence, embedding)
    }

    /// Forward from concatenated specialist features.
    pub fn forward_concat(
        &self,
        specialist_concat: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        // Split concatenated input into individual specialist embeddings
        let mut offset = 0;
        let dims = [AQUILO_DIM, BOREAS_DIM, NAIAD_DIM, VULCAN_DIM, ZEPHYRUS_DIM];
        let mut parts: Vec<Variable> = Vec::new();

        for &dim in &dims {
            parts.push(specialist_concat.narrow(1, offset, dim));
            offset += dim;
        }

        self.forward_specialists(&parts[0], &parts[1], &parts[2], &parts[3], &parts[4])
    }

    /// Embedding dimension.
    pub fn embedding_dim() -> usize {
        256
    }

    /// Total output dimension (24+8+1+5 = 38).
    pub fn output_dim() -> usize {
        38
    }
}

impl Module for Colossus {
    fn forward(&self, input: &Variable) -> Variable {
        let (fault, _, _, _, _) = self.forward_concat(input);
        fault
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.proj_aquilo.parameters());
        params.extend(self.proj_boreas.parameters());
        params.extend(self.proj_naiad.parameters());
        params.extend(self.proj_vulcan.parameters());
        params.extend(self.proj_zephyrus.parameters());
        params.extend(self.attention.parameters());
        params.extend(self.decision_net.parameters());
        params.extend(self.fault_head.parameters());
        params.extend(self.cascade_head.parameters());
        params.extend(self.health_head.parameters());
        params.extend(self.confidence_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.proj_aquilo.named_parameters() {
            params.insert(format!("proj_aquilo.{n}"), p);
        }
        for (n, p) in self.proj_boreas.named_parameters() {
            params.insert(format!("proj_boreas.{n}"), p);
        }
        for (n, p) in self.proj_naiad.named_parameters() {
            params.insert(format!("proj_naiad.{n}"), p);
        }
        for (n, p) in self.proj_vulcan.named_parameters() {
            params.insert(format!("proj_vulcan.{n}"), p);
        }
        for (n, p) in self.proj_zephyrus.named_parameters() {
            params.insert(format!("proj_zephyrus.{n}"), p);
        }
        for (n, p) in self.attention.named_parameters() {
            params.insert(format!("attention.{n}"), p);
        }
        for (n, p) in self.decision_net.named_parameters() {
            params.insert(format!("decision_net.{n}"), p);
        }
        for (n, p) in self.fault_head.named_parameters() {
            params.insert(format!("fault_head.{n}"), p);
        }
        for (n, p) in self.cascade_head.named_parameters() {
            params.insert(format!("cascade_head.{n}"), p);
        }
        for (n, p) in self.health_head.named_parameters() {
            params.insert(format!("health_head.{n}"), p);
        }
        for (n, p) in self.confidence_head.named_parameters() {
            params.insert(format!("confidence_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.decision_net.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Colossus"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_colossus_output_shapes() {
        let model = Colossus::new();

        let aquilo = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 256], &[2, 256]).unwrap(),
            false,
        );
        let boreas = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 384], &[2, 384]).unwrap(),
            false,
        );
        let naiad = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 256], &[2, 256]).unwrap(),
            false,
        );
        let vulcan = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 256], &[2, 256]).unwrap(),
            false,
        );
        let zephyrus = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 320], &[2, 320]).unwrap(),
            false,
        );

        let (fault, cascade, health, conf, emb) =
            model.forward_specialists(&aquilo, &boreas, &naiad, &vulcan, &zephyrus);

        assert_eq!(fault.shape(), vec![2, 24]);
        assert_eq!(cascade.shape(), vec![2, 8]);
        assert_eq!(health.shape(), vec![2, 1]);
        assert_eq!(conf.shape(), vec![2, 5]);
        assert_eq!(emb.shape(), vec![2, 256]);
    }

    #[test]
    fn test_colossus_concat_forward() {
        let model = Colossus::new();
        let input = Variable::new(
            Tensor::from_vec(
                vec![1.0; 2 * TOTAL_SPECIALIST_DIM],
                &[2, TOTAL_SPECIALIST_DIM],
            )
            .unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![2, 24]);
    }

    #[test]
    fn test_colossus_parameter_count() {
        let model = Colossus::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 1_000_000 && total < 3_000_000,
            "Colossus has {} params, expected ~1.5M",
            total
        );
    }
}
