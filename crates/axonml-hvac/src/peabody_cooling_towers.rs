//! PEABODY_COOLING_TOWERS — Peabody Retirement Community Cooling Tower Controller (~350K params)
//!
//! # File
//! `crates/axonml-hvac/src/peabody_cooling_towers.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 29, 2026 1:30 AM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm1d, Dropout, Linear, Module, Parameter, ReLU, Sequential};

// =============================================================================
// PeabodyCoolingTowers Model
// =============================================================================

/// Peabody Retirement Community cooling tower controller.
///
/// Architecture (Multi-scale parallel paths with learned attention fusion):
/// - Path A (narrow/deep):   1920 → 128 → 128 → 64  (3 layers, narrow)
/// - Path B (medium):        1920 → 256 → 128        (2 layers, medium)
/// - Path C (wide/shallow):  1920 → 512 → 128        (2 layers, wide first layer)
/// - Attention gate:         1920 → 64 → 3            (softmax over 3 paths)
///   Each path output is 128-dim; weighted sum by attention → 128
/// - Post-fusion MLP:        128 → 192 → 128 with BN + Dropout
///
/// Input: (batch, 1920) — flattened 120 timesteps x 16 features
///   water_in(2) + water_out(2) + wet_bulb(1) + dry_bulb(1) + OAT(1)
///   + fan_speed(4) + valve_pos(2) + flow(1) + sump_level(1) + basin_temp(1)
///
/// Outputs: water_prediction(10), fan_staging(10)
pub struct PeabodyCoolingTowers {
    // Multi-scale parallel paths
    path_a: Sequential,  // narrow/deep
    path_b: Sequential,  // medium
    path_c: Sequential,  // wide/shallow
    // Learned attention gate
    attn_gate: Sequential,
    attn_softmax: Linear,
    // Post-fusion
    post_fusion: Sequential,
    // Output heads
    water_pred_head: Linear,
    fan_staging_head: Linear,
    training: bool,
}

impl Default for PeabodyCoolingTowers {
    fn default() -> Self {
        Self::new()
    }
}

impl PeabodyCoolingTowers {
    /// Creates a new PeabodyCoolingTowers model.
    pub fn new() -> Self {
        // Path A: narrow/deep — 1920 → 128 → 128 → 128
        let path_a = Sequential::new()
            .add(Linear::new(1920, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Dropout::new(0.15))
            .add(Linear::new(128, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Linear::new(128, 128))
            .add(ReLU);

        // Path B: medium — 1920 → 256 → 128
        let path_b = Sequential::new()
            .add(Linear::new(1920, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(256, 128))
            .add(ReLU);

        // Path C: wide/shallow — 1920 → 512 → 128
        let path_c = Sequential::new()
            .add(Linear::new(1920, 512))
            .add(BatchNorm1d::new(512))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(512, 128))
            .add(ReLU);

        // Attention gate: input → 64 → 3 weights (one per path)
        let attn_gate = Sequential::new()
            .add(Linear::new(1920, 64))
            .add(ReLU);
        let attn_softmax = Linear::new(64, 3);

        // Post-fusion MLP: 128 → 192 → 128
        let post_fusion = Sequential::new()
            .add(Linear::new(128, 192))
            .add(BatchNorm1d::new(192))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(192, 128))
            .add(ReLU);

        // Output heads
        let water_pred_head = Linear::new(128, 10);
        let fan_staging_head = Linear::new(128, 10);

        Self {
            path_a,
            path_b,
            path_c,
            attn_gate,
            attn_softmax,
            post_fusion,
            water_pred_head,
            fan_staging_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (water_prediction, fan_staging, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable) {
        // Multi-scale parallel paths
        let out_a = self.path_a.forward(input);  // (batch, 128)
        let out_b = self.path_b.forward(input);  // (batch, 128)
        let out_c = self.path_c.forward(input);  // (batch, 128)

        // Attention gate: compute 3 weights via softmax
        let gate_hidden = self.attn_gate.forward(input);         // (batch, 64)
        let gate_logits = self.attn_softmax.forward(&gate_hidden); // (batch, 3)
        let gate_weights = gate_logits.softmax(1);                 // (batch, 3)

        // Extract per-path weights: narrow on dim 1
        let w_a = gate_weights.narrow(1, 0, 1); // (batch, 1)
        let w_b = gate_weights.narrow(1, 1, 1); // (batch, 1)
        let w_c = gate_weights.narrow(1, 2, 1); // (batch, 1)

        // Weighted sum: each weight broadcasts across 128 features
        let weighted_a = &out_a * &w_a; // (batch, 128)
        let weighted_b = &out_b * &w_b; // (batch, 128)
        let weighted_c = &out_c * &w_c; // (batch, 128)

        // Sum the weighted paths
        let fused = &(&weighted_a + &weighted_b) + &weighted_c; // (batch, 128)

        // Post-fusion MLP
        let embedding = self.post_fusion.forward(&fused); // (batch, 128)

        // Output heads
        let water_pred = self.water_pred_head.forward(&embedding);
        let fan_staging = self.fan_staging_head.forward(&embedding);

        (water_pred, fan_staging, embedding)
    }

    /// Returns the embedding dimension for downstream consumers.
    pub fn embedding_dim() -> usize {
        128
    }

    /// Returns total output dimension (10 + 10 = 20).
    pub fn output_dim() -> usize {
        20
    }
}

impl Module for PeabodyCoolingTowers {
    fn forward(&self, input: &Variable) -> Variable {
        let (water_pred, _, _) = self.forward_all(input);
        water_pred
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.path_a.parameters());
        params.extend(self.path_b.parameters());
        params.extend(self.path_c.parameters());
        params.extend(self.attn_gate.parameters());
        params.extend(self.attn_softmax.parameters());
        params.extend(self.post_fusion.parameters());
        params.extend(self.water_pred_head.parameters());
        params.extend(self.fan_staging_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.path_a.named_parameters() {
            params.insert(format!("path_a.{n}"), p);
        }
        for (n, p) in self.path_b.named_parameters() {
            params.insert(format!("path_b.{n}"), p);
        }
        for (n, p) in self.path_c.named_parameters() {
            params.insert(format!("path_c.{n}"), p);
        }
        for (n, p) in self.attn_gate.named_parameters() {
            params.insert(format!("attn_gate.{n}"), p);
        }
        for (n, p) in self.attn_softmax.named_parameters() {
            params.insert(format!("attn_softmax.{n}"), p);
        }
        for (n, p) in self.post_fusion.named_parameters() {
            params.insert(format!("post_fusion.{n}"), p);
        }
        for (n, p) in self.water_pred_head.named_parameters() {
            params.insert(format!("water_pred_head.{n}"), p);
        }
        for (n, p) in self.fan_staging_head.named_parameters() {
            params.insert(format!("fan_staging_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.path_a.set_training(training);
        self.path_b.set_training(training);
        self.path_c.set_training(training);
        self.post_fusion.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "PeabodyCoolingTowers"
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
    fn test_peabody_cooling_towers_output_shapes() {
        let model = PeabodyCoolingTowers::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 1920], &[2, 1920]).unwrap(),
            false,
        );
        let (water, fan, emb) = model.forward_all(&input);

        assert_eq!(water.shape(), vec![2, 10]);
        assert_eq!(fan.shape(), vec![2, 10]);
        assert_eq!(emb.shape(), vec![2, 128]);
    }

    #[test]
    fn test_peabody_cooling_towers_parameter_count() {
        let model = PeabodyCoolingTowers::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 250_000 && total < 450_000,
            "PeabodyCoolingTowers has {} params, expected ~350K",
            total
        );
    }

    #[test]
    fn test_peabody_cooling_towers_forward_module_trait() {
        let model = PeabodyCoolingTowers::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 1920], &[4, 1920]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 10]);
    }
}
