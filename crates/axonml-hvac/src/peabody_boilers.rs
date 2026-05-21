//! PEABODY_BOILERS — Peabody Retirement Community Boiler Plant Controller (~450K params)
//!
//! # File
//! `crates/axonml-hvac/src/peabody_boilers.rs`
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
// PeabodyBoilers Model
// =============================================================================

/// Peabody Retirement Community boiler plant controller.
///
/// Architecture (Deep MLP with progressive temporal compression + per-head expansion):
/// - Compression stage 1:  2400 → 512  with BN + ReLU + Dropout(0.3)
/// - Compression stage 2:  512  → 256  with BN + ReLU + Dropout(0.25)
/// - Compression stage 3:  256  → 128  with BN + ReLU + Dropout(0.2)
/// - Shared embedding extracted at 128-dim
/// - Efficiency head:      128  → 64 → 10  (2-layer MLP)
/// - Staging head:         128  → 64 → 10  (2-layer MLP)
/// - Safety head:          128  → 64 → 10  (2-layer MLP)
///
/// Input: (batch, 2400) — flattened 120 timesteps x 20 features
///   supply_temp(2) + return_temp(2) + flue_temp(2) + stack_temp(1)
///   + gas_valve(2) + flame_signal(2) + OAT(1) + inlet_water(1)
///   + pump_speed(2) + DHW_temp(1) + circ_temp(1) + pressure(2)
///
/// Outputs: efficiency(10), staging(10), safety(10)
pub struct PeabodyBoilers {
    // Progressive compression stages
    compress1: Sequential,
    compress2: Sequential,
    compress3: Sequential,
    // Per-head expansion MLPs
    efficiency_mlp: Sequential,
    efficiency_head: Linear,
    staging_mlp: Sequential,
    staging_head: Linear,
    safety_mlp: Sequential,
    safety_head: Linear,
    training: bool,
}

impl Default for PeabodyBoilers {
    fn default() -> Self {
        Self::new()
    }
}

impl PeabodyBoilers {
    /// Creates a new PeabodyBoilers model.
    pub fn new() -> Self {
        // Progressive compression: 2400 → 512 → 256 → 128
        let compress1 = Sequential::new()
            .add(Linear::new(2400, 512))
            .add(BatchNorm1d::new(512))
            .add(ReLU)
            .add(Dropout::new(0.3));

        let compress2 = Sequential::new()
            .add(Linear::new(512, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU)
            .add(Dropout::new(0.25));

        let compress3 = Sequential::new()
            .add(Linear::new(256, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Dropout::new(0.2));

        // Efficiency head: 128 → 64 → 10
        let efficiency_mlp = Sequential::new()
            .add(Linear::new(128, 64))
            .add(BatchNorm1d::new(64))
            .add(ReLU);
        let efficiency_head = Linear::new(64, 10);

        // Staging head: 128 → 64 → 10
        let staging_mlp = Sequential::new()
            .add(Linear::new(128, 64))
            .add(BatchNorm1d::new(64))
            .add(ReLU);
        let staging_head = Linear::new(64, 10);

        // Safety head: 128 → 64 → 10
        let safety_mlp = Sequential::new()
            .add(Linear::new(128, 64))
            .add(BatchNorm1d::new(64))
            .add(ReLU);
        let safety_head = Linear::new(64, 10);

        Self {
            compress1,
            compress2,
            compress3,
            efficiency_mlp,
            efficiency_head,
            staging_mlp,
            staging_head,
            safety_mlp,
            safety_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (efficiency, staging, safety, embedding)
    pub fn forward_all(&self, input: &Variable) -> (Variable, Variable, Variable, Variable) {
        // Progressive compression
        let h = self.compress1.forward(input); // (batch, 512)
        let h = self.compress2.forward(&h); // (batch, 256)
        let embedding = self.compress3.forward(&h); // (batch, 128)

        // Per-head expansion
        let eff_h = self.efficiency_mlp.forward(&embedding); // (batch, 64)
        let efficiency = self.efficiency_head.forward(&eff_h); // (batch, 10)

        let stg_h = self.staging_mlp.forward(&embedding); // (batch, 64)
        let staging = self.staging_head.forward(&stg_h); // (batch, 10)

        let saf_h = self.safety_mlp.forward(&embedding); // (batch, 64)
        let safety = self.safety_head.forward(&saf_h); // (batch, 10)

        (efficiency, staging, safety, embedding)
    }

    /// Returns the embedding dimension for downstream consumers.
    pub fn embedding_dim() -> usize {
        128
    }

    /// Returns total output dimension (10 + 10 + 10 = 30).
    pub fn output_dim() -> usize {
        30
    }
}

impl Module for PeabodyBoilers {
    fn forward(&self, input: &Variable) -> Variable {
        let (efficiency, _, _, _) = self.forward_all(input);
        efficiency
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.compress1.parameters());
        params.extend(self.compress2.parameters());
        params.extend(self.compress3.parameters());
        params.extend(self.efficiency_mlp.parameters());
        params.extend(self.efficiency_head.parameters());
        params.extend(self.staging_mlp.parameters());
        params.extend(self.staging_head.parameters());
        params.extend(self.safety_mlp.parameters());
        params.extend(self.safety_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.compress1.named_parameters() {
            params.insert(format!("compress1.{n}"), p);
        }
        for (n, p) in self.compress2.named_parameters() {
            params.insert(format!("compress2.{n}"), p);
        }
        for (n, p) in self.compress3.named_parameters() {
            params.insert(format!("compress3.{n}"), p);
        }
        for (n, p) in self.efficiency_mlp.named_parameters() {
            params.insert(format!("efficiency_mlp.{n}"), p);
        }
        for (n, p) in self.efficiency_head.named_parameters() {
            params.insert(format!("efficiency_head.{n}"), p);
        }
        for (n, p) in self.staging_mlp.named_parameters() {
            params.insert(format!("staging_mlp.{n}"), p);
        }
        for (n, p) in self.staging_head.named_parameters() {
            params.insert(format!("staging_head.{n}"), p);
        }
        for (n, p) in self.safety_mlp.named_parameters() {
            params.insert(format!("safety_mlp.{n}"), p);
        }
        for (n, p) in self.safety_head.named_parameters() {
            params.insert(format!("safety_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.compress1.set_training(training);
        self.compress2.set_training(training);
        self.compress3.set_training(training);
        self.efficiency_mlp.set_training(training);
        self.staging_mlp.set_training(training);
        self.safety_mlp.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "PeabodyBoilers"
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
    fn test_peabody_boilers_output_shapes() {
        let model = PeabodyBoilers::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 2400], &[2, 2400]).unwrap(),
            false,
        );
        let (eff, stg, saf, emb) = model.forward_all(&input);

        assert_eq!(eff.shape(), vec![2, 10]);
        assert_eq!(stg.shape(), vec![2, 10]);
        assert_eq!(saf.shape(), vec![2, 10]);
        assert_eq!(emb.shape(), vec![2, 128]);
    }

    #[test]
    fn test_peabody_boilers_parameter_count() {
        let model = PeabodyBoilers::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 350_000 && total < 550_000,
            "PeabodyBoilers has {} params, expected ~450K",
            total
        );
    }

    #[test]
    fn test_peabody_boilers_forward_module_trait() {
        let model = PeabodyBoilers::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 2400], &[4, 2400]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 10]);
    }
}
