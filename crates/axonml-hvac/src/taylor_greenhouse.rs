//! TAYLOR_GREENHOUSE — Taylor University Greenhouse Controller (~300K params)
//!
//! # File
//! `crates/axonml-hvac/src/taylor_greenhouse.rs`
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

use super::aquilo::concat_variables;

// =============================================================================
// TaylorGreenhouse Model
// =============================================================================

/// Taylor University greenhouse controller model.
///
/// Architecture (Multi-branch with squeeze-excite attention fusion):
/// - Temperature branch:  Linear chain (480 -> 128 -> 64)
/// - Humidity branch:     Linear chain (360 -> 128 -> 64)
/// - Environmental branch: Linear chain (840 -> 256 -> 128 -> 64)
/// - Squeeze-excite:      Fused(192) -> squeeze(32) -> ReLU -> excite(192) -> sigmoid
///                         Element-wise multiply with fused features
/// - Post-SE MLP:         192 -> 128 -> ReLU
///
/// Input: (batch, 1680) — flattened 120 timesteps x 14 features
///   temp_zones(4) + humidity(3) + solar(2) + soil(2) + OAT(1) + wind(1) + CO2(1)
///   Sliced: temp = 120*4=480, humidity = 120*3=360, env = 120*7=840
///
/// Outputs: zone_temps(10), humidity_targets(10), ventilation(10)
pub struct TaylorGreenhouse {
    // --- branches ---
    temp_branch: Sequential,
    humidity_branch: Sequential,
    env_branch: Sequential,
    // --- squeeze-excite ---
    se_squeeze: Linear,
    se_excite: Linear,
    // --- post-SE ---
    post_se: Sequential,
    // --- output heads ---
    zone_temps_head: Linear,
    humidity_targets_head: Linear,
    ventilation_head: Linear,
    training: bool,
}

impl Default for TaylorGreenhouse {
    fn default() -> Self {
        Self::new()
    }
}

impl TaylorGreenhouse {
    /// Creates a new TaylorGreenhouse model.
    pub fn new() -> Self {
        // Temperature branch: 480 -> 128 -> 64
        let temp_branch = Sequential::new()
            .add(Linear::new(480, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(128, 64))
            .add(ReLU);

        // Humidity branch: 360 -> 128 -> 64
        let humidity_branch = Sequential::new()
            .add(Linear::new(360, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(128, 64))
            .add(ReLU);

        // Environmental branch: 840 -> 256 -> 128 -> 64
        let env_branch = Sequential::new()
            .add(Linear::new(840, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(256, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Linear::new(128, 64))
            .add(ReLU);

        // Squeeze-excite on fused 192: squeeze(32) -> ReLU -> excite(192)
        let se_squeeze = Linear::new(192, 32);
        let se_excite = Linear::new(32, 192);

        // Post-SE MLP: 192 -> 128
        let post_se = Sequential::new()
            .add(Linear::new(192, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Dropout::new(0.15));

        // Output heads
        let zone_temps_head = Linear::new(128, 10);
        let humidity_targets_head = Linear::new(128, 10);
        let ventilation_head = Linear::new(128, 10);

        Self {
            temp_branch,
            humidity_branch,
            env_branch,
            se_squeeze,
            se_excite,
            post_se,
            zone_temps_head,
            humidity_targets_head,
            ventilation_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (zone_temps, humidity_targets, ventilation, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable) {
        let shape = input.shape();
        let batch = shape[0];

        // Split input: temp(480) | humidity(360) | env(840) = 1680
        let temp_in = input.narrow(1, 0, 480);
        let hum_in = input.narrow(1, 480, 360);
        let env_in = input.narrow(1, 840, 840);

        // Branch forward passes
        let temp_out = self.temp_branch.forward(&temp_in);       // (batch, 64)
        let hum_out = self.humidity_branch.forward(&hum_in);     // (batch, 64)
        let env_out = self.env_branch.forward(&env_in);          // (batch, 64)

        // Fuse branches: concat -> (batch, 192)
        let fused = concat_variables(&[&temp_out, &hum_out, &env_out], batch);

        // Squeeze-excite attention
        let squeezed = self.se_squeeze.forward(&fused).relu();   // (batch, 32)
        let excited = self.se_excite.forward(&squeezed).sigmoid(); // (batch, 192)
        let attended = &fused * &excited;                        // element-wise

        // Post-SE MLP
        let embedding = self.post_se.forward(&attended);         // (batch, 128)

        // Output heads
        let zone_temps = self.zone_temps_head.forward(&embedding);
        let humidity_targets = self.humidity_targets_head.forward(&embedding);
        let ventilation = self.ventilation_head.forward(&embedding);

        (zone_temps, humidity_targets, ventilation, embedding)
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

impl Module for TaylorGreenhouse {
    fn forward(&self, input: &Variable) -> Variable {
        let (zone_temps, _, _, _) = self.forward_all(input);
        zone_temps
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.temp_branch.parameters());
        params.extend(self.humidity_branch.parameters());
        params.extend(self.env_branch.parameters());
        params.extend(self.se_squeeze.parameters());
        params.extend(self.se_excite.parameters());
        params.extend(self.post_se.parameters());
        params.extend(self.zone_temps_head.parameters());
        params.extend(self.humidity_targets_head.parameters());
        params.extend(self.ventilation_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.temp_branch.named_parameters() {
            params.insert(format!("temp_branch.{n}"), p);
        }
        for (n, p) in self.humidity_branch.named_parameters() {
            params.insert(format!("humidity_branch.{n}"), p);
        }
        for (n, p) in self.env_branch.named_parameters() {
            params.insert(format!("env_branch.{n}"), p);
        }
        for (n, p) in self.se_squeeze.named_parameters() {
            params.insert(format!("se_squeeze.{n}"), p);
        }
        for (n, p) in self.se_excite.named_parameters() {
            params.insert(format!("se_excite.{n}"), p);
        }
        for (n, p) in self.post_se.named_parameters() {
            params.insert(format!("post_se.{n}"), p);
        }
        for (n, p) in self.zone_temps_head.named_parameters() {
            params.insert(format!("zone_temps_head.{n}"), p);
        }
        for (n, p) in self.humidity_targets_head.named_parameters() {
            params.insert(format!("humidity_targets_head.{n}"), p);
        }
        for (n, p) in self.ventilation_head.named_parameters() {
            params.insert(format!("ventilation_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.temp_branch.set_training(training);
        self.humidity_branch.set_training(training);
        self.env_branch.set_training(training);
        self.post_se.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "TaylorGreenhouse"
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
    fn test_taylor_greenhouse_output_shapes() {
        let model = TaylorGreenhouse::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 1680], &[2, 1680]).unwrap(),
            false,
        );
        let (zone_temps, hum_targets, vent, emb) = model.forward_all(&input);

        assert_eq!(zone_temps.shape(), vec![2, 10]);
        assert_eq!(hum_targets.shape(), vec![2, 10]);
        assert_eq!(vent.shape(), vec![2, 10]);
        assert_eq!(emb.shape(), vec![2, 128]);
    }

    #[test]
    fn test_taylor_greenhouse_parameter_count() {
        let model = TaylorGreenhouse::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 200_000 && total < 400_000,
            "TaylorGreenhouse has {} params, expected ~300K",
            total
        );
    }

    #[test]
    fn test_taylor_greenhouse_forward_module_trait() {
        let model = TaylorGreenhouse::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 1680], &[4, 1680]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 10]);
    }
}
