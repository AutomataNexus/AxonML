//! TAYLOR_NATORIUM — Taylor University Natorium Climate Controller (~400K params)
//!
//! # File
//! `crates/axonml-hvac/src/taylor_natorium.rs`
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
// Residual MLP Block
// =============================================================================

/// A single residual MLP block: Linear -> BN -> ReLU -> Linear -> BN + skip.
///
/// When `in_dim != out_dim` a learned projection shortcut is used.
struct ResidualMlpBlock {
    layer1: Linear,
    bn1: BatchNorm1d,
    layer2: Linear,
    bn2: BatchNorm1d,
    shortcut: Option<Linear>,
}

impl ResidualMlpBlock {
    fn new(in_dim: usize, hidden: usize, out_dim: usize) -> Self {
        let shortcut = if in_dim != out_dim {
            Some(Linear::new(in_dim, out_dim))
        } else {
            None
        };
        Self {
            layer1: Linear::new(in_dim, hidden),
            bn1: BatchNorm1d::new(hidden),
            layer2: Linear::new(hidden, out_dim),
            bn2: BatchNorm1d::new(out_dim),
            shortcut,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let h = self.layer1.forward(x);
        let h = self.bn1.forward(&h).relu();
        let h = self.layer2.forward(&h);
        let h = self.bn2.forward(&h);
        let skip = match &self.shortcut {
            Some(proj) => proj.forward(x),
            None => x.clone(),
        };
        (&h + &skip).relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.layer1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.layer2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(proj) = &self.shortcut {
            params.extend(proj.parameters());
        }
        params
    }

    fn named_parameters(&self, prefix: &str) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.layer1.named_parameters() {
            params.insert(format!("{prefix}.layer1.{n}"), p);
        }
        for (n, p) in self.bn1.named_parameters() {
            params.insert(format!("{prefix}.bn1.{n}"), p);
        }
        for (n, p) in self.layer2.named_parameters() {
            params.insert(format!("{prefix}.layer2.{n}"), p);
        }
        for (n, p) in self.bn2.named_parameters() {
            params.insert(format!("{prefix}.bn2.{n}"), p);
        }
        if let Some(proj) = &self.shortcut {
            for (n, p) in proj.named_parameters() {
                params.insert(format!("{prefix}.shortcut.{n}"), p);
            }
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.bn1.set_training(training);
        self.bn2.set_training(training);
    }
}

// =============================================================================
// TaylorNatorium Model
// =============================================================================

/// Taylor University natorium (indoor pool) climate controller.
///
/// Architecture (Deep residual MLP with depthwise-style mixing):
/// - Input projection:  2160 -> 512 with BN + ReLU
/// - ResBlock 1:        512 -> 384 -> 512 (identity skip)
/// - ResBlock 2:        512 -> 384 -> 512 (identity skip)
/// - ResBlock 3:        512 -> 256 -> 256 (projection skip)
/// - ResBlock 4:        256 -> 192 -> 256 (identity skip)
/// - Depthwise mixing:  256 -> 128 -> 256 (two parallel MLPs, summed)
/// - Final:             256 -> 128 with Dropout
///
/// Input: (batch, 2160) — flattened 120 timesteps x 18 features
///   pool_temp(2) + air_temp(3) + humidity(3) + chlorine(1) + pH(1) + ORP(1)
///   + exhaust_fans(2) + supply_fans(2) + dampers(2) + OAT(1)
///
/// Outputs: climate(12), ventilation(12), chemical_dosing(12)
pub struct TaylorNatorium {
    input_proj: Sequential,
    res_block1: ResidualMlpBlock,
    res_block2: ResidualMlpBlock,
    res_block3: ResidualMlpBlock,
    res_block4: ResidualMlpBlock,
    // Depthwise-style mixing: two parallel paths summed
    mix_path_a: Sequential,
    mix_path_b: Sequential,
    final_net: Sequential,
    // Output heads
    climate_head: Linear,
    ventilation_head: Linear,
    dosing_head: Linear,
    training: bool,
}

impl Default for TaylorNatorium {
    fn default() -> Self {
        Self::new()
    }
}

impl TaylorNatorium {
    /// Creates a new TaylorNatorium model.
    pub fn new() -> Self {
        // Input projection: 2160 -> 512
        let input_proj = Sequential::new()
            .add(Linear::new(2160, 512))
            .add(BatchNorm1d::new(512))
            .add(ReLU);

        // Residual blocks
        let res_block1 = ResidualMlpBlock::new(512, 384, 512);
        let res_block2 = ResidualMlpBlock::new(512, 384, 512);
        let res_block3 = ResidualMlpBlock::new(512, 256, 256); // downsample
        let res_block4 = ResidualMlpBlock::new(256, 192, 256);

        // Depthwise-style mixing: two parallel 256->128->256 paths, summed
        let mix_path_a = Sequential::new()
            .add(Linear::new(256, 128))
            .add(ReLU)
            .add(Linear::new(128, 256));

        let mix_path_b = Sequential::new()
            .add(Linear::new(256, 128))
            .add(ReLU)
            .add(Linear::new(128, 256));

        // Final compression: 256 -> 128
        let final_net = Sequential::new()
            .add(Linear::new(256, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Dropout::new(0.2));

        // Output heads
        let climate_head = Linear::new(128, 12);
        let ventilation_head = Linear::new(128, 12);
        let dosing_head = Linear::new(128, 12);

        Self {
            input_proj,
            res_block1,
            res_block2,
            res_block3,
            res_block4,
            mix_path_a,
            mix_path_b,
            final_net,
            climate_head,
            ventilation_head,
            dosing_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (climate, ventilation, chemical_dosing, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable) {
        // Input projection
        let h = self.input_proj.forward(input);          // (batch, 512)

        // Residual blocks
        let h = self.res_block1.forward(&h);             // (batch, 512)
        let h = self.res_block2.forward(&h);             // (batch, 512)
        let h = self.res_block3.forward(&h);             // (batch, 256)
        let h = self.res_block4.forward(&h);             // (batch, 256)

        // Depthwise-style mixing: sum of two parallel paths
        let mix_a = self.mix_path_a.forward(&h);         // (batch, 256)
        let mix_b = self.mix_path_b.forward(&h);         // (batch, 256)
        let mixed = (&mix_a + &mix_b).relu();            // (batch, 256)

        // Final compression
        let embedding = self.final_net.forward(&mixed);  // (batch, 128)

        // Output heads
        let climate = self.climate_head.forward(&embedding);
        let ventilation = self.ventilation_head.forward(&embedding);
        let dosing = self.dosing_head.forward(&embedding);

        (climate, ventilation, dosing, embedding)
    }

    /// Returns the embedding dimension for downstream consumers.
    pub fn embedding_dim() -> usize {
        128
    }

    /// Returns total output dimension (12 + 12 + 12 = 36).
    pub fn output_dim() -> usize {
        36
    }
}

impl Module for TaylorNatorium {
    fn forward(&self, input: &Variable) -> Variable {
        let (climate, _, _, _) = self.forward_all(input);
        climate
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.input_proj.parameters());
        params.extend(self.res_block1.parameters());
        params.extend(self.res_block2.parameters());
        params.extend(self.res_block3.parameters());
        params.extend(self.res_block4.parameters());
        params.extend(self.mix_path_a.parameters());
        params.extend(self.mix_path_b.parameters());
        params.extend(self.final_net.parameters());
        params.extend(self.climate_head.parameters());
        params.extend(self.ventilation_head.parameters());
        params.extend(self.dosing_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.input_proj.named_parameters() {
            params.insert(format!("input_proj.{n}"), p);
        }
        params.extend(self.res_block1.named_parameters("res_block1"));
        params.extend(self.res_block2.named_parameters("res_block2"));
        params.extend(self.res_block3.named_parameters("res_block3"));
        params.extend(self.res_block4.named_parameters("res_block4"));
        for (n, p) in self.mix_path_a.named_parameters() {
            params.insert(format!("mix_path_a.{n}"), p);
        }
        for (n, p) in self.mix_path_b.named_parameters() {
            params.insert(format!("mix_path_b.{n}"), p);
        }
        for (n, p) in self.final_net.named_parameters() {
            params.insert(format!("final_net.{n}"), p);
        }
        for (n, p) in self.climate_head.named_parameters() {
            params.insert(format!("climate_head.{n}"), p);
        }
        for (n, p) in self.ventilation_head.named_parameters() {
            params.insert(format!("ventilation_head.{n}"), p);
        }
        for (n, p) in self.dosing_head.named_parameters() {
            params.insert(format!("dosing_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.input_proj.set_training(training);
        self.res_block1.set_training(training);
        self.res_block2.set_training(training);
        self.res_block3.set_training(training);
        self.res_block4.set_training(training);
        self.final_net.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "TaylorNatorium"
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
    fn test_taylor_natorium_output_shapes() {
        let model = TaylorNatorium::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 2160], &[2, 2160]).unwrap(),
            false,
        );
        let (climate, vent, dosing, emb) = model.forward_all(&input);

        assert_eq!(climate.shape(), vec![2, 12]);
        assert_eq!(vent.shape(), vec![2, 12]);
        assert_eq!(dosing.shape(), vec![2, 12]);
        assert_eq!(emb.shape(), vec![2, 128]);
    }

    #[test]
    fn test_taylor_natorium_parameter_count() {
        let model = TaylorNatorium::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 300_000 && total < 500_000,
            "TaylorNatorium has {} params, expected ~400K",
            total
        );
    }

    #[test]
    fn test_taylor_natorium_forward_module_trait() {
        let model = TaylorNatorium::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 2160], &[4, 2160]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 12]);
    }
}
