//! TAYLOR_CHILLER — Taylor University Chiller Plant Controller (~500K params)
//!
//! # File
//! `crates/axonml-hvac/src/taylor_chiller.rs`
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
// Bottleneck Block
// =============================================================================

/// Bottleneck MLP block: Linear(wide→narrow) → BN → ReLU → Linear(narrow→wide) + residual.
///
/// Compresses features to a narrow bottleneck then expands back, with a skip
/// connection from input to output. When `in_dim != out_dim` a projection
/// shortcut is used.
struct BottleneckBlock {
    compress: Linear,
    bn_mid: BatchNorm1d,
    expand: Linear,
    bn_out: BatchNorm1d,
    shortcut: Option<Linear>,
}

impl BottleneckBlock {
    fn new(in_dim: usize, bottleneck: usize, out_dim: usize) -> Self {
        let shortcut = if in_dim != out_dim {
            Some(Linear::new(in_dim, out_dim))
        } else {
            None
        };
        Self {
            compress: Linear::new(in_dim, bottleneck),
            bn_mid: BatchNorm1d::new(bottleneck),
            expand: Linear::new(bottleneck, out_dim),
            bn_out: BatchNorm1d::new(out_dim),
            shortcut,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let h = self.compress.forward(x);
        let h = self.bn_mid.forward(&h).relu();
        let h = self.expand.forward(&h);
        let h = self.bn_out.forward(&h);
        let skip = match &self.shortcut {
            Some(proj) => proj.forward(x),
            None => x.clone(),
        };
        (&h + &skip).relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.compress.parameters());
        params.extend(self.bn_mid.parameters());
        params.extend(self.expand.parameters());
        params.extend(self.bn_out.parameters());
        if let Some(proj) = &self.shortcut {
            params.extend(proj.parameters());
        }
        params
    }

    fn named_parameters(&self, prefix: &str) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.compress.named_parameters() {
            params.insert(format!("{prefix}.compress.{n}"), p);
        }
        for (n, p) in self.bn_mid.named_parameters() {
            params.insert(format!("{prefix}.bn_mid.{n}"), p);
        }
        for (n, p) in self.expand.named_parameters() {
            params.insert(format!("{prefix}.expand.{n}"), p);
        }
        for (n, p) in self.bn_out.named_parameters() {
            params.insert(format!("{prefix}.bn_out.{n}"), p);
        }
        if let Some(proj) = &self.shortcut {
            for (n, p) in proj.named_parameters() {
                params.insert(format!("{prefix}.shortcut.{n}"), p);
            }
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.bn_mid.set_training(training);
        self.bn_out.set_training(training);
    }
}

// =============================================================================
// TaylorChiller Model
// =============================================================================

/// Taylor University chiller plant controller.
///
/// Architecture (Bottleneck blocks with dual regression heads):
/// - Input projection:  2640 → 512 with BN + ReLU
/// - Bottleneck 1:      512  → 128 → 512  (identity skip)
/// - Bottleneck 2:      512  → 128 → 512  (identity skip)
/// - Bottleneck 3:      512  → 64  → 256  (projection skip)
/// - Bottleneck 4:      256  → 64  → 256  (identity skip)
/// - Shared trunk:      256  → 192 with BN + ReLU + Dropout
/// - COP head:          192  → 96  → 8
/// - Staging head:      192  → 96  → 8
///
/// Input: (batch, 2640) — flattened 120 timesteps x 22 features
///   chiller_temps(4) + CW_temps(4) + CHW_temps(4) + flow(2) + amps(2)
///   + OAT(1) + wet_bulb(1) + load(1) + valve_pos(2) + pump_speed(1)
///
/// Outputs: cop_prediction(8), staging_recommendation(8)
pub struct TaylorChiller {
    input_proj: Sequential,
    bottleneck1: BottleneckBlock,
    bottleneck2: BottleneckBlock,
    bottleneck3: BottleneckBlock,
    bottleneck4: BottleneckBlock,
    shared_trunk: Sequential,
    // Dual output heads with their own hidden layers
    cop_hidden: Sequential,
    cop_head: Linear,
    staging_hidden: Sequential,
    staging_head: Linear,
    training: bool,
}

impl Default for TaylorChiller {
    fn default() -> Self {
        Self::new()
    }
}

impl TaylorChiller {
    /// Creates a new TaylorChiller model.
    pub fn new() -> Self {
        // Input projection: 2640 → 512
        let input_proj = Sequential::new()
            .add(Linear::new(2640, 512))
            .add(BatchNorm1d::new(512))
            .add(ReLU);

        // Bottleneck blocks
        let bottleneck1 = BottleneckBlock::new(512, 128, 512);
        let bottleneck2 = BottleneckBlock::new(512, 128, 512);
        let bottleneck3 = BottleneckBlock::new(512, 64, 256);  // downsample
        let bottleneck4 = BottleneckBlock::new(256, 64, 256);

        // Shared trunk: 256 → 192
        let shared_trunk = Sequential::new()
            .add(Linear::new(256, 192))
            .add(BatchNorm1d::new(192))
            .add(ReLU)
            .add(Dropout::new(0.2));

        // COP prediction head: 192 → 96 → 8
        let cop_hidden = Sequential::new()
            .add(Linear::new(192, 96))
            .add(ReLU);
        let cop_head = Linear::new(96, 8);

        // Staging recommendation head: 192 → 96 → 8
        let staging_hidden = Sequential::new()
            .add(Linear::new(192, 96))
            .add(ReLU);
        let staging_head = Linear::new(96, 8);

        Self {
            input_proj,
            bottleneck1,
            bottleneck2,
            bottleneck3,
            bottleneck4,
            shared_trunk,
            cop_hidden,
            cop_head,
            staging_hidden,
            staging_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (cop_prediction, staging_recommendation, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable) {
        // Input projection
        let h = self.input_proj.forward(input);           // (batch, 512)

        // Bottleneck blocks
        let h = self.bottleneck1.forward(&h);             // (batch, 512)
        let h = self.bottleneck2.forward(&h);             // (batch, 512)
        let h = self.bottleneck3.forward(&h);             // (batch, 256)
        let h = self.bottleneck4.forward(&h);             // (batch, 256)

        // Shared trunk
        let embedding = self.shared_trunk.forward(&h);    // (batch, 192)

        // Dual output heads
        let cop_h = self.cop_hidden.forward(&embedding);  // (batch, 96)
        let cop = self.cop_head.forward(&cop_h);           // (batch, 8)

        let stg_h = self.staging_hidden.forward(&embedding); // (batch, 96)
        let staging = self.staging_head.forward(&stg_h);     // (batch, 8)

        (cop, staging, embedding)
    }

    /// Returns the embedding dimension for downstream consumers.
    pub fn embedding_dim() -> usize {
        192
    }

    /// Returns total output dimension (8 + 8 = 16).
    pub fn output_dim() -> usize {
        16
    }
}

impl Module for TaylorChiller {
    fn forward(&self, input: &Variable) -> Variable {
        let (cop, _, _) = self.forward_all(input);
        cop
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.input_proj.parameters());
        params.extend(self.bottleneck1.parameters());
        params.extend(self.bottleneck2.parameters());
        params.extend(self.bottleneck3.parameters());
        params.extend(self.bottleneck4.parameters());
        params.extend(self.shared_trunk.parameters());
        params.extend(self.cop_hidden.parameters());
        params.extend(self.cop_head.parameters());
        params.extend(self.staging_hidden.parameters());
        params.extend(self.staging_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.input_proj.named_parameters() {
            params.insert(format!("input_proj.{n}"), p);
        }
        params.extend(self.bottleneck1.named_parameters("bottleneck1"));
        params.extend(self.bottleneck2.named_parameters("bottleneck2"));
        params.extend(self.bottleneck3.named_parameters("bottleneck3"));
        params.extend(self.bottleneck4.named_parameters("bottleneck4"));
        for (n, p) in self.shared_trunk.named_parameters() {
            params.insert(format!("shared_trunk.{n}"), p);
        }
        for (n, p) in self.cop_hidden.named_parameters() {
            params.insert(format!("cop_hidden.{n}"), p);
        }
        for (n, p) in self.cop_head.named_parameters() {
            params.insert(format!("cop_head.{n}"), p);
        }
        for (n, p) in self.staging_hidden.named_parameters() {
            params.insert(format!("staging_hidden.{n}"), p);
        }
        for (n, p) in self.staging_head.named_parameters() {
            params.insert(format!("staging_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.input_proj.set_training(training);
        self.bottleneck1.set_training(training);
        self.bottleneck2.set_training(training);
        self.bottleneck3.set_training(training);
        self.bottleneck4.set_training(training);
        self.shared_trunk.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "TaylorChiller"
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
    fn test_taylor_chiller_output_shapes() {
        let model = TaylorChiller::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 2640], &[2, 2640]).unwrap(),
            false,
        );
        let (cop, staging, emb) = model.forward_all(&input);

        assert_eq!(cop.shape(), vec![2, 8]);
        assert_eq!(staging.shape(), vec![2, 8]);
        assert_eq!(emb.shape(), vec![2, 192]);
    }

    #[test]
    fn test_taylor_chiller_parameter_count() {
        let model = TaylorChiller::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 400_000 && total < 600_000,
            "TaylorChiller has {} params, expected ~500K",
            total
        );
    }

    #[test]
    fn test_taylor_chiller_forward_module_trait() {
        let model = TaylorChiller::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 2640], &[4, 2640]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 8]);
    }
}
