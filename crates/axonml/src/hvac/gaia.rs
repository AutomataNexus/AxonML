//! GAIA — Safety Validator Model (~896K params)
//!
//! # File
//! `crates/axonml/src/hvac/gaia.rs`
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
use axonml_nn::{BatchNorm1d, Dropout, Linear, Module, Parameter, ReLU, Sequential, GELU};
#[cfg(test)]
use axonml_tensor::Tensor;

use super::colossus;

// =============================================================================
// Gaia Model
// =============================================================================

/// Safety validator model with adversarial robustness.
///
/// Architecture:
/// - Safety encoder: processes all specialist outputs
/// - Validation network: independent validation path
/// - Fusion with safety-critical output heads
///
/// Input: specialist_features(1472) + colossus_output(256) = 1728
/// Outputs: validation_state(5), safety_score(1), override(8), confidence(4)
pub struct Gaia {
    safety_encoder: Sequential,
    validation_net: Sequential,
    fusion: Sequential,
    // Output heads
    validation_head: Linear,
    safety_head: Linear,
    override_head: Linear,
    confidence_head: Linear,
    training: bool,
}

/// Total input dimension: specialist features + colossus embedding (256).
pub const GAIA_INPUT_DIM: usize = colossus::TOTAL_SPECIALIST_DIM + 256;

impl Gaia {
    /// Creates a new Gaia safety validator.
    pub fn new() -> Self {
        let input_dim = GAIA_INPUT_DIM; // 1472 + 256 = 1728

        let safety_encoder = Sequential::new()
            .add(Linear::new(input_dim, 512))
            .add(ReLU)
            .add(Linear::new(512, 256));

        let validation_net = Sequential::new()
            .add(Linear::new(input_dim, 512))
            .add(BatchNorm1d::new(512))
            .add(GELU)
            .add(Dropout::new(0.3))
            .add(Linear::new(512, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(256, 128))
            .add(ReLU);

        // Fusion: safety(256) + validation(128) = 384 → 256
        let fusion = Sequential::new().add(Linear::new(384, 256)).add(ReLU);

        let validation_head = Linear::new(256, 5);
        let safety_head = Linear::new(256, 1);
        let override_head = Linear::new(256, 8);
        let confidence_head = Linear::new(256, 4);

        Self {
            safety_encoder,
            validation_net,
            fusion,
            validation_head,
            safety_head,
            override_head,
            confidence_head,
            training: true,
        }
    }

    /// Forward pass returning all heads.
    ///
    /// # Arguments
    /// * `specialist_features` - Concatenated specialist embeddings (batch, 1472)
    /// * `colossus_embedding` - Colossus aggregator output (batch, 256)
    ///
    /// Returns (validation_state, safety_score, override, confidence, embedding)
    pub fn forward_parts(
        &self,
        specialist_features: &Variable,
        colossus_embedding: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        let batch = specialist_features.shape()[0];

        // Concat specialist + colossus
        let input =
            super::aquilo::concat_variables(&[specialist_features, colossus_embedding], batch);

        self.forward_all(&input)
    }

    /// Forward from pre-concatenated input.
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        let batch = input.shape()[0];

        let safety_out = self.safety_encoder.forward(input); // (batch, 256)
        let validation_out = self.validation_net.forward(input); // (batch, 128)

        let fused = super::aquilo::concat_variables(&[&safety_out, &validation_out], batch);
        let embedding = self.fusion.forward(&fused); // (batch, 256)

        let validation = self.validation_head.forward(&embedding);
        let safety = self.safety_head.forward(&embedding);
        let override_out = self.override_head.forward(&embedding);
        let confidence = self.confidence_head.forward(&embedding);

        (validation, safety, override_out, confidence, embedding)
    }

    /// Embedding dimension.
    pub fn embedding_dim() -> usize {
        256
    }

    /// Total output dimension (5+1+8+4 = 18).
    pub fn output_dim() -> usize {
        18
    }
}

impl Module for Gaia {
    fn forward(&self, input: &Variable) -> Variable {
        let (validation, _, _, _, _) = self.forward_all(input);
        validation
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.safety_encoder.parameters());
        params.extend(self.validation_net.parameters());
        params.extend(self.fusion.parameters());
        params.extend(self.validation_head.parameters());
        params.extend(self.safety_head.parameters());
        params.extend(self.override_head.parameters());
        params.extend(self.confidence_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.safety_encoder.named_parameters() {
            params.insert(format!("safety_encoder.{n}"), p);
        }
        for (n, p) in self.validation_net.named_parameters() {
            params.insert(format!("validation_net.{n}"), p);
        }
        for (n, p) in self.fusion.named_parameters() {
            params.insert(format!("fusion.{n}"), p);
        }
        for (n, p) in self.validation_head.named_parameters() {
            params.insert(format!("validation_head.{n}"), p);
        }
        for (n, p) in self.safety_head.named_parameters() {
            params.insert(format!("safety_head.{n}"), p);
        }
        for (n, p) in self.override_head.named_parameters() {
            params.insert(format!("override_head.{n}"), p);
        }
        for (n, p) in self.confidence_head.named_parameters() {
            params.insert(format!("confidence_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.safety_encoder.set_training(training);
        self.validation_net.set_training(training);
        self.fusion.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Gaia"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaia_output_shapes() {
        let model = Gaia::new();

        let specialist = Variable::new(
            Tensor::from_vec(
                vec![1.0; 2 * colossus::TOTAL_SPECIALIST_DIM],
                &[2, colossus::TOTAL_SPECIALIST_DIM],
            )
            .unwrap(),
            false,
        );
        let colossus_emb = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 256], &[2, 256]).unwrap(),
            false,
        );

        let (val, safety, override_out, conf, emb) =
            model.forward_parts(&specialist, &colossus_emb);

        assert_eq!(val.shape(), vec![2, 5]);
        assert_eq!(safety.shape(), vec![2, 1]);
        assert_eq!(override_out.shape(), vec![2, 8]);
        assert_eq!(conf.shape(), vec![2, 4]);
        assert_eq!(emb.shape(), vec![2, 256]);
    }

    #[test]
    fn test_gaia_concat_forward() {
        let model = Gaia::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * GAIA_INPUT_DIM], &[2, GAIA_INPUT_DIM]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![2, 5]);
    }

    #[test]
    fn test_gaia_parameter_count() {
        let model = Gaia::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        // Dual-path with 1728-dim input yields ~2.2M params
        assert!(
            total > 1_500_000 && total < 3_000_000,
            "Gaia has {} params, expected ~2.2M",
            total
        );
    }
}
