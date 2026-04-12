//! APOLLO — Master Coordinator Model (~1.8M params)
//!
//! # File
//! `crates/axonml-hvac/src/apollo.rs`
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
use axonml_nn::{
    BatchNorm1d, Dropout, GELU, Linear, Module, MultiHeadAttention, Parameter, ReLU, Sequential,
};

// Model dimensions are defined as constants below

// =============================================================================
// Apollo Model
// =============================================================================

/// Master coordinator — final diagnosis from all model outputs + raw sensors.
///
/// Architecture:
/// - Specialist attention: MultiHeadAttention over 7 model embeddings
/// - Sensor validation: raw sensor features → compressed representation
/// - Decision network: fused features → multi-head diagnosis
///
/// Input: 7 model embeddings + raw sensor summary
/// Outputs: diagnosis(12), cost_optim(6), action_priority(8), confidence(4)
pub struct Apollo {
    // Project all 7 models to common dim for attention
    proj_models: Vec<Linear>,
    // Specialist attention over 7 model outputs
    specialist_attention: MultiHeadAttention,
    // Raw sensor validation
    sensor_encoder: Sequential,
    // Decision network
    decision_net: Sequential,
    // Output heads
    diagnosis_head: Linear,
    cost_head: Linear,
    action_head: Linear,
    confidence_head: Linear,
    training: bool,
}

/// Raw sensor feature dimension (summary of all subsystems).
/// Derived from flattening a small summary per subsystem:
/// electrical(7) + refrigeration(7) + water(7) + mechanical(7) + airflow(7) = 35
/// We use the mean across time for each sensor channel.
pub const RAW_SENSOR_DIM: usize = 35;

/// Model embedding dimensions: aquilo(256) + boreas(384) + naiad(256) + vulcan(256) + zephyrus(320) + colossus(256) + gaia(256)
pub const MODEL_DIMS: [usize; 7] = [256, 384, 256, 256, 320, 256, 256];
/// Sum of all model embedding dimensions.
pub const TOTAL_MODEL_DIM: usize = 1984;

impl Default for Apollo {
    fn default() -> Self {
        Self::new()
    }
}

impl Apollo {
    /// Creates a new Apollo master coordinator.
    pub fn new() -> Self {
        // Project each model output to 256
        let proj_models: Vec<Linear> = MODEL_DIMS
            .iter()
            .map(|&dim| Linear::new(dim, 256))
            .collect();

        // 7 models as sequence tokens, dim=256, 8 heads
        let specialist_attention = MultiHeadAttention::new(256, 8);

        // Raw sensor encoder
        let sensor_encoder = Sequential::new()
            .add(Linear::new(RAW_SENSOR_DIM, 256))
            .add(ReLU)
            .add(Linear::new(256, 128));

        // After attention: mean pool 7 tokens → 256, concat with sensor(128) = 384
        let decision_net = Sequential::new()
            .add(Linear::new(384, 512))
            .add(BatchNorm1d::new(512))
            .add(GELU)
            .add(Dropout::new(0.3))
            .add(Linear::new(512, 512))
            .add(BatchNorm1d::new(512))
            .add(GELU)
            .add(Dropout::new(0.2))
            .add(Linear::new(512, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU);

        let diagnosis_head = Linear::new(256, 12);
        let cost_head = Linear::new(256, 6);
        let action_head = Linear::new(256, 8);
        let confidence_head = Linear::new(256, 4);

        Self {
            proj_models,
            specialist_attention,
            sensor_encoder,
            decision_net,
            diagnosis_head,
            cost_head,
            action_head,
            confidence_head,
            training: true,
        }
    }

    /// Forward pass with individual model embeddings and raw sensor summary.
    ///
    /// # Arguments
    /// * `model_embeddings` - Slice of 7 model output Variables
    /// * `raw_sensors` - Summarized raw sensor features (batch, 35)
    ///
    /// Returns (diagnosis, cost_optim, action_priority, confidence, embedding)
    pub fn forward_parts(
        &self,
        model_embeddings: &[&Variable],
        raw_sensors: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        assert_eq!(
            model_embeddings.len(),
            7,
            "Apollo expects 7 model embeddings"
        );
        let batch = model_embeddings[0].shape()[0];

        // Project all models to 256
        let projected: Vec<Variable> = model_embeddings
            .iter()
            .zip(self.proj_models.iter())
            .map(|(emb, proj)| proj.forward(emb))
            .collect();

        // Stack as (batch, 7, 256)
        let unsqueezed: Vec<Variable> = projected.iter().map(|p| p.unsqueeze(1)).collect();
        let unsqueezed_refs: Vec<&Variable> = unsqueezed.iter().collect();
        let stacked_var = Variable::cat(&unsqueezed_refs, 1); // (batch, 7, 256)

        // Cross-model attention
        let attn_out = self.specialist_attention.forward(&stacked_var); // (batch, 7, 256)

        // Mean pool over 7 models → (batch, 256)
        let model_features = attn_out.mean_dim(1, false);

        // Sensor encoder
        let sensor_features = self.sensor_encoder.forward(raw_sensors); // (batch, 128)

        // Fuse: model(256) + sensor(128) = 384
        let fused = super::aquilo::concat_variables(&[&model_features, &sensor_features], batch);
        let embedding = self.decision_net.forward(&fused); // (batch, 256)

        let diagnosis = self.diagnosis_head.forward(&embedding);
        let cost = self.cost_head.forward(&embedding);
        let action = self.action_head.forward(&embedding);
        let confidence = self.confidence_head.forward(&embedding);

        (diagnosis, cost, action, confidence, embedding)
    }

    /// Forward from concatenated model embeddings + sensor features.
    pub fn forward_concat(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        // Split into model embeddings and raw sensor
        let mut model_parts: Vec<Variable> = Vec::new();
        let mut offset = 0;
        for &dim in &MODEL_DIMS {
            model_parts.push(input.narrow(1, offset, dim));
            offset += dim;
        }

        let raw_sensors = input.narrow(1, offset, RAW_SENSOR_DIM);

        let model_refs: Vec<&Variable> = model_parts.iter().collect();
        self.forward_parts(&model_refs, &raw_sensors)
    }

    /// Embedding dimension.
    pub fn embedding_dim() -> usize {
        256
    }

    /// Total output dimension (12+6+8+4 = 30).
    pub fn output_dim() -> usize {
        30
    }
}

impl Module for Apollo {
    fn forward(&self, input: &Variable) -> Variable {
        let (diagnosis, _, _, _, _) = self.forward_concat(input);
        diagnosis
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for proj in &self.proj_models {
            params.extend(proj.parameters());
        }
        params.extend(self.specialist_attention.parameters());
        params.extend(self.sensor_encoder.parameters());
        params.extend(self.decision_net.parameters());
        params.extend(self.diagnosis_head.parameters());
        params.extend(self.cost_head.parameters());
        params.extend(self.action_head.parameters());
        params.extend(self.confidence_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, proj) in self.proj_models.iter().enumerate() {
            for (n, p) in proj.named_parameters() {
                params.insert(format!("proj_models.{i}.{n}"), p);
            }
        }
        for (n, p) in self.specialist_attention.named_parameters() {
            params.insert(format!("specialist_attention.{n}"), p);
        }
        for (n, p) in self.sensor_encoder.named_parameters() {
            params.insert(format!("sensor_encoder.{n}"), p);
        }
        for (n, p) in self.decision_net.named_parameters() {
            params.insert(format!("decision_net.{n}"), p);
        }
        for (n, p) in self.diagnosis_head.named_parameters() {
            params.insert(format!("diagnosis_head.{n}"), p);
        }
        for (n, p) in self.cost_head.named_parameters() {
            params.insert(format!("cost_head.{n}"), p);
        }
        for (n, p) in self.action_head.named_parameters() {
            params.insert(format!("action_head.{n}"), p);
        }
        for (n, p) in self.confidence_head.named_parameters() {
            params.insert(format!("confidence_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.sensor_encoder.set_training(training);
        self.decision_net.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Apollo"
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
    fn test_apollo_output_shapes() {
        let model = Apollo::new();

        let embs: Vec<Variable> = MODEL_DIMS
            .iter()
            .map(|&dim| {
                Variable::new(
                    Tensor::from_vec(vec![1.0; 2 * dim], &[2, dim]).unwrap(),
                    false,
                )
            })
            .collect();
        let emb_refs: Vec<&Variable> = embs.iter().collect();
        let sensors = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * RAW_SENSOR_DIM], &[2, RAW_SENSOR_DIM]).unwrap(),
            false,
        );

        let (diag, cost, action, conf, emb) = model.forward_parts(&emb_refs, &sensors);

        assert_eq!(diag.shape(), vec![2, 12]);
        assert_eq!(cost.shape(), vec![2, 6]);
        assert_eq!(action.shape(), vec![2, 8]);
        assert_eq!(conf.shape(), vec![2, 4]);
        assert_eq!(emb.shape(), vec![2, 256]);
    }

    #[test]
    fn test_apollo_concat_forward() {
        let model = Apollo::new();
        let total_in = TOTAL_MODEL_DIM + RAW_SENSOR_DIM;
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * total_in], &[2, total_in]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![2, 12]);
    }

    #[test]
    fn test_apollo_parameter_count() {
        let model = Apollo::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 1_200_000 && total < 2_400_000,
            "Apollo has {} params, expected ~1.8M",
            total
        );
    }
}
