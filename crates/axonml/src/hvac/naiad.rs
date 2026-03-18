//! NAIAD — Water Systems Diagnostic Model (~533K params)
//!
//! # File
//! `crates/axonml/src/hvac/naiad.rs`
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
use axonml_nn::{Conv1d, Linear, MaxPool1d, Module, MultiHeadAttention, Parameter, ReLU};

// =============================================================================
// Naiad Model
// =============================================================================

/// Water systems diagnostic model.
///
/// Architecture:
/// - 3-layer Conv1d backbone with MaxPool and GlobalAvgPool
/// - Multi-head self-attention on the 256-dim embedding
/// - Multi-head output for fault, flow anomaly, water quality, pump efficiency
///
/// Input: (batch, 7, 64) — Conv1d format (7 channels, 64 timesteps)
/// Outputs: fault(11), flow_anomaly(2), water_quality(1), pump_efficiency(4)
pub struct Naiad {
    conv1: Conv1d,
    relu1: ReLU,
    pool1: MaxPool1d,
    conv2: Conv1d,
    relu2: ReLU,
    pool2: MaxPool1d,
    conv3: Conv1d,
    relu3: ReLU,
    attention: MultiHeadAttention,
    fault_head: Linear,
    flow_anomaly_head: Linear,
    water_quality_head: Linear,
    pump_efficiency_head: Linear,
    training: bool,
}

impl Naiad {
    /// Creates a new Naiad model.
    pub fn new() -> Self {
        // Conv backbone
        let conv1 = Conv1d::new(7, 64, 3); // (batch, 7, 64) -> (batch, 64, 62)
        let relu1 = ReLU;
        let pool1 = MaxPool1d::new(2); // -> (batch, 64, 31)

        let conv2 = Conv1d::new(64, 128, 3); // -> (batch, 128, 29)
        let relu2 = ReLU;
        let pool2 = MaxPool1d::new(2); // -> (batch, 128, 14)

        let conv3 = Conv1d::new(128, 256, 3); // -> (batch, 256, 12)
        let relu3 = ReLU;
        // GlobalAvgPool over time dim -> (batch, 256)

        // Self-attention on 256-dim embedding
        let attention = MultiHeadAttention::new(256, 4);

        // Output heads
        let fault_head = Linear::new(256, 11);
        let flow_anomaly_head = Linear::new(256, 2);
        let water_quality_head = Linear::new(256, 1);
        let pump_efficiency_head = Linear::new(256, 4);

        Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            conv3,
            relu3,
            attention,
            fault_head,
            flow_anomaly_head,
            water_quality_head,
            pump_efficiency_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (fault_logits, flow_anomaly_logits, water_quality, pump_efficiency, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        let shape = input.shape();
        let batch = shape[0];

        // Conv1d backbone: input (batch, 7, 64)
        let x = self.conv1.forward(input); // (batch, 64, 62)
        let x = self.relu1.forward(&x);
        let x = self.pool1.forward(&x); // (batch, 64, 31)

        let x = self.conv2.forward(&x); // (batch, 128, 29)
        let x = self.relu2.forward(&x);
        let x = self.pool2.forward(&x); // (batch, 128, 14)

        let x = self.conv3.forward(&x); // (batch, 256, 12)
        let x = self.relu3.forward(&x);

        // Global average pooling over the time dimension
        // x shape: (batch, 256, time_len)
        let channels = x.shape()[1];
        let embedding = x.mean_dim(2, false); // (batch, 256)

        // Self-attention: reshape (batch, 256) -> (batch, 1, 256) for MHA
        let attn_input = embedding.reshape(&[batch, 1, channels]);
        let attn_out = self.attention.forward(&attn_input); // (batch, 1, 256)

        // Squeeze back to (batch, 256)
        let embedding = attn_out.reshape(&[batch, channels]);

        // Output heads
        let fault = self.fault_head.forward(&embedding);
        let flow_anomaly = self.flow_anomaly_head.forward(&embedding);
        let water_quality = self.water_quality_head.forward(&embedding);
        let pump_efficiency = self.pump_efficiency_head.forward(&embedding);

        (
            fault,
            flow_anomaly,
            water_quality,
            pump_efficiency,
            embedding,
        )
    }

    /// Returns the embedding dimension for downstream aggregators.
    pub fn embedding_dim() -> usize {
        256
    }

    /// Returns total output dimension (11 + 2 + 1 + 4 = 18).
    pub fn output_dim() -> usize {
        18
    }
}

impl Module for Naiad {
    fn forward(&self, input: &Variable) -> Variable {
        let (fault, _, _, _, _) = self.forward_all(input);
        fault
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.attention.parameters());
        params.extend(self.fault_head.parameters());
        params.extend(self.flow_anomaly_head.parameters());
        params.extend(self.water_quality_head.parameters());
        params.extend(self.pump_efficiency_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.conv1.named_parameters() {
            params.insert(format!("conv1.{n}"), p);
        }
        for (n, p) in self.conv2.named_parameters() {
            params.insert(format!("conv2.{n}"), p);
        }
        for (n, p) in self.conv3.named_parameters() {
            params.insert(format!("conv3.{n}"), p);
        }
        for (n, p) in self.attention.named_parameters() {
            params.insert(format!("attention.{n}"), p);
        }
        for (n, p) in self.fault_head.named_parameters() {
            params.insert(format!("fault_head.{n}"), p);
        }
        for (n, p) in self.flow_anomaly_head.named_parameters() {
            params.insert(format!("flow_anomaly_head.{n}"), p);
        }
        for (n, p) in self.water_quality_head.named_parameters() {
            params.insert(format!("water_quality_head.{n}"), p);
        }
        for (n, p) in self.pump_efficiency_head.named_parameters() {
            params.insert(format!("pump_efficiency_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Naiad"
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
    fn test_naiad_output_shapes() {
        let model = Naiad::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 7 * 64], &[2, 7, 64]).unwrap(),
            false,
        );
        let (fault, flow, wq, pump, emb) = model.forward_all(&input);

        assert_eq!(fault.shape(), vec![2, 11]);
        assert_eq!(flow.shape(), vec![2, 2]);
        assert_eq!(wq.shape(), vec![2, 1]);
        assert_eq!(pump.shape(), vec![2, 4]);
        assert_eq!(emb.shape(), vec![2, 256]);
    }

    #[test]
    fn test_naiad_parameter_count() {
        let model = Naiad::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        // Expected ~533K params
        assert!(
            total > 350_000 && total < 700_000,
            "Naiad has {} params, expected ~533K",
            total
        );
    }

    #[test]
    fn test_naiad_forward_module_trait() {
        let model = Naiad::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 4 * 7 * 64], &[4, 7, 64]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 11]);
    }

    #[test]
    fn test_naiad_embedding_dim() {
        assert_eq!(Naiad::embedding_dim(), 256);
    }

    #[test]
    fn test_naiad_output_dim() {
        assert_eq!(Naiad::output_dim(), 18);
    }
}
