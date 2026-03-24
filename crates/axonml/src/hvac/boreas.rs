//! BOREAS — Refrigeration Systems Diagnostic Model (~1.2M params)
//!
//! # File
//! `crates/axonml/src/hvac/boreas.rs`
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
    BatchNorm1d, Conv1d, Dropout, LSTM, Linear, Module, MultiHeadAttention, Parameter, ReLU,
    ResidualBlock, Sequential,
};
#[cfg(test)]
use axonml_tensor::Tensor;

// =============================================================================
// Boreas Model
// =============================================================================

/// Refrigeration systems diagnostic model.
///
/// Architecture:
/// - 3 thermodynamic analyzers (pressure, temperature, flow)
/// - 3 ResidualBlock with Conv1d for temporal patterns
/// - LSTM for sequence dependencies
/// - Multi-head attention for long-range correlation
///
/// Input: (batch, 80, 7) → transposed to (batch, 7, 80) for Conv1d
/// Outputs: fault(16), efficiency(1), charge(5), component_health(8)
pub struct Boreas {
    // Thermo analyzers (operate on flattened segments)
    pressure_analyzer: Sequential,
    temp_analyzer: Sequential,
    flow_analyzer: Sequential,
    // ResNet blocks (Conv1d based)
    res_block1: ResidualBlock,
    res_block2: ResidualBlock,
    res_block3: ResidualBlock,
    // Sequence modeling
    lstm: LSTM,
    attention: MultiHeadAttention,
    // Pre-head layers
    pre_head: Sequential,
    // Output heads
    fault_head: Linear,
    efficiency_head: Linear,
    charge_head: Linear,
    health_head: Linear,
    training: bool,
}

impl Default for Boreas {
    fn default() -> Self {
        Self::new()
    }
}

impl Boreas {
    /// Creates a new Boreas model.
    pub fn new() -> Self {
        // Thermo analyzers: each takes a subset of features across time
        // Pressure: channels 0,1 (suction/discharge pressure) → 160 vals → 64
        let pressure_analyzer = Sequential::new()
            .add(Linear::new(160, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Linear::new(128, 64));

        // Temperature: channels 2,3 (suction/discharge temp) → 160 vals → 64
        let temp_analyzer = Sequential::new()
            .add(Linear::new(160, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Linear::new(128, 64));

        // Flow: channels 4,5,6 (subcool, superheat, flow) → 240 vals → 64
        let flow_analyzer = Sequential::new()
            .add(Linear::new(240, 128))
            .add(BatchNorm1d::new(128))
            .add(ReLU)
            .add(Linear::new(128, 64));

        // ResNet blocks: operate on (batch, 7, 80) Conv1d format
        // Block 1: 7 → 32 channels (needs downsample)
        let res1_main = Sequential::new()
            .add(Conv1d::new(7, 32, 3))
            .add(BatchNorm1d::new(32))
            .add(ReLU)
            .add(Conv1d::new(32, 32, 3))
            .add(BatchNorm1d::new(32));
        let res1_down = Sequential::new()
            .add(Conv1d::new(7, 32, 5)) // 80→76, matches 80→78→76
            .add(BatchNorm1d::new(32));
        let res_block1 = ResidualBlock::new(res1_main).with_downsample(res1_down);

        // Block 2: 32 → 32 channels (same)
        let res2_main = Sequential::new()
            .add(Conv1d::new(32, 32, 3))
            .add(BatchNorm1d::new(32))
            .add(ReLU)
            .add(Conv1d::new(32, 32, 3))
            .add(BatchNorm1d::new(32));
        let res2_down = Sequential::new()
            .add(Conv1d::new(32, 32, 5))
            .add(BatchNorm1d::new(32));
        let res_block2 = ResidualBlock::new(res2_main).with_downsample(res2_down);

        // Block 3: 32 → 64 channels
        let res3_main = Sequential::new()
            .add(Conv1d::new(32, 64, 3))
            .add(BatchNorm1d::new(64))
            .add(ReLU)
            .add(Conv1d::new(64, 64, 3))
            .add(BatchNorm1d::new(64));
        let res3_down = Sequential::new()
            .add(Conv1d::new(32, 64, 5))
            .add(BatchNorm1d::new(64));
        let res_block3 = ResidualBlock::new(res3_main).with_downsample(res3_down);

        // LSTM: takes reshaped conv output → sequence of 256 features
        // After 3 res blocks: (batch, 64, T') where T' depends on conv reductions
        // We'll flatten conv features and project to LSTM input dim
        let lstm = LSTM::new(64, 256, 1);

        // Multi-head attention on LSTM output
        let attention = MultiHeadAttention::new(256, 8);

        // Pre-head fusion: analyzer(192) + attention(256) = 448 → 384
        let pre_head = Sequential::new()
            .add(Linear::new(448, 384))
            .add(BatchNorm1d::new(384))
            .add(ReLU)
            .add(Dropout::new(0.2));

        let fault_head = Linear::new(384, 16);
        let efficiency_head = Linear::new(384, 1);
        let charge_head = Linear::new(384, 5);
        let health_head = Linear::new(384, 8);

        Self {
            pressure_analyzer,
            temp_analyzer,
            flow_analyzer,
            res_block1,
            res_block2,
            res_block3,
            lstm,
            attention,
            pre_head,
            fault_head,
            efficiency_head,
            charge_head,
            health_head,
            training: true,
        }
    }

    /// Forward pass returning all heads.
    ///
    /// Returns (fault, efficiency, charge, health, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        let shape = input.shape();
        let batch = shape[0];
        // Input: (batch, 80, 7) — extract per-channel features using Variable ops

        // Extract pressure channels (0,1): narrow on last dim, then flatten
        // (batch, 80, 7) → narrow → (batch, 80, 2) → reshape → (batch, 160)
        let pressure_var = input.narrow(2, 0, 2).reshape(&[batch, 160]);
        // Extract temperature channels (2,3): (batch, 80, 2) → (batch, 160)
        let temp_var = input.narrow(2, 2, 2).reshape(&[batch, 160]);
        // Extract flow channels (4,5,6): (batch, 80, 3) → (batch, 240)
        let flow_var = input.narrow(2, 4, 3).reshape(&[batch, 240]);

        let press_out = self.pressure_analyzer.forward(&pressure_var); // (batch, 64)
        let temp_out = self.temp_analyzer.forward(&temp_var); // (batch, 64)
        let flow_out = self.flow_analyzer.forward(&flow_var); // (batch, 64)

        // Transpose input for Conv1d: (batch, 80, 7) → (batch, 7, 80)
        let conv_input = input.transpose(1, 2);

        // ResNet blocks
        let res_out = self.res_block1.forward(&conv_input); // (batch, 32, 76)
        let res_out = self.res_block2.forward(&res_out); // (batch, 32, 72)
        let res_out = self.res_block3.forward(&res_out); // (batch, 64, 68)

        // Transpose for LSTM: (batch, 64, T) → (batch, T, 64)
        let lstm_input = res_out.transpose(1, 2);

        // LSTM
        let lstm_out = self.lstm.forward(&lstm_input); // (batch, time_len, 256)

        // Attention on LSTM output
        let attn_out = self.attention.forward(&lstm_out); // (batch, time_len, 256)

        // Take last timestep as the sequence representation: (batch, T, 256) → (batch, 256)
        let attn_time = attn_out.shape()[1];
        let seq_features = attn_out.select(1, attn_time - 1);

        // Concat analyzers + sequence: (64+64+64) + 256 = 448
        let analyzer_features =
            super::aquilo::concat_variables(&[&press_out, &temp_out, &flow_out], batch);
        let fused = super::aquilo::concat_variables(&[&analyzer_features, &seq_features], batch);

        let embedding = self.pre_head.forward(&fused); // (batch, 384)

        let fault = self.fault_head.forward(&embedding);
        let efficiency = self.efficiency_head.forward(&embedding);
        let charge = self.charge_head.forward(&embedding);
        let health = self.health_head.forward(&embedding);

        (fault, efficiency, charge, health, embedding)
    }

    /// Embedding dimension for downstream aggregators.
    pub fn embedding_dim() -> usize {
        384
    }

    /// Total output dimension (16+1+5+8 = 30).
    pub fn output_dim() -> usize {
        30
    }
}

impl Module for Boreas {
    fn forward(&self, input: &Variable) -> Variable {
        let (fault, _, _, _, _) = self.forward_all(input);
        fault
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.pressure_analyzer.parameters());
        params.extend(self.temp_analyzer.parameters());
        params.extend(self.flow_analyzer.parameters());
        params.extend(self.res_block1.parameters());
        params.extend(self.res_block2.parameters());
        params.extend(self.res_block3.parameters());
        params.extend(self.lstm.parameters());
        params.extend(self.attention.parameters());
        params.extend(self.pre_head.parameters());
        params.extend(self.fault_head.parameters());
        params.extend(self.efficiency_head.parameters());
        params.extend(self.charge_head.parameters());
        params.extend(self.health_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.pressure_analyzer.named_parameters() {
            params.insert(format!("pressure_analyzer.{n}"), p);
        }
        for (n, p) in self.temp_analyzer.named_parameters() {
            params.insert(format!("temp_analyzer.{n}"), p);
        }
        for (n, p) in self.flow_analyzer.named_parameters() {
            params.insert(format!("flow_analyzer.{n}"), p);
        }
        for (n, p) in self.res_block1.named_parameters() {
            params.insert(format!("res_block1.{n}"), p);
        }
        for (n, p) in self.res_block2.named_parameters() {
            params.insert(format!("res_block2.{n}"), p);
        }
        for (n, p) in self.res_block3.named_parameters() {
            params.insert(format!("res_block3.{n}"), p);
        }
        for (n, p) in self.lstm.named_parameters() {
            params.insert(format!("lstm.{n}"), p);
        }
        for (n, p) in self.attention.named_parameters() {
            params.insert(format!("attention.{n}"), p);
        }
        for (n, p) in self.pre_head.named_parameters() {
            params.insert(format!("pre_head.{n}"), p);
        }
        for (n, p) in self.fault_head.named_parameters() {
            params.insert(format!("fault_head.{n}"), p);
        }
        for (n, p) in self.efficiency_head.named_parameters() {
            params.insert(format!("efficiency_head.{n}"), p);
        }
        for (n, p) in self.charge_head.named_parameters() {
            params.insert(format!("charge_head.{n}"), p);
        }
        for (n, p) in self.health_head.named_parameters() {
            params.insert(format!("health_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.pressure_analyzer.set_training(training);
        self.temp_analyzer.set_training(training);
        self.flow_analyzer.set_training(training);
        self.res_block1.set_training(training);
        self.res_block2.set_training(training);
        self.res_block3.set_training(training);
        self.pre_head.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Boreas"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boreas_output_shapes() {
        let model = Boreas::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 80 * 7], &[2, 80, 7]).unwrap(),
            false,
        );
        let (fault, eff, charge, health, emb) = model.forward_all(&input);

        assert_eq!(fault.shape(), vec![2, 16]);
        assert_eq!(eff.shape(), vec![2, 1]);
        assert_eq!(charge.shape(), vec![2, 5]);
        assert_eq!(health.shape(), vec![2, 8]);
        assert_eq!(emb.shape(), vec![2, 384]);
    }

    #[test]
    fn test_boreas_parameter_count() {
        let model = Boreas::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert!(
            total > 800_000 && total < 1_600_000,
            "Boreas has {} params, expected ~1.2M",
            total
        );
    }

    #[test]
    fn test_boreas_forward_trait() {
        let model = Boreas::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 80 * 7], &[4, 80, 7]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 16]);
    }
}
