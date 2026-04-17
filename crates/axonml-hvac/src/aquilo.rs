//! AQUILO — Electrical Systems Diagnostic Model (~608K params)
//!
//! # File
//! `crates/axonml-hvac/src/aquilo.rs`
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
use axonml_nn::{BatchNorm1d, Dropout, FFT1d, GELU, Linear, Module, Parameter, ReLU, Sequential};

// =============================================================================
// Aquilo Model
// =============================================================================

/// Electrical systems diagnostic model.
///
/// Architecture:
/// - 3 parallel analyzers (voltage, current, power quality)
/// - FFT spectral feature extraction on voltage/current
/// - Fusion network with multi-head output
///
/// Input: (batch, 168) — flattened 64 timesteps × 7 features, split 3×56
/// Outputs: fault(13), severity(5), phase_health(3), power_quality(1)
pub struct Aquilo {
    voltage_analyzer: Sequential,
    current_analyzer: Sequential,
    power_quality: Sequential,
    fft: FFT1d,
    main_net: Sequential,
    fault_head: Linear,
    severity_head: Linear,
    phase_health_head: Linear,
    pq_head: Linear,
    training: bool,
}

impl Default for Aquilo {
    fn default() -> Self {
        Self::new()
    }
}

impl Aquilo {
    /// Creates a new Aquilo model.
    pub fn new() -> Self {
        let voltage_analyzer = Sequential::new()
            .add(Linear::new(56, 64))
            .add(BatchNorm1d::new(64))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(64, 64));

        let current_analyzer = Sequential::new()
            .add(Linear::new(56, 64))
            .add(BatchNorm1d::new(64))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(64, 64));

        let power_quality = Sequential::new()
            .add(Linear::new(56, 32))
            .add(ReLU)
            .add(Linear::new(32, 32));

        // FFT on 56-length segments → 29 bins each, 2 segments = 58 features
        let fft = FFT1d::new(56);

        // Concat: raw(168) + voltage(64) + current(64) + pq(32) + fft(58) = 386
        let main_net = Sequential::new()
            .add(Linear::new(386, 512))
            .add(BatchNorm1d::new(512))
            .add(GELU)
            .add(Dropout::new(0.3))
            .add(Linear::new(512, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU)
            .add(Dropout::new(0.2));

        let fault_head = Linear::new(256, 13);
        let severity_head = Linear::new(256, 5);
        let phase_health_head = Linear::new(256, 3);
        let pq_head = Linear::new(256, 1);

        Self {
            voltage_analyzer,
            current_analyzer,
            power_quality,
            fft,
            main_net,
            fault_head,
            severity_head,
            phase_health_head,
            pq_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (fault_logits, severity_logits, phase_health, power_quality, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        let shape = input.shape();
        let batch = shape[0];

        // Split input into 3 segments of 56 using narrow (GPU-compatible)
        let volt_var = input.narrow(1, 0, 56); // (batch, 56)
        let curr_var = input.narrow(1, 56, 56); // (batch, 56)
        let pq_var = input.narrow(1, 112, 56); // (batch, 56)

        // Analyzer branches
        let volt_out = self.voltage_analyzer.forward(&volt_var); // (batch, 64)
        let curr_out = self.current_analyzer.forward(&curr_var); // (batch, 64)
        let pq_out = self.power_quality.forward(&pq_var); // (batch, 32)

        // FFT spectral features on voltage and current
        let volt_fft = self.fft.forward(&volt_var); // (batch, 29)
        let curr_fft = self.fft.forward(&curr_var); // (batch, 29)

        // Concatenate everything: raw(168) + volt(64) + curr(64) + pq(32) + fft(58) = 386
        let all_features = concat_variables(
            &[input, &volt_out, &curr_out, &pq_out, &volt_fft, &curr_fft],
            batch,
        );

        // Main network
        let embedding = self.main_net.forward(&all_features); // (batch, 256)

        // Output heads
        let fault = self.fault_head.forward(&embedding);
        let severity = self.severity_head.forward(&embedding);
        let phase_health = self.phase_health_head.forward(&embedding);
        let power_quality = self.pq_head.forward(&embedding);

        (fault, severity, phase_health, power_quality, embedding)
    }

    /// Returns the embedding dimension for downstream aggregators.
    pub fn embedding_dim() -> usize {
        256
    }

    /// Returns total output dimension (13 + 5 + 3 + 1 = 22).
    pub fn output_dim() -> usize {
        22
    }
}

impl Module for Aquilo {
    fn forward(&self, input: &Variable) -> Variable {
        let (fault, _, _, _, _) = self.forward_all(input);
        fault
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.voltage_analyzer.parameters());
        params.extend(self.current_analyzer.parameters());
        params.extend(self.power_quality.parameters());
        params.extend(self.main_net.parameters());
        params.extend(self.fault_head.parameters());
        params.extend(self.severity_head.parameters());
        params.extend(self.phase_health_head.parameters());
        params.extend(self.pq_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.voltage_analyzer.named_parameters() {
            params.insert(format!("voltage_analyzer.{n}"), p);
        }
        for (n, p) in self.current_analyzer.named_parameters() {
            params.insert(format!("current_analyzer.{n}"), p);
        }
        for (n, p) in self.power_quality.named_parameters() {
            params.insert(format!("power_quality.{n}"), p);
        }
        for (n, p) in self.main_net.named_parameters() {
            params.insert(format!("main_net.{n}"), p);
        }
        for (n, p) in self.fault_head.named_parameters() {
            params.insert(format!("fault_head.{n}"), p);
        }
        for (n, p) in self.severity_head.named_parameters() {
            params.insert(format!("severity_head.{n}"), p);
        }
        for (n, p) in self.phase_health_head.named_parameters() {
            params.insert(format!("phase_health_head.{n}"), p);
        }
        for (n, p) in self.pq_head.named_parameters() {
            params.insert(format!("pq_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.voltage_analyzer.set_training(training);
        self.current_analyzer.set_training(training);
        self.power_quality.set_training(training);
        self.main_net.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Aquilo"
    }
}

// =============================================================================
// Utility
// =============================================================================

/// Concatenate multiple Variables along the last dimension.
pub(crate) fn concat_variables(vars: &[&Variable], _batch: usize) -> Variable {
    Variable::cat(vars, vars[0].shape().len() - 1)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_aquilo_output_shapes() {
        let model = Aquilo::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 168], &[2, 168]).unwrap(),
            false,
        );
        let (fault, severity, phase, pq, emb) = model.forward_all(&input);

        assert_eq!(fault.shape(), vec![2, 13]);
        assert_eq!(severity.shape(), vec![2, 5]);
        assert_eq!(phase.shape(), vec![2, 3]);
        assert_eq!(pq.shape(), vec![2, 1]);
        assert_eq!(emb.shape(), vec![2, 256]);
    }

    #[test]
    fn test_aquilo_parameter_count() {
        let model = Aquilo::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        // Architecture yields ~355K params
        assert!(
            total > 250_000 && total < 500_000,
            "Aquilo has {} params, expected ~355K",
            total
        );
    }

    #[test]
    fn test_aquilo_forward_module_trait() {
        let model = Aquilo::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 168], &[4, 168]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 13]);
    }
}
