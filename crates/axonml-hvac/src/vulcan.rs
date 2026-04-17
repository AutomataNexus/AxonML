//! VULCAN — Mechanical Systems Diagnostic Model (~1.1M params)
//!
//! # File
//! `crates/axonml-hvac/src/vulcan.rs`
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
use axonml_nn::{BatchNorm1d, Dropout, FFT1d, Linear, Module, Parameter, ReLU, Sequential};

use super::aquilo::concat_variables;

// =============================================================================
// Vulcan Model
// =============================================================================

/// Mechanical systems diagnostic model.
///
/// Architecture (Wide & Deep with FFT vibration analysis):
/// - Wide branch:   Linear(672, 256)
/// - Deep branch:   5-layer MLP with BatchNorm and Dropout
/// - Vibration:     FFT1d(672) → 337 freq bins → 2-layer MLP
/// - Fusion:        Concat(256+128+64=448) → Linear(448, 256) → ReLU
///
/// Input: (batch, 672) — flattened 96 timesteps × 7 features
/// Outputs: mech_fault(15), bearing_health(4), vib_severity(4), rul(1)
pub struct Vulcan {
    wide_branch: Sequential,
    deep_branch: Sequential,
    fft: FFT1d,
    vib_linear1: Linear,
    vib_linear2: Linear,
    fusion: Sequential,
    mech_fault_head: Linear,
    bearing_health_head: Linear,
    vib_severity_head: Linear,
    rul_head: Linear,
    training: bool,
}

impl Default for Vulcan {
    fn default() -> Self {
        Self::new()
    }
}

impl Vulcan {
    /// Creates a new Vulcan model.
    pub fn new() -> Self {
        // Wide branch: Linear(672, 256)
        let wide_branch = Sequential::new().add(Linear::new(672, 256));

        // Deep branch: 5 layers with BN/ReLU/Dropout
        let deep_branch = Sequential::new()
            .add(Linear::new(672, 512))
            .add(BatchNorm1d::new(512))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(512, 512))
            .add(BatchNorm1d::new(512))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(512, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU)
            .add(Dropout::new(0.2))
            .add(Linear::new(256, 256))
            .add(BatchNorm1d::new(256))
            .add(ReLU)
            .add(Linear::new(256, 128))
            .add(ReLU);

        // FFT on full 672-length input → 337 frequency bins
        let fft = FFT1d::new(672);

        // Vibration branch linear layers (applied after FFT)
        let vib_linear1 = Linear::new(337, 256);
        let vib_linear2 = Linear::new(256, 64);

        // Fusion: concat wide(256) + deep(128) + vibration(64) = 448
        let fusion = Sequential::new().add(Linear::new(448, 256)).add(ReLU);

        // Output heads
        let mech_fault_head = Linear::new(256, 15);
        let bearing_health_head = Linear::new(256, 4);
        let vib_severity_head = Linear::new(256, 4);
        let rul_head = Linear::new(256, 1);

        Self {
            wide_branch,
            deep_branch,
            fft,
            vib_linear1,
            vib_linear2,
            fusion,
            mech_fault_head,
            bearing_health_head,
            vib_severity_head,
            rul_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (mech_fault, bearing_health, vib_severity, rul, embedding)
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable, Variable) {
        let shape = input.shape();
        let batch = shape[0];

        // Wide branch: (batch, 672) → (batch, 256)
        let wide_out = self.wide_branch.forward(input);

        // Deep branch: (batch, 672) → (batch, 128)
        let deep_out = self.deep_branch.forward(input);

        // Vibration branch: FFT → Linear → ReLU → Linear
        let fft_out = self.fft.forward(input); // (batch, 337)
        let vib_hidden = self.vib_linear1.forward(&fft_out); // (batch, 256)
        let vib_hidden = vib_hidden.relu(); // ReLU
        let vib_out = self.vib_linear2.forward(&vib_hidden); // (batch, 64)

        // Fusion: concat wide(256) + deep(128) + vibration(64) = 448
        let fused = concat_variables(&[&wide_out, &deep_out, &vib_out], batch);
        let embedding = self.fusion.forward(&fused); // (batch, 256)

        // Output heads
        let mech_fault = self.mech_fault_head.forward(&embedding);
        let bearing_health = self.bearing_health_head.forward(&embedding);
        let vib_severity = self.vib_severity_head.forward(&embedding);
        let rul = self.rul_head.forward(&embedding);

        (mech_fault, bearing_health, vib_severity, rul, embedding)
    }

    /// Returns the embedding dimension for downstream aggregators.
    pub fn embedding_dim() -> usize {
        256
    }

    /// Returns total output dimension (15 + 4 + 4 + 1 = 24).
    pub fn output_dim() -> usize {
        24
    }
}

impl Module for Vulcan {
    fn forward(&self, input: &Variable) -> Variable {
        let (mech_fault, _, _, _, _) = self.forward_all(input);
        mech_fault
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.wide_branch.parameters());
        params.extend(self.deep_branch.parameters());
        params.extend(self.vib_linear1.parameters());
        params.extend(self.vib_linear2.parameters());
        params.extend(self.fusion.parameters());
        params.extend(self.mech_fault_head.parameters());
        params.extend(self.bearing_health_head.parameters());
        params.extend(self.vib_severity_head.parameters());
        params.extend(self.rul_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.wide_branch.named_parameters() {
            params.insert(format!("wide_branch.{n}"), p);
        }
        for (n, p) in self.deep_branch.named_parameters() {
            params.insert(format!("deep_branch.{n}"), p);
        }
        for (n, p) in self.vib_linear1.named_parameters() {
            params.insert(format!("vib_linear1.{n}"), p);
        }
        for (n, p) in self.vib_linear2.named_parameters() {
            params.insert(format!("vib_linear2.{n}"), p);
        }
        for (n, p) in self.fusion.named_parameters() {
            params.insert(format!("fusion.{n}"), p);
        }
        for (n, p) in self.mech_fault_head.named_parameters() {
            params.insert(format!("mech_fault_head.{n}"), p);
        }
        for (n, p) in self.bearing_health_head.named_parameters() {
            params.insert(format!("bearing_health_head.{n}"), p);
        }
        for (n, p) in self.vib_severity_head.named_parameters() {
            params.insert(format!("vib_severity_head.{n}"), p);
        }
        for (n, p) in self.rul_head.named_parameters() {
            params.insert(format!("rul_head.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.wide_branch.set_training(training);
        self.deep_branch.set_training(training);
        self.fusion.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Vulcan"
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
    fn test_vulcan_output_shapes() {
        let model = Vulcan::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 672], &[2, 672]).unwrap(),
            false,
        );
        let (mech_fault, bearing_health, vib_severity, rul, emb) = model.forward_all(&input);

        assert_eq!(mech_fault.shape(), vec![2, 15]);
        assert_eq!(bearing_health.shape(), vec![2, 4]);
        assert_eq!(vib_severity.shape(), vec![2, 4]);
        assert_eq!(rul.shape(), vec![2, 1]);
        assert_eq!(emb.shape(), vec![2, 256]);
    }

    #[test]
    fn test_vulcan_parameter_count() {
        let model = Vulcan::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        // Expected ~1.1M params
        assert!(
            total > 900_000 && total < 1_300_000,
            "Vulcan has {} params, expected ~1.1M",
            total
        );
    }

    #[test]
    fn test_vulcan_forward_module_trait() {
        let model = Vulcan::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 672], &[4, 672]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 15]);
    }
}
