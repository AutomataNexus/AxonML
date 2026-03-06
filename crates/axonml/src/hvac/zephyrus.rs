//! ZEPHYRUS — Airflow Systems Diagnostic Model (~845K params)
//!
//! Analyzes airflow sensor data using Graph Neural Networks to model
//! sensor correlations and Conv1d for temporal patterns.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;
use axonml_nn::{
    Conv1d, Dropout, Linear, Module, Parameter,
    Sequential, ReLU, GCNConv,
};

// =============================================================================
// Zephyrus Model
// =============================================================================

/// Airflow systems diagnostic model with Graph Neural Network.
///
/// Architecture:
/// - GCN branch: Learned adjacency matrix + 2 GCNConv layers for sensor correlations
/// - Conv branch: 3 Conv1d layers for temporal pattern extraction
/// - Fusion: concat GCN(256) + Conv(256) → Linear → 320
///
/// Input: (batch, 7, 72) — 7 sensor channels, 72 timesteps
/// Outputs: fault(12), filter_loading(1), duct_leakage(8), iaq(1)
pub struct Zephyrus {
    // Learned adjacency matrix for sensor graph
    adjacency: Parameter,
    // GCN layers
    gcn1: GCNConv,
    gcn2: GCNConv,
    gcn_relu: ReLU,
    // Conv branch
    conv_branch: Sequential,
    // Fusion
    fusion: Sequential,
    // Output heads
    fault_head: Linear,
    filter_head: Linear,
    duct_head: Linear,
    iaq_head: Linear,
    training: bool,
}

impl Zephyrus {
    /// Creates a new Zephyrus model.
    pub fn new() -> Self {
        // Learned adjacency: 7x7 sensor correlation matrix, initialized near-identity
        let mut adj_data = vec![0.1f32; 49];
        for i in 0..7 {
            adj_data[i * 7 + i] = 1.0; // Self-connections strong
        }
        let adjacency = Parameter::named(
            "adjacency",
            Tensor::from_vec(adj_data, &[7, 7]).unwrap(),
            true,
        );

        // GCN layers: (7 nodes, 72 features) → (7, 128) → (7, 256)
        let gcn1 = GCNConv::new(72, 128);
        let gcn2 = GCNConv::new(128, 256);
        let gcn_relu = ReLU;

        // Conv branch: (batch, 7, 72) → temporal features
        let conv_branch = Sequential::new()
            .add(Conv1d::new(7, 64, 3))    // → (batch, 64, 70)
            .add(ReLU)
            .add(Conv1d::new(64, 128, 3))  // → (batch, 128, 68)
            .add(ReLU)
            .add(Conv1d::new(128, 256, 3)) // → (batch, 256, 66)
            .add(ReLU);
            // Global avg pool applied manually in forward

        // Fusion: GCN(256) + Conv(256) = 512 → 320
        let fusion = Sequential::new()
            .add(Linear::new(512, 320))
            .add(ReLU)
            .add(Dropout::new(0.2));

        let fault_head = Linear::new(320, 12);
        let filter_head = Linear::new(320, 1);
        let duct_head = Linear::new(320, 8);
        let iaq_head = Linear::new(320, 1);

        Self {
            adjacency,
            gcn1,
            gcn2,
            gcn_relu,
            conv_branch,
            fusion,
            fault_head,
            filter_head,
            duct_head,
            iaq_head,
            training: true,
        }
    }

    /// Forward pass returning all output heads.
    ///
    /// Returns (fault, filter_loading, duct_leakage, iaq, embedding)
    pub fn forward_all(&self, input: &Variable) -> (Variable, Variable, Variable, Variable, Variable) {
        let shape = input.shape();
        let batch = shape[0];

        // --- GCN Branch ---
        // Input is (batch, 7, 72) — treat as (batch, num_nodes=7, features=72)
        // Apply softmax to adjacency for proper graph attention weights
        let adj_var = self.adjacency.variable();
        let adj_data = adj_var.data().to_vec();
        let mut adj_softmax = vec![0.0f32; 49];
        for i in 0..7 {
            let row_start = i * 7;
            let max_val = adj_data[row_start..row_start + 7].iter()
                .cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for j in 0..7 {
                let e = (adj_data[row_start + j] - max_val).exp();
                adj_softmax[row_start + j] = e;
                sum_exp += e;
            }
            for j in 0..7 {
                adj_softmax[row_start + j] /= sum_exp;
            }
        }
        let adj_norm = Variable::new(
            Tensor::from_vec(adj_softmax, &[7, 7]).unwrap(),
            false,
        );

        let gcn_out = self.gcn1.forward_graph(input, &adj_norm);  // (batch, 7, 128)
        let gcn_out = self.gcn_relu.forward(&gcn_out);
        let gcn_out = self.gcn2.forward_graph(&gcn_out, &adj_norm); // (batch, 7, 256)
        let gcn_out = self.gcn_relu.forward(&gcn_out);

        // Global mean pool over nodes: (batch, 7, 256) → (batch, 256)
        let gcn_features = gcn_out.mean_dim(1, false);

        // --- Conv Branch ---
        let conv_out = self.conv_branch.forward(input); // (batch, 256, 66)
        // Global avg pool over time: (batch, 256, 66) → (batch, 256)
        let conv_features = conv_out.mean_dim(2, false);

        // --- Fusion ---
        let fused = super::aquilo::concat_variables(&[&gcn_features, &conv_features], batch);
        let embedding = self.fusion.forward(&fused); // (batch, 320)

        // --- Heads ---
        let fault = self.fault_head.forward(&embedding);
        let filter = self.filter_head.forward(&embedding);
        let duct = self.duct_head.forward(&embedding);
        let iaq = self.iaq_head.forward(&embedding);

        (fault, filter, duct, iaq, embedding)
    }

    /// Embedding dimension for downstream aggregators.
    pub fn embedding_dim() -> usize {
        320
    }

    /// Total output dimension (12+1+8+1 = 22).
    pub fn output_dim() -> usize {
        22
    }
}

impl Module for Zephyrus {
    fn forward(&self, input: &Variable) -> Variable {
        let (fault, _, _, _, _) = self.forward_all(input);
        fault
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.adjacency.clone()];
        params.extend(self.gcn1.parameters());
        params.extend(self.gcn2.parameters());
        params.extend(self.conv_branch.parameters());
        params.extend(self.fusion.parameters());
        params.extend(self.fault_head.parameters());
        params.extend(self.filter_head.parameters());
        params.extend(self.duct_head.parameters());
        params.extend(self.iaq_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("adjacency".to_string(), self.adjacency.clone());
        for (n, p) in self.gcn1.named_parameters() { params.insert(format!("gcn1.{n}"), p); }
        for (n, p) in self.gcn2.named_parameters() { params.insert(format!("gcn2.{n}"), p); }
        for (n, p) in self.conv_branch.named_parameters() { params.insert(format!("conv_branch.{n}"), p); }
        for (n, p) in self.fusion.named_parameters() { params.insert(format!("fusion.{n}"), p); }
        for (n, p) in self.fault_head.named_parameters() { params.insert(format!("fault_head.{n}"), p); }
        for (n, p) in self.filter_head.named_parameters() { params.insert(format!("filter_head.{n}"), p); }
        for (n, p) in self.duct_head.named_parameters() { params.insert(format!("duct_head.{n}"), p); }
        for (n, p) in self.iaq_head.named_parameters() { params.insert(format!("iaq_head.{n}"), p); }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.conv_branch.set_training(training);
        self.fusion.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Zephyrus"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zephyrus_output_shapes() {
        let model = Zephyrus::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 7 * 72], &[2, 7, 72]).unwrap(),
            false,
        );
        let (fault, filter, duct, iaq, emb) = model.forward_all(&input);

        assert_eq!(fault.shape(), vec![2, 12]);
        assert_eq!(filter.shape(), vec![2, 1]);
        assert_eq!(duct.shape(), vec![2, 8]);
        assert_eq!(iaq.shape(), vec![2, 1]);
        assert_eq!(emb.shape(), vec![2, 320]);
    }

    #[test]
    fn test_zephyrus_parameter_count() {
        let model = Zephyrus::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        // GCN+Conv architecture yields ~338K params
        assert!(total > 250_000 && total < 500_000,
            "Zephyrus has {} params, expected ~338K", total);
    }

    #[test]
    fn test_zephyrus_has_learned_adjacency() {
        let model = Zephyrus::new();
        let named = model.named_parameters();
        assert!(named.contains_key("adjacency"));
        let adj = named.get("adjacency").unwrap();
        assert_eq!(adj.data().shape(), &[7, 7]);
    }

    #[test]
    fn test_zephyrus_forward_trait() {
        let model = Zephyrus::new();
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 7 * 72], &[4, 7, 72]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![4, 12]);
    }
}
