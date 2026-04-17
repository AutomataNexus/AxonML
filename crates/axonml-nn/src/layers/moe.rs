//! Mixture of Experts - Sparse Expert Routing
//!
//! Implements a Mixture of Experts (MoE) layer where each token is routed to
//! a subset of expert MLPs. This allows massive model capacity (many experts)
//! while keeping per-token compute constant (only top-k experts activate).
//!
//! Key components:
//! - `Expert` — Standard SiLU-gated MLP (up_proj + gate_proj + down_proj)
//! - `MoERouter` — Learned routing: Linear -> softmax -> top-k selection
//! - `MoELayer` — Full MoE: routes tokens to top-k experts, combines outputs
//! - `load_balancing_loss()` — Auxiliary loss preventing expert collapse
//!
//! # File
//! `crates/axonml-nn/src/layers/moe.rs`
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
use axonml_tensor::Tensor;

use crate::layers::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// Expert MLP
// =============================================================================

/// A single expert MLP with SiLU-gated architecture (SwiGLU variant).
///
/// Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x))
///
/// This is the same MLP architecture used in LLaMA/Mistral/Phi models.
///
/// # Shape
/// - Input: (*, d_model)
/// - Output: (*, d_model)
pub struct Expert {
    /// Up projection: d_model -> intermediate_size
    up_proj: Linear,
    /// Gate projection: d_model -> intermediate_size (for SiLU gating)
    gate_proj: Linear,
    /// Down projection: intermediate_size -> d_model
    down_proj: Linear,
}

impl Expert {
    /// Creates a new Expert MLP.
    ///
    /// # Arguments
    /// * `d_model` - Input/output dimension
    /// * `intermediate_size` - Hidden dimension (typically 4 * d_model or 8/3 * d_model)
    pub fn new(d_model: usize, intermediate_size: usize) -> Self {
        Self {
            up_proj: Linear::with_bias(d_model, intermediate_size, false),
            gate_proj: Linear::with_bias(d_model, intermediate_size, false),
            down_proj: Linear::with_bias(intermediate_size, d_model, false),
        }
    }
}

impl Module for Expert {
    fn forward(&self, input: &Variable) -> Variable {
        // SwiGLU: down(SiLU(gate(x)) * up(x))
        let gate = self.gate_proj.forward(input).silu();
        let up = self.up_proj.forward(input);
        let hidden = gate.mul_var(&up);
        self.down_proj.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.up_proj.parameters());
        params.extend(self.gate_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.up_proj.named_parameters() {
            params.insert(format!("up_proj.{name}"), param);
        }
        for (name, param) in self.gate_proj.named_parameters() {
            params.insert(format!("gate_proj.{name}"), param);
        }
        for (name, param) in self.down_proj.named_parameters() {
            params.insert(format!("down_proj.{name}"), param);
        }
        params
    }

    fn name(&self) -> &'static str {
        "Expert"
    }
}

impl std::fmt::Debug for Expert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expert")
            .field("up_proj", &self.up_proj)
            .field("gate_proj", &self.gate_proj)
            .field("down_proj", &self.down_proj)
            .finish()
    }
}

// =============================================================================
// MoE Router
// =============================================================================

/// Router for Mixture of Experts.
///
/// Maps each token's hidden state to expert selection probabilities via a
/// learned linear projection followed by softmax, then selects the top-k
/// experts per token.
///
/// # Routing
/// ```text
/// gate_logits = Linear(x)           [num_tokens, num_experts]
/// gate_probs  = softmax(gate_logits) [num_tokens, num_experts]
/// top_k_probs, top_k_indices = topk(gate_probs, k)
/// ```
pub struct MoERouter {
    /// Gate projection: d_model -> num_experts
    gate: Linear,
    /// Number of experts
    num_experts: usize,
    /// Number of experts to select per token
    top_k: usize,
}

impl MoERouter {
    /// Creates a new MoE router.
    ///
    /// # Arguments
    /// * `d_model` - Input hidden dimension
    /// * `num_experts` - Total number of experts
    /// * `top_k` - Number of experts activated per token
    pub fn new(d_model: usize, num_experts: usize, top_k: usize) -> Self {
        assert!(
            top_k <= num_experts,
            "top_k ({top_k}) must be <= num_experts ({num_experts})"
        );
        Self {
            gate: Linear::with_bias(d_model, num_experts, false),
            num_experts,
            top_k,
        }
    }

    /// Routes tokens to experts.
    ///
    /// # Arguments
    /// * `x` - Hidden states [num_tokens, d_model]
    ///
    /// # Returns
    /// * `gate_probs` - Full probability distribution [num_tokens, num_experts] (for load balancing)
    /// * `top_k_weights` - Normalized weights for selected experts [num_tokens, top_k]
    /// * `top_k_indices` - Indices of selected experts [num_tokens, top_k]
    pub fn route(&self, x: &Variable) -> (Variable, Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let gate_logits = self.gate.forward(x);
        let gate_probs = gate_logits.softmax(-1);

        let probs_data = gate_probs.data();
        let probs_vec = probs_data.to_vec();
        let num_tokens = probs_data.shape()[0];

        let mut top_k_weights = Vec::with_capacity(num_tokens);
        let mut top_k_indices = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let offset = t * self.num_experts;
            let token_probs = &probs_vec[offset..offset + self.num_experts];

            // Find top-k experts by probability
            let mut indexed: Vec<(usize, f32)> = token_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_indices: Vec<usize> = indexed[..self.top_k].iter().map(|(i, _)| *i).collect();
            let top_weights: Vec<f32> = indexed[..self.top_k].iter().map(|(_, w)| *w).collect();

            // Normalize top-k weights to sum to 1
            let weight_sum: f32 = top_weights.iter().sum();
            let normalized: Vec<f32> = if weight_sum > 0.0 {
                top_weights.iter().map(|w| w / weight_sum).collect()
            } else {
                vec![1.0 / self.top_k as f32; self.top_k]
            };

            top_k_weights.push(normalized);
            top_k_indices.push(top_indices);
        }

        (gate_probs, top_k_weights, top_k_indices)
    }

    /// Returns the number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Returns top-k value.
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

impl std::fmt::Debug for MoERouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoERouter")
            .field("num_experts", &self.num_experts)
            .field("top_k", &self.top_k)
            .finish()
    }
}

// =============================================================================
// MoE Layer
// =============================================================================

/// Mixture of Experts layer.
///
/// Routes each token to the top-k experts (out of N total), applies the
/// selected expert MLPs, and combines outputs using the routing weights.
///
/// This provides N times the model capacity while only using top-k/N of
/// the compute per token.
///
/// # Shape
/// - Input: (batch, seq_len, d_model)
/// - Output: (batch, seq_len, d_model)
///
/// # Load Balancing
/// Call `load_balancing_loss()` after forward to get the auxiliary loss
/// that prevents expert collapse (all tokens routing to same expert).
pub struct MoELayer {
    /// Expert MLPs
    experts: Vec<Expert>,
    /// Router for expert selection
    router: MoERouter,
    /// Model dimension
    d_model: usize,
    /// Number of experts
    num_experts: usize,
    /// Top-k routing
    top_k: usize,
    /// Cached gate probabilities from last forward pass (for load balancing loss)
    last_gate_probs: std::sync::RwLock<Option<Variable>>,
    /// Cached expert assignments from last forward (for utilization stats)
    last_expert_counts: std::sync::RwLock<Vec<usize>>,
}

impl MoELayer {
    /// Creates a new MoE layer.
    ///
    /// # Arguments
    /// * `d_model` - Hidden dimension
    /// * `intermediate_size` - Expert MLP hidden dimension
    /// * `num_experts` - Total number of expert MLPs
    /// * `top_k` - Number of experts activated per token
    pub fn new(d_model: usize, intermediate_size: usize, num_experts: usize, top_k: usize) -> Self {
        let experts: Vec<Expert> = (0..num_experts)
            .map(|_| Expert::new(d_model, intermediate_size))
            .collect();
        let router = MoERouter::new(d_model, num_experts, top_k);

        Self {
            experts,
            router,
            d_model,
            num_experts,
            top_k,
            last_gate_probs: std::sync::RwLock::new(None),
            last_expert_counts: std::sync::RwLock::new(vec![0; num_experts]),
        }
    }

    /// Computes load balancing auxiliary loss.
    ///
    /// This loss encourages uniform routing across experts to prevent
    /// expert collapse. Uses the formulation from Switch Transformer:
    ///
    ///   L_bal = num_experts * sum_i(f_i * P_i)
    ///
    /// where f_i = fraction of tokens assigned to expert i,
    ///       P_i = mean routing probability for expert i.
    ///
    /// Returns zero if no forward pass has been done yet.
    pub fn load_balancing_loss(&self) -> Variable {
        let gate_probs_opt = self.last_gate_probs.read().unwrap();
        if gate_probs_opt.is_none() {
            return Variable::new(
                Tensor::from_vec(vec![0.0f32], &[1]).expect("tensor creation failed"),
                false,
            );
        }

        let gate_probs = gate_probs_opt.as_ref().unwrap();
        let probs_data = gate_probs.data();
        let probs_vec = probs_data.to_vec();
        let shape = probs_data.shape();
        let num_tokens = shape[0];
        let num_experts = shape[1];

        if num_tokens == 0 {
            return Variable::new(
                Tensor::from_vec(vec![0.0f32], &[1]).expect("tensor creation failed"),
                false,
            );
        }

        let expert_counts = self.last_expert_counts.read().unwrap();

        // f_i: fraction of tokens routed to expert i
        let token_fractions: Vec<f32> = expert_counts
            .iter()
            .map(|&c| c as f32 / num_tokens as f32)
            .collect();

        // P_i: mean routing probability for expert i
        let mut mean_probs = vec![0.0f32; num_experts];
        for t in 0..num_tokens {
            for e in 0..num_experts {
                mean_probs[e] += probs_vec[t * num_experts + e];
            }
        }
        for p in &mut mean_probs {
            *p /= num_tokens as f32;
        }

        // L_bal = num_experts * sum(f_i * P_i)
        let mut loss_val = 0.0f32;
        for e in 0..num_experts {
            loss_val += token_fractions[e] * mean_probs[e];
        }
        loss_val *= num_experts as f32;

        Variable::new(
            Tensor::from_vec(vec![loss_val], &[1]).expect("tensor creation failed"),
            false,
        )
    }

    /// Returns expert utilization counts from the last forward pass.
    ///
    /// Each element is the number of tokens routed to that expert.
    pub fn expert_utilization(&self) -> Vec<usize> {
        self.last_expert_counts.read().unwrap().clone()
    }

    /// Returns the number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Returns the top-k value.
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

impl Module for MoELayer {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];
        let num_tokens = batch_size * seq_len;

        // Flatten to [num_tokens, d_model]
        let flat_input = input.reshape(&[num_tokens, d_model]);

        // Route tokens to experts
        let (gate_probs, top_k_weights, top_k_indices) = self.router.route(&flat_input);

        // Track expert utilization
        let mut expert_counts = vec![0usize; self.num_experts];
        for indices in &top_k_indices {
            for &idx in indices {
                expert_counts[idx] += 1;
            }
        }
        *self.last_expert_counts.write().unwrap() = expert_counts;
        *self.last_gate_probs.write().unwrap() = Some(gate_probs);

        // Initialize output as zeros
        let mut output_data = vec![0.0f32; num_tokens * d_model];

        // Process each expert: gather tokens, forward, scatter back
        for expert_idx in 0..self.num_experts {
            // Find which tokens go to this expert and their weights
            let mut token_indices = Vec::new();
            let mut token_weights = Vec::new();

            for (t, (indices, weights)) in
                top_k_indices.iter().zip(top_k_weights.iter()).enumerate()
            {
                for (k, (&idx, &w)) in indices.iter().zip(weights.iter()).enumerate() {
                    if idx == expert_idx {
                        token_indices.push(t);
                        token_weights.push(w);
                        let _ = k;
                    }
                }
            }

            if token_indices.is_empty() {
                continue;
            }

            // Gather tokens for this expert
            let flat_data = flat_input.data();
            let flat_vec = flat_data.to_vec();
            let n = token_indices.len();
            let mut expert_input_data = Vec::with_capacity(n * d_model);
            for &t in &token_indices {
                let offset = t * d_model;
                expert_input_data.extend_from_slice(&flat_vec[offset..offset + d_model]);
            }
            let expert_input = Variable::new(
                Tensor::from_vec(expert_input_data, &[n, d_model]).expect("tensor creation failed"),
                true,
            );

            // Forward through expert
            let expert_output = self.experts[expert_idx].forward(&expert_input);
            let expert_out_vec = expert_output.data().to_vec();

            // Scatter weighted outputs back
            for (local_idx, &global_idx) in token_indices.iter().enumerate() {
                let weight = token_weights[local_idx];
                let src_offset = local_idx * d_model;
                let dst_offset = global_idx * d_model;
                for d in 0..d_model {
                    output_data[dst_offset + d] += weight * expert_out_vec[src_offset + d];
                }
            }
        }

        let output_tensor =
            Tensor::from_vec(output_data, &[num_tokens, d_model]).expect("tensor creation failed");
        let output = Variable::new(output_tensor, true);

        // Reshape back to [batch, seq_len, d_model]
        output.reshape(&[batch_size, seq_len, d_model])
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.router.gate.parameters());
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.router.gate.named_parameters() {
            params.insert(format!("router.gate.{name}"), param);
        }
        for (i, expert) in self.experts.iter().enumerate() {
            for (name, param) in expert.named_parameters() {
                params.insert(format!("experts.{i}.{name}"), param);
            }
        }
        params
    }

    fn name(&self) -> &'static str {
        "MoELayer"
    }
}

impl std::fmt::Debug for MoELayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoELayer")
            .field("d_model", &self.d_model)
            .field("num_experts", &self.num_experts)
            .field("top_k", &self.top_k)
            .field("experts", &self.experts.len())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(64, 256);
        let params = expert.parameters();
        // up_proj(w) + gate_proj(w) + down_proj(w) = 3 weights (no bias)
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(64, 256);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 4 * 64], &[4, 64]).expect("tensor creation failed"),
            false,
        );
        let output = expert.forward(&input);
        assert_eq!(output.shape(), vec![4, 64]);
    }

    #[test]
    fn test_router_creation() {
        let router = MoERouter::new(64, 8, 2);
        assert_eq!(router.num_experts(), 8);
        assert_eq!(router.top_k(), 2);
    }

    #[test]
    fn test_router_route() {
        let router = MoERouter::new(64, 8, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 4 * 64], &[4, 64]).expect("tensor creation failed"),
            false,
        );
        let (_gate_probs, weights, indices) = router.route(&input);

        assert_eq!(weights.len(), 4); // 4 tokens
        assert_eq!(indices.len(), 4);
        for w in &weights {
            assert_eq!(w.len(), 2); // top-2
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Weights should sum to 1");
        }
        for idx in &indices {
            assert_eq!(idx.len(), 2);
            for &i in idx {
                assert!(i < 8, "Expert index should be < num_experts");
            }
        }
    }

    #[test]
    fn test_moe_layer_forward() {
        let moe = MoELayer::new(64, 256, 8, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 5 * 64], &[2, 5, 64]).expect("tensor creation failed"),
            false,
        );
        let output = moe.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_moe_layer_parameters() {
        let moe = MoELayer::new(64, 256, 8, 2);
        let params = moe.parameters();
        // Router: 1 weight (no bias)
        // 8 experts * 3 weights each = 24
        // Total = 25
        assert_eq!(params.len(), 25);
    }

    #[test]
    fn test_moe_load_balancing_loss() {
        let moe = MoELayer::new(64, 256, 4, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 5 * 64], &[2, 5, 64]).expect("tensor creation failed"),
            false,
        );
        let _output = moe.forward(&input);

        let lb_loss = moe.load_balancing_loss();
        let loss_val = lb_loss.data().to_vec()[0];
        // Load balancing loss should be positive
        assert!(loss_val > 0.0, "Load balancing loss should be > 0");
    }

    #[test]
    fn test_moe_expert_utilization() {
        let moe = MoELayer::new(64, 256, 4, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 5 * 64], &[2, 5, 64]).expect("tensor creation failed"),
            false,
        );
        let _output = moe.forward(&input);

        let util = moe.expert_utilization();
        assert_eq!(util.len(), 4);
        let total: usize = util.iter().sum();
        // Each of 10 tokens selects top-2 experts = 20 assignments total
        assert_eq!(total, 20);
    }

    #[test]
    fn test_moe_named_parameters() {
        let moe = MoELayer::new(64, 256, 4, 2);
        let named = moe.named_parameters();
        assert!(named.contains_key("router.gate.weight"));
        assert!(named.contains_key("experts.0.up_proj.weight"));
        assert!(named.contains_key("experts.3.down_proj.weight"));
    }
}
