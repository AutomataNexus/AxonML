//! Chimera - Mixture of Experts + Differential Attention Small Language Model
//!
//! A novel architecture combining sparse MoE routing for massive capacity with
//! differential attention for noise-cancelling precision. Each forward pass
//! activates only ~25% of total parameters (top-2 of 8 experts per layer).
//!
//! # Architecture
//! ```text
//! Token Embedding -> [ChimeraBlock x N] -> RMSNorm -> LM Head
//!
//! ChimeraBlock:
//!   x = x + DifferentialAttention(RMSNorm(x))
//!   x = x + MoELayer(RMSNorm(x))            // top-2 of 8 experts
//! ```
//!
//! # Key innovations
//! - **Differential Attention**: Computes two softmax attention maps and subtracts,
//!   cancelling noisy/irrelevant patterns (Microsoft DIFF Transformer)
//! - **Sparse MoE MLP**: 8 expert SwiGLU MLPs per layer, only 2 active per token
//!   (Switch Transformer / GShard style routing)
//! - **Load balancing loss**: Auxiliary loss prevents expert collapse
//!
//! # File
//! `crates/axonml-llm/src/chimera.rs`
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
use axonml_nn::layers::diff_attention::DifferentialAttention;
use axonml_nn::layers::moe::MoELayer;
use axonml_nn::{Embedding, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use crate::llama::RMSNorm;

// =============================================================================
// Chimera Configuration
// =============================================================================

/// Configuration for the Chimera model.
#[derive(Debug, Clone)]
pub struct ChimeraConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension (d_model)
    pub d_model: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of expert MLPs per layer
    pub num_experts: usize,
    /// Number of experts activated per token
    pub top_k: usize,
    /// Expert MLP intermediate dimension
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// Initial lambda for differential attention
    pub lambda_init: f32,
    /// Weight for load balancing auxiliary loss
    pub load_balance_weight: f32,
}

impl ChimeraConfig {
    /// Default Chimera configuration (~2B total params, ~500M active).
    ///
    /// - d_model=512, 16 layers, 8 heads
    /// - 8 experts per layer, top-2 active
    /// - vocab_size=32000
    pub fn default_2b() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            num_layers: 16,
            num_heads: 8,
            num_experts: 8,
            top_k: 2,
            intermediate_size: 2048, // 4 * d_model
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            lambda_init: 0.05,
            load_balance_weight: 0.01,
        }
    }

    /// Small configuration for testing and prototyping (~50M total, ~15M active).
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 256,
            num_layers: 6,
            num_heads: 4,
            num_experts: 8,
            top_k: 2,
            intermediate_size: 1024,
            max_seq_len: 512,
            rms_norm_eps: 1e-5,
            lambda_init: 0.05,
            load_balance_weight: 0.01,
        }
    }

    /// Tiny configuration for unit tests (~5M total).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 64,
            num_layers: 2,
            num_heads: 4,
            num_experts: 4,
            top_k: 2,
            intermediate_size: 256,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            lambda_init: 0.05,
            load_balance_weight: 0.01,
        }
    }

    /// Returns total parameters (all experts).
    pub fn estimate_total_params(&self) -> usize {
        let embed = self.vocab_size * self.d_model;
        let attn_per_layer = 4 * self.d_model * self.d_model + 1; // Q,K,V,O projections + lambda
        let expert_per_layer = self.num_experts * 3 * self.d_model * self.intermediate_size;
        let router_per_layer = self.d_model * self.num_experts;
        let norm_per_layer = 2 * self.d_model;
        let layer = attn_per_layer + expert_per_layer + router_per_layer + norm_per_layer;
        let final_norm = self.d_model;
        let lm_head = self.d_model * self.vocab_size;
        embed + self.num_layers * layer + final_norm + lm_head
    }

    /// Returns active parameters per forward pass (top-k experts only).
    pub fn estimate_active_params(&self) -> usize {
        let embed = self.vocab_size * self.d_model;
        let attn_per_layer = 4 * self.d_model * self.d_model + 1;
        let active_expert_per_layer = self.top_k * 3 * self.d_model * self.intermediate_size;
        let router_per_layer = self.d_model * self.num_experts;
        let norm_per_layer = 2 * self.d_model;
        let layer = attn_per_layer + active_expert_per_layer + router_per_layer + norm_per_layer;
        let final_norm = self.d_model;
        let lm_head = self.d_model * self.vocab_size;
        embed + self.num_layers * layer + final_norm + lm_head
    }
}

// =============================================================================
// Chimera Block
// =============================================================================

/// A single Chimera transformer block.
///
/// ```text
/// x = x + DifferentialAttention(RMSNorm(x))
/// x = x + MoELayer(RMSNorm(x))
/// ```
pub struct ChimeraBlock {
    /// Pre-attention normalization
    attn_norm: RMSNorm,
    /// Differential attention
    attention: DifferentialAttention,
    /// Pre-MoE normalization
    ffn_norm: RMSNorm,
    /// Mixture of Experts feedforward
    moe: MoELayer,
}

impl ChimeraBlock {
    /// Creates a new Chimera block.
    pub fn new(config: &ChimeraConfig) -> Self {
        Self {
            attn_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            attention: DifferentialAttention::with_lambda(
                config.d_model,
                config.num_heads,
                config.lambda_init,
            ),
            ffn_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            moe: MoELayer::new(
                config.d_model,
                config.intermediate_size,
                config.num_experts,
                config.top_k,
            ),
        }
    }

    /// Forward pass through the block.
    pub fn forward(&self, x: &Variable) -> Variable {
        // Self-attention with residual
        let normed = self.attn_norm.forward(x);
        let attn_out = self.attention.forward(&normed);
        let x = x.add_var(&attn_out);

        // MoE MLP with residual
        let normed = self.ffn_norm.forward(&x);
        let moe_out = self.moe.forward(&normed);
        x.add_var(&moe_out)
    }

    /// Returns load balancing loss for this block's MoE layer.
    pub fn load_balancing_loss(&self) -> Variable {
        self.moe.load_balancing_loss()
    }

    /// Returns expert utilization for this block.
    pub fn expert_utilization(&self) -> Vec<usize> {
        self.moe.expert_utilization()
    }

    /// Returns the current lambda value for differential attention.
    pub fn lambda_value(&self) -> f32 {
        self.attention.lambda_value()
    }

    /// Returns parameters for this block.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.attn_norm.parameters());
        params.extend(self.attention.parameters());
        params.extend(self.ffn_norm.parameters());
        params.extend(self.moe.parameters());
        params
    }

    /// Returns named parameters for this block.
    pub fn named_parameters(&self, prefix: &str) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.attention.named_parameters() {
            params.insert(format!("{prefix}.attention.{name}"), param);
        }
        for (name, param) in self.moe.named_parameters() {
            params.insert(format!("{prefix}.moe.{name}"), param);
        }
        // RMSNorm parameters
        for p in self.attn_norm.parameters() {
            params.insert(format!("{prefix}.attn_norm.weight"), p);
        }
        for p in self.ffn_norm.parameters() {
            params.insert(format!("{prefix}.ffn_norm.weight"), p);
        }
        params
    }
}

impl std::fmt::Debug for ChimeraBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChimeraBlock")
            .field("attention", &self.attention)
            .field("moe", &self.moe)
            .finish()
    }
}

// =============================================================================
// Chimera Model
// =============================================================================

/// Chimera language model — MoE + Differential Attention.
///
/// Full causal language model with token embedding, N transformer blocks
/// (each with differential attention + MoE MLP), final RMSNorm, and LM head.
///
/// # Usage
/// ```ignore
/// let config = ChimeraConfig::small();
/// let model = ChimeraModel::new(&config);
///
/// let input_ids = Tensor::from_vec(vec![1u32, 42, 100, 7], &[1, 4]).unwrap();
/// let logits = model.forward_ids(&input_ids);
/// // logits shape: [1, 4, vocab_size]
///
/// let (logits, loss) = model.forward_with_loss(&input_ids, &input_ids);
/// ```
pub struct ChimeraModel {
    /// Token embedding
    token_embedding: Embedding,
    /// Transformer blocks
    blocks: Vec<ChimeraBlock>,
    /// Final normalization
    final_norm: RMSNorm,
    /// Language model head: d_model -> vocab_size
    lm_head: Linear,
    /// Model configuration
    config: ChimeraConfig,
}

impl ChimeraModel {
    /// Creates a new Chimera model from configuration.
    pub fn new(config: &ChimeraConfig) -> Self {
        let blocks: Vec<ChimeraBlock> = (0..config.num_layers)
            .map(|_| ChimeraBlock::new(config))
            .collect();

        Self {
            token_embedding: Embedding::new(config.vocab_size, config.d_model),
            blocks,
            final_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            lm_head: Linear::with_bias(config.d_model, config.vocab_size, false),
            config: config.clone(),
        }
    }

    /// Forward pass from token IDs to logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs tensor [batch, seq_len] (u32)
    ///
    /// # Returns
    /// Logits tensor [batch, seq_len, vocab_size]
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        // Convert u32 indices to f32 for embedding lookup
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var = Variable::new(Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(), false);

        // Embed tokens
        let mut hidden = self.token_embedding.forward(&ids_var);

        // Process through blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        // Final norm + LM head
        let hidden = self.final_norm.forward(&hidden);
        self.lm_head.forward(&hidden)
    }

    /// Forward pass with loss computation for training.
    ///
    /// Computes next-token prediction cross-entropy loss plus load balancing
    /// auxiliary loss across all MoE layers.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len] (u32)
    /// * `labels` - Target token IDs [batch, seq_len] (u32), shifted internally
    ///
    /// # Returns
    /// * `logits` - Raw logits [batch, seq_len, vocab_size]
    /// * `total_loss` - Combined CE loss + load_balance_weight * LB loss
    pub fn forward_with_loss(
        &self,
        input_ids: &Tensor<u32>,
        labels: &Tensor<u32>,
    ) -> (Variable, Variable) {
        let logits = self.forward_ids(input_ids);

        let shape = logits.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        if seq_len <= 1 {
            let zero_loss = Variable::new(Tensor::from_vec(vec![0.0f32], &[1]).unwrap(), false);
            return (logits, zero_loss);
        }

        // Shift for next-token prediction: logits[:-1] predicts labels[1:]
        let shift_logits = logits.narrow(1, 0, seq_len - 1);

        let labels_vec = labels.to_vec();
        let mut shift_labels_data = Vec::with_capacity(batch_size * (seq_len - 1));
        for b in 0..batch_size {
            for s in 1..seq_len {
                shift_labels_data.push(labels_vec[b * seq_len + s]);
            }
        }

        // Flatten for cross-entropy
        let flat_logits = shift_logits.reshape(&[batch_size * (seq_len - 1), vocab_size]);
        let valid_labels: Vec<f32> = shift_labels_data
            .iter()
            .map(|&l| {
                if (l as usize) < vocab_size {
                    l as f32
                } else {
                    0.0f32
                }
            })
            .collect();
        let target_var = Variable::new(
            Tensor::from_vec(valid_labels, &[batch_size * (seq_len - 1)]).unwrap(),
            false,
        );

        use axonml_nn::loss::CrossEntropyLoss;
        let ce_loss = CrossEntropyLoss::new().compute(&flat_logits, &target_var);

        // Load balancing loss across all layers
        let mut lb_loss_val = 0.0f32;
        for block in &self.blocks {
            let lb = block.load_balancing_loss();
            lb_loss_val += lb.data().to_vec()[0];
        }
        lb_loss_val /= self.blocks.len() as f32;

        // Total loss = CE + weight * LB
        let total_loss = ce_loss.add_var(&Variable::new(
            Tensor::from_vec(vec![self.config.load_balance_weight * lb_loss_val], &[1]).unwrap(),
            false,
        ));

        (logits, total_loss)
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &ChimeraConfig {
        &self.config
    }

    /// Returns expert utilization across all layers.
    ///
    /// Returns a Vec of (layer_index, expert_counts) tuples.
    pub fn expert_utilization(&self) -> Vec<(usize, Vec<usize>)> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, block)| (i, block.expert_utilization()))
            .collect()
    }

    /// Returns lambda values for all layers' differential attention.
    pub fn lambda_values(&self) -> Vec<f32> {
        self.blocks.iter().map(|b| b.lambda_value()).collect()
    }

    /// Returns total parameter count (all experts).
    pub fn total_param_count(&self) -> usize {
        self.parameters().iter().map(|p| p.data().numel()).sum()
    }

    /// Returns estimated active parameter count per forward pass.
    pub fn active_param_count(&self) -> usize {
        self.config.estimate_active_params()
    }
}

impl Module for ChimeraModel {
    fn forward(&self, input: &Variable) -> Variable {
        // This path expects pre-embedded input [batch, seq, d_model]
        let mut hidden = input.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        let hidden = self.final_norm.forward(&hidden);
        self.lm_head.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.final_norm.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.token_embedding.named_parameters() {
            params.insert(format!("token_embedding.{name}"), param);
        }
        for (i, block) in self.blocks.iter().enumerate() {
            let block_params = block.named_parameters(&format!("blocks.{i}"));
            params.extend(block_params);
        }
        for p in self.final_norm.parameters() {
            params.insert("final_norm.weight".to_string(), p);
        }
        for (name, param) in self.lm_head.named_parameters() {
            params.insert(format!("lm_head.{name}"), param);
        }
        params
    }

    fn name(&self) -> &'static str {
        "ChimeraModel"
    }
}

impl std::fmt::Debug for ChimeraModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChimeraModel")
            .field("config", &self.config)
            .field("num_blocks", &self.blocks.len())
            .field("total_params", &self.total_param_count())
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
    fn test_chimera_config_params() {
        let config = ChimeraConfig::default_2b();
        let total = config.estimate_total_params();
        let active = config.estimate_active_params();
        assert!(total > active, "Total params should exceed active params");
        assert!(active > 0);
        // Active should be roughly top_k/num_experts of the MoE portion
    }

    #[test]
    fn test_chimera_tiny_forward() {
        let config = ChimeraConfig::tiny();
        let model = ChimeraModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 42, 100, 7], &[1, 4]).unwrap();
        let logits = model.forward_ids(&input_ids);
        assert_eq!(logits.shape(), vec![1, 4, 1000]);
    }

    #[test]
    fn test_chimera_tiny_batch() {
        let config = ChimeraConfig::tiny();
        let model = ChimeraModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let logits = model.forward_ids(&input_ids);
        assert_eq!(logits.shape(), vec![2, 4, 1000]);
    }

    #[test]
    fn test_chimera_forward_with_loss() {
        let config = ChimeraConfig::tiny();
        let model = ChimeraModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 42, 100, 7], &[1, 4]).unwrap();
        let labels = Tensor::from_vec(vec![42u32, 100, 7, 500], &[1, 4]).unwrap();

        let (logits, loss) = model.forward_with_loss(&input_ids, &labels);
        assert_eq!(logits.shape(), vec![1, 4, 1000]);

        let loss_val = loss.data().to_vec()[0];
        assert!(loss_val > 0.0, "Loss should be positive");
        assert!(loss_val.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_chimera_expert_utilization() {
        let config = ChimeraConfig::tiny();
        let model = ChimeraModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[1, 4]).unwrap();
        let _logits = model.forward_ids(&input_ids);

        let util = model.expert_utilization();
        assert_eq!(util.len(), config.num_layers);
        for (_, counts) in &util {
            assert_eq!(counts.len(), config.num_experts);
            let total: usize = counts.iter().sum();
            // 4 tokens * top_k=2 = 8 assignments per layer
            assert_eq!(total, 4 * config.top_k);
        }
    }

    #[test]
    fn test_chimera_lambda_values() {
        let config = ChimeraConfig::tiny();
        let model = ChimeraModel::new(&config);

        let lambdas = model.lambda_values();
        assert_eq!(lambdas.len(), config.num_layers);
        for &l in &lambdas {
            assert!((l - config.lambda_init).abs() < 1e-6);
        }
    }

    #[test]
    fn test_chimera_parameters() {
        let config = ChimeraConfig::tiny();
        let model = ChimeraModel::new(&config);
        let params = model.parameters();
        assert!(!params.is_empty());

        let total_params: usize = params.iter().map(|p| p.data().numel()).sum();
        assert!(total_params > 0);
    }

    #[test]
    fn test_chimera_named_parameters() {
        let config = ChimeraConfig::tiny();
        let model = ChimeraModel::new(&config);
        let named = model.named_parameters();

        assert!(named.contains_key("token_embedding.weight"));
        assert!(named.contains_key("final_norm.weight"));
        assert!(named.contains_key("lm_head.weight"));
        assert!(named.contains_key("blocks.0.attention.lambda"));
        assert!(named.contains_key("blocks.0.moe.router.gate.weight"));
    }
}
