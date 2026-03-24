//! Trident - 1.58-bit Ternary Weight Small Language Model
//!
//! A BitNet b1.58-style transformer language model using ternary weights {-1, 0, +1}.
//! All linear projections in the transformer blocks use TernaryLinear, while
//! embeddings and the LM head remain in full precision (fp32).
//!
//! Architecture:
//!   Token Embedding (fp32) -> [TridentBlock x N] -> RMSNorm -> LM Head (fp32)
//!
//! Each TridentBlock:
//!   RMSNorm -> TernaryLinear(QKV) -> Multi-Head Attention -> TernaryLinear(out_proj) -> residual
//!   RMSNorm -> TernaryLinear(up) -> SiLU -> TernaryLinear(down) -> residual
//!
//! Key properties:
//! - 16x memory compression for transformer weights (2-bit vs 32-bit)
//! - Inference matmul reduces to addition/subtraction (no FP multiply for weights)
//! - Full-precision activations throughout for accuracy
//! - Trained with Straight-Through Estimator (STE) on shadow weights
//!
//! # File
//! `crates/axonml-llm/src/trident.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 19, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::any::Any;

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_nn::layers::ternary::TernaryLinear;
use axonml_nn::{Embedding, Linear, Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// Trident Configuration
// =============================================================================

/// Configuration for the Trident 1.58-bit SLM.
#[derive(Debug, Clone)]
pub struct TridentConfig {
    /// Vocabulary size (default: 32000)
    pub vocab_size: usize,
    /// Hidden dimension / model dimension (default: 512)
    pub d_model: usize,
    /// Number of transformer layers (default: 12)
    pub num_layers: usize,
    /// Number of attention heads (default: 8)
    pub num_heads: usize,
    /// Intermediate size for MLP (default: 4 * d_model)
    pub intermediate_size: usize,
    /// Maximum sequence length (default: 2048)
    pub max_seq_len: usize,
    /// RMSNorm epsilon (default: 1e-6)
    pub rms_norm_eps: f32,
}

impl TridentConfig {
    /// Default 150M-equivalent parameter configuration.
    pub fn default_150m() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            num_layers: 12,
            num_heads: 8,
            intermediate_size: 2048,
            max_seq_len: 2048,
            rms_norm_eps: 1e-6,
        }
    }

    /// Small configuration for testing.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 64,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 256,
            max_seq_len: 128,
            rms_norm_eps: 1e-6,
        }
    }

    /// Medium configuration (~300M equivalent).
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 768,
            num_layers: 16,
            num_heads: 12,
            intermediate_size: 3072,
            max_seq_len: 2048,
            rms_norm_eps: 1e-6,
        }
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Estimate total parameters (fp32-equivalent).
    pub fn estimated_params(&self) -> usize {
        let embedding = self.vocab_size * self.d_model;
        let lm_head = self.d_model * self.vocab_size;
        let per_layer = {
            // QKV + output projection
            let attn = 4 * self.d_model * self.d_model;
            // Up + down MLP
            let mlp = 2 * self.d_model * self.intermediate_size;
            // RMSNorm weights (2 per layer)
            let norms = 2 * self.d_model;
            attn + mlp + norms
        };
        embedding + lm_head + self.num_layers * per_layer
    }

    /// Estimate ternary storage in bytes (transformer weights only).
    pub fn ternary_storage_bytes(&self) -> usize {
        let per_layer = {
            let attn = 4 * self.d_model * self.d_model;
            let mlp = 2 * self.d_model * self.intermediate_size;
            attn + mlp
        };
        let total_ternary_weights = self.num_layers * per_layer;
        // 2 bits per weight, packed 4 per byte, + scale factors
        let packed_bytes = total_ternary_weights.div_ceil(4);
        let scale_bytes = self.num_layers * 6 * 4; // 6 TernaryLinear layers per block, 4 bytes per scale
        // Add fp32 embeddings + lm_head + norms
        let fp32_bytes = (self.vocab_size * self.d_model
            + self.d_model * self.vocab_size
            + self.num_layers * 2 * self.d_model)
            * 4;
        packed_bytes + scale_bytes + fp32_bytes
    }

    /// Estimate fp32 storage in bytes (for comparison).
    pub fn fp32_storage_bytes(&self) -> usize {
        self.estimated_params() * 4
    }
}

// =============================================================================
// RMSNorm (local copy to avoid LLM crate circular dep)
// =============================================================================

/// Root Mean Square Layer Normalization for Trident.
#[derive(Debug)]
struct TridentRMSNorm {
    weight: Tensor<f32>,
    eps: f32,
    #[allow(dead_code)]
    hidden_size: usize,
}

impl TridentRMSNorm {
    fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::ones(&[hidden_size]),
            eps,
            hidden_size,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let x_data = x.data();
        let shape = x_data.shape();
        let last_dim = shape[shape.len() - 1];
        let x_vec = x_data.to_vec();
        let batch_elements: usize = shape.iter().take(shape.len() - 1).product();

        let mut output = vec![0.0f32; x_vec.len()];
        let mut rms_vals = vec![0.0f32; batch_elements];
        let weight_vec = self.weight.to_vec();

        for (b, rms_val) in rms_vals.iter_mut().enumerate() {
            let offset = b * last_dim;
            let mut sum_sq = 0.0f32;
            for i in 0..last_dim {
                sum_sq += x_vec[offset + i] * x_vec[offset + i];
            }
            let rms = (sum_sq / last_dim as f32 + self.eps).sqrt();
            *rms_val = rms;

            for i in 0..last_dim {
                output[offset + i] = (x_vec[offset + i] / rms) * weight_vec[i];
            }
        }

        let output_tensor = Tensor::from_vec(output, shape).unwrap();
        let requires_grad = x.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(TridentRMSNormBackward {
                next_fns: vec![x.grad_fn().cloned()],
                saved_input: x_data.clone(),
                weight: self.weight.clone(),
                rms_vals,
                last_dim,
            });
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::named("weight", self.weight.clone(), true)]
    }
}

/// RMSNorm backward for Trident.
#[derive(Debug)]
struct TridentRMSNormBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    weight: Tensor<f32>,
    rms_vals: Vec<f32>,
    last_dim: usize,
}

impl GradientFunction for TridentRMSNormBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let x_vec = self.saved_input.to_vec();
        let w_vec = self.weight.to_vec();
        let g_vec = grad_output.to_vec();
        let d = self.last_dim;
        let batch_elements = self.rms_vals.len();

        let mut grad_input = vec![0.0f32; x_vec.len()];

        for b in 0..batch_elements {
            let off = b * d;
            let rms = self.rms_vals[b];
            let rms_inv = 1.0 / rms;
            let rms3_inv = rms_inv * rms_inv * rms_inv;

            let mut dot = 0.0f32;
            for i in 0..d {
                dot += x_vec[off + i] * w_vec[i] * g_vec[off + i];
            }

            for i in 0..d {
                grad_input[off + i] = w_vec[i] * g_vec[off + i] * rms_inv
                    - x_vec[off + i] * dot * rms3_inv / d as f32;
            }
        }

        let gi = Tensor::from_vec(grad_input, self.saved_input.shape()).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "TridentRMSNormBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Trident Attention
// =============================================================================

/// Multi-head attention with ternary weight projections.
///
/// Uses TernaryLinear for Q, K, V, and output projections.
/// Activations remain in full precision (fp32) throughout.
#[derive(Debug)]
struct TridentAttention {
    /// Query projection (ternary)
    q_proj: TernaryLinear,
    /// Key projection (ternary)
    k_proj: TernaryLinear,
    /// Value projection (ternary)
    v_proj: TernaryLinear,
    /// Output projection (ternary)
    o_proj: TernaryLinear,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Hidden size
    hidden_size: usize,
}

impl TridentAttention {
    fn new(config: &TridentConfig) -> Self {
        let head_dim = config.head_dim();
        Self {
            q_proj: TernaryLinear::with_bias(config.d_model, config.d_model, false),
            k_proj: TernaryLinear::with_bias(config.d_model, config.d_model, false),
            v_proj: TernaryLinear::with_bias(config.d_model, config.d_model, false),
            o_proj: TernaryLinear::with_bias(config.d_model, config.d_model, false),
            num_heads: config.num_heads,
            head_dim,
            hidden_size: config.d_model,
        }
    }

    fn forward(&self, hidden_states: &Variable) -> Variable {
        let data = hidden_states.data();
        let shape = data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V through ternary layers
        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);

        // Reshape for multi-head attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)).mul_scalar(scale);

        // Causal mask
        let mask = self.create_causal_mask(seq_len);
        let attn_weights = attn_weights.add(&Variable::new(mask, false));

        // Softmax
        let attn_weights = attn_weights.softmax(-1);

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output =
            attn_output
                .transpose(1, 2)
                .reshape(&[batch_size, seq_len, self.hidden_size]);

        // Output projection (ternary)
        self.o_proj.forward(&attn_output)
    }

    fn create_causal_mask(&self, seq_len: usize) -> Tensor<f32> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len]).unwrap()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params
    }

    fn quantize_for_inference(&mut self) {
        self.q_proj.quantize_for_inference();
        self.k_proj.quantize_for_inference();
        self.v_proj.quantize_for_inference();
        self.o_proj.quantize_for_inference();
    }
}

// =============================================================================
// Trident MLP
// =============================================================================

/// Feed-forward MLP with ternary weight projections and SiLU activation.
///
/// Architecture: TernaryLinear(up) -> SiLU -> TernaryLinear(down)
/// Activations remain fp32 throughout.
#[derive(Debug)]
struct TridentMLP {
    /// Up projection (ternary)
    up_proj: TernaryLinear,
    /// Down projection (ternary)
    down_proj: TernaryLinear,
}

impl TridentMLP {
    fn new(config: &TridentConfig) -> Self {
        Self {
            up_proj: TernaryLinear::with_bias(config.d_model, config.intermediate_size, false),
            down_proj: TernaryLinear::with_bias(config.intermediate_size, config.d_model, false),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let up = self.up_proj.forward(x).silu();
        self.down_proj.forward(&up)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }

    fn quantize_for_inference(&mut self) {
        self.up_proj.quantize_for_inference();
        self.down_proj.quantize_for_inference();
    }
}

// =============================================================================
// Trident Transformer Block
// =============================================================================

/// Single Trident transformer block with ternary weights.
///
/// Architecture (pre-norm):
///   residual + TridentAttention(RMSNorm(x))
///   residual + TridentMLP(RMSNorm(x))
#[derive(Debug)]
struct TridentBlock {
    /// Pre-attention normalization
    attn_norm: TridentRMSNorm,
    /// Multi-head self-attention (ternary projections)
    attention: TridentAttention,
    /// Pre-MLP normalization
    mlp_norm: TridentRMSNorm,
    /// Feed-forward MLP (ternary projections)
    mlp: TridentMLP,
}

impl TridentBlock {
    fn new(config: &TridentConfig) -> Self {
        Self {
            attn_norm: TridentRMSNorm::new(config.d_model, config.rms_norm_eps),
            attention: TridentAttention::new(config),
            mlp_norm: TridentRMSNorm::new(config.d_model, config.rms_norm_eps),
            mlp: TridentMLP::new(config),
        }
    }

    fn forward(&self, hidden_states: &Variable) -> Variable {
        // Self-attention with pre-norm + residual
        let residual = hidden_states.clone();
        let normed = self.attn_norm.forward(hidden_states);
        let attn_out = self.attention.forward(&normed);
        let hidden_states = residual.add(&attn_out);

        // MLP with pre-norm + residual
        let residual = hidden_states.clone();
        let normed = self.mlp_norm.forward(&hidden_states);
        let mlp_out = self.mlp.forward(&normed);
        residual.add(&mlp_out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.attn_norm.parameters());
        params.extend(self.attention.parameters());
        params.extend(self.mlp_norm.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    fn quantize_for_inference(&mut self) {
        self.attention.quantize_for_inference();
        self.mlp.quantize_for_inference();
    }
}

// =============================================================================
// Trident Model
// =============================================================================

/// Trident: a 1.58-bit ternary weight small language model.
///
/// Uses BitNet b1.58 ternary quantization for all transformer linear layers,
/// keeping embeddings and the LM head in full precision.
///
/// # Example
/// ```ignore
/// let config = TridentConfig::default_150m();
/// let model = TridentModel::new(&config);
///
/// let input_ids = Tensor::from_vec(vec![1u32, 42, 100], &[1, 3]).unwrap();
/// let logits = model.forward_ids(&input_ids);
/// // logits shape: [1, 3, 32000]
/// ```
pub struct TridentModel {
    /// Token embedding (fp32)
    embed_tokens: Embedding,
    /// Transformer blocks with ternary weights
    blocks: Vec<TridentBlock>,
    /// Final RMSNorm
    final_norm: TridentRMSNorm,
    /// Language model head (fp32)
    lm_head: Linear,
    /// Model configuration
    config: TridentConfig,
}

impl TridentModel {
    /// Create a new Trident model.
    pub fn new(config: &TridentConfig) -> Self {
        let blocks = (0..config.num_layers)
            .map(|_| TridentBlock::new(config))
            .collect();

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.d_model),
            blocks,
            final_norm: TridentRMSNorm::new(config.d_model, config.rms_norm_eps),
            lm_head: Linear::with_bias(config.d_model, config.vocab_size, false),
            config: config.clone(),
        }
    }

    /// Forward pass with token IDs, returning logits.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        // Convert token IDs to float for embedding lookup
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var = Variable::new(Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(), false);

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(&ids_var);

        // Pass through transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        // Final norm
        hidden = self.final_norm.forward(&hidden);

        // LM head
        self.lm_head.forward(&hidden)
    }

    /// Forward pass with loss computation (for training).
    ///
    /// Internally shifts logits and labels for next-token prediction
    /// and computes cross-entropy loss.
    pub fn forward_with_loss(
        &self,
        input_ids: &Tensor<u32>,
        labels: &Tensor<u32>,
    ) -> (Variable, Variable) {
        let logits = self.forward_ids(input_ids);

        let logits_data = logits.data();
        let shape = logits_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let _vocab_size = shape[2];

        if seq_len > 1 {
            // Shift for next-token prediction: logits[..., :-1, :] predicts labels[..., 1:]
            // Use narrow() instead of slice() to preserve the autograd graph
            let shift_logits = logits.narrow(1, 0, seq_len - 1);

            // Build shifted labels
            let labels_vec = labels.to_vec();
            let mut shift_labels_data = Vec::with_capacity(batch_size * (seq_len - 1));
            for b in 0..batch_size {
                for s in 1..seq_len {
                    shift_labels_data.push(labels_vec[b * seq_len + s]);
                }
            }
            let shift_labels =
                Tensor::from_vec(shift_labels_data, &[batch_size, seq_len - 1]).unwrap();

            let loss = Self::cross_entropy_loss(&shift_logits, &shift_labels);
            (logits, loss)
        } else {
            let zero_loss = Variable::new(Tensor::from_vec(vec![0.0f32], &[1]).unwrap(), false);
            (logits, zero_loss)
        }
    }

    /// Cross-entropy loss computation.
    fn cross_entropy_loss(logits: &Variable, labels: &Tensor<u32>) -> Variable {
        let logits_data = logits.data();
        let shape = logits_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        let logits_flat = logits.reshape(&[batch_size * seq_len, vocab_size]);

        let labels_vec = labels.to_vec();
        let valid_labels: Vec<f32> = labels_vec
            .iter()
            .map(|&l| {
                let label = l as usize;
                if label < vocab_size { l as f32 } else { 0.0f32 }
            })
            .collect();
        let target_var = Variable::new(
            Tensor::from_vec(valid_labels, &[batch_size * seq_len]).unwrap(),
            false,
        );

        use axonml_nn::loss::CrossEntropyLoss;
        CrossEntropyLoss::new().compute(&logits_flat, &target_var)
    }

    /// Quantize all ternary layers for inference.
    pub fn quantize_for_inference(&mut self) {
        for block in &mut self.blocks {
            block.quantize_for_inference();
        }
    }

    /// Get average weight sparsity across all ternary layers.
    pub fn average_sparsity(&self) -> f32 {
        let mut total_sparsity = 0.0f32;
        let mut count = 0usize;

        for block in &self.blocks {
            // Attention projections
            total_sparsity += block.attention.q_proj.weight_sparsity();
            total_sparsity += block.attention.k_proj.weight_sparsity();
            total_sparsity += block.attention.v_proj.weight_sparsity();
            total_sparsity += block.attention.o_proj.weight_sparsity();
            // MLP projections
            total_sparsity += block.mlp.up_proj.weight_sparsity();
            total_sparsity += block.mlp.down_proj.weight_sparsity();
            count += 6;
        }

        if count > 0 {
            total_sparsity / count as f32
        } else {
            0.0
        }
    }

    /// Get the model configuration.
    pub fn config(&self) -> &TridentConfig {
        &self.config
    }

    /// Report model statistics.
    pub fn report(&self) {
        let param_count: usize = self.parameters().iter().map(|p| p.data().numel()).sum();
        let fp32_mb = self.config.fp32_storage_bytes() as f32 / (1024.0 * 1024.0);
        let ternary_mb = self.config.ternary_storage_bytes() as f32 / (1024.0 * 1024.0);
        let compression = fp32_mb / ternary_mb;

        println!("Trident Model Report");
        println!("====================");
        println!("Layers       : {}", self.config.num_layers);
        println!("d_model      : {}", self.config.d_model);
        println!("Heads        : {}", self.config.num_heads);
        println!("Vocab        : {}", self.config.vocab_size);
        println!("Parameters   : {}", param_count);
        println!("FP32 size    : {:.1} MB", fp32_mb);
        println!("Ternary size : {:.1} MB", ternary_mb);
        println!("Compression  : {:.1}x", compression);
        println!("Sparsity     : {:.1}%", self.average_sparsity() * 100.0);
    }
}

impl Module for TridentModel {
    fn forward(&self, input: &Variable) -> Variable {
        // When called via Module trait, assume input contains token ID floats
        let mut hidden = self.embed_tokens.forward(input);

        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        hidden = self.final_norm.forward(&hidden);
        self.lm_head.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.embed_tokens.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.final_norm.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    fn name(&self) -> &'static str {
        "TridentModel"
    }
}

impl std::fmt::Debug for TridentModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TridentModel")
            .field("vocab_size", &self.config.vocab_size)
            .field("d_model", &self.config.d_model)
            .field("num_layers", &self.config.num_layers)
            .field("num_heads", &self.config.num_heads)
            .field("intermediate_size", &self.config.intermediate_size)
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
    fn test_trident_config() {
        let config = TridentConfig::default_150m();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_trident_config_storage() {
        let config = TridentConfig::default_150m();
        let fp32 = config.fp32_storage_bytes();
        let ternary = config.ternary_storage_bytes();
        // Ternary should be significantly smaller
        assert!(ternary < fp32);
        println!(
            "FP32: {} bytes, Ternary: {} bytes, Ratio: {:.1}x",
            fp32,
            ternary,
            fp32 as f32 / ternary as f32
        );
    }

    #[test]
    fn test_trident_tiny_forward() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let logits = model.forward_ids(&input_ids);
        assert_eq!(logits.data().shape(), &[2, 2, config.vocab_size]);
    }

    #[test]
    fn test_trident_parameters() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);
        let params = model.parameters();
        assert!(!params.is_empty());

        let total: usize = params.iter().map(|p| p.data().numel()).sum();
        println!("Tiny Trident params: {}", total);
    }

    #[test]
    fn test_trident_forward_with_loss() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let labels = Tensor::from_vec(vec![2u32, 3, 4, 5, 6, 7], &[2, 3]).unwrap();

        let (logits, loss) = model.forward_with_loss(&input_ids, &labels);
        assert_eq!(logits.data().shape(), &[2, 3, config.vocab_size]);
        assert_eq!(loss.data().numel(), 1);

        let loss_val = loss.data().to_vec()[0];
        assert!(loss_val > 0.0, "Loss should be positive, got {}", loss_val);
    }

    #[test]
    fn test_trident_sparsity() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);
        let sparsity = model.average_sparsity();
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
    }

    #[test]
    fn test_trident_quantize_inference() {
        let config = TridentConfig::tiny();
        let mut model = TridentModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], &[1, 3]).unwrap();

        // Training forward
        let logits_train = model.forward_ids(&input_ids);

        // Quantize for inference
        model.quantize_for_inference();
        let logits_infer = model.forward_ids(&input_ids);

        // Should produce same shape
        assert_eq!(logits_train.data().shape(), logits_infer.data().shape());

        // Values should be very close (same quantization)
        let train_vec = logits_train.data().to_vec();
        let infer_vec = logits_infer.data().to_vec();
        for (a, b) in train_vec.iter().zip(infer_vec.iter()) {
            assert!((a - b).abs() < 1e-4, "Train {} vs infer {} mismatch", a, b);
        }
    }
}
