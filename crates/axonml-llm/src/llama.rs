//! LLaMA - Large Language Model Meta AI
//!
//! Implementation of the LLaMA architecture with RoPE (Rotary Position Embedding),
//! RMSNorm, and SwiGLU activation.
//!
//! Reference: "LLaMA: Open and Efficient Foundation Language Models"
//! https://arxiv.org/abs/2302.13971
//!
//! # Example
//! ```rust,ignore
//! use axonml_llm::{LLaMA, LLaMAConfig};
//!
//! let config = LLaMAConfig::llama2_7b();
//! let model = LLaMA::new(&config);
//! ```

use axonml_autograd::Variable;
use axonml_nn::{Module, Linear, Dropout, Parameter, Embedding};
use axonml_tensor::Tensor;

use crate::attention::{KVCache, LayerKVCache};

// =============================================================================
// LLaMA Configuration
// =============================================================================

/// Configuration for LLaMA models.
#[derive(Debug, Clone)]
pub struct LLaMAConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size (embedding dimension)
    pub hidden_size: usize,
    /// Intermediate size for MLP (typically 4 * hidden_size or 8/3 * hidden_size for SwiGLU)
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for grouped-query attention)
    pub num_key_value_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta (base for rotary embeddings)
    pub rope_theta: f32,
    /// Attention dropout
    pub attention_dropout: f32,
    /// Hidden dropout
    pub hidden_dropout: f32,
}

impl LLaMAConfig {
    /// LLaMA 2 7B configuration
    pub fn llama2_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
        }
    }

    /// LLaMA 2 13B configuration
    pub fn llama2_13b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 40,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
        }
    }

    /// LLaMA 3 8B configuration (with GQA)
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8, // Grouped-query attention
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
        }
    }

    /// Tiny LLaMA for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
        }
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// =============================================================================
// RMSNorm
// =============================================================================

/// Root Mean Square Layer Normalization.
///
/// Unlike LayerNorm, RMSNorm only normalizes by RMS without centering.
/// This is more efficient and works well for LLMs.
#[derive(Debug)]
pub struct RMSNorm {
    /// Learnable scale parameter
    weight: Tensor<f32>,
    /// Epsilon for numerical stability
    eps: f32,
    /// Hidden size
    hidden_size: usize,
}

impl RMSNorm {
    /// Create new RMSNorm layer.
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::ones(&[hidden_size]),
            eps,
            hidden_size,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Variable) -> Variable {
        let x_data = x.data();
        let shape = x_data.shape();
        let last_dim = shape[shape.len() - 1];

        // Compute RMS: sqrt(mean(x^2))
        let x_vec = x_data.to_vec();
        let batch_elements: usize = shape.iter().take(shape.len() - 1).product();

        let mut output = vec![0.0f32; x_vec.len()];

        for b in 0..batch_elements {
            let offset = b * last_dim;

            // Compute mean of squares
            let mut sum_sq = 0.0f32;
            for i in 0..last_dim {
                sum_sq += x_vec[offset + i] * x_vec[offset + i];
            }
            let rms = (sum_sq / last_dim as f32 + self.eps).sqrt();

            // Normalize and scale
            let weight_vec = self.weight.to_vec();
            for i in 0..last_dim {
                output[offset + i] = (x_vec[offset + i] / rms) * weight_vec[i];
            }
        }

        Variable::new(
            Tensor::from_vec(output, shape).unwrap(),
            x.requires_grad(),
        )
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::named("weight", self.weight.clone(), true)]
    }
}

// =============================================================================
// Rotary Position Embedding (RoPE)
// =============================================================================

/// Rotary Position Embedding.
///
/// Encodes position information by rotating pairs of dimensions.
#[derive(Debug)]
pub struct RotaryEmbedding {
    /// Dimension of the embedding
    dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Base theta
    theta: f32,
    /// Precomputed cosine values
    cos_cached: Tensor<f32>,
    /// Precomputed sine values
    sin_cached: Tensor<f32>,
}

impl RotaryEmbedding {
    /// Create new rotary embedding.
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        // Compute inverse frequencies
        let half_dim = dim / 2;
        let mut inv_freq = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            inv_freq[i] = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
        }

        // Precompute cos and sin for all positions
        let mut cos_data = vec![0.0f32; max_seq_len * dim];
        let mut sin_data = vec![0.0f32; max_seq_len * dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                cos_data[pos * dim + i] = angle.cos();
                cos_data[pos * dim + half_dim + i] = angle.cos();
                sin_data[pos * dim + i] = angle.sin();
                sin_data[pos * dim + half_dim + i] = angle.sin();
            }
        }

        Self {
            dim,
            max_seq_len,
            theta,
            cos_cached: Tensor::from_vec(cos_data, &[max_seq_len, dim]).unwrap(),
            sin_cached: Tensor::from_vec(sin_data, &[max_seq_len, dim]).unwrap(),
        }
    }

    /// Apply rotary embedding to query and key tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, num_heads, seq_len, head_dim]
    /// * `position_offset` - Starting position (for KV-cache)
    pub fn apply(&self, q: &Variable, k: &Variable, position_offset: usize) -> (Variable, Variable) {
        let q_data = q.data();
        let k_data = k.data();
        let shape = q_data.shape();
        let seq_len = shape[2];
        let head_dim = shape[3];

        let q_rotated = self.rotate_tensor(&q_data, seq_len, head_dim, position_offset);
        let k_rotated = self.rotate_tensor(&k_data, seq_len, head_dim, position_offset);

        (
            Variable::new(q_rotated, q.requires_grad()),
            Variable::new(k_rotated, k.requires_grad()),
        )
    }

    fn rotate_tensor(&self, x: &Tensor<f32>, seq_len: usize, head_dim: usize, offset: usize) -> Tensor<f32> {
        let shape = x.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let x_vec = x.to_vec();
        let cos_vec = self.cos_cached.to_vec();
        let sin_vec = self.sin_cached.to_vec();

        let mut output = vec![0.0f32; x_vec.len()];
        let half_dim = head_dim / 2;

        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let pos = offset + s;
                    let x_offset = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let rope_offset = pos * self.dim;

                    for i in 0..half_dim {
                        let cos_val = cos_vec[rope_offset + i];
                        let sin_val = sin_vec[rope_offset + i];

                        let x1 = x_vec[x_offset + i];
                        let x2 = x_vec[x_offset + half_dim + i];

                        // Rotate pairs
                        output[x_offset + i] = x1 * cos_val - x2 * sin_val;
                        output[x_offset + half_dim + i] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }

        Tensor::from_vec(output, shape).unwrap()
    }
}

// =============================================================================
// LLaMA Attention
// =============================================================================

/// LLaMA attention with RoPE and optional grouped-query attention (GQA).
#[derive(Debug)]
pub struct LLaMAAttention {
    /// Query projection
    q_proj: Linear,
    /// Key projection
    k_proj: Linear,
    /// Value projection
    v_proj: Linear,
    /// Output projection
    o_proj: Linear,
    /// Rotary embedding
    rotary_emb: RotaryEmbedding,
    /// Number of attention heads
    num_heads: usize,
    /// Number of key-value heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Hidden size
    hidden_size: usize,
    /// Attention dropout
    attn_dropout: Dropout,
}

impl LLaMAAttention {
    /// Create new LLaMA attention layer.
    pub fn new(config: &LLaMAConfig) -> Self {
        let head_dim = config.head_dim();
        let kv_hidden = config.num_key_value_heads * head_dim;

        Self {
            q_proj: Linear::new(config.hidden_size, config.hidden_size),
            k_proj: Linear::new(config.hidden_size, kv_hidden),
            v_proj: Linear::new(config.hidden_size, kv_hidden),
            o_proj: Linear::new(config.hidden_size, config.hidden_size),
            rotary_emb: RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            hidden_size: config.hidden_size,
            attn_dropout: Dropout::new(config.attention_dropout),
        }
    }

    /// Forward pass with optional KV-cache.
    pub fn forward_with_cache(
        &self,
        hidden_states: &Variable,
        kv_cache: Option<&mut KVCache>,
        position_offset: usize,
    ) -> Variable {
        let data = hidden_states.data();
        let shape = data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);

        // Reshape for multi-head attention
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]).transpose(1, 2);
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]).transpose(1, 2);

        // Apply rotary embeddings
        let (q, k) = self.rotary_emb.apply(&q, &k, position_offset);

        // Handle KV-cache
        let (k, v, total_seq_len) = if let Some(cache) = kv_cache {
            let (cached_k, cached_v) = cache.update(&k.data(), &v.data());
            (
                Variable::new(cached_k.clone(), false),
                Variable::new(cached_v, false),
                cached_k.shape()[2],
            )
        } else {
            (k, v, seq_len)
        };

        // Repeat KV heads for grouped-query attention
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            (
                self.repeat_kv(&k, repeat),
                self.repeat_kv(&v, repeat),
            )
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)).mul_scalar(scale);

        // Apply causal mask
        let mask = self.create_causal_mask(seq_len, total_seq_len, position_offset);
        let attn_weights = attn_weights.add(&Variable::new(mask, false));

        // Softmax and dropout
        let attn_weights = attn_weights.softmax(-1);
        let attn_weights = self.attn_dropout.forward(&attn_weights);

        // Compute output
        let attn_output = attn_weights.matmul(&v);
        let attn_output = attn_output.transpose(1, 2).reshape(&[batch_size, seq_len, self.hidden_size]);

        self.o_proj.forward(&attn_output)
    }

    fn repeat_kv(&self, x: &Variable, n_rep: usize) -> Variable {
        if n_rep == 1 {
            return x.clone();
        }

        let data = x.data();
        let shape = data.shape();
        let batch = shape[0];
        let num_kv_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        let data_vec = data.to_vec();
        let mut output = Vec::with_capacity(data_vec.len() * n_rep);

        for b in 0..batch {
            for h in 0..num_kv_heads {
                for _ in 0..n_rep {
                    for s in 0..seq_len {
                        let offset = ((b * num_kv_heads + h) * seq_len + s) * head_dim;
                        output.extend_from_slice(&data_vec[offset..offset + head_dim]);
                    }
                }
            }
        }

        Variable::new(
            Tensor::from_vec(output, &[batch, num_kv_heads * n_rep, seq_len, head_dim]).unwrap(),
            x.requires_grad(),
        )
    }

    fn create_causal_mask(&self, q_len: usize, kv_len: usize, offset: usize) -> Tensor<f32> {
        let mut mask_data = vec![0.0f32; q_len * kv_len];

        for i in 0..q_len {
            let pos = offset + i;
            for j in 0..kv_len {
                if j > pos {
                    mask_data[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        Tensor::from_vec(mask_data, &[1, 1, q_len, kv_len]).unwrap()
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params
    }
}

// =============================================================================
// LLaMA MLP (SwiGLU)
// =============================================================================

/// LLaMA MLP with SwiGLU activation.
#[derive(Debug)]
pub struct LLaMAMLP {
    /// Gate projection
    gate_proj: Linear,
    /// Up projection
    up_proj: Linear,
    /// Down projection
    down_proj: Linear,
}

impl LLaMAMLP {
    /// Create new LLaMA MLP.
    pub fn new(config: &LLaMAConfig) -> Self {
        Self {
            gate_proj: Linear::new(config.hidden_size, config.intermediate_size),
            up_proj: Linear::new(config.hidden_size, config.intermediate_size),
            down_proj: Linear::new(config.intermediate_size, config.hidden_size),
        }
    }

    /// Forward pass with SwiGLU activation.
    pub fn forward(&self, x: &Variable) -> Variable {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        let hidden = gate.mul(&up);
        self.down_proj.forward(&hidden)
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.gate_proj.parameters());
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}

// =============================================================================
// LLaMA Decoder Layer
// =============================================================================

/// Single LLaMA transformer decoder layer.
#[derive(Debug)]
pub struct LLaMADecoderLayer {
    /// Self attention
    self_attn: LLaMAAttention,
    /// MLP
    mlp: LLaMAMLP,
    /// Input layer norm
    input_layernorm: RMSNorm,
    /// Post-attention layer norm
    post_attention_layernorm: RMSNorm,
}

impl LLaMADecoderLayer {
    /// Create new decoder layer.
    pub fn new(config: &LLaMAConfig) -> Self {
        Self {
            self_attn: LLaMAAttention::new(config),
            mlp: LLaMAMLP::new(config),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        }
    }

    /// Forward pass with optional KV-cache.
    pub fn forward_with_cache(
        &self,
        hidden_states: &Variable,
        kv_cache: Option<&mut KVCache>,
        position_offset: usize,
    ) -> Variable {
        // Self attention with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward_with_cache(&hidden_states, kv_cache, position_offset);
        let hidden_states = residual.add(&hidden_states);

        // MLP with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states);
        let hidden_states = self.mlp.forward(&hidden_states);
        residual.add(&hidden_states)
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.mlp.parameters());
        params.extend(self.input_layernorm.parameters());
        params.extend(self.post_attention_layernorm.parameters());
        params
    }
}

// =============================================================================
// LLaMA Model
// =============================================================================

/// LLaMA language model.
#[derive(Debug)]
pub struct LLaMA {
    /// Token embeddings
    embed_tokens: Embedding,
    /// Decoder layers
    layers: Vec<LLaMADecoderLayer>,
    /// Final layer norm
    norm: RMSNorm,
    /// Configuration
    config: LLaMAConfig,
}

impl LLaMA {
    /// Create new LLaMA model.
    pub fn new(config: &LLaMAConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| LLaMADecoderLayer::new(config))
            .collect();

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            config: config.clone(),
        }
    }

    /// Forward pass with token IDs.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        self.forward_with_cache(input_ids, None).0
    }

    /// Forward pass with KV-cache support.
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor<u32>,
        kv_cache: Option<&mut LayerKVCache>,
    ) -> (Variable, usize) {
        let position_offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        // Convert token IDs to Variable for embedding lookup
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var = Variable::new(
            Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(),
            false,
        );

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(&ids_var);

        // Pass through decoder layers
        if let Some(cache) = kv_cache {
            for (i, layer) in self.layers.iter().enumerate() {
                let layer_cache = cache.get_mut(i);
                hidden_states = layer.forward_with_cache(&hidden_states, layer_cache, position_offset);
            }
        } else {
            for layer in &self.layers {
                hidden_states = layer.forward_with_cache(&hidden_states, None, position_offset);
            }
        }

        // Final norm
        let hidden_states = self.norm.forward(&hidden_states);

        (hidden_states, position_offset)
    }

    /// Create KV-cache for this model.
    pub fn create_kv_cache(&self, batch_size: usize) -> LayerKVCache {
        LayerKVCache::new(
            self.config.num_hidden_layers,
            batch_size,
            self.config.num_key_value_heads,
            self.config.max_position_embeddings,
            self.config.head_dim(),
        )
    }
}

impl Module for LLaMA {
    fn forward(&self, input: &Variable) -> Variable {
        // Assume input is already embedded
        let mut hidden_states = input.clone();
        for layer in &self.layers {
            hidden_states = layer.forward_with_cache(&hidden_states, None, 0);
        }
        self.norm.forward(&hidden_states)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.embed_tokens.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        params
    }
}

/// LLaMA with language modeling head.
#[derive(Debug)]
pub struct LLaMAForCausalLM {
    /// Base LLaMA model
    model: LLaMA,
    /// Language modeling head (tied to embeddings)
    lm_head: Linear,
}

impl LLaMAForCausalLM {
    /// Create new LLaMA for causal LM.
    pub fn new(config: &LLaMAConfig) -> Self {
        Self {
            model: LLaMA::new(config),
            lm_head: Linear::new(config.hidden_size, config.vocab_size),
        }
    }

    /// Forward pass returning logits.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        let hidden_states = self.model.forward_ids(input_ids);
        self.lm_head.forward(&hidden_states)
    }

    /// Forward with KV-cache.
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor<u32>,
        kv_cache: Option<&mut LayerKVCache>,
    ) -> Variable {
        let (hidden_states, _) = self.model.forward_with_cache(input_ids, kv_cache);
        self.lm_head.forward(&hidden_states)
    }

    /// Create KV-cache.
    pub fn create_kv_cache(&self, batch_size: usize) -> LayerKVCache {
        self.model.create_kv_cache(batch_size)
    }
}

impl Module for LLaMAForCausalLM {
    fn forward(&self, input: &Variable) -> Variable {
        let hidden_states = self.model.forward(input);
        self.lm_head.forward(&hidden_states)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.model.parameters();
        params.extend(self.lm_head.parameters());
        params
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_config() {
        let config = LLaMAConfig::tiny();
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_rms_norm() {
        let norm = RMSNorm::new(64, 1e-5);
        let input = Variable::new(Tensor::randn(&[2, 8, 64]), false);
        let output = norm.forward(&input);
        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_rotary_embedding() {
        let rope = RotaryEmbedding::new(64, 512, 10000.0);
        let q = Variable::new(Tensor::randn(&[2, 4, 8, 64]), false);
        let k = Variable::new(Tensor::randn(&[2, 4, 8, 64]), false);
        let (q_rot, k_rot) = rope.apply(&q, &k, 0);
        assert_eq!(q_rot.data().shape(), &[2, 4, 8, 64]);
        assert_eq!(k_rot.data().shape(), &[2, 4, 8, 64]);
    }

    #[test]
    fn test_llama_attention() {
        let config = LLaMAConfig::tiny();
        let attn = LLaMAAttention::new(&config);
        let input = Variable::new(Tensor::randn(&[2, 8, 256]), false);
        let output = attn.forward_with_cache(&input, None, 0);
        assert_eq!(output.data().shape(), &[2, 8, 256]);
    }

    #[test]
    fn test_llama_mlp() {
        let config = LLaMAConfig::tiny();
        let mlp = LLaMAMLP::new(&config);
        let input = Variable::new(Tensor::randn(&[2, 8, 256]), false);
        let output = mlp.forward(&input);
        assert_eq!(output.data().shape(), &[2, 8, 256]);
    }

    #[test]
    fn test_llama_decoder_layer() {
        let config = LLaMAConfig::tiny();
        let layer = LLaMADecoderLayer::new(&config);
        let input = Variable::new(Tensor::randn(&[2, 8, 256]), false);
        let output = layer.forward_with_cache(&input, None, 0);
        assert_eq!(output.data().shape(), &[2, 8, 256]);
    }

    #[test]
    fn test_llama_forward() {
        let config = LLaMAConfig::tiny();
        let model = LLaMA::new(&config);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let output = model.forward_ids(&input_ids);
        assert_eq!(output.data().shape(), &[2, 4, 256]);
    }

    #[test]
    fn test_llama_with_cache() {
        let config = LLaMAConfig::tiny();
        let model = LLaMA::new(&config);
        let mut cache = model.create_kv_cache(2);

        // First forward with prompt
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let (output1, _) = model.forward_with_cache(&prompt, Some(&mut cache));
        assert_eq!(output1.data().shape(), &[2, 2, 256]);
        assert_eq!(cache.seq_len(), 2);

        // Second forward with single token
        let token = Tensor::from_vec(vec![5u32, 6], &[2, 1]).unwrap();
        let (output2, _) = model.forward_with_cache(&token, Some(&mut cache));
        assert_eq!(output2.data().shape(), &[2, 1, 256]);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_llama_causal_lm() {
        let config = LLaMAConfig::tiny();
        let model = LLaMAForCausalLM::new(&config);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let logits = model.forward_ids(&input_ids);
        assert_eq!(logits.data().shape(), &[2, 2, config.vocab_size]);
    }
}
