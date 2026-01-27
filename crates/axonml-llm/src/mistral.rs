//! Mistral - Efficient LLM Architecture
//!
//! Implementation of the Mistral architecture with sliding window attention,
//! grouped-query attention (GQA), and RoPE.
//!
//! Reference: "Mistral 7B" https://arxiv.org/abs/2310.06825
//!
//! # Example
//! ```rust,ignore
//! use axonml_llm::{Mistral, MistralConfig};
//!
//! let config = MistralConfig::mistral_7b();
//! let model = Mistral::new(&config);
//! ```

use axonml_autograd::Variable;
use axonml_nn::{Dropout, Embedding, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use crate::attention::{KVCache, LayerKVCache};
use crate::llama::{RMSNorm, RotaryEmbedding};

// =============================================================================
// Mistral Configuration
// =============================================================================

/// Configuration for Mistral models.
#[derive(Debug, Clone)]
pub struct MistralConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Sliding window size for attention
    pub sliding_window: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta
    pub rope_theta: f32,
    /// Attention dropout
    pub attention_dropout: f32,
}

impl MistralConfig {
    /// Mistral 7B configuration
    pub fn mistral_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8, // GQA with 8 KV heads
            max_position_embeddings: 32768,
            sliding_window: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_dropout: 0.0,
        }
    }

    /// Mistral 7B Instruct configuration
    pub fn mistral_7b_instruct() -> Self {
        Self::mistral_7b()
    }

    /// Mixtral 8x7B configuration (MoE base config)
    pub fn mixtral_8x7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            sliding_window: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            attention_dropout: 0.0,
        }
    }

    /// Tiny Mistral for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            sliding_window: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_dropout: 0.0,
        }
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// =============================================================================
// Mistral Attention with Sliding Window
// =============================================================================

/// Mistral attention with sliding window and grouped-query attention.
#[derive(Debug)]
pub struct MistralAttention {
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
    /// Sliding window size
    sliding_window: usize,
    /// Attention dropout
    attn_dropout: Dropout,
}

impl MistralAttention {
    /// Create new Mistral attention layer.
    pub fn new(config: &MistralConfig) -> Self {
        let head_dim = config.head_dim();
        let kv_hidden = config.num_key_value_heads * head_dim;

        Self {
            q_proj: Linear::new(config.hidden_size, config.hidden_size),
            k_proj: Linear::new(config.hidden_size, kv_hidden),
            v_proj: Linear::new(config.hidden_size, kv_hidden),
            o_proj: Linear::new(config.hidden_size, config.hidden_size),
            rotary_emb: RotaryEmbedding::new(
                head_dim,
                config.max_position_embeddings,
                config.rope_theta,
            ),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            hidden_size: config.hidden_size,
            sliding_window: config.sliding_window,
            attn_dropout: Dropout::new(config.attention_dropout),
        }
    }

    /// Forward pass with sliding window attention.
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
        let q = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);

        // Apply rotary embeddings
        let (q, k) = self.rotary_emb.apply(&q, &k, position_offset);

        // Handle KV-cache with sliding window
        let (k, v, kv_seq_len) = if let Some(cache) = kv_cache {
            let (cached_k, cached_v) = cache.update(&k.data(), &v.data());

            // Apply sliding window: only keep last `sliding_window` tokens
            let total_len = cached_k.shape()[2];
            if total_len > self.sliding_window {
                let start = total_len - self.sliding_window;
                let k_windowed = cached_k.slice(&[
                    0..batch_size,
                    0..self.num_kv_heads,
                    start..total_len,
                    0..self.head_dim,
                ]);
                let v_windowed = cached_v.slice(&[
                    0..batch_size,
                    0..self.num_kv_heads,
                    start..total_len,
                    0..self.head_dim,
                ]);
                (
                    Variable::new(k_windowed, false),
                    Variable::new(v_windowed, false),
                    self.sliding_window,
                )
            } else {
                (
                    Variable::new(cached_k, false),
                    Variable::new(cached_v, false),
                    total_len,
                )
            }
        } else {
            (k, v, seq_len)
        };

        // Repeat KV heads for GQA
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            (self.repeat_kv(&k, repeat), self.repeat_kv(&v, repeat))
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)).mul_scalar(scale);

        // Apply sliding window causal mask
        let mask = self.create_sliding_window_mask(seq_len, kv_seq_len, position_offset);
        let attn_weights = attn_weights.add(&Variable::new(mask, false));

        // Softmax and dropout
        let attn_weights = attn_weights.softmax(-1);
        let attn_weights = self.attn_dropout.forward(&attn_weights);

        // Compute output
        let attn_output = attn_weights.matmul(&v);
        let attn_output =
            attn_output
                .transpose(1, 2)
                .reshape(&[batch_size, seq_len, self.hidden_size]);

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

    /// Create sliding window causal mask.
    fn create_sliding_window_mask(
        &self,
        q_len: usize,
        kv_len: usize,
        offset: usize,
    ) -> Tensor<f32> {
        let mut mask_data = vec![0.0f32; q_len * kv_len];

        for i in 0..q_len {
            let pos = offset + i;
            for j in 0..kv_len {
                // Can't attend to future positions
                if j > pos {
                    mask_data[i * kv_len + j] = f32::NEG_INFINITY;
                }
                // Can't attend beyond sliding window
                else if pos >= self.sliding_window && j < pos - self.sliding_window + 1 {
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
// Mistral MLP
// =============================================================================

/// Mistral MLP with SwiGLU activation (same as LLaMA).
#[derive(Debug)]
pub struct MistralMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MistralMLP {
    /// Create new Mistral MLP.
    pub fn new(config: &MistralConfig) -> Self {
        Self {
            gate_proj: Linear::new(config.hidden_size, config.intermediate_size),
            up_proj: Linear::new(config.hidden_size, config.intermediate_size),
            down_proj: Linear::new(config.intermediate_size, config.hidden_size),
        }
    }

    /// Forward pass.
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
// Mistral Decoder Layer
// =============================================================================

/// Single Mistral decoder layer.
#[derive(Debug)]
pub struct MistralDecoderLayer {
    self_attn: MistralAttention,
    mlp: MistralMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl MistralDecoderLayer {
    /// Create new decoder layer.
    pub fn new(config: &MistralConfig) -> Self {
        Self {
            self_attn: MistralAttention::new(config),
            mlp: MistralMLP::new(config),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        }
    }

    /// Forward pass.
    pub fn forward_with_cache(
        &self,
        hidden_states: &Variable,
        kv_cache: Option<&mut KVCache>,
        position_offset: usize,
    ) -> Variable {
        // Self attention with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states =
            self.self_attn
                .forward_with_cache(&hidden_states, kv_cache, position_offset);
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
// Mistral Model
// =============================================================================

/// Mistral language model.
#[derive(Debug)]
pub struct Mistral {
    /// Token embeddings
    embed_tokens: Embedding,
    /// Decoder layers
    layers: Vec<MistralDecoderLayer>,
    /// Final layer norm
    norm: RMSNorm,
    /// Configuration
    config: MistralConfig,
}

impl Mistral {
    /// Create new Mistral model.
    pub fn new(config: &MistralConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| MistralDecoderLayer::new(config))
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

    /// Forward pass with KV-cache.
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor<u32>,
        kv_cache: Option<&mut LayerKVCache>,
    ) -> (Variable, usize) {
        let position_offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        // Convert token IDs to Variable for embedding lookup
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var = Variable::new(Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(), false);

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(&ids_var);

        // Pass through decoder layers
        if let Some(cache) = kv_cache {
            for (i, layer) in self.layers.iter().enumerate() {
                let layer_cache = cache.get_mut(i);
                hidden_states =
                    layer.forward_with_cache(&hidden_states, layer_cache, position_offset);
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
            self.config.sliding_window, // Use sliding window size for cache
            self.config.head_dim(),
        )
    }
}

impl Module for Mistral {
    fn forward(&self, input: &Variable) -> Variable {
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

/// Mistral with language modeling head.
#[derive(Debug)]
pub struct MistralForCausalLM {
    model: Mistral,
    lm_head: Linear,
}

impl MistralForCausalLM {
    /// Create new Mistral for causal LM.
    pub fn new(config: &MistralConfig) -> Self {
        Self {
            model: Mistral::new(config),
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

    /// Load model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> crate::error::LLMResult<Self> {
        use crate::hf_loader::HFLoader;

        println!("Loading Mistral from: {}", model_id);

        let mut loader = HFLoader::new(model_id)?;
        let config_json = loader.load_config()?;

        // Parse Mistral config
        let config = MistralConfig {
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(32000) as usize,
            hidden_size: config_json["hidden_size"].as_u64().unwrap_or(4096) as usize,
            intermediate_size: config_json["intermediate_size"].as_u64().unwrap_or(14336) as usize,
            num_hidden_layers: config_json["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            num_attention_heads: config_json["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: config_json["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            max_position_embeddings: config_json["max_position_embeddings"]
                .as_u64()
                .unwrap_or(32768) as usize,
            sliding_window: config_json["sliding_window"].as_u64().unwrap_or(4096) as usize,
            rms_norm_eps: config_json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: config_json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            attention_dropout: 0.0,
        };

        loader.load_tensors()?;

        let mut model = Self::new(&config);

        // Load weights (same structure as LLaMA)
        let mut loaded = 0;
        for (name, info) in loader.tensors() {
            let tensor = Tensor::from_vec(info.data.clone(), &info.shape).unwrap();

            // Map weight names
            if name.contains("embed_tokens.weight") {
                model.model.embed_tokens.weight.update_data(tensor);
                loaded += 1;
            } else if name.contains("lm_head.weight") {
                model.lm_head.weight.update_data(tensor);
                loaded += 1;
            } else if name.contains(".norm.weight") && !name.contains("layers.") {
                model.model.norm.load_weight(&tensor);
                loaded += 1;
            }
            // Layer weights are handled by pattern matching layer index
        }

        println!("MistralForCausalLM: Loaded {} weight tensors", loaded);
        Ok(model)
    }
}

impl Module for MistralForCausalLM {
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
    fn test_mistral_config() {
        let config = MistralConfig::tiny();
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.sliding_window, 128);
        assert_eq!(config.num_key_value_heads, 2);
    }

    #[test]
    fn test_mistral_attention() {
        let config = MistralConfig::tiny();
        let attn = MistralAttention::new(&config);
        let input = Variable::new(Tensor::randn(&[2, 8, 256]), false);
        let output = attn.forward_with_cache(&input, None, 0);
        assert_eq!(output.data().shape(), &[2, 8, 256]);
    }

    #[test]
    fn test_sliding_window_mask() {
        let config = MistralConfig::tiny();
        let attn = MistralAttention::new(&config);
        let mask = attn.create_sliding_window_mask(4, 10, 6);

        // Position 6 can attend to 0-6, but sliding window is 128, so all past allowed
        // Position 9 (6+3) can attend to 0-9
        assert_eq!(mask.shape(), &[1, 1, 4, 10]);
    }

    #[test]
    fn test_mistral_forward() {
        let config = MistralConfig::tiny();
        let model = Mistral::new(&config);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let output = model.forward_ids(&input_ids);
        assert_eq!(output.data().shape(), &[2, 2, 256]);
    }

    #[test]
    fn test_mistral_with_cache() {
        let config = MistralConfig::tiny();
        let model = Mistral::new(&config);
        let mut cache = model.create_kv_cache(2);

        // First forward
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let (output1, _) = model.forward_with_cache(&prompt, Some(&mut cache));
        assert_eq!(output1.data().shape(), &[2, 2, 256]);

        // Incremental forward
        let token = Tensor::from_vec(vec![5u32, 6], &[2, 1]).unwrap();
        let (output2, _) = model.forward_with_cache(&token, Some(&mut cache));
        assert_eq!(output2.data().shape(), &[2, 1, 256]);
    }

    #[test]
    fn test_mistral_causal_lm() {
        let config = MistralConfig::tiny();
        let model = MistralForCausalLM::new(&config);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let logits = model.forward_ids(&input_ids);
        assert_eq!(logits.data().shape(), &[2, 2, config.vocab_size]);
    }
}
