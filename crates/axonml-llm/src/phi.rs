//! Phi - Microsoft's Small Language Models
//!
//! Implementation of the Phi architecture (Phi-1, Phi-2, Phi-3).
//! Phi models are efficient small language models optimized for coding and reasoning.
//!
//! Reference: "Textbooks Are All You Need" https://arxiv.org/abs/2306.11644
//!
//! # Example
//! ```rust,ignore
//! use axonml_llm::{Phi, PhiConfig};
//!
//! let config = PhiConfig::phi2();
//! let model = Phi::new(&config);
//! ```

use axonml_autograd::Variable;
use axonml_nn::{Module, Linear, Dropout, Parameter, Embedding};
use axonml_tensor::{Tensor, view::cat};

use crate::attention::{KVCache, LayerKVCache};
use crate::llama::{RotaryEmbedding, RMSNorm};

// =============================================================================
// Phi Configuration
// =============================================================================

/// Configuration for Phi models.
#[derive(Debug, Clone)]
pub struct PhiConfig {
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
    /// Number of key-value heads
    pub num_key_value_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
    /// RoPE theta
    pub rope_theta: f32,
    /// Partial rotary factor (Phi uses partial RoPE)
    pub partial_rotary_factor: f32,
    /// Attention dropout
    pub attention_dropout: f32,
    /// Hidden dropout
    pub hidden_dropout: f32,
    /// Use bias in linear layers
    pub use_bias: bool,
}

impl PhiConfig {
    /// Phi-1 (1.3B) configuration
    pub fn phi1() -> Self {
        Self {
            vocab_size: 51200,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: true,
        }
    }

    /// Phi-2 (2.7B) configuration
    pub fn phi2() -> Self {
        Self {
            vocab_size: 51200,
            hidden_size: 2560,
            intermediate_size: 10240,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.4,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: true,
        }
    }

    /// Phi-3 Mini (3.8B) configuration
    pub fn phi3_mini() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 4096,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0, // Full RoPE
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: false,
        }
    }

    /// Tiny Phi for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 51200,
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: true,
        }
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Rotary dimension (partial RoPE)
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim() as f32 * self.partial_rotary_factor) as usize
    }
}

// =============================================================================
// Phi Attention
// =============================================================================

/// Phi attention with partial rotary embeddings.
#[derive(Debug)]
pub struct PhiAttention {
    /// Query projection
    q_proj: Linear,
    /// Key projection
    k_proj: Linear,
    /// Value projection
    v_proj: Linear,
    /// Output projection
    dense: Linear,
    /// Rotary embedding
    rotary_emb: RotaryEmbedding,
    /// Number of attention heads
    num_heads: usize,
    /// Number of key-value heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Rotary dimension
    rotary_dim: usize,
    /// Hidden size
    hidden_size: usize,
    /// Attention dropout
    attn_dropout: Dropout,
}

impl PhiAttention {
    /// Create new Phi attention layer.
    pub fn new(config: &PhiConfig) -> Self {
        let head_dim = config.head_dim();
        let rotary_dim = config.rotary_dim();
        let kv_hidden = config.num_key_value_heads * head_dim;

        Self {
            q_proj: Linear::new(config.hidden_size, config.hidden_size),
            k_proj: Linear::new(config.hidden_size, kv_hidden),
            v_proj: Linear::new(config.hidden_size, kv_hidden),
            dense: Linear::new(config.hidden_size, config.hidden_size),
            rotary_emb: RotaryEmbedding::new(rotary_dim, config.max_position_embeddings, config.rope_theta),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rotary_dim,
            hidden_size: config.hidden_size,
            attn_dropout: Dropout::new(config.attention_dropout),
        }
    }

    /// Forward pass with partial RoPE and optional KV-cache.
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

        // Apply partial rotary embeddings (only to first rotary_dim dimensions)
        let (q, k) = if self.rotary_dim < self.head_dim {
            self.apply_partial_rotary(&q, &k, position_offset)
        } else {
            self.rotary_emb.apply(&q, &k, position_offset)
        };

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

        // Repeat KV heads if using GQA
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            (self.repeat_kv(&k, repeat), self.repeat_kv(&v, repeat))
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

        self.dense.forward(&attn_output)
    }

    /// Apply partial rotary embeddings.
    fn apply_partial_rotary(
        &self,
        q: &Variable,
        k: &Variable,
        position_offset: usize,
    ) -> (Variable, Variable) {
        let q_data = q.data();
        let k_data = k.data();
        let shape = q_data.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];

        // Split into rotary and pass-through parts
        let q_rot = q_data.slice(&[0..batch_size, 0..num_heads, 0..seq_len, 0..self.rotary_dim]);
        let q_pass = q_data.slice(&[0..batch_size, 0..num_heads, 0..seq_len, self.rotary_dim..self.head_dim]);

        let k_rot = k_data.slice(&[0..batch_size, 0..self.num_kv_heads, 0..seq_len, 0..self.rotary_dim]);
        let k_pass = k_data.slice(&[0..batch_size, 0..self.num_kv_heads, 0..seq_len, self.rotary_dim..self.head_dim]);

        // Apply rotary to the rotary part
        let q_rot_var = Variable::new(q_rot, q.requires_grad());
        let k_rot_var = Variable::new(k_rot, k.requires_grad());
        let (q_rotated, k_rotated) = self.rotary_emb.apply(&q_rot_var, &k_rot_var, position_offset);

        // Concatenate back
        let q_out = cat(&[q_rotated.data(), q_pass], 3).unwrap();
        let k_out = cat(&[k_rotated.data(), k_pass], 3).unwrap();

        (
            Variable::new(q_out, q.requires_grad()),
            Variable::new(k_out, k.requires_grad()),
        )
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
        params.extend(self.dense.parameters());
        params
    }
}

// =============================================================================
// Phi MLP
// =============================================================================

/// Phi MLP (uses GELU activation, not SwiGLU).
#[derive(Debug)]
pub struct PhiMLP {
    fc1: Linear,
    fc2: Linear,
}

impl PhiMLP {
    /// Create new Phi MLP.
    pub fn new(config: &PhiConfig) -> Self {
        Self {
            fc1: Linear::new(config.hidden_size, config.intermediate_size),
            fc2: Linear::new(config.intermediate_size, config.hidden_size),
        }
    }

    /// Forward pass with GELU activation.
    pub fn forward(&self, x: &Variable) -> Variable {
        let hidden = self.fc1.forward(x).gelu();
        self.fc2.forward(&hidden)
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

// =============================================================================
// Phi Decoder Layer
// =============================================================================

/// Single Phi decoder layer with parallel attention and MLP.
#[derive(Debug)]
pub struct PhiDecoderLayer {
    self_attn: PhiAttention,
    mlp: PhiMLP,
    input_layernorm: RMSNorm,
    /// Whether to use parallel attention (Phi-1/2 style)
    parallel_attn: bool,
}

impl PhiDecoderLayer {
    /// Create new decoder layer.
    pub fn new(config: &PhiConfig, parallel_attn: bool) -> Self {
        Self {
            self_attn: PhiAttention::new(config),
            mlp: PhiMLP::new(config),
            input_layernorm: RMSNorm::new(config.hidden_size, config.layer_norm_eps),
            parallel_attn,
        }
    }

    /// Forward pass.
    pub fn forward_with_cache(
        &self,
        hidden_states: &Variable,
        kv_cache: Option<&mut KVCache>,
        position_offset: usize,
    ) -> Variable {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);

        if self.parallel_attn {
            // Parallel: attention and MLP computed on same input, then summed
            let attn_output = self.self_attn.forward_with_cache(&hidden_states, kv_cache, position_offset);
            let mlp_output = self.mlp.forward(&hidden_states);
            residual.add(&attn_output).add(&mlp_output)
        } else {
            // Sequential: standard transformer block
            let attn_output = self.self_attn.forward_with_cache(&hidden_states, kv_cache, position_offset);
            let hidden_states = residual.add(&attn_output);
            let residual = hidden_states.clone();
            let hidden_states = self.input_layernorm.forward(&hidden_states);
            let mlp_output = self.mlp.forward(&hidden_states);
            residual.add(&mlp_output)
        }
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.mlp.parameters());
        params.extend(self.input_layernorm.parameters());
        params
    }
}

// =============================================================================
// Phi Model
// =============================================================================

/// Phi language model.
#[derive(Debug)]
pub struct Phi {
    /// Token embeddings
    embed_tokens: Embedding,
    /// Decoder layers
    layers: Vec<PhiDecoderLayer>,
    /// Final layer norm
    final_layernorm: RMSNorm,
    /// Configuration
    config: PhiConfig,
}

impl Phi {
    /// Create new Phi model.
    pub fn new(config: &PhiConfig) -> Self {
        Self::with_parallel_attn(config, true) // Phi-1/2 use parallel attention
    }

    /// Create Phi with configurable parallel attention.
    pub fn with_parallel_attn(config: &PhiConfig, parallel_attn: bool) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| PhiDecoderLayer::new(config, parallel_attn))
            .collect();

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers,
            final_layernorm: RMSNorm::new(config.hidden_size, config.layer_norm_eps),
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
        let hidden_states = self.final_layernorm.forward(&hidden_states);

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

impl Module for Phi {
    fn forward(&self, input: &Variable) -> Variable {
        let mut hidden_states = input.clone();
        for layer in &self.layers {
            hidden_states = layer.forward_with_cache(&hidden_states, None, 0);
        }
        self.final_layernorm.forward(&hidden_states)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.embed_tokens.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.final_layernorm.parameters());
        params
    }
}

/// Phi with language modeling head.
#[derive(Debug)]
pub struct PhiForCausalLM {
    model: Phi,
    lm_head: Linear,
}

impl PhiForCausalLM {
    /// Create new Phi for causal LM.
    pub fn new(config: &PhiConfig) -> Self {
        Self {
            model: Phi::new(config),
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

        println!("Loading Phi from: {}", model_id);

        let mut loader = HFLoader::new(model_id)?;
        let config_json = loader.load_config()?;

        // Parse Phi config
        let config = PhiConfig {
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(51200) as usize,
            hidden_size: config_json["hidden_size"].as_u64().unwrap_or(2560) as usize,
            intermediate_size: config_json["intermediate_size"].as_u64().unwrap_or(10240) as usize,
            num_hidden_layers: config_json["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            num_attention_heads: config_json["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: config_json["num_key_value_heads"]
                .as_u64()
                .unwrap_or(config_json["num_attention_heads"].as_u64().unwrap_or(32)) as usize,
            max_position_embeddings: config_json["max_position_embeddings"].as_u64().unwrap_or(2048) as usize,
            layer_norm_eps: config_json["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: config_json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            partial_rotary_factor: config_json["partial_rotary_factor"].as_f64().unwrap_or(0.5) as f32,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: config_json["use_bias"].as_bool().unwrap_or(true),
        };

        loader.load_tensors()?;

        let mut model = Self::new(&config);

        // Load weights
        let mut loaded = 0;
        for (name, info) in loader.tensors() {
            let tensor = Tensor::from_vec(info.data.clone(), &info.shape).unwrap();

            if name.contains("embed_tokens.weight") {
                model.model.embed_tokens.weight.update_data(tensor);
                loaded += 1;
            } else if name.contains("lm_head.weight") {
                model.lm_head.weight.update_data(tensor);
                loaded += 1;
            } else if name.contains("final_layernorm.weight") {
                model.model.final_layernorm.load_weight(&tensor);
                loaded += 1;
            }
        }

        println!("PhiForCausalLM: Loaded {} weight tensors", loaded);
        Ok(model)
    }
}

impl Module for PhiForCausalLM {
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
    fn test_phi_config() {
        let config = PhiConfig::tiny();
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.rotary_dim(), 32); // 50% partial rotary
    }

    #[test]
    fn test_phi_attention() {
        let config = PhiConfig::tiny();
        let attn = PhiAttention::new(&config);
        let input = Variable::new(Tensor::randn(&[2, 8, 256]), false);
        let output = attn.forward_with_cache(&input, None, 0);
        assert_eq!(output.data().shape(), &[2, 8, 256]);
    }

    #[test]
    fn test_phi_mlp() {
        let config = PhiConfig::tiny();
        let mlp = PhiMLP::new(&config);
        let input = Variable::new(Tensor::randn(&[2, 8, 256]), false);
        let output = mlp.forward(&input);
        assert_eq!(output.data().shape(), &[2, 8, 256]);
    }

    #[test]
    fn test_phi_forward() {
        let config = PhiConfig::tiny();
        let model = Phi::new(&config);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let output = model.forward_ids(&input_ids);
        assert_eq!(output.data().shape(), &[2, 2, 256]);
    }

    #[test]
    fn test_phi_with_cache() {
        let config = PhiConfig::tiny();
        let model = Phi::new(&config);
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
    fn test_phi_causal_lm() {
        let config = PhiConfig::tiny();
        let model = PhiForCausalLM::new(&config);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let logits = model.forward_ids(&input_ids);
        assert_eq!(logits.data().shape(), &[2, 2, config.vocab_size]);
    }

    #[test]
    fn test_phi_parallel_vs_sequential() {
        let config = PhiConfig::tiny();

        // Parallel attention (default)
        let model_parallel = Phi::with_parallel_attn(&config, true);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let output_parallel = model_parallel.forward_ids(&input_ids);
        assert_eq!(output_parallel.data().shape(), &[2, 2, 256]);

        // Sequential attention
        let model_sequential = Phi::with_parallel_attn(&config, false);
        let output_sequential = model_sequential.forward_ids(&input_ids);
        assert_eq!(output_sequential.data().shape(), &[2, 2, 256]);
    }
}
