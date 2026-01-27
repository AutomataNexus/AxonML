//! Attention Mechanisms Module
//!
//! Implements multi-head self-attention and causal (masked) self-attention
//! for transformer models with KV-cache support for efficient inference.

use axonml_autograd::Variable;
use axonml_nn::{Dropout, Linear, Module, Parameter};
use axonml_tensor::{view::cat, Tensor};

// =============================================================================
// KV Cache
// =============================================================================

/// Key-Value cache for efficient autoregressive generation.
///
/// Stores the key and value tensors from previous forward passes to avoid
/// recomputation during incremental decoding.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached key tensor: [batch, num_heads, seq_len, head_dim]
    pub key: Tensor<f32>,
    /// Cached value tensor: [batch, num_heads, seq_len, head_dim]
    pub value: Tensor<f32>,
    /// Current sequence length in cache
    pub seq_len: usize,
}

impl KVCache {
    /// Create a new empty KV cache.
    pub fn new(batch_size: usize, num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        Self {
            key: Tensor::zeros(&[batch_size, num_heads, max_seq_len, head_dim]),
            value: Tensor::zeros(&[batch_size, num_heads, max_seq_len, head_dim]),
            seq_len: 0,
        }
    }

    /// Update cache with new key/value tensors.
    ///
    /// # Arguments
    /// * `new_key` - New key tensor [batch, num_heads, new_seq_len, head_dim]
    /// * `new_value` - New value tensor [batch, num_heads, new_seq_len, head_dim]
    ///
    /// # Returns
    /// Updated key and value tensors including cached values
    pub fn update(
        &mut self,
        new_key: &Tensor<f32>,
        new_value: &Tensor<f32>,
    ) -> (Tensor<f32>, Tensor<f32>) {
        let new_seq_len = new_key.shape()[2];

        if self.seq_len == 0 {
            // First token(s), just store and return
            self.key = new_key.clone();
            self.value = new_value.clone();
            self.seq_len = new_seq_len;
            return (new_key.clone(), new_value.clone());
        }

        // Concatenate cached and new along sequence dimension
        let key = cat(&[self.key.clone(), new_key.clone()], 2).unwrap();
        let value = cat(&[self.value.clone(), new_value.clone()], 2).unwrap();

        self.key = key.clone();
        self.value = value.clone();
        self.seq_len += new_seq_len;

        (key, value)
    }

    /// Get current sequence length in cache.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}

/// Layer-wise KV cache for full model.
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Cache for each layer
    pub layers: Vec<KVCache>,
}

impl LayerKVCache {
    /// Create cache for all layers.
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| KVCache::new(batch_size, num_heads, max_seq_len, head_dim))
            .collect();
        Self { layers }
    }

    /// Get mutable reference to layer cache.
    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut KVCache> {
        self.layers.get_mut(layer_idx)
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for cache in &mut self.layers {
            cache.clear();
        }
    }

    /// Get current sequence length (same for all layers).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|c| c.seq_len).unwrap_or(0)
    }
}

/// Multi-head self-attention layer.
#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    /// Query projection
    pub query: Linear,
    /// Key projection
    pub key: Linear,
    /// Value projection
    pub value: Linear,
    /// Output projection
    pub out_proj: Linear,
    /// Attention dropout
    pub attn_dropout: Dropout,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Scale factor for attention scores
    pub scale: f32,
}

impl MultiHeadSelfAttention {
    /// Creates a new multi-head self-attention layer.
    pub fn new(hidden_size: usize, num_heads: usize, dropout: f32) -> Self {
        assert!(
            hidden_size % num_heads == 0,
            "hidden_size must be divisible by num_heads"
        );

        let head_dim = hidden_size / num_heads;

        Self {
            query: Linear::new(hidden_size, hidden_size),
            key: Linear::new(hidden_size, hidden_size),
            value: Linear::new(hidden_size, hidden_size),
            out_proj: Linear::new(hidden_size, hidden_size),
            attn_dropout: Dropout::new(dropout),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Splits tensor into multiple heads.
    fn split_heads(&self, x: &Variable, batch_size: usize, seq_len: usize) -> Variable {
        // x: [batch, seq_len, hidden_size]
        // Output: [batch, num_heads, seq_len, head_dim]
        let reshaped = x.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);
        reshaped.transpose(1, 2)
    }

    /// Merges multiple heads back into single tensor.
    fn merge_heads(&self, x: &Variable, batch_size: usize, seq_len: usize) -> Variable {
        // x: [batch, num_heads, seq_len, head_dim]
        // Output: [batch, seq_len, hidden_size]
        let transposed = x.transpose(1, 2);
        transposed.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])
    }

    /// Forward pass with optional attention mask.
    pub fn forward_with_mask(
        &self,
        hidden_states: &Variable,
        attention_mask: Option<&Tensor<f32>>,
    ) -> Variable {
        let data = hidden_states.data();
        let shape = data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V
        let q = self.query.forward(hidden_states);
        let k = self.key.forward(hidden_states);
        let v = self.value.forward(hidden_states);

        // Split into heads
        let q = self.split_heads(&q, batch_size, seq_len);
        let k = self.split_heads(&k, batch_size, seq_len);
        let v = self.split_heads(&v, batch_size, seq_len);

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let k_t = k.transpose(2, 3);
        let mut attn_scores = q.matmul(&k_t);
        attn_scores = attn_scores.mul_scalar(self.scale);

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            let mask_var = Variable::new(mask.clone(), false);
            // Mask is typically 0 for positions to attend, -inf for positions to mask
            attn_scores = attn_scores.add(&mask_var);
        }

        // Softmax over last dimension
        let attn_probs = attn_scores.softmax(-1);

        // Apply dropout
        let attn_probs = self.attn_dropout.forward(&attn_probs);

        // Attention output: attn_probs @ V
        let context = attn_probs.matmul(&v);

        // Merge heads
        let context = self.merge_heads(&context, batch_size, seq_len);

        // Output projection
        self.out_proj.forward(&context)
    }
}

impl Module for MultiHeadSelfAttention {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.query.parameters());
        params.extend(self.key.parameters());
        params.extend(self.value.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn train(&mut self) {
        self.attn_dropout.train();
    }

    fn eval(&mut self) {
        self.attn_dropout.eval();
    }
}

/// Causal (masked) self-attention for autoregressive models like GPT.
#[derive(Debug)]
pub struct CausalSelfAttention {
    /// Combined Q, K, V projection for efficiency
    pub c_attn: Linear,
    /// Output projection
    pub c_proj: Linear,
    /// Attention dropout
    pub attn_dropout: Dropout,
    /// Residual dropout
    pub resid_dropout: Dropout,
    /// Number of attention heads
    pub num_heads: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl CausalSelfAttention {
    /// Creates a new causal self-attention layer.
    pub fn new(n_embd: usize, num_heads: usize, max_seq_len: usize, dropout: f32) -> Self {
        assert!(
            n_embd % num_heads == 0,
            "n_embd must be divisible by num_heads"
        );

        let head_dim = n_embd / num_heads;

        Self {
            c_attn: Linear::new(n_embd, 3 * n_embd), // Q, K, V combined
            c_proj: Linear::new(n_embd, n_embd),
            attn_dropout: Dropout::new(dropout),
            resid_dropout: Dropout::new(dropout),
            num_heads,
            n_embd,
            head_dim,
            max_seq_len,
        }
    }

    /// Creates the causal mask for autoregressive attention.
    pub fn create_causal_mask(&self, seq_len: usize) -> Tensor<f32> {
        // Lower triangular mask: 1s for positions we can attend to, 0s otherwise
        let mut mask_data = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // Can't attend to future positions
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len]).unwrap()
    }

    /// Forward pass with causal masking.
    pub fn forward_causal(&self, x: &Variable) -> Variable {
        self.forward_with_cache(x, None).0
    }

    /// Forward pass with KV-cache support for efficient generation.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    /// * `kv_cache` - Optional mutable reference to KV cache
    ///
    /// # Returns
    /// Tuple of (output, updated_cache) where cache contains new key/value pairs
    pub fn forward_with_cache(
        &self,
        x: &Variable,
        kv_cache: Option<&mut KVCache>,
    ) -> (Variable, Option<(Tensor<f32>, Tensor<f32>)>) {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Combined Q, K, V projection
        let qkv = self.c_attn.forward(x);

        // Split into Q, K, V
        let qkv_data = qkv.data();

        // Split along last dimension: [batch, seq, 3*n_embd] -> 3x [batch, seq, n_embd]
        let q_data = qkv_data.slice(&[0..batch_size, 0..seq_len, 0..self.n_embd]);
        let k_data = qkv_data.slice(&[0..batch_size, 0..seq_len, self.n_embd..2 * self.n_embd]);
        let v_data = qkv_data.slice(&[0..batch_size, 0..seq_len, 2 * self.n_embd..3 * self.n_embd]);

        let q = Variable::new(q_data, qkv.requires_grad());
        let k_new = Variable::new(k_data.clone(), qkv.requires_grad());
        let v_new = Variable::new(v_data.clone(), qkv.requires_grad());

        // Reshape for multi-head attention
        // [batch, seq, n_embd] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        let q = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k_new = k_new
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v_new = v_new
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // Apply KV cache if provided
        let (k, v, total_seq_len, new_cache) = if let Some(cache) = kv_cache {
            let (cached_k, cached_v) = cache.update(&k_new.data(), &v_new.data());
            let total_len = cached_k.shape()[2];
            (
                Variable::new(cached_k.clone(), false),
                Variable::new(cached_v.clone(), false),
                total_len,
                Some((cached_k, cached_v)),
            )
        } else {
            (
                k_new.clone(),
                v_new.clone(),
                seq_len,
                Some((k_new.data(), v_new.data())),
            )
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)).mul_scalar(scale);

        // Apply causal mask (only mask future positions relative to query position)
        let causal_mask = self.create_causal_mask_for_cache(seq_len, total_seq_len);
        let mask_var = Variable::new(causal_mask, false);
        let attn = attn.add(&mask_var);

        // Softmax
        let attn = attn.softmax(-1);

        // Apply attention dropout
        let attn = self.attn_dropout.forward(&attn);

        // Compute output
        let output = attn.matmul(&v);

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, n_embd]
        let output = output
            .transpose(1, 2)
            .reshape(&[batch_size, seq_len, self.n_embd]);

        // Output projection and dropout
        let output = self.c_proj.forward(&output);
        let output = self.resid_dropout.forward(&output);

        (output, new_cache)
    }

    /// Creates causal mask for cached attention.
    ///
    /// When using KV-cache, query has length `q_len` but key/value have length `kv_len`.
    /// We need to mask so position i in query can only attend to positions 0..=(cache_len + i).
    fn create_causal_mask_for_cache(&self, q_len: usize, kv_len: usize) -> Tensor<f32> {
        let mut mask_data = vec![0.0f32; q_len * kv_len];
        let start_pos = kv_len - q_len; // Position offset for new tokens

        for i in 0..q_len {
            let query_pos = start_pos + i;
            for j in 0..kv_len {
                if j > query_pos {
                    // Can't attend to future positions
                    mask_data[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        Tensor::from_vec(mask_data, &[1, 1, q_len, kv_len]).unwrap()
    }
}

impl Module for CausalSelfAttention {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_causal(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.c_attn.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn train(&mut self) {
        self.attn_dropout.train();
        self.resid_dropout.train();
    }

    fn eval(&mut self) {
        self.attn_dropout.eval();
        self.resid_dropout.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multihead_attention_shape() {
        let attn = MultiHeadSelfAttention::new(64, 4, 0.0);

        let input_data = Tensor::randn(&[2, 8, 64]); // [batch, seq, hidden]
        let input = Variable::new(input_data, false);

        let output = attn.forward(&input);
        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_causal_attention_shape() {
        let attn = CausalSelfAttention::new(64, 4, 128, 0.0);

        let input_data = Tensor::randn(&[2, 8, 64]); // [batch, seq, hidden]
        let input = Variable::new(input_data, false);

        let output = attn.forward(&input);
        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_causal_mask() {
        let attn = CausalSelfAttention::new(64, 4, 128, 0.0);
        let mask = attn.create_causal_mask(4);

        // Check shape
        assert_eq!(mask.shape(), &[1, 1, 4, 4]);

        // Check that future positions are masked
        let data = mask.to_vec();
        assert!(data[0 * 4 + 1].is_infinite()); // position 0 can't attend to 1
        assert!(data[0 * 4 + 2].is_infinite()); // position 0 can't attend to 2
        assert!(data[0 * 4 + 3].is_infinite()); // position 0 can't attend to 3
        assert!(data[1 * 4 + 2].is_infinite()); // position 1 can't attend to 2
        assert!(data[1 * 4 + 3].is_infinite()); // position 1 can't attend to 3

        // Check that current and past positions are not masked
        assert_eq!(data[0 * 4 + 0], 0.0); // position 0 can attend to 0
        assert_eq!(data[1 * 4 + 0], 0.0); // position 1 can attend to 0
        assert_eq!(data[1 * 4 + 1], 0.0); // position 1 can attend to 1
        assert_eq!(data[3 * 4 + 0], 0.0); // position 3 can attend to 0
    }

    #[test]
    fn test_attention_parameters() {
        let attn = MultiHeadSelfAttention::new(64, 4, 0.0);
        let params = attn.parameters();

        // Should have 4 sets of weight/bias for Q, K, V, and output projection
        assert_eq!(params.len(), 8); // 4 weights + 4 biases
    }

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(2, 4, 128, 16);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_update() {
        let mut cache = KVCache::new(2, 4, 128, 16);

        // First update
        let k1 = Tensor::randn(&[2, 4, 5, 16]);
        let v1 = Tensor::randn(&[2, 4, 5, 16]);
        let (k_out, v_out) = cache.update(&k1, &v1);

        assert_eq!(cache.len(), 5);
        assert_eq!(k_out.shape(), &[2, 4, 5, 16]);
        assert_eq!(v_out.shape(), &[2, 4, 5, 16]);

        // Second update (incremental)
        let k2 = Tensor::randn(&[2, 4, 1, 16]);
        let v2 = Tensor::randn(&[2, 4, 1, 16]);
        let (k_out, v_out) = cache.update(&k2, &v2);

        assert_eq!(cache.len(), 6);
        assert_eq!(k_out.shape(), &[2, 4, 6, 16]);
        assert_eq!(v_out.shape(), &[2, 4, 6, 16]);
    }

    #[test]
    fn test_layer_kv_cache() {
        let mut cache = LayerKVCache::new(12, 2, 4, 128, 16);
        assert_eq!(cache.layers.len(), 12);
        assert_eq!(cache.seq_len(), 0);

        // Update first layer
        let k = Tensor::randn(&[2, 4, 3, 16]);
        let v = Tensor::randn(&[2, 4, 3, 16]);
        cache.get_mut(0).unwrap().update(&k, &v);

        assert_eq!(cache.layers[0].len(), 3);

        // Clear all
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_causal_attention_with_cache() {
        let attn = CausalSelfAttention::new(64, 4, 128, 0.0);
        let mut cache = KVCache::new(2, 4, 128, 16);

        // First forward with prompt
        let prompt = Tensor::randn(&[2, 5, 64]);
        let prompt_var = Variable::new(prompt, false);
        let (output1, _) = attn.forward_with_cache(&prompt_var, Some(&mut cache));

        assert_eq!(output1.data().shape(), &[2, 5, 64]);
        assert_eq!(cache.len(), 5);

        // Incremental forward with single token
        let token = Tensor::randn(&[2, 1, 64]);
        let token_var = Variable::new(token, false);
        let (output2, _) = attn.forward_with_cache(&token_var, Some(&mut cache));

        assert_eq!(output2.data().shape(), &[2, 1, 64]);
        assert_eq!(cache.len(), 6);
    }

    #[test]
    fn test_causal_mask_for_cache() {
        let attn = CausalSelfAttention::new(64, 4, 128, 0.0);

        // Query length 1, KV length 5 (4 cached + 1 new)
        let mask = attn.create_causal_mask_for_cache(1, 5);
        assert_eq!(mask.shape(), &[1, 1, 1, 5]);

        // The single query at position 4 can attend to all 5 positions
        let data = mask.to_vec();
        assert_eq!(data[0], 0.0); // can attend to 0
        assert_eq!(data[1], 0.0); // can attend to 1
        assert_eq!(data[2], 0.0); // can attend to 2
        assert_eq!(data[3], 0.0); // can attend to 3
        assert_eq!(data[4], 0.0); // can attend to 4 (self)
    }

    #[test]
    fn test_flash_attention_basic() {
        let flash = FlashAttention::new(4, 0.0, true);

        let q = Tensor::randn(&[2, 4, 8, 16]); // [batch, heads, seq, head_dim]
        let k = Tensor::randn(&[2, 4, 8, 16]);
        let v = Tensor::randn(&[2, 4, 8, 16]);

        let output = flash.forward_qkv(&q, &k, &v);
        assert_eq!(output.shape(), &[2, 4, 8, 16]);
    }

    #[test]
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.block_size_kv, 64);
        assert!(config.causal);
    }
}

// =============================================================================
// Flash Attention
// =============================================================================

/// Configuration for Flash Attention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for query chunking
    pub block_size_q: usize,
    /// Block size for key/value chunking
    pub block_size_kv: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Softmax scale (typically 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
    /// Dropout probability
    pub dropout_p: f32,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            causal: true,
            softmax_scale: None,
            dropout_p: 0.0,
        }
    }
}

/// Flash Attention - Memory-efficient attention mechanism.
///
/// Implements the Flash Attention algorithm that computes attention in tiles/blocks
/// to reduce memory usage from O(N²) to O(N) while maintaining numerical precision.
///
/// Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
/// https://arxiv.org/abs/2205.14135
///
/// # Example
/// ```rust,ignore
/// use axonml_llm::attention::FlashAttention;
///
/// let flash_attn = FlashAttention::new(8, 0.0, true);
/// let output = flash_attn.forward_qkv(&query, &key, &value);
/// ```
#[derive(Debug)]
pub struct FlashAttention {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout probability
    pub dropout_p: f32,
    /// Whether to use causal masking
    pub causal: bool,
    /// Block size for tiling
    pub block_size: usize,
}

impl FlashAttention {
    /// Creates a new Flash Attention module.
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    /// * `dropout_p` - Dropout probability (0.0 for inference)
    /// * `causal` - Whether to apply causal masking
    pub fn new(num_heads: usize, dropout_p: f32, causal: bool) -> Self {
        Self {
            num_heads,
            dropout_p,
            causal,
            block_size: 64, // Default block size
        }
    }

    /// Creates Flash Attention with custom block size.
    pub fn with_block_size(
        num_heads: usize,
        dropout_p: f32,
        causal: bool,
        block_size: usize,
    ) -> Self {
        Self {
            num_heads,
            dropout_p,
            causal,
            block_size,
        }
    }

    /// Forward pass with pre-computed Q, K, V tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, num_heads, seq_len, head_dim]
    /// * `v` - Value tensor [batch, num_heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Output tensor [batch, num_heads, seq_len, head_dim]
    pub fn forward_qkv(&self, q: &Tensor<f32>, k: &Tensor<f32>, v: &Tensor<f32>) -> Tensor<f32> {
        let shape = q.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        let scale = 1.0 / (head_dim as f32).sqrt();

        // For smaller sequences, use standard attention
        if seq_len <= self.block_size * 2 {
            return self.standard_attention(q, k, v, scale);
        }

        // Flash Attention with tiling
        self.tiled_attention(q, k, v, batch_size, num_heads, seq_len, head_dim, scale)
    }

    /// Standard attention implementation for small sequences.
    fn standard_attention(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        scale: f32,
    ) -> Tensor<f32> {
        let shape = q.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        // Compute attention scores: Q @ K^T * scale
        let q_data = q.to_vec();
        let k_data = k.to_vec();
        let v_data = v.to_vec();

        let mut output = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];

        for b in 0..batch_size {
            for h in 0..num_heads {
                // Compute attention for this batch/head
                let mut attn_scores = vec![0.0f32; seq_len * seq_len];

                // Q @ K^T
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        // Apply causal mask
                        if self.causal && j > i {
                            attn_scores[i * seq_len + j] = f32::NEG_INFINITY;
                            continue;
                        }

                        let mut score = 0.0;
                        for d in 0..head_dim {
                            let q_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                            let k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                            score += q_data[q_idx] * k_data[k_idx];
                        }
                        attn_scores[i * seq_len + j] = score * scale;
                    }
                }

                // Softmax over each row
                for i in 0..seq_len {
                    let row_start = i * seq_len;
                    let row_end = row_start + seq_len;
                    let row = &mut attn_scores[row_start..row_end];

                    // Find max for numerical stability
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    // Exp and sum
                    let mut sum = 0.0;
                    for val in row.iter_mut() {
                        *val = (*val - max_val).exp();
                        sum += *val;
                    }

                    // Normalize
                    for val in row.iter_mut() {
                        *val /= sum;
                    }
                }

                // Attn @ V
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        let mut val = 0.0;
                        for j in 0..seq_len {
                            let attn_val = attn_scores[i * seq_len + j];
                            let v_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                            val += attn_val * v_data[v_idx];
                        }
                        let out_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                        output[out_idx] = val;
                    }
                }
            }
        }

        Tensor::from_vec(output, &[batch_size, num_heads, seq_len, head_dim]).unwrap()
    }

    /// Tiled attention implementation for memory efficiency.
    ///
    /// Uses the Flash Attention algorithm to compute attention in blocks,
    /// reducing peak memory usage from O(N²) to O(N).
    fn tiled_attention(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Tensor<f32> {
        let q_data = q.to_vec();
        let k_data = k.to_vec();
        let v_data = v.to_vec();

        let mut output = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];
        let block_size = self.block_size;
        let num_blocks = (seq_len + block_size - 1) / block_size;

        for b in 0..batch_size {
            for h in 0..num_heads {
                // Running statistics for online softmax
                let mut row_max = vec![f32::NEG_INFINITY; seq_len];
                let mut row_sum = vec![0.0f32; seq_len];
                let mut row_out = vec![vec![0.0f32; head_dim]; seq_len];

                // Process key-value blocks
                for kv_block in 0..num_blocks {
                    let kv_start = kv_block * block_size;
                    let kv_end = (kv_start + block_size).min(seq_len);

                    // Process query blocks
                    for q_block in 0..num_blocks {
                        let q_start = q_block * block_size;
                        let q_end = (q_start + block_size).min(seq_len);

                        // Skip if causal and this block is fully masked
                        if self.causal && kv_start > q_end - 1 {
                            continue;
                        }

                        // Compute block attention scores
                        for i in q_start..q_end {
                            let mut block_scores = Vec::with_capacity(kv_end - kv_start);
                            let mut block_max = f32::NEG_INFINITY;

                            for j in kv_start..kv_end {
                                // Apply causal mask
                                if self.causal && j > i {
                                    block_scores.push(f32::NEG_INFINITY);
                                    continue;
                                }

                                let mut score = 0.0;
                                for d in 0..head_dim {
                                    let q_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                                    let k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                                    score += q_data[q_idx] * k_data[k_idx];
                                }
                                score *= scale;
                                block_max = block_max.max(score);
                                block_scores.push(score);
                            }

                            // Online softmax update
                            let prev_max = row_max[i];
                            let new_max = prev_max.max(block_max);

                            // Rescale previous sum and output
                            let scale_prev = (prev_max - new_max).exp();
                            row_sum[i] *= scale_prev;
                            for d in 0..head_dim {
                                row_out[i][d] *= scale_prev;
                            }

                            // Add new block contribution
                            for (local_j, j) in (kv_start..kv_end).enumerate() {
                                let score = block_scores[local_j];
                                if score.is_finite() {
                                    let p = (score - new_max).exp();
                                    row_sum[i] += p;

                                    for d in 0..head_dim {
                                        let v_idx =
                                            ((b * num_heads + h) * seq_len + j) * head_dim + d;
                                        row_out[i][d] += p * v_data[v_idx];
                                    }
                                }
                            }

                            row_max[i] = new_max;
                        }
                    }
                }

                // Normalize output
                for i in 0..seq_len {
                    let sum = row_sum[i].max(1e-9);
                    for d in 0..head_dim {
                        let out_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                        output[out_idx] = row_out[i][d] / sum;
                    }
                }
            }
        }

        Tensor::from_vec(output, &[batch_size, num_heads, seq_len, head_dim]).unwrap()
    }

    /// Computes memory usage estimate for standard vs flash attention.
    ///
    /// # Arguments
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Head dimension
    ///
    /// # Returns
    /// Tuple of (standard_memory_mb, flash_memory_mb)
    pub fn memory_estimate(
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> (f32, f32) {
        let bytes_per_float = 4;

        // Standard attention stores full N×N attention matrix
        let standard_attn_matrix = batch_size * num_heads * seq_len * seq_len * bytes_per_float;
        let standard_qkv = 3 * batch_size * num_heads * seq_len * head_dim * bytes_per_float;
        let standard_total = (standard_attn_matrix + standard_qkv) as f32 / (1024.0 * 1024.0);

        // Flash attention only stores block-sized tiles
        let block_size = 64;
        let flash_tile = batch_size * num_heads * block_size * block_size * bytes_per_float;
        let flash_qkv = 3 * batch_size * num_heads * seq_len * head_dim * bytes_per_float;
        let flash_running = batch_size * num_heads * seq_len * (head_dim + 2) * bytes_per_float;
        let flash_total = (flash_tile + flash_qkv + flash_running) as f32 / (1024.0 * 1024.0);

        (standard_total, flash_total)
    }
}

/// Scaled Dot-Product Attention with optional Flash Attention optimization.
///
/// Automatically selects between standard and flash attention based on sequence length.
pub fn scaled_dot_product_attention(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    attn_mask: Option<&Tensor<f32>>,
    dropout_p: f32,
    is_causal: bool,
    scale: Option<f32>,
) -> Tensor<f32> {
    let shape = query.shape();
    let seq_len = shape[2];
    let head_dim = shape[3];

    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Use flash attention for longer sequences
    if seq_len > 128 && attn_mask.is_none() {
        let flash = FlashAttention::new(shape[1], dropout_p, is_causal);
        return flash.forward_qkv(query, key, value);
    }

    // Standard attention for shorter sequences or when mask is provided
    let flash = FlashAttention::new(shape[1], dropout_p, is_causal);
    flash.standard_attention(query, key, value, scale)
}
