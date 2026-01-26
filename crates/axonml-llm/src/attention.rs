//! Attention Mechanisms Module
//!
//! Implements multi-head self-attention and causal (masked) self-attention
//! for transformer models with KV-cache support for efficient inference.

use axonml_autograd::Variable;
use axonml_nn::{Module, Linear, Dropout, Parameter};
use axonml_tensor::{Tensor, view::cat};

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
    pub fn update(&mut self, new_key: &Tensor<f32>, new_value: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
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
    pub fn new(num_layers: usize, batch_size: usize, num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
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
        let k_data = qkv_data.slice(&[0..batch_size, 0..seq_len, self.n_embd..2*self.n_embd]);
        let v_data = qkv_data.slice(&[0..batch_size, 0..seq_len, 2*self.n_embd..3*self.n_embd]);

        let q = Variable::new(q_data, qkv.requires_grad());
        let k_new = Variable::new(k_data.clone(), qkv.requires_grad());
        let v_new = Variable::new(v_data.clone(), qkv.requires_grad());

        // Reshape for multi-head attention
        // [batch, seq, n_embd] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let k_new = k_new.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let v_new = v_new.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);

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
            (k_new.clone(), v_new.clone(), seq_len, Some((k_new.data(), v_new.data())))
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
        let output = output.transpose(1, 2).reshape(&[batch_size, seq_len, self.n_embd]);

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
}
