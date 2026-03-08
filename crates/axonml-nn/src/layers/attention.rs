//! Attention Mechanisms - Multi-Head Attention
//!
//! # File
//! `crates/axonml-nn/src/layers/attention.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
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
// MultiHeadAttention
// =============================================================================

/// Multi-Head Attention mechanism.
///
/// Allows the model to jointly attend to information from different
/// representation subspaces at different positions.
///
/// # Arguments
/// * `embed_dim` - Total dimension of the model
/// * `num_heads` - Number of parallel attention heads
/// * `dropout` - Dropout probability (default: 0.0)
///
/// # Shape
/// - Query: (L, N, E) or (N, L, E) if batch_first
/// - Key: (S, N, E) or (N, S, E) if batch_first
/// - Value: (S, N, E) or (N, S, E) if batch_first
/// - Output: (L, N, E) or (N, L, E) if batch_first
pub struct MultiHeadAttention {
    /// Query projection.
    q_proj: Linear,
    /// Key projection.
    k_proj: Linear,
    /// Value projection.
    v_proj: Linear,
    /// Output projection.
    out_proj: Linear,
    /// Embedding dimension.
    embed_dim: usize,
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Scaling factor.
    scale: f32,
    /// Whether input is batch first.
    batch_first: bool,
}

impl MultiHeadAttention {
    /// Creates a new MultiHeadAttention module.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self::with_options(embed_dim, num_heads, 0.0, true)
    }

    /// Creates MultiHeadAttention with all options.
    pub fn with_options(
        embed_dim: usize,
        num_heads: usize,
        _dropout: f32,
        batch_first: bool,
    ) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );

        let head_dim = embed_dim / num_heads;
        let scale = (head_dim as f32).sqrt().recip();

        Self {
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            embed_dim,
            num_heads,
            head_dim,
            scale,
            batch_first,
        }
    }

    /// Try to expand attention mask on GPU via CUDA kernels.
    /// Returns None to fall through to CPU expansion.
    #[allow(unused_variables)]
    #[allow(dead_code)]
    fn try_gpu_mask_expand(
        mask_data: &Tensor<f32>,
        mask_shape: &[usize],
        scores_shape: &[usize],
        device: axonml_core::Device,
        total: usize,
        batch_size: usize,
        num_heads: usize,
        tgt_len: usize,
        src_len: usize,
    ) -> Option<Variable> {
        // TODO: CUDA kernel for mask broadcast expansion
        None
    }

    /// Computes attention using batched matmul (BLAS-accelerated).
    pub fn attention(
        &self,
        query: &Variable,
        key: &Variable,
        value: &Variable,
        attn_mask: Option<&Variable>,
    ) -> Variable {
        let q_shape = query.shape();
        let (batch_size, tgt_len, _) = if self.batch_first {
            (q_shape[0], q_shape[1], q_shape[2])
        } else {
            (q_shape[1], q_shape[0], q_shape[2])
        };
        let src_len = if self.batch_first {
            key.shape()[1]
        } else {
            key.shape()[0]
        };

        // Project Q, K, V  (all tracked through autograd)
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        // Reshape to multi-head: [batch, seq, embed] → [batch, seq, heads, head_dim]
        // Then transpose to:     [batch, heads, seq, head_dim]
        let q = q
            .reshape(&[batch_size, tgt_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape(&[batch_size, src_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape(&[batch_size, src_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // Scaled dot-product attention: scores = Q @ K^T * scale
        // K^T: [batch, heads, head_dim, src_len]
        let k_t = k.transpose(2, 3);
        // scores: [batch, heads, tgt_len, src_len]
        let scores = q.matmul(&k_t).mul_scalar(self.scale);

        // Apply attention mask (0 → -1e9 additive mask)
        // Mask shapes: [tgt_len, src_len] (causal) or [batch, src_len] (padding)
        // Scores shape: [batch, heads, tgt_len, src_len]
        let scores = if let Some(mask) = attn_mask {
            let mask_shape = mask.shape();
            let mask_data = mask.data();
            let scores_shape = scores.shape();
            let total = scores_shape.iter().product::<usize>();

            // GPU fast path: expand mask entirely on GPU via CUDA kernel
            // Avoids GPU→CPU→GPU round-trip (9 mask expansions per forward pass)
            #[cfg(feature = "cuda")]
            if scores.data().device().is_gpu() {
                // Ensure mask is on GPU (it's small, so upload is cheap if needed)
                let mask_gpu = if mask_data.device().is_gpu() {
                    mask_data.clone()
                } else {
                    mask_data.to_device(scores.data().device()).unwrap()
                };

                if let Some(expanded_tensor) = mask_gpu.mask_expand_cuda(
                    &scores_shape, batch_size, self.num_heads, tgt_len, src_len,
                ) {
                    let additive_mask = Variable::new(expanded_tensor, false);
                    return self.finish_attention(
                        scores.add_var(&additive_mask), &v, batch_size, tgt_len,
                    );
                }
                // Fall through to CPU path on unsupported shape
            }

            // CPU fallback: expand mask with nested loops
            let mask_vec = mask_data.to_vec();
            let additive: Vec<f32> = mask_vec
                .iter()
                .map(|&v| if v == 0.0 { -1e9 } else { 0.0 })
                .collect();

            let mut expanded = vec![0.0f32; total];

            if mask_shape.len() == 2 && mask_shape[0] == tgt_len && mask_shape[1] == src_len {
                // Causal mask [tgt_len, src_len] → broadcast over batch & heads
                for b in 0..batch_size {
                    for h in 0..self.num_heads {
                        for i in 0..tgt_len {
                            for j in 0..src_len {
                                let idx = b * self.num_heads * tgt_len * src_len
                                    + h * tgt_len * src_len
                                    + i * src_len
                                    + j;
                                expanded[idx] = additive[i * src_len + j];
                            }
                        }
                    }
                }
            } else if mask_shape.len() == 2 && mask_shape[0] == batch_size && mask_shape[1] == src_len {
                // Padding mask [batch, src_len] → broadcast over heads & tgt positions
                for b in 0..batch_size {
                    for h in 0..self.num_heads {
                        for i in 0..tgt_len {
                            for j in 0..src_len {
                                let idx = b * self.num_heads * tgt_len * src_len
                                    + h * tgt_len * src_len
                                    + i * src_len
                                    + j;
                                expanded[idx] = additive[b * src_len + j];
                            }
                        }
                    }
                }
            } else {
                // General: tile mask across scores using modular indexing
                for (i, val) in expanded.iter_mut().enumerate() {
                    *val = additive[i % additive.len()];
                }
            }

            let mut additive_tensor = Tensor::from_vec(expanded, &scores_shape).unwrap();
            let scores_device = scores.data().device();
            if scores_device.is_gpu() {
                additive_tensor = additive_tensor.to_device(scores_device).unwrap();
            }
            let additive_mask = Variable::new(additive_tensor, false);
            scores.add_var(&additive_mask)
        } else {
            scores
        };

        self.finish_attention(scores, &v, batch_size, tgt_len)
    }

    /// Softmax → weighted sum → reshape → output projection.
    /// Shared by both GPU and CPU mask expansion paths.
    fn finish_attention(
        &self,
        scores: Variable,
        v: &Variable,
        batch_size: usize,
        tgt_len: usize,
    ) -> Variable {
        let attn_weights = scores.softmax(-1);
        let attn_output = attn_weights.matmul(v);
        let attn_output = attn_output
            .transpose(1, 2)
            .reshape(&[batch_size, tgt_len, self.embed_dim]);
        self.out_proj.forward(&attn_output)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Variable) -> Variable {
        // Self-attention: query = key = value = input
        self.attention(input, input, input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.q_proj.named_parameters() {
            params.insert(format!("q_proj.{name}"), param);
        }
        for (name, param) in self.k_proj.named_parameters() {
            params.insert(format!("k_proj.{name}"), param);
        }
        for (name, param) in self.v_proj.named_parameters() {
            params.insert(format!("v_proj.{name}"), param);
        }
        for (name, param) in self.out_proj.named_parameters() {
            params.insert(format!("out_proj.{name}"), param);
        }
        params
    }

    fn name(&self) -> &'static str {
        "MultiHeadAttention"
    }
}

// =============================================================================
// CrossAttention
// =============================================================================

/// Cross-Attention mechanism for encoder-decoder architectures.
///
/// Queries come from the decoder, keys and values come from the encoder.
/// This is the standard cross-attention used in Transformer decoders,
/// seq2seq models, and vision-language models.
///
/// # Shape (batch_first=true)
/// - Query (decoder): (N, T, E)
/// - Memory (encoder): (N, S, E)
/// - Output: (N, T, E)
///
/// where N=batch, T=target seq len, S=source seq len, E=embed_dim.
pub struct CrossAttention {
    /// Underlying multi-head attention.
    mha: MultiHeadAttention,
}

impl CrossAttention {
    /// Creates a new CrossAttention module.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(embed_dim, num_heads),
        }
    }

    /// Creates CrossAttention with all options.
    pub fn with_options(embed_dim: usize, num_heads: usize, dropout: f32, batch_first: bool) -> Self {
        Self {
            mha: MultiHeadAttention::with_options(embed_dim, num_heads, dropout, batch_first),
        }
    }

    /// Computes cross-attention.
    ///
    /// # Arguments
    /// * `query` - Decoder hidden states (N, T, E)
    /// * `memory` - Encoder output (N, S, E)
    /// * `attn_mask` - Optional attention mask
    pub fn cross_attention(
        &self,
        query: &Variable,
        memory: &Variable,
        attn_mask: Option<&Variable>,
    ) -> Variable {
        self.mha.attention(query, memory, memory, attn_mask)
    }

    /// Returns the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.mha.embed_dim
    }

    /// Returns the number of heads.
    pub fn num_heads(&self) -> usize {
        self.mha.num_heads
    }
}

impl Module for CrossAttention {
    fn forward(&self, input: &Variable) -> Variable {
        // When called as Module (single input), acts as self-attention.
        // Use cross_attention() for encoder-decoder attention.
        self.mha.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.mha.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.mha.named_parameters() {
            params.insert(format!("mha.{name}"), param);
        }
        params
    }

    fn name(&self) -> &'static str {
        "CrossAttention"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multihead_attention_creation() {
        let mha = MultiHeadAttention::new(512, 8);
        assert_eq!(mha.embed_dim, 512);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.head_dim, 64);
    }

    #[test]
    fn test_multihead_attention_forward() {
        let mha = MultiHeadAttention::new(64, 4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = mha.forward(&input);
        assert_eq!(output.shape(), vec![2, 10, 64]);
    }

    #[test]
    fn test_cross_attention() {
        let mha = MultiHeadAttention::new(64, 4);
        let query = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        let key_value = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = mha.attention(&query, &key_value, &key_value, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_multihead_attention_parameters() {
        let mha = MultiHeadAttention::new(64, 4);
        let params = mha.parameters();
        // Q, K, V, Out projections each have weight + bias = 8 total
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_cross_attention_creation() {
        let ca = CrossAttention::new(256, 8);
        assert_eq!(ca.embed_dim(), 256);
        assert_eq!(ca.num_heads(), 8);
    }

    #[test]
    fn test_cross_attention_forward() {
        let ca = CrossAttention::new(64, 4);
        // Decoder query: (batch=2, tgt_len=5, embed=64)
        let query = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        // Encoder memory: (batch=2, src_len=10, embed=64)
        let memory = Variable::new(
            Tensor::from_vec(vec![0.2; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = ca.cross_attention(&query, &memory, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_cross_attention_self_attention_fallback() {
        let ca = CrossAttention::new(64, 4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 8 * 64], &[2, 8, 64]).unwrap(),
            false,
        );
        // Module::forward does self-attention
        let output = ca.forward(&input);
        assert_eq!(output.shape(), vec![2, 8, 64]);
    }

    #[test]
    fn test_cross_attention_parameters() {
        let ca = CrossAttention::new(64, 4);
        let params = ca.parameters();
        assert_eq!(params.len(), 8); // Q, K, V, Out × (weight + bias)
        let named = ca.named_parameters();
        assert!(named.contains_key("mha.q_proj.weight"));
        assert!(named.contains_key("mha.out_proj.bias"));
    }
}
