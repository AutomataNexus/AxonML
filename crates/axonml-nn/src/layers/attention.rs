//! Attention Mechanisms - Multi-Head Attention
//!
//! Implements scaled dot-product and multi-head attention.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

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

    /// Computes attention.
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

        // Project Q, K, V
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        // Reshape for multi-head: (batch, seq, embed) -> (batch, heads, seq, head_dim)
        // For simplicity, we'll work with the flat representation
        let q_vec = q.data().to_vec();
        let k_vec = k.data().to_vec();
        let v_vec = v.data().to_vec();

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let mut attn_scores = vec![0.0f32; batch_size * self.num_heads * tgt_len * src_len];

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..tgt_len {
                    for j in 0..src_len {
                        let mut score = 0.0f32;
                        for d in 0..self.head_dim {
                            let q_idx = b * tgt_len * self.embed_dim
                                + i * self.embed_dim
                                + h * self.head_dim
                                + d;
                            let k_idx = b * src_len * self.embed_dim
                                + j * self.embed_dim
                                + h * self.head_dim
                                + d;
                            score += q_vec[q_idx] * k_vec[k_idx];
                        }
                        let attn_idx = b * self.num_heads * tgt_len * src_len
                            + h * tgt_len * src_len
                            + i * src_len
                            + j;
                        attn_scores[attn_idx] = score * self.scale;
                    }
                }
            }
        }

        // Apply attention mask if provided
        if let Some(mask) = attn_mask {
            let mask_vec = mask.data().to_vec();
            for (i, score) in attn_scores.iter_mut().enumerate() {
                if mask_vec[i % mask_vec.len()] == 0.0 {
                    *score = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax over source sequence
        let mut attn_weights = vec![0.0f32; batch_size * self.num_heads * tgt_len * src_len];
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..tgt_len {
                    let base = b * self.num_heads * tgt_len * src_len
                        + h * tgt_len * src_len
                        + i * src_len;

                    // Find max for numerical stability
                    let max_score = (0..src_len)
                        .map(|j| attn_scores[base + j])
                        .fold(f32::NEG_INFINITY, f32::max);

                    // Compute exp and sum
                    let mut sum = 0.0f32;
                    for j in 0..src_len {
                        let exp_val = (attn_scores[base + j] - max_score).exp();
                        attn_weights[base + j] = exp_val;
                        sum += exp_val;
                    }

                    // Normalize
                    for j in 0..src_len {
                        attn_weights[base + j] /= sum;
                    }
                }
            }
        }

        // Apply attention to values
        let mut output_vec = vec![0.0f32; batch_size * tgt_len * self.embed_dim];
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..tgt_len {
                    for d in 0..self.head_dim {
                        let mut weighted_sum = 0.0f32;
                        for j in 0..src_len {
                            let attn_idx = b * self.num_heads * tgt_len * src_len
                                + h * tgt_len * src_len
                                + i * src_len
                                + j;
                            let v_idx = b * src_len * self.embed_dim
                                + j * self.embed_dim
                                + h * self.head_dim
                                + d;
                            weighted_sum += attn_weights[attn_idx] * v_vec[v_idx];
                        }
                        let out_idx = b * tgt_len * self.embed_dim
                            + i * self.embed_dim
                            + h * self.head_dim
                            + d;
                        output_vec[out_idx] = weighted_sum;
                    }
                }
            }
        }

        let output_shape = if self.batch_first {
            vec![batch_size, tgt_len, self.embed_dim]
        } else {
            vec![tgt_len, batch_size, self.embed_dim]
        };

        let output = Variable::new(
            Tensor::from_vec(output_vec, &output_shape).unwrap(),
            query.requires_grad(),
        );

        // Output projection
        self.out_proj.forward(&output)
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
}
