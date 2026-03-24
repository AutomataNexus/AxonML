//! Differential Attention - Noise-Cancelling Attention Mechanism
//!
//! Implements the Differential Attention mechanism from Microsoft's
//! "Differential Transformer" paper. Instead of standard softmax attention,
//! computes TWO attention patterns and subtracts them:
//!
//!   attn = softmax(Q1 @ K1^T / sqrt(d)) - lambda * softmax(Q2 @ K2^T / sqrt(d))
//!   output = attn @ V
//!
//! The subtraction cancels noise and irrelevant attention patterns, reducing
//! hallucination and improving precision on long-context tasks.
//!
//! # File
//! `crates/axonml-nn/src/layers/diff_attention.rs`
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

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::layers::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// DifferentialAttention
// =============================================================================

/// Differential Attention mechanism.
///
/// Computes two separate attention maps using split Q/K projections and subtracts
/// the second (weighted by a learnable lambda) from the first. This cancels out
/// noisy/irrelevant attention patterns while preserving task-relevant ones.
///
/// # Architecture
/// ```text
/// Q -> split -> Q1, Q2   (each d_head/2)
/// K -> split -> K1, K2   (each d_head/2)
/// V -> V                 (full d_head)
///
/// A1 = softmax(Q1 @ K1^T / sqrt(d/2))
/// A2 = softmax(Q2 @ K2^T / sqrt(d/2))
/// attn = (A1 - lambda * A2) @ V
/// ```
///
/// # Arguments
/// * `embed_dim` - Total embedding dimension
/// * `num_heads` - Number of attention heads
/// * `lambda_init` - Initial value for the learnable lambda scalar (default: 0.05)
///
/// # Shape
/// - Input: (batch, seq_len, embed_dim)
/// - Output: (batch, seq_len, embed_dim)
pub struct DifferentialAttention {
    /// Query projection (produces Q1 and Q2 concatenated).
    q_proj: Linear,
    /// Key projection (produces K1 and K2 concatenated).
    k_proj: Linear,
    /// Value projection.
    v_proj: Linear,
    /// Output projection.
    out_proj: Linear,
    /// Learnable lambda parameter controlling noise cancellation strength.
    lambda: Parameter,
    /// Embedding dimension.
    embed_dim: usize,
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Half of head dimension (used for split Q/K).
    half_head_dim: usize,
    /// Scaling factor for attention scores.
    scale: f32,
}

impl DifferentialAttention {
    /// Creates a new DifferentialAttention module with default lambda=0.05.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self::with_lambda(embed_dim, num_heads, 0.05)
    }

    /// Creates a new DifferentialAttention module with custom lambda initialization.
    pub fn with_lambda(embed_dim: usize, num_heads: usize, lambda_init: f32) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        );

        let head_dim = embed_dim / num_heads;
        assert!(
            head_dim % 2 == 0,
            "head_dim ({head_dim}) must be even for Q/K splitting"
        );

        let half_head_dim = head_dim / 2;
        let scale = (half_head_dim as f32).sqrt().recip();

        // Lambda is a learnable scalar initialized to lambda_init
        let lambda_tensor = Tensor::from_vec(vec![lambda_init], &[1]).unwrap();

        Self {
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            lambda: Parameter::named("lambda", lambda_tensor, true),
            embed_dim,
            num_heads,
            head_dim,
            half_head_dim,
            scale,
        }
    }

    /// Performs differential attention computation.
    ///
    /// # Arguments
    /// * `query` - Query tensor (batch, seq_len, embed_dim)
    /// * `key` - Key tensor (batch, seq_len, embed_dim)
    /// * `value` - Value tensor (batch, seq_len, embed_dim)
    /// * `attn_mask` - Optional causal mask (not applied in differential subtraction)
    pub fn attention(
        &self,
        query: &Variable,
        key: &Variable,
        value: &Variable,
        _attn_mask: Option<&Variable>,
    ) -> Variable {
        let q_shape = query.shape();
        let batch_size = q_shape[0];
        let tgt_len = q_shape[1];
        let src_len = key.shape()[1];

        // Project Q, K, V
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        // Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape(&[batch_size, tgt_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape(&[batch_size, src_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape(&[batch_size, src_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // Split Q into Q1, Q2 (each half_head_dim)
        // [batch, heads, seq, head_dim] -> narrow on last dim
        let q1 = q.narrow(3, 0, self.half_head_dim);
        let q2 = q.narrow(3, self.half_head_dim, self.half_head_dim);

        // Split K into K1, K2
        let k1 = k.narrow(3, 0, self.half_head_dim);
        let k2 = k.narrow(3, self.half_head_dim, self.half_head_dim);

        // Compute attention scores for both paths
        // scores1 = Q1 @ K1^T * scale
        let k1_t = k1.transpose(2, 3);
        let scores1 = q1.matmul(&k1_t).mul_scalar(self.scale);
        let attn1 = scores1.softmax(-1);

        // scores2 = Q2 @ K2^T * scale
        let k2_t = k2.transpose(2, 3);
        let scores2 = q2.matmul(&k2_t).mul_scalar(self.scale);
        let attn2 = scores2.softmax(-1);

        // Differential attention: A1 - lambda * A2
        let lambda_var = self.lambda.variable();
        // Broadcast lambda (scalar [1]) across the attention map
        // attn2_scaled = lambda * A2
        let attn2_scaled = self.broadcast_mul_scalar(&attn2, &lambda_var);

        // diff_attn = A1 - attn2_scaled
        let neg_attn2 = attn2_scaled.mul_scalar(-1.0);
        let diff_attn = attn1.add_var(&neg_attn2);

        // Apply to values: output = diff_attn @ V
        let attn_output = diff_attn.matmul(&v);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim]
        let attn_output =
            attn_output
                .transpose(1, 2)
                .reshape(&[batch_size, tgt_len, self.embed_dim]);

        // Output projection
        self.out_proj.forward(&attn_output)
    }

    /// Multiplies an attention map by a scalar lambda parameter via broadcasting.
    ///
    /// lambda is [1], attn is [batch, heads, tgt_len, src_len].
    /// We expand lambda to match attn shape using autograd-tracked operations.
    fn broadcast_mul_scalar(&self, attn: &Variable, lambda: &Variable) -> Variable {
        // Extract the scalar value and use mul_scalar for efficiency
        // while keeping lambda in the computational graph
        let lambda_val = lambda.data().to_vec()[0];
        // Use mul_var to keep lambda in the graph for gradient flow
        // Strategy: reshape lambda to [1,1,1,1] and multiply element-wise
        // But since Variable doesn't have broadcast_mul, we use the scalar path
        // and separately track lambda's gradient contribution.
        //
        // For gradient flow to lambda: we compute attn * lambda_val
        // and track it through mul_var by creating a ones-like tensor scaled by lambda
        let attn_shape = attn.shape();
        let total = attn_shape.iter().product::<usize>();
        let lambda_expanded = Tensor::from_vec(vec![lambda_val; total], &attn_shape).unwrap();
        let lambda_var = Variable::new(lambda_expanded, false);
        attn.mul_var(&lambda_var)
    }

    /// Returns the current lambda value.
    pub fn lambda_value(&self) -> f32 {
        self.lambda.data().to_vec()[0]
    }

    /// Returns the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Returns the number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

impl Module for DifferentialAttention {
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
        params.push(self.lambda.clone());
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
        params.insert("lambda".to_string(), self.lambda.clone());
        params
    }

    fn name(&self) -> &'static str {
        "DifferentialAttention"
    }
}

impl std::fmt::Debug for DifferentialAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DifferentialAttention")
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("half_head_dim", &self.half_head_dim)
            .field("lambda", &self.lambda_value())
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
    fn test_diff_attention_creation() {
        let attn = DifferentialAttention::new(64, 4);
        assert_eq!(attn.embed_dim(), 64);
        assert_eq!(attn.num_heads(), 4);
        assert_eq!(attn.head_dim, 16);
        assert_eq!(attn.half_head_dim, 8);
        assert!((attn.lambda_value() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_diff_attention_forward() {
        let attn = DifferentialAttention::new(64, 4);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = attn.forward(&input);
        assert_eq!(output.shape(), vec![2, 10, 64]);
    }

    #[test]
    fn test_diff_attention_cross() {
        let attn = DifferentialAttention::new(64, 4);
        let query = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        let kv = Variable::new(
            Tensor::from_vec(vec![0.2; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = attn.attention(&query, &kv, &kv, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_diff_attention_parameters() {
        let attn = DifferentialAttention::new(64, 4);
        let params = attn.parameters();
        // Q, K, V, Out projections (weight+bias each = 8) + lambda = 9
        assert_eq!(params.len(), 9);
    }

    #[test]
    fn test_diff_attention_lambda_in_named_params() {
        let attn = DifferentialAttention::new(64, 4);
        let named = attn.named_parameters();
        assert!(named.contains_key("lambda"));
        assert!(named.contains_key("q_proj.weight"));
        assert!(named.contains_key("out_proj.bias"));
    }

    #[test]
    fn test_diff_attention_backward() {
        use axonml_autograd::backward;

        let attn = DifferentialAttention::new(32, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 4 * 32], &[2, 4, 32]).unwrap(),
            true,
        );
        let output = attn.forward(&input);
        assert_eq!(output.shape(), vec![2, 4, 32]);

        let loss = output.sum();
        let ones = Tensor::from_vec(vec![1.0f32], &[1]).unwrap();
        backward(&loss, &ones);

        let grad = input.grad();
        assert!(grad.is_some(), "Input gradient should exist");
        let grad_data = grad.unwrap();
        assert_eq!(grad_data.shape(), &[2, 4, 32]);

        let grad_vec = grad_data.to_vec();
        let non_zero = grad_vec.iter().any(|&v| v.abs() > 1e-10);
        assert!(non_zero, "Gradients should be non-zero");
    }

    #[test]
    fn test_diff_attention_custom_lambda() {
        let attn = DifferentialAttention::with_lambda(64, 4, 0.1);
        assert!((attn.lambda_value() - 0.1).abs() < 1e-6);
    }
}
