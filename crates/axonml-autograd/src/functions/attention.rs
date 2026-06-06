//! Backward function for fused scaled dot-product attention.
//!
//! 343 lines. `FusedAttentionBackward` recomputes the attention matrix from
//! saved Q/K/V and the forward-pass row-max/row-sum, then computes grad_Q,
//! grad_K, grad_V without materializing the full N×N attention matrix in
//! memory. Supports causal masking. GPU-accelerated via the
//! `fused_attention_bwd_f32` CUDA kernel when the `cuda` feature is enabled.
//!
//! # File
//! `crates/axonml-autograd/src/functions/attention.rs`
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

use std::any::Any;

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// Fused Attention Backward
// =============================================================================

/// Gradient function for fused scaled dot-product attention.
///
/// For output = softmax(Q @ K^T * scale) @ V:
/// - grad_Q = grad_scores @ K * scale
/// - grad_K = grad_scores^T @ Q * scale
/// - grad_V = attn_weights^T @ grad_output
///
/// where grad_scores = P * (grad_output @ V^T - sum(grad_output * output))
///
/// On GPU, this uses a CUDA kernel that recomputes attention weights per query
/// row (memory-efficient). On CPU, falls back to standard matmul-based backward.
#[derive(Debug)]
pub struct FusedAttentionBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_q: Tensor<f32>,
    saved_k: Tensor<f32>,
    saved_v: Tensor<f32>,
    saved_output: Tensor<f32>,
    scale: f32,
    is_causal: bool,
}

impl FusedAttentionBackward {
    /// Creates a new `FusedAttentionBackward`.
    ///
    /// # Arguments
    /// * `q_grad_fn`, `k_grad_fn`, `v_grad_fn` - Gradient functions for Q, K, V inputs
    /// * `q`, `k`, `v` - Saved input tensors [B, H, Tq/Tk, D]
    /// * `output` - Forward output [B, H, Tq, D]
    /// * `scale` - Attention scale factor (1/sqrt(head_dim))
    /// * `is_causal` - Whether causal masking was applied
    #[must_use]
    pub fn new(
        q_grad_fn: Option<GradFn>,
        k_grad_fn: Option<GradFn>,
        v_grad_fn: Option<GradFn>,
        q: Tensor<f32>,
        k: Tensor<f32>,
        v: Tensor<f32>,
        output: Tensor<f32>,
        scale: f32,
        is_causal: bool,
    ) -> Self {
        Self {
            next_fns: vec![q_grad_fn, k_grad_fn, v_grad_fn],
            saved_q: q,
            saved_k: k,
            saved_v: v,
            saved_output: output,
            scale,
            is_causal,
        }
    }

    /// CPU fallback for attention backward using standard matmul operations.
    ///
    /// Recomputes attention weights and computes gradients without the CUDA kernel.
    fn backward_cpu(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let q_shape = self.saved_q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let tgt_len = q_shape[2];
        let head_dim = q_shape[3];
        let src_len = self.saved_k.shape()[2];

        let q_data = self.saved_q.to_vec();
        let k_data = self.saved_k.to_vec();
        let v_data = self.saved_v.to_vec();
        let o_data = self.saved_output.to_vec();
        let go_data = grad_output.to_vec();

        let total_q = batch_size * num_heads * tgt_len * head_dim;
        let total_kv = batch_size * num_heads * src_len * head_dim;

        let mut grad_q = vec![0.0f32; total_q];
        let mut grad_k = vec![0.0f32; total_kv];
        let mut grad_v = vec![0.0f32; total_kv];

        // Parallel over (batch, head) pairs — heads are completely independent,
        // disjoint memory regions for grad_*[h]. Use usize (Send+Sync) cast of raw ptrs
        // for safe mutation from rayon for_each closure (no overlapping writes across bh).
        // Good for CPU attention backward in single-GPU or pure-CPU training (distributed bottom tier).
        {
            use rayon::prelude::*;
            let num_bh = batch_size * num_heads;
            let gq_ptr = grad_q.as_mut_ptr() as usize;
            let gk_ptr = grad_k.as_mut_ptr() as usize;
            let gv_ptr = grad_v.as_mut_ptr() as usize;
            if num_bh > 1 {
                (0..num_bh).into_par_iter().for_each(|bh| {
                    let b = bh / num_heads;
                    let h = bh % num_heads;
                    let gq_ptr = gq_ptr as *mut f32;
                    let gk_ptr = gk_ptr as *mut f32;
                    let gv_ptr = gv_ptr as *mut f32;
                    for i in 0..tgt_len {
                        let eff_src = if self.is_causal {
                            (i + 1).min(src_len)
                        } else {
                            src_len
                        };
                        let qi_base = ((b * num_heads + h) * tgt_len + i) * head_dim;

                        // Recompute attention scores and softmax
                        let mut max_score = f32::NEG_INFINITY;
                        let mut scores = vec![0.0f32; eff_src];
                        for j in 0..eff_src {
                            let kj_base = ((b * num_heads + h) * src_len + j) * head_dim;
                            let mut s = 0.0f32;
                            for d in 0..head_dim {
                                s += q_data[qi_base + d] * k_data[kj_base + d];
                            }
                            s *= self.scale;
                            scores[j] = s;
                            if s > max_score {
                                max_score = s;
                            }
                        }

                        // Softmax
                        let mut sum_exp = 0.0f32;
                        for s in &mut scores {
                            *s = (*s - max_score).exp();
                            sum_exp += *s;
                        }
                        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                        for s in &mut scores {
                            *s *= inv_sum;
                        }

                        // D_i = sum_d(grad_O[i,d] * O[i,d])
                        let mut d_i = 0.0f32;
                        for d in 0..head_dim {
                            d_i += go_data[qi_base + d] * o_data[qi_base + d];
                        }

                        // For each key position j
                        for j in 0..eff_src {
                            let kj_base = ((b * num_heads + h) * src_len + j) * head_dim;
                            let p_ij = scores[j];

                            // grad_attn[i,j] = sum_d(grad_O[i,d] * V[j,d])
                            let mut grad_attn_ij = 0.0f32;
                            for d in 0..head_dim {
                                grad_attn_ij += go_data[qi_base + d] * v_data[kj_base + d];
                            }

                            // grad_score[i,j] = P[i,j] * (grad_attn[i,j] - D_i)
                            let grad_score_ij = p_ij * (grad_attn_ij - d_i);
                            let scaled_gs = grad_score_ij * self.scale;

                            for d in 0..head_dim {
                                unsafe {
                                    *gv_ptr.add(kj_base + d) += p_ij * go_data[qi_base + d];
                                    *gq_ptr.add(qi_base + d) += scaled_gs * k_data[kj_base + d];
                                    *gk_ptr.add(kj_base + d) += scaled_gs * q_data[qi_base + d];
                                }
                            }
                        }
                    }
                });
            } else {
                // small case, sequential
                for b in 0..batch_size {
                    for h in 0..num_heads {
                        for i in 0..tgt_len {
                            let eff_src = if self.is_causal {
                                (i + 1).min(src_len)
                            } else {
                                src_len
                            };
                            let qi_base = ((b * num_heads + h) * tgt_len + i) * head_dim;

                            // Recompute attention scores and softmax
                            let mut max_score = f32::NEG_INFINITY;
                            let mut scores = vec![0.0f32; eff_src];
                            for j in 0..eff_src {
                                let kj_base = ((b * num_heads + h) * src_len + j) * head_dim;
                                let mut s = 0.0f32;
                                for d in 0..head_dim {
                                    s += q_data[qi_base + d] * k_data[kj_base + d];
                                }
                                s *= self.scale;
                                scores[j] = s;
                                if s > max_score {
                                    max_score = s;
                                }
                            }

                            // Softmax
                            let mut sum_exp = 0.0f32;
                            for s in &mut scores {
                                *s = (*s - max_score).exp();
                                sum_exp += *s;
                            }
                            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                            for s in &mut scores {
                                *s *= inv_sum;
                            }

                            // D_i = sum_d(grad_O[i,d] * O[i,d])
                            let mut d_i = 0.0f32;
                            for d in 0..head_dim {
                                d_i += go_data[qi_base + d] * o_data[qi_base + d];
                            }

                            // For each key position j
                            for j in 0..eff_src {
                                let kj_base = ((b * num_heads + h) * src_len + j) * head_dim;
                                let p_ij = scores[j];

                                // grad_attn[i,j] = sum_d(grad_O[i,d] * V[j,d])
                                let mut grad_attn_ij = 0.0f32;
                                for d in 0..head_dim {
                                    grad_attn_ij += go_data[qi_base + d] * v_data[kj_base + d];
                                }

                                // grad_score[i,j] = P[i,j] * (grad_attn[i,j] - D_i)
                                let grad_score_ij = p_ij * (grad_attn_ij - d_i);
                                let scaled_gs = grad_score_ij * self.scale;

                                for d in 0..head_dim {
                                    grad_v[kj_base + d] += p_ij * go_data[qi_base + d];
                                    grad_q[qi_base + d] += scaled_gs * k_data[kj_base + d];
                                    grad_k[kj_base + d] += scaled_gs * q_data[qi_base + d];
                                }
                            }
                        }
                    }
                }
            }
        }

        let gq = Tensor::from_vec(grad_q, q_shape).expect("backward: tensor creation failed");
        let gk = Tensor::from_vec(grad_k, self.saved_k.shape()).unwrap();
        let gv = Tensor::from_vec(grad_v, self.saved_v.shape()).unwrap();

        vec![Some(gq), Some(gk), Some(gv)]
    }
}

impl GradientFunction for FusedAttentionBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // Try GPU backward kernel
        #[cfg(feature = "cuda")]
        if self.saved_q.device().is_gpu() {
            // Ensure grad_output is on GPU
            let go_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_q.device()).unwrap()
            };

            if let Some((gq, gk, gv)) = self.saved_q.fused_attention_bwd_cuda(
                &self.saved_k,
                &self.saved_v,
                &self.saved_output,
                &go_gpu,
                self.scale,
                self.is_causal,
            ) {
                return vec![Some(gq), Some(gk), Some(gv)];
            }
            // Fall through to CPU on failure
        }

        // CPU fallback
        self.backward_cpu(grad_output)
    }

    fn name(&self) -> &'static str {
        "FusedAttentionBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_backward_shapes() {
        let batch = 1;
        let heads = 2;
        let tgt_len = 3;
        let src_len = 3;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let shape_q = [batch, heads, tgt_len, head_dim];
        let shape_kv = [batch, heads, src_len, head_dim];
        let n_q = batch * heads * tgt_len * head_dim;
        let n_kv = batch * heads * src_len * head_dim;

        let q = Tensor::from_vec(vec![0.5f32; n_q], &shape_q)
            .expect("backward: tensor creation failed");
        let k = Tensor::from_vec(vec![0.3f32; n_kv], &shape_kv)
            .expect("backward: tensor creation failed");
        let v = Tensor::from_vec(vec![0.1f32; n_kv], &shape_kv)
            .expect("backward: tensor creation failed");
        let output = Tensor::from_vec(vec![0.1f32; n_q], &shape_q)
            .expect("backward: tensor creation failed");

        let backward = FusedAttentionBackward::new(None, None, None, q, k, v, output, scale, false);

        let grad_output = Tensor::from_vec(vec![1.0f32; n_q], &shape_q)
            .expect("backward: tensor creation failed");
        let grads = backward.apply(&grad_output);

        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0].as_ref().unwrap().shape(), &shape_q);
        assert_eq!(grads[1].as_ref().unwrap().shape(), &shape_kv);
        assert_eq!(grads[2].as_ref().unwrap().shape(), &shape_kv);
    }

    #[test]
    fn test_attention_backward_finite_nonzero() {
        let batch = 1;
        let heads = 1;
        let seq_len = 2;
        let head_dim = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let shape = [batch, heads, seq_len, head_dim];
        let n = batch * heads * seq_len * head_dim;

        let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &shape)
            .expect("backward: tensor creation failed");
        let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &shape)
            .expect("backward: tensor creation failed");
        let v = Tensor::from_vec(vec![0.5, 0.3, 0.2, 0.8], &shape)
            .expect("backward: tensor creation failed");
        let output = Tensor::from_vec(vec![0.35, 0.55, 0.35, 0.55], &shape)
            .expect("backward: tensor creation failed");

        let backward = FusedAttentionBackward::new(None, None, None, q, k, v, output, scale, false);

        let grad =
            Tensor::from_vec(vec![1.0; n], &shape).expect("backward: tensor creation failed");
        let grads = backward.apply(&grad);

        for (name, g) in [("Q", &grads[0]), ("K", &grads[1]), ("V", &grads[2])] {
            let data = g.as_ref().unwrap().to_vec();
            for &val in &data {
                assert!(val.is_finite(), "grad_{name} not finite: {val}");
            }
        }

        // At least V gradient should be nonzero
        let v_nonzero = grads[2]
            .as_ref()
            .unwrap()
            .to_vec()
            .iter()
            .filter(|v| v.abs() > 1e-10)
            .count();
        assert!(v_nonzero > 0, "Expected nonzero V gradients");
    }

    #[test]
    fn test_attention_backward_causal() {
        let batch = 1;
        let heads = 1;
        let seq_len = 3;
        let head_dim = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let shape = [batch, heads, seq_len, head_dim];
        let n = batch * heads * seq_len * head_dim;

        let q =
            Tensor::from_vec(vec![0.5f32; n], &shape).expect("backward: tensor creation failed");
        let k =
            Tensor::from_vec(vec![0.3f32; n], &shape).expect("backward: tensor creation failed");
        let v =
            Tensor::from_vec(vec![0.1f32; n], &shape).expect("backward: tensor creation failed");
        let output =
            Tensor::from_vec(vec![0.1f32; n], &shape).expect("backward: tensor creation failed");

        let backward = FusedAttentionBackward::new(
            None, None, None, q, k, v, output, scale, true, // causal=true
        );

        let grad =
            Tensor::from_vec(vec![1.0f32; n], &shape).expect("backward: tensor creation failed");
        let grads = backward.apply(&grad);

        assert_eq!(grads.len(), 3);
        for g in &grads {
            for &val in &g.as_ref().unwrap().to_vec() {
                assert!(val.is_finite(), "Causal attention grad not finite: {val}");
            }
        }
    }
}
