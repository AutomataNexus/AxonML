//! Hydra - Hybrid SSM + Sparse Attention Small Language Model
//!
//! # File
//! `crates/axonml-llm/src/hydra.rs`
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
use axonml_nn::{Embedding, Linear, Module, Parameter};
use axonml_tensor::Tensor;
use rand::Rng;

use crate::llama::RMSNorm;
use crate::ssm::{SSMBlock, SSMConfig};

// =============================================================================
// Hydra Configuration
// =============================================================================

/// Configuration for the Hydra hybrid SSM + Attention model.
#[derive(Debug, Clone)]
pub struct HydraConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of total layers (half SSM, half attention, alternating)
    pub num_layers: usize,
    /// Number of attention heads (for attention layers)
    pub num_heads: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// SSM convolution kernel size
    pub d_conv: usize,
    /// SSM expansion factor (d_inner = expansion * d_model)
    pub ssm_expansion: usize,
    /// MLP intermediate size
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Windowed attention window size
    pub window_size: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// Dropout rate
    pub dropout: f32,
}

impl HydraConfig {
    /// Standard ~300M parameter configuration.
    pub fn base() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 768,
            num_layers: 24,
            num_heads: 12,
            d_state: 16,
            d_conv: 4,
            ssm_expansion: 2,
            intermediate_size: 768 * 4,
            max_seq_len: 8192,
            window_size: 256,
            rms_norm_eps: 1e-5,
            dropout: 0.0,
        }
    }

    /// Small configuration for testing/development.
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 256,
            num_layers: 8,
            num_heads: 4,
            d_state: 16,
            d_conv: 4,
            ssm_expansion: 2,
            intermediate_size: 256 * 4,
            max_seq_len: 2048,
            window_size: 128,
            rms_norm_eps: 1e-5,
            dropout: 0.0,
        }
    }

    /// Tiny configuration for unit tests.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 64,
            num_layers: 4,
            num_heads: 4,
            d_state: 8,
            d_conv: 4,
            ssm_expansion: 2,
            intermediate_size: 256,
            max_seq_len: 512,
            window_size: 64,
            rms_norm_eps: 1e-5,
            dropout: 0.0,
        }
    }

    /// Head dimension for attention layers.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// SSM inner dimension.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.ssm_expansion
    }
}

// =============================================================================
// Windowed Local Attention
// =============================================================================

/// Multi-head attention with a sliding window (local attention).
///
/// Each query attends only to keys within a window of `window_size` tokens,
/// giving O(n * w * d) complexity instead of O(n^2 * d).
#[derive(Debug)]
pub struct WindowedAttention {
    /// Query projection
    q_proj: Linear,
    /// Key projection
    k_proj: Linear,
    /// Value projection
    v_proj: Linear,
    /// Output projection
    o_proj: Linear,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Window size for local attention
    window_size: usize,
    /// Attention dropout probability
    attn_drop_p: f32,
}

impl WindowedAttention {
    /// Create new windowed attention layer.
    pub fn new(config: &HydraConfig) -> Self {
        let head_dim = config.head_dim();
        Self {
            q_proj: Linear::new(config.d_model, config.d_model),
            k_proj: Linear::new(config.d_model, config.d_model),
            v_proj: Linear::new(config.d_model, config.d_model),
            o_proj: Linear::new(config.d_model, config.d_model),
            num_heads: config.num_heads,
            head_dim,
            window_size: config.window_size,
            attn_drop_p: config.dropout,
        }
    }

    /// Forward pass: [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    pub fn forward(&self, x: &Variable) -> Variable {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, num_heads, head_dim] then transpose to
        // [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // Compute windowed attention manually
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let q_vec = q_data.to_vec();
        let k_vec = k_data.to_vec();
        let v_vec = v_data.to_vec();

        let w = self.window_size;
        let mut output = vec![0.0f32; batch_size * self.num_heads * seq_len * self.head_dim];

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    // Window: attend to [max(0, i-w+1) .. i] (causal + windowed)
                    let win_start = if i >= w { i - w + 1 } else { 0 };
                    let win_end = i + 1; // causal: only attend to positions <= i
                    let win_len = win_end - win_start;

                    // Compute attention scores for this window
                    let mut scores = vec![0.0f32; win_len];
                    let q_off = ((b * self.num_heads + h) * seq_len + i) * self.head_dim;

                    for (wi, j) in (win_start..win_end).enumerate() {
                        let k_off = ((b * self.num_heads + h) * seq_len + j) * self.head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            dot += q_vec[q_off + d] * k_vec[k_off + d];
                        }
                        scores[wi] = dot * scale;
                    }

                    // Softmax over window
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    if exp_sum > 0.0 {
                        for s in &mut scores {
                            *s /= exp_sum;
                        }
                    }

                    // Apply attention dropout (training only)
                    if self.attn_drop_p > 0.0 && is_grad_enabled() {
                        let mut rng = rand::thread_rng();
                        let keep_scale = 1.0 / (1.0 - self.attn_drop_p);
                        for s in &mut scores {
                            if rng.r#gen::<f32>() < self.attn_drop_p {
                                *s = 0.0;
                            } else {
                                *s *= keep_scale;
                            }
                        }
                    }

                    // Weighted sum of values
                    let o_off = ((b * self.num_heads + h) * seq_len + i) * self.head_dim;
                    for (wi, j) in (win_start..win_end).enumerate() {
                        let v_off = ((b * self.num_heads + h) * seq_len + j) * self.head_dim;
                        for d in 0..self.head_dim {
                            output[o_off + d] += scores[wi] * v_vec[v_off + d];
                        }
                    }
                }
            }
        }

        let attn_out = Tensor::from_vec(
            output,
            &[batch_size, self.num_heads, seq_len, self.head_dim],
        )
        .unwrap();

        // Build variable with backward support through Q,K,V projections
        let requires_grad = x.requires_grad() && is_grad_enabled();

        // Transpose back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        let attn_var = if requires_grad {
            let grad_fn = GradFn::new(WindowedAttnBackward {
                next_fns: vec![x.grad_fn().cloned()],
                saved_q: q_data.clone(),
                saved_k: k_data.clone(),
                saved_v: v_data.clone(),
                num_heads: self.num_heads,
                head_dim: self.head_dim,
                window_size: self.window_size,
                scale,
            });
            Variable::from_operation(attn_out, grad_fn, true)
        } else {
            Variable::new(attn_out, false)
        };

        // Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        let reshaped = attn_var
            .transpose(1, 2)
            .reshape(&[batch_size, seq_len, d_model]);

        // Output projection
        self.o_proj.forward(&reshaped)
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
// WindowedAttnBackward
// =============================================================================

/// Gradient for windowed attention.
///
/// Recomputes attention weights from saved Q, K, V and distributes gradients
/// correctly through the softmax(QK^T/sqrt(d)) * V computation.
///
/// Given output = softmax(scores) @ V where scores = Q @ K^T * scale:
///   dV = attn_weights^T @ dO
///   d_attn = dO @ V^T
///   d_scores = d_attn * attn_weights - attn_weights * sum(d_attn * attn_weights, dim=-1)
///   dQ = d_scores @ K * scale
///   dK = d_scores^T @ Q * scale
///
/// The gradient w.r.t. x is then dQ + dK + dV (summed since x feeds all three projections
/// through the same input path before this backward node).
#[derive(Debug)]
struct WindowedAttnBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_q: Tensor<f32>,
    saved_k: Tensor<f32>,
    saved_v: Tensor<f32>,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
    scale: f32,
}

impl GradientFunction for WindowedAttnBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // grad_output shape: [batch, num_heads, seq_len, head_dim]
        let shape = grad_output.shape();
        let batch_size = shape[0];
        let num_heads = self.num_heads;
        let seq_len = shape[2];
        let head_dim = self.head_dim;
        let w = self.window_size;
        let scale = self.scale;

        let q_vec = self.saved_q.to_vec();
        let k_vec = self.saved_k.to_vec();
        let v_vec = self.saved_v.to_vec();
        let go_vec = grad_output.to_vec();

        // Accumulate grad_q, grad_k, grad_v in same layout as saved tensors
        let total = batch_size * num_heads * seq_len * head_dim;
        let mut grad_q = vec![0.0f32; total];
        let mut grad_k = vec![0.0f32; total];
        let mut grad_v = vec![0.0f32; total];

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    let win_start = if i >= w { i - w + 1 } else { 0 };
                    let win_end = i + 1;
                    let win_len = win_end - win_start;

                    let q_off = ((b * num_heads + h) * seq_len + i) * head_dim;

                    // Recompute attention scores and weights for this query position
                    let mut scores = vec![0.0f32; win_len];
                    for (wi, j) in (win_start..win_end).enumerate() {
                        let k_off = ((b * num_heads + h) * seq_len + j) * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[q_off + d] * k_vec[k_off + d];
                        }
                        scores[wi] = dot * scale;
                    }

                    // Softmax
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut attn_w = vec![0.0f32; win_len];
                    let mut exp_sum = 0.0f32;
                    for (wi, &s) in scores.iter().enumerate() {
                        attn_w[wi] = (s - max_s).exp();
                        exp_sum += attn_w[wi];
                    }
                    if exp_sum > 0.0 {
                        for a in &mut attn_w {
                            *a /= exp_sum;
                        }
                    }

                    // grad_output for position i
                    let go_off = ((b * num_heads + h) * seq_len + i) * head_dim;

                    // dV: for each j in window, grad_v[j] += attn_w[j] * grad_output[i]
                    for (wi, j) in (win_start..win_end).enumerate() {
                        let v_off = ((b * num_heads + h) * seq_len + j) * head_dim;
                        for d in 0..head_dim {
                            grad_v[v_off + d] += attn_w[wi] * go_vec[go_off + d];
                        }
                    }

                    // d_attn: dO @ V^T for each (i, j) in window
                    let mut d_attn = vec![0.0f32; win_len];
                    for (wi, j) in (win_start..win_end).enumerate() {
                        let v_off = ((b * num_heads + h) * seq_len + j) * head_dim;
                        for d in 0..head_dim {
                            d_attn[wi] += go_vec[go_off + d] * v_vec[v_off + d];
                        }
                    }

                    // d_scores = softmax_backward(d_attn, attn_w)
                    // = attn_w * (d_attn - sum(d_attn * attn_w))
                    let sum_da_aw: f32 = d_attn
                        .iter()
                        .zip(attn_w.iter())
                        .map(|(da, aw)| da * aw)
                        .sum();
                    let mut d_scores = vec![0.0f32; win_len];
                    for wi in 0..win_len {
                        d_scores[wi] = attn_w[wi] * (d_attn[wi] - sum_da_aw);
                    }

                    // dQ: d_scores @ K * scale (for query position i)
                    for (wi, j) in (win_start..win_end).enumerate() {
                        let k_off = ((b * num_heads + h) * seq_len + j) * head_dim;
                        for d in 0..head_dim {
                            grad_q[q_off + d] += d_scores[wi] * k_vec[k_off + d] * scale;
                        }
                    }

                    // dK: d_scores^T @ Q * scale (for each key position j)
                    for (wi, j) in (win_start..win_end).enumerate() {
                        let k_off = ((b * num_heads + h) * seq_len + j) * head_dim;
                        for d in 0..head_dim {
                            grad_k[k_off + d] += d_scores[wi] * q_vec[q_off + d] * scale;
                        }
                    }
                }
            }
        }

        // The input x feeds into q_proj, k_proj, v_proj independently.
        // This backward node sits before all three projections, so we sum gradients.
        // Each projection's backward will further distribute through its own weights.
        let mut grad_input = vec![0.0f32; total];
        for i in 0..total {
            grad_input[i] = grad_q[i] + grad_k[i] + grad_v[i];
        }

        let gi = Tensor::from_vec(grad_input, shape).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "WindowedAttnBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Hydra SwiGLU MLP
// =============================================================================

/// SwiGLU MLP for Hydra model (same as LLaMA MLP).
#[derive(Debug)]
pub struct HydraMLP {
    /// Gate projection
    gate_proj: Linear,
    /// Up projection
    up_proj: Linear,
    /// Down projection
    down_proj: Linear,
}

impl HydraMLP {
    /// Create new SwiGLU MLP.
    pub fn new(d_model: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::new(d_model, intermediate_size),
            up_proj: Linear::new(d_model, intermediate_size),
            down_proj: Linear::new(intermediate_size, d_model),
        }
    }

    /// Forward with SwiGLU activation.
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
// Hydra Block (SSM or Attention)
// =============================================================================

/// A single Hydra layer — either an SSM block or an Attention block.
///
/// Both variants include pre-norm (RMSNorm) + residual connection + MLP.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum HydraBlock {
    /// SSM (Mamba-style) block
    SSM {
        /// Pre-norm for SSM
        norm: RMSNorm,
        /// SSM block
        ssm: SSMBlock,
        /// Post-SSM norm for MLP
        mlp_norm: RMSNorm,
        /// Feed-forward MLP
        mlp: HydraMLP,
    },
    /// Windowed local attention block
    Attention {
        /// Pre-norm for attention
        norm: RMSNorm,
        /// Windowed attention
        attn: WindowedAttention,
        /// Post-attention norm for MLP
        mlp_norm: RMSNorm,
        /// Feed-forward MLP
        mlp: HydraMLP,
    },
}

impl HydraBlock {
    /// Create an SSM block.
    pub fn new_ssm(config: &HydraConfig) -> Self {
        let ssm_config = SSMConfig {
            d_model: config.d_model,
            d_state: config.d_state,
            d_inner: config.d_inner(),
            d_conv: config.d_conv,
            dt_rank: config.d_model.div_ceil(16),
        };
        HydraBlock::SSM {
            norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            ssm: SSMBlock::new(&ssm_config),
            mlp_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            mlp: HydraMLP::new(config.d_model, config.intermediate_size),
        }
    }

    /// Create an Attention block.
    pub fn new_attention(config: &HydraConfig) -> Self {
        HydraBlock::Attention {
            norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            attn: WindowedAttention::new(config),
            mlp_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            mlp: HydraMLP::new(config.d_model, config.intermediate_size),
        }
    }

    /// Forward pass with residual connections.
    pub fn forward(&self, x: &Variable) -> Variable {
        match self {
            HydraBlock::SSM {
                norm,
                ssm,
                mlp_norm,
                mlp,
            } => {
                // SSM with pre-norm + residual
                let residual = x.clone();
                let h = norm.forward(x);
                let h = ssm.forward(&h);
                let h = residual.add(&h);

                // MLP with pre-norm + residual
                let residual = h.clone();
                let h = mlp_norm.forward(&h);
                let h = mlp.forward(&h);
                residual.add(&h)
            }
            HydraBlock::Attention {
                norm,
                attn,
                mlp_norm,
                mlp,
            } => {
                // Attention with pre-norm + residual
                let residual = x.clone();
                let h = norm.forward(x);
                let h = attn.forward(&h);
                let h = residual.add(&h);

                // MLP with pre-norm + residual
                let residual = h.clone();
                let h = mlp_norm.forward(&h);
                let h = mlp.forward(&h);
                residual.add(&h)
            }
        }
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        match self {
            HydraBlock::SSM {
                norm,
                ssm,
                mlp_norm,
                mlp,
            } => {
                params.extend(norm.parameters());
                params.extend(ssm.parameters());
                params.extend(mlp_norm.parameters());
                params.extend(mlp.parameters());
            }
            HydraBlock::Attention {
                norm,
                attn,
                mlp_norm,
                mlp,
            } => {
                params.extend(norm.parameters());
                params.extend(attn.parameters());
                params.extend(mlp_norm.parameters());
                params.extend(mlp.parameters());
            }
        }
        params
    }
}

// =============================================================================
// Hydra Model
// =============================================================================

/// Hydra — a hybrid SSM + Sparse Attention small language model.
///
/// Interleaves State Space Model (Mamba-style S6) layers with windowed local
/// attention layers. SSM layers capture long-range dependencies in O(n) time,
/// while attention layers provide precise local token interactions.
///
/// Architecture:
///   Token Embedding -> [SSM Block, Attn Block] x N -> RMSNorm -> LM Head
#[derive(Debug)]
pub struct HydraModel {
    /// Token embeddings
    embed_tokens: Embedding,
    /// Alternating SSM and Attention blocks
    blocks: Vec<HydraBlock>,
    /// Final layer norm
    final_norm: RMSNorm,
    /// Language model head
    lm_head: Linear,
    /// Configuration
    config: HydraConfig,
}

impl HydraModel {
    /// Create new Hydra model from configuration.
    pub fn new(config: &HydraConfig) -> Self {
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            if i % 2 == 0 {
                blocks.push(HydraBlock::new_ssm(config));
            } else {
                blocks.push(HydraBlock::new_attention(config));
            }
        }

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.d_model),
            blocks,
            final_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            lm_head: Linear::new(config.d_model, config.vocab_size),
            config: config.clone(),
        }
    }

    /// Forward pass: token IDs -> logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token ID tensor [batch, seq_len] as u32
    ///
    /// # Returns
    /// Logits tensor [batch, seq_len, vocab_size]
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        // Convert to f32 for embedding lookup
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var = Variable::new(Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(), false);

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(&ids_var);

        // Pass through alternating blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        // Final norm
        let hidden = self.final_norm.forward(&hidden);

        // LM head
        self.lm_head.forward(&hidden)
    }

    /// Forward pass with cross-entropy loss for next-token prediction.
    ///
    /// Shifts logits and labels internally for causal LM training.
    pub fn forward_with_loss(
        &self,
        input_ids: &Tensor<u32>,
        labels: &Tensor<u32>,
    ) -> (Variable, Variable) {
        let logits = self.forward_ids(input_ids);

        let logits_data = logits.data();
        let shape = logits_data.shape().to_vec();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let _vocab_size = shape[2];
        drop(logits_data);

        if seq_len > 1 {
            // Shift: predict next token (use narrow to preserve grad graph)
            let shift_logits = logits.narrow(1, 0, seq_len - 1);

            // Shift labels
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
            let zero = Variable::new(Tensor::from_vec(vec![0.0f32], &[1]).unwrap(), false);
            (logits, zero)
        }
    }

    /// Compute cross-entropy loss.
    fn cross_entropy_loss(logits: &Variable, labels: &Tensor<u32>) -> Variable {
        let logits_data = logits.data();
        let shape = logits_data.shape().to_vec();
        drop(logits_data);
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        let logits_flat = logits.reshape(&[batch_size * seq_len, vocab_size]);

        let labels_vec = labels.to_vec();
        let valid_labels: Vec<f32> = labels_vec
            .iter()
            .map(|&l| {
                if (l as usize) < vocab_size {
                    l as f32
                } else {
                    0.0f32
                }
            })
            .collect();
        let target_var = Variable::new(
            Tensor::from_vec(valid_labels, &[batch_size * seq_len]).unwrap(),
            false,
        );

        use axonml_nn::loss::CrossEntropyLoss;
        CrossEntropyLoss::new().compute(&logits_flat, &target_var)
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        self.parameters().iter().map(|p| p.data().numel()).sum()
    }

    /// Get the configuration.
    pub fn config(&self) -> &HydraConfig {
        &self.config
    }

    /// Set training mode.
    pub fn train(&mut self) {
        // Dropout layers would switch here if needed
    }

    /// Set evaluation mode.
    pub fn eval(&mut self) {
        // Dropout layers would switch here if needed
    }
}

impl Module for HydraModel {
    fn forward(&self, input: &Variable) -> Variable {
        // Assume input is already embedded
        let mut hidden = input.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        let hidden = self.final_norm.forward(&hidden);
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
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydra_tiny_forward() {
        let config = HydraConfig::tiny();
        let model = HydraModel::new(&config);

        let input_ids = Tensor::<u32>::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let logits = model.forward_ids(&input_ids);

        assert_eq!(logits.data().shape()[0], 2);
        assert_eq!(logits.data().shape()[1], 4);
        assert_eq!(logits.data().shape()[2], config.vocab_size);
    }

    #[test]
    fn test_hydra_tiny_with_loss() {
        let config = HydraConfig::tiny();
        let model = HydraModel::new(&config);

        let input_ids = Tensor::<u32>::from_vec(vec![1, 2, 3, 4], &[1, 4]).unwrap();
        let labels = Tensor::<u32>::from_vec(vec![2, 3, 4, 5], &[1, 4]).unwrap();

        let (_logits, loss) = model.forward_with_loss(&input_ids, &labels);
        let loss_val = loss.data().to_vec()[0];
        assert!(loss_val > 0.0, "Cross-entropy loss should be positive");
    }

    #[test]
    fn test_hydra_param_count() {
        let config = HydraConfig::tiny();
        let model = HydraModel::new(&config);
        let count = model.param_count();
        assert!(count > 0, "Model should have parameters");
        println!("Hydra tiny params: {count}");
    }

    #[test]
    fn test_hydra_alternating_blocks() {
        let config = HydraConfig::tiny();
        let model = HydraModel::new(&config);

        // Verify alternating pattern: even=SSM, odd=Attention
        for (i, block) in model.blocks.iter().enumerate() {
            match block {
                HydraBlock::SSM { .. } => assert_eq!(i % 2, 0, "SSM at even index"),
                HydraBlock::Attention { .. } => assert_eq!(i % 2, 1, "Attn at odd index"),
            }
        }
    }

    #[test]
    fn test_windowed_attention_shape() {
        let config = HydraConfig::tiny();
        let attn = WindowedAttention::new(&config);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1f32; 2 * 8 * 64], &[2, 8, 64]).unwrap(),
            true,
        );
        let y = attn.forward(&x);
        assert_eq!(y.data().shape(), &[2, 8, 64]);
    }
}
