//! LLaMA - Large Language Model Meta AI
//!
//! # File
//! `crates/axonml-llm/src/llama.rs`
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

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_nn::{Dropout, Embedding, Linear, Module, Parameter};
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
    pub weight: Tensor<f32>,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Hidden size (used for serialization/debug)
    pub hidden_size: usize,
    /// Cached GPU-side weight. Populated on first GPU forward so subsequent
    /// forwards reuse the same device allocation — critical for CUDA graph
    /// capture, where every kernel arg's host-side cu_device_ptr must remain
    /// at a stable address across captures and replays.
    #[allow(clippy::type_complexity)]
    weight_gpu_cache: parking_lot::Mutex<Option<Tensor<f32>>>,
}

// Manual Clone — `weight_gpu_cache` (parking_lot::Mutex) is not Clone.
// Cloning resets the cache; the new instance will re-populate it on first
// GPU forward, which is the same behavior as a freshly-constructed RMSNorm.
// Cheap because Tensor::clone is Arc-shared.
impl Clone for RMSNorm {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            eps: self.eps,
            hidden_size: self.hidden_size,
            weight_gpu_cache: parking_lot::Mutex::new(None),
        }
    }
}

impl RMSNorm {
    /// Create new RMSNorm layer.
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::ones(&[hidden_size]),
            eps,
            hidden_size,
            weight_gpu_cache: parking_lot::Mutex::new(None),
        }
    }

    /// Forward pass. GPU-accelerated via `rms_norm_batched` when input is
    /// on device; CPU fallback computes row-wise rms for the same math.
    /// The backward kernel recomputes rms from the saved input so we don't
    /// need to stash per-row rms values.
    pub fn forward(&self, x: &Variable) -> Variable {
        let x_data = x.data();
        let shape = x_data.shape();
        let last_dim = shape[shape.len() - 1];
        let batch_elements: usize = shape.iter().take(shape.len() - 1).product();

        let output_tensor = if x_data.device().is_gpu() {
            // GPU path: one kernel launch per forward (down from full D2H).
            // Cache the moved weight on first GPU forward so subsequent
            // forwards reuse the same CudaSlice (identical cu_device_ptr
            // host address → CUDA graph capture / replay sees a stable ref).
            let weight_gpu = {
                let mut cache = self.weight_gpu_cache.lock();
                if let Some(ref w) = *cache {
                    if w.device() == x_data.device() {
                        w.clone()
                    } else {
                        // device drifted — re-upload
                        let moved = self
                            .weight
                            .to_device(x_data.device())
                            .expect("RMSNorm: failed to move weight to GPU");
                        *cache = Some(moved.clone());
                        moved
                    }
                } else if self.weight.device().is_gpu() {
                    *cache = Some(self.weight.clone());
                    self.weight.clone()
                } else {
                    let moved = self
                        .weight
                        .to_device(x_data.device())
                        .expect("RMSNorm: failed to move weight to GPU");
                    *cache = Some(moved.clone());
                    moved
                }
            };
            let x_2d = x_data
                .reshape(&[batch_elements as isize, last_dim as isize])
                .expect("RMSNorm: reshape to 2D");
            let out_2d = x_2d.rms_norm_batched(&weight_gpu, batch_elements, last_dim, self.eps);
            let shape_isize: Vec<isize> = shape.iter().map(|&s| s as isize).collect();
            out_2d
                .reshape(&shape_isize)
                .expect("RMSNorm: reshape output back to original shape")
        } else {
            // CPU fallback: original per-row rms computation.
            let x_vec = x_data.to_vec();
            let mut output = vec![0.0f32; x_vec.len()];
            let weight_vec = self.weight.to_vec();
            for b in 0..batch_elements {
                let offset = b * last_dim;
                let mut sum_sq = 0.0f32;
                for i in 0..last_dim {
                    sum_sq += x_vec[offset + i] * x_vec[offset + i];
                }
                let rms = (sum_sq / last_dim as f32 + self.eps).sqrt();
                for i in 0..last_dim {
                    output[offset + i] = (x_vec[offset + i] / rms) * weight_vec[i];
                }
            }
            Tensor::from_vec(output, shape).unwrap()
        };

        let requires_grad = x.requires_grad() && is_grad_enabled();
        if requires_grad {
            // Save the weight on whatever device the input was on. For a
            // GPU forward we've already cached a GPU copy above; cloning
            // the cache yields a stable Arc-shared GPU tensor that the
            // backward kernel can use without another host→device upload
            // (critical for CUDA-graph-captured steps).
            let saved_weight = if x_data.device().is_gpu() {
                self.weight_gpu_cache
                    .lock()
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| self.weight.clone())
            } else {
                self.weight.clone()
            };
            let grad_fn = GradFn::new(RMSNormBackward {
                next_fns: vec![x.grad_fn().cloned()],
                saved_input: x_data.clone(),
                weight: saved_weight,
                last_dim,
                eps: self.eps,
            });
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::named("weight", self.weight.clone(), true)]
    }

    /// Load weight from tensor.
    pub fn load_weight(&mut self, weight: &Tensor<f32>) {
        self.weight = weight.clone();
    }
}

// =============================================================================
// RMSNormBackward
// =============================================================================

/// Gradient function for RMSNorm.
///
/// y = (x / rms) * w
/// dy/dx = w/rms - x * (x . (w * dy)) / (rms^3 * D)
#[derive(Debug)]
struct RMSNormBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    weight: Tensor<f32>,
    last_dim: usize,
    eps: f32,
}

impl GradientFunction for RMSNormBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let x_shape = self.saved_input.shape();
        let numel = self.saved_input.numel();
        let d = self.last_dim;
        let m = numel / d;

        // Reshape saved_input, weight, grad_output to canonical [m, n] for the
        // batched kernel, then reshape result back to the saved shape.
        let saved_2d = self
            .saved_input
            .reshape(&[m as isize, d as isize])
            .expect("RMSNormBackward: reshape saved_input to 2D");

        // Move grad_output to the saved_input's device if needed so the kernel
        // (or CPU fallback) sees a consistent device set.
        let target = saved_2d.device();
        let grad_dev = if grad_output.device() == target {
            grad_output.clone()
        } else {
            grad_output
                .to_device(target)
                .expect("RMSNormBackward: failed to move grad_output to target device")
        };
        let grad_2d = grad_dev
            .reshape(&[m as isize, d as isize])
            .expect("RMSNormBackward: reshape grad_output to 2D");

        let weight_dev = if self.weight.device() == target {
            self.weight.clone()
        } else {
            self.weight
                .to_device(target)
                .expect("RMSNormBackward: failed to move weight to target device")
        };

        let grad_input_2d = saved_2d.rms_norm_bwd_batched(&weight_dev, &grad_2d, m, d, self.eps);
        let shape_isize: Vec<isize> = x_shape.iter().map(|&s| s as isize).collect();
        let grad_input = grad_input_2d
            .reshape(&shape_isize)
            .expect("RMSNormBackward: reshape output to saved shape");
        vec![Some(grad_input)]
    }

    fn name(&self) -> &'static str {
        "RMSNormBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Rotary Position Embedding (RoPE)
// =============================================================================

/// Rotary Position Embedding.
///
/// Encodes position information by rotating pairs of dimensions.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Dimension of the embedding
    dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Base theta
    pub theta: f32,
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
        for (i, freq) in inv_freq.iter_mut().enumerate() {
            *freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
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
    pub fn apply(
        &self,
        q: &Variable,
        k: &Variable,
        position_offset: usize,
    ) -> (Variable, Variable) {
        let q_data = q.data();
        let k_data = k.data();
        let shape = q_data.shape();
        let seq_len = shape[2];
        let head_dim = shape[3];

        let q_rotated = self.rotate_tensor(&q_data, seq_len, head_dim, position_offset);
        let k_rotated = self.rotate_tensor(&k_data, seq_len, head_dim, position_offset);

        let q_out = if q.requires_grad() && is_grad_enabled() {
            let grad_fn = GradFn::new(RoPEBackward {
                next_fns: vec![q.grad_fn().cloned()],
                cos_cached: self.cos_cached.clone(),
                sin_cached: self.sin_cached.clone(),
                rope_dim: self.dim,
                position_offset,
                theta: self.theta,
            });
            Variable::from_operation(q_rotated, grad_fn, true)
        } else {
            Variable::new(q_rotated, false)
        };

        let k_out = if k.requires_grad() && is_grad_enabled() {
            let grad_fn = GradFn::new(RoPEBackward {
                next_fns: vec![k.grad_fn().cloned()],
                cos_cached: self.cos_cached.clone(),
                sin_cached: self.sin_cached.clone(),
                rope_dim: self.dim,
                position_offset,
                theta: self.theta,
            });
            Variable::from_operation(k_rotated, grad_fn, true)
        } else {
            Variable::new(k_rotated, false)
        };

        (q_out, k_out)
    }

    fn rotate_tensor(
        &self,
        x: &Tensor<f32>,
        seq_len: usize,
        head_dim: usize,
        offset: usize,
    ) -> Tensor<f32> {
        let shape = x.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];

        // GPU fast path: one kernel launch, no D2H round-trip. Matches the
        // head-major `[bs, n_heads, seq, head_dim]` layout Qwen3/LLaMA
        // produces after their reshape+transpose chain.
        if x.device().is_gpu() {
            return x.apply_rope_split_halves_bhsd(
                batch_size, num_heads, seq_len, head_dim, self.theta, offset,
            );
        }

        let x_vec = x.to_vec();
        // Only copy the needed slice of cached cos/sin tables for positions [offset..offset+seq_len]
        let cos_slice = self.cos_cached.narrow(0, offset, seq_len).unwrap();
        let sin_slice = self.sin_cached.narrow(0, offset, seq_len).unwrap();
        let cos_vec = cos_slice.to_vec();
        let sin_vec = sin_slice.to_vec();

        let mut output = vec![0.0f32; x_vec.len()];
        let half_dim = head_dim / 2;

        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let x_offset = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let rope_offset = s * self.dim;

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
// RoPEBackward
// =============================================================================

/// Gradient function for Rotary Position Embedding.
///
/// Forward: y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos
/// Backward (transpose of rotation): dx1 = dy1*cos + dy2*sin, dx2 = -dy1*sin + dy2*cos
#[derive(Debug)]
struct RoPEBackward {
    next_fns: Vec<Option<GradFn>>,
    cos_cached: Tensor<f32>,
    sin_cached: Tensor<f32>,
    rope_dim: usize,
    position_offset: usize,
    /// RoPE theta, used by the GPU kernel to recompute angles on device.
    theta: f32,
}

impl GradientFunction for RoPEBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape = grad_output.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];
        let half_dim = head_dim / 2;

        // GPU fast path: single kernel, no D2H round-trip.
        if grad_output.device().is_gpu() {
            let gi = grad_output.rope_split_halves_bhsd_bwd(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                self.theta,
                self.position_offset,
            );
            return vec![Some(gi)];
        }

        let g_vec = grad_output.to_vec();
        // Only copy the needed slice of cached cos/sin tables
        let cos_slice = self
            .cos_cached
            .narrow(0, self.position_offset, seq_len)
            .unwrap();
        let sin_slice = self
            .sin_cached
            .narrow(0, self.position_offset, seq_len)
            .unwrap();
        let cos_vec = cos_slice.to_vec();
        let sin_vec = sin_slice.to_vec();

        let mut grad_input = vec![0.0f32; g_vec.len()];

        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let off = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let rope_off = s * self.rope_dim;

                    for i in 0..half_dim {
                        let cos_val = cos_vec[rope_off + i];
                        let sin_val = sin_vec[rope_off + i];

                        let dy1 = g_vec[off + i];
                        let dy2 = g_vec[off + half_dim + i];

                        // Inverse rotation (transpose)
                        grad_input[off + i] = dy1 * cos_val + dy2 * sin_val;
                        grad_input[off + half_dim + i] = -dy1 * sin_val + dy2 * cos_val;
                    }
                }
            }
        }

        let gi = Tensor::from_vec(grad_input, shape).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "RoPEBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// RepeatKVBackward
// =============================================================================

/// Gradient function for repeat_kv (GQA key-value repeat).
///
/// Forward repeats each KV head n_rep times.
/// Backward sums gradients over the repeated heads.
#[derive(Debug)]
pub(crate) struct RepeatKVBackward {
    pub(crate) next_fns: Vec<Option<GradFn>>,
    pub(crate) num_kv_heads: usize,
    pub(crate) n_rep: usize,
}

impl GradientFunction for RepeatKVBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape = grad_output.shape();
        let batch = shape[0];
        let seq_len = shape[2];
        let head_dim = shape[3];

        let g_vec = grad_output.to_vec();
        let mut grad_input = vec![0.0f32; batch * self.num_kv_heads * seq_len * head_dim];

        for b in 0..batch {
            for h in 0..self.num_kv_heads {
                for r in 0..self.n_rep {
                    for s in 0..seq_len {
                        let src_off = ((b * self.num_kv_heads * self.n_rep + h * self.n_rep + r)
                            * seq_len
                            + s)
                            * head_dim;
                        let dst_off = ((b * self.num_kv_heads + h) * seq_len + s) * head_dim;
                        for d in 0..head_dim {
                            grad_input[dst_off + d] += g_vec[src_off + d];
                        }
                    }
                }
            }
        }

        let gi =
            Tensor::from_vec(grad_input, &[batch, self.num_kv_heads, seq_len, head_dim]).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "RepeatKVBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
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
            rotary_emb: RotaryEmbedding::new(
                head_dim,
                config.max_position_embeddings,
                config.rope_theta,
            ),
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

        let output_tensor =
            Tensor::from_vec(output, &[batch, num_kv_heads * n_rep, seq_len, head_dim]).unwrap();

        if x.requires_grad() && is_grad_enabled() {
            let grad_fn = GradFn::new(RepeatKVBackward {
                next_fns: vec![x.grad_fn().cloned()],
                num_kv_heads,
                n_rep,
            });
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
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

    /// Load weights from state dict.
    pub fn load_weights(
        &mut self,
        prefix: &str,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = 0;

        if let Some(w) = weights.get(&format!("{}.q_proj.weight", prefix)) {
            self.q_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{}.k_proj.weight", prefix)) {
            self.k_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{}.v_proj.weight", prefix)) {
            self.v_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{}.o_proj.weight", prefix)) {
            self.o_proj.weight.update_data(w.clone());
            loaded += 1;
        }

        loaded
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

    /// Load weights from state dict.
    pub fn load_weights(
        &mut self,
        prefix: &str,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = 0;

        if let Some(w) = weights.get(&format!("{}.gate_proj.weight", prefix)) {
            self.gate_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{}.up_proj.weight", prefix)) {
            self.up_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{}.down_proj.weight", prefix)) {
            self.down_proj.weight.update_data(w.clone());
            loaded += 1;
        }

        loaded
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

    /// Load weights from state dict.
    pub fn load_weights(
        &mut self,
        prefix: &str,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = 0;

        // Attention weights
        loaded += self
            .self_attn
            .load_weights(&format!("{}.self_attn", prefix), weights);

        // MLP weights
        loaded += self.mlp.load_weights(&format!("{}.mlp", prefix), weights);

        // Layer norms
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            self.input_layernorm.load_weight(w);
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            self.post_attention_layernorm.load_weight(w);
            loaded += 1;
        }

        loaded
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
            self.config.max_position_embeddings,
            self.config.head_dim(),
        )
    }

    /// Load state dict from HuggingFace format weights.
    ///
    /// Expects weights with names like:
    /// - model.embed_tokens.weight
    /// - model.layers.0.self_attn.q_proj.weight
    /// - model.layers.0.mlp.gate_proj.weight
    /// - model.norm.weight
    pub fn load_state_dict(
        &mut self,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = 0;

        // Embedding
        if let Some(w) = weights
            .get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight"))
        {
            self.embed_tokens.weight.update_data(w.clone());
            loaded += 1;
        }

        // Layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Try both with and without "model." prefix
            let prefix1 = format!("model.layers.{}", i);
            let prefix2 = format!("layers.{}", i);

            let layer_loaded = layer.load_weights(&prefix1, weights);
            if layer_loaded == 0 {
                loaded += layer.load_weights(&prefix2, weights);
            } else {
                loaded += layer_loaded;
            }
        }

        // Final norm
        if let Some(w) = weights
            .get("model.norm.weight")
            .or_else(|| weights.get("norm.weight"))
        {
            self.norm.load_weight(w);
            loaded += 1;
        }

        println!("LLaMA: Loaded {} weight tensors", loaded);
        loaded
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

    /// Get the config.
    pub fn config(&self) -> &LLaMAConfig {
        &self.model.config
    }

    /// Generate text autoregressively.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch_size, seq_len]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = normal, <1 = more deterministic)
    /// * `top_k` - If set, only sample from top k tokens
    ///
    /// # Returns
    /// Generated token IDs [batch_size, seq_len + max_new_tokens]
    pub fn generate(
        &self,
        input_ids: &Tensor<u32>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        eos_token_id: Option<u32>,
    ) -> Tensor<u32> {
        let batch_size = input_ids.shape()[0];
        let mut cache = self.create_kv_cache(batch_size);

        // Start with input tokens
        let mut all_tokens: Vec<Vec<u32>> = (0..batch_size)
            .map(|b| {
                let start = b * input_ids.shape()[1];
                let end = start + input_ids.shape()[1];
                input_ids.to_vec()[start..end].to_vec()
            })
            .collect();

        // Process initial prompt
        let logits = self.forward_with_cache(input_ids, Some(&mut cache));

        // Get next token from last position
        let mut next_tokens = self.sample_next_token(&logits, temperature, top_k);

        // Check for EOS
        let mut finished = vec![false; batch_size];
        if let Some(eos_id) = eos_token_id {
            for (b, &token) in next_tokens.iter().enumerate() {
                if token == eos_id {
                    finished[b] = true;
                }
            }
        }

        // Append first generated token
        for (b, &token) in next_tokens.iter().enumerate() {
            all_tokens[b].push(token);
        }

        // Generate remaining tokens
        for _ in 1..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            // Create input tensor for next token
            let next_input = Tensor::from_vec(next_tokens.clone(), &[batch_size, 1]).unwrap();

            // Forward with cache (only processes 1 new token)
            let logits = self.forward_with_cache(&next_input, Some(&mut cache));

            // Sample next tokens
            next_tokens = self.sample_next_token(&logits, temperature, top_k);

            // Check for EOS and append
            for (b, &token) in next_tokens.iter().enumerate() {
                if !finished[b] {
                    all_tokens[b].push(token);
                    if Some(token) == eos_token_id {
                        finished[b] = true;
                    }
                }
            }
        }

        // Find max length and pad
        let max_len = all_tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        let mut output = vec![0u32; batch_size * max_len];
        for (b, tokens) in all_tokens.iter().enumerate() {
            for (i, &token) in tokens.iter().enumerate() {
                output[b * max_len + i] = token;
            }
        }

        Tensor::from_vec(output, &[batch_size, max_len]).unwrap()
    }

    /// Sample next token from logits.
    fn sample_next_token(
        &self,
        logits: &Variable,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Vec<u32> {
        let logits_data = logits.data();
        let shape = logits_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        let logits_vec = logits_data.to_vec();
        let mut next_tokens = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Get logits for last position
            let start = (b * seq_len + seq_len - 1) * vocab_size;
            let end = start + vocab_size;
            let mut token_logits: Vec<(usize, f32)> = logits_vec[start..end]
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v / temperature))
                .collect();

            // Apply top-k filtering
            if let Some(k) = top_k {
                token_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                token_logits.truncate(k);
            }

            // Softmax
            let max_logit = token_logits
                .iter()
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = token_logits
                .iter()
                .map(|(_, v)| (v - max_logit).exp())
                .sum();
            let probs: Vec<(usize, f32)> = token_logits
                .iter()
                .map(|(i, v)| (*i, (v - max_logit).exp() / exp_sum))
                .collect();

            // Sample from the distribution
            let next_token = {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let sample: f32 = rng.r#gen();
                let mut cumsum = 0.0f32;
                let mut selected = probs[0].0 as u32;
                for &(idx, p) in &probs {
                    cumsum += p;
                    if sample < cumsum {
                        selected = idx as u32;
                        break;
                    }
                }
                selected
            };

            next_tokens.push(next_token);
        }

        next_tokens
    }

    /// Load state dict from HuggingFace format weights.
    pub fn load_state_dict(
        &mut self,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = self.model.load_state_dict(weights);

        // LM head weight (may be tied to embeddings)
        if let Some(w) = weights.get("lm_head.weight") {
            self.lm_head.weight.update_data(w.clone());
            loaded += 1;
        } else if let Some(w) = weights
            .get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight"))
        {
            // Tie lm_head to embeddings (common in LLaMA)
            self.lm_head.weight.update_data(w.clone());
            loaded += 1;
        }

        println!("LLaMAForCausalLM: Loaded {} total weight tensors", loaded);
        loaded
    }

    /// Load model from HuggingFace Hub.
    ///
    /// # Example
    /// ```rust,ignore
    /// let model = LLaMAForCausalLM::from_pretrained("meta-llama/Llama-2-7b-hf")?;
    /// ```
    pub fn from_pretrained(model_id: &str) -> crate::error::LLMResult<Self> {
        use crate::hf_loader::HFLoader;

        println!("Loading LLaMA from: {}", model_id);

        // Create loader and download weights
        let mut loader = HFLoader::new(model_id)?;

        // Load config
        let config_json = loader.load_config()?;
        let config = crate::hf_loader::parse_llama_config_from_json(&config_json)?;

        // Load tensors
        loader.load_tensors()?;

        // Create model
        let mut model = Self::new(&config);

        // Load weights
        let weights: std::collections::HashMap<String, Tensor<f32>> = loader
            .tensors()
            .iter()
            .map(|(k, v)| {
                let tensor = Tensor::from_vec(v.data.clone(), &v.shape).unwrap();
                (k.clone(), tensor)
            })
            .collect();

        model.load_state_dict(&weights);

        Ok(model)
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
