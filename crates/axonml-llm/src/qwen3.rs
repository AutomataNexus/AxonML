//! Qwen3 — Alibaba's third-generation dense LLM.
//!
//! Architecture is LLaMA-style (RoPE + GQA + SwiGLU + RMSNorm) with ONE
//! additional feature: **QK-norm**. Before RoPE, each query and key head
//! gets a per-head RMSNorm applied over its `head_dim` axis, with a
//! shared `[head_dim]` weight broadcast across all heads. The innovation
//! is from Gemma 2 / Grok, adopted by Qwen3 family (0.6B / 1.7B / 4B /
//! 8B / 14B / 32B).
//!
//! `Qwen3Config::head_dim` is **independent** of `hidden_size /
//! num_attention_heads` (Qwen3 decouples them — e.g. Qwen3-0.6B has
//! hidden=1024, num_heads=16, head_dim=128 → attention-dim=2048 ≠ hidden).
//!
//! Everything else (RMSNorm, RotaryEmbedding, MLP with SwiGLU, decoder
//! layer pattern, causal LM head) is identical to LLaMA and reused from
//! the `llama` module.
//!
//! # File
//! `crates/axonml-llm/src/qwen3.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Disclaimer
//! Use at own risk.

use axonml_autograd::Variable;
use axonml_nn::{Dropout, Embedding, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use crate::attention::{KVCache, LayerKVCache};
use crate::llama::{RMSNorm, RotaryEmbedding};

// =============================================================================
// Qwen3 Configuration
// =============================================================================

/// Configuration for Qwen3 models.
///
/// Unlike LLaMA, Qwen3 allows `head_dim` to be independent of
/// `hidden_size / num_attention_heads`, so it's a first-class field.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    /// Vocabulary size (152064 for Qwen3 family).
    pub vocab_size: usize,
    /// Hidden / embedding dimension.
    pub hidden_size: usize,
    /// MLP intermediate size (SwiGLU).
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads (Q heads).
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA; less than num_attention_heads).
    pub num_key_value_heads: usize,
    /// Per-head dimension — independent of hidden_size in Qwen3.
    pub head_dim: usize,
    /// Maximum sequence length (context window).
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE theta (base for rotary embeddings).
    pub rope_theta: f32,
    /// Attention dropout.
    pub attention_dropout: f32,
    /// Hidden dropout.
    pub hidden_dropout: f32,
    /// Whether to tie LM head weights to token embeddings. Qwen3-0.6B /
    /// 1.7B / 4B tie; larger variants do not.
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    /// Qwen3-0.6B configuration.
    pub fn qwen3_0_6b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            tie_word_embeddings: true,
        }
    }

    /// Qwen3-1.7B configuration.
    pub fn qwen3_1_7b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            tie_word_embeddings: true,
        }
    }

    /// Qwen3-4B configuration.
    pub fn qwen3_4b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            tie_word_embeddings: true,
        }
    }

    /// Tiny Qwen3 for unit-test smoke checks.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1024,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            tie_word_embeddings: true,
        }
    }

    /// Total dimension of the query projection (num_heads × head_dim).
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Total dimension of the key / value projections (num_kv_heads × head_dim).
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

// =============================================================================
// Qwen3 Attention — LLaMA-style with QK-norm
// =============================================================================

/// Qwen3 multi-head attention with QK-norm.
///
/// Per-head RMSNorm is applied to Q and K after projection and before
/// RoPE. The norm weight is `[head_dim]` and is broadcast across every
/// head (same weight for Q across all n_heads, same weight for K across
/// all n_kv_heads). This matches Qwen3's published architecture.
#[derive(Debug, Clone)]
pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Per-head RMSNorm for queries (weight shape `[head_dim]`).
    q_norm: RMSNorm,
    /// Per-head RMSNorm for keys (weight shape `[head_dim]`).
    k_norm: RMSNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_dropout: Dropout,
}

impl Qwen3Attention {
    /// Create new Qwen3 attention layer.
    pub fn new(config: &Qwen3Config) -> Self {
        let q_hidden = config.q_dim();
        let kv_hidden = config.kv_dim();

        // Qwen3 projections carry no bias (per the HF reference config).
        Self {
            q_proj: Linear::with_bias(config.hidden_size, q_hidden, false),
            k_proj: Linear::with_bias(config.hidden_size, kv_hidden, false),
            v_proj: Linear::with_bias(config.hidden_size, kv_hidden, false),
            o_proj: Linear::with_bias(q_hidden, config.hidden_size, false),
            q_norm: RMSNorm::new(config.head_dim, config.rms_norm_eps),
            k_norm: RMSNorm::new(config.head_dim, config.rms_norm_eps),
            rotary_emb: RotaryEmbedding::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
            ),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            attn_dropout: Dropout::new(config.attention_dropout),
        }
    }

    /// Projection / per-head-norm accessors (read-only) — for a sub-bit converter.
    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }
    /// The key projection.
    pub fn k_proj(&self) -> &Linear {
        &self.k_proj
    }
    /// The value projection.
    pub fn v_proj(&self) -> &Linear {
        &self.v_proj
    }
    /// The output projection.
    pub fn o_proj(&self) -> &Linear {
        &self.o_proj
    }
    /// The per-head query RMSNorm.
    pub fn q_norm(&self) -> &RMSNorm {
        &self.q_norm
    }
    /// The per-head key RMSNorm.
    pub fn k_norm(&self) -> &RMSNorm {
        &self.k_norm
    }

    /// Forward pass with optional KV-cache.
    ///
    /// Difference from `LLaMAAttention::forward_with_cache`: Q and K pass
    /// through `q_norm` / `k_norm` (per-head RMSNorm) between the reshape-
    /// to-heads step and RoPE. Every other detail — GQA repeat, causal
    /// mask, softmax, output projection — is identical.
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

        // Project Q, K, V.
        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);

        // Reshape for multi-head attention.
        // [B, T, n_heads * head_dim] → [B, T, n_heads, head_dim]
        //                            → [B, n_heads, T, head_dim]
        let q = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);

        // QK-norm — the only architectural difference from LLaMA. RMSNorm
        // normalizes over the last axis (head_dim), so calling it on
        // [B, n_heads, T, head_dim] applies per-head per-token norm with
        // the shared `[head_dim]` weight broadcast across every position.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Apply rotary embeddings — same split-halves convention as LLaMA.
        let (q, k) = self.rotary_emb.apply(&q, &k, position_offset);

        // KV-cache update.
        let (k, v, total_seq_len) = if let Some(cache) = kv_cache {
            let (cached_k, cached_v) = cache.update(&k.data(), &v.data());
            let tot = cached_k.shape()[2];
            (
                Variable::new(cached_k, false),
                Variable::new(cached_v, false),
                tot,
            )
        } else {
            (k, v, seq_len)
        };

        // Repeat KV heads for grouped-query attention.
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            (repeat_kv(&k, repeat), repeat_kv(&v, repeat))
        } else {
            (k, v)
        };

        // Scaled dot-product attention with fused mul_scalar + causal-mask +
        // softmax. Three separate autograd ops (+ the CPU-built mask H2D)
        // collapse into one kernel launch; backward is one kernel too
        // (SoftmaxCausalScaledBackward, no MulScalarBackward/AddBackward).
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose(2, 3));
        let attn_weights =
            scores.softmax_causal_scaled(seq_len, total_seq_len, position_offset, scale);

        // Dropout (still a separate op — training-only).
        let attn_weights = self.attn_dropout.forward(&attn_weights);

        // Compute output and project back to hidden.
        let attn_output = attn_weights.matmul(&v);
        let attn_output = attn_output.transpose(1, 2).reshape(&[
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ]);

        self.o_proj.forward(&attn_output)
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params.extend(self.q_norm.parameters());
        params.extend(self.k_norm.parameters());
        params
    }

    /// Load weights from state dict using HuggingFace naming.
    pub fn load_weights(
        &mut self,
        prefix: &str,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = 0;

        if let Some(w) = weights.get(&format!("{prefix}.q_proj.weight")) {
            self.q_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.k_proj.weight")) {
            self.k_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.v_proj.weight")) {
            self.v_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.o_proj.weight")) {
            self.o_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.q_norm.weight")) {
            self.q_norm.load_weight(w);
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.k_norm.weight")) {
            self.k_norm.load_weight(w);
            loaded += 1;
        }

        loaded
    }
}

// =============================================================================
// Helpers shared with LLaMA (free-standing to avoid crossing trait boundaries)
// =============================================================================

/// Repeat KV heads for grouped-query attention.
///
/// Input `[B, num_kv_heads, T, head_dim]` → output `[B, num_kv_heads * n_rep, T, head_dim]`.
/// Non-graph-preserving version — sufficient for training where we only need
/// the forward pass to be correct and the backward pass goes through the
/// standard tensor ops. If you need gradient-aware repeat_kv, the LLaMA
/// module has `RepeatKVBackward`; wire that in if training with GQA-ratio
/// changes becomes a live concern.
fn repeat_kv(x: &Variable, n_rep: usize) -> Variable {
    if n_rep == 1 {
        return x.clone();
    }
    let data = x.data();
    let shape = data.shape();
    let batch = shape[0];
    let num_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // GPU fast path: one kernel, no D2H / H2D.
    // Backward for GQA isn't gradient-tracked here (matches the existing
    // un-fused CPU path — see comment above this function).
    let t = data.repeat_kv(batch, num_kv_heads, n_rep, seq_len, head_dim);
    Variable::new(t, x.requires_grad())
}

/// Causal attention mask: `[1, 1, q_len, kv_len]` with `-inf` above the
/// diagonal of the `(q_pos, k_pos)` square, accounting for position offset.
///
/// Kept for reference / tests. The live forward path now uses the fused
/// `Variable::softmax_causal_scaled`, which bakes the mask into the softmax
/// kernel and never materializes this tensor.
#[allow(dead_code)]
fn create_causal_mask(q_len: usize, kv_len: usize, offset: usize) -> Tensor<f32> {
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

// =============================================================================
// Qwen3 MLP (SwiGLU, bias-free — matches the Qwen3 HF reference)
// =============================================================================

/// Qwen3 MLP: SwiGLU with bias-free projections. Structurally identical
/// to `LLaMAMLP` but uses `Linear::with_bias(..., false)` everywhere so
/// the parameter count and tensor names line up 1:1 with Qwen3 GGUFs.
#[derive(Debug, Clone)]
pub struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MLP {
    /// Create new Qwen3 MLP from a config.
    pub fn new(cfg: &Qwen3Config) -> Self {
        Self {
            gate_proj: Linear::with_bias(cfg.hidden_size, cfg.intermediate_size, false),
            up_proj: Linear::with_bias(cfg.hidden_size, cfg.intermediate_size, false),
            down_proj: Linear::with_bias(cfg.intermediate_size, cfg.hidden_size, false),
        }
    }

    /// Projection accessors (read-only) — for a sub-bit converter.
    pub fn gate_proj(&self) -> &Linear {
        &self.gate_proj
    }
    /// The up projection.
    pub fn up_proj(&self) -> &Linear {
        &self.up_proj
    }
    /// The down projection.
    pub fn down_proj(&self) -> &Linear {
        &self.down_proj
    }

    /// Forward pass: `down(silu(gate(x)) * up(x))`. Uses the fused SwiGLU op
    /// (`silu + elementwise mul` collapsed into one kernel forward + one
    /// kernel backward).
    pub fn forward(&self, x: &Variable) -> Variable {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let hidden = gate.swiglu(&up);
        self.down_proj.forward(&hidden)
    }

    /// Trainable parameters: gate, up, and down projections.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.gate_proj.parameters());
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }

    /// Load MLP weights from a flat `{prefix}.{proj}.weight` map; returns the
    /// number of projections actually populated (gate/up/down).
    pub fn load_weights(
        &mut self,
        prefix: &str,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = 0;
        if let Some(w) = weights.get(&format!("{prefix}.gate_proj.weight")) {
            self.gate_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.up_proj.weight")) {
            self.up_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.down_proj.weight")) {
            self.down_proj.weight.update_data(w.clone());
            loaded += 1;
        }
        loaded
    }
}

// =============================================================================
// Qwen3 Decoder Layer
// =============================================================================

/// Single Qwen3 transformer decoder layer: pre-norm attention + pre-norm MLP.
#[derive(Debug, Clone)]
pub struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen3DecoderLayer {
    /// Create new decoder layer.
    pub fn new(config: &Qwen3Config) -> Self {
        Self {
            self_attn: Qwen3Attention::new(config),
            mlp: Qwen3MLP::new(config),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        }
    }

    /// Self-attention block (read-only) — for a sub-bit converter.
    pub fn self_attn(&self) -> &Qwen3Attention {
        &self.self_attn
    }

    /// MLP block (read-only).
    pub fn mlp(&self) -> &Qwen3MLP {
        &self.mlp
    }

    /// Input RMSNorm (read-only).
    pub fn input_layernorm(&self) -> &RMSNorm {
        &self.input_layernorm
    }

    /// Post-attention RMSNorm (read-only).
    pub fn post_attention_layernorm(&self) -> &RMSNorm {
        &self.post_attention_layernorm
    }

    /// Forward pass with optional KV-cache.
    ///
    /// Uses the fused `add_rmsnorm_split` op on the post-attention residual
    /// path. The split form returns both `normed = RMSNorm(residual + attn_out)`
    /// (feeding the MLP) and `sum = residual + attn_out` (the un-normalized
    /// residual that gets re-added to the MLP output at layer exit). Collapses
    /// the un-fused `broadcast_add + rms_norm_batched` kernel pair into one
    /// kernel launch per layer on the forward path, plus drops one AddBackward
    /// on the backward path (its job is merged into AddRMSNormBackward).
    pub fn forward_with_cache(
        &self,
        hidden_states: &Variable,
        kv_cache: Option<&mut KVCache>,
        position_offset: usize,
    ) -> Variable {
        // Self attention with pre-norm.
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states =
            self.self_attn
                .forward_with_cache(&hidden_states, kv_cache, position_offset);

        // Fused (residual + attn_out) → post_attention_layernorm.
        // `normed` goes into the MLP; `mlp_residual` is the raw sum used for
        // the layer-exit add.
        let (normed, mlp_residual) = residual.add_rmsnorm_split(
            &hidden_states,
            &self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        );

        let mlp_out = self.mlp.forward(&normed);
        mlp_residual.add(&mlp_out)
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
        loaded += self
            .self_attn
            .load_weights(&format!("{prefix}.self_attn"), weights);
        loaded += self.mlp.load_weights(&format!("{prefix}.mlp"), weights);
        if let Some(w) = weights.get(&format!("{prefix}.input_layernorm.weight")) {
            self.input_layernorm.load_weight(w);
            loaded += 1;
        }
        if let Some(w) = weights.get(&format!("{prefix}.post_attention_layernorm.weight")) {
            self.post_attention_layernorm.load_weight(w);
            loaded += 1;
        }
        loaded
    }
}

// =============================================================================
// Qwen3 Model
// =============================================================================

/// Qwen3 base model (no LM head).
#[derive(Debug)]
pub struct Qwen3 {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RMSNorm,
    config: Qwen3Config,
}

impl Qwen3 {
    /// Create new Qwen3 model.
    pub fn new(config: &Qwen3Config) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(config));
        }
        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            config: config.clone(),
        }
    }

    /// Forward pass taking raw token IDs.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        self.forward_with_cache(input_ids, None).0
    }

    /// Forward with KV-cache for incremental decoding.
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor<u32>,
        kv_cache: Option<&mut LayerKVCache>,
    ) -> (Variable, usize) {
        let position_offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);

        // Embedding lookup needs Variable<f32> input. Convert u32 ids → f32
        // and move to the model's device so the entire forward stays on GPU.
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let mut ids_tensor = Tensor::from_vec(ids_f32, input_ids.shape()).unwrap();
        let model_device = self
            .embed_tokens
            .parameters()
            .first()
            .map(|p| p.data().device())
            .unwrap_or(axonml_core::Device::Cpu);
        if !matches!(model_device, axonml_core::Device::Cpu) {
            ids_tensor = ids_tensor.to_device(model_device).unwrap();
        }
        let ids_var = Variable::new(ids_tensor, false);
        let mut hidden_states = self.embed_tokens.forward(&ids_var);

        if let Some(cache) = kv_cache {
            for (i, layer) in self.layers.iter().enumerate() {
                let layer_cache = cache.get_mut(i);
                hidden_states =
                    layer.forward_with_cache(&hidden_states, layer_cache, position_offset);
            }
        } else if crate::llama::checkpoint_layers() {
            // ── AXONML_CKPT_LAYERS=1: recompute each layer's activations in backward instead of holding all of
            //    them — peak memory O(1 layer) instead of O(layers), one extra forward per layer ──
            for layer in &self.layers {
                let l = layer.clone();
                hidden_states = axonml_autograd::checkpoint_with_params(
                    move |h| l.forward_with_cache(h, None, position_offset),
                    &hidden_states,
                );
            }
        } else {
            for layer in &self.layers {
                hidden_states = layer.forward_with_cache(&hidden_states, None, position_offset);
            }
        }

        let hidden_states = self.norm.forward(&hidden_states);
        (hidden_states, position_offset)
    }

    /// Create a KV-cache sized for this model's layers.
    pub fn create_kv_cache(&self, batch_size: usize) -> LayerKVCache {
        LayerKVCache::new(
            self.config.num_hidden_layers,
            batch_size,
            self.config.num_key_value_heads,
            self.config.max_position_embeddings,
            self.config.head_dim,
        )
    }

    /// Get config.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Token embedding (read-only).
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Decoder layers (read-only) — for a sub-bit converter to VQ-quantize projections.
    pub fn layers(&self) -> &[Qwen3DecoderLayer] {
        &self.layers
    }

    /// Final norm (read-only).
    pub fn norm(&self) -> &RMSNorm {
        &self.norm
    }

    /// Capture the per-layer INPUT hidden states of a full fp forward.
    ///
    /// Returns `[H_0, H_1, …, H_L]` (`num_hidden_layers + 1` tensors), where
    /// `H_0` is the token-embedding output (the input to layer 0), `H_i` is the
    /// input to layer `i`, and `H_L` is the output of the last decoder layer
    /// (the input to the final RMSNorm). This is exactly the teacher trace that
    /// a cascade-distillation trainer consumes (`[H0..H_depth]`,
    /// `depth = num_hidden_layers`): each sub-bit student layer `i` is distilled
    /// to reproduce `H_{i+1}` from its own realized input. Runs under `no_grad`
    /// with no KV cache (`position_offset = 0`), so `input_ids` is a single
    /// prompt `[1, T]` (or `[B, T]`).
    pub fn layer_input_hiddens(&self, input_ids: &Tensor<u32>) -> Vec<Tensor<f32>> {
        axonml_autograd::no_grad(|| {
            let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
            let mut ids_tensor = Tensor::from_vec(ids_f32, input_ids.shape()).unwrap();
            let model_device = self
                .embed_tokens
                .parameters()
                .first()
                .map(|p| p.data().device())
                .unwrap_or(axonml_core::Device::Cpu);
            if !matches!(model_device, axonml_core::Device::Cpu) {
                ids_tensor = ids_tensor.to_device(model_device).unwrap();
            }
            let ids_var = Variable::new(ids_tensor, false);
            let mut hidden_states = self.embed_tokens.forward(&ids_var);

            let mut hiddens = Vec::with_capacity(self.layers.len() + 1);
            hiddens.push(hidden_states.data());
            for layer in &self.layers {
                hidden_states = layer.forward_with_cache(&hidden_states, None, 0);
                hiddens.push(hidden_states.data());
            }
            hiddens
        })
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.embed_tokens.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        params
    }

    /// Load weights from state dict using HuggingFace naming convention.
    pub fn load_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = 0;
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.embed_tokens.weight.update_data(w.clone());
            loaded += 1;
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            loaded += layer.load_weights(&format!("model.layers.{i}"), weights);
        }
        if let Some(w) = weights.get("model.norm.weight") {
            self.norm.load_weight(w);
            loaded += 1;
        }
        loaded
    }
}

impl Module for Qwen3 {
    fn forward(&self, input: &Variable) -> Variable {
        // Treat input.data() as token IDs cast to f32 (Module trait's
        // forward takes a Variable). Prefer `forward_ids` when you have a
        // `Tensor<u32>` directly — that's what training/inference loops use.
        let input_data = input.data();
        let shape: Vec<usize> = input_data.shape().to_vec();
        let ids: Vec<u32> = input_data.to_vec().iter().map(|&x| x as u32).collect();
        let input_ids = Tensor::from_vec(ids, &shape).unwrap();
        self.forward_ids(&input_ids)
    }

    fn parameters(&self) -> Vec<Parameter> {
        Qwen3::parameters(self)
    }
}

// =============================================================================
// Qwen3 For Causal LM
// =============================================================================

/// Qwen3 with language modeling head on top.
#[derive(Debug)]
pub struct Qwen3ForCausalLM {
    model: Qwen3,
    /// LM head. Tied to embed_tokens when `config.tie_word_embeddings` is
    /// true (standard for Qwen3-0.6B / 1.7B / 4B). The tied case is
    /// implemented as a shared weight tensor in `load_weights`.
    lm_head: Linear,
}

impl Qwen3ForCausalLM {
    /// Create a new Qwen3 causal-LM wrapper.
    pub fn new(config: &Qwen3Config) -> Self {
        Self {
            model: Qwen3::new(config),
            lm_head: Linear::new(config.hidden_size, config.vocab_size),
        }
    }

    /// Forward returning logits `[B, T, vocab_size]`.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        let hidden = self.model.forward_ids(input_ids);
        self.lm_head.forward(&hidden)
    }

    /// Teacher hidden trace for cascade distillation
    /// (see [`Qwen3::layer_input_hiddens`]).
    pub fn layer_input_hiddens(&self, input_ids: &Tensor<u32>) -> Vec<Tensor<f32>> {
        self.model.layer_input_hiddens(input_ids)
    }

    /// Forward with KV-cache returning logits.
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor<u32>,
        kv_cache: Option<&mut LayerKVCache>,
    ) -> Variable {
        let (hidden, _pos) = self.model.forward_with_cache(input_ids, kv_cache);
        self.lm_head.forward(&hidden)
    }

    /// Create a KV-cache for autoregressive decoding.
    pub fn create_kv_cache(&self, batch_size: usize) -> LayerKVCache {
        self.model.create_kv_cache(batch_size)
    }

    /// Config accessor.
    pub fn config(&self) -> &Qwen3Config {
        self.model.config()
    }

    /// Base model (read-only) — exposes layers/projections/norms for a sub-bit
    /// converter that VQ-quantizes the projections while reusing the fp math.
    pub fn model(&self) -> &Qwen3 {
        &self.model
    }

    /// LM head (read-only).
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    /// Get parameters (combined base model + LM head).
    ///
    /// Always includes the LM-head weight, even when `tie_word_embeddings` is
    /// true. The tying performed in `load_weights` (via `update_data`) only
    /// copies the embedding tensor's *data* into the LM head's Variable — they
    /// stay as distinct Parameters backed by separate storage. A caller iterating
    /// parameters to move them to a device would otherwise leave the LM-head
    /// weight stranded on CPU. `Parameter::to_device` is idempotent, so exposing
    /// it twice under a true alias (future refactor) would not hurt either.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.model.parameters();
        params.extend(self.lm_head.parameters());
        params
    }

    // ── per-domain modules: a rank-r adapter on every projection, trained alone against a frozen base ──

    /// Share the embedding Parameter with the LM head instead of holding a second copy. `load_weights`
    /// copies the embedding tensor into a separate LM-head Parameter, so a tied model costs
    /// `vocab x hidden` twice on the device — 2.5 GB of a 12 GB card at Qwen3-1.7B.
    pub fn tie_lm_head(&mut self) {
        self.lm_head.weight = self.model.embed_tokens.weight.clone();
    }

    /// Attach an adapter to all seven projections in every decoder layer and return its parameters.
    /// Attaching is an exact identity (each `b` is zero), so the model is unchanged until training.
    pub fn attach_lora(&mut self, rank: usize, alpha: f32) -> Vec<Parameter> {
        let mut ps = Vec::new();
        for l in &mut self.model.layers {
            for lin in [
                &mut l.self_attn.q_proj,
                &mut l.self_attn.k_proj,
                &mut l.self_attn.v_proj,
                &mut l.self_attn.o_proj,
                &mut l.mlp.gate_proj,
                &mut l.mlp.up_proj,
                &mut l.mlp.down_proj,
            ] {
                ps.extend(lin.attach_lora(rank, alpha));
            }
        }
        ps
    }

    /// Attach LoRA to layers `from..` only. Lower layers keep no trainable
    /// tensor at all, so when the embedding and every other input is frozen the
    /// backward pass stops at the lowest adapted layer instead of walking the
    /// whole stack. Additive: `attach_lora` is unchanged.
    pub fn attach_lora_from(&mut self, rank: usize, alpha: f32, from: usize) -> Vec<Parameter> {
        let mut ps = Vec::new();
        for l in self.model.layers.iter_mut().skip(from) {
            for lin in [
                &mut l.self_attn.q_proj,
                &mut l.self_attn.k_proj,
                &mut l.self_attn.v_proj,
                &mut l.self_attn.o_proj,
                &mut l.mlp.gate_proj,
                &mut l.mlp.up_proj,
                &mut l.mlp.down_proj,
            ] {
                ps.extend(lin.attach_lora(rank, alpha));
            }
        }
        ps
    }

    /// Freeze every base tensor — projections, embedding and LM head — so only the adapters train.
    /// The embedding and head matter here in a way they do not for a 250M model: at Qwen3-1.7B their
    /// gradients alone are 2.5 GB.
    pub fn freeze_lora_base(&mut self) {
        for l in &mut self.model.layers {
            for lin in [
                &mut l.self_attn.q_proj,
                &mut l.self_attn.k_proj,
                &mut l.self_attn.v_proj,
                &mut l.self_attn.o_proj,
                &mut l.mlp.gate_proj,
                &mut l.mlp.up_proj,
                &mut l.mlp.down_proj,
            ] {
                lin.freeze_base();
            }
        }
        let e = self.model.embed_tokens.weight.data();
        self.model.embed_tokens.weight = Parameter::from_variable(Variable::new(e, false));
        let h = self.lm_head.weight.data();
        self.lm_head.weight = Parameter::from_variable(Variable::new(h, false));
    }

    /// Adapter parameters keyed by tensor name, for saving or loading one module on its own.
    pub fn lora_named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        for (i, l) in self.model.layers.iter().enumerate() {
            for (tag, lin) in [
                ("self_attn.q_proj", &l.self_attn.q_proj),
                ("self_attn.k_proj", &l.self_attn.k_proj),
                ("self_attn.v_proj", &l.self_attn.v_proj),
                ("self_attn.o_proj", &l.self_attn.o_proj),
                ("mlp.gate_proj", &l.mlp.gate_proj),
                ("mlp.up_proj", &l.mlp.up_proj),
                ("mlp.down_proj", &l.mlp.down_proj),
            ] {
                for (k, p) in lin.lora_parameters().into_iter().enumerate() {
                    out.push((
                        format!(
                            "model.layers.{i}.{tag}.lora_{}",
                            if k == 0 { "a" } else { "b" }
                        ),
                        p,
                    ));
                }
            }
        }
        out
    }

    /// Fold every adapter into its base weight and drop it. Required before an export that walks
    /// `parameters()` positionally — an attached adapter adds two tensors per projection.
    pub fn merge_lora(&mut self) {
        for l in &mut self.model.layers {
            for lin in [
                &mut l.self_attn.q_proj,
                &mut l.self_attn.k_proj,
                &mut l.self_attn.v_proj,
                &mut l.self_attn.o_proj,
                &mut l.mlp.gate_proj,
                &mut l.mlp.up_proj,
                &mut l.mlp.down_proj,
            ] {
                lin.merge_lora();
            }
        }
    }

    /// Load weights from state dict. Honors `tie_word_embeddings` by
    /// aliasing the LM head weight to `model.embed_tokens.weight` if set.
    pub fn load_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Tensor<f32>>,
    ) -> usize {
        let mut loaded = self.model.load_weights(weights);
        if self.config().tie_word_embeddings {
            // Tie: reuse embed_tokens weight as the LM head projection.
            let embed = self.model.embed_tokens.weight.data();
            self.lm_head.weight.update_data(embed);
        } else if let Some(w) = weights.get("lm_head.weight") {
            self.lm_head.weight.update_data(w.clone());
            loaded += 1;
        }
        loaded
    }
}

impl Module for Qwen3ForCausalLM {
    fn forward(&self, input: &Variable) -> Variable {
        let hidden = self.model.forward(input);
        self.lm_head.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Parameter> {
        Qwen3ForCausalLM::parameters(self)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_config_0_6b() {
        let c = Qwen3Config::qwen3_0_6b();
        assert_eq!(c.vocab_size, 151936);
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128);
        // Key assertion: q_dim = 2048 ≠ hidden_size = 1024 (Qwen3 decouples these).
        assert_eq!(c.q_dim(), 2048);
        assert_eq!(c.kv_dim(), 1024);
        assert!(c.tie_word_embeddings);
    }

    #[test]
    fn test_qwen3_config_4b() {
        let c = Qwen3Config::qwen3_4b();
        assert_eq!(c.hidden_size, 2560);
        assert_eq!(c.num_hidden_layers, 36);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.q_dim(), 32 * 128);
    }

    #[test]
    fn test_qwen3_tiny_forward_shapes() {
        // Just verify the module graph wires together at tiny size. We
        // don't assert on output values — that's for the full-fidelity
        // distillation runner to validate against a reference model.
        let cfg = Qwen3Config::tiny();
        let model = Qwen3::new(&cfg);
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[1, 4]).unwrap();
        let out = model.forward_ids(&ids);
        let s = out.data().shape().to_vec();
        assert_eq!(s, vec![1, 4, cfg.hidden_size]);
    }

    #[test]
    fn test_qwen3_causal_lm_tiny_forward_shapes() {
        let cfg = Qwen3Config::tiny();
        let model = Qwen3ForCausalLM::new(&cfg);
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[1, 4]).unwrap();
        let logits = model.forward_ids(&ids);
        let s = logits.data().shape().to_vec();
        assert_eq!(s, vec![1, 4, cfg.vocab_size]);
    }
}
