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
#[derive(Debug)]
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

        // Scaled dot-product attention.
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)).mul_scalar(scale);

        // Causal mask.
        let mask = create_causal_mask(seq_len, total_seq_len, position_offset);
        let attn_weights = attn_weights.add(&Variable::new(mask, false));

        // Softmax + dropout.
        let attn_weights = attn_weights.softmax(-1);
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
    let shape_out = [batch, num_kv_heads * n_rep, seq_len, head_dim];
    let t = Tensor::from_vec(output, &shape_out).unwrap();
    Variable::new(t, x.requires_grad())
}

/// Causal attention mask: `[1, 1, q_len, kv_len]` with `-inf` above the
/// diagonal of the `(q_pos, k_pos)` square, accounting for position offset.
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
#[derive(Debug)]
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

    /// Forward pass: `down(silu(gate(x)) * up(x))`.
    pub fn forward(&self, x: &Variable) -> Variable {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        let hidden = gate.mul(&up);
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
#[derive(Debug)]
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

    /// Forward pass with optional KV-cache.
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
        let hidden_states = residual.add(&hidden_states);

        // MLP with pre-norm.
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

        // Embedding lookup needs Variable<f32> input (Embedding::forward's
        // signature). Convert u32 ids → f32 Variable the same way llama.rs
        // does; Embedding's internals downcast back to an index.
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var = Variable::new(Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(), false);
        let mut hidden_states = self.embed_tokens.forward(&ids_var);

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
