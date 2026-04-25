//! Trident - 1.58-bit Ternary Weight Small Language Model
//!
//! A BitNet b1.58-style transformer language model using ternary weights {-1, 0, +1}.
//! All linear projections in the transformer blocks use TernaryLinear, while
//! embeddings and the LM head remain in full precision (fp32).
//!
//! Architecture:
//!   Token Embedding (fp32) -> [TridentBlock x N] -> RMSNorm -> LM Head (fp32)
//!
//! Each TridentBlock:
//!   RMSNorm -> TernaryLinear(QKV) -> Multi-Head Attention -> TernaryLinear(out_proj) -> residual
//!   RMSNorm -> TernaryLinear(up) -> SiLU -> TernaryLinear(down) -> residual
//!
//! Key properties:
//! - 16x memory compression for transformer weights (2-bit vs 32-bit)
//! - Inference matmul reduces to addition/subtraction (no FP multiply for weights)
//! - Full-precision activations throughout for accuracy
//! - Trained with Straight-Through Estimator (STE) on shadow weights
//!
//! # File
//! `crates/axonml-llm/src/trident.rs`
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

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, Variable};
use axonml_nn::layers::ternary::TernaryLinear;
use axonml_nn::{Embedding, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use crate::llama::{RMSNorm, RepeatKVBackward, RotaryEmbedding};

// =============================================================================
// Trident Configuration
// =============================================================================

/// Configuration for the Trident 1.58-bit SLM.
#[derive(Debug, Clone)]
pub struct TridentConfig {
    /// Vocabulary size (default: 32000)
    pub vocab_size: usize,
    /// Hidden dimension / model dimension (default: 512)
    pub d_model: usize,
    /// Number of transformer layers (default: 12)
    pub num_layers: usize,
    /// Number of attention heads (default: 8)
    pub num_heads: usize,
    /// Number of key-value heads for grouped-query attention.
    /// Must divide `num_heads`. Equal to `num_heads` for vanilla MHA.
    pub num_kv_heads: usize,
    /// Intermediate size for MLP (default: 4 * d_model)
    pub intermediate_size: usize,
    /// Maximum sequence length (default: 2048)
    pub max_seq_len: usize,
    /// RMSNorm epsilon (default: 1e-6)
    pub rms_norm_eps: f32,
    // ------------------------------------------------------------------
    // BitNet b1.58-compatible architecture flags. Default to `false` so
    // legacy callers (tiny/default_150m/medium) keep the original MHA +
    // SiLU-MLP behavior. `trident_1b` / `trident_3b` turn them on.
    // ------------------------------------------------------------------
    /// Apply rotary position embedding inside attention. When false, positions
    /// are unencoded (matches the original 150M Trident toy).
    pub use_rope: bool,
    /// RoPE base θ. Only read when `use_rope = true` (default 10_000).
    pub rope_theta: f32,
    /// Use ReLU² (`relu(gate)^2 * up`) FFN gating instead of SwiGLU/SiLU.
    /// Matches BitNet b1.58-2B-4T. When false, the MLP falls back to the
    /// legacy 2-linear SiLU stack (`up → SiLU → down`).
    pub use_squared_relu: bool,
    /// Insert the two BitNet-style SubLN (RMSNorm) layers: one before
    /// `o_proj` in attention, one before `ffn_down` in the MLP.
    pub use_sub_ln: bool,
}

impl TridentConfig {
    /// Default 150M-equivalent parameter configuration.
    pub fn default_150m() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            num_layers: 12,
            num_heads: 8,
            num_kv_heads: 8,
            intermediate_size: 2048,
            max_seq_len: 2048,
            rms_norm_eps: 1e-6,
            use_rope: false,
            rope_theta: 10_000.0,
            use_squared_relu: false,
            use_sub_ln: false,
        }
    }

    /// Small configuration for testing.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 256,
            max_seq_len: 128,
            rms_norm_eps: 1e-6,
            use_rope: false,
            rope_theta: 10_000.0,
            use_squared_relu: false,
            use_sub_ln: false,
        }
    }

    /// Medium configuration (~300M equivalent).
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 768,
            num_layers: 16,
            num_heads: 12,
            num_kv_heads: 12,
            intermediate_size: 3072,
            max_seq_len: 2048,
            rms_norm_eps: 1e-6,
            use_rope: false,
            rope_theta: 10_000.0,
            use_squared_relu: false,
            use_sub_ln: false,
        }
    }

    /// 1B-parameter config for Trident-Coder training on The Stack v2.
    ///
    /// Matches the BitNet b1.58 training recipe:
    /// - d_model=2048, 24 layers, 16 Q heads, 4 KV heads (4:1 GQA)
    /// - ReLU²-gated FFN with BitNet SubLN
    /// - RoPE θ=500_000 (LLaMA-3 long-context scaling)
    /// - Context 4096
    ///
    /// NOTE: GQA (num_kv_heads < num_heads) currently severs the autograd
    /// graph on the KV-repeat step (see `TridentAttention::repeat_kv`).
    /// Before the real 1B training run, either (a) move LLaMA's
    /// `RepeatKVBackward` helper into a shared crate and wire it here, or
    /// (b) flip `num_kv_heads` to match `num_heads`. For inference this is
    /// fine (no backward pass).
    pub fn trident_1b(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 2048,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            intermediate_size: 5504,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            use_rope: true,
            rope_theta: 500_000.0,
            use_squared_relu: true,
            use_sub_ln: true,
        }
    }

    /// 3B-parameter config — stub for later scaling. Not trained yet.
    pub fn trident_3b(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 3200,
            num_layers: 26,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 8640,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            use_rope: true,
            rope_theta: 500_000.0,
            use_squared_relu: true,
            use_sub_ln: true,
        }
    }

    /// Smoke-test config (~30M) — tiny 1B-shaped model so local CPU runs of
    /// `train_trident_code --config smoke` exercise the real architecture
    /// (RoPE + ReLU²-gated FFN + SubLN) without taking hours to warm up.
    ///
    /// Uses vanilla MHA (num_kv_heads == num_heads) so autograd flows
    /// cleanly through `repeat_kv` (which would otherwise sever the graph —
    /// see comment in `TridentAttention::repeat_kv`).
    pub fn smoke(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 256,
            num_layers: 4,
            // GQA 4:1 — mirrors `trident_1b`'s shape at toy scale so the
            // smoke test actually exercises `repeat_kv` (with its
            // graph-preserving `RepeatKVBackward`). Before the fix landed
            // this had to be `num_kv_heads == num_heads` to avoid severing
            // the autograd graph.
            num_heads: 8,
            num_kv_heads: 2,
            intermediate_size: 688,
            max_seq_len: 512,
            rms_norm_eps: 1e-5,
            use_rope: true,
            rope_theta: 500_000.0,
            use_squared_relu: true,
            use_sub_ln: true,
        }
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Estimate total parameters (fp32-equivalent).
    pub fn estimated_params(&self) -> usize {
        let embedding = self.vocab_size * self.d_model;
        let lm_head = self.d_model * self.vocab_size;
        let head_dim = self.head_dim();
        let kv_hidden = self.num_kv_heads * head_dim;
        let per_layer = {
            // Q + O on d_model, K + V on kv_hidden (GQA aware)
            let attn = 2 * self.d_model * self.d_model + 2 * self.d_model * kv_hidden;
            // MLP: up + down (always), gate (if squared_relu)
            let mlp_linears = if self.use_squared_relu { 3 } else { 2 };
            let mlp = mlp_linears * self.d_model * self.intermediate_size;
            // RMSNorm weights (2 per layer) + optional sub-norms
            let mut norms = 2 * self.d_model;
            if self.use_sub_ln {
                norms += self.d_model + self.intermediate_size;
            }
            attn + mlp + norms
        };
        embedding + lm_head + self.num_layers * per_layer
    }

    /// Estimate ternary storage in bytes (transformer weights only).
    pub fn ternary_storage_bytes(&self) -> usize {
        let head_dim = self.head_dim();
        let kv_hidden = self.num_kv_heads * head_dim;
        let per_layer = {
            let attn = 2 * self.d_model * self.d_model + 2 * self.d_model * kv_hidden;
            let mlp_linears = if self.use_squared_relu { 3 } else { 2 };
            let mlp = mlp_linears * self.d_model * self.intermediate_size;
            attn + mlp
        };
        let total_ternary_weights = self.num_layers * per_layer;
        let packed_bytes = total_ternary_weights.div_ceil(4);
        let linears_per_block = 4 + if self.use_squared_relu { 3 } else { 2 };
        let scale_bytes = self.num_layers * linears_per_block * 4;
        let mut fp32_bytes = (self.vocab_size * self.d_model
            + self.d_model * self.vocab_size
            + self.num_layers * 2 * self.d_model)
            * 4;
        if self.use_sub_ln {
            fp32_bytes += self.num_layers * (self.d_model + self.intermediate_size) * 4;
        }
        packed_bytes + scale_bytes + fp32_bytes
    }

    /// Estimate fp32 storage in bytes (for comparison).
    pub fn fp32_storage_bytes(&self) -> usize {
        self.estimated_params() * 4
    }
}

// =============================================================================
// Trident Attention
// =============================================================================

/// Multi-head attention with ternary weight projections.
///
/// Uses TernaryLinear for Q, K, V, and output projections.
/// Activations remain in full precision (fp32) throughout.
///
/// Supports:
/// - Grouped-query attention (GQA) when `num_kv_heads < num_heads`
/// - Rotary position embedding (RoPE) when `config.use_rope`
/// - BitNet SubLN (RMSNorm) before the output projection when
///   `config.use_sub_ln`
#[derive(Debug)]
struct TridentAttention {
    /// Query projection (ternary)
    q_proj: TernaryLinear,
    /// Key projection (ternary)
    k_proj: TernaryLinear,
    /// Value projection (ternary)
    v_proj: TernaryLinear,
    /// Output projection (ternary)
    o_proj: TernaryLinear,
    /// Optional RoPE (shared head_dim rotary embedding table)
    rotary_emb: Option<RotaryEmbedding>,
    /// Optional SubLN — RMSNorm(d_model) applied before `o_proj`
    attn_sub_norm: Option<RMSNorm>,
    /// Number of query heads
    num_heads: usize,
    /// Number of KV heads (= num_heads for vanilla MHA, < for GQA)
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Hidden size
    hidden_size: usize,
}

impl TridentAttention {
    fn new(config: &TridentConfig) -> Self {
        let head_dim = config.head_dim();
        let kv_hidden = config.num_kv_heads * head_dim;
        let rotary_emb = if config.use_rope {
            Some(RotaryEmbedding::new(
                head_dim,
                config.max_seq_len,
                config.rope_theta,
            ))
        } else {
            None
        };
        let attn_sub_norm = if config.use_sub_ln {
            Some(RMSNorm::new(config.d_model, config.rms_norm_eps))
        } else {
            None
        };
        Self {
            q_proj: TernaryLinear::with_bias(config.d_model, config.d_model, false),
            k_proj: TernaryLinear::with_bias(config.d_model, kv_hidden, false),
            v_proj: TernaryLinear::with_bias(config.d_model, kv_hidden, false),
            o_proj: TernaryLinear::with_bias(config.d_model, config.d_model, false),
            rotary_emb,
            attn_sub_norm,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            hidden_size: config.d_model,
        }
    }

    fn forward(&self, hidden_states: &Variable) -> Variable {
        let data = hidden_states.data();
        let shape = data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V through ternary layers
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

        // Apply RoPE if configured
        let (q, k) = if let Some(rope) = &self.rotary_emb {
            rope.apply(&q, &k, 0)
        } else {
            (q, k)
        };

        // Expand KV heads for GQA
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            (self.repeat_kv(&k, n_rep), self.repeat_kv(&v, n_rep))
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)).mul_scalar(scale);

        // Causal mask
        let mask = self.create_causal_mask(seq_len);
        let attn_weights = attn_weights.add(&Variable::new(mask, false));

        // Softmax
        let attn_weights = attn_weights.softmax(-1);

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output =
            attn_output
                .transpose(1, 2)
                .reshape(&[batch_size, seq_len, self.hidden_size]);

        // BitNet SubLN: RMSNorm before output projection
        let attn_output = if let Some(norm) = &self.attn_sub_norm {
            norm.forward(&attn_output)
        } else {
            attn_output
        };

        // Output projection (ternary)
        self.o_proj.forward(&attn_output)
    }

    /// Repeat KV heads `n_rep` times along the head dimension for GQA.
    /// Data-only copy (no autograd); acceptable because KV projections
    /// contribute to the attention output via `matmul(Q, K^T)` and
    /// `matmul(W, V)`; gradients flow through those matmuls back into
    /// the repeated tensor. The repeat itself is a view-like
    /// reshape-broadcast; for the training smoke test we use a plain
    /// rebroadcast and let the matmul autograd handle gradient accumulation
    /// via standard broadcasting backward.
    /// Repeat each KV head `n_rep` times along the head axis so GQA Q
    /// heads can each attend to a broadcasted K/V. Graph-preserving:
    /// builds the repeated tensor eagerly (same pattern as `phi.rs` and
    /// `mistral.rs`) and attaches a `RepeatKVBackward` grad fn that sums
    /// the upstream gradient across each group of `n_rep` copies back
    /// into the original KV head. This is what unblocks `trident_1b`'s
    /// GQA 16Q/4KV config for training.
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

        // Device-aware GQA broadcast — `Tensor::repeat_kv` dispatches to
        // `repeat_kv_f32` PTX on GPU tensors and to a contiguous CPU walk
        // otherwise. Eliminates the per-layer CPU round-trip that
        // dominated step time on `Cuda(0)` (smoke=13.8 s/step).
        let output_tensor = data.repeat_kv(batch, num_kv_heads, n_rep, seq_len, head_dim);

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

    fn create_causal_mask(&self, seq_len: usize) -> Tensor<f32> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(mask_data, &[1, 1, seq_len, seq_len]).unwrap()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        if let Some(n) = &self.attn_sub_norm {
            params.extend(n.parameters());
        }
        params
    }

    fn quantize_for_inference(&mut self) {
        self.q_proj.quantize_for_inference();
        self.k_proj.quantize_for_inference();
        self.v_proj.quantize_for_inference();
        self.o_proj.quantize_for_inference();
    }
}

// =============================================================================
// Trident MLP
// =============================================================================

/// Feed-forward MLP with ternary weight projections.
///
/// Two modes, selected by `TridentConfig::use_squared_relu`:
///
/// - **Legacy SiLU (2-linear)**: `up → SiLU → down`. Keeps the original
///   150M toy model unchanged.
/// - **BitNet ReLU² gated (3-linear)**: `gate_act = relu(gate)^2`,
///   `inner = ffn_sub_norm(gate_act * up)`, `out = down(inner)`. Matches
///   Microsoft BitNet b1.58-2B-4T.
#[derive(Debug)]
struct TridentMLP {
    /// Up projection (ternary)
    up_proj: TernaryLinear,
    /// Gate projection — only populated in ReLU² mode (None in legacy SiLU)
    gate_proj: Option<TernaryLinear>,
    /// Down projection (ternary)
    down_proj: TernaryLinear,
    /// Optional SubLN — RMSNorm(intermediate_size) applied before `down_proj`
    ffn_sub_norm: Option<RMSNorm>,
    /// Whether to use the BitNet ReLU²-gated pathway
    use_squared_relu: bool,
}

impl TridentMLP {
    fn new(config: &TridentConfig) -> Self {
        let gate_proj = if config.use_squared_relu {
            Some(TernaryLinear::with_bias(
                config.d_model,
                config.intermediate_size,
                false,
            ))
        } else {
            None
        };
        let ffn_sub_norm = if config.use_sub_ln {
            Some(RMSNorm::new(config.intermediate_size, config.rms_norm_eps))
        } else {
            None
        };
        Self {
            up_proj: TernaryLinear::with_bias(config.d_model, config.intermediate_size, false),
            gate_proj,
            down_proj: TernaryLinear::with_bias(config.intermediate_size, config.d_model, false),
            ffn_sub_norm,
            use_squared_relu: config.use_squared_relu,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        if self.use_squared_relu {
            let gate = self
                .gate_proj
                .as_ref()
                .expect("gate_proj set when use_squared_relu")
                .forward(x);
            let up = self.up_proj.forward(x);
            // squared_relu(gate) = relu(gate)^2
            let gate_act = gate.relu().pow(2.0);
            let gated = gate_act.mul(&up);
            let normed = if let Some(n) = &self.ffn_sub_norm {
                n.forward(&gated)
            } else {
                gated
            };
            self.down_proj.forward(&normed)
        } else {
            let up = self.up_proj.forward(x).silu();
            self.down_proj.forward(&up)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.up_proj.parameters());
        if let Some(g) = &self.gate_proj {
            params.extend(g.parameters());
        }
        params.extend(self.down_proj.parameters());
        if let Some(n) = &self.ffn_sub_norm {
            params.extend(n.parameters());
        }
        params
    }

    fn quantize_for_inference(&mut self) {
        self.up_proj.quantize_for_inference();
        if let Some(g) = &mut self.gate_proj {
            g.quantize_for_inference();
        }
        self.down_proj.quantize_for_inference();
    }
}

// =============================================================================
// Trident Transformer Block
// =============================================================================

/// Single Trident transformer block with ternary weights.
///
/// Architecture (pre-norm):
///   residual + TridentAttention(RMSNorm(x))
///   residual + TridentMLP(RMSNorm(x))
#[derive(Debug)]
struct TridentBlock {
    /// Pre-attention normalization
    attn_norm: RMSNorm,
    /// Multi-head self-attention (ternary projections)
    attention: TridentAttention,
    /// Pre-MLP normalization
    mlp_norm: RMSNorm,
    /// Feed-forward MLP (ternary projections)
    mlp: TridentMLP,
}

impl TridentBlock {
    fn new(config: &TridentConfig) -> Self {
        Self {
            attn_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            attention: TridentAttention::new(config),
            mlp_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            mlp: TridentMLP::new(config),
        }
    }

    fn forward(&self, hidden_states: &Variable) -> Variable {
        // Self-attention with pre-norm + residual
        let residual = hidden_states.clone();
        let normed = self.attn_norm.forward(hidden_states);
        let attn_out = self.attention.forward(&normed);
        let hidden_states = residual.add(&attn_out);

        // MLP with pre-norm + residual
        let residual = hidden_states.clone();
        let normed = self.mlp_norm.forward(&hidden_states);
        let mlp_out = self.mlp.forward(&normed);
        residual.add(&mlp_out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.attn_norm.parameters());
        params.extend(self.attention.parameters());
        params.extend(self.mlp_norm.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    fn quantize_for_inference(&mut self) {
        self.attention.quantize_for_inference();
        self.mlp.quantize_for_inference();
    }
}

// =============================================================================
// Trident Model
// =============================================================================

/// Trident: a 1.58-bit ternary weight small language model.
///
/// Uses BitNet b1.58 ternary quantization for all transformer linear layers,
/// keeping embeddings and the LM head in full precision.
///
/// # Example
/// ```ignore
/// let config = TridentConfig::default_150m();
/// let model = TridentModel::new(&config);
///
/// let input_ids = Tensor::from_vec(vec![1u32, 42, 100], &[1, 3]).unwrap();
/// let logits = model.forward_ids(&input_ids);
/// // logits shape: [1, 3, 32000]
/// ```
pub struct TridentModel {
    /// Token embedding (fp32)
    embed_tokens: Embedding,
    /// Transformer blocks with ternary weights
    blocks: Vec<TridentBlock>,
    /// Final RMSNorm
    final_norm: RMSNorm,
    /// Language model head (fp32)
    lm_head: Linear,
    /// Model configuration
    config: TridentConfig,
}

impl TridentModel {
    /// Create a new Trident model.
    pub fn new(config: &TridentConfig) -> Self {
        let blocks = (0..config.num_layers)
            .map(|_| TridentBlock::new(config))
            .collect();

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.d_model),
            blocks,
            final_norm: RMSNorm::new(config.d_model, config.rms_norm_eps),
            lm_head: Linear::with_bias(config.d_model, config.vocab_size, false),
            config: config.clone(),
        }
    }

    /// Forward pass with token IDs, returning logits.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        // Convert token IDs to float for embedding lookup
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var = Variable::new(Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(), false);

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(&ids_var);

        // Pass through transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        // Final norm
        hidden = self.final_norm.forward(&hidden);

        // LM head
        self.lm_head.forward(&hidden)
    }

    /// Forward pass with loss computation (for training).
    ///
    /// Internally shifts logits and labels for next-token prediction
    /// and computes cross-entropy loss.
    pub fn forward_with_loss(
        &self,
        input_ids: &Tensor<u32>,
        labels: &Tensor<u32>,
    ) -> (Variable, Variable) {
        let logits = self.forward_ids(input_ids);

        let logits_data = logits.data();
        let shape = logits_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let _vocab_size = shape[2];

        if seq_len > 1 {
            // Shift for next-token prediction: logits[..., :-1, :] predicts labels[..., 1:]
            // Use narrow() instead of slice() to preserve the autograd graph
            let shift_logits = logits.narrow(1, 0, seq_len - 1);

            // Build shifted labels
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
            let zero_loss = Variable::new(Tensor::from_vec(vec![0.0f32], &[1]).unwrap(), false);
            (logits, zero_loss)
        }
    }

    /// Cross-entropy loss computation.
    fn cross_entropy_loss(logits: &Variable, labels: &Tensor<u32>) -> Variable {
        let logits_data = logits.data();
        let shape = logits_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        let logits_flat = logits.reshape(&[batch_size * seq_len, vocab_size]);

        let labels_vec = labels.to_vec();
        let valid_labels: Vec<f32> = labels_vec
            .iter()
            .map(|&l| {
                let label = l as usize;
                if label < vocab_size { l as f32 } else { 0.0f32 }
            })
            .collect();
        let target_var = Variable::new(
            Tensor::from_vec(valid_labels, &[batch_size * seq_len]).unwrap(),
            false,
        );

        use axonml_nn::loss::CrossEntropyLoss;
        CrossEntropyLoss::new().compute(&logits_flat, &target_var)
    }

    /// Quantize all ternary layers for inference.
    pub fn quantize_for_inference(&mut self) {
        for block in &mut self.blocks {
            block.quantize_for_inference();
        }
    }

    /// Get average weight sparsity across all ternary layers.
    pub fn average_sparsity(&self) -> f32 {
        let mut total_sparsity = 0.0f32;
        let mut count = 0usize;

        for block in &self.blocks {
            // Attention projections
            total_sparsity += block.attention.q_proj.weight_sparsity();
            total_sparsity += block.attention.k_proj.weight_sparsity();
            total_sparsity += block.attention.v_proj.weight_sparsity();
            total_sparsity += block.attention.o_proj.weight_sparsity();
            count += 4;
            // MLP projections
            total_sparsity += block.mlp.up_proj.weight_sparsity();
            total_sparsity += block.mlp.down_proj.weight_sparsity();
            count += 2;
            if let Some(g) = &block.mlp.gate_proj {
                total_sparsity += g.weight_sparsity();
                count += 1;
            }
        }

        if count > 0 {
            total_sparsity / count as f32
        } else {
            0.0
        }
    }

    /// Get the model configuration.
    pub fn config(&self) -> &TridentConfig {
        &self.config
    }

    /// Report model statistics.
    pub fn report(&self) {
        let param_count: usize = self.parameters().iter().map(|p| p.data().numel()).sum();
        let fp32_mb = self.config.fp32_storage_bytes() as f32 / (1024.0 * 1024.0);
        let ternary_mb = self.config.ternary_storage_bytes() as f32 / (1024.0 * 1024.0);
        let compression = fp32_mb / ternary_mb;

        println!("Trident Model Report");
        println!("====================");
        println!("Layers       : {}", self.config.num_layers);
        println!("d_model      : {}", self.config.d_model);
        println!("Heads        : {}", self.config.num_heads);
        println!("Vocab        : {}", self.config.vocab_size);
        println!("Parameters   : {}", param_count);
        println!("FP32 size    : {:.1} MB", fp32_mb);
        println!("Ternary size : {:.1} MB", ternary_mb);
        println!("Compression  : {:.1}x", compression);
        println!("Sparsity     : {:.1}%", self.average_sparsity() * 100.0);
    }
}

impl Module for TridentModel {
    fn forward(&self, input: &Variable) -> Variable {
        // When called via Module trait, assume input contains token ID floats
        let mut hidden = self.embed_tokens.forward(input);

        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        hidden = self.final_norm.forward(&hidden);
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

    fn name(&self) -> &'static str {
        "TridentModel"
    }
}

impl std::fmt::Debug for TridentModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TridentModel")
            .field("vocab_size", &self.config.vocab_size)
            .field("d_model", &self.config.d_model)
            .field("num_layers", &self.config.num_layers)
            .field("num_heads", &self.config.num_heads)
            .field("intermediate_size", &self.config.intermediate_size)
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
    fn test_trident_config() {
        let config = TridentConfig::default_150m();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_trident_config_storage() {
        let config = TridentConfig::default_150m();
        let fp32 = config.fp32_storage_bytes();
        let ternary = config.ternary_storage_bytes();
        // Ternary should be significantly smaller
        assert!(ternary < fp32);
        println!(
            "FP32: {} bytes, Ternary: {} bytes, Ratio: {:.1}x",
            fp32,
            ternary,
            fp32 as f32 / ternary as f32
        );
    }

    #[test]
    fn test_trident_tiny_forward() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let logits = model.forward_ids(&input_ids);
        assert_eq!(logits.data().shape(), &[2, 2, config.vocab_size]);
    }

    #[test]
    fn test_trident_parameters() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);
        let params = model.parameters();
        assert!(!params.is_empty());

        let total: usize = params.iter().map(|p| p.data().numel()).sum();
        println!("Tiny Trident params: {}", total);
    }

    #[test]
    fn test_trident_forward_with_loss() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let labels = Tensor::from_vec(vec![2u32, 3, 4, 5, 6, 7], &[2, 3]).unwrap();

        let (logits, loss) = model.forward_with_loss(&input_ids, &labels);
        assert_eq!(logits.data().shape(), &[2, 3, config.vocab_size]);
        assert_eq!(loss.data().numel(), 1);

        let loss_val = loss.data().to_vec()[0];
        assert!(loss_val > 0.0, "Loss should be positive, got {}", loss_val);
    }

    #[test]
    fn test_trident_sparsity() {
        let config = TridentConfig::tiny();
        let model = TridentModel::new(&config);
        let sparsity = model.average_sparsity();
        assert!((0.0..=1.0).contains(&sparsity));
    }

    /// Convergence test for the BitNet-shaped architecture: runs 100 opt
    /// steps on tiny synthetic data and checks loss drops meaningfully.
    /// Exercises the full path — RoPE + squared-ReLU-gated FFN + SubLN +
    /// ternary linears — with `num_kv_heads == num_heads` so autograd
    /// flows through the KV projections unimpeded.
    #[test]
    fn test_trident_1b_shape_converges() {
        use axonml_optim::{Adam, Optimizer};
        // Shrunken 1B-shaped config: smoke-sized but same feature switches.
        let config = TridentConfig {
            vocab_size: 64,
            d_model: 48,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 128,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            use_rope: true,
            rope_theta: 10_000.0,
            use_squared_relu: true,
            use_sub_ln: true,
        };
        let model = TridentModel::new(&config);
        let mut optimizer = Adam::new(model.parameters(), 3e-3);

        // Deterministic tiny corpus: 3 repeating patterns of length 8.
        let seq_len = 8usize;
        let batch_size = 4usize;
        let patterns: Vec<Vec<u32>> = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![9, 10, 11, 12, 13, 14, 15, 16],
            vec![17, 18, 19, 20, 21, 22, 23, 24],
        ];

        let mut flat_batch = Vec::with_capacity(batch_size * seq_len);
        for b in 0..batch_size {
            flat_batch.extend_from_slice(&patterns[b % patterns.len()]);
        }
        let ids = Tensor::from_vec(flat_batch.clone(), &[batch_size, seq_len]).unwrap();
        let labels = Tensor::from_vec(flat_batch, &[batch_size, seq_len]).unwrap();

        // Initial loss — ln(vocab) ≈ 4.16 for vocab=64 if uniform.
        let (_, loss0) = model.forward_with_loss(&ids, &labels);
        let start_loss = loss0.data().to_vec()[0];

        let mut last_loss = start_loss;
        for _ in 0..100 {
            optimizer.zero_grad();
            let (_, loss) = model.forward_with_loss(&ids, &labels);
            last_loss = loss.data().to_vec()[0];
            loss.backward();
            optimizer.step();
        }

        println!("[trident convergence] start_loss={start_loss:.4} end_loss={last_loss:.4}");
        // On a tiny 2-layer model with 3 fixed patterns, 100 steps of
        // Adam 3e-3 should push loss down by at least ~30% from the
        // near-uniform starting point. Very generous bound to avoid
        // flakiness on RNG-sensitive inits; the key signal is that the
        // gradient flows and loss moves monotonically.
        assert!(
            last_loss < start_loss * 0.7,
            "Trident did not converge: start={start_loss:.4}, end={last_loss:.4}"
        );
        assert!(last_loss.is_finite(), "Loss went NaN/Inf: {last_loss}");
    }

    #[test]
    fn test_trident_1b_config_shapes() {
        let cfg = TridentConfig::trident_1b(32_000);
        assert_eq!(cfg.d_model, 2048);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.intermediate_size, 5504);
        assert_eq!(cfg.max_seq_len, 4096);
        assert!((cfg.rope_theta - 500_000.0).abs() < 1e-3);
        assert!(cfg.use_rope);
        assert!(cfg.use_squared_relu);
        assert!(cfg.use_sub_ln);
        // Rough param count check — should be in the 1B ballpark.
        let n = cfg.estimated_params();
        assert!(
            (800_000_000..1_500_000_000).contains(&n),
            "1B config estimated_params={n} outside [0.8B, 1.5B]"
        );
    }

    #[test]
    fn test_trident_smoke_forward() {
        let cfg = TridentConfig::smoke(64);
        let model = TridentModel::new(&cfg);
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[1, 4]).unwrap();
        let logits = model.forward_ids(&ids);
        assert_eq!(logits.data().shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_trident_quantize_inference() {
        let config = TridentConfig::tiny();
        let mut model = TridentModel::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], &[1, 3]).unwrap();

        // Training forward
        let logits_train = model.forward_ids(&input_ids);

        // Quantize for inference
        model.quantize_for_inference();
        let logits_infer = model.forward_ids(&input_ids);

        // Should produce same shape
        assert_eq!(logits_train.data().shape(), logits_infer.data().shape());

        // Values should be very close (same quantization)
        let train_vec = logits_train.data().to_vec();
        let infer_vec = logits_infer.data().to_vec();
        for (a, b) in train_vec.iter().zip(infer_vec.iter()) {
            assert!((a - b).abs() < 1e-4, "Train {} vs infer {} mismatch", a, b);
        }
    }
}
