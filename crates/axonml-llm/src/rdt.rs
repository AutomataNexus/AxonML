//! rdt — Recurrent-Depth Transformer (Huginn-style test-time compute).
//!
//! Architecture:
//!
//! ```text
//!   tokens ──► Embedding ──► Prelude (N_p × Qwen3DecoderLayer) ──► e
//!                                                                  │
//!                              ┌───────────────────────────────────┘
//!                              ▼
//!                       h_0 := e  (seed hidden state)
//!                              │
//!                              ▼
//!   ┌──► Core(h_t + e)   N_c layers, SHARED across iterations   ──► Block(h_t + e)
//!   │                                                                │
//!   │             h_{t+1} = α · h_t + β · e + Block(h_t + e)         │
//!   │                           ▲                                    │
//!   └───────────────────────────┴────────────────────────────────────┘   × K iterations
//!
//!                      h_K ──► Coda (N_d layers) ──► output_norm ──► logits
//! ```
//!
//! At training time, K is sampled uniformly from `[k_min, k_max]` per batch
//! so the model generalizes across iteration counts. At inference, K is a
//! per-request compute knob (test-time compute scaling) — more K for harder
//! queries, less for easy ones.
//!
//! # Design choices
//!
//! - **Reuse Qwen3DecoderLayer** as the atomic block — inherits QK-norm,
//!   split-halves RoPE, SwiGLU MLP, RMSNorm, and all the GPU-resident decode
//!   kernels nexus-serve has been tuned against.
//! - **α, β start as fixed f32 scalars** (default 0.5 each) per the paper's
//!   initial formulation. Upgrading to learnable scalars or per-layer
//!   coefficients is a planned v2 ablation.
//! - **Prelude and Coda weights are NOT shared with the Core.** Only the
//!   core iterates — prelude and coda run once per forward.
//! - **No KV caching across recurrent iterations in v1.** Paper notes
//!   negligible quality impact; nexus-serve inference will rebuild the core
//!   KV per token. Prelude/Coda KV caches are per-session as normal.
//!
//! # Reference
//!
//! Geiping et al. 2025, *Scaling up Test-Time Compute with Latent
//! Reasoning: A Recurrent Depth Approach* (Huginn-3.5B).
//!
//! Design doc: `/opt/AxonML/docs/RDT_DESIGN.md`.
//!
//! # File
//! `crates/axonml-llm/src/rdt.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Created
//! April 20, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of
//! any kind, express or implied. The author and AutomataNexus shall not be
//! held liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::{Embedding, Linear, Module, Parameter};
// (Linear is imported via the paths below; used through a bias-less `with_bias`.)
use axonml_tensor::Tensor;

use crate::attention::LayerKVCache;
use crate::llama::RMSNorm;
use crate::qwen3::{Qwen3Config, Qwen3DecoderLayer};

// =============================================================================
// Config
// =============================================================================

/// Configuration for Recurrent-Depth Transformer.
///
/// Wraps a standard [`Qwen3Config`] for per-layer hyperparameters (head
/// counts, hidden size, RoPE theta, etc.) and adds RDT-specific fields for
/// the prelude/core/coda layer counts and recurrent update coefficients.
#[derive(Debug, Clone)]
pub struct RDTConfig {
    /// Per-layer transformer config (shared across prelude/core/coda).
    pub base: Qwen3Config,
    /// Number of transformer layers in the prelude stack (runs once).
    pub n_prelude: usize,
    /// Number of transformer layers inside one core iteration (shared
    /// across the K iterations).
    pub n_core: usize,
    /// Number of transformer layers in the coda stack (runs once).
    pub n_coda: usize,
    /// Minimum core iterations sampled at training time.
    pub k_min: usize,
    /// Maximum core iterations sampled at training time.
    pub k_max: usize,
    /// Default K at inference. Tunable per-request via the
    /// `num_steps` field on the Messages API.
    pub k_default: usize,
    /// Recurrent update coefficient on h_t. Paper default: 0.5.
    pub alpha: f32,
    /// Recurrent update coefficient on e (prelude output). Paper default: 0.5.
    pub beta: f32,
}

impl RDTConfig {
    /// Smoke-test config — tiny, fits on a laptop GPU. ~200M params.
    pub fn rdt_tiny() -> Self {
        let mut base = Qwen3Config::qwen3_0_6b();
        // Collapse base layer count — we'll split layers across prelude/core/coda.
        base.num_hidden_layers = 8;
        base.hidden_size = 1024;
        base.intermediate_size = 3072;
        Self {
            base,
            n_prelude: 2,
            n_core: 4,
            n_coda: 2,
            k_min: 4,
            k_max: 16,
            k_default: 8,
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Mid-size config. ~500M params. Target for first serious training run.
    pub fn rdt_small() -> Self {
        let mut base = Qwen3Config::qwen3_1_7b();
        base.num_hidden_layers = 10;
        base.hidden_size = 1536;
        base.intermediate_size = 4608;
        base.num_attention_heads = 24;
        base.num_key_value_heads = 8;
        base.head_dim = 64;
        Self {
            base,
            n_prelude: 2,
            n_core: 6,
            n_coda: 2,
            k_min: 4,
            k_max: 16,
            k_default: 8,
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Upper-middleweight. ~1.2B params. Serious test-time-compute lever.
    pub fn rdt_mid() -> Self {
        let mut base = Qwen3Config::qwen3_1_7b();
        base.num_hidden_layers = 16;
        base.hidden_size = 2048;
        base.intermediate_size = 6144;
        base.num_attention_heads = 32;
        base.num_key_value_heads = 8;
        base.head_dim = 64;
        Self {
            base,
            n_prelude: 4,
            n_core: 8,
            n_coda: 4,
            k_min: 4,
            k_max: 16,
            k_default: 8,
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Total flat layer count (prelude + core + coda). Convenience for
    /// parameter-count estimation.
    pub fn total_layer_count(&self) -> usize {
        self.n_prelude + self.n_core + self.n_coda
    }
}

// =============================================================================
// Stacks
// =============================================================================

/// Prelude — input-side transformer stack that produces the frozen
/// embedding `e` fed into every core iteration.
#[derive(Debug)]
pub struct RDTPrelude {
    layers: Vec<Qwen3DecoderLayer>,
}

impl RDTPrelude {
    /// Build a fresh prelude with `n` transformer layers.
    pub fn new(cfg: &Qwen3Config, n: usize) -> Self {
        let layers = (0..n).map(|_| Qwen3DecoderLayer::new(cfg)).collect();
        Self { layers }
    }

    /// Forward: apply each layer in order, no KV cache. Called once per
    /// token step (the prelude output `e` is held frozen across the K
    /// core iterations).
    pub fn forward(
        &self,
        hidden: &Variable,
        kv_cache: Option<&mut LayerKVCache>,
        position_offset: usize,
    ) -> Variable {
        let mut x = hidden.clone();
        if let Some(cache) = kv_cache {
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward_with_cache(&x, cache.get_mut(i), position_offset);
            }
        } else {
            for layer in &self.layers {
                x = layer.forward_with_cache(&x, None, position_offset);
            }
        }
        x
    }

    /// Collect all parameters for optimizer registration.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

/// Recurrent core — the stack that gets re-applied K times per forward.
///
/// Holds α and β as public f32 fields (v1 fixed; will migrate to learnable
/// Parameters in v2). Invariant: `alpha + beta ≤ 1` roughly ensures
/// numerical stability — the core's own residual contribution handles the
/// rest.
#[derive(Debug)]
pub struct RDTCore {
    layers: Vec<Qwen3DecoderLayer>,
    /// Update coefficient on h_t.
    pub alpha: f32,
    /// Update coefficient on e.
    pub beta: f32,
}

impl RDTCore {
    /// Build a fresh core with `n` transformer layers.
    pub fn new(cfg: &Qwen3Config, n: usize, alpha: f32, beta: f32) -> Self {
        let layers = (0..n).map(|_| Qwen3DecoderLayer::new(cfg)).collect();
        Self { layers, alpha, beta }
    }

    /// Run one core iteration: compute `Block(h_t + e)`. The caller handles
    /// the α·h_t + β·e + block-output mixing step outside this function so
    /// the block itself can be cleanly unit-tested against a single-shot
    /// transformer stack.
    pub fn block_forward(&self, h: &Variable, e: &Variable) -> Variable {
        let mut x = h.add(e);
        // No KV cache across recurrent iterations — fresh attention each
        // time per the paper's v1 formulation.
        for layer in &self.layers {
            x = layer.forward_with_cache(&x, None, 0);
        }
        x
    }

    /// Parameters (shared across K iterations — one copy of weights).
    pub fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// Core layer count (for metadata / param-count estimates).
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Coda — output-side transformer stack that reads the final recurrent
/// state h_K and prepares it for the LM head.
#[derive(Debug)]
pub struct RDTCoda {
    layers: Vec<Qwen3DecoderLayer>,
}

impl RDTCoda {
    /// Build a fresh coda with `n` transformer layers.
    pub fn new(cfg: &Qwen3Config, n: usize) -> Self {
        let layers = (0..n).map(|_| Qwen3DecoderLayer::new(cfg)).collect();
        Self { layers }
    }

    /// Forward through the coda layers.
    pub fn forward(
        &self,
        hidden: &Variable,
        kv_cache: Option<&mut LayerKVCache>,
        position_offset: usize,
    ) -> Variable {
        let mut x = hidden.clone();
        if let Some(cache) = kv_cache {
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward_with_cache(&x, cache.get_mut(i), position_offset);
            }
        } else {
            for layer in &self.layers {
                x = layer.forward_with_cache(&x, None, position_offset);
            }
        }
        x
    }

    /// Collect all parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

// =============================================================================
// Full model
// =============================================================================

/// Recurrent-Depth Transformer — backbone without an LM head.
#[derive(Debug)]
pub struct RDT {
    embed_tokens: Embedding,
    prelude: RDTPrelude,
    core: RDTCore,
    coda: RDTCoda,
    output_norm: RMSNorm,
    cfg: RDTConfig,
}

impl RDT {
    /// Build a fresh RDT with random-initialized weights.
    pub fn new(cfg: &RDTConfig) -> Self {
        Self {
            embed_tokens: Embedding::new(cfg.base.vocab_size, cfg.base.hidden_size),
            prelude: RDTPrelude::new(&cfg.base, cfg.n_prelude),
            core: RDTCore::new(&cfg.base, cfg.n_core, cfg.alpha, cfg.beta),
            coda: RDTCoda::new(&cfg.base, cfg.n_coda),
            output_norm: RMSNorm::new(cfg.base.hidden_size, cfg.base.rms_norm_eps),
            cfg: cfg.clone(),
        }
    }

    /// Forward pass over raw token ids, running the core for exactly `k`
    /// iterations.
    ///
    /// `input_ids`: `[B, T]` token ids (cast to f32 internally — see Qwen3
    /// forward for the same pattern).
    ///
    /// Returns `[B, T, hidden]` post-norm hidden states, ready for the LM
    /// head.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>, k: usize) -> Variable {
        let ids_f32: Vec<f32> = input_ids.to_vec().iter().map(|&x| x as f32).collect();
        let ids_var =
            Variable::new(Tensor::from_vec(ids_f32, input_ids.shape()).unwrap(), false);
        let embeds = self.embed_tokens.forward(&ids_var);

        // e := Prelude(Embed(input))  — frozen across the K iterations.
        let e = self.prelude.forward(&embeds, None, 0);

        // h_0 := e   (seed the hidden state from the prelude output).
        let mut h = e.clone();

        // Recurrent core: h_{t+1} = α·h_t + β·e + Block(h_t + e)
        for _ in 0..k {
            let block_out = self.core.block_forward(&h, &e);
            h = h
                .mul_scalar(self.core.alpha)
                .add(&e.mul_scalar(self.core.beta))
                .add(&block_out);
        }

        // Coda → final norm.
        let coda_out = self.coda.forward(&h, None, 0);
        self.output_norm.forward(&coda_out)
    }

    /// Get config.
    pub fn config(&self) -> &RDTConfig {
        &self.cfg
    }

    /// Collect all parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.embed_tokens.parameters());
        params.extend(self.prelude.parameters());
        params.extend(self.core.parameters());
        params.extend(self.coda.parameters());
        params.extend(self.output_norm.parameters());
        params
    }
}

impl Module for RDT {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape: Vec<usize> = input_data.shape().to_vec();
        let ids: Vec<u32> = input_data.to_vec().iter().map(|&x| x as u32).collect();
        let input_ids = Tensor::from_vec(ids, &shape).unwrap();
        self.forward_ids(&input_ids, self.cfg.k_default)
    }

    fn parameters(&self) -> Vec<Parameter> {
        RDT::parameters(self)
    }
}

// =============================================================================
// For Causal LM
// =============================================================================

/// RDT with a language-modeling head on top.
#[derive(Debug)]
pub struct RDTForCausalLM {
    model: RDT,
    lm_head: Linear,
}

impl RDTForCausalLM {
    /// Build a fresh RDT-for-causal-LM. LM head is an untied nn::Linear
    /// from `hidden` → `vocab_size`. Tied-weights variant is a planned
    /// v2 option (save ~10% params at tiny/small sizes).
    pub fn new(cfg: &RDTConfig) -> Self {
        Self {
            model: RDT::new(cfg),
            // LM heads are bias-less by convention (LLaMA/Qwen family).
            // Keeping with_bias=false also keeps the parameter count
            // predictable for the GGUF exporter manifest.
            lm_head: Linear::with_bias(cfg.base.hidden_size, cfg.base.vocab_size, false),
        }
    }

    /// Forward through backbone (K core iterations) + LM head.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>, k: usize) -> Variable {
        let hidden = self.model.forward_ids(input_ids, k);
        self.lm_head.forward(&hidden)
    }

    /// Expose the underlying RDT backbone (for state-dict load/save paths).
    pub fn model(&self) -> &RDT {
        &self.model
    }

    /// Get config (delegates to backbone).
    pub fn config(&self) -> &RDTConfig {
        self.model.config()
    }

    /// Collect all parameters (backbone + lm_head).
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.model.parameters();
        params.extend(self.lm_head.parameters());
        params
    }
}

impl Module for RDTForCausalLM {
    fn forward(&self, input: &Variable) -> Variable {
        let hidden = self.model.forward(input);
        self.lm_head.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Parameter> {
        RDTForCausalLM::parameters(self)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_presets_produce_valid_layer_counts() {
        for cfg in [RDTConfig::rdt_tiny(), RDTConfig::rdt_small(), RDTConfig::rdt_mid()] {
            assert!(cfg.n_prelude >= 1);
            assert!(cfg.n_core >= 1);
            assert!(cfg.n_coda >= 1);
            assert!(cfg.k_min <= cfg.k_max);
            assert!(cfg.k_default >= cfg.k_min && cfg.k_default <= cfg.k_max);
            assert!(cfg.alpha >= 0.0 && cfg.alpha <= 1.0);
            assert!(cfg.beta >= 0.0 && cfg.beta <= 1.0);
        }
    }

    #[test]
    fn rdt_tiny_builds_and_has_non_zero_params() {
        let cfg = RDTConfig::rdt_tiny();
        let model = RDT::new(&cfg);
        let params = model.parameters();
        assert!(!params.is_empty(), "expected non-empty parameter list");
        assert_eq!(model.prelude.layers.len(), cfg.n_prelude);
        assert_eq!(model.core.layers.len(), cfg.n_core);
        assert_eq!(model.coda.layers.len(), cfg.n_coda);
    }

    #[test]
    fn rdt_core_shares_weights_across_iterations() {
        // Invariant: the core holds ONE copy of weights regardless of K.
        // Param count must not change when we vary K at the forward level.
        let cfg = RDTConfig::rdt_tiny();
        let model = RDTForCausalLM::new(&cfg);
        let n_params = model.parameters().len();

        // Running forward_ids with different K values doesn't add params.
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[1, 4]).unwrap();
        let _ = model.forward_ids(&ids, 1);
        let _ = model.forward_ids(&ids, 8);
        assert_eq!(model.parameters().len(), n_params);
    }

    #[test]
    fn rdt_forward_output_shape_is_b_t_v() {
        // Sanity: forward produces [batch, seq_len, vocab_size] logits.
        let cfg = RDTConfig::rdt_tiny();
        let model = RDTForCausalLM::new(&cfg);
        let ids = Tensor::from_vec(vec![1u32, 2, 3], &[1, 3]).unwrap();
        let logits = model.forward_ids(&ids, 4);
        let shape = logits.data().shape().to_vec();
        assert_eq!(shape, vec![1, 3, cfg.base.vocab_size]);
    }

    #[test]
    fn total_layer_count_sums_stacks() {
        let cfg = RDTConfig::rdt_small();
        assert_eq!(cfg.total_layer_count(), cfg.n_prelude + cfg.n_core + cfg.n_coda);
    }

    #[test]
    fn rdt_exports_to_gguf_and_file_is_valid() {
        // Round-trip check: fresh rdt-tiny → export_rdt_to_gguf → verify
        // the file starts with the GGUF magic and has a non-trivial size.
        // Full nexus-serve load-side round-trip comes with the inference
        // dispatch work (task #58 step 5).
        use std::fs;
        let cfg = RDTConfig::rdt_tiny();
        let model = RDTForCausalLM::new(&cfg);
        let tmp = std::env::temp_dir().join("rdt_export_test.gguf");
        let _ = fs::remove_file(&tmp);
        crate::gguf_export::export_rdt_to_gguf(&model, &tmp, "rdt-tiny-test", None)
            .expect("export should succeed");

        let bytes = fs::read(&tmp).expect("read exported file");
        assert!(bytes.len() > 1024, "exported file unexpectedly small: {} bytes", bytes.len());
        // GGUF magic = 0x4655_4747 little-endian = "GGUF"
        assert_eq!(&bytes[0..4], b"GGUF", "missing GGUF magic at start of file");
        // Version = 3 at offset 4 (u32 LE)
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, 3, "GGUF version should be 3");
        let _ = fs::remove_file(&tmp);
    }
}
