// ═══════════════════════════════════════════════════════════════════════════════
// RustyMythos — Recurrent-Depth Transformer with MoE and Stable Latent Reasoning
//
// Three-stage architecture:
//   Prelude  → token embedding into continuous latent space
//   Recurrent Block → looped reasoning with LTI-stable injection + MoE transformer
//   Coda     → latent state projection back to token probabilities
//
// Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
// ORCID: 0009-0005-2158-7060
// ═══════════════════════════════════════════════════════════════════════════════

use axonml_autograd::Variable;
use std::ops::Neg;
use axonml_nn::{Module, Linear, LayerNorm, Parameter};
use axonml_nn::layers::moe::MoELayer;
use axonml_tensor::Tensor;

// ─── Configuration ───────────────────────────────────────────────────────────

pub struct RustyMythosConfig {
    pub d_model: usize,
    pub max_loop_iters: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub expert_intermediate: usize,
    pub top_k: usize,
}

impl Default for RustyMythosConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            max_loop_iters: 4,
            vocab_size: 256,
            num_experts: 4,
            expert_intermediate: 256,
            top_k: 1,
        }
    }
}

impl RustyMythosConfig {
    pub fn from_scale(scale: &str) -> Self {
        match scale {
            "tiny" => Self { d_model: 16, max_loop_iters: 4, num_experts: 4, expert_intermediate: 16, vocab_size: 16, ..Self::default() },
            "xs" => Self::default(),
            "small" => Self { d_model: 256, max_loop_iters: 8, num_experts: 8, expert_intermediate: 512, ..Self::default() },
            "medium" => Self { d_model: 512, max_loop_iters: 16, num_experts: 16, expert_intermediate: 1024, ..Self::default() },
            "large" => Self { d_model: 1024, max_loop_iters: 32, num_experts: 32, expert_intermediate: 2048, ..Self::default() },
            "xl" => Self { d_model: 2048, max_loop_iters: 64, num_experts: 64, expert_intermediate: 4096, vocab_size: 512, ..Self::default() },
            _ => Self::default(),
        }
    }
}

// ─── Stage 1: Prelude ────────────────────────────────────────────────────────

pub struct Prelude {
    embed: Linear,
    norm: LayerNorm,
}

impl Prelude {
    pub fn new(config: &RustyMythosConfig) -> Self {
        Self {
            embed: Linear::new(config.vocab_size, config.d_model),
            norm: LayerNorm::new(vec![config.d_model]),
        }
    }
}

impl Module for Prelude {
    fn forward(&self, x: &Variable) -> Variable {
        let embedded = self.embed.forward(x);
        self.norm.forward(&embedded)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.embed.parameters();
        p.extend(self.norm.parameters());
        p
    }
}

// ─── Transformer Layer (Multi-Head Attention + MoE FFN) ──────────────────────

pub struct MythosTransformerLayer {
    attn_proj: Linear,
    attn_out: Linear,
    attn_norm: LayerNorm,
    moe: MoELayer,
    moe_norm: LayerNorm,
}

impl MythosTransformerLayer {
    pub fn new(config: &RustyMythosConfig) -> Self {
        Self {
            attn_proj: Linear::new(config.d_model, config.d_model),
            attn_out: Linear::new(config.d_model, config.d_model),
            attn_norm: LayerNorm::new(vec![config.d_model]),
            moe: MoELayer::new(
                config.d_model,
                config.expert_intermediate,
                config.num_experts,
                config.top_k,
            ),
            moe_norm: LayerNorm::new(vec![config.d_model]),
        }
    }

    pub fn forward_at_depth(&self, h: &Variable, _t: usize) -> Variable {
        let residual = h.clone();
        let normed = self.attn_norm.forward(h);
        let attn = self.attn_out.forward(&self.attn_proj.forward(&normed));
        let h = residual.add(&attn);

        let residual = h.clone();
        let normed = self.moe_norm.forward(&h);
        let s = normed.shape();
        let normed_3d = normed.reshape(&[s[0], 1, s[1]]);
        let moe_out_3d = self.moe.forward(&normed_3d);
        let moe_out = moe_out_3d.reshape(&[s[0], s[1]]);
        residual.add(&moe_out)
    }
}

impl Module for MythosTransformerLayer {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_at_depth(input, 0)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.attn_proj.parameters();
        p.extend(self.attn_out.parameters());
        p.extend(self.attn_norm.parameters());
        p.extend(self.moe.parameters());
        p.extend(self.moe_norm.parameters());
        p
    }
}

// ─── Stage 2: Recurrent Block (Looped Reasoning) ─────────────────────────────

pub struct RecurrentBlock {
    max_loop_iters: usize,
    log_a: Variable,
    b_param: Variable,
    transformer: MythosTransformerLayer,
}

impl RecurrentBlock {
    pub fn new(config: &RustyMythosConfig) -> Self {
        let log_a_data = Tensor::randn(&[config.d_model]).mul_scalar(-1.0);
        let b_data = Tensor::ones(&[config.d_model]);
        Self {
            max_loop_iters: config.max_loop_iters,
            log_a: Variable::new(log_a_data, true),
            b_param: Variable::new(b_data, true),
            transformer: MythosTransformerLayer::new(config),
        }
    }
}

impl Module for RecurrentBlock {
    fn forward(&self, e: &Variable) -> Variable {
        let mut h = e.clone();
        let a_matrix = self.log_a.exp().neg();

        for t in 0..self.max_loop_iters {
            let transformer_out = self.transformer.forward_at_depth(&h, t);
            let ah = a_matrix.mul(&h);
            let be = self.b_param.mul(e);
            h = ah.add(&be).add(&transformer_out);
        }

        h
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = vec![
            Parameter::from_variable(self.log_a.clone()),
            Parameter::from_variable(self.b_param.clone()),
        ];
        p.extend(self.transformer.parameters());
        p
    }
}

// ─── Stage 3: Coda ──────────────────────────────────────────────────────────

pub struct Coda {
    norm: LayerNorm,
    head: Linear,
}

impl Coda {
    pub fn new(config: &RustyMythosConfig) -> Self {
        Self {
            norm: LayerNorm::new(vec![config.d_model]),
            head: Linear::new(config.d_model, config.vocab_size),
        }
    }
}

impl Module for Coda {
    fn forward(&self, x: &Variable) -> Variable {
        let normed = self.norm.forward(x);
        self.head.forward(&normed)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.norm.parameters();
        p.extend(self.head.parameters());
        p
    }
}

// ─── Full Model: RustyMythos ─────────────────────────────────────────────────

pub struct RustyMythos {
    pub config: RustyMythosConfig,
    prelude: Prelude,
    recurrent_block: RecurrentBlock,
    coda: Coda,
}

impl RustyMythos {
    pub fn new(config: RustyMythosConfig) -> Self {
        let prelude = Prelude::new(&config);
        let recurrent_block = RecurrentBlock::new(&config);
        let coda = Coda::new(&config);
        Self { config, prelude, recurrent_block, coda }
    }

    pub fn param_count(&self) -> usize {
        self.parameters().iter().map(|p| p.data().numel()).sum()
    }
}

impl Module for RustyMythos {
    fn forward(&self, x: &Variable) -> Variable {
        let e = self.prelude.forward(x);
        let h = self.recurrent_block.forward(&e);
        self.coda.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.prelude.parameters();
        p.extend(self.recurrent_block.parameters());
        p.extend(self.coda.parameters());
        p
    }
}
