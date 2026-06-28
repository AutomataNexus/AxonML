# axonml-llm Documentation

> Large-language-model architectures for AxonML.

## Overview

`axonml-llm` ships pure-Rust LLM architectures on top of
`axonml-nn` + `axonml-autograd`, with shared building blocks (attention,
embeddings, RMSNorm, rotary positions), a HuggingFace weight loader, a
pretrained model hub, GGUF import/export, and a text-generation engine with
top-k / top-p / temperature sampling.

## Architectures

| Module       | Model / Variant                                                                                |
|--------------|------------------------------------------------------------------------------------------------|
| `gpt2`       | `GPT2`, `GPT2LMHead` (classic decoder-only)                                                    |
| `bert`       | `Bert`, `BertForSequenceClassification`, `BertForMaskedLM` (bidirectional encoder)             |
| `llama`      | `LLaMA`, `LLaMAForCausalLM` (split-halves RoPE + GQA + SwiGLU)                                 |
| `mistral`    | `Mistral`, `MistralForCausalLM` (sliding-window attention)                                     |
| `phi`        | `Phi`, `PhiForCausalLM` (partial RoPE + GELU)                                                  |
| `qwen3`      | `Qwen3ForCausalLM` (trainable Qwen3 with QK-norm; teacher/student + distillation)              |
| `ssm`        | `SSMBlock`, `SSMConfig`, `SSMForCausalLM` (Mamba: selective S6 scan + depthwise conv)          |
| `hydra`      | `HydraModel`, `HydraConfig` (hybrid SSM + windowed attention)                                  |
| `chimera`    | `ChimeraModel`, `ChimeraConfig` (sparse MoE + differential attention)                          |
| `rdt`        | `RDTForCausalLM` (Recurrent-Depth Transformer, Huginn-style test-time compute: Prelude → latent block iterated K times → Coda). `rdt` GGUF arch id; distillation via `train_rdt_distill` |

GGUF support covers both `qwen2` and `qwen3` loaders for distillation /
teacher-student workflows. Training paths use the device-native CPU
parallelism (see [training](../training.md)); on GPU, the 1.58-bit ternary
quantized linear (BitNet b1.58) has fused forward/backward kernels.

The `gpt2`, `bert`, `llama`, `mistral`, `phi`, and `qwen3` modules are faithful
reimplementations of published architectures (trainable from scratch and
loadable from HuggingFace `safetensors` / GGUF). The remaining three —
**Chimera, Hydra, and RDT** — are original AutomataNexus
architectures, detailed below.

## Novel Architectures

These are designed and implemented in-house, not ports. Each is trainable
end-to-end on the AxonML autograd engine and several have a dedicated
whitepaper and Hailo-NPU compile path.

### Chimera — sparse MoE + Differential Attention

A small language model that pairs **massive capacity with noise-cancelling
precision**. Each `ChimeraBlock` is:

```text
x = x + DifferentialAttention(RMSNorm(x))
x = x + MoELayer(RMSNorm(x))            // top-2 of 8 SwiGLU experts
```

- **Differential Attention** (Microsoft DIFF-Transformer style): computes two
  softmax attention maps and subtracts them, cancelling attention noise on
  irrelevant tokens for sharper retrieval.
- **Sparse MoE MLP**: 8 expert SwiGLU MLPs per layer, top-2 routed — only
  ~25% of parameters activate per token (Switch/GShard-style).
- **Load-balancing auxiliary loss** prevents expert collapse.

`ChimeraModel` / `ChimeraConfig`.

### Hydra — hybrid SSM + sparse attention

Alternates **Mamba-style SSM blocks** (`SSMBlock`: selective S6 scan +
depthwise Conv1d, linear-time in sequence length) with **windowed (local)
attention** layers. The SSM layers carry long-range state cheaply while the
windowed-attention layers recover precise local token interactions — a
sub-quadratic hybrid. `HydraModel` / `HydraConfig`.

### RDT — Recurrent-Depth Transformer (test-time compute)

Huginn-style latent reasoning (Geiping et al. 2025): instead of more layers,
**iterate a shared core block K times** in latent space, where K is a per-request
compute knob.

```text
tokens → Embedding → Prelude (N_p Qwen3 layers) → e
h_0 = e
repeat K times (shared Core, N_c layers):
    h_{t+1} = α·h_t + β·e + Block(h_t + e)
h_K → Coda (N_d layers) → output_norm → logits
```

- **K is sampled uniformly from `[k_min, k_max]` per batch at training** so the
  model generalizes across iteration counts; at inference K trades latency for
  quality (more for hard prompts, less for easy ones).
- Reuses `Qwen3DecoderLayer` as the atomic block, inheriting QK-norm,
  split-halves RoPE, SwiGLU, and the tuned GPU decode kernels.
- Prelude/Coda weights are **not** shared with the iterated Core.
- Configs: `rdt_tiny` (~265M), `rdt_small` (~540M), `rdt_mid` (~1.18B).
- **GGUF arch id `rdt`**; production distillation via `train_rdt_distill`
  (a 7B teacher → RDT student). Design doc: `docs/RDT_DESIGN.md`.

`RDTForCausalLM` / `RDTConfig`.

## Shared Building Blocks

### `attention`

- `MultiHeadSelfAttention` — BERT-style bidirectional self-attention
- `CausalSelfAttention` — GPT-style causal self-attention
- `FlashAttention`, `FlashAttentionConfig` — tiled / block-sparse attention
- `KVCache`, `LayerKVCache` — per-layer KV cache for incremental decode
- `scaled_dot_product_attention(q, k, v, mask, dropout)` — low-level primitive

### `embedding`

- `TokenEmbedding` — vocab -> hidden projection
- `PositionalEmbedding` — learned absolute positions
- `BertEmbedding` — BERT embedding stack (token + position + segment +
  LayerNorm + dropout)
- `GPT2Embedding` — GPT-2 embedding stack

Rotary embeddings and RMSNorm live inside the individual model modules
where they are used (LLaMA, Mistral).

### `transformer`

`TransformerBlock`, `TransformerEncoder`, `TransformerDecoder` — generic
building blocks used by several models.

### `config`

`BertConfig`, `GPT2Config`, `TransformerConfig`. Architecture-specific
configs live alongside their models (`LLaMAConfig`, `MistralConfig`,
`PhiConfig`, `SSMConfig`, `HydraConfig`, `ChimeraConfig`).

### `generation`

Text generation with greedy, sampling, top-k, and nucleus (top-p) decoding.

```rust
pub struct GenerationConfig {
    pub max_length: usize,
    pub min_length: usize,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl GenerationConfig {
    pub fn greedy() -> Self;
    pub fn sampling(temperature: f32) -> Self;
    pub fn top_k_sampling(k: usize, temperature: f32) -> Self;
    pub fn nucleus_sampling(p: f32, temperature: f32) -> Self;
}
```

`TextGenerator<M>::generate(&[u32]) -> Vec<u32>` runs token-by-token
autoregressive decoding.

### `tokenizer`

`HFTokenizer` (HuggingFace-compatible tokenizer shim) and `SpecialTokens`
helper.

### `hf_loader`

`HFLoader`, `load_llama_from_hf`, `load_mistral_from_hf` — reads
`safetensors` checkpoints from HuggingFace-style directories into AxonML
`state_dict` form.

### `hub`

`PretrainedLLM`, `download_weights`, and `llm_registry()` — curated index
of downloadable model checkpoints.

### `state_dict`

`LoadStateDict` trait + `LoadResult` for loading parameter tensors into
`Module` trees from flat key -> tensor maps.

### `error`

`LLMError` + `LLMResult<T>`.

## Usage

### BERT classification

```rust
use axonml_llm::{BertConfig, BertForSequenceClassification};
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let config = BertConfig::base();
let model = BertForSequenceClassification::new(&config, 2);

let input_ids = Variable::new(
    Tensor::from_vec(vec![101.0, 2054.0, 2003.0, 2023.0, 102.0], &[1, 5]).unwrap(),
    false,
);
let logits = model.forward(&input_ids, None);
```

### BERT masked LM

```rust
use axonml_llm::{BertConfig, BertForMaskedLM};
let config = BertConfig::base();
let model  = BertForMaskedLM::new(&config);
let logits = model.forward(&input_ids, None); // [batch, seq, vocab]
```

### GPT-2 text generation

```rust
use axonml_llm::{GPT2Config, GPT2LMHead, GenerationConfig, TextGenerator};

let config = GPT2Config::small();
let model  = GPT2LMHead::new(&config);

let gen_config = GenerationConfig::top_k_sampling(50, 0.8);
let generator  = TextGenerator::new(model, gen_config);

let prompt = vec![15496u32, 11, 314]; // "Hello, I"
let output = generator.generate(&prompt);
```

### RDT (recurrent-depth, test-time compute)

```rust
use axonml_llm::{RDTConfig, RDTForCausalLM};
use axonml_tensor::Tensor;

let config = RDTConfig::rdt_small();     // rdt_tiny / rdt_small / rdt_mid
let model  = RDTForCausalLM::new(&config);

// `k` (recurrent iterations) is the per-request compute knob — pass more for
// harder prompts. At train time it is sampled from [k_min, k_max].
let logits = model.forward_ids(&input_ids, /*k=*/8); // input_ids: Tensor<u32>
```

### Fine-tuning BERT

```rust
use axonml_llm::{BertConfig, BertForSequenceClassification};
use axonml::prelude::*;

let config = BertConfig::base();
let mut model = BertForSequenceClassification::new(&config, 3);
let mut opt = Adam::new(model.parameters(), 2e-5);

for (input_ids, labels) in dataset {
    let logits = model.forward(&input_ids, None);
    let loss = CrossEntropyLoss::new().compute(&logits, &labels);
    loss.backward();
    opt.step();
    opt.zero_grad();
}
```

## Model Sizes

### BERT variants

| Variant | Layers | Hidden | Heads | Params |
|---------|--------|--------|-------|--------|
| tiny    | 2      | 128    | 2     | ~4M    |
| base    | 12     | 768    | 12    | ~110M  |
| large   | 24     | 1024   | 16    | ~340M  |

### GPT-2 variants

| Variant | Layers | Hidden | Heads | Params |
|---------|--------|--------|-------|--------|
| tiny    | 2      | 64     | 2     | ~1M    |
| small   | 12     | 768    | 12    | ~117M  |
| medium  | 24     | 1024   | 16    | ~345M  |
| large   | 36     | 1280   | 20    | ~762M  |
| xl      | 48     | 1600   | 25    | ~1.5B  |

## Attention Patterns

- **BERT** — bidirectional; every token attends to every other.
- **GPT-2 / LLaMA / Phi** — causal; token *t* attends to tokens
  *0..=t*.
- **Mistral** — sliding-window causal; token *t* attends to tokens
  *max(0, t-W)..=t*.
- **Hydra** — hybrid SSM state + windowed attention over short spans.

## Feature Flags

- `default` — base features
- `pretrained` — load pretrained weight sets

## Dependencies of Note

`safetensors = "0.3"` for checkpoint IO, `half = "2.3"` for f16 conversion,
`reqwest` + `indicatif` for `hub` downloads.

## Last updated

0.6.5 (2026-06-06)
