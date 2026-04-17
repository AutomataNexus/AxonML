# axonml-llm Documentation

> Large-language-model architectures for AxonML.

## Overview

`axonml-llm` ships nine full pure-Rust LLM architectures on top of
`axonml-nn` + `axonml-autograd`, with shared building blocks (attention,
embeddings, RMSNorm, rotary positions), a HuggingFace weight loader, a
pretrained model hub, and a text-generation engine with top-k / top-p /
temperature sampling.

## Architectures

| Module       | Model / Variant                                                                                |
|--------------|------------------------------------------------------------------------------------------------|
| `gpt2`       | `GPT2`, `GPT2LMHead` (classic decoder-only)                                                    |
| `bert`       | `Bert`, `BertForSequenceClassification`, `BertForMaskedLM` (bidirectional encoder)             |
| `llama`      | `LLaMA`, `LLaMAForCausalLM` (split-halves RoPE + GQA + SwiGLU)                                 |
| `mistral`    | `Mistral`, `MistralForCausalLM` (sliding-window attention)                                     |
| `phi`        | `Phi`, `PhiForCausalLM` (partial RoPE + GELU)                                                  |
| `ssm`        | `SSMBlock`, `SSMConfig`, `SSMForCausalLM` (Mamba: selective S6 scan + depthwise conv)          |
| `hydra`      | `HydraModel`, `HydraConfig` (hybrid SSM + windowed attention)                                  |
| `chimera`    | `ChimeraModel`, `ChimeraConfig` (sparse MoE + differential attention)                          |
| `trident`    | `TridentModel`, `TridentConfig` (1.58-bit ternary weights: RoPE + GQA + ReLU^2 FFN + SubLN)    |

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
where they are used (LLaMA, Mistral, Trident).

### `transformer`

`TransformerBlock`, `TransformerEncoder`, `TransformerDecoder` — generic
building blocks used by several models.

### `config`

`BertConfig`, `GPT2Config`, `TransformerConfig`. Architecture-specific
configs live alongside their models (`LLaMAConfig`, `MistralConfig`,
`PhiConfig`, `SSMConfig`, `HydraConfig`, `ChimeraConfig`, `TridentConfig`).

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

### Trident (1.58-bit ternary)

```rust
use axonml_llm::{TridentConfig, TridentModel};
let config = TridentConfig::smoke(); // or ::trident_1b(), ::trident_3b()
let model  = TridentModel::new(&config);
```

Each Trident block uses `TernaryLinear` from `axonml-nn` with trained
shadow weights + Straight-Through-Estimator, giving ~16x memory compression
for transformer weights while keeping activations in fp32 for accuracy.

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
- **GPT-2 / LLaMA / Phi / Trident** — causal; token *t* attends to tokens
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

0.6.1 (2026-04-16)
