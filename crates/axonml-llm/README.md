# axonml-llm

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust 1.75+"></a>
  <a href="https://crates.io/crates/axonml-llm"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Crate Version"></a>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

`axonml-llm` provides nine large-language-model architectures for the AxonML framework, all implemented in pure Rust on top of `axonml-tensor` and `axonml-autograd`. Shared infrastructure includes multi-head / causal self-attention with a KV cache, a FlashAttention kernel, RoPE, RMSNorm, a HuggingFace weight loader (safetensors), a state-dict name-mapping helper, a pretrained-weights hub with on-disk caching, a configurable text-generation sampler, and an HF-style tokenizer.

---

## Supported architectures

| Architecture | Module | Notes |
|--------------|--------|-------|
| **GPT-2** | `gpt2` | Decoder-only transformer with learned positional embeddings and `GPT2LMHead`. |
| **BERT** | `bert` | Bidirectional encoder with `BertForSequenceClassification` and `BertForMaskedLM`. |
| **LLaMA** | `llama` | LLaMA-2/3-style: split-halves RoPE, grouped-query attention, SwiGLU MLP, RMSNorm. `LLaMAForCausalLM`. |
| **Mistral** | `mistral` | LLaMA + sliding-window attention. `MistralForCausalLM`. |
| **Phi** | `phi` | Phi-1/2/3-style: partial RoPE, GELU MLP, optional parallel-attn block. `PhiForCausalLM`. |
| **SSM** | `ssm` | Mamba/S6-style selective state-space model: depthwise Conv1d + selective scan + RMSNorm. `SSMBlock`, `SSMForCausalLM`. |
| **Hydra** | `hydra` | Hybrid: alternates SSM blocks and windowed (local) attention. `HydraModel`. |
| **Chimera** | `chimera` | Sparse MoE (top-k routing) + differential attention, with load-balancing auxiliary loss. `ChimeraModel`. |
| **Trident** | `trident` | 1.58-bit ternary weights (`TernaryLinear`) + RoPE + GQA + ReLU²-gated FFN + SubLN (BitNet b1.58-2B-4T recipe). `TridentModel`. |

### Preset configurations

| Config | Presets |
|--------|---------|
| `GPT2Config` | `tiny`, `small`, `medium`, `large`, `xl` |
| `BertConfig` | `tiny`, `base`, `large` |
| `LLaMAConfig` | `llama2_7b`, `llama2_13b`, `llama3_8b`, `tiny` |
| `MistralConfig` | `mistral_7b`, `mistral_7b_instruct`, `mixtral_8x7b`, `tiny` |
| `PhiConfig` | `phi1`, `phi2`, `phi3_mini`, `tiny` |
| `SSMConfig` | `from_d_model(d_model, vocab_size)` builder |
| `HydraConfig` | `base` (~300M), `small`, `tiny` |
| `ChimeraConfig` | `default_2b` (8 experts, top-2), `small`, `tiny` |
| `TridentConfig` | `default_150m`, `tiny`, `medium`, plus 1B/3B/smoke constructors exposed for training |

---

## Shared building blocks

- **Attention** (`attention`) — `MultiHeadSelfAttention`, `CausalSelfAttention`, `scaled_dot_product_attention`, plus `FlashAttention` + `FlashAttentionConfig`. `KVCache` and per-layer `LayerKVCache` for incremental decoding.
- **Transformer** (`transformer`) — `TransformerBlock`, `TransformerEncoder`, `TransformerDecoder` with configurable depth/width/heads/activation and pre- or post-norm.
- **Embeddings** (`embedding`) — `TokenEmbedding`, `PositionalEmbedding` (sinusoidal), `GPT2Embedding`, `BertEmbedding`.
- **Generation** (`generation`) — `GenerationConfig` with `greedy()`, `sampling(temp)`, `top_k_sampling(k, temp)`, `nucleus_sampling(p, temp)`, `beam_search(beams)`, plus builder methods `with_max_tokens`, `with_eos_token`, `with_repetition_penalty`. `TextGenerator` drives the next-token logic.
- **Weight loading** — `HFLoader` (safetensors), `load_llama_from_hf`, `load_mistral_from_hf`, and generic `LoadStateDict` trait with `LoadResult` + `map_hf_to_axonml` / `map_axonml_to_hf` name mappers.
- **Hub** — `PretrainedLLM` registry (`llm_registry()`, `list_models()`), `download_weights(name, force)` with an on-disk cache under `$XDG_CACHE_HOME/axonml/hub/llm`.
- **Tokenizer** — `HFTokenizer` + `SpecialTokens` (HuggingFace-compatible tokenizer.json).

---

## Modules

| Module | Description |
|--------|-------------|
| `attention` | Multi-head / causal / flash attention, KV cache |
| `bert` | BERT encoder + classification and MLM heads |
| `chimera` | Sparse MoE + differential attention model |
| `config` | `GPT2Config`, `BertConfig`, `TransformerConfig` |
| `embedding` | Token, positional, BERT/GPT-2 combined embeddings |
| `error` | `LLMError` / `LLMResult` |
| `generation` | `GenerationConfig`, `TextGenerator`, sampling strategies |
| `gpt2` | GPT-2 + `GPT2LMHead` |
| `hf_loader` | HuggingFace safetensors loading (LLaMA, Mistral, …) |
| `hub` | Pretrained-weight registry and downloader |
| `hydra` | SSM + windowed-attention hybrid |
| `llama` | LLaMA with RoPE, GQA, SwiGLU, RMSNorm |
| `mistral` | Mistral (LLaMA + sliding-window attention) |
| `phi` | Phi with partial RoPE / GELU / optional parallel-attn |
| `ssm` | Mamba-style selective SSM blocks and LM head |
| `state_dict` | `LoadStateDict` trait, HF ↔ AxonML name mapping |
| `tokenizer` | HuggingFace-compatible `HFTokenizer` |
| `transformer` | Encoder / decoder blocks, feed-forward, layer norm |
| `trident` | 1.58-bit ternary SLM (BitNet b1.58 recipe) |

---

## Usage

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
axonml-llm = "0.6.1"
```

### GPT-2 text generation

```rust
use axonml_llm::{GPT2LMHead, GPT2Config};
use axonml_tensor::Tensor;

let config = GPT2Config::small();
let model = GPT2LMHead::new(&config);

let input_ids = Tensor::from_vec(vec![50256u32, 1, 2, 3], &[1, 4]).unwrap();
let output  = model.generate(&input_ids, 50, 0.8, Some(50));
let greedy  = model.generate_greedy(&input_ids, 50);
```

### BERT sequence classification

```rust
use axonml_llm::{BertForSequenceClassification, BertConfig};
use axonml_tensor::Tensor;

let config = BertConfig::base();
let model  = BertForSequenceClassification::new(&config, /*num_labels=*/ 2);

let input_ids = Tensor::from_vec(vec![101u32, 2054, 2003, 1996, 102], &[1, 5]).unwrap();
let logits = model.forward_classification(&input_ids);
```

### BERT masked language modeling

```rust
use axonml_llm::{BertForMaskedLM, BertConfig};
use axonml_tensor::Tensor;

let model = BertForMaskedLM::new(&BertConfig::base());
let input_ids = Tensor::from_vec(vec![101u32, 2054, 103, 1996, 102], &[1, 5]).unwrap();
let logits = model.forward_mlm(&input_ids); // [batch, seq, vocab]
```

### LLaMA / Mistral / Phi

```rust
use axonml_llm::{LLaMAForCausalLM, LLaMAConfig, MistralForCausalLM, MistralConfig, PhiForCausalLM, PhiConfig};

let llama   = LLaMAForCausalLM::new(&LLaMAConfig::llama2_7b());
let mistral = MistralForCausalLM::new(&MistralConfig::mistral_7b());
let phi     = PhiForCausalLM::new(&PhiConfig::phi2());
```

### SSM (Mamba-style) causal LM

```rust
use axonml_llm::{SSMForCausalLM, SSMConfig};

let cfg = SSMConfig::from_d_model(/*d_model=*/ 512, /*vocab_size=*/ 32000);
let ssm = SSMForCausalLM::new(&cfg);
```

### Hydra hybrid SSM + windowed attention

```rust
use axonml_llm::{HydraModel, HydraConfig};

let hydra = HydraModel::new(&HydraConfig::base()); // 768d, 24 layers, window 256
```

### Chimera sparse MoE + differential attention

```rust
use axonml_llm::{ChimeraModel, ChimeraConfig};

let cfg = ChimeraConfig::default_2b(); // 8 experts per layer, top-2 active
let model = ChimeraModel::new(&cfg);
// Returns (logits, lb_loss) when used via forward_with_loss for training.
```

### Trident 1.58-bit ternary LM

```rust
use axonml_llm::{TridentModel, TridentConfig};

let cfg = TridentConfig::default_150m(); // ternary weights + RoPE + GQA + ReLU² + SubLN
let model = TridentModel::new(&cfg);
```

### Custom transformer encoder

```rust
use axonml_llm::TransformerEncoder;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let encoder = TransformerEncoder::new(
    6,       // num_layers
    512,     // hidden_size
    8,       // num_heads
    2048,    // intermediate_size
    0.1,     // dropout
    1e-12,   // layer_norm_eps
    "gelu",  // activation
    false,   // pre_norm
);

let input  = Variable::new(Tensor::randn(&[2, 128, 512]), false);
let output = encoder.forward(&input);
```

### Generation configuration

```rust
use axonml_llm::{GenerationConfig, TextGenerator};

let config = GenerationConfig::nucleus_sampling(0.95, 0.8)
    .with_max_tokens(100)
    .with_repetition_penalty(1.2)
    .with_eos_token(50256);

let generator = TextGenerator::new(config);
let next_token = generator.get_next_token(&logits, &generated_so_far);
```

### Loading HuggingFace weights

```rust
use axonml_llm::{load_llama_from_hf, LLaMAConfig};

let mut model = axonml_llm::LLaMAForCausalLM::new(&LLaMAConfig::llama2_7b());
load_llama_from_hf(&mut model, "path/to/model.safetensors")?;
```

### Pretrained hub

```rust
use axonml_llm::{llm_registry, download_llm_weights};

for m in llm_registry().values() {
    println!("{}  ({} params)", m.name, m.num_parameters);
}

let path = download_llm_weights("bert-base-uncased", /*force=*/ false)?;
```

---

## Configuration reference

### BERT

| Config | Hidden size | Layers | Heads |
|--------|-------------|--------|-------|
| `BertConfig::tiny()`  | 128  | 2  | 2  |
| `BertConfig::base()`  | 768  | 12 | 12 |
| `BertConfig::large()` | 1024 | 24 | 16 |

### GPT-2

| Config | Embedding dim | Layers | Heads |
|--------|---------------|--------|-------|
| `GPT2Config::tiny()`   | 128  | 2  | 2  |
| `GPT2Config::small()`  | 768  | 12 | 12 |
| `GPT2Config::medium()` | 1024 | 24 | 16 |
| `GPT2Config::large()`  | 1280 | 36 | 20 |
| `GPT2Config::xl()`     | 1600 | 48 | 25 |

### LLaMA / Mistral / Phi

| Config | Presets |
|--------|---------|
| LLaMA   | `llama2_7b`, `llama2_13b`, `llama3_8b`, `tiny` |
| Mistral | `mistral_7b`, `mistral_7b_instruct`, `mixtral_8x7b`, `tiny` |
| Phi     | `phi1`, `phi2`, `phi3_mini`, `tiny` |

### Hybrid / SSM / MoE / Ternary

| Config | Presets / builder |
|--------|--------------------|
| `SSMConfig`     | `from_d_model(d_model, vocab)` |
| `HydraConfig`   | `base`, `small`, `tiny` |
| `ChimeraConfig` | `default_2b`, `small`, `tiny` |
| `TridentConfig` | `default_150m`, `tiny`, `medium` (plus 1B/3B/smoke constructors) |

---

## Generation strategies

| Strategy    | Method                                 | Description |
|-------------|----------------------------------------|-------------|
| Greedy      | `GenerationConfig::greedy()`           | Always takes the argmax |
| Sampling    | `GenerationConfig::sampling(temp)`     | Temperature-scaled softmax sampling |
| Top-K       | `GenerationConfig::top_k_sampling(k, temp)` | Sample from top-k tokens |
| Nucleus     | `GenerationConfig::nucleus_sampling(p, temp)` | Sample from top-p probability mass |
| Beam Search | `GenerationConfig::beam_search(beams)` | Beam search decoding |

---

## Tests

```bash
cargo test -p axonml-llm
cargo test -p axonml-llm -- --nocapture
```

---

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.

---

_Last updated: 2026-04-16 (v0.6.1)_
