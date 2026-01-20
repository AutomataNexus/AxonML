# axonml-llm

<!-- Logo placeholder -->
<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust 1.75+"></a>
  <a href="https://crates.io/crates/axonml-llm"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Crate Version"></a>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

`axonml-llm` provides implementations of popular transformer-based large language model architectures for the AxonML framework. This crate includes complete implementations of BERT and GPT-2 models, along with modular building blocks for constructing custom LLM architectures.

Built on top of `axonml-tensor` and `axonml-autograd`, this crate enables training and inference of transformer models entirely in pure Rust.

---

## Features

- **BERT Implementation** - Full Bidirectional Encoder Representations from Transformers with support for sequence classification, masked language modeling, and custom heads.

- **GPT-2 Implementation** - Complete Generative Pre-trained Transformer 2 with all model sizes (Small, Medium, Large, XL) and language modeling head.

- **Multi-Head Attention** - Efficient multi-head self-attention and causal self-attention implementations with configurable heads and dimensions.

- **Transformer Building Blocks** - Modular encoder and decoder blocks with layer normalization, feed-forward networks, and residual connections.

- **Embedding Layers** - Token embeddings, learned positional embeddings, sinusoidal positional encodings, and BERT/GPT-2 combined embeddings.

- **Text Generation** - Comprehensive generation utilities including greedy decoding, temperature sampling, top-k/top-p filtering, and beam search.

- **Configurable Architectures** - Pre-defined configurations for BERT-base, BERT-large, GPT-2 Small/Medium/Large/XL, and tiny variants for testing.

---

## Modules

| Module | Description |
|--------|-------------|
| `attention` | Multi-head self-attention and causal self-attention mechanisms |
| `bert` | BERT model with classification and masked LM variants |
| `config` | Configuration structs for BERT, GPT-2, and base transformers |
| `embedding` | Token, positional, and combined embedding layers |
| `error` | Error types and result definitions for LLM operations |
| `generation` | Text generation utilities, sampling strategies, and beam search |
| `gpt2` | GPT-2 model with language modeling head |
| `transformer` | Encoder/decoder blocks, layer norm, and feed-forward networks |

---

## Usage

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
axonml-llm = "0.1.0"
```

### GPT-2 Text Generation

```rust
use axonml_llm::{GPT2LMHead, GPT2Config};
use axonml_tensor::Tensor;

// Create a GPT-2 model
let config = GPT2Config::small();
let model = GPT2LMHead::new(&config);

// Input token IDs
let input_ids = Tensor::from_vec(vec![50256u32, 1, 2, 3], &[1, 4]).unwrap();

// Generate text with sampling
let output = model.generate(&input_ids, 50, 0.8, Some(50));
println!("Generated tokens: {:?}", output.to_vec());

// Or use greedy decoding
let greedy_output = model.generate_greedy(&input_ids, 50);
```

### BERT for Sequence Classification

```rust
use axonml_llm::{BertForSequenceClassification, BertConfig};
use axonml_tensor::Tensor;

// Create BERT for binary classification
let config = BertConfig::base();
let model = BertForSequenceClassification::new(&config, 2);

// Input token IDs
let input_ids = Tensor::from_vec(vec![101u32, 2054, 2003, 1996, 102], &[1, 5]).unwrap();

// Get classification logits
let logits = model.forward_classification(&input_ids);
println!("Logits shape: {:?}", logits.data().shape());
```

### Custom Transformer Encoder

```rust
use axonml_llm::{TransformerEncoder, MultiHeadSelfAttention};
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// Build a custom encoder stack
let encoder = TransformerEncoder::new(
    6,          // num_layers
    512,        // hidden_size
    8,          // num_heads
    2048,       // intermediate_size
    0.1,        // dropout
    1e-12,      // layer_norm_eps
    "gelu",     // activation
    false,      // pre_norm
);

// Forward pass
let input = Variable::new(Tensor::randn(&[2, 128, 512]), false);
let output = encoder.forward(&input);
```

### Generation Configuration

```rust
use axonml_llm::{GenerationConfig, TextGenerator};

// Configure generation parameters
let config = GenerationConfig::nucleus_sampling(0.95, 0.8)
    .with_max_tokens(100)
    .with_repetition_penalty(1.2)
    .with_eos_token(50256);

let generator = TextGenerator::new(config);

// Use with model logits
let next_token = generator.get_next_token(&logits, &generated_so_far);
```

### BERT for Masked Language Modeling

```rust
use axonml_llm::{BertForMaskedLM, BertConfig};
use axonml_tensor::Tensor;

// Create BERT for MLM
let config = BertConfig::base();
let model = BertForMaskedLM::new(&config);

// Input with [MASK] token
let input_ids = Tensor::from_vec(
    vec![101u32, 2054, 103, 1996, 102], // 103 = [MASK]
    &[1, 5]
).unwrap();

// Get MLM logits
let logits = model.forward_mlm(&input_ids);
// Shape: [batch, seq_len, vocab_size]
```

---

## Model Configurations

### BERT Configurations

| Config | Hidden Size | Layers | Heads | Parameters |
|--------|-------------|--------|-------|------------|
| `BertConfig::tiny()` | 128 | 2 | 2 | ~4M |
| `BertConfig::base()` | 768 | 12 | 12 | ~110M |
| `BertConfig::large()` | 1024 | 24 | 16 | ~340M |

### GPT-2 Configurations

| Config | Embedding Dim | Layers | Heads | Parameters |
|--------|---------------|--------|-------|------------|
| `GPT2Config::tiny()` | 128 | 2 | 2 | ~4M |
| `GPT2Config::small()` | 768 | 12 | 12 | ~117M |
| `GPT2Config::medium()` | 1024 | 24 | 16 | ~345M |
| `GPT2Config::large()` | 1280 | 36 | 20 | ~774M |
| `GPT2Config::xl()` | 1600 | 48 | 25 | ~1.5B |

---

## Generation Strategies

| Strategy | Method | Description |
|----------|--------|-------------|
| Greedy | `GenerationConfig::greedy()` | Always selects highest probability token |
| Sampling | `GenerationConfig::sampling(temp)` | Temperature-controlled sampling |
| Top-K | `GenerationConfig::top_k_sampling(k, temp)` | Sample from top-k tokens |
| Nucleus | `GenerationConfig::nucleus_sampling(p, temp)` | Sample from top-p probability mass |
| Beam Search | `GenerationConfig::beam_search(beams)` | Beam search decoding |

---

## Tests

Run the test suite:

```bash
cargo test -p axonml-llm
```

Run with verbose output:

```bash
cargo test -p axonml-llm -- --nocapture
```

---

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
