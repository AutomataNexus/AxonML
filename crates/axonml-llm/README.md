# axonml-llm

[![Crates.io](https://img.shields.io/crates/v/axonml-llm.svg)](https://crates.io/crates/axonml-llm)
[![Docs.rs](https://docs.rs/axonml-llm/badge.svg)](https://docs.rs/axonml-llm)
[![Downloads](https://img.shields.io/crates/d/axonml-llm.svg)](https://crates.io/crates/axonml-llm)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Large language model architectures for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-llm` provides implementations of popular large language model architectures including BERT, GPT-2, LLaMA, and more. Includes tokenizers, pretrained weights loading, and optimized inference with KV-cache.

## Features

### Architectures
- **BERT** - Bidirectional encoder (base, large)
- **GPT-2** - Autoregressive decoder (small, medium, large, xl)
- **LLaMA** - Meta's LLaMA architecture
- **Mistral** - Mistral 7B architecture
- **Phi** - Microsoft Phi models

### Inference Optimizations
- **KV-cache** - Cached key-value for fast generation
- **Flash Attention** - Memory-efficient attention
- **Grouped Query Attention** - GQA support
- **Rotary Embeddings** - RoPE position encoding

### Tokenization
- **BPE tokenizer** - Byte-pair encoding
- **WordPiece** - BERT-style tokenization
- **SentencePiece** - Universal tokenizer
- **Tiktoken** - OpenAI-compatible tokenization

### Utilities
- **Weight loading** - Load from HuggingFace, GGUF, SafeTensors
- **Quantization** - INT8/INT4 inference
- **Streaming** - Token-by-token generation
- **Batched inference** - Process multiple sequences

## Installation

```toml
[dependencies]
axonml-llm = "0.1"
```

## Usage

### Text Generation with GPT-2

```rust
use axonml_llm::{GPT2, GPT2Tokenizer, GenerationConfig};

// Load model and tokenizer
let model = GPT2::from_pretrained("gpt2")?;
let tokenizer = GPT2Tokenizer::from_pretrained("gpt2")?;

// Encode input
let input_ids = tokenizer.encode("The future of AI is")?;

// Generate
let config = GenerationConfig {
    max_new_tokens: 50,
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};

let output_ids = model.generate(&input_ids, &config)?;
let text = tokenizer.decode(&output_ids)?;

println!("{}", text);
```

### BERT for Classification

```rust
use axonml_llm::{BertForSequenceClassification, BertTokenizer};

// Load fine-tuned model
let model = BertForSequenceClassification::from_pretrained(
    "bert-base-uncased",
    num_labels: 2,
)?;
let tokenizer = BertTokenizer::from_pretrained("bert-base-uncased")?;

// Tokenize
let encoding = tokenizer.encode_pair(
    "This movie was great!",
    None,
    max_length: 512,
)?;

// Classify
let logits = model.forward(&encoding.input_ids, &encoding.attention_mask)?;
let prediction = logits.argmax(-1);
```

### LLaMA with Quantization

```rust
use axonml_llm::{LlamaForCausalLM, LlamaTokenizer};
use axonml_quant::QuantFormat;

// Load quantized model (INT4)
let model = LlamaForCausalLM::from_pretrained("meta-llama/Llama-2-7b")?
    .quantize(QuantFormat::Q4_0)?;

let tokenizer = LlamaTokenizer::from_pretrained("meta-llama/Llama-2-7b")?;

// Generate with KV-cache
let mut kv_cache = model.create_kv_cache();

let input_ids = tokenizer.encode("Hello, how are you?")?;
let output = model.generate_with_cache(&input_ids, &mut kv_cache, 100)?;
```

### Streaming Generation

```rust
use axonml_llm::{GPT2, GenerationConfig};

let model = GPT2::from_pretrained("gpt2-medium")?;
let tokenizer = GPT2Tokenizer::from_pretrained("gpt2-medium")?;

let input_ids = tokenizer.encode("Once upon a time")?;

// Stream tokens one by one
for token in model.generate_stream(&input_ids, &config) {
    let text = tokenizer.decode_single(token?)?;
    print!("{}", text);
    std::io::stdout().flush()?;
}
```

### Embeddings with BERT

```rust
use axonml_llm::{BertModel, BertTokenizer};

let model = BertModel::from_pretrained("bert-base-uncased")?;
let tokenizer = BertTokenizer::from_pretrained("bert-base-uncased")?;

let encoding = tokenizer.encode("Hello world")?;

// Get embeddings
let output = model.forward(&encoding.input_ids, &encoding.attention_mask)?;

// Use [CLS] token embedding for sentence representation
let sentence_embedding = output.last_hidden_state.select(1, 0);

// Or use mean pooling
let mean_embedding = output.last_hidden_state.mean(1);
```

### Load GGUF Models

```rust
use axonml_llm::{load_gguf, GGUFModel};

// Load GGUF format (llama.cpp compatible)
let model = load_gguf("model.gguf")?;

let input_ids = vec![1, 15043, 29892, 920, 526, 366, 29973];
let output = model.generate(&input_ids, max_tokens: 100)?;
```

## Supported Models

| Model | Sizes | Features |
|-------|-------|----------|
| BERT | base, large | Embeddings, Classification, QA |
| GPT-2 | small, medium, large, xl | Generation |
| LLaMA | 7B, 13B, 70B | Generation, Chat |
| LLaMA 2 | 7B, 13B, 70B | Generation, Chat |
| Mistral | 7B | Generation, Chat |
| Phi-2 | 2.7B | Generation |

## API Reference

### GenerationConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 20 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `top_k` | 50 | Top-k sampling |
| `repetition_penalty` | 1.0 | Penalty for repeated tokens |
| `do_sample` | true | Use sampling vs greedy |

### Tokenizer Methods

| Method | Description |
|--------|-------------|
| `encode(text)` | Text to token IDs |
| `decode(ids)` | Token IDs to text |
| `encode_pair(a, b)` | Encode sentence pair |
| `vocab_size()` | Vocabulary size |

## CLI Usage

```bash
# Generate text
axonml llm generate --model gpt2 --prompt "Hello world"

# Interactive chat
axonml llm chat --model llama-2-7b-chat

# Convert model to GGUF
axonml llm convert model.safetensors --format gguf -o model.gguf

# Benchmark inference
axonml llm benchmark --model gpt2 --batch-size 8
```

## Part of Axonml

```toml
[dependencies]
axonml = { version = "0.1", features = ["llm"] }
```

## License

MIT OR Apache-2.0
