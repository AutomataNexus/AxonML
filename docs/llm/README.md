# axonml-llm Documentation

> Large Language Model architectures for Axonml.

## Overview

`axonml-llm` provides production-ready implementations of popular LLM architectures including BERT (encoder-only) and GPT-2 (decoder-only). It includes pre-built model configurations, text generation utilities, and fine-tuning support.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Text Generation                      │
│        TextGenerator, GenerationConfig                   │
├─────────────────────────────────────────────────────────┤
│                    Task-Specific Heads                   │
│   BertForSequenceClassification, BertForMaskedLM        │
│   GPT2LMHead                                             │
├─────────────────────────────────────────────────────────┤
│                     Base Models                          │
│   Bert (encoder), GPT2 (decoder)                         │
├─────────────────────────────────────────────────────────┤
│                    Transformer Blocks                    │
│   BertLayer, GPT2Block                                   │
├─────────────────────────────────────────────────────────┤
│                    Core Components                       │
│   MultiHeadAttention, FeedForward, LayerNorm            │
├─────────────────────────────────────────────────────────┤
│                      Embeddings                          │
│   TokenEmbedding, PositionalEmbedding                   │
└─────────────────────────────────────────────────────────┘
```

## Modules

### config.rs

Model configuration for BERT and GPT-2.

```rust
pub struct BertConfig {
    pub vocab_size: usize,         // 30522
    pub hidden_size: usize,        // 768
    pub num_hidden_layers: usize,  // 12
    pub num_attention_heads: usize, // 12
    pub intermediate_size: usize,   // 3072
    pub hidden_dropout_prob: f32,   // 0.1
    pub attention_dropout_prob: f32, // 0.1
    pub max_position_embeddings: usize, // 512
    pub layer_norm_eps: f32,        // 1e-12
}

impl BertConfig {
    pub fn base() -> Self;      // BERT-base (110M params)
    pub fn large() -> Self;     // BERT-large (340M params)
    pub fn tiny() -> Self;      // BERT-tiny (for testing)
}

pub struct GPT2Config {
    pub vocab_size: usize,         // 50257
    pub n_positions: usize,        // 1024
    pub n_embd: usize,             // 768
    pub n_layer: usize,            // 12
    pub n_head: usize,             // 12
    pub dropout: f32,              // 0.1
    pub layer_norm_eps: f32,       // 1e-5
}

impl GPT2Config {
    pub fn small() -> Self;     // GPT-2 small (117M params)
    pub fn medium() -> Self;    // GPT-2 medium (345M params)
    pub fn large() -> Self;     // GPT-2 large (762M params)
    pub fn xl() -> Self;        // GPT-2 XL (1.5B params)
    pub fn tiny() -> Self;      // Tiny (for testing)
}
```

### bert.rs

BERT encoder model and task heads.

```rust
pub struct Bert {
    embeddings: BertEmbedding,
    layers: Vec<BertLayer>,
    pooler: Linear,
}

impl Bert {
    pub fn new(config: &BertConfig) -> Self;
    pub fn forward(&self, input_ids: &Variable, attention_mask: Option<&Variable>)
        -> (Variable, Variable);  // (sequence_output, pooled_output)
}

pub struct BertForSequenceClassification {
    bert: Bert,
    classifier: Linear,
    num_labels: usize,
}

impl BertForSequenceClassification {
    pub fn new(config: &BertConfig, num_labels: usize) -> Self;
    pub fn forward(&self, input_ids: &Variable, attention_mask: Option<&Variable>)
        -> Variable;  // logits [batch_size, num_labels]
}

pub struct BertForMaskedLM {
    bert: Bert,
    lm_head: Linear,
}

impl BertForMaskedLM {
    pub fn new(config: &BertConfig) -> Self;
    pub fn forward(&self, input_ids: &Variable, attention_mask: Option<&Variable>)
        -> Variable;  // logits [batch_size, seq_len, vocab_size]
}
```

### gpt2.rs

GPT-2 decoder model.

```rust
pub struct GPT2 {
    wte: Embedding,      // Token embeddings
    wpe: Embedding,      // Position embeddings
    blocks: Vec<GPT2Block>,
    ln_f: LayerNorm,     // Final layer norm
}

impl GPT2 {
    pub fn new(config: &GPT2Config) -> Self;
    pub fn forward(&self, input_ids: &Variable) -> Variable;
        // hidden_states [batch_size, seq_len, n_embd]
}

pub struct GPT2LMHead {
    transformer: GPT2,
    lm_head: Linear,  // Tied with wte
}

impl GPT2LMHead {
    pub fn new(config: &GPT2Config) -> Self;
    pub fn forward(&self, input_ids: &Variable) -> Variable;
        // logits [batch_size, seq_len, vocab_size]
}
```

### attention.rs

Multi-head attention implementations.

```rust
pub struct MultiHeadSelfAttention {
    num_heads: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    dropout: f32,
}

impl MultiHeadSelfAttention {
    pub fn new(hidden_size: usize, num_heads: usize, dropout: f32) -> Self;
    pub fn forward(&self, hidden_states: &Variable, attention_mask: Option<&Variable>)
        -> Variable;
}

pub struct CausalSelfAttention {
    // Same as MultiHeadSelfAttention but with causal masking
}

impl CausalSelfAttention {
    pub fn forward(&self, hidden_states: &Variable) -> Variable;
}
```

### generation.rs

Text generation utilities.

```rust
pub struct GenerationConfig {
    pub max_length: usize,       // Maximum tokens to generate
    pub min_length: usize,       // Minimum tokens
    pub do_sample: bool,         // Use sampling vs greedy
    pub temperature: f32,        // Sampling temperature
    pub top_k: Option<usize>,    // Top-k filtering
    pub top_p: Option<f32>,      // Nucleus (top-p) sampling
    pub repetition_penalty: f32, // Penalize repetition
    pub eos_token_id: Option<u32>, // End of sequence token
    pub pad_token_id: Option<u32>, // Padding token
}

impl GenerationConfig {
    pub fn greedy() -> Self;
    pub fn sampling(temperature: f32) -> Self;
    pub fn top_k_sampling(k: usize, temperature: f32) -> Self;
    pub fn nucleus_sampling(p: f32, temperature: f32) -> Self;
}

pub struct TextGenerator<M> {
    model: M,
    config: GenerationConfig,
}

impl<M: LanguageModel> TextGenerator<M> {
    pub fn new(model: M, config: GenerationConfig) -> Self;
    pub fn generate(&self, input_ids: &[u32]) -> Vec<u32>;
    pub fn generate_batch(&self, input_ids: &[Vec<u32>]) -> Vec<Vec<u32>>;
}
```

### embedding.rs

Embedding layers for transformers.

```rust
pub struct BertEmbedding {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: f32,
}

impl BertEmbedding {
    pub fn forward(&self, input_ids: &Variable, token_type_ids: Option<&Variable>)
        -> Variable;
}
```

## Usage

### BERT for Classification

```rust
use axonml_llm::{BertConfig, BertForSequenceClassification};
use axonml::prelude::*;

// Create model
let config = BertConfig::base();
let model = BertForSequenceClassification::new(&config, 2);  // Binary classification

// Prepare input (token IDs)
let input_ids = Variable::new(
    Tensor::from_vec(vec![101u32, 2054, 2003, 2023, 102], &[1, 5]).unwrap(),
    false
);

// Forward pass
let logits = model.forward(&input_ids, None);
// logits shape: [1, 2]
```

### BERT for Masked Language Modeling

```rust
use axonml_llm::{BertConfig, BertForMaskedLM};

let config = BertConfig::base();
let model = BertForMaskedLM::new(&config);

// Input with [MASK] token (103)
let input_ids = Variable::new(
    Tensor::from_vec(vec![101u32, 2023, 103, 1037, 3231, 102], &[1, 6]).unwrap(),
    false
);

let logits = model.forward(&input_ids, None);
// logits shape: [1, 6, 30522] - predictions for each position
```

### GPT-2 Text Generation

```rust
use axonml_llm::{GPT2Config, GPT2LMHead, GenerationConfig, TextGenerator};

// Create model
let config = GPT2Config::small();
let model = GPT2LMHead::new(&config);

// Create generator with sampling
let gen_config = GenerationConfig::top_k_sampling(50, 0.8);
let generator = TextGenerator::new(model, gen_config);

// Generate text
let prompt = vec![15496u32, 11, 314];  // "Hello, I"
let output = generator.generate(&prompt);
// output: token IDs for generated text
```

### Custom Generation Config

```rust
let config = GenerationConfig {
    max_length: 100,
    min_length: 10,
    do_sample: true,
    temperature: 0.9,
    top_k: Some(40),
    top_p: Some(0.95),
    repetition_penalty: 1.2,
    eos_token_id: Some(50256),
    pad_token_id: Some(50256),
};
```

### Fine-tuning BERT

```rust
use axonml_llm::{BertConfig, BertForSequenceClassification};
use axonml::prelude::*;

// Create model
let config = BertConfig::base();
let model = BertForSequenceClassification::new(&config, 3);  // 3 classes

// Create optimizer
let mut optimizer = Adam::new(model.parameters(), 2e-5);

// Training loop
for (input_ids, labels) in dataset {
    // Forward
    let logits = model.forward(&input_ids, None);

    // Compute loss
    let loss = cross_entropy_loss(&logits, &labels);

    // Backward
    loss.backward();

    // Update
    optimizer.step();
    optimizer.zero_grad();
}
```

## Model Sizes

### BERT Variants

| Variant | Layers | Hidden | Heads | Params |
|---------|--------|--------|-------|--------|
| tiny    | 2      | 128    | 2     | ~4M    |
| base    | 12     | 768    | 12    | ~110M  |
| large   | 24     | 1024   | 16    | ~340M  |

### GPT-2 Variants

| Variant | Layers | Hidden | Heads | Params |
|---------|--------|--------|-------|--------|
| tiny    | 2      | 64     | 2     | ~1M    |
| small   | 12     | 768    | 12    | ~117M  |
| medium  | 24     | 1024   | 16    | ~345M  |
| large   | 36     | 1280   | 20    | ~762M  |
| xl      | 48     | 1600   | 25    | ~1.5B  |

## Attention Patterns

### BERT (Bidirectional)
```
Tokens:  [CLS] The cat sat [SEP]
Attends:   ←────────────────→
Each token can attend to all other tokens.
```

### GPT-2 (Causal/Autoregressive)
```
Tokens:  The cat sat on
Attends: ←── ←── ←── ←──
Each token can only attend to previous tokens.
```

## Best Practices

1. **Use appropriate precision**: FP16 for inference, FP32 for training
2. **Batch for efficiency**: Process multiple sequences together
3. **Truncate long sequences**: Respect max_position_embeddings
4. **Use attention masks**: Properly mask padding tokens
5. **Gradient checkpointing**: For memory-constrained training

## Feature Flags

- Default: Basic BERT and GPT-2
- `pretrained` - Enable loading pretrained weights
- `flash-attention` - Use Flash Attention for efficiency
