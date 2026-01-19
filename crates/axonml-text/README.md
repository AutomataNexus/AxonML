# axonml-text

[![Crates.io](https://img.shields.io/crates/v/axonml-text.svg)](https://crates.io/crates/axonml-text)
[![Docs.rs](https://docs.rs/axonml-text/badge.svg)](https://docs.rs/axonml-text)
[![Downloads](https://img.shields.io/crates/d/axonml-text.svg)](https://crates.io/crates/axonml-text)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> NLP utilities for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-text` provides text processing utilities including tokenizers, vocabulary management, and text datasets for natural language processing tasks.

## Features

### Tokenizers
- **WhitespaceTokenizer** - Split on whitespace
- **CharTokenizer** - Character-level tokenization
- **BPETokenizer** - Byte-pair encoding

### Vocabulary
- **Vocabulary** - Token to index mapping
- **Special tokens** - PAD, UNK, BOS, EOS, MASK
- **Frequency filtering** - Min/max frequency cutoffs

### Datasets
- **SyntheticSentimentDataset** - Sentiment classification

## Installation

```toml
[dependencies]
axonml-text = "0.1"
```

## Usage

### Whitespace Tokenizer

```rust
use axonml_text::tokenizers::WhitespaceTokenizer;
use axonml_text::Vocabulary;

let tokenizer = WhitespaceTokenizer::new();

// Build vocabulary from corpus
let texts = vec!["hello world", "hello rust"];
let vocab = Vocabulary::build_from_texts(&texts, &tokenizer)
    .min_freq(1)
    .max_size(10000)
    .add_special_tokens(&["<pad>", "<unk>"]);

// Tokenize and encode
let tokens = tokenizer.tokenize("hello world");  // ["hello", "world"]
let ids = vocab.encode(&tokens);                  // [2, 3]
```

### BPE Tokenizer

```rust
use axonml_text::tokenizers::BPETokenizer;

let tokenizer = BPETokenizer::new()
    .vocab_size(8000)
    .train(&corpus);

let tokens = tokenizer.tokenize("unbelievable");  // ["un", "believ", "able"]
let ids = tokenizer.encode("hello world");
let text = tokenizer.decode(&ids);
```

### Sentiment Dataset

```rust
use axonml_text::datasets::SyntheticSentimentDataset;
use axonml_data::DataLoader;

let dataset = SyntheticSentimentDataset::new(true, 10000);  // train, samples
let dataloader = DataLoader::new(dataset, 32).shuffle(true);

for (text_ids, label) in dataloader.iter() {
    // text_ids: [32, max_len] padded token ids
    // label: [32] (0=negative, 1=positive)
}
```

### Text Preprocessing

```rust
use axonml_text::preprocessing::{lowercase, remove_punctuation, pad_sequence};

let text = "Hello, World!";
let processed = lowercase(&remove_punctuation(text));  // "hello world"

// Pad sequences to same length
let sequences = vec![vec![1, 2, 3], vec![1, 2]];
let padded = pad_sequence(&sequences, 0, 5);  // [[1,2,3,0,0], [1,2,0,0,0]]
```

## API Reference

### Tokenizers

| Tokenizer | Description |
|-----------|-------------|
| `WhitespaceTokenizer` | Split on whitespace |
| `CharTokenizer` | Character-level |
| `BPETokenizer` | Byte-pair encoding |

### Vocabulary Methods

| Method | Description |
|--------|-------------|
| `build_from_texts()` | Build from corpus |
| `encode(tokens)` | Tokens to indices |
| `decode(ids)` | Indices to tokens |
| `get_id(token)` | Get single token ID |
| `get_token(id)` | Get token from ID |
| `size()` | Vocabulary size |

## Part of Axonml

```toml
[dependencies]
axonml = { version = "0.1", features = ["text"] }
```

## License

MIT OR Apache-2.0
