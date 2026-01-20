# axonml-text

<p align="center">
  <!-- Logo placeholder -->
  <img src="../../assets/logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part%20of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

`axonml-text` provides natural language processing utilities for the AxonML machine learning framework. It includes vocabulary management, multiple tokenization strategies, and dataset implementations for common NLP tasks like text classification, language modeling, and sequence-to-sequence learning.

## Features

- **Vocabulary Management** - Token-to-index mapping with special tokens (PAD, UNK, BOS, EOS, MASK) and frequency-based filtering
- **Multiple Tokenizers** - Whitespace, character-level, word-punctuation, n-gram, BPE, and unigram tokenization strategies
- **Text Classification Datasets** - Build datasets from labeled text samples with automatic vocabulary construction
- **Language Modeling Datasets** - Create next-token prediction datasets with configurable sequence lengths
- **Synthetic Datasets** - Pre-built sentiment and seq2seq datasets for testing and prototyping
- **Prelude Module** - Convenient re-exports for common imports

## Modules

| Module | Description |
|--------|-------------|
| `vocab` | Vocabulary management with token-to-index mapping and special token support |
| `tokenizer` | Tokenizer trait and implementations (Whitespace, Char, WordPunct, NGram, BPE, Unigram) |
| `datasets` | Dataset implementations for text classification, language modeling, and seq2seq tasks |

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-text = "0.1.0"
```

### Building a Vocabulary

```rust
use axonml_text::prelude::*;

// Build vocabulary from text with minimum frequency threshold
let text = "the quick brown fox jumps over the lazy dog";
let vocab = Vocab::from_text(text, 1);

// Or create with special tokens
let mut vocab = Vocab::with_special_tokens();
vocab.add_token("hello");
vocab.add_token("world");

// Encode and decode
let indices = vocab.encode(&["hello", "world"]);
let tokens = vocab.decode(&indices);
```

### Tokenization

```rust
use axonml_text::prelude::*;

// Whitespace tokenizer
let tokenizer = WhitespaceTokenizer::new();
let tokens = tokenizer.tokenize("Hello World");  // ["Hello", "World"]

// Character-level tokenizer
let char_tokenizer = CharTokenizer::new();
let chars = char_tokenizer.tokenize("Hi!");  // ["H", "i", "!"]

// Word-punctuation tokenizer
let wp_tokenizer = WordPunctTokenizer::lowercase();
let tokens = wp_tokenizer.tokenize("Hello, World!");  // ["hello", ",", "world", "!"]

// N-gram tokenizer
let bigrams = NGramTokenizer::word_ngrams(2);
let tokens = bigrams.tokenize("one two three");  // ["one two", "two three"]

// BPE tokenizer
let mut bpe = BasicBPETokenizer::new();
bpe.train("low lower lowest newer newest", 10);
let tokens = bpe.tokenize("lowest");
```

### Text Classification Dataset

```rust
use axonml_text::prelude::*;

let samples = vec![
    ("good movie".to_string(), 1),
    ("bad movie".to_string(), 0),
    ("great film".to_string(), 1),
    ("terrible movie".to_string(), 0),
];

let tokenizer = WhitespaceTokenizer::new();
let dataset = TextDataset::from_samples(&samples, &tokenizer, 1, 10);

// Use with DataLoader
let loader = DataLoader::new(dataset, 16);
for batch in loader.iter() {
    // batch.data: [batch_size, max_length]
    // batch.target: [batch_size, num_classes]
}
```

### Language Modeling Dataset

```rust
use axonml_text::prelude::*;

let text = "one two three four five six seven eight nine ten";
let dataset = LanguageModelDataset::from_text(text, 3, 1);

let (input, target) = dataset.get(0).unwrap();
// input: [seq_length] - tokens at positions 0..seq_length
// target: [seq_length] - tokens at positions 1..seq_length+1
```

### Synthetic Datasets

```rust
use axonml_text::prelude::*;

// Sentiment dataset for testing
let sentiment = SyntheticSentimentDataset::small();  // 100 samples
let sentiment = SyntheticSentimentDataset::train();  // 10000 samples

// Seq2seq copy/reverse task
let seq2seq = SyntheticSeq2SeqDataset::copy_task(100, 5, 50);
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-text
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
