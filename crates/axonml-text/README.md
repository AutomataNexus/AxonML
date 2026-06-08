# axonml-text

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.6.5-green.svg" alt="Version 0.6.5">
  <img src="https://img.shields.io/badge/part%20of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

`axonml-text` provides natural language processing utilities for the AxonML machine learning framework: a serializable `Vocab`, six tokenizers, and three dataset families (classification, language modeling, synthetic seq2seq). Labels are emitted as class-index tensors of shape `[1]` — directly compatible with AxonML's `CrossEntropyLoss`.

## Features

- **Vocabulary management** — `Vocab` with token/index maps, special-token indices (PAD, UNK, BOS, EOS, MASK), frequency-threshold construction via `Vocab::from_text`, auto-inserted UNK/PAD in `from_tokens`, and JSON save/load (`serde`).
- **Six tokenizers** implementing a common `Tokenizer` trait:
  - `WhitespaceTokenizer` (with optional lowercasing)
  - `CharTokenizer` (optional whitespace filtering)
  - `WordPunctTokenizer` (separates words and punctuation)
  - `NGramTokenizer` (word- or character-level n-grams)
  - `BasicBPETokenizer` (trainable byte-pair encoding with priority-ordered merges and `</w>` end markers)
  - `UnigramTokenizer` (Viterbi-optimal segmentation from scored vocabulary)
- **Text classification dataset** — `TextDataset` stores a tokenizer and pads/truncates to `max_length`; `from_samples` builds the vocab from tokenized text with a `min_freq` threshold.
- **Language modeling dataset** — `LanguageModelDataset` produces next-token (input, target) pairs of shape `[seq_length]`.
- **Synthetic datasets** — `SyntheticSentimentDataset` (small/train/test presets, deterministic per-index generation) and `SyntheticSeq2SeqDataset` (reverse / copy task).
- **Prelude module** for concise imports.

## Modules

| Module | Description |
|--------|-------------|
| `vocab` | `Vocab` struct, special-token constants (`PAD_TOKEN`, `UNK_TOKEN`, `BOS_TOKEN`, `EOS_TOKEN`, `MASK_TOKEN`), JSON save/load |
| `tokenizer` | `Tokenizer` trait plus Whitespace, Char, WordPunct, NGram, BasicBPE, Unigram implementations |
| `datasets` | `TextDataset`, `LanguageModelDataset`, `SyntheticSentimentDataset`, `SyntheticSeq2SeqDataset` |

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-text = "0.6.5"
```

### Building a Vocabulary

```rust
use axonml_text::prelude::*;

// Frequency-threshold construction (adds special tokens automatically)
let text = "the quick brown fox jumps over the lazy dog";
let vocab = Vocab::from_text(text, /*min_freq=*/ 1);

// Or build manually
let mut vocab = Vocab::with_special_tokens();
vocab.add_token("hello");
vocab.add_token("world");

// Encode and decode
let indices = vocab.encode(&["hello", "world"]);
let tokens = vocab.decode(&indices);

// Unknown tokens resolve to the UNK index
assert_eq!(vocab.token_to_index("foo"), vocab.unk_index().unwrap());

// Persistence
vocab.save(std::path::Path::new("vocab.json")).unwrap();
let loaded = Vocab::load(std::path::Path::new("vocab.json")).unwrap();
```

### Tokenization

```rust
use axonml_text::prelude::*;

let ws = WhitespaceTokenizer::new();
let tokens = ws.tokenize("Hello World");            // ["Hello", "World"]

let chars = CharTokenizer::new();
let t = chars.tokenize("Hi!");                      // ["H", "i", "!"]

let wp = WordPunctTokenizer::lowercase();
let t = wp.tokenize("Hello, World!");               // ["hello", ",", "world", "!"]

let bigrams = NGramTokenizer::word_ngrams(2);
let t = bigrams.tokenize("one two three");          // ["one two", "two three"]

let trigrams = NGramTokenizer::char_ngrams(3);
let t = trigrams.tokenize("hello");                 // ["hel", "ell", "llo"]

// Trainable BPE
let mut bpe = BasicBPETokenizer::new();
bpe.train("low lower lowest newer newest", /*num_merges=*/ 10);
let tokens = bpe.tokenize("lowest");                // uses priority-ordered merges

// Unigram (Viterbi-optimal segmentation)
let unigram = UnigramTokenizer::from_tokens(&["hel", "lo", "wor", "ld"]);
let tokens = unigram.tokenize("hello world");
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
// Builds vocab from tokenized samples with min_freq=1, pads/truncates to 10.
let dataset = TextDataset::from_samples(&samples, &tokenizer, 1, 10);

assert_eq!(dataset.num_classes(), 2);

let loader = DataLoader::new(dataset, 16);
for batch in loader.iter() {
    // batch.data    : [batch_size, max_length] (float token indices)
    // batch.targets : [batch_size, 1]          (float class index)
}
```

### Language Modeling Dataset

```rust
use axonml_text::prelude::*;

let text = "one two three four five six seven eight nine ten";
let dataset = LanguageModelDataset::from_text(text, /*seq_len=*/ 3, /*min_freq=*/ 1);

let (input, target) = dataset.get(0).unwrap();
// input  : [seq_length] — tokens at positions i..i+seq_length
// target : [seq_length] — tokens at positions i+1..i+seq_length+1
```

### Synthetic Datasets

```rust
use axonml_text::prelude::*;

// Deterministic sentiment dataset (binary, reproducible per-index)
let sentiment = SyntheticSentimentDataset::small(); // 100 samples, max_len=32, vocab=1000
let sentiment = SyntheticSentimentDataset::train(); // 10000 samples, max_len=64, vocab=10000
let sentiment = SyntheticSentimentDataset::test();  // 2000 samples

// Seq2seq reverse (copy_task makes src_len == tgt_len)
let seq2seq = SyntheticSeq2SeqDataset::copy_task(/*size=*/ 100, /*length=*/ 5, /*vocab_size=*/ 50);
// For each sample: tgt is src reversed.
```

## Tests

```bash
cargo test -p axonml-text
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

---

_Last updated: 2026-06-06 (v0.6.5)_
