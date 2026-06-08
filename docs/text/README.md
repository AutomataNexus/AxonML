# axonml-text Documentation

> Natural language processing utilities for the AxonML ML framework.

## Overview

`axonml-text` provides six tokenizers, a `Vocab` with special-token
management, and four datasets (classification, language modelling, and
seq2seq). The AxonML counterpart to `torchtext`.

## Modules

### `tokenizer`

All tokenizers implement:

```rust
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
    fn encode(&self, text: &str, vocab: &Vocab) -> Vec<usize>;
    fn decode(&self, indices: &[usize], vocab: &Vocab) -> String;
}
```

Available implementations:

| Tokenizer              | Notes                                                           |
|------------------------|-----------------------------------------------------------------|
| `WhitespaceTokenizer`  | Split on whitespace — simplest baseline                         |
| `CharTokenizer`        | One token per Unicode character                                 |
| `WordPunctTokenizer`   | Separates words and punctuation                                 |
| `NGramTokenizer`       | Configurable word- or char-level n-grams                        |
| `BasicBPETokenizer`    | Byte-Pair Encoding; `train(corpus, num_merges)` learns merges   |
| `UnigramTokenizer`     | Unigram language-model tokenizer                                |

```rust
use axonml_text::{WhitespaceTokenizer, CharTokenizer, BasicBPETokenizer, NGramTokenizer};

let ws = WhitespaceTokenizer::new();
let chars = CharTokenizer::new();

let mut bpe = BasicBPETokenizer::new();
bpe.train("low lower lowest newer newest", 10);
let tokens = bpe.tokenize("lower");

let bigrams = NGramTokenizer::word_ngrams(2);
let trigrams = NGramTokenizer::char_ngrams(3);
```

### `vocab`

`Vocab` maps tokens to indices and back. Special tokens are defined as
string constants: `PAD_TOKEN`, `UNK_TOKEN`, `BOS_TOKEN`, `EOS_TOKEN`,
`MASK_TOKEN`.

```rust
use axonml_text::{Vocab, PAD_TOKEN, UNK_TOKEN};

let vocab = Vocab::from_text("the quick brown fox", 1); // min_freq=1

vocab.len();
vocab.token_to_index("the");
vocab.index_to_token(0);
vocab.unk_index();
vocab.pad_index();

let mut v = Vocab::with_special_tokens();
v.add_token("hello");
v.add_token("world");
```

### `datasets`

All implement `axonml_data::Dataset`.

#### `TextDataset`

Token-ID-encoded classification dataset.

```rust
use axonml_text::{TextDataset, WhitespaceTokenizer};

let samples: Vec<(String, usize)> = vec![
    ("good movie".into(), 1),
    ("bad movie".into(), 0),
];
let tokenizer = WhitespaceTokenizer::new();
let dataset = TextDataset::from_samples(&samples, &tokenizer, 1 /*min_freq*/, 10 /*max_len*/);
```

#### `LanguageModelDataset`

Next-token prediction. `from_text(text, seq_len, min_freq)` returns
`(input_seq, target_seq)` pairs of shape `[seq_len]`.

```rust
use axonml_text::LanguageModelDataset;
let ds = LanguageModelDataset::from_text(&corpus, 128, 5);
```

#### `SyntheticSentimentDataset`

Synthetic binary-sentiment dataset for smoke tests.

```rust
use axonml_text::SyntheticSentimentDataset;
let ds = SyntheticSentimentDataset::small();
```

#### `SyntheticSeq2SeqDataset`

Synthetic sequence-to-sequence tasks (copy / reverse) for seq2seq model
testing.

```rust
use axonml_text::SyntheticSeq2SeqDataset;
let ds = SyntheticSeq2SeqDataset::copy_task(num_samples, seq_len, vocab_size);
```

## Usage Examples

### Tokenize and encode

```rust
use axonml::prelude::*;
use axonml_text::{Vocab, WhitespaceTokenizer, Tokenizer};

let vocab = Vocab::from_text("the quick brown fox jumps over the lazy dog", 1);
let tok = WhitespaceTokenizer::new();

let tokens  = tok.tokenize("the quick fox");
let encoded = tok.encode("the quick fox", &vocab);
let decoded = tok.decode(&encoded, &vocab);
```

### BPE training

```rust
use axonml_text::BasicBPETokenizer;

let corpus = "low lower lowest new newer newest show showed shown slow slower slowest";
let mut bpe = BasicBPETokenizer::new();
bpe.train(corpus, 50);

let tokens = bpe.tokenize("showing");
```

### Sentiment classification pipeline

```rust
use axonml::prelude::*;
use axonml_text::SyntheticSentimentDataset;

let ds = SyntheticSentimentDataset::small();
let loader = DataLoader::with_shuffle(ds, 32, true);

let vocab_size  = 1000;
let embed_dim   = 128;
let hidden_size = 256;

let embedding  = Embedding::new(vocab_size, embed_dim);
let lstm       = LSTM::new(embed_dim, hidden_size, 1);
let classifier = Linear::new(hidden_size, 2);

let params = [
    embedding.parameters(),
    lstm.parameters(),
    classifier.parameters(),
].concat();
let mut opt = Adam::new(params, 0.001);

for batch in loader.iter() {
    let emb  = embedding.forward(&batch.data);
    let out  = lstm.forward(&emb);
    let last = out.select(1, -1).unwrap();
    let logits = classifier.forward(&last);

    let loss = cross_entropy(&logits, &batch.targets);
    loss.backward();
    opt.step();
    opt.zero_grad();
}
```

### Language modelling

```rust
use axonml::prelude::*;
use axonml_text::{LanguageModelDataset, Vocab};

let corpus = std::fs::read_to_string("corpus.txt")?;
let vocab = Vocab::from_text(&corpus, 5);
let ds = LanguageModelDataset::from_text(&corpus, 128, 5);

let model = create_transformer_lm(vocab.len(), 512, 8, 6);
let loader = DataLoader::with_shuffle(ds, 32, true);
let mut opt = AdamW::new(model.parameters(), 0.0001);

for batch in loader.iter() {
    let logits = model.forward(&batch.data);
    let loss = cross_entropy(&logits, &batch.targets);
    loss.backward();
    opt.step();
    opt.zero_grad();
}
```

## Special Tokens

| Constant      | String   | Purpose                         |
|---------------|----------|---------------------------------|
| `PAD_TOKEN`   | `<pad>`  | Batch alignment padding         |
| `UNK_TOKEN`   | `<unk>`  | Out-of-vocabulary placeholder   |
| `BOS_TOKEN`   | `<bos>`  | Beginning of sequence           |
| `EOS_TOKEN`   | `<eos>`  | End of sequence                 |
| `MASK_TOKEN`  | `<mask>` | Masked token (MLM)              |

## Related Modules

- [Data](../../crates/axonml-data) — `DataLoader`, `Dataset`
- [Neural Networks](../nn/README.md) — `Embedding`, `RNN`, `Attention`
- [LLM](../llm/README.md) — large language model training / inference

## Last updated

0.6.5 (2026-06-06)
