# axonml-text Documentation

> Natural language processing utilities for the Axonml ML framework.

## Overview

`axonml-text` provides text processing capabilities including tokenizers, vocabulary management, and text datasets. It's the Axonml equivalent of PyTorch's torchtext.

## Modules

### tokenizer.rs

Text tokenization implementations.

#### Tokenizer Trait

```rust
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
    fn encode(&self, text: &str, vocab: &Vocab) -> Vec<usize>;
    fn decode(&self, indices: &[usize], vocab: &Vocab) -> String;
}
```

#### WhitespaceTokenizer

Simple whitespace-based tokenization:

```rust
use axonml_text::WhitespaceTokenizer;

let tokenizer = WhitespaceTokenizer::new();
let tokens = tokenizer.tokenize("Hello, world!");
// Result: ["Hello,", "world!"]
```

#### CharTokenizer

Character-level tokenization:

```rust
use axonml_text::CharTokenizer;

let tokenizer = CharTokenizer::new();
let tokens = tokenizer.tokenize("Hello");
// Result: ["H", "e", "l", "l", "o"]
```

#### BasicBPETokenizer

Byte-Pair Encoding tokenizer:

```rust
use axonml_text::BasicBPETokenizer;

let mut bpe = BasicBPETokenizer::new();

// Train on corpus
bpe.train("low lower lowest newer newest", 10);

// Tokenize
let tokens = bpe.tokenize("lower");
println!("Tokens: {:?}", tokens);
println!("Vocab size: {}", bpe.get_vocab().len());
```

### vocab.rs

Vocabulary management for mapping tokens to indices.

```rust
use axonml_text::Vocab;

// Build vocabulary from text
let corpus = "the quick brown fox jumps over the lazy dog";
let vocab = Vocab::from_text(corpus, min_freq);

// Vocabulary operations
println!("Size: {}", vocab.len());
println!("'the' index: {:?}", vocab.token_to_index("the"));
println!("Index 0 token: {:?}", vocab.index_to_token(0));

// Special tokens
let unk_idx = vocab.unk_index();
let pad_idx = vocab.pad_index();
```

#### Building Custom Vocabulary

```rust
use axonml_text::Vocab;

let mut vocab = Vocab::new();

// Add special tokens
vocab.add_special_token("<pad>");
vocab.add_special_token("<unk>");
vocab.add_special_token("<bos>");
vocab.add_special_token("<eos>");

// Add tokens from corpus
vocab.add_token("hello");
vocab.add_token("world");
```

### datasets/

Text datasets for common NLP tasks.

#### TextDataset

Basic text dataset:

```rust
use axonml_text::TextDataset;

let dataset = TextDataset::from_texts(texts, labels, vocab, max_len);

let (text_tensor, label) = dataset.get(0).unwrap();
```

#### LanguageModelDataset

Dataset for language modeling (next token prediction):

```rust
use axonml_text::LanguageModelDataset;

let dataset = LanguageModelDataset::from_text(corpus, vocab, seq_len);

// Returns (input_sequence, target_sequence)
let (input, target) = dataset.get(0).unwrap();
```

#### SyntheticSentimentDataset

Synthetic sentiment analysis dataset:

```rust
use axonml_text::SyntheticSentimentDataset;

let dataset = SyntheticSentimentDataset::small();

let (text_tensor, sentiment) = dataset.get(0).unwrap();
// text_tensor: [max_len] encoded text
// sentiment: [2] binary sentiment (positive/negative)
```

## Usage Examples

### Text Classification

```rust
use axonml::prelude::*;

fn main() {
    // 1. Create dataset
    let dataset = SyntheticSentimentDataset::small();
    let loader = DataLoader::with_shuffle(dataset, 32, true);

    // 2. Create model
    let vocab_size = 1000;
    let embed_dim = 128;
    let hidden_size = 256;
    let num_classes = 2;

    let embedding = Embedding::new(vocab_size, embed_dim);
    let lstm = LSTM::new(embed_dim, hidden_size);
    let classifier = Linear::new(hidden_size, num_classes);

    // 3. Training loop
    let mut optimizer = Adam::new(
        [
            embedding.parameters(),
            lstm.parameters(),
            classifier.parameters(),
        ].concat(),
        0.001
    );

    for batch in loader.iter() {
        // Embed tokens
        let embedded = embedding.forward(&batch.data);

        // Process with LSTM
        let (output, _) = lstm.forward_seq(&embedded);

        // Use last hidden state for classification
        let last = output.select(1, -1)?;
        let logits = classifier.forward(&last);

        let loss = cross_entropy(&logits, &batch.targets);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

### Tokenization and Encoding

```rust
use axonml::prelude::*;

// Build vocabulary from training data
let train_corpus = "the quick brown fox jumps over the lazy dog";
let vocab = Vocab::from_text(train_corpus, 1);

// Tokenize and encode
let tokenizer = WhitespaceTokenizer::new();
let text = "the quick fox";

let tokens = tokenizer.tokenize(text);
println!("Tokens: {:?}", tokens);

let encoded = tokenizer.encode(text, &vocab);
println!("Encoded: {:?}", encoded);

let decoded = tokenizer.decode(&encoded, &vocab);
println!("Decoded: {}", decoded);
```

### BPE Training

```rust
use axonml_text::BasicBPETokenizer;

// Training corpus
let corpus = r#"
    low lower lowest
    new newer newest
    show showed shown
    slow slower slowest
"#;

// Train BPE
let mut bpe = BasicBPETokenizer::new();
bpe.train(corpus, 50);  // Learn 50 merge operations

// Tokenize new text
let tokens = bpe.tokenize("showing");
println!("BPE tokens: {:?}", tokens);
```

### Language Modeling

```rust
use axonml::prelude::*;

// Create language model dataset
let corpus = load_text_file("corpus.txt")?;
let vocab = Vocab::from_text(&corpus, 5);  // min_freq = 5
let dataset = LanguageModelDataset::from_text(&corpus, &vocab, 128);

// Create model
let model = create_transformer_lm(vocab.len(), 512, 8, 6);

// Training
let loader = DataLoader::with_shuffle(dataset, 32, true);
let mut optimizer = AdamW::new(model.parameters(), 0.0001);

for batch in loader.iter() {
    let logits = model.forward(&batch.data);
    let loss = cross_entropy(&logits, &batch.targets);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

## Special Tokens

Standard special tokens used in NLP:

| Token | Purpose |
|-------|---------|
| `<pad>` | Padding for batch alignment |
| `<unk>` | Unknown/out-of-vocabulary words |
| `<bos>` | Beginning of sequence |
| `<eos>` | End of sequence |
| `<mask>` | Masked token (for MLM) |

## Related Modules

- [Data](../data/README.md) - DataLoader and Dataset traits
- [Neural Networks](../nn/README.md) - Embedding, RNN, Attention
- [Autograd](../autograd/README.md) - Training with gradients

@version 0.1.0
@author AutomataNexus Development Team
