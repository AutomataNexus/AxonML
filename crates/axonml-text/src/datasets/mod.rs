//! Text Datasets
//!
//! Provides dataset implementations for common NLP tasks.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;
use axonml_data::Dataset;
use axonml_tensor::Tensor;

// =============================================================================
// TextDataset
// =============================================================================

/// A dataset of text samples with labels.
pub struct TextDataset {
    texts: Vec<String>,
    labels: Vec<usize>,
    vocab: Vocab,
    max_length: usize,
    num_classes: usize,
}

impl TextDataset {
    /// Creates a new `TextDataset`.
    #[must_use]
    pub fn new(texts: Vec<String>, labels: Vec<usize>, vocab: Vocab, max_length: usize) -> Self {
        let num_classes = labels.iter().max().map_or(0, |&m| m + 1);
        Self {
            texts,
            labels,
            vocab,
            max_length,
            num_classes,
        }
    }

    /// Creates a `TextDataset` from raw text samples with a tokenizer.
    pub fn from_samples<T: Tokenizer>(
        samples: &[(String, usize)],
        tokenizer: &T,
        min_freq: usize,
        max_length: usize,
    ) -> Self {
        use std::collections::HashMap;

        // Build vocabulary from tokenized text
        let mut freq: HashMap<String, usize> = HashMap::new();
        for (text, _) in samples {
            for token in tokenizer.tokenize(text) {
                *freq.entry(token).or_insert(0) += 1;
            }
        }

        // Create vocabulary with tokens meeting min_freq threshold
        let mut vocab = Vocab::with_special_tokens();
        let mut tokens: Vec<_> = freq
            .into_iter()
            .filter(|(_, count)| *count >= min_freq)
            .collect();
        tokens.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (token, _) in tokens {
            vocab.add_token(&token);
        }

        let texts: Vec<String> = samples.iter().map(|(t, _)| t.clone()).collect();
        let labels: Vec<usize> = samples.iter().map(|(_, l)| *l).collect();

        Self::new(texts, labels, vocab, max_length)
    }

    /// Returns the vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Returns the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Returns the maximum sequence length.
    #[must_use]
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Encodes text to padded tensor.
    fn encode_text(&self, text: &str) -> Tensor<f32> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut indices: Vec<f32> = tokens
            .iter()
            .take(self.max_length)
            .map(|t| self.vocab.token_to_index(t) as f32)
            .collect();

        // Pad to max_length
        let pad_idx = self.vocab.pad_index().unwrap_or(0) as f32;
        while indices.len() < self.max_length {
            indices.push(pad_idx);
        }

        Tensor::from_vec(indices, &[self.max_length]).unwrap()
    }
}

impl Dataset for TextDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.texts.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.texts.len() {
            return None;
        }

        let text = self.encode_text(&self.texts[index]);

        // One-hot encode label
        let mut label_vec = vec![0.0f32; self.num_classes];
        label_vec[self.labels[index]] = 1.0;
        let label = Tensor::from_vec(label_vec, &[self.num_classes]).unwrap();

        Some((text, label))
    }
}

// =============================================================================
// LanguageModelDataset
// =============================================================================

/// A dataset for language modeling (next token prediction).
pub struct LanguageModelDataset {
    tokens: Vec<usize>,
    sequence_length: usize,
    vocab: Vocab,
}

impl LanguageModelDataset {
    /// Creates a new `LanguageModelDataset`.
    #[must_use]
    pub fn new(text: &str, vocab: Vocab, sequence_length: usize) -> Self {
        let tokens: Vec<usize> = text
            .split_whitespace()
            .map(|t| vocab.token_to_index(t))
            .collect();

        Self {
            tokens,
            sequence_length,
            vocab,
        }
    }

    /// Creates a dataset from text, building vocabulary automatically.
    #[must_use]
    pub fn from_text(text: &str, sequence_length: usize, min_freq: usize) -> Self {
        let vocab = Vocab::from_text(text, min_freq);
        Self::new(text, vocab, sequence_length)
    }

    /// Returns the vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

impl Dataset for LanguageModelDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        if self.tokens.len() <= self.sequence_length {
            0
        } else {
            self.tokens.len() - self.sequence_length
        }
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.len() {
            return None;
        }

        // Input: tokens[index..index+sequence_length]
        let input: Vec<f32> = self.tokens[index..index + self.sequence_length]
            .iter()
            .map(|&t| t as f32)
            .collect();

        // Target: tokens[index+1..index+sequence_length+1]
        let target: Vec<f32> = self.tokens[(index + 1)..=(index + self.sequence_length)]
            .iter()
            .map(|&t| t as f32)
            .collect();

        Some((
            Tensor::from_vec(input, &[self.sequence_length]).unwrap(),
            Tensor::from_vec(target, &[self.sequence_length]).unwrap(),
        ))
    }
}

// =============================================================================
// SyntheticSentimentDataset
// =============================================================================

/// A synthetic sentiment analysis dataset for testing.
pub struct SyntheticSentimentDataset {
    size: usize,
    max_length: usize,
    vocab_size: usize,
}

impl SyntheticSentimentDataset {
    /// Creates a new synthetic sentiment dataset.
    #[must_use]
    pub fn new(size: usize, max_length: usize, vocab_size: usize) -> Self {
        Self {
            size,
            max_length,
            vocab_size,
        }
    }

    /// Creates a small test dataset.
    #[must_use]
    pub fn small() -> Self {
        Self::new(100, 32, 1000)
    }

    /// Creates a standard training dataset.
    #[must_use]
    pub fn train() -> Self {
        Self::new(10000, 64, 10000)
    }

    /// Creates a standard test dataset.
    #[must_use]
    pub fn test() -> Self {
        Self::new(2000, 64, 10000)
    }
}

impl Dataset for SyntheticSentimentDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.size {
            return None;
        }

        // Generate deterministic "random" sequence
        let seed = index as u32;
        let label = index % 2; // Binary sentiment

        let mut text = Vec::with_capacity(self.max_length);
        for i in 0..self.max_length {
            let token_seed = seed.wrapping_mul(1103515245).wrapping_add(12345 + i as u32);
            let token = (token_seed as usize) % self.vocab_size;
            // Bias tokens based on sentiment
            let biased_token = if label == 1 {
                (token + self.vocab_size / 2) % self.vocab_size
            } else {
                token
            };
            text.push(biased_token as f32);
        }

        let text_tensor = Tensor::from_vec(text, &[self.max_length]).unwrap();

        // One-hot label
        let mut label_vec = vec![0.0f32; 2];
        label_vec[label] = 1.0;
        let label_tensor = Tensor::from_vec(label_vec, &[2]).unwrap();

        Some((text_tensor, label_tensor))
    }
}

// =============================================================================
// SyntheticSequenceDataset
// =============================================================================

/// A synthetic sequence-to-sequence dataset for testing.
pub struct SyntheticSeq2SeqDataset {
    size: usize,
    src_length: usize,
    tgt_length: usize,
    vocab_size: usize,
}

impl SyntheticSeq2SeqDataset {
    /// Creates a new synthetic seq2seq dataset.
    #[must_use]
    pub fn new(size: usize, src_length: usize, tgt_length: usize, vocab_size: usize) -> Self {
        Self {
            size,
            src_length,
            tgt_length,
            vocab_size,
        }
    }

    /// Creates a copy task dataset (target = reversed source).
    #[must_use]
    pub fn copy_task(size: usize, length: usize, vocab_size: usize) -> Self {
        Self::new(size, length, length, vocab_size)
    }
}

impl Dataset for SyntheticSeq2SeqDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.size {
            return None;
        }

        let seed = index as u32;

        // Generate source sequence
        let mut src = Vec::with_capacity(self.src_length);
        for i in 0..self.src_length {
            let token_seed = seed.wrapping_mul(1103515245).wrapping_add(12345 + i as u32);
            let token = (token_seed as usize) % self.vocab_size;
            src.push(token as f32);
        }

        // Target is reversed source (simple copy task)
        let tgt: Vec<f32> = src.iter().rev().copied().collect();

        Some((
            Tensor::from_vec(src, &[self.src_length]).unwrap(),
            Tensor::from_vec(tgt, &[self.tgt_length]).unwrap(),
        ))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_dataset() {
        let vocab = Vocab::from_tokens(&["hello", "world", "good", "bad", "<pad>", "<unk>"]);
        let texts = vec!["hello world".to_string(), "good bad".to_string()];
        let labels = vec![0, 1];

        let dataset = TextDataset::new(texts, labels, vocab, 10);

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.num_classes(), 2);

        let (text, label) = dataset.get(0).unwrap();
        assert_eq!(text.shape(), &[10]);
        assert_eq!(label.shape(), &[2]);
    }

    #[test]
    fn test_language_model_dataset() {
        let text = "the quick brown fox jumps over the lazy dog";
        let dataset = LanguageModelDataset::from_text(text, 3, 1);

        assert!(dataset.len() > 0);

        let (input, target) = dataset.get(0).unwrap();
        assert_eq!(input.shape(), &[3]);
        assert_eq!(target.shape(), &[3]);
    }

    #[test]
    fn test_synthetic_sentiment_dataset() {
        let dataset = SyntheticSentimentDataset::small();

        assert_eq!(dataset.len(), 100);

        let (text, label) = dataset.get(0).unwrap();
        assert_eq!(text.shape(), &[32]);
        assert_eq!(label.shape(), &[2]);

        // Check label is one-hot
        let label_vec = label.to_vec();
        let sum: f32 = label_vec.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_synthetic_sentiment_deterministic() {
        let dataset = SyntheticSentimentDataset::small();

        let (text1, label1) = dataset.get(5).unwrap();
        let (text2, label2) = dataset.get(5).unwrap();

        assert_eq!(text1.to_vec(), text2.to_vec());
        assert_eq!(label1.to_vec(), label2.to_vec());
    }

    #[test]
    fn test_synthetic_seq2seq_dataset() {
        let dataset = SyntheticSeq2SeqDataset::copy_task(100, 10, 50);

        assert_eq!(dataset.len(), 100);

        let (src, tgt) = dataset.get(0).unwrap();
        assert_eq!(src.shape(), &[10]);
        assert_eq!(tgt.shape(), &[10]);

        // Target should be reversed source
        let src_vec = src.to_vec();
        let tgt_vec = tgt.to_vec();
        let reversed: Vec<f32> = src_vec.iter().rev().copied().collect();
        assert_eq!(tgt_vec, reversed);
    }

    #[test]
    fn test_text_dataset_padding() {
        let vocab = Vocab::with_special_tokens();
        let texts = vec!["a b".to_string()];
        let labels = vec![0];

        let dataset = TextDataset::new(texts, labels, vocab, 10);
        let (text, _) = dataset.get(0).unwrap();

        // Should be padded to length 10
        assert_eq!(text.shape(), &[10]);
    }
}
