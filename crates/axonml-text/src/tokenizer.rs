//! Tokenizer - Text Tokenization
//!
//! Provides various tokenization strategies for text processing.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use crate::vocab::Vocab;
use std::collections::HashMap;

// =============================================================================
// Tokenizer Trait
// =============================================================================

/// Trait for text tokenization.
pub trait Tokenizer: Send + Sync {
    /// Tokenizes a string into tokens.
    fn tokenize(&self, text: &str) -> Vec<String>;

    /// Tokenizes and encodes to indices using a vocabulary.
    fn encode(&self, text: &str, vocab: &Vocab) -> Vec<usize> {
        let tokens = self.tokenize(text);
        let token_refs: Vec<&str> = tokens.iter().map(std::string::String::as_str).collect();
        vocab.encode(&token_refs)
    }
}

// =============================================================================
// WhitespaceTokenizer
// =============================================================================

/// Simple whitespace-based tokenizer.
#[derive(Debug, Clone, Default)]
pub struct WhitespaceTokenizer {
    lowercase: bool,
}

impl WhitespaceTokenizer {
    /// Creates a new `WhitespaceTokenizer`.
    #[must_use] pub fn new() -> Self {
        Self { lowercase: false }
    }

    /// Creates a tokenizer that lowercases all tokens.
    #[must_use] pub fn lowercase() -> Self {
        Self { lowercase: true }
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| {
                if self.lowercase {
                    s.to_lowercase()
                } else {
                    s.to_string()
                }
            })
            .collect()
    }
}

// =============================================================================
// CharTokenizer
// =============================================================================

/// Character-level tokenizer.
#[derive(Debug, Clone, Default)]
pub struct CharTokenizer {
    include_whitespace: bool,
}

impl CharTokenizer {
    /// Creates a new `CharTokenizer`.
    #[must_use] pub fn new() -> Self {
        Self {
            include_whitespace: true,
        }
    }

    /// Creates a tokenizer that excludes whitespace.
    #[must_use] pub fn no_whitespace() -> Self {
        Self {
            include_whitespace: false,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        if self.include_whitespace {
            text.chars().map(|c| c.to_string()).collect()
        } else {
            text.chars()
                .filter(|c| !c.is_whitespace())
                .map(|c| c.to_string())
                .collect()
        }
    }
}

// =============================================================================
// WordPunctTokenizer
// =============================================================================

/// Tokenizer that separates words and punctuation.
#[derive(Debug, Clone, Default)]
pub struct WordPunctTokenizer {
    lowercase: bool,
}

impl WordPunctTokenizer {
    /// Creates a new `WordPunctTokenizer`.
    #[must_use] pub fn new() -> Self {
        Self { lowercase: false }
    }

    /// Creates a tokenizer that lowercases all tokens.
    #[must_use] pub fn lowercase() -> Self {
        Self { lowercase: true }
    }
}

impl Tokenizer for WordPunctTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            if c.is_alphanumeric() {
                current.push(c);
            } else {
                if !current.is_empty() {
                    tokens.push(if self.lowercase {
                        current.to_lowercase()
                    } else {
                        current.clone()
                    });
                    current.clear();
                }
                if !c.is_whitespace() {
                    tokens.push(c.to_string());
                }
            }
        }

        if !current.is_empty() {
            tokens.push(if self.lowercase {
                current.to_lowercase()
            } else {
                current
            });
        }

        tokens
    }
}

// =============================================================================
// NGramTokenizer
// =============================================================================

/// N-gram tokenizer for subword or character n-grams.
#[derive(Debug, Clone)]
pub struct NGramTokenizer {
    n: usize,
    char_level: bool,
}

impl NGramTokenizer {
    /// Creates a word-level n-gram tokenizer.
    #[must_use] pub fn word_ngrams(n: usize) -> Self {
        Self {
            n: n.max(1),
            char_level: false,
        }
    }

    /// Creates a character-level n-gram tokenizer.
    #[must_use] pub fn char_ngrams(n: usize) -> Self {
        Self {
            n: n.max(1),
            char_level: true,
        }
    }
}

impl Tokenizer for NGramTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        if self.char_level {
            // Character n-grams
            let chars: Vec<char> = text.chars().collect();
            if chars.len() < self.n {
                return vec![text.to_string()];
            }

            chars
                .windows(self.n)
                .map(|w| w.iter().collect::<String>())
                .collect()
        } else {
            // Word n-grams
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.len() < self.n {
                return vec![text.to_string()];
            }

            words.windows(self.n).map(|w| w.join(" ")).collect()
        }
    }
}

// =============================================================================
// BasicBPETokenizer
// =============================================================================

/// A basic Byte-Pair Encoding tokenizer.
#[derive(Debug, Clone)]
pub struct BasicBPETokenizer {
    merges: HashMap<(String, String), String>,
    vocab: Vec<String>,
}

impl BasicBPETokenizer {
    /// Creates a new BPE tokenizer.
    #[must_use] pub fn new() -> Self {
        Self {
            merges: HashMap::new(),
            vocab: Vec::new(),
        }
    }

    /// Trains the BPE tokenizer on text.
    pub fn train(&mut self, text: &str, num_merges: usize) {
        // Initialize vocabulary with characters
        let mut vocab: HashMap<String, usize> = HashMap::new();

        // Split text into words and add space markers
        for word in text.split_whitespace() {
            let word_with_end = format!("{word}</w>");
            let chars: Vec<String> = word_with_end.chars().map(|c| c.to_string()).collect();
            *vocab.entry(chars.join(" ")).or_insert(0) += 1;
        }

        for _ in 0..num_merges {
            // Count pairs
            let mut pairs: HashMap<(String, String), usize> = HashMap::new();
            for (word, count) in &vocab {
                let symbols: Vec<&str> = word.split(' ').collect();
                for i in 0..symbols.len().saturating_sub(1) {
                    let pair = (symbols[i].to_string(), symbols[i + 1].to_string());
                    *pairs.entry(pair).or_insert(0) += count;
                }
            }

            if pairs.is_empty() {
                break;
            }

            // Find most frequent pair
            let best_pair = pairs
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(pair, _)| pair);

            if let Some((a, b)) = best_pair {
                let merged = format!("{a}{b}");
                self.merges.insert((a.clone(), b.clone()), merged.clone());

                // Update vocabulary
                let pattern = format!("{a} {b}");
                let mut new_vocab = HashMap::new();
                for (word, count) in vocab {
                    let new_word = word.replace(&pattern, &merged);
                    *new_vocab.entry(new_word).or_insert(0) += count;
                }
                vocab = new_vocab;
            }
        }

        // Extract final vocabulary
        let mut all_symbols: std::collections::HashSet<String> = std::collections::HashSet::new();
        for word in vocab.keys() {
            for symbol in word.split(' ') {
                all_symbols.insert(symbol.to_string());
            }
        }
        self.vocab = all_symbols.into_iter().collect();
        self.vocab.sort();
    }

    /// Returns the vocabulary.
    #[must_use] pub fn get_vocab(&self) -> &[String] {
        &self.vocab
    }

    /// Applies BPE merges to a word.
    fn apply_bpe(&self, word: &str) -> Vec<String> {
        let word_with_end = format!("{word}</w>");
        let mut symbols: Vec<String> = word_with_end.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best_pair: Option<(usize, &str)> = None;

            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(merged) = self.merges.get(&pair) {
                    best_pair = Some((i, merged));
                    break;
                }
            }

            match best_pair {
                Some((i, merged)) => {
                    symbols[i] = merged.to_string();
                    symbols.remove(i + 1);
                }
                None => break,
            }
        }

        symbols
    }
}

impl Default for BasicBPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BasicBPETokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let word_tokens = self.apply_bpe(word);
            tokens.extend(word_tokens);
        }

        tokens
    }
}

// =============================================================================
// SentencePieceTokenizer (Simplified)
// =============================================================================

/// A simplified unigram-style tokenizer.
#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    vocab: HashMap<String, f32>,
    max_token_length: usize,
}

impl UnigramTokenizer {
    /// Creates a new unigram tokenizer from a vocabulary with scores.
    #[must_use] pub fn new(vocab: HashMap<String, f32>) -> Self {
        let max_len = vocab.keys().map(std::string::String::len).max().unwrap_or(1);
        Self {
            vocab,
            max_token_length: max_len,
        }
    }

    /// Creates a tokenizer from a list of tokens (equal scores).
    #[must_use] pub fn from_tokens(tokens: &[&str]) -> Self {
        let vocab: HashMap<String, f32> = tokens.iter().map(|&t| (t.to_string(), 1.0)).collect();
        Self::new(vocab)
    }

    /// Tokenizes using a greedy longest-match approach.
    fn greedy_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut best_len = 1;
            let mut best_token = chars[i].to_string();

            // Try to find the longest matching token
            for len in 1..=self.max_token_length.min(chars.len() - i) {
                let candidate: String = chars[i..i + len].iter().collect();
                if self.vocab.contains_key(&candidate) {
                    best_len = len;
                    best_token = candidate;
                }
            }

            tokens.push(best_token);
            i += best_len;
        }

        tokens
    }
}

impl Tokenizer for UnigramTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        // Split by whitespace first, then tokenize each word
        let mut all_tokens = Vec::new();

        for word in text.split_whitespace() {
            let word_tokens = self.greedy_tokenize(word);
            all_tokens.extend(word_tokens);
        }

        all_tokens
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Hello World");

        assert_eq!(tokens, vec!["Hello", "World"]);
    }

    #[test]
    fn test_whitespace_tokenizer_lowercase() {
        let tokenizer = WhitespaceTokenizer::lowercase();
        let tokens = tokenizer.tokenize("Hello World");

        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_char_tokenizer() {
        let tokenizer = CharTokenizer::new();
        let tokens = tokenizer.tokenize("Hi!");

        assert_eq!(tokens, vec!["H", "i", "!"]);
    }

    #[test]
    fn test_char_tokenizer_no_whitespace() {
        let tokenizer = CharTokenizer::no_whitespace();
        let tokens = tokenizer.tokenize("Hi there!");

        assert_eq!(tokens, vec!["H", "i", "t", "h", "e", "r", "e", "!"]);
    }

    #[test]
    fn test_word_punct_tokenizer() {
        let tokenizer = WordPunctTokenizer::new();
        let tokens = tokenizer.tokenize("Hello, World!");

        assert_eq!(tokens, vec!["Hello", ",", "World", "!"]);
    }

    #[test]
    fn test_word_punct_tokenizer_lowercase() {
        let tokenizer = WordPunctTokenizer::lowercase();
        let tokens = tokenizer.tokenize("Hello, World!");

        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_ngram_word_tokenizer() {
        let tokenizer = NGramTokenizer::word_ngrams(2);
        let tokens = tokenizer.tokenize("one two three");

        assert_eq!(tokens, vec!["one two", "two three"]);
    }

    #[test]
    fn test_ngram_char_tokenizer() {
        let tokenizer = NGramTokenizer::char_ngrams(3);
        let tokens = tokenizer.tokenize("hello");

        assert_eq!(tokens, vec!["hel", "ell", "llo"]);
    }

    #[test]
    fn test_bpe_tokenizer_basic() {
        let mut tokenizer = BasicBPETokenizer::new();
        tokenizer.train("low lower lowest", 10);

        // Should have learned some merges
        assert!(!tokenizer.get_vocab().is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_apply() {
        let mut tokenizer = BasicBPETokenizer::new();
        tokenizer.train("low low low lower lowest", 5);

        let tokens = tokenizer.tokenize("low");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_unigram_tokenizer() {
        let tokenizer = UnigramTokenizer::from_tokens(&[
            "hel", "lo", "wor", "ld", "h", "e", "l", "o", "w", "r", "d",
        ]);
        let tokens = tokenizer.tokenize("hello world");

        // Should produce some tokens
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_tokenizer_encode() {
        let tokenizer = WhitespaceTokenizer::new();
        let mut vocab = Vocab::new();
        vocab.add_token("hello");
        vocab.add_token("world");

        let indices = tokenizer.encode("hello world", &vocab);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_tokenizer_with_multiple_spaces() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("hello    world");

        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("");

        assert!(tokens.is_empty());
    }
}
