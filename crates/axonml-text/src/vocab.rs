//! Vocabulary - Token to Index Mapping
//!
//! Provides vocabulary management for text processing.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::collections::HashMap;

// =============================================================================
// Special Tokens
// =============================================================================

/// Default special token for unknown words.
pub const UNK_TOKEN: &str = "<unk>";
/// Default special token for padding.
pub const PAD_TOKEN: &str = "<pad>";
/// Default special token for beginning of sequence.
pub const BOS_TOKEN: &str = "<bos>";
/// Default special token for end of sequence.
pub const EOS_TOKEN: &str = "<eos>";
/// Default special token for masking (used in BERT-style models).
pub const MASK_TOKEN: &str = "<mask>";

// =============================================================================
// Vocabulary
// =============================================================================

/// A vocabulary that maps tokens to indices and vice versa.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Token to index mapping.
    token_to_idx: HashMap<String, usize>,
    /// Index to token mapping.
    idx_to_token: Vec<String>,
    /// Unknown token.
    unk_token: Option<String>,
    /// Padding token.
    pad_token: Option<String>,
    /// Beginning of sequence token.
    bos_token: Option<String>,
    /// End of sequence token.
    eos_token: Option<String>,
}

impl Vocab {
    /// Creates a new empty vocabulary.
    #[must_use] pub fn new() -> Self {
        Self {
            token_to_idx: HashMap::new(),
            idx_to_token: Vec::new(),
            unk_token: None,
            pad_token: None,
            bos_token: None,
            eos_token: None,
        }
    }

    /// Creates a vocabulary with default special tokens.
    #[must_use] pub fn with_special_tokens() -> Self {
        let mut vocab = Self::new();
        vocab.add_special_tokens(&[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]);
        vocab.unk_token = Some(UNK_TOKEN.to_string());
        vocab.pad_token = Some(PAD_TOKEN.to_string());
        vocab.bos_token = Some(BOS_TOKEN.to_string());
        vocab.eos_token = Some(EOS_TOKEN.to_string());
        vocab
    }

    /// Creates a vocabulary from a list of tokens.
    #[must_use] pub fn from_tokens(tokens: &[&str]) -> Self {
        let mut vocab = Self::new();
        for token in tokens {
            vocab.add_token(token);
        }
        vocab
    }

    /// Creates a vocabulary from text by extracting unique tokens.
    #[must_use] pub fn from_text(text: &str, min_freq: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();

        for word in text.split_whitespace() {
            *freq.entry(word.to_string()).or_insert(0) += 1;
        }

        let mut vocab = Self::with_special_tokens();

        // Sort by frequency (descending) then alphabetically for determinism
        let mut tokens: Vec<_> = freq
            .into_iter()
            .filter(|(_, count)| *count >= min_freq)
            .collect();
        tokens.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for (token, _) in tokens {
            vocab.add_token(&token);
        }

        vocab
    }

    /// Adds a token to the vocabulary.
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&idx) = self.token_to_idx.get(token) {
            return idx;
        }

        let idx = self.idx_to_token.len();
        self.token_to_idx.insert(token.to_string(), idx);
        self.idx_to_token.push(token.to_string());
        idx
    }

    /// Adds multiple special tokens.
    pub fn add_special_tokens(&mut self, tokens: &[&str]) {
        for token in tokens {
            self.add_token(token);
        }
    }

    /// Returns the index for a token, or the UNK index if not found.
    #[must_use] pub fn token_to_index(&self, token: &str) -> usize {
        if let Some(&idx) = self.token_to_idx.get(token) {
            return idx;
        }

        // Return UNK index if available
        if let Some(ref unk) = self.unk_token {
            if let Some(&idx) = self.token_to_idx.get(unk) {
                return idx;
            }
        }

        0 // Default to first token if no UNK
    }

    /// Returns the token for an index.
    #[must_use] pub fn index_to_token(&self, idx: usize) -> Option<&str> {
        self.idx_to_token.get(idx).map(std::string::String::as_str)
    }

    /// Returns the vocabulary size.
    #[must_use] pub fn len(&self) -> usize {
        self.idx_to_token.len()
    }

    /// Returns true if the vocabulary is empty.
    #[must_use] pub fn is_empty(&self) -> bool {
        self.idx_to_token.is_empty()
    }

    /// Checks if a token is in the vocabulary.
    #[must_use] pub fn contains(&self, token: &str) -> bool {
        self.token_to_idx.contains_key(token)
    }

    /// Returns the UNK token index.
    #[must_use] pub fn unk_index(&self) -> Option<usize> {
        self.unk_token
            .as_ref()
            .and_then(|t| self.token_to_idx.get(t).copied())
    }

    /// Returns the PAD token index.
    #[must_use] pub fn pad_index(&self) -> Option<usize> {
        self.pad_token
            .as_ref()
            .and_then(|t| self.token_to_idx.get(t).copied())
    }

    /// Returns the BOS token index.
    #[must_use] pub fn bos_index(&self) -> Option<usize> {
        self.bos_token
            .as_ref()
            .and_then(|t| self.token_to_idx.get(t).copied())
    }

    /// Returns the EOS token index.
    #[must_use] pub fn eos_index(&self) -> Option<usize> {
        self.eos_token
            .as_ref()
            .and_then(|t| self.token_to_idx.get(t).copied())
    }

    /// Encodes a sequence of tokens to indices.
    #[must_use] pub fn encode(&self, tokens: &[&str]) -> Vec<usize> {
        tokens.iter().map(|t| self.token_to_index(t)).collect()
    }

    /// Decodes a sequence of indices to tokens.
    #[must_use] pub fn decode(&self, indices: &[usize]) -> Vec<String> {
        indices
            .iter()
            .filter_map(|&idx| self.index_to_token(idx).map(std::string::ToString::to_string))
            .collect()
    }

    /// Sets the UNK token.
    pub fn set_unk_token(&mut self, token: &str) {
        self.add_token(token);
        self.unk_token = Some(token.to_string());
    }

    /// Sets the PAD token.
    pub fn set_pad_token(&mut self, token: &str) {
        self.add_token(token);
        self.pad_token = Some(token.to_string());
    }

    /// Returns all tokens in the vocabulary.
    #[must_use] pub fn tokens(&self) -> &[String] {
        &self.idx_to_token
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_new() {
        let vocab = Vocab::new();
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
    }

    #[test]
    fn test_vocab_add_token() {
        let mut vocab = Vocab::new();

        let idx1 = vocab.add_token("hello");
        let idx2 = vocab.add_token("world");
        let idx3 = vocab.add_token("hello"); // Duplicate

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx3, 0); // Should return existing index
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_vocab_token_to_index() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello");
        vocab.add_token("world");

        assert_eq!(vocab.token_to_index("hello"), 0);
        assert_eq!(vocab.token_to_index("world"), 1);
    }

    #[test]
    fn test_vocab_index_to_token() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello");
        vocab.add_token("world");

        assert_eq!(vocab.index_to_token(0), Some("hello"));
        assert_eq!(vocab.index_to_token(1), Some("world"));
        assert_eq!(vocab.index_to_token(2), None);
    }

    #[test]
    fn test_vocab_with_special_tokens() {
        let vocab = Vocab::with_special_tokens();

        assert!(vocab.contains(PAD_TOKEN));
        assert!(vocab.contains(UNK_TOKEN));
        assert!(vocab.contains(BOS_TOKEN));
        assert!(vocab.contains(EOS_TOKEN));

        assert!(vocab.pad_index().is_some());
        assert!(vocab.unk_index().is_some());
        assert!(vocab.bos_index().is_some());
        assert!(vocab.eos_index().is_some());
    }

    #[test]
    fn test_vocab_unknown_token() {
        let vocab = Vocab::with_special_tokens();
        let unk_idx = vocab.unk_index().unwrap();

        // Unknown tokens should map to UNK
        assert_eq!(vocab.token_to_index("nonexistent"), unk_idx);
    }

    #[test]
    fn test_vocab_encode_decode() {
        let mut vocab = Vocab::with_special_tokens();
        vocab.add_token("hello");
        vocab.add_token("world");

        let tokens = vec!["hello", "world", "hello"];
        let encoded = vocab.encode(&tokens);
        let decoded = vocab.decode(&encoded);

        assert_eq!(decoded, vec!["hello", "world", "hello"]);
    }

    #[test]
    fn test_vocab_from_tokens() {
        let vocab = Vocab::from_tokens(&["a", "b", "c"]);

        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.token_to_index("a"), 0);
        assert_eq!(vocab.token_to_index("b"), 1);
        assert_eq!(vocab.token_to_index("c"), 2);
    }

    #[test]
    fn test_vocab_from_text() {
        let text = "the quick brown fox jumps over the lazy dog the";
        let vocab = Vocab::from_text(text, 1);

        // Should have all unique words plus special tokens
        assert!(vocab.contains("the"));
        assert!(vocab.contains("quick"));
        assert!(vocab.contains("fox"));
    }

    #[test]
    fn test_vocab_from_text_min_freq() {
        let text = "the the the quick quick brown";
        let vocab = Vocab::from_text(text, 2);

        // Only "the" and "quick" have freq >= 2
        assert!(vocab.contains("the"));
        assert!(vocab.contains("quick"));
        assert!(!vocab.contains("brown"));
    }

    #[test]
    fn test_vocab_contains() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello");

        assert!(vocab.contains("hello"));
        assert!(!vocab.contains("world"));
    }
}
