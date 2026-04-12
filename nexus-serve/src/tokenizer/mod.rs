//! Tokenizer loading — GGUF-embedded BPE, HuggingFace tokenizer.json, or char-level fallback.
//!
//! Priority order:
//! 1. HuggingFace tokenizer.json (if found alongside model file)
//! 2. GGUF-embedded vocabulary (tokenizer.ggml.tokens + tokenizer.ggml.merges)
//! 3. Char-level fallback (last resort — produces garbage for real models)

use std::collections::HashMap;
use std::path::Path;

use crate::model::gguf::{GgufFile, GgufValue};

/// A loaded tokenizer that can encode/decode text.
pub enum Tokenizer {
    /// HuggingFace tokenizers (BPE, WordPiece, etc.)
    HuggingFace(tokenizers::Tokenizer),
    /// BPE tokenizer built from GGUF-embedded vocabulary.
    GgufBpe {
        /// Token ID → token string
        id_to_token: Vec<String>,
        /// Token string → token ID
        token_to_id: HashMap<String, u32>,
        /// BPE merge pairs (if available)
        merges: Vec<(String, String)>,
    },
    /// Simple character-level fallback for AxonML-trained models.
    CharLevel {
        chars: Vec<char>,
        char_to_id: HashMap<char, u32>,
    },
}

impl Tokenizer {
    /// Load a HuggingFace tokenizer.json file.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self::HuggingFace(tok))
    }

    /// Build a tokenizer from GGUF-embedded vocabulary metadata.
    ///
    /// GGUF files contain:
    /// - `tokenizer.ggml.tokens`: Array of token strings (the vocabulary)
    /// - `tokenizer.ggml.scores`: Array of f32 merge priorities (optional)
    /// - `tokenizer.ggml.merges`: Array of merge rule strings "token_a token_b" (optional)
    /// - `tokenizer.ggml.token_type`: Array of i32 token types (optional)
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        let tokens = gguf.get_meta("tokenizer.ggml.tokens")?;
        let token_strings: Vec<String> = match tokens {
            GgufValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    GgufValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => return None,
        };

        if token_strings.is_empty() {
            return None;
        }

        let mut token_to_id = HashMap::with_capacity(token_strings.len());
        for (i, tok) in token_strings.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Load merges if available
        let merges = gguf
            .get_meta("tokenizer.ggml.merges")
            .and_then(|v| match v {
                GgufValue::Array(arr) => {
                    let pairs: Vec<(String, String)> = arr
                        .iter()
                        .filter_map(|v| match v {
                            GgufValue::String(s) => {
                                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                                if parts.len() == 2 {
                                    Some((parts[0].to_string(), parts[1].to_string()))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        })
                        .collect();
                    Some(pairs)
                }
                _ => None,
            })
            .unwrap_or_default();

        Some(Self::GgufBpe {
            id_to_token: token_strings,
            token_to_id,
            merges,
        })
    }

    /// Build a char-level tokenizer from a corpus string.
    pub fn char_level(corpus: &str) -> Self {
        use std::collections::BTreeSet;
        let mut chars: Vec<char> = corpus.chars().collect::<BTreeSet<_>>().into_iter().collect();
        if !chars.contains(&'\0') {
            chars.insert(0, '\0');
        }
        let mut char_to_id = HashMap::with_capacity(chars.len());
        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i as u32);
        }
        Self::CharLevel { chars, char_to_id }
    }

    /// Encode text to token IDs.
    ///
    /// Passes `add_special_tokens=true` so that ChatML markers like
    /// `<|im_start|>` and `<|im_end|>` get their proper single-token IDs
    /// when present in the tokenizer's added_tokens list.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::HuggingFace(tok) => tok
                .encode(text, true)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default(),

            Self::GgufBpe {
                token_to_id,
                merges,
                id_to_token,
                ..
            } => {
                if !merges.is_empty() {
                    bpe_encode(text, token_to_id, merges, id_to_token)
                } else {
                    // No merges — greedy longest-match encoding
                    greedy_encode(text, token_to_id)
                }
            }

            Self::CharLevel { char_to_id, .. } => text
                .chars()
                .map(|c| *char_to_id.get(&c).unwrap_or(&0))
                .collect(),
        }
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Self::HuggingFace(tok) => tok.decode(ids, true).unwrap_or_default(),

            Self::GgufBpe { id_to_token, .. } => {
                let mut result = String::new();
                for &id in ids {
                    if let Some(tok) = id_to_token.get(id as usize) {
                        // GGUF BPE tokens use byte-level encoding for special chars.
                        // Common patterns: "Ġ" = leading space, "Ċ" = newline
                        // Also raw byte tokens like <0xNN>
                        result.push_str(&decode_bpe_token(tok));
                    }
                }
                result
            }

            Self::CharLevel { chars, .. } => ids
                .iter()
                .map(|&id| chars.get(id as usize).copied().unwrap_or('\0'))
                .collect(),
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(tok) => tok.get_vocab_size(true),
            Self::GgufBpe { id_to_token, .. } => id_to_token.len(),
            Self::CharLevel { chars, .. } => chars.len(),
        }
    }

    /// Get the variant name.
    pub fn variant(&self) -> &'static str {
        match self {
            Self::HuggingFace(_) => "HuggingFace BPE",
            Self::GgufBpe { merges, .. } => {
                if merges.is_empty() {
                    "GGUF vocab (greedy)"
                } else {
                    "GGUF BPE"
                }
            }
            Self::CharLevel { .. } => "char-level fallback",
        }
    }
}

// =============================================================================
// BPE encoding
// =============================================================================

/// BPE encode using merge rules from GGUF.
fn bpe_encode(
    text: &str,
    token_to_id: &HashMap<String, u32>,
    merges: &[(String, String)],
    _vocab: &[String],
) -> Vec<u32> {
    // Start with UTF-8 bytes as individual tokens
    let bytes = text.as_bytes();
    let mut symbols: Vec<String> = bytes
        .iter()
        .map(|&b| {
            // Try single byte as char first
            if b.is_ascii() && !b.is_ascii_control() {
                String::from(b as char)
            } else {
                format!("<0x{:02X}>", b)
            }
        })
        .collect();

    // Apply merges greedily
    for (left, right) in merges {
        let mut i = 0;
        while i + 1 < symbols.len() {
            if symbols[i] == *left && symbols[i + 1] == *right {
                let merged = format!("{}{}", left, right);
                symbols[i] = merged;
                symbols.remove(i + 1);
                // Don't increment i — check if the merged token can merge again
            } else {
                i += 1;
            }
        }
    }

    // Map to token IDs
    symbols
        .iter()
        .map(|s| *token_to_id.get(s).unwrap_or(&0))
        .collect()
}

/// Greedy longest-match encoding (when no merge rules available).
fn greedy_encode(text: &str, token_to_id: &HashMap<String, u32>) -> Vec<u32> {
    let mut ids = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let mut best_len = 0;
        let mut best_id = 0u32;

        // Try progressively shorter substrings
        let max_len = (chars.len() - i).min(32); // cap at 32 chars
        for len in (1..=max_len).rev() {
            let substr: String = chars[i..i + len].iter().collect();
            if let Some(&id) = token_to_id.get(&substr) {
                best_len = len;
                best_id = id;
                break;
            }
        }

        if best_len == 0 {
            // Single char not in vocab — try byte fallback
            let c = chars[i];
            for b in c.to_string().as_bytes() {
                let byte_tok = format!("<0x{:02X}>", b);
                if let Some(&id) = token_to_id.get(&byte_tok) {
                    ids.push(id);
                } else {
                    ids.push(0); // unknown
                }
            }
            i += 1;
        } else {
            ids.push(best_id);
            i += best_len;
        }
    }

    ids
}

/// Decode a BPE token string, handling byte-level encoding.
fn decode_bpe_token(token: &str) -> String {
    // Handle byte tokens: <0xNN>
    if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
            return String::from(byte as char);
        }
    }

    // Handle common BPE special tokens
    if token == "<s>" || token == "</s>" || token == "<|endoftext|>" || token == "<|im_end|>" {
        return String::new();
    }
    if token == "<|im_start|>" {
        return String::new();
    }

    // SentencePiece-style: leading ▁ (U+2581) = space
    token.replace('▁', " ")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_level_roundtrip() {
        let tok = Tokenizer::char_level("hello world");
        let ids = tok.encode("hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_greedy_encode() {
        let mut vocab = HashMap::new();
        vocab.insert("he".to_string(), 1);
        vocab.insert("llo".to_string(), 2);
        vocab.insert(" ".to_string(), 3);
        vocab.insert("world".to_string(), 4);

        let ids = greedy_encode("hello world", &vocab);
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_decode_bpe_token() {
        assert_eq!(decode_bpe_token("hello"), "hello");
        assert_eq!(decode_bpe_token("▁hello"), " hello");
        assert_eq!(decode_bpe_token("<0x0A>"), "\n");
        assert_eq!(decode_bpe_token("<s>"), "");
    }
}
