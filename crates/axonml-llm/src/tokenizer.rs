//! HuggingFace Tokenizer Support
//!
//! Loads and uses tokenizers from HuggingFace tokenizer.json files.
//!
//! # Example
//! ```rust,ignore
//! use axonml_llm::tokenizer::HFTokenizer;
//!
//! let tokenizer = HFTokenizer::from_pretrained("meta-llama/Llama-2-7b-hf")?;
//! let tokens = tokenizer.encode("Hello, world!")?;
//! let text = tokenizer.decode(&tokens)?;
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::{LLMError, LLMResult};
use crate::hf_loader::HFLoader;

// =============================================================================
// Tokenizer
// =============================================================================

/// HuggingFace-compatible tokenizer.
pub struct HFTokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// BPE merges (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Added tokens (not in base vocab)
    added_tokens: HashMap<String, u32>,
}

/// Special tokens configuration.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token: Option<String>,
    /// End of sequence token
    pub eos_token: Option<String>,
    /// Unknown token
    pub unk_token: Option<String>,
    /// Padding token
    pub pad_token: Option<String>,
    /// BOS token ID
    pub bos_token_id: Option<u32>,
    /// EOS token ID
    pub eos_token_id: Option<u32>,
    /// UNK token ID
    pub unk_token_id: Option<u32>,
    /// PAD token ID
    pub pad_token_id: Option<u32>,
}

impl HFTokenizer {
    /// Load tokenizer from a HuggingFace model.
    pub fn from_pretrained(model_id: &str) -> LLMResult<Self> {
        let loader = HFLoader::new(model_id)?;
        let cache_dir = loader.cache_dir();

        // Download tokenizer files
        Self::download_tokenizer_files(&loader)?;

        // Load from cache
        Self::from_directory(cache_dir)
    }

    /// Load tokenizer from a local directory.
    pub fn from_directory<P: AsRef<Path>>(path: P) -> LLMResult<Self> {
        let path = path.as_ref();

        // Try tokenizer.json first (fast tokenizer format)
        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists() {
            return Self::load_tokenizer_json(&tokenizer_json);
        }

        // Fall back to vocab.json + merges.txt (legacy format)
        let vocab_json = path.join("vocab.json");
        let merges_txt = path.join("merges.txt");

        if vocab_json.exists() {
            return Self::load_legacy_format(&vocab_json, &merges_txt, path);
        }

        Err(LLMError::ModelNotFound(
            "No tokenizer.json or vocab.json found".to_string()
        ))
    }

    /// Download tokenizer files from HuggingFace.
    fn download_tokenizer_files(loader: &HFLoader) -> LLMResult<()> {
        // Try to download tokenizer.json
        if loader.download_file_if_exists("tokenizer.json")? {
            // Also try tokenizer_config.json for special tokens
            let _ = loader.download_file_if_exists("tokenizer_config.json");
            return Ok(());
        }

        // Fall back to legacy files
        loader.download_file("vocab.json")?;
        let _ = loader.download_file_if_exists("merges.txt");
        let _ = loader.download_file_if_exists("special_tokens_map.json");

        Ok(())
    }

    /// Load from tokenizer.json (HuggingFace fast tokenizer format).
    fn load_tokenizer_json(path: &Path) -> LLMResult<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| LLMError::IoError(e.to_string()))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| LLMError::ParseError(e.to_string()))?;

        // Extract vocab from model.vocab
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        if let Some(model) = json.get("model") {
            if let Some(v) = model.get("vocab").and_then(|v| v.as_object()) {
                for (token, id) in v {
                    if let Some(id) = id.as_u64() {
                        vocab.insert(token.clone(), id as u32);
                        id_to_token.insert(id as u32, token.clone());
                    }
                }
            }
        }

        // Extract merges
        let mut merges = Vec::new();
        if let Some(model) = json.get("model") {
            if let Some(m) = model.get("merges").and_then(|m| m.as_array()) {
                for merge in m {
                    if let Some(s) = merge.as_str() {
                        let parts: Vec<&str> = s.split(' ').collect();
                        if parts.len() == 2 {
                            merges.push((parts[0].to_string(), parts[1].to_string()));
                        }
                    }
                }
            }
        }

        // Extract added tokens
        let mut added_tokens = HashMap::new();
        if let Some(tokens) = json.get("added_tokens").and_then(|t| t.as_array()) {
            for token in tokens {
                if let (Some(content), Some(id)) = (
                    token.get("content").and_then(|c| c.as_str()),
                    token.get("id").and_then(|i| i.as_u64())
                ) {
                    added_tokens.insert(content.to_string(), id as u32);
                    id_to_token.insert(id as u32, content.to_string());
                }
            }
        }

        // Extract special tokens
        let special_tokens = Self::extract_special_tokens(&json, &vocab, &added_tokens);

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            special_tokens,
            added_tokens,
        })
    }

    /// Load from legacy vocab.json + merges.txt format.
    fn load_legacy_format(vocab_path: &Path, merges_path: &Path, dir: &Path) -> LLMResult<Self> {
        // Load vocab
        let vocab_content = fs::read_to_string(vocab_path)
            .map_err(|e| LLMError::IoError(e.to_string()))?;
        let vocab_json: HashMap<String, u32> = serde_json::from_str(&vocab_content)
            .map_err(|e| LLMError::ParseError(e.to_string()))?;

        let vocab = vocab_json;
        let id_to_token: HashMap<u32, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        // Load merges if present
        let mut merges = Vec::new();
        if merges_path.exists() {
            let merges_content = fs::read_to_string(merges_path)
                .map_err(|e| LLMError::IoError(e.to_string()))?;
            for line in merges_content.lines().skip(1) {  // Skip header
                let parts: Vec<&str> = line.split(' ').collect();
                if parts.len() == 2 {
                    merges.push((parts[0].to_string(), parts[1].to_string()));
                }
            }
        }

        // Load special tokens
        let mut special_tokens = SpecialTokens::default();
        let special_path = dir.join("special_tokens_map.json");
        if special_path.exists() {
            if let Ok(content) = fs::read_to_string(&special_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(bos) = json.get("bos_token").and_then(|t| t.as_str()) {
                        special_tokens.bos_token = Some(bos.to_string());
                        special_tokens.bos_token_id = vocab.get(bos).copied();
                    }
                    if let Some(eos) = json.get("eos_token").and_then(|t| t.as_str()) {
                        special_tokens.eos_token = Some(eos.to_string());
                        special_tokens.eos_token_id = vocab.get(eos).copied();
                    }
                    if let Some(unk) = json.get("unk_token").and_then(|t| t.as_str()) {
                        special_tokens.unk_token = Some(unk.to_string());
                        special_tokens.unk_token_id = vocab.get(unk).copied();
                    }
                    if let Some(pad) = json.get("pad_token").and_then(|t| t.as_str()) {
                        special_tokens.pad_token = Some(pad.to_string());
                        special_tokens.pad_token_id = vocab.get(pad).copied();
                    }
                }
            }
        }

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            special_tokens,
            added_tokens: HashMap::new(),
        })
    }

    /// Extract special tokens from tokenizer.json.
    fn extract_special_tokens(
        json: &serde_json::Value,
        _vocab: &HashMap<String, u32>,
        _added_tokens: &HashMap<String, u32>,
    ) -> SpecialTokens {
        let mut special = SpecialTokens::default();

        // Check added_tokens for special token flags
        if let Some(tokens) = json.get("added_tokens").and_then(|t| t.as_array()) {
            for token in tokens {
                let content = token.get("content").and_then(|c| c.as_str());
                let id = token.get("id").and_then(|i| i.as_u64()).map(|i| i as u32);
                let special_flag = token.get("special").and_then(|s| s.as_bool()).unwrap_or(false);

                if let (Some(content), Some(id)) = (content, id) {
                    if special_flag {
                        // Try to identify which special token this is
                        let lower = content.to_lowercase();
                        if lower.contains("bos") || lower == "<s>" {
                            special.bos_token = Some(content.to_string());
                            special.bos_token_id = Some(id);
                        } else if lower.contains("eos") || lower == "</s>" {
                            special.eos_token = Some(content.to_string());
                            special.eos_token_id = Some(id);
                        } else if lower.contains("unk") {
                            special.unk_token = Some(content.to_string());
                            special.unk_token_id = Some(id);
                        } else if lower.contains("pad") {
                            special.pad_token = Some(content.to_string());
                            special.pad_token_id = Some(id);
                        }
                    }
                }
            }
        }

        special
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> LLMResult<Vec<u32>> {
        let mut tokens = Vec::new();

        // Add BOS token if configured
        if let Some(bos_id) = self.special_tokens.bos_token_id {
            tokens.push(bos_id);
        }

        // Tokenize using BPE
        let text_tokens = self.bpe_encode(text)?;
        tokens.extend(text_tokens);

        Ok(tokens)
    }

    /// Encode text with options.
    pub fn encode_with_options(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> LLMResult<Vec<u32>> {
        let mut tokens = Vec::new();

        if add_bos {
            if let Some(bos_id) = self.special_tokens.bos_token_id {
                tokens.push(bos_id);
            }
        }

        let text_tokens = self.bpe_encode(text)?;
        tokens.extend(text_tokens);

        if add_eos {
            if let Some(eos_id) = self.special_tokens.eos_token_id {
                tokens.push(eos_id);
            }
        }

        Ok(tokens)
    }

    /// BPE encoding of text.
    fn bpe_encode(&self, text: &str) -> LLMResult<Vec<u32>> {
        let mut tokens = Vec::new();

        // Split into words (simple whitespace split for now)
        for word in text.split_inclusive(|c: char| c.is_whitespace() || c.is_ascii_punctuation()) {
            if word.is_empty() {
                continue;
            }

            // Check for exact match first (handles special tokens and common words)
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
                continue;
            }

            // Check added tokens
            if let Some(&id) = self.added_tokens.get(word) {
                tokens.push(id);
                continue;
            }

            // Apply BPE
            let word_tokens = self.bpe_tokenize_word(word)?;
            tokens.extend(word_tokens);
        }

        Ok(tokens)
    }

    /// Apply BPE to a single word.
    fn bpe_tokenize_word(&self, word: &str) -> LLMResult<Vec<u32>> {
        if word.is_empty() {
            return Ok(vec![]);
        }

        // Start with characters
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply merges
        for (a, b) in &self.merges {
            let mut i = 0;
            while i < parts.len().saturating_sub(1) {
                if &parts[i] == a && &parts[i + 1] == b {
                    let merged = format!("{}{}", a, b);
                    parts[i] = merged;
                    parts.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Convert to IDs
        let mut ids = Vec::new();
        for part in parts {
            if let Some(&id) = self.vocab.get(&part) {
                ids.push(id);
            } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                ids.push(unk_id);
            } else {
                // Try byte fallback for unknown characters
                for byte in part.as_bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.vocab.get(&byte_token) {
                        ids.push(id);
                    }
                }
            }
        }

        Ok(ids)
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> LLMResult<String> {
        self.decode_with_options(ids, true)
    }

    /// Decode with options.
    pub fn decode_with_options(&self, ids: &[u32], skip_special: bool) -> LLMResult<String> {
        let mut text = String::new();

        for &id in ids {
            // Skip special tokens if requested
            if skip_special {
                if Some(id) == self.special_tokens.bos_token_id
                    || Some(id) == self.special_tokens.eos_token_id
                    || Some(id) == self.special_tokens.pad_token_id
                {
                    continue;
                }
            }

            if let Some(token) = self.id_to_token.get(&id) {
                // Handle byte tokens
                if token.starts_with("<0x") && token.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        text.push(byte as char);
                        continue;
                    }
                }
                text.push_str(token);
            }
        }

        // Clean up common BPE artifacts
        let text = text.replace("Ġ", " ");  // GPT-2 style space
        let text = text.replace("▁", " ");  // SentencePiece style space

        Ok(text)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.added_tokens.len()
    }

    /// Get special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.special_tokens.bos_token_id
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.special_tokens.eos_token_id
    }

    /// Convert a single token ID to string.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Convert a token string to ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
            .or_else(|| self.added_tokens.get(token).copied())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_default() {
        let tokens = SpecialTokens::default();
        assert!(tokens.bos_token.is_none());
        assert!(tokens.eos_token.is_none());
    }

    #[test]
    fn test_bpe_simple() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert(" ".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let tokenizer = HFTokenizer {
            vocab: vocab.clone(),
            id_to_token: vocab.iter().map(|(k, v)| (*v, k.clone())).collect(),
            merges: vec![],
            special_tokens: SpecialTokens::default(),
            added_tokens: HashMap::new(),
        };

        // Test decode
        let text = tokenizer.decode(&[0, 1, 2]).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_vocab_size() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);

        let mut added = HashMap::new();
        added.insert("<special>".to_string(), 2);

        let tokenizer = HFTokenizer {
            vocab,
            id_to_token: HashMap::new(),
            merges: vec![],
            special_tokens: SpecialTokens::default(),
            added_tokens: added,
        };

        assert_eq!(tokenizer.vocab_size(), 3);
    }
}
