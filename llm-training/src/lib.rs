//! llm-training — shared utilities for training the nine AxonML LLM architectures
//!
//! Provides:
//! - [`CharTokenizer`] — deterministic character-level tokenizer built from a corpus
//! - [`TextDataset`] — sliding-window dataset for next-token prediction
//! - [`lcg_range`] — simple seedable RNG for batch sampling (no external dep)
//! - Checkpoint resume helpers that work with any AxonML `Module`
//!
//! Each LLM has its own binary under `src/bin/train_<name>.rs`.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use axonml_nn::Module;
use axonml_serialize::{load_checkpoint, Checkpoint, TrainingState};

// =============================================================================
// Character Tokenizer
// =============================================================================

/// A minimal character-level tokenizer.
///
/// Each unique character in the training corpus is assigned a token ID
/// starting from 0. Token 0 is reserved for the unknown / padding token
/// (`'\0'`), which is never produced by `encode` for in-vocabulary characters.
pub struct CharTokenizer {
    char_to_id: HashMap<char, u32>,
    id_to_char: Vec<char>,
}

impl CharTokenizer {
    /// Build a tokenizer from a corpus string.
    /// Token 0 is reserved for the unknown/padding token (`'\0'`).
    pub fn from_corpus(corpus: &str) -> Self {
        let mut chars: Vec<char> = corpus
            .chars()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if !chars.contains(&'\0') {
            chars.insert(0, '\0');
        }

        let mut char_to_id = HashMap::with_capacity(chars.len());
        for (i, c) in chars.iter().enumerate() {
            char_to_id.insert(*c, i as u32);
        }

        Self {
            char_to_id,
            id_to_char: chars,
        }
    }

    /// Number of distinct tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    /// Encode a string into a sequence of token IDs.
    /// Unknown chars are mapped to token 0.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&0))
            .collect()
    }

    /// Decode a sequence of token IDs back into a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                self.id_to_char
                    .get(id as usize)
                    .copied()
                    .unwrap_or('\0')
            })
            .collect()
    }
}

// =============================================================================
// Text Dataset
// =============================================================================

/// Sliding-window dataset over a tokenized corpus.
///
/// Each "window" is a contiguous slice of `seq_len` tokens. Windows overlap by
/// `seq_len - 1` — a stride-1 window — so every position in the corpus serves
/// as a training start point.
pub struct TextDataset {
    /// All corpus tokens as a flat buffer.
    tokens: Vec<u32>,
    /// Sequence length (context window).
    seq_len: usize,
}

impl TextDataset {
    /// Build a dataset from a corpus string using the given tokenizer.
    pub fn from_string(corpus: &str, tokenizer: &CharTokenizer, seq_len: usize) -> Self {
        let tokens = tokenizer.encode(corpus);
        Self { tokens, seq_len }
    }

    /// Number of valid starting positions for a sliding window.
    pub fn len(&self) -> usize {
        self.tokens.len().saturating_sub(self.seq_len + 1)
    }

    /// Sample a batch of `batch_size` windows uniformly at random.
    /// Returns a flat `Vec<u32>` of shape `[batch_size * seq_len]`.
    pub fn sample_batch(&self, batch_size: usize, rng: &mut u64) -> Vec<u32> {
        let n = self.len().max(1);
        let mut batch = Vec::with_capacity(batch_size * self.seq_len);
        for _ in 0..batch_size {
            let start = lcg_range(rng, n);
            batch.extend_from_slice(&self.tokens[start..start + self.seq_len]);
        }
        batch
    }

    /// Get the raw token buffer (for statistics / validation).
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

// =============================================================================
// Seedable LCG RNG (no external crate)
// =============================================================================

/// Linear congruential generator — same constants as Numerical Recipes.
/// Returns a uniform sample in `[0, max)`.
pub fn lcg_range(state: &mut u64, max: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as usize) % max.max(1)
}

// =============================================================================
// Number formatting (for reporting param counts, dataset sizes, etc.)
// =============================================================================

/// Format a number with thousand separators: `1234567` → `1,234,567`.
pub fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let len = s.len();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

// =============================================================================
// Checkpoint resume helper
// =============================================================================

/// Resume mode: start fresh, load the latest checkpoint, load best, or a custom path.
pub enum ResumeMode {
    None,
    Latest,
    Best,
    Path(PathBuf),
}

impl ResumeMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "none" | "" => Self::None,
            "latest" => Self::Latest,
            "best" => Self::Best,
            path => Self::Path(PathBuf::from(path)),
        }
    }
}

/// Find a checkpoint file given a resume mode.
pub fn find_checkpoint(output_dir: &Path, mode: &ResumeMode) -> Option<PathBuf> {
    match mode {
        ResumeMode::None => None,
        ResumeMode::Latest => {
            let p = output_dir.join("checkpoint_latest.axonml");
            if p.exists() {
                Some(p)
            } else {
                let p2 = output_dir.join("checkpoint_best.axonml");
                if p2.exists() { Some(p2) } else { None }
            }
        }
        ResumeMode::Best => {
            let p = output_dir.join("checkpoint_best.axonml");
            if p.exists() { Some(p) } else { None }
        }
        ResumeMode::Path(p) => {
            if p.exists() { Some(p.clone()) } else { None }
        }
    }
}

/// Load a checkpoint into a model by matching shapes in-order.
/// Falls back to shape-based matching if named parameters don't align.
/// Returns `(starting_epoch, training_state)`.
pub fn load_model_from_checkpoint<M: Module>(
    model: &M,
    path: &Path,
) -> Result<(usize, TrainingState), String> {
    let checkpoint: Checkpoint =
        load_checkpoint(path).map_err(|e| format!("Failed to load checkpoint: {e}"))?;

    let state_dict = &checkpoint.model_state;
    let params = model.parameters();

    // Try name-based matching first via StateDict entries
    let mut loaded = 0usize;
    let named = model.named_parameters();
    if !named.is_empty() {
        for (name, param) in &named {
            if let Some(entry) = state_dict.get(name) {
                if let Ok(tensor) = entry.data.to_tensor() {
                    if tensor.shape() == param.data().shape() {
                        param.update_data(tensor);
                        loaded += 1;
                    }
                }
            }
        }
    }

    // Fallback: shape-based matching in-order (for models without named_parameters)
    if loaded == 0 {
        let saved_tensors: Vec<_> = state_dict
            .entries()
            .filter_map(|(_, entry)| entry.data.to_tensor().ok())
            .collect();
        let mut used = vec![false; saved_tensors.len()];
        for param in &params {
            let param_data = param.data();
            let param_shape = param_data.shape();
            for (i, saved) in saved_tensors.iter().enumerate() {
                if !used[i] && saved.shape() == param_shape {
                    param.update_data(saved.clone());
                    used[i] = true;
                    loaded += 1;
                    break;
                }
            }
        }
    }

    println!(
        "  Loaded {}/{} params from {} (epoch {})",
        loaded,
        params.len(),
        path.display(),
        checkpoint.epoch()
    );

    Ok((checkpoint.epoch(), checkpoint.training_state.clone()))
}

// =============================================================================
// I/O helpers
// =============================================================================

/// Read a text corpus from disk with a friendly error message.
pub fn read_corpus(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read corpus from {}: {}", path.display(), e);
        eprintln!();
        eprintln!("Expected Shakespeare corpus at /opt/datasets/text/shakespeare.txt");
        eprintln!("Or pass a different path with: --corpus /path/to/text.txt");
        std::process::exit(1);
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_roundtrip() {
        let corpus = "Hello, World!";
        let tok = CharTokenizer::from_corpus(corpus);
        let ids = tok.encode(corpus);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, corpus);
    }

    #[test]
    fn test_dataset_sample() {
        let corpus = "abcdefghijklmnopqrstuvwxyz";
        let tok = CharTokenizer::from_corpus(corpus);
        let ds = TextDataset::from_string(corpus, &tok, 4);
        let mut rng = 42u64;
        let batch = ds.sample_batch(3, &mut rng);
        assert_eq!(batch.len(), 3 * 4);
    }
}
