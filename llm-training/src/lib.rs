//! llm-training — Shared Utilities for AxonML LLM Training Binaries
//!
//! Crate root that holds the pieces every `train_<arch>.rs` binary reuses:
//! - [`CharTokenizer`] — deterministic character-level tokenizer built from a
//!   corpus via [`CharTokenizer::from_corpus`], with `encode` / `decode` /
//!   `vocab_size` methods and token 0 reserved for the unknown / padding
//!   character (`'\0'`).
//! - [`TextDataset`] — sliding-window dataset for next-token prediction, with
//!   [`TextDataset::sample_batch`] returning a flat `Vec<u32>` of shape
//!   `[batch_size * seq_len]`.
//! - [`lcg_range`] — seedable linear congruential generator (no external RNG
//!   crate) used for batch sampling.
//! - [`format_count`] — thousands-separator formatter for reporting parameter
//!   counts and dataset sizes.
//! - [`ResumeMode`] / [`find_checkpoint`] / [`load_model_from_checkpoint`] —
//!   checkpoint-resume helpers that match saved tensors either by name or by
//!   falling back to shape-based in-order matching, so any AxonML [`Module`]
//!   can be resumed.
//! - [`shifted_cross_entropy`] — causal-LM loss that shifts logits/labels by
//!   one position, flattens to `[N, V]`, and moves the f32 target tensor onto
//!   the logits' device so the fused GPU cross-entropy kernel is used when
//!   available.
//! - [`read_corpus`] — opinionated corpus loader that prints a friendly
//!   Shakespeare-path hint on failure.
//! - [`lifecycle`] — re-export of the pause/resume/stop/checkpoint +
//!   always-on monitor subsystem (hard rule; every training binary adopts it).
//!
//! Each LLM architecture has its own binary under `src/bin/train_<name>.rs`
//! which pulls these utilities in plus its architecture crate.
//!
//! # File
//! `llm-training/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Module Exports & Imports
// =============================================================================

pub mod lifecycle;
pub use lifecycle::{LoopAction, TrainingLifecycle, TrainingLifecycleBuilder};

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_nn::loss::CrossEntropyLoss;
use axonml_serialize::{Checkpoint, TrainingState, load_checkpoint};
use axonml_tensor::Tensor;

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
            .map(|&id| self.id_to_char.get(id as usize).copied().unwrap_or('\0'))
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

    /// Load a pre-tokenized flat uint32 stream from disk.
    ///
    /// Produced by `tokenize_corpus` — each 4 bytes LE is one token ID.
    /// Intended for distillation against a real pre-trained teacher,
    /// where the student must see the exact token IDs from the
    /// teacher's tokenizer (not char IDs from `CharTokenizer`).
    pub fn from_tokens_bin(path: &Path, seq_len: usize) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "token bin file has {} bytes, not a multiple of 4 (corrupt?)",
                    bytes.len()
                ),
            ));
        }
        let n_tokens = bytes.len() / 4;
        let mut tokens: Vec<u32> = Vec::with_capacity(n_tokens);
        for chunk in bytes.chunks_exact(4) {
            tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Self { tokens, seq_len })
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
            if p.exists() {
                Some(p.clone())
            } else {
                None
            }
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
// Shifted cross-entropy loss for causal language modeling
// =============================================================================

/// Compute the standard causal-LM cross-entropy loss from a [B, S, V] logits
/// Variable and a [B, S] u32 labels Tensor: logits[:-1] predicts labels[1:].
///
/// This is the shift-then-reshape pattern that GPT-2, LLaMA, Mistral, Phi,
/// and SSM/Mamba all need. Hydra and Chimera expose their own
/// `forward_with_loss` methods and do not need this helper.
///
/// # Device handling
/// `labels` is expected on CPU (u32 tensors cannot be moved to GPU — moving a
/// `Tensor<u32>` with `--features cuda` enabled panics at `tensor.rs:682`
/// with *"GPU tensors are only supported for f32"*). The shifted-label f32
/// target tensor this function builds is moved to the logits' device so the
/// fused GPU cross-entropy kernel triggers when training on CUDA.
///
/// # Out-of-range labels
/// Any label index `>= vocab_size` is defensively clamped to 0 (padding),
/// matching the pattern used internally by `GPT2LMHead::cross_entropy_loss`.
pub fn shifted_cross_entropy(logits: &Variable, labels: &Tensor<u32>) -> Variable {
    let logits_data = logits.data();
    let shape = logits_data.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let vocab_size = shape[2];

    if seq_len <= 1 {
        // Degenerate case — return zero loss.
        let zero = Tensor::from_vec(vec![0.0f32], &[1]).unwrap();
        return Variable::new(zero, false);
    }

    // Shift logits: drop the last position → predict positions 1..S from 0..S-1.
    let shift_logits = logits.narrow(1, 0, seq_len - 1);
    let n = batch_size * (seq_len - 1);
    let logits_flat = shift_logits.reshape(&[n, vocab_size]);

    // Shift labels: drop position 0, keep positions 1..S, flatten to [N].
    let labels_vec = labels.to_vec();
    let mut shift_labels = Vec::with_capacity(n);
    for b in 0..batch_size {
        for s in 1..seq_len {
            let l = labels_vec[b * seq_len + s] as usize;
            shift_labels.push(if l < vocab_size { l as f32 } else { 0.0 });
        }
    }
    let mut target_tensor = Tensor::from_vec(shift_labels, &[n]).unwrap();

    // Move targets to the logits' device so the fused GPU CE kernel triggers.
    let logits_device = logits_data.device();
    if logits_device.is_gpu() {
        target_tensor = target_tensor.to_device(logits_device).unwrap();
    }
    let target_var = Variable::new(target_tensor, false);

    CrossEntropyLoss::new().compute(&logits_flat, &target_var)
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
