//! biometric-training — Shared Utilities for the Aegis Biometric Suite
//!
//! Crate root that holds the pieces every `train_*.rs` modality trainer
//! (Argus / Ariadne / Mnemosyne) reuses:
//! - [`IdentityRecord`] + [`IdentityDataset`] — generic per-identity binary
//!   loader. Every modality preprocesses its data into `identity_NNNN.bin`
//!   files with a 16-byte `u32 LE` header (num_samples, channels, height,
//!   width) followed by flat f32 pixel data. [`IdentityDataset::load`]
//!   sorts filenames to produce stable label IDs, validates shape
//!   consistency, and exposes `sample_len`, `num_identities`,
//!   `total_samples`, and `count_with_at_least` accessors.
//! - [`mine_triplet_batch`] — random (anchor, positive, negative) triplet
//!   sampler used by Argus training.
//! - [`mine_identity_sequence_batches`] — per-step triplet sequences used
//!   by Mnemosyne's crystallization training.
//! - [`mine_pair_batch`] — 50/50 same/different pair sampler used by
//!   Ariadne contrastive training.
//! - [`l2_normalize_var`] — graph-tracked L2 norm (via
//!   `mul_var`/`sum`/`sqrt`/`div_var`) used by every modality head so
//!   embeddings live on the unit hypersphere.
//! - [`lcg_range`] — Numerical Recipes LCG for deterministic batch mining.
//! - [`format_count`] — thousand-separator formatter.
//! - [`ResumeMode`] / [`find_checkpoint`] / [`load_model_from_checkpoint`]
//!   — checkpoint-resume helpers. Duplicated from `llm-training` on
//!   purpose so `biometric-training` stays a self-contained standalone
//!   crate.
//!
//! Each modality has its own training binary under `src/bin/train_*.rs`
//! which pulls these utilities in plus its per-modality model crate.
//!
//! # File
//! `biometric-training/src/lib.rs`
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
// Imports
// =============================================================================

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_serialize::{load_checkpoint, Checkpoint, TrainingState};

// =============================================================================
// Generic per-identity binary dataset
// =============================================================================
//
// All three biometric modalities preprocess their data into a uniform binary
// format: one file per identity, with a 16-byte header followed by flat f32
// pixel data. The header layout is:
//
//   bytes  0..4   : num_samples (u32, little-endian)
//   bytes  4..8   : channels    (u32)
//   bytes  8..12  : height      (u32)
//   bytes 12..16  : width       (u32)
//   bytes 16..N   : num_samples * channels * height * width * 4 bytes of f32
//
// Filename convention: `identity_NNNN.bin` where NNNN is the zero-padded
// label index. The loader sorts by filename to produce stable label IDs.

/// A single identity's samples stored as flat f32 vectors.
pub struct IdentityRecord {
    /// Each sample is `channels * height * width` floats.
    pub samples: Vec<Vec<f32>>,
}

/// A loaded per-identity dataset — just a vector of `IdentityRecord`s.
/// Labels are implicit: index in the vector.
pub struct IdentityDataset {
    pub identities: Vec<IdentityRecord>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl IdentityDataset {
    /// Sample size in floats: `channels * height * width`.
    pub fn sample_len(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Load every `identity_NNNN.bin` file in `data_dir` (sorted).
    pub fn load(data_dir: &Path) -> Self {
        let mut files: Vec<_> = fs::read_dir(data_dir)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to read dataset directory {}: {}",
                    data_dir.display(),
                    e
                )
            })
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .map(|f| f.to_string_lossy().starts_with("identity_"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort_by_key(|e| e.file_name());

        let mut identities = Vec::with_capacity(files.len());
        let mut channels = 0usize;
        let mut height = 0usize;
        let mut width = 0usize;

        for entry in &files {
            let path = entry.path();
            let mut file = fs::File::open(&path)
                .unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e));
            let mut header = [0u8; 16];
            file.read_exact(&mut header)
                .unwrap_or_else(|e| panic!("Short header in {}: {}", path.display(), e));

            let num = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
            let c = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
            let h = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
            let w = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;

            if channels == 0 {
                channels = c;
                height = h;
                width = w;
            } else if channels != c || height != h || width != w {
                panic!(
                    "Shape mismatch in {}: expected [{},{},{}], got [{},{},{}]",
                    path.display(),
                    channels,
                    height,
                    width,
                    c,
                    h,
                    w
                );
            }

            let sample_bytes = c * h * w * 4;
            let mut byte_buf = vec![0u8; num * sample_bytes];
            file.read_exact(&mut byte_buf)
                .unwrap_or_else(|e| panic!("Short data in {}: {}", path.display(), e));

            let all: Vec<f32> = byte_buf
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            let sample_size = c * h * w;
            let mut samples = Vec::with_capacity(num);
            for i in 0..num {
                samples.push(all[i * sample_size..(i + 1) * sample_size].to_vec());
            }

            identities.push(IdentityRecord { samples });
        }

        IdentityDataset {
            identities,
            channels,
            height,
            width,
        }
    }

    /// Number of identities loaded.
    pub fn num_identities(&self) -> usize {
        self.identities.len()
    }

    /// Total sample count across all identities.
    pub fn total_samples(&self) -> usize {
        self.identities.iter().map(|id| id.samples.len()).sum()
    }

    /// Number of identities with at least `k` samples.
    pub fn count_with_at_least(&self, k: usize) -> usize {
        self.identities
            .iter()
            .filter(|id| id.samples.len() >= k)
            .count()
    }
}

// =============================================================================
// Triplet mining (random)
// =============================================================================

/// Mine a batch of triplets uniformly at random: anchor + positive from the
/// same identity, negative from a different identity. Requires identities
/// with at least 2 samples; falls back to reusing the anchor sample if an
/// identity has only 1 sample.
///
/// Returns three flat `Vec<f32>` buffers of shape `[batch * sample_len]`.
pub fn mine_triplet_batch(
    dataset: &IdentityDataset,
    batch_size: usize,
    rng: &mut u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sample_len = dataset.sample_len();
    let mut anchor_buf = Vec::with_capacity(batch_size * sample_len);
    let mut pos_buf = Vec::with_capacity(batch_size * sample_len);
    let mut neg_buf = Vec::with_capacity(batch_size * sample_len);

    let n_ids = dataset.num_identities();
    for _ in 0..batch_size {
        let anchor_id = lcg_range(rng, n_ids);
        let anchor_rec = &dataset.identities[anchor_id];
        let a_idx = lcg_range(rng, anchor_rec.samples.len());
        let p_idx = if anchor_rec.samples.len() > 1 {
            let mut p = lcg_range(rng, anchor_rec.samples.len());
            while p == a_idx {
                p = lcg_range(rng, anchor_rec.samples.len());
            }
            p
        } else {
            a_idx
        };

        let mut neg_id = lcg_range(rng, n_ids);
        while neg_id == anchor_id {
            neg_id = lcg_range(rng, n_ids);
        }
        let neg_rec = &dataset.identities[neg_id];
        let n_idx = lcg_range(rng, neg_rec.samples.len());

        anchor_buf.extend_from_slice(&anchor_rec.samples[a_idx]);
        pos_buf.extend_from_slice(&anchor_rec.samples[p_idx]);
        neg_buf.extend_from_slice(&neg_rec.samples[n_idx]);
    }

    (anchor_buf, pos_buf, neg_buf)
}

/// Mine a batch of identity sequences: for each item in the batch, sample
/// `seq_len` samples from the same identity. Returns `seq_len` buffers, each
/// of shape `[batch * sample_len]` — one per sequence step. The caller feeds
/// these step-by-step to the model's temporal loop.
///
/// Used by Mnemosyne for crystallization training.
pub fn mine_identity_sequence_batches(
    dataset: &IdentityDataset,
    batch_size: usize,
    seq_len: usize,
    rng: &mut u64,
) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // Pre-pick anchor_id + neg_id for each triplet in the batch
    let n_ids = dataset.num_identities();
    let valid_ids: Vec<usize> = (0..n_ids)
        .filter(|&i| dataset.identities[i].samples.len() >= 1)
        .collect();

    let mut triplet_ids = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let a = valid_ids[lcg_range(rng, valid_ids.len())];
        let mut n = lcg_range(rng, n_ids);
        while n == a {
            n = lcg_range(rng, n_ids);
        }
        triplet_ids.push((a, n));
    }

    let sample_len = dataset.sample_len();
    let mut steps = Vec::with_capacity(seq_len);
    for _ in 0..seq_len {
        let mut anchor_buf = Vec::with_capacity(batch_size * sample_len);
        let mut pos_buf = Vec::with_capacity(batch_size * sample_len);
        let mut neg_buf = Vec::with_capacity(batch_size * sample_len);

        for &(a_id, n_id) in &triplet_ids {
            let a_rec = &dataset.identities[a_id];
            let n_rec = &dataset.identities[n_id];

            let a_idx = lcg_range(rng, a_rec.samples.len());
            let p_idx = if a_rec.samples.len() > 1 {
                let mut p = lcg_range(rng, a_rec.samples.len());
                while p == a_idx {
                    p = lcg_range(rng, a_rec.samples.len());
                }
                p
            } else {
                a_idx
            };
            let n_idx = lcg_range(rng, n_rec.samples.len());

            anchor_buf.extend_from_slice(&a_rec.samples[a_idx]);
            pos_buf.extend_from_slice(&a_rec.samples[p_idx]);
            neg_buf.extend_from_slice(&n_rec.samples[n_idx]);
        }

        steps.push((anchor_buf, pos_buf, neg_buf));
    }

    steps
}

/// Mine a batch of pairs (50% same-identity, 50% different-identity).
/// Used by Ariadne contrastive training.
pub fn mine_pair_batch(
    dataset: &IdentityDataset,
    batch_size: usize,
    rng: &mut u64,
) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let sample_len = dataset.sample_len();
    let mut a_buf = Vec::with_capacity(batch_size * sample_len);
    let mut b_buf = Vec::with_capacity(batch_size * sample_len);
    let mut same = Vec::with_capacity(batch_size);

    let n_ids = dataset.num_identities();
    for i in 0..batch_size {
        let is_same = i % 2 == 0;
        same.push(is_same);

        if is_same {
            let id = lcg_range(rng, n_ids);
            let rec = &dataset.identities[id];
            let a = lcg_range(rng, rec.samples.len());
            let b = if rec.samples.len() > 1 {
                let mut bb = lcg_range(rng, rec.samples.len());
                while bb == a {
                    bb = lcg_range(rng, rec.samples.len());
                }
                bb
            } else {
                a
            };
            a_buf.extend_from_slice(&rec.samples[a]);
            b_buf.extend_from_slice(&rec.samples[b]);
        } else {
            let id_a = lcg_range(rng, n_ids);
            let mut id_b = lcg_range(rng, n_ids);
            while id_b == id_a {
                id_b = lcg_range(rng, n_ids);
            }
            let rec_a = &dataset.identities[id_a];
            let rec_b = &dataset.identities[id_b];
            let a = lcg_range(rng, rec_a.samples.len());
            let b = lcg_range(rng, rec_b.samples.len());
            a_buf.extend_from_slice(&rec_a.samples[a]);
            b_buf.extend_from_slice(&rec_b.samples[b]);
        }
    }

    (a_buf, b_buf, same)
}

// =============================================================================
// L2 normalization (graph-tracked)
// =============================================================================

/// Graph-tracked L2 normalization of a `[B, D]` Variable. Divides each row
/// by its Euclidean norm so the output lies on the unit hypersphere.
pub fn l2_normalize_var(x: &Variable) -> Variable {
    let sq = x.mul_var(x);
    let sum_sq = sq.sum();
    let norm = sum_sq.add_scalar(1e-8).sqrt();
    x.div_var(&norm)
}

// =============================================================================
// Seedable LCG RNG
// =============================================================================

pub fn lcg_range(state: &mut u64, max: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as usize) % max.max(1)
}

// =============================================================================
// Number formatting
// =============================================================================

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
// Checkpoint resume
// =============================================================================

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

/// Load a checkpoint into a model. Prefers named-parameter matching; falls
/// back to in-order shape matching for models without `named_parameters`
/// implementations. Returns `(starting_epoch, training_state)`.
pub fn load_model_from_checkpoint<M: Module>(
    model: &M,
    path: &Path,
) -> Result<(usize, TrainingState), String> {
    let checkpoint: Checkpoint =
        load_checkpoint(path).map_err(|e| format!("Failed to load checkpoint: {e}"))?;

    let state_dict = &checkpoint.model_state;
    let params = model.parameters();
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
