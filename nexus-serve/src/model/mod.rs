//! model — Model Loading + Inference Module Root
//!
//! Groups the model-facing submodules of nexus-serve:
//!
//! - [`gguf`]: zero-copy GGUF file reader (magic-byte check, metadata KV
//!   parser, tensor directory, [`gguf::GgufFile`], [`gguf::GgufValue`],
//!   [`gguf::GgufTensor`]).
//! - [`inference`]: the inference engine proper — [`inference::InferenceEngine`],
//!   [`inference::MappedGguf`] (memory-mapped tensor slices), per-architecture
//!   forward passes (LLaMA / Qwen / BitNet / Gemma), KV cache, split-halves
//!   RoPE, CUDA dispatch, fused flash-decode + prefill attention, sampling,
//!   and the `generate` / `generate_stream` entry points.
//! - [`registry`]: [`registry::ModelRegistry`] + [`registry::ModelInfo`] — an
//!   `RwLock`-backed map of loaded models plus alias resolution.
//! - [`weight`]: [`weight::Weight`] enum wrapping f32 row-major tensors and
//!   packed GGUF quant blocks (Q4_K / Q6_K / Q8_0 / F16), with CPU matmul,
//!   CUDA upload caches, and dequant helpers.
//!
//! Supports three on-disk formats: GGUF (primary), SafeTensors (HuggingFace),
//! and AxonML native checkpoints.
//!
//! # File
//! `nexus-serve/src/model/mod.rs`
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
// Submodules
// =============================================================================

pub mod gguf;
pub mod inference;
pub mod registry;
pub mod weight;
