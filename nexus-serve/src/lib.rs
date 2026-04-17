//! nexus-serve — Pure-Rust LLM Inference Server Library Root
//!
//! Top-level library crate for nexus-serve, AutomataNexus's native Rust LLM
//! inference stack. This is NOT a llama.cpp wrapper; it is its own inference
//! engine with custom CUDA kernels (Q4_K / Q6_K dequant-in-shader GEMV, fused
//! flash-decode + prefill attention), OnceLock-cached GPU weight uploads, and
//! an Anthropic Messages API with SSE streaming.
//!
//! Re-exports three top-level modules:
//! - [`api`]: HTTP surface (routes, Messages API, OpenAI-compat types, SSE)
//! - [`model`]: GGUF/SafeTensors/AxonML loading, inference loop, registry
//! - [`tokenizer`]: HuggingFace `tokenizer.json` + char-level fallback
//!
//! Supported input formats: GGUF (quantized, from HuggingFace / llama.cpp),
//! SafeTensors (HuggingFace format), and AxonML native checkpoints (`.axonml`).
//! Supported architectures: LLaMA-family (LLaMA, Qwen2, Mistral) with
//! split-halves RoPE, plus Gemma 3 / BitNet via architecture dispatch in
//! [`model::inference`]. CUDA acceleration is gated behind `--features cuda`.
//!
//! # File
//! `nexus-serve/src/lib.rs`
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
// Public Modules
// =============================================================================

pub mod api;
pub mod model;
pub mod tokenizer;
