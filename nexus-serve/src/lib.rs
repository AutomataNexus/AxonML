//! nexus-serve — Pure-Rust LLM inference server
//!
//! Replaces ollama with native AxonML inference. Exposes an OpenAI-compatible
//! REST API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`).
//!
//! Supports:
//! - GGUF model files (quantized, from HuggingFace / llama.cpp)
//! - SafeTensors (HuggingFace format)
//! - AxonML native checkpoints (`.axonml`)
//! - HuggingFace `tokenizer.json` + char-level fallback
//! - CUDA GPU acceleration via `--features cuda`
//! - LLaMA-family architectures (LLaMA, Qwen2, Mistral) with split-halves RoPE

pub mod api;
pub mod model;
pub mod tokenizer;
