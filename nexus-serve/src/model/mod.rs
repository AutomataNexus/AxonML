//! Model loading and management.
//!
//! Supports:
//! - GGUF files (llama.cpp format, quantized)
//! - SafeTensors (HuggingFace format)
//! - AxonML native checkpoints

pub mod gguf;
pub mod inference;
pub mod registry;
