//! axonml-llm - Large Language Model Architectures
//!
//! This crate provides implementations of popular transformer-based language models
//! including BERT, GPT-2, and building blocks for custom LLM architectures.
//!
//! # Key Features
//! - BERT (Bidirectional Encoder Representations from Transformers)
//! - GPT-2 (Generative Pre-trained Transformer 2)
//! - Transformer building blocks (attention, feed-forward, positional encoding)
//! - Text generation utilities
//!
//! # Example
//! ```ignore
//! use axonml_llm::{GPT2, GPT2Config};
//! use axonml_tensor::Tensor;
//!
//! // Create a GPT-2 model
//! let config = GPT2Config::small();
//! let model = GPT2::new(&config);
//!
//! // Generate text
//! let input_ids = Tensor::from_vec(vec![50256u32], &[1, 1]).unwrap();
//! let output = model.forward(&input_ids);
//! ```
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod config;
pub mod attention;
pub mod embedding;
pub mod hub;
pub mod transformer;
pub mod bert;
pub mod gpt2;
pub mod generation;

pub use error::{LLMError, LLMResult};
pub use config::{BertConfig, GPT2Config, TransformerConfig};
pub use attention::{MultiHeadSelfAttention, CausalSelfAttention};
pub use embedding::{TokenEmbedding, PositionalEmbedding, BertEmbedding, GPT2Embedding};
pub use hub::{PretrainedLLM, llm_registry, download_weights as download_llm_weights};
pub use transformer::{TransformerBlock, TransformerEncoder, TransformerDecoder};
pub use bert::{Bert, BertForSequenceClassification, BertForMaskedLM};
pub use gpt2::{GPT2, GPT2LMHead};
pub use generation::{GenerationConfig, TextGenerator};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_config() {
        let config = GPT2Config::small();
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
        assert_eq!(config.n_embd, 768);
    }

    #[test]
    fn test_bert_config() {
        let config = BertConfig::base();
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.hidden_size, 768);
    }
}
