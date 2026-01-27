//! axonml-llm - Large Language Model Architectures
//!
//! This crate provides implementations of popular transformer-based language models
//! including BERT, GPT-2, LLaMA, Mistral, and Phi, along with building blocks for
//! custom LLM architectures.
//!
//! # Key Features
//! - BERT (Bidirectional Encoder Representations from Transformers)
//! - GPT-2 (Generative Pre-trained Transformer 2)
//! - LLaMA (Large Language Model Meta AI) with RoPE and SwiGLU
//! - Mistral with sliding window attention
//! - Phi with partial rotary embeddings
//! - KV-cache for efficient autoregressive generation
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
//! @version 0.2.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod attention;
pub mod bert;
pub mod config;
pub mod embedding;
pub mod error;
pub mod generation;
pub mod gpt2;
pub mod hf_loader;
pub mod hub;
pub mod llama;
pub mod mistral;
pub mod phi;
pub mod state_dict;
pub mod tokenizer;
pub mod transformer;

pub use attention::{
    scaled_dot_product_attention, CausalSelfAttention, FlashAttention, FlashAttentionConfig,
    KVCache, LayerKVCache, MultiHeadSelfAttention,
};
pub use bert::{Bert, BertForMaskedLM, BertForSequenceClassification};
pub use config::{BertConfig, GPT2Config, TransformerConfig};
pub use embedding::{BertEmbedding, GPT2Embedding, PositionalEmbedding, TokenEmbedding};
pub use error::{LLMError, LLMResult};
pub use generation::{GenerationConfig, TextGenerator};
pub use gpt2::{GPT2LMHead, GPT2};
pub use hf_loader::{load_llama_from_hf, load_mistral_from_hf, HFLoader};
pub use hub::{download_weights as download_llm_weights, llm_registry, PretrainedLLM};
pub use llama::{LLaMA, LLaMAConfig, LLaMAForCausalLM};
pub use mistral::{Mistral, MistralConfig, MistralForCausalLM};
pub use phi::{Phi, PhiConfig, PhiForCausalLM};
pub use state_dict::{LoadResult, LoadStateDict};
pub use tokenizer::{HFTokenizer, SpecialTokens};
pub use transformer::{TransformerBlock, TransformerDecoder, TransformerEncoder};

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
