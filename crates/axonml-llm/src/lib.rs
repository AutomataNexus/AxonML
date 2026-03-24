//! axonml-llm - Large Language Model Architectures
//!
//! # File
//! `crates/axonml-llm/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod attention;
pub mod bert;
pub mod chimera;
pub mod config;
pub mod embedding;
pub mod error;
pub mod generation;
pub mod gpt2;
pub mod hf_loader;
pub mod hub;
pub mod hydra;
pub mod llama;
pub mod mistral;
pub mod phi;
pub mod ssm;
pub mod state_dict;
pub mod tokenizer;
pub mod transformer;
pub mod trident;

pub use attention::{
    scaled_dot_product_attention, CausalSelfAttention, FlashAttention, FlashAttentionConfig,
    KVCache, LayerKVCache, MultiHeadSelfAttention,
};
pub use bert::{Bert, BertForMaskedLM, BertForSequenceClassification};
pub use chimera::{ChimeraConfig, ChimeraModel};
pub use config::{BertConfig, GPT2Config, TransformerConfig};
pub use embedding::{BertEmbedding, GPT2Embedding, PositionalEmbedding, TokenEmbedding};
pub use error::{LLMError, LLMResult};
pub use generation::{GenerationConfig, TextGenerator};
pub use gpt2::{GPT2LMHead, GPT2};
pub use hf_loader::{load_llama_from_hf, load_mistral_from_hf, HFLoader};
pub use hub::{download_weights as download_llm_weights, llm_registry, PretrainedLLM};
pub use hydra::{HydraConfig, HydraModel};
pub use llama::{LLaMA, LLaMAConfig, LLaMAForCausalLM};
pub use mistral::{Mistral, MistralConfig, MistralForCausalLM};
pub use phi::{Phi, PhiConfig, PhiForCausalLM};
pub use ssm::{SSMBlock, SSMConfig};
pub use state_dict::{LoadResult, LoadStateDict};
pub use tokenizer::{HFTokenizer, SpecialTokens};
pub use transformer::{TransformerBlock, TransformerDecoder, TransformerEncoder};
pub use trident::{TridentConfig, TridentModel};

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
