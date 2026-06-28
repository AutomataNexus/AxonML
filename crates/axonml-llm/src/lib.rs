//! LLM architectures for the AxonML framework.
//!
//! Complete pure-Rust implementations: GPT-2 (decoder-only), LLaMA (split-
//! halves RoPE + GQA + SwiGLU), Mistral (sliding-window attention), Phi
//! (partial RoPE + GELU), BERT (bidirectional encoder + classification/MLM),
//! SSM/Mamba (selective S6 scan + depthwise conv + SSMForCausalLM), Qwen3
//! (GQA + QK-norm). Shared building blocks: attention, RMSNorm,
//! RotaryEmbedding, embedding, text generation (top-k/top-p/temperature),
//! HuggingFace weight loader, and pretrained model hub.
//!
//! # File
//! `crates/axonml-llm/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod attention;
pub mod bert;
pub mod config;
pub mod embedding;
pub mod error;
pub mod generation;
pub mod gguf_export;
pub mod gguf_loader;
pub mod gpt2;
pub mod hf_loader;
pub mod hub;
pub mod llama;
pub mod mistral;
pub mod phi;
pub mod qwen3;
pub mod ssm;
pub mod state_dict;
pub mod tokenizer;
pub mod transformer;

pub use attention::{
    CausalSelfAttention, FlashAttention, FlashAttentionConfig, KVCache, LayerKVCache,
    MultiHeadSelfAttention, scaled_dot_product_attention,
};
pub use bert::{Bert, BertForMaskedLM, BertForSequenceClassification};
pub use config::{BertConfig, GPT2Config, TransformerConfig};
pub use embedding::{BertEmbedding, GPT2Embedding, PositionalEmbedding, TokenEmbedding};
pub use error::{LLMError, LLMResult};
pub use generation::{GenerationConfig, TextGenerator};
pub use gguf_export::export_qwen3_to_gguf;
pub use gguf_loader::{load_qwen3_from_gguf, read_gguf_metadata_raw_bytes, read_gguf_tokenizer};
pub use gpt2::{GPT2, GPT2LMHead};
pub use hf_loader::{HFLoader, load_llama_from_hf, load_mistral_from_hf};
pub use hub::{PretrainedLLM, download_weights as download_llm_weights, llm_registry};
pub use llama::{LLaMA, LLaMAConfig, LLaMAForCausalLM};
pub use mistral::{Mistral, MistralConfig, MistralForCausalLM};
pub use phi::{Phi, PhiConfig, PhiForCausalLM};
pub use qwen3::{Qwen3, Qwen3Attention, Qwen3Config, Qwen3DecoderLayer, Qwen3ForCausalLM};
pub use ssm::{SSMBlock, SSMConfig, SSMForCausalLM};
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
