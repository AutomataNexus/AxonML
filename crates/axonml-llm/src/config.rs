//! Model Configuration Module
//!
//! Configuration structs for BERT, GPT-2, and transformer models.

use serde::{Serialize, Deserialize};

/// Base transformer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Hidden size / embedding dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate (feed-forward) size
    pub intermediate_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Dropout probability
    pub dropout_prob: f32,
    /// Attention dropout probability
    pub attention_dropout_prob: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Activation function (gelu, relu, etc.)
    pub activation: String,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_attention_heads: 12,
            intermediate_size: 3072,
            num_layers: 12,
            max_position_embeddings: 512,
            vocab_size: 30522,
            dropout_prob: 0.1,
            attention_dropout_prob: 0.1,
            layer_norm_eps: 1e-12,
            activation: "gelu".to_string(),
        }
    }
}

/// BERT model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Type vocabulary size (for segment embeddings)
    pub type_vocab_size: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Pad token ID
    pub pad_token_id: usize,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self::base()
    }
}

impl BertConfig {
    /// Creates a BERT-base configuration.
    pub fn base() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    /// Creates a BERT-large configuration.
    pub fn large() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    /// Creates a tiny BERT configuration for testing.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 128,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    /// Returns the head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// GPT-2 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPT2Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Context length (maximum sequence length)
    pub n_ctx: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Attention dropout probability
    pub attn_dropout: f32,
    /// Residual dropout probability
    pub resid_dropout: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Activation function
    pub activation: String,
    /// BOS token ID
    pub bos_token_id: usize,
    /// EOS token ID
    pub eos_token_id: usize,
}

impl Default for GPT2Config {
    fn default() -> Self {
        Self::small()
    }
}

impl GPT2Config {
    /// Creates a GPT-2 Small (117M) configuration.
    pub fn small() -> Self {
        Self {
            vocab_size: 50257,
            n_ctx: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            dropout: 0.1,
            attn_dropout: 0.1,
            resid_dropout: 0.1,
            layer_norm_eps: 1e-5,
            activation: "gelu".to_string(),
            bos_token_id: 50256,
            eos_token_id: 50256,
        }
    }

    /// Creates a GPT-2 Medium (345M) configuration.
    pub fn medium() -> Self {
        Self {
            vocab_size: 50257,
            n_ctx: 1024,
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            dropout: 0.1,
            attn_dropout: 0.1,
            resid_dropout: 0.1,
            layer_norm_eps: 1e-5,
            activation: "gelu".to_string(),
            bos_token_id: 50256,
            eos_token_id: 50256,
        }
    }

    /// Creates a GPT-2 Large (774M) configuration.
    pub fn large() -> Self {
        Self {
            vocab_size: 50257,
            n_ctx: 1024,
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            dropout: 0.1,
            attn_dropout: 0.1,
            resid_dropout: 0.1,
            layer_norm_eps: 1e-5,
            activation: "gelu".to_string(),
            bos_token_id: 50256,
            eos_token_id: 50256,
        }
    }

    /// Creates a GPT-2 XL (1.5B) configuration.
    pub fn xl() -> Self {
        Self {
            vocab_size: 50257,
            n_ctx: 1024,
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            dropout: 0.1,
            attn_dropout: 0.1,
            resid_dropout: 0.1,
            layer_norm_eps: 1e-5,
            activation: "gelu".to_string(),
            bos_token_id: 50256,
            eos_token_id: 50256,
        }
    }

    /// Creates a tiny GPT-2 configuration for testing.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            n_ctx: 128,
            n_embd: 128,
            n_layer: 2,
            n_head: 2,
            dropout: 0.1,
            attn_dropout: 0.1,
            resid_dropout: 0.1,
            layer_norm_eps: 1e-5,
            activation: "gelu".to_string(),
            bos_token_id: 0,
            eos_token_id: 0,
        }
    }

    /// Returns the head dimension.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_configs() {
        let base = BertConfig::base();
        assert_eq!(base.hidden_size, 768);
        assert_eq!(base.num_hidden_layers, 12);
        assert_eq!(base.head_dim(), 64);

        let large = BertConfig::large();
        assert_eq!(large.hidden_size, 1024);
        assert_eq!(large.num_hidden_layers, 24);
        assert_eq!(large.head_dim(), 64);
    }

    #[test]
    fn test_gpt2_configs() {
        let small = GPT2Config::small();
        assert_eq!(small.n_embd, 768);
        assert_eq!(small.n_layer, 12);
        assert_eq!(small.head_dim(), 64);

        let medium = GPT2Config::medium();
        assert_eq!(medium.n_embd, 1024);
        assert_eq!(medium.n_layer, 24);

        let large = GPT2Config::large();
        assert_eq!(large.n_embd, 1280);
        assert_eq!(large.n_layer, 36);

        let xl = GPT2Config::xl();
        assert_eq!(xl.n_embd, 1600);
        assert_eq!(xl.n_layer, 48);
    }
}
