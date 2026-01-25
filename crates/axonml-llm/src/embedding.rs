//! Embedding Module
//!
//! Token, positional, and combined embeddings for transformer models.

use axonml_autograd::Variable;
use axonml_nn::{Module, Embedding, Parameter, Dropout};
use axonml_tensor::Tensor;
use axonml_tensor::creation::{zeros, ones};

/// Token embedding layer.
#[derive(Debug)]
pub struct TokenEmbedding {
    /// Embedding layer
    pub embedding: Embedding,
}

impl TokenEmbedding {
    /// Creates a new token embedding.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
        }
    }

    /// Gets embeddings for token IDs.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        // Convert u32 to indices and lookup
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        let embed_dim = self.embedding.embedding_dim();

        let ids_vec = input_ids.to_vec();
        let mut output_data = vec![0.0f32; batch_size * seq_len * embed_dim];

        let weight = &self.embedding.weight;
        let weight_data = weight.data().to_vec();

        for b in 0..batch_size {
            for s in 0..seq_len {
                let idx = ids_vec[b * seq_len + s] as usize;
                let src_offset = idx * embed_dim;
                let dst_offset = (b * seq_len + s) * embed_dim;

                for e in 0..embed_dim {
                    output_data[dst_offset + e] = weight_data[src_offset + e];
                }
            }
        }

        let output_tensor = Tensor::from_vec(output_data, &[batch_size, seq_len, embed_dim]).unwrap();
        Variable::new(output_tensor, weight.requires_grad())
    }
}

impl Module for TokenEmbedding {
    fn forward(&self, input: &Variable) -> Variable {
        self.embedding.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.embedding.parameters()
    }
}

/// Learned positional embedding.
#[derive(Debug)]
pub struct PositionalEmbedding {
    /// Position embedding weights
    pub embedding: Embedding,
    /// Maximum sequence length
    pub max_len: usize,
}

impl PositionalEmbedding {
    /// Creates a new learned positional embedding.
    pub fn new(max_len: usize, embed_dim: usize) -> Self {
        Self {
            embedding: Embedding::new(max_len, embed_dim),
            max_len,
        }
    }

    /// Gets positional embeddings for a sequence length.
    pub fn forward_positions(&self, seq_len: usize, batch_size: usize) -> Variable {
        let embed_dim = self.embedding.embedding_dim();

        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let position_tensor = Tensor::from_vec(positions.clone(), &[1, seq_len]).unwrap();
        let position_var = Variable::new(position_tensor, false);

        // Lookup embeddings
        let pos_embeds = self.embedding.forward(&position_var);

        // Expand to batch size
        if batch_size > 1 {
            pos_embeds.expand(&[batch_size, seq_len, embed_dim])
        } else {
            pos_embeds
        }
    }
}

impl Module for PositionalEmbedding {
    fn forward(&self, input: &Variable) -> Variable {
        self.embedding.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.embedding.parameters()
    }
}

/// Sinusoidal positional encoding (fixed, not learned).
#[derive(Debug)]
pub struct SinusoidalPositionalEncoding {
    /// Precomputed positional encodings
    pub encodings: Tensor<f32>,
    /// Maximum sequence length
    pub max_len: usize,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl SinusoidalPositionalEncoding {
    /// Creates sinusoidal positional encodings.
    pub fn new(max_len: usize, embed_dim: usize) -> Self {
        let mut encodings = vec![0.0f32; max_len * embed_dim];

        for pos in 0..max_len {
            for i in 0..embed_dim / 2 {
                let div_term = (10000.0f32).powf(2.0 * i as f32 / embed_dim as f32);
                let angle = pos as f32 / div_term;

                encodings[pos * embed_dim + 2 * i] = angle.sin();
                encodings[pos * embed_dim + 2 * i + 1] = angle.cos();
            }
        }

        Self {
            encodings: Tensor::from_vec(encodings, &[max_len, embed_dim]).unwrap(),
            max_len,
            embed_dim,
        }
    }

    /// Gets positional encodings for a sequence.
    pub fn forward_seq(&self, seq_len: usize) -> Variable {
        let sliced = self.encodings.slice(&[0..seq_len, 0..self.embed_dim]);
        Variable::new(sliced, false)
    }
}

/// BERT-style embeddings (token + position + segment).
#[derive(Debug)]
pub struct BertEmbedding {
    /// Token embeddings
    pub word_embeddings: Embedding,
    /// Position embeddings
    pub position_embeddings: Embedding,
    /// Token type embeddings (segment embeddings)
    pub token_type_embeddings: Embedding,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Dropout
    pub dropout: Dropout,
    /// Embedding dimension
    pub embed_dim: usize,
}

/// Simple layer norm implementation for embeddings.
#[derive(Debug)]
pub struct LayerNorm {
    weight: Parameter,
    bias: Parameter,
    eps: f32,
}

impl LayerNorm {
    fn new(dim: usize, eps: f32) -> Self {
        let weight = Parameter::new(ones::<f32>(&[dim]), true);
        let bias = Parameter::new(zeros::<f32>(&[dim]), true);
        Self { weight, bias, eps }
    }

    fn forward(&self, x: &Variable) -> Variable {
        // Normalize over last dimension
        let mean = x.mean_dim(-1, true);
        let variance = x.var_dim(-1, true);

        let x_normalized = x.sub(&mean).div(&variance.add_scalar(self.eps).sqrt());

        // Scale and shift
        let weight_var = Variable::from_tensor_with_grad(self.weight.data().clone(), self.weight.requires_grad());
        let bias_var = Variable::from_tensor_with_grad(self.bias.data().clone(), self.bias.requires_grad());

        x_normalized.mul(&weight_var).add(&bias_var)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl BertEmbedding {
    /// Creates BERT embeddings.
    pub fn new(
        vocab_size: usize,
        max_position_embeddings: usize,
        type_vocab_size: usize,
        hidden_size: usize,
        layer_norm_eps: f32,
        dropout_prob: f32,
    ) -> Self {
        Self {
            word_embeddings: Embedding::new(vocab_size, hidden_size),
            position_embeddings: Embedding::new(max_position_embeddings, hidden_size),
            token_type_embeddings: Embedding::new(type_vocab_size, hidden_size),
            layer_norm: LayerNorm::new(hidden_size, layer_norm_eps),
            dropout: Dropout::new(dropout_prob),
            embed_dim: hidden_size,
        }
    }

    /// Forward pass with token IDs, position IDs, and token type IDs.
    pub fn forward_with_ids(
        &self,
        input_ids: &Tensor<u32>,
        token_type_ids: Option<&Tensor<u32>>,
        position_ids: Option<&Tensor<u32>>,
    ) -> Variable {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Token embeddings
        let input_ids_f32 = Self::u32_to_f32_tensor(input_ids);
        let word_embeds = self.word_embeddings.forward(&Variable::new(input_ids_f32, false));

        // Position embeddings
        let pos_ids = if let Some(ids) = position_ids {
            Self::u32_to_f32_tensor(ids)
        } else {
            let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
            let pos_data: Vec<f32> = (0..batch_size).flat_map(|_| positions.iter().cloned()).collect();
            Tensor::from_vec(pos_data, &[batch_size, seq_len]).unwrap()
        };
        let position_embeds = self.position_embeddings.forward(&Variable::new(pos_ids, false));

        // Token type embeddings
        let type_ids = if let Some(ids) = token_type_ids {
            Self::u32_to_f32_tensor(ids)
        } else {
            zeros::<f32>(&[batch_size, seq_len])
        };
        let token_type_embeds = self.token_type_embeddings.forward(&Variable::new(type_ids, false));

        // Combine embeddings
        let embeddings = word_embeds.add(&position_embeds).add(&token_type_embeds);

        // Layer norm and dropout
        let embeddings = self.layer_norm.forward(&embeddings);
        self.dropout.forward(&embeddings)
    }

    fn u32_to_f32_tensor(t: &Tensor<u32>) -> Tensor<f32> {
        let data: Vec<f32> = t.to_vec().iter().map(|&x| x as f32).collect();
        Tensor::from_vec(data, t.shape()).unwrap()
    }
}

impl Module for BertEmbedding {
    fn forward(&self, input: &Variable) -> Variable {
        // For Module trait, assume input is already f32 token indices
        let input_data = input.data();
        let shape = input_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        let word_embeds = self.word_embeddings.forward(input);

        // Generate position IDs
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let pos_data: Vec<f32> = (0..batch_size).flat_map(|_| positions.iter().cloned()).collect();
        let pos_tensor = Tensor::from_vec(pos_data, &[batch_size, seq_len]).unwrap();
        let position_embeds = self.position_embeddings.forward(&Variable::new(pos_tensor, false));

        // Token type embeddings (assume all zeros)
        let type_tensor = zeros::<f32>(&[batch_size, seq_len]);
        let token_type_embeds = self.token_type_embeddings.forward(&Variable::new(type_tensor, false));

        let embeddings = word_embeds.add(&position_embeds).add(&token_type_embeds);
        let embeddings = self.layer_norm.forward(&embeddings);
        self.dropout.forward(&embeddings)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.word_embeddings.parameters());
        params.extend(self.position_embeddings.parameters());
        params.extend(self.token_type_embeddings.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }
}

/// GPT-2 style embeddings (token + position).
#[derive(Debug)]
pub struct GPT2Embedding {
    /// Token embeddings
    pub wte: Embedding,
    /// Position embeddings
    pub wpe: Embedding,
    /// Dropout
    pub dropout: Dropout,
    /// Embedding dimension
    pub n_embd: usize,
}

impl GPT2Embedding {
    /// Creates GPT-2 embeddings.
    pub fn new(vocab_size: usize, n_ctx: usize, n_embd: usize, dropout: f32) -> Self {
        Self {
            wte: Embedding::new(vocab_size, n_embd),
            wpe: Embedding::new(n_ctx, n_embd),
            dropout: Dropout::new(dropout),
            n_embd,
        }
    }

    /// Forward pass with token IDs.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Token embeddings
        let input_ids_f32 = Self::u32_to_f32_tensor(input_ids);
        let token_embeds = self.wte.forward(&Variable::new(input_ids_f32, false));

        // Position embeddings
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let pos_data: Vec<f32> = (0..batch_size).flat_map(|_| positions.iter().cloned()).collect();
        let pos_tensor = Tensor::from_vec(pos_data, &[batch_size, seq_len]).unwrap();
        let position_embeds = self.wpe.forward(&Variable::new(pos_tensor, false));

        // Combine and apply dropout
        let embeddings = token_embeds.add(&position_embeds);
        self.dropout.forward(&embeddings)
    }

    fn u32_to_f32_tensor(t: &Tensor<u32>) -> Tensor<f32> {
        let data: Vec<f32> = t.to_vec().iter().map(|&x| x as f32).collect();
        Tensor::from_vec(data, t.shape()).unwrap()
    }
}

impl Module for GPT2Embedding {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape = input_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        let token_embeds = self.wte.forward(input);

        // Position embeddings
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let pos_data: Vec<f32> = (0..batch_size).flat_map(|_| positions.iter().cloned()).collect();
        let pos_tensor = Tensor::from_vec(pos_data, &[batch_size, seq_len]).unwrap();
        let position_embeds = self.wpe.forward(&Variable::new(pos_tensor, false));

        let embeddings = token_embeds.add(&position_embeds);
        self.dropout.forward(&embeddings)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.wte.parameters());
        params.extend(self.wpe.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding() {
        let embed = TokenEmbedding::new(1000, 64);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let output = embed.forward_ids(&input_ids);

        assert_eq!(output.data().shape(), &[2, 2, 64]);
    }

    #[test]
    fn test_positional_embedding() {
        let embed = PositionalEmbedding::new(128, 64);
        let output = embed.forward_positions(16, 2);

        assert_eq!(output.data().shape(), &[2, 16, 64]);
    }

    #[test]
    fn test_sinusoidal_encoding() {
        let encoding = SinusoidalPositionalEncoding::new(100, 64);
        let output = encoding.forward_seq(16);

        assert_eq!(output.data().shape(), &[16, 64]);
    }

    #[test]
    fn test_gpt2_embedding() {
        let embed = GPT2Embedding::new(1000, 128, 64, 0.0);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
        let output = embed.forward_ids(&input_ids);

        assert_eq!(output.data().shape(), &[2, 2, 64]);
    }
}
