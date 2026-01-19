//! Transformer Building Blocks
//!
//! Transformer encoder and decoder layers, blocks, and stacks.

use axonml_autograd::Variable;
use axonml_nn::{Module, Linear, Dropout, Parameter};
use axonml_tensor::Tensor;
use axonml_tensor::creation::{zeros, ones};

use crate::attention::{MultiHeadSelfAttention, CausalSelfAttention};

/// Layer normalization.
#[derive(Debug)]
pub struct LayerNorm {
    /// Scale parameter
    pub weight: Parameter,
    /// Bias parameter
    pub bias: Parameter,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Normalized dimension
    pub dim: usize,
}

impl LayerNorm {
    /// Creates a new layer normalization.
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: Parameter::new(ones::<f32>(&[dim]), true),
            bias: Parameter::new(zeros::<f32>(&[dim]), true),
            eps,
            dim,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Variable {
        // Normalize over last dimension
        let mean = input.mean_dim(-1, true);
        let variance = input.var_dim(-1, true);

        let x_normalized = input.sub(&mean).div(&variance.add_scalar(self.eps).sqrt());

        // Scale and shift
        let weight_var = Variable::from_tensor_with_grad(self.weight.data().clone(), self.weight.requires_grad());
        let bias_var = Variable::from_tensor_with_grad(self.bias.data().clone(), self.bias.requires_grad());

        x_normalized.mul(&weight_var).add(&bias_var)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Feed-forward network (MLP) used in transformers.
#[derive(Debug)]
pub struct FeedForward {
    /// First linear layer
    pub fc1: Linear,
    /// Second linear layer
    pub fc2: Linear,
    /// Dropout
    pub dropout: Dropout,
    /// Activation function type
    pub activation: String,
}

impl FeedForward {
    /// Creates a new feed-forward network.
    pub fn new(hidden_size: usize, intermediate_size: usize, dropout: f32, activation: &str) -> Self {
        Self {
            fc1: Linear::new(hidden_size, intermediate_size),
            fc2: Linear::new(intermediate_size, hidden_size),
            dropout: Dropout::new(dropout),
            activation: activation.to_string(),
        }
    }

    /// Applies the activation function.
    fn activate(&self, x: &Variable) -> Variable {
        match self.activation.as_str() {
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            "silu" | "swish" => x.silu(),
            "tanh" => x.tanh(),
            _ => x.gelu(), // Default to GELU
        }
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Variable) -> Variable {
        let x = self.fc1.forward(input);
        let x = self.activate(&x);
        let x = self.dropout.forward(&x);
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }
}

/// Transformer encoder block (BERT-style).
#[derive(Debug)]
pub struct TransformerEncoderBlock {
    /// Self-attention layer
    pub attention: MultiHeadSelfAttention,
    /// First layer norm (pre-attention or post-attention)
    pub ln1: LayerNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Second layer norm (pre-FFN or post-FFN)
    pub ln2: LayerNorm,
    /// Residual dropout
    pub dropout: Dropout,
    /// Whether to use pre-norm (like GPT) or post-norm (like original BERT)
    pub pre_norm: bool,
}

impl TransformerEncoderBlock {
    /// Creates a new transformer encoder block.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        dropout: f32,
        layer_norm_eps: f32,
        activation: &str,
        pre_norm: bool,
    ) -> Self {
        Self {
            attention: MultiHeadSelfAttention::new(hidden_size, num_heads, dropout),
            ln1: LayerNorm::new(hidden_size, layer_norm_eps),
            ffn: FeedForward::new(hidden_size, intermediate_size, dropout, activation),
            ln2: LayerNorm::new(hidden_size, layer_norm_eps),
            dropout: Dropout::new(dropout),
            pre_norm,
        }
    }

    /// Forward pass with optional attention mask.
    pub fn forward_with_mask(
        &self,
        hidden_states: &Variable,
        attention_mask: Option<&Tensor<f32>>,
    ) -> Variable {
        if self.pre_norm {
            // Pre-norm: LN -> Attention -> Residual, LN -> FFN -> Residual
            let residual = hidden_states.clone();
            let x = self.ln1.forward(hidden_states);
            let x = self.attention.forward_with_mask(&x, attention_mask);
            let x = self.dropout.forward(&x);
            let x = x.add(&residual);

            let residual = x.clone();
            let x = self.ln2.forward(&x);
            let x = self.ffn.forward(&x);
            let x = self.dropout.forward(&x);
            x.add(&residual)
        } else {
            // Post-norm: Attention -> Residual -> LN, FFN -> Residual -> LN
            let residual = hidden_states.clone();
            let x = self.attention.forward_with_mask(hidden_states, attention_mask);
            let x = self.dropout.forward(&x);
            let x = self.ln1.forward(&x.add(&residual));

            let residual = x.clone();
            let x = self.ffn.forward(&x);
            let x = self.dropout.forward(&x);
            self.ln2.forward(&x.add(&residual))
        }
    }
}

impl Module for TransformerEncoderBlock {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.ln1.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.ln2.parameters());
        params
    }

    fn train(&mut self) {
        self.attention.train();
        self.ffn.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.ffn.eval();
        self.dropout.eval();
    }
}

/// Transformer decoder block (GPT-style with causal attention).
#[derive(Debug)]
pub struct TransformerDecoderBlock {
    /// Causal self-attention
    pub attention: CausalSelfAttention,
    /// First layer norm
    pub ln1: LayerNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Second layer norm
    pub ln2: LayerNorm,
}

impl TransformerDecoderBlock {
    /// Creates a new transformer decoder block.
    pub fn new(
        n_embd: usize,
        n_head: usize,
        max_seq_len: usize,
        dropout: f32,
        layer_norm_eps: f32,
        activation: &str,
    ) -> Self {
        Self {
            attention: CausalSelfAttention::new(n_embd, n_head, max_seq_len, dropout),
            ln1: LayerNorm::new(n_embd, layer_norm_eps),
            ffn: FeedForward::new(n_embd, 4 * n_embd, dropout, activation),
            ln2: LayerNorm::new(n_embd, layer_norm_eps),
        }
    }
}

impl Module for TransformerDecoderBlock {
    fn forward(&self, input: &Variable) -> Variable {
        // GPT-2 style: Pre-norm with residual connections
        let x = input.clone();

        // Attention block
        let residual = x.clone();
        let x = self.ln1.forward(&x);
        let x = self.attention.forward(&x);
        let x = x.add(&residual);

        // FFN block
        let residual = x.clone();
        let x = self.ln2.forward(&x);
        let x = self.ffn.forward(&x);
        x.add(&residual)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.ln1.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.ln2.parameters());
        params
    }

    fn train(&mut self) {
        self.attention.train();
        self.ffn.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.ffn.eval();
    }
}

/// Stack of transformer encoder blocks.
#[derive(Debug)]
pub struct TransformerEncoder {
    /// Encoder layers
    pub layers: Vec<TransformerEncoderBlock>,
}

impl TransformerEncoder {
    /// Creates a new transformer encoder stack.
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        dropout: f32,
        layer_norm_eps: f32,
        activation: &str,
        pre_norm: bool,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| {
                TransformerEncoderBlock::new(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    dropout,
                    layer_norm_eps,
                    activation,
                    pre_norm,
                )
            })
            .collect();

        Self { layers }
    }

    /// Forward pass with optional attention mask.
    pub fn forward_with_mask(
        &self,
        hidden_states: &Variable,
        attention_mask: Option<&Tensor<f32>>,
    ) -> Variable {
        let mut output = hidden_states.clone();
        for layer in &self.layers {
            output = layer.forward_with_mask(&output, attention_mask);
        }
        output
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

/// Stack of transformer decoder blocks.
#[derive(Debug)]
pub struct TransformerDecoder {
    /// Decoder layers
    pub layers: Vec<TransformerDecoderBlock>,
    /// Final layer norm
    pub ln_f: LayerNorm,
}

impl TransformerDecoder {
    /// Creates a new transformer decoder stack.
    pub fn new(
        num_layers: usize,
        n_embd: usize,
        n_head: usize,
        max_seq_len: usize,
        dropout: f32,
        layer_norm_eps: f32,
        activation: &str,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| {
                TransformerDecoderBlock::new(
                    n_embd,
                    n_head,
                    max_seq_len,
                    dropout,
                    layer_norm_eps,
                    activation,
                )
            })
            .collect();

        Self {
            layers,
            ln_f: LayerNorm::new(n_embd, layer_norm_eps),
        }
    }
}

impl Module for TransformerDecoder {
    fn forward(&self, input: &Variable) -> Variable {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        self.ln_f.forward(&output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params: Vec<Parameter> = self.layers.iter().flat_map(|l| l.parameters()).collect();
        params.extend(self.ln_f.parameters());
        params
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

/// Generic transformer block type selection.
#[derive(Debug)]
pub enum TransformerBlock {
    /// Encoder block (bidirectional attention)
    Encoder(TransformerEncoderBlock),
    /// Decoder block (causal attention)
    Decoder(TransformerDecoderBlock),
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Variable) -> Variable {
        match self {
            TransformerBlock::Encoder(block) => block.forward(input),
            TransformerBlock::Decoder(block) => block.forward(input),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        match self {
            TransformerBlock::Encoder(block) => block.parameters(),
            TransformerBlock::Decoder(block) => block.parameters(),
        }
    }

    fn train(&mut self) {
        match self {
            TransformerBlock::Encoder(block) => block.train(),
            TransformerBlock::Decoder(block) => block.train(),
        }
    }

    fn eval(&mut self) {
        match self {
            TransformerBlock::Encoder(block) => block.eval(),
            TransformerBlock::Decoder(block) => block.eval(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64, 1e-5);
        let input = Variable::new(Tensor::randn(&[2, 8, 64]), false);
        let output = ln.forward(&input);

        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_feed_forward() {
        let ffn = FeedForward::new(64, 256, 0.0, "gelu");
        let input = Variable::new(Tensor::randn(&[2, 8, 64]), false);
        let output = ffn.forward(&input);

        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_encoder_block() {
        let block = TransformerEncoderBlock::new(64, 4, 256, 0.0, 1e-5, "gelu", false);
        let input = Variable::new(Tensor::randn(&[2, 8, 64]), false);
        let output = block.forward(&input);

        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_decoder_block() {
        let block = TransformerDecoderBlock::new(64, 4, 128, 0.0, 1e-5, "gelu");
        let input = Variable::new(Tensor::randn(&[2, 8, 64]), false);
        let output = block.forward(&input);

        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_transformer_encoder() {
        let encoder = TransformerEncoder::new(2, 64, 4, 256, 0.0, 1e-5, "gelu", false);
        let input = Variable::new(Tensor::randn(&[2, 8, 64]), false);
        let output = encoder.forward(&input);

        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_transformer_decoder() {
        let decoder = TransformerDecoder::new(2, 64, 4, 128, 0.0, 1e-5, "gelu");
        let input = Variable::new(Tensor::randn(&[2, 8, 64]), false);
        let output = decoder.forward(&input);

        assert_eq!(output.data().shape(), &[2, 8, 64]);
    }
}
