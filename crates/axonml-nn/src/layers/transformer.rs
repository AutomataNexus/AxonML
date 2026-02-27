//! Transformer Architecture - Encoder-Decoder Transformer
//!
//! Implements the full Transformer architecture following PyTorch's API:
//! - `TransformerEncoderLayer` - Single encoder layer (self-attn + FFN)
//! - `TransformerDecoderLayer` - Single decoder layer (self-attn + cross-attn + FFN)
//! - `TransformerEncoder` - Stack of encoder layers
//! - `TransformerDecoder` - Stack of decoder layers
//! - `Seq2SeqTransformer` - Full encoder-decoder Transformer
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::layers::attention::MultiHeadAttention;
use crate::layers::linear::Linear;
use crate::layers::norm::LayerNorm;
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// TransformerEncoderLayer
// =============================================================================

/// A single Transformer encoder layer.
///
/// Consists of multi-head self-attention followed by a position-wise
/// feedforward network, each with residual connections and layer normalization.
///
/// # Shape
/// - Input: (N, S, E) if batch_first (default)
/// - Output: (N, S, E)
///
/// where N=batch, S=source seq len, E=d_model.
pub struct TransformerEncoderLayer {
    /// Self-attention.
    self_attn: MultiHeadAttention,
    /// Feedforward network — first linear.
    linear1: Linear,
    /// Feedforward network — second linear.
    linear2: Linear,
    /// Layer norm after self-attention.
    norm1: LayerNorm,
    /// Layer norm after feedforward.
    norm2: LayerNorm,
    /// Model dimension.
    d_model: usize,
}

impl TransformerEncoderLayer {
    /// Creates a new TransformerEncoderLayer.
    ///
    /// # Arguments
    /// * `d_model` - Embedding dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Hidden dimension of feedforward network (default 2048)
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::single(d_model),
            norm2: LayerNorm::single(d_model),
            d_model,
        }
    }

    /// Forward pass with optional source mask.
    ///
    /// # Arguments
    /// * `src` - Source sequence (N, S, E)
    /// * `src_mask` - Optional attention mask
    pub fn forward_with_mask(&self, src: &Variable, src_mask: Option<&Variable>) -> Variable {
        // Self-attention sublayer: x + self_attn(x)
        let attn_out = self.self_attn.attention(src, src, src, src_mask);
        let x = src.add_var(&attn_out);
        let x = self.norm1.forward(&x);

        // Feedforward sublayer: x + FFN(x)
        let ff_out = self.linear1.forward(&x).relu();
        let ff_out = self.linear2.forward(&ff_out);
        let x = x.add_var(&ff_out);
        self.norm2.forward(&x)
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.self_attn.named_parameters() {
            params.insert(format!("self_attn.{name}"), param);
        }
        for (name, param) in self.linear1.named_parameters() {
            params.insert(format!("linear1.{name}"), param);
        }
        for (name, param) in self.linear2.named_parameters() {
            params.insert(format!("linear2.{name}"), param);
        }
        for (name, param) in self.norm1.named_parameters() {
            params.insert(format!("norm1.{name}"), param);
        }
        for (name, param) in self.norm2.named_parameters() {
            params.insert(format!("norm2.{name}"), param);
        }
        params
    }

    fn name(&self) -> &'static str {
        "TransformerEncoderLayer"
    }
}

// =============================================================================
// TransformerDecoderLayer
// =============================================================================

/// A single Transformer decoder layer.
///
/// Consists of:
/// 1. Masked multi-head self-attention (causal)
/// 2. Multi-head cross-attention over encoder output
/// 3. Position-wise feedforward network
///
/// Each sublayer has residual connections and layer normalization.
///
/// # Shape
/// - Target: (N, T, E)
/// - Memory: (N, S, E)
/// - Output: (N, T, E)
pub struct TransformerDecoderLayer {
    /// Masked self-attention (causal).
    self_attn: MultiHeadAttention,
    /// Cross-attention over encoder output.
    cross_attn: MultiHeadAttention,
    /// Feedforward network — first linear.
    linear1: Linear,
    /// Feedforward network — second linear.
    linear2: Linear,
    /// Layer norm after self-attention.
    norm1: LayerNorm,
    /// Layer norm after cross-attention.
    norm2: LayerNorm,
    /// Layer norm after feedforward.
    norm3: LayerNorm,
    /// Model dimension.
    d_model: usize,
}

impl TransformerDecoderLayer {
    /// Creates a new TransformerDecoderLayer.
    ///
    /// # Arguments
    /// * `d_model` - Embedding dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Hidden dimension of feedforward network
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            cross_attn: MultiHeadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::single(d_model),
            norm2: LayerNorm::single(d_model),
            norm3: LayerNorm::single(d_model),
            d_model,
        }
    }

    /// Forward pass with encoder memory and optional masks.
    ///
    /// # Arguments
    /// * `tgt` - Target sequence (N, T, E)
    /// * `memory` - Encoder output (N, S, E)
    /// * `tgt_mask` - Optional causal mask for self-attention
    /// * `memory_mask` - Optional mask for cross-attention
    pub fn forward_with_memory(
        &self,
        tgt: &Variable,
        memory: &Variable,
        tgt_mask: Option<&Variable>,
        memory_mask: Option<&Variable>,
    ) -> Variable {
        // 1. Masked self-attention sublayer
        let self_attn_out = self.self_attn.attention(tgt, tgt, tgt, tgt_mask);
        let x = tgt.add_var(&self_attn_out);
        let x = self.norm1.forward(&x);

        // 2. Cross-attention sublayer (query=decoder, key/value=encoder)
        let cross_attn_out = self.cross_attn.attention(&x, memory, memory, memory_mask);
        let x = x.add_var(&cross_attn_out);
        let x = self.norm2.forward(&x);

        // 3. Feedforward sublayer
        let ff_out = self.linear1.forward(&x).relu();
        let ff_out = self.linear2.forward(&ff_out);
        let x = x.add_var(&ff_out);
        self.norm3.forward(&x)
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, input: &Variable) -> Variable {
        // Without memory, can only do self-attention pass.
        // Use forward_with_memory() for full decoder behavior.
        let self_attn_out = self.self_attn.attention(input, input, input, None);
        let x = input.add_var(&self_attn_out);
        let x = self.norm1.forward(&x);

        // Skip cross-attention (no memory), go straight to FFN
        let x_after_norm2 = self.norm2.forward(&x);
        let ff_out = self.linear1.forward(&x_after_norm2).relu();
        let ff_out = self.linear2.forward(&ff_out);
        let x = x_after_norm2.add_var(&ff_out);
        self.norm3.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.self_attn.named_parameters() {
            params.insert(format!("self_attn.{name}"), param);
        }
        for (name, param) in self.cross_attn.named_parameters() {
            params.insert(format!("cross_attn.{name}"), param);
        }
        for (name, param) in self.linear1.named_parameters() {
            params.insert(format!("linear1.{name}"), param);
        }
        for (name, param) in self.linear2.named_parameters() {
            params.insert(format!("linear2.{name}"), param);
        }
        for (name, param) in self.norm1.named_parameters() {
            params.insert(format!("norm1.{name}"), param);
        }
        for (name, param) in self.norm2.named_parameters() {
            params.insert(format!("norm2.{name}"), param);
        }
        for (name, param) in self.norm3.named_parameters() {
            params.insert(format!("norm3.{name}"), param);
        }
        params
    }

    fn name(&self) -> &'static str {
        "TransformerDecoderLayer"
    }
}

// =============================================================================
// TransformerEncoder
// =============================================================================

/// Stack of N TransformerEncoderLayers.
///
/// # Shape
/// - Input: (N, S, E)
/// - Output: (N, S, E)
pub struct TransformerEncoder {
    /// Encoder layers.
    layers: Vec<TransformerEncoderLayer>,
    /// Optional final layer norm.
    norm: Option<LayerNorm>,
}

impl TransformerEncoder {
    /// Creates a TransformerEncoder with the given number of layers.
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, nhead, dim_feedforward))
            .collect();

        Self {
            layers,
            norm: Some(LayerNorm::single(d_model)),
        }
    }

    /// Creates a TransformerEncoder without final layer norm.
    pub fn without_norm(d_model: usize, nhead: usize, dim_feedforward: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, nhead, dim_feedforward))
            .collect();

        Self { layers, norm: None }
    }

    /// Forward pass with optional source mask.
    pub fn forward_with_mask(&self, src: &Variable, src_mask: Option<&Variable>) -> Variable {
        let mut x = src.clone();
        for layer in &self.layers {
            x = layer.forward_with_mask(&x, src_mask);
        }
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x);
        }
        x
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params: Vec<Parameter> = self.layers.iter()
            .flat_map(|l| l.parameters())
            .collect();
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layers.{i}.{name}"), param);
            }
        }
        if let Some(ref norm) = self.norm {
            for (name, param) in norm.named_parameters() {
                params.insert(format!("norm.{name}"), param);
            }
        }
        params
    }

    fn name(&self) -> &'static str {
        "TransformerEncoder"
    }
}

// =============================================================================
// TransformerDecoder
// =============================================================================

/// Stack of N TransformerDecoderLayers.
///
/// # Shape
/// - Target: (N, T, E)
/// - Memory: (N, S, E)
/// - Output: (N, T, E)
pub struct TransformerDecoder {
    /// Decoder layers.
    layers: Vec<TransformerDecoderLayer>,
    /// Optional final layer norm.
    norm: Option<LayerNorm>,
}

impl TransformerDecoder {
    /// Creates a TransformerDecoder with the given number of layers.
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerDecoderLayer::new(d_model, nhead, dim_feedforward))
            .collect();

        Self {
            layers,
            norm: Some(LayerNorm::single(d_model)),
        }
    }

    /// Creates a TransformerDecoder without final layer norm.
    pub fn without_norm(d_model: usize, nhead: usize, dim_feedforward: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerDecoderLayer::new(d_model, nhead, dim_feedforward))
            .collect();

        Self { layers, norm: None }
    }

    /// Forward pass with encoder memory and optional masks.
    pub fn forward_with_memory(
        &self,
        tgt: &Variable,
        memory: &Variable,
        tgt_mask: Option<&Variable>,
        memory_mask: Option<&Variable>,
    ) -> Variable {
        let mut x = tgt.clone();
        for layer in &self.layers {
            x = layer.forward_with_memory(&x, memory, tgt_mask, memory_mask);
        }
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x);
        }
        x
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Module for TransformerDecoder {
    fn forward(&self, input: &Variable) -> Variable {
        // Without memory, runs self-attention only (for pretraining/testing)
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        if let Some(ref norm) = self.norm {
            x = norm.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params: Vec<Parameter> = self.layers.iter()
            .flat_map(|l| l.parameters())
            .collect();
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layers.{i}.{name}"), param);
            }
        }
        if let Some(ref norm) = self.norm {
            for (name, param) in norm.named_parameters() {
                params.insert(format!("norm.{name}"), param);
            }
        }
        params
    }

    fn name(&self) -> &'static str {
        "TransformerDecoder"
    }
}

// =============================================================================
// Seq2SeqTransformer
// =============================================================================

/// Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
///
/// Combines a TransformerEncoder and TransformerDecoder into a single module.
/// Follows PyTorch's `nn.Transformer` API.
///
/// # Architecture
/// ```text
/// Source → [Encoder] → Memory
///                         ↓
/// Target → [Decoder] → Output
/// ```
///
/// # Shape
/// - Source: (N, S, E)
/// - Target: (N, T, E)
/// - Output: (N, T, E)
pub struct Seq2SeqTransformer {
    /// Encoder stack.
    encoder: TransformerEncoder,
    /// Decoder stack.
    decoder: TransformerDecoder,
    /// Model dimension.
    d_model: usize,
    /// Number of attention heads.
    nhead: usize,
}

impl Seq2SeqTransformer {
    /// Creates a new Seq2SeqTransformer.
    ///
    /// # Arguments
    /// * `d_model` - Embedding/model dimension
    /// * `nhead` - Number of attention heads
    /// * `num_encoder_layers` - Number of encoder layers
    /// * `num_decoder_layers` - Number of decoder layers
    /// * `dim_feedforward` - Hidden dimension of feedforward networks
    pub fn new(
        d_model: usize,
        nhead: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dim_feedforward: usize,
    ) -> Self {
        Self {
            encoder: TransformerEncoder::new(d_model, nhead, dim_feedforward, num_encoder_layers),
            decoder: TransformerDecoder::new(d_model, nhead, dim_feedforward, num_decoder_layers),
            d_model,
            nhead,
        }
    }

    /// Creates a Seq2SeqTransformer with default settings (6 layers, 2048 FFN).
    pub fn default_config(d_model: usize, nhead: usize) -> Self {
        Self::new(d_model, nhead, 6, 6, 2048)
    }

    /// Full forward pass: encode source, then decode target conditioned on encoder output.
    ///
    /// # Arguments
    /// * `src` - Source sequence (N, S, E)
    /// * `tgt` - Target sequence (N, T, E)
    /// * `src_mask` - Optional mask for encoder self-attention
    /// * `tgt_mask` - Optional causal mask for decoder self-attention
    /// * `memory_mask` - Optional mask for decoder cross-attention
    pub fn forward_seq2seq(
        &self,
        src: &Variable,
        tgt: &Variable,
        src_mask: Option<&Variable>,
        tgt_mask: Option<&Variable>,
        memory_mask: Option<&Variable>,
    ) -> Variable {
        let memory = self.encoder.forward_with_mask(src, src_mask);
        self.decoder.forward_with_memory(tgt, &memory, tgt_mask, memory_mask)
    }

    /// Encode source sequence only (useful for inference).
    pub fn encode(&self, src: &Variable, src_mask: Option<&Variable>) -> Variable {
        self.encoder.forward_with_mask(src, src_mask)
    }

    /// Decode target given pre-computed encoder memory (useful for inference).
    pub fn decode(
        &self,
        tgt: &Variable,
        memory: &Variable,
        tgt_mask: Option<&Variable>,
        memory_mask: Option<&Variable>,
    ) -> Variable {
        self.decoder.forward_with_memory(tgt, memory, tgt_mask, memory_mask)
    }

    /// Generates a causal (upper-triangular) mask for autoregressive decoding.
    ///
    /// Returns a mask of shape (seq_len, seq_len) where future positions are 0.0
    /// and valid positions are 1.0.
    pub fn generate_square_subsequent_mask(seq_len: usize) -> Variable {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        Variable::new(
            Tensor::from_vec(mask_data, &[seq_len, seq_len]).unwrap(),
            false,
        )
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Returns the number of attention heads.
    pub fn nhead(&self) -> usize {
        self.nhead
    }

    /// Returns a reference to the encoder.
    pub fn encoder(&self) -> &TransformerEncoder {
        &self.encoder
    }

    /// Returns a reference to the decoder.
    pub fn decoder(&self) -> &TransformerDecoder {
        &self.decoder
    }
}

impl Module for Seq2SeqTransformer {
    fn forward(&self, input: &Variable) -> Variable {
        // Single-input forward: encode only (use forward_seq2seq for full pipeline)
        self.encoder.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.encoder.named_parameters() {
            params.insert(format!("encoder.{name}"), param);
        }
        for (name, param) in self.decoder.named_parameters() {
            params.insert(format!("decoder.{name}"), param);
        }
        params
    }

    fn name(&self) -> &'static str {
        "Seq2SeqTransformer"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_layer_creation() {
        let layer = TransformerEncoderLayer::new(64, 4, 256);
        assert_eq!(layer.d_model(), 64);
    }

    #[test]
    fn test_encoder_layer_forward() {
        let layer = TransformerEncoderLayer::new(64, 4, 256);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![2, 10, 64]);
    }

    #[test]
    fn test_decoder_layer_with_memory() {
        let layer = TransformerDecoderLayer::new(64, 4, 256);
        let tgt = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        let memory = Variable::new(
            Tensor::from_vec(vec![0.2; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = layer.forward_with_memory(&tgt, &memory, None, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_encoder_stack() {
        let encoder = TransformerEncoder::new(64, 4, 256, 3);
        assert_eq!(encoder.num_layers(), 3);

        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 8 * 64], &[2, 8, 64]).unwrap(),
            false,
        );
        let output = encoder.forward(&input);
        assert_eq!(output.shape(), vec![2, 8, 64]);
    }

    #[test]
    fn test_decoder_stack() {
        let decoder = TransformerDecoder::new(64, 4, 256, 3);
        assert_eq!(decoder.num_layers(), 3);

        let tgt = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        let memory = Variable::new(
            Tensor::from_vec(vec![0.2; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = decoder.forward_with_memory(&tgt, &memory, None, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_seq2seq_transformer() {
        let transformer = Seq2SeqTransformer::new(64, 4, 2, 2, 256);
        assert_eq!(transformer.d_model(), 64);
        assert_eq!(transformer.nhead(), 4);

        let src = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let tgt = Variable::new(
            Tensor::from_vec(vec![0.2; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        let output = transformer.forward_seq2seq(&src, &tgt, None, None, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_seq2seq_encode_decode_separate() {
        let transformer = Seq2SeqTransformer::new(64, 4, 2, 2, 256);

        let src = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let tgt = Variable::new(
            Tensor::from_vec(vec![0.2; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );

        // Encode once, decode multiple times (autoregressive inference)
        let memory = transformer.encode(&src, None);
        assert_eq!(memory.shape(), vec![2, 10, 64]);

        let output = transformer.decode(&tgt, &memory, None, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = Seq2SeqTransformer::generate_square_subsequent_mask(4);
        let mask_data = mask.data().to_vec();
        // Row 0: [1, 0, 0, 0]
        // Row 1: [1, 1, 0, 0]
        // Row 2: [1, 1, 1, 0]
        // Row 3: [1, 1, 1, 1]
        assert_eq!(mask_data[0], 1.0); // (0,0) = visible
        assert_eq!(mask_data[1], 0.0); // (0,1) = masked
        assert_eq!(mask_data[4], 1.0); // (1,0) = visible
        assert_eq!(mask_data[5], 1.0); // (1,1) = visible
        assert_eq!(mask_data[6], 0.0); // (1,2) = masked
        assert_eq!(mask_data[15], 1.0); // (3,3) = visible
    }

    #[test]
    fn test_default_config() {
        let transformer = Seq2SeqTransformer::default_config(512, 8);
        assert_eq!(transformer.encoder().num_layers(), 6);
        assert_eq!(transformer.decoder().num_layers(), 6);
    }

    #[test]
    fn test_parameter_count() {
        let layer = TransformerEncoderLayer::new(64, 4, 256);
        let params = layer.parameters();
        // self_attn: 4 projections × (weight + bias) = 8
        // linear1: weight + bias = 2
        // linear2: weight + bias = 2
        // norm1: weight + bias = 2
        // norm2: weight + bias = 2
        assert_eq!(params.len(), 16);
    }

    #[test]
    fn test_decoder_parameter_count() {
        let layer = TransformerDecoderLayer::new(64, 4, 256);
        let params = layer.parameters();
        // self_attn: 8, cross_attn: 8, linear1: 2, linear2: 2, norm1: 2, norm2: 2, norm3: 2
        assert_eq!(params.len(), 26);
    }

    #[test]
    fn test_named_parameters_hierarchy() {
        let transformer = Seq2SeqTransformer::new(64, 4, 1, 1, 256);
        let named = transformer.named_parameters();
        // Verify hierarchical naming
        assert!(named.contains_key("encoder.layers.0.self_attn.q_proj.weight"));
        assert!(named.contains_key("decoder.layers.0.cross_attn.q_proj.weight"));
        assert!(named.contains_key("encoder.norm.weight"));
        assert!(named.contains_key("decoder.norm.weight"));
    }

    #[test]
    fn test_seq2seq_with_causal_mask() {
        let transformer = Seq2SeqTransformer::new(64, 4, 2, 2, 256);
        let src = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let tgt = Variable::new(
            Tensor::from_vec(vec![0.2; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        let tgt_mask = Seq2SeqTransformer::generate_square_subsequent_mask(5);
        let output = transformer.forward_seq2seq(&src, &tgt, None, Some(&tgt_mask), None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }
}
