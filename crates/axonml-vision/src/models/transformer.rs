//! Transformer - Attention-based Neural Networks
//!
//! Implementation of Transformer architectures for various tasks.
//!
//! # Models
//!
//! - **`TransformerEncoder`**: Stack of encoder layers
//! - **`TransformerDecoder`**: Stack of decoder layers
//! - **Transformer**: Full encoder-decoder Transformer
//! - **`VisionTransformer` (`ViT`)**: Transformer for image classification
//!
//! # Reference
//!
//! "Attention Is All You Need" (Vaswani et al., 2017)
//! <https://arxiv.org/abs/1706.03762>
//!
//! "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
//! <https://arxiv.org/abs/2010.11929>

use axonml_autograd::Variable;
use axonml_nn::{Dropout, LayerNorm, Linear, Module, MultiHeadAttention, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// Positional Encoding
// =============================================================================

/// Positional encoding using sinusoidal functions.
pub struct PositionalEncoding {
    encoding: Tensor<f32>,
    max_len: usize,
    d_model: usize,
}

impl PositionalEncoding {
    /// Create positional encoding.
    #[must_use] pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut pe = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let div_term = (-(i as f32 / d_model as f32) * (10000.0f32).ln()).exp();
                if i % 2 == 0 {
                    pe[pos * d_model + i] = (pos as f32 * div_term).sin();
                } else {
                    pe[pos * d_model + i] = (pos as f32 * div_term).cos();
                }
            }
        }

        Self {
            encoding: Tensor::from_vec(pe, &[max_len, d_model]).unwrap(),
            max_len,
            d_model,
        }
    }

    /// Add positional encoding to input.
    #[must_use] pub fn forward(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        let seq_len = shape[1];
        let x_data = x.data().to_vec();
        let pe_data = self.encoding.to_vec();

        // Broadcasting PE across batch
        let batch_size = shape[0];
        let mut result = x_data.clone();

        for b in 0..batch_size {
            for s in 0..seq_len.min(self.max_len) {
                for d in 0..self.d_model {
                    let idx = b * seq_len * self.d_model + s * self.d_model + d;
                    result[idx] += pe_data[s * self.d_model + d];
                }
            }
        }

        Variable::new(Tensor::from_vec(result, &shape).unwrap(), x.requires_grad())
    }
}

// =============================================================================
// Transformer Encoder Layer
// =============================================================================

/// A single Transformer encoder layer.
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    ff_linear1: Linear,
    ff_linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    d_model: usize,
}

impl TransformerEncoderLayer {
    /// Create encoder layer.
    #[must_use] pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f32) -> Self {
        Self {
            self_attn: MultiHeadAttention::with_options(d_model, nhead, dropout, true),
            ff_linear1: Linear::new(d_model, dim_feedforward),
            ff_linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(vec![d_model]),
            norm2: LayerNorm::new(vec![d_model]),
            dropout: Dropout::new(dropout),
            d_model,
        }
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Forward with optional attention mask.
    pub fn forward_with_mask(&self, src: &Variable, src_mask: Option<&Variable>) -> Variable {
        // Self-attention with residual
        let attn_out = self.self_attn.attention(src, src, src, src_mask);
        let attn_out = self.dropout.forward(&attn_out);
        let src = src.add_var(&attn_out);
        let src = self.norm1.forward(&src);

        // Feed-forward with residual
        let ff_out = self.ff_linear1.forward(&src);
        let ff_out = ff_out.relu();
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = self.ff_linear2.forward(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let src = src.add_var(&ff_out);

        self.norm2.forward(&src)
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.dropout.is_training()
    }
}

// =============================================================================
// Transformer Encoder
// =============================================================================

/// Stack of Transformer encoder layers.
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    norm: Option<LayerNorm>,
}

impl TransformerEncoder {
    /// Create encoder with specified layers.
    #[must_use] pub fn new(
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        dim_feedforward: usize,
        dropout: f32,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, nhead, dim_feedforward, dropout))
            .collect();

        Self {
            layers,
            norm: Some(LayerNorm::new(vec![d_model])),
        }
    }

    /// Forward with optional mask.
    #[must_use] pub fn forward_with_mask(&self, src: &Variable, mask: Option<&Variable>) -> Variable {
        let mut output = src.clone();
        for layer in &self.layers {
            output = layer.forward_with_mask(&output, mask);
        }
        if let Some(norm) = &self.norm {
            output = norm.forward(&output);
        }
        output
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Variable) -> Variable {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(norm) = &self.norm {
            params.extend(norm.parameters());
        }
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

    fn is_training(&self) -> bool {
        self.layers.first().map_or(true, axonml_nn::Module::is_training)
    }
}

// =============================================================================
// Transformer Decoder Layer
// =============================================================================

/// A single Transformer decoder layer.
pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ff_linear1: Linear,
    ff_linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
}

impl TransformerDecoderLayer {
    /// Create decoder layer.
    #[must_use] pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f32) -> Self {
        Self {
            self_attn: MultiHeadAttention::with_options(d_model, nhead, dropout, true),
            cross_attn: MultiHeadAttention::with_options(d_model, nhead, dropout, true),
            ff_linear1: Linear::new(d_model, dim_feedforward),
            ff_linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(vec![d_model]),
            norm2: LayerNorm::new(vec![d_model]),
            norm3: LayerNorm::new(vec![d_model]),
            dropout: Dropout::new(dropout),
        }
    }

    /// Forward with memory and masks.
    pub fn forward_with_memory(
        &self,
        tgt: &Variable,
        memory: &Variable,
        tgt_mask: Option<&Variable>,
        memory_mask: Option<&Variable>,
    ) -> Variable {
        // Self-attention with residual
        let attn_out = self.self_attn.attention(tgt, tgt, tgt, tgt_mask);
        let attn_out = self.dropout.forward(&attn_out);
        let tgt = tgt.add_var(&attn_out);
        let tgt = self.norm1.forward(&tgt);

        // Cross-attention with residual
        let cross_out = self.cross_attn.attention(&tgt, memory, memory, memory_mask);
        let cross_out = self.dropout.forward(&cross_out);
        let tgt = tgt.add_var(&cross_out);
        let tgt = self.norm2.forward(&tgt);

        // Feed-forward with residual
        let ff_out = self.ff_linear1.forward(&tgt);
        let ff_out = ff_out.relu();
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = self.ff_linear2.forward(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let tgt = tgt.add_var(&ff_out);

        self.norm3.forward(&tgt)
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, input: &Variable) -> Variable {
        // For standard forward, use self-attention only
        self.self_attn.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.dropout.is_training()
    }
}

// =============================================================================
// Transformer Decoder
// =============================================================================

/// Stack of Transformer decoder layers.
pub struct TransformerDecoder {
    layers: Vec<TransformerDecoderLayer>,
    norm: Option<LayerNorm>,
}

impl TransformerDecoder {
    /// Create decoder with specified layers.
    #[must_use] pub fn new(
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        dim_feedforward: usize,
        dropout: f32,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerDecoderLayer::new(d_model, nhead, dim_feedforward, dropout))
            .collect();

        Self {
            layers,
            norm: Some(LayerNorm::new(vec![d_model])),
        }
    }

    /// Forward with memory and masks.
    #[must_use] pub fn forward_with_memory(
        &self,
        tgt: &Variable,
        memory: &Variable,
        tgt_mask: Option<&Variable>,
        memory_mask: Option<&Variable>,
    ) -> Variable {
        let mut output = tgt.clone();
        for layer in &self.layers {
            output = layer.forward_with_memory(&output, memory, tgt_mask, memory_mask);
        }
        if let Some(norm) = &self.norm {
            output = norm.forward(&output);
        }
        output
    }
}

impl Module for TransformerDecoder {
    fn forward(&self, input: &Variable) -> Variable {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        if let Some(norm) = &self.norm {
            output = norm.forward(&output);
        }
        output
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(norm) = &self.norm {
            params.extend(norm.parameters());
        }
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

    fn is_training(&self) -> bool {
        self.layers.first().map_or(true, axonml_nn::Module::is_training)
    }
}

// =============================================================================
// Full Transformer
// =============================================================================

/// Full Transformer model (encoder-decoder).
pub struct Transformer {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    d_model: usize,
}

impl Transformer {
    /// Create a Transformer model.
    #[must_use] pub fn new(
        d_model: usize,
        nhead: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dim_feedforward: usize,
        dropout: f32,
    ) -> Self {
        Self {
            encoder: TransformerEncoder::new(
                d_model,
                nhead,
                num_encoder_layers,
                dim_feedforward,
                dropout,
            ),
            decoder: TransformerDecoder::new(
                d_model,
                nhead,
                num_decoder_layers,
                dim_feedforward,
                dropout,
            ),
            d_model,
        }
    }

    /// Returns the model dimension.
    #[must_use] pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Forward pass with source and target.
    #[must_use] pub fn forward_full(
        &self,
        src: &Variable,
        tgt: &Variable,
        src_mask: Option<&Variable>,
        tgt_mask: Option<&Variable>,
        memory_mask: Option<&Variable>,
    ) -> Variable {
        let memory = self.encoder.forward_with_mask(src, src_mask);
        self.decoder
            .forward_with_memory(tgt, &memory, tgt_mask, memory_mask)
    }
}

impl Module for Transformer {
    fn forward(&self, input: &Variable) -> Variable {
        // Encoder-only forward for classification tasks
        self.encoder.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params
    }

    fn train(&mut self) {
        self.encoder.train();
        self.decoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
        self.decoder.eval();
    }

    fn is_training(&self) -> bool {
        self.encoder.is_training()
    }
}

// =============================================================================
// Vision Transformer (ViT)
// =============================================================================

/// Vision Transformer for image classification.
///
/// Converts images into patches and processes them with a Transformer encoder.
pub struct VisionTransformer {
    patch_embedding: Linear,
    pos_encoding: PositionalEncoding,
    encoder: TransformerEncoder,
    mlp_head: Linear,
    cls_token: Parameter,
    patch_size: usize,
    num_patches: usize,
    d_model: usize,
}

impl VisionTransformer {
    /// Create a Vision Transformer.
    ///
    /// # Arguments
    /// * `image_size` - Input image size (assumes square)
    /// * `patch_size` - Size of each patch
    /// * `in_channels` - Number of input channels (3 for RGB)
    /// * `num_classes` - Number of output classes
    /// * `d_model` - Model dimension
    /// * `nhead` - Number of attention heads
    /// * `num_layers` - Number of encoder layers
    /// * `dim_feedforward` - Feed-forward dimension
    /// * `dropout` - Dropout probability
    #[must_use] pub fn new(
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        num_classes: usize,
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        dim_feedforward: usize,
        dropout: f32,
    ) -> Self {
        assert!(
            image_size % patch_size == 0,
            "Image size must be divisible by patch size"
        );

        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let patch_dim = in_channels * patch_size * patch_size;

        // CLS token as learnable parameter
        let cls_data = Tensor::from_vec(vec![0.0f32; d_model], &[1, 1, d_model]).unwrap();
        let cls_token = Parameter::named("cls_token", cls_data, true);

        Self {
            patch_embedding: Linear::new(patch_dim, d_model),
            pos_encoding: PositionalEncoding::new(d_model, num_patches + 1), // +1 for CLS
            encoder: TransformerEncoder::new(d_model, nhead, num_layers, dim_feedforward, dropout),
            mlp_head: Linear::new(d_model, num_classes),
            cls_token,
            patch_size,
            num_patches,
            d_model,
        }
    }

    /// Create ViT-Tiny.
    #[must_use] pub fn vit_tiny(image_size: usize, num_classes: usize) -> Self {
        Self::new(image_size, 16, 3, num_classes, 192, 3, 12, 768, 0.0)
    }

    /// Create ViT-Small.
    #[must_use] pub fn vit_small(image_size: usize, num_classes: usize) -> Self {
        Self::new(image_size, 16, 3, num_classes, 384, 6, 12, 1536, 0.0)
    }

    /// Create ViT-Base.
    #[must_use] pub fn vit_base(image_size: usize, num_classes: usize) -> Self {
        Self::new(image_size, 16, 3, num_classes, 768, 12, 12, 3072, 0.0)
    }

    /// Create ViT-Large.
    #[must_use] pub fn vit_large(image_size: usize, num_classes: usize) -> Self {
        Self::new(image_size, 16, 3, num_classes, 1024, 16, 24, 4096, 0.0)
    }

    /// Extract patches from image.
    fn extract_patches(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        let patch_dim = channels * self.patch_size * self.patch_size;

        let x_data = x.data().to_vec();
        let mut patches = vec![0.0f32; batch_size * self.num_patches * patch_dim];

        for b in 0..batch_size {
            for ph in 0..num_patches_h {
                for pw in 0..num_patches_w {
                    let patch_idx = ph * num_patches_w + pw;
                    for c in 0..channels {
                        for i in 0..self.patch_size {
                            for j in 0..self.patch_size {
                                let img_h = ph * self.patch_size + i;
                                let img_w = pw * self.patch_size + j;
                                let img_idx = b * channels * height * width
                                    + c * height * width
                                    + img_h * width
                                    + img_w;
                                let patch_offset =
                                    c * self.patch_size * self.patch_size + i * self.patch_size + j;
                                let out_idx = b * self.num_patches * patch_dim
                                    + patch_idx * patch_dim
                                    + patch_offset;
                                patches[out_idx] = x_data[img_idx];
                            }
                        }
                    }
                }
            }
        }

        Variable::new(
            Tensor::from_vec(patches, &[batch_size, self.num_patches, patch_dim]).unwrap(),
            x.requires_grad(),
        )
    }
}

impl Module for VisionTransformer {
    fn forward(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        let batch_size = shape[0];

        // Extract patches: [B, C, H, W] -> [B, num_patches, patch_dim]
        let patches = self.extract_patches(x);

        // Embed patches: [B, num_patches, patch_dim] -> [B, num_patches, d_model]
        let patch_emb = self.patch_embedding.forward(&patches);

        // Prepend CLS token
        let cls_data = self.cls_token.data().to_vec();
        let patch_emb_data = patch_emb.data().to_vec();

        let mut tokens = vec![0.0f32; batch_size * (self.num_patches + 1) * self.d_model];

        for b in 0..batch_size {
            // CLS token
            for d in 0..self.d_model {
                tokens[b * (self.num_patches + 1) * self.d_model + d] = cls_data[d];
            }
            // Patch embeddings
            for p in 0..self.num_patches {
                for d in 0..self.d_model {
                    let src_idx = b * self.num_patches * self.d_model + p * self.d_model + d;
                    let dst_idx =
                        b * (self.num_patches + 1) * self.d_model + (p + 1) * self.d_model + d;
                    tokens[dst_idx] = patch_emb_data[src_idx];
                }
            }
        }

        let tokens = Variable::new(
            Tensor::from_vec(tokens, &[batch_size, self.num_patches + 1, self.d_model]).unwrap(),
            x.requires_grad(),
        );

        // Add positional encoding
        let tokens = self.pos_encoding.forward(&tokens);

        // Pass through encoder
        let encoded = self.encoder.forward(&tokens);

        // Extract CLS token output: [B, num_patches+1, d_model] -> [B, d_model]
        let encoded_data = encoded.data().to_vec();
        let mut cls_output = vec![0.0f32; batch_size * self.d_model];
        for b in 0..batch_size {
            for d in 0..self.d_model {
                cls_output[b * self.d_model + d] =
                    encoded_data[b * (self.num_patches + 1) * self.d_model + d];
            }
        }

        let cls_output = Variable::new(
            Tensor::from_vec(cls_output, &[batch_size, self.d_model]).unwrap(),
            x.requires_grad(),
        );

        // Classification head
        self.mlp_head.forward(&cls_output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.push(self.cls_token.clone());
        params.extend(self.patch_embedding.parameters());
        params.extend(self.encoder.parameters());
        params.extend(self.mlp_head.parameters());
        params
    }

    fn train(&mut self) {
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
    }

    fn is_training(&self) -> bool {
        self.encoder.is_training()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create ViT-Base for `ImageNet` (224x224, 1000 classes).
#[must_use] pub fn vit_base() -> VisionTransformer {
    VisionTransformer::vit_base(224, 1000)
}

/// Create ViT-Large for `ImageNet` (224x224, 1000 classes).
#[must_use] pub fn vit_large() -> VisionTransformer {
    VisionTransformer::vit_large(224, 1000)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(64, 100);
        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = pe.forward(&input);
        assert_eq!(output.shape(), vec![2, 10, 64]);
    }

    #[test]
    fn test_encoder_layer() {
        let layer = TransformerEncoderLayer::new(64, 4, 256, 0.1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![2, 10, 64]);
    }

    #[test]
    fn test_transformer_encoder() {
        let encoder = TransformerEncoder::new(64, 4, 2, 256, 0.1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let output = encoder.forward(&input);
        assert_eq!(output.shape(), vec![2, 10, 64]);
    }

    #[test]
    fn test_transformer() {
        let transformer = Transformer::new(64, 4, 2, 2, 256, 0.1);
        let src = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 10 * 64], &[2, 10, 64]).unwrap(),
            false,
        );
        let tgt = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 5 * 64], &[2, 5, 64]).unwrap(),
            false,
        );
        let output = transformer.forward_full(&src, &tgt, None, None, None);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }

    #[test]
    fn test_vit_creation() {
        let vit = VisionTransformer::new(
            32,  // image_size
            8,   // patch_size
            3,   // channels
            10,  // num_classes
            64,  // d_model
            4,   // nhead
            2,   // num_layers
            256, // dim_ff
            0.1, // dropout
        );
        let params = vit.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_vit_forward() {
        let vit = VisionTransformer::new(32, 8, 3, 10, 64, 4, 2, 256, 0.1);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 3 * 32 * 32], &[2, 3, 32, 32]).unwrap(),
            false,
        );
        let output = vit.forward(&input);
        assert_eq!(output.shape(), vec![2, 10]);
    }

    #[test]
    fn test_vit_tiny() {
        let vit = VisionTransformer::vit_tiny(32, 10);
        let params = vit.parameters();
        assert!(!params.is_empty());
    }
}
