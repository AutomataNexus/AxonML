//! Visual Question Answering — Multi-Modal Vision+Language Model
//!
//! VQA model combining image and text understanding via cross-attention fusion.
//! `VisionEncoder` converts images to patch tokens (Conv2d patch embedding +
//! transformer encoder layers + LayerNorm). `TextEncoder` embeds token IDs via
//! `Embedding` and processes through transformer encoder layers. `VQAModel` fuses
//! the two via `CrossAttention` (text queries attend to image features), mean-pools
//! over the sequence dimension, and classifies through a 2-layer MLP head.
//! `forward_vqa()` takes image [N,3,H,W] and question token IDs [N,seq_len],
//! returning [N, num_answers] logits. `small()` factory creates a test-sized model.
//!
//! # File
//! `crates/axonml-vision/src/models/vqa.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::{
    Conv2d, CrossAttention, Embedding, LayerNorm, Linear, Module, MultiHeadAttention, Parameter,
    ReLU,
};

// =============================================================================
// Text Encoder
// =============================================================================

/// Transformer-based text encoder for VQA.
struct TextEncoder {
    embedding: Embedding,
    layers: Vec<TextEncoderLayer>,
    norm: LayerNorm,
    _d_model: usize,
}

struct TextEncoderLayer {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
    norm2: LayerNorm,
    relu: ReLU,
}

impl TextEncoderLayer {
    fn new(d_model: usize, nhead: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            norm1: LayerNorm::single(d_model),
            ffn1: Linear::new(d_model, d_model * 4),
            ffn2: Linear::new(d_model * 4, d_model),
            norm2: LayerNorm::single(d_model),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let attn = self.self_attn.forward(x);
        let x = self.norm1.forward(&x.add_var(&attn));
        let ffn = self
            .ffn2
            .forward(&self.relu.forward(&self.ffn1.forward(&x)));
        self.norm2.forward(&x.add_var(&ffn))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.ffn1.parameters());
        p.extend(self.ffn2.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

impl TextEncoder {
    fn new(vocab_size: usize, d_model: usize, nhead: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TextEncoderLayer::new(d_model, nhead))
            .collect();

        Self {
            embedding: Embedding::new(vocab_size, d_model),
            layers,
            norm: LayerNorm::single(d_model),
            _d_model: d_model,
        }
    }

    fn forward(&self, token_ids: &Variable) -> Variable {
        let mut x = self.embedding.forward(token_ids);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.norm.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.embedding.parameters());
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.norm.parameters());
        p
    }
}

// =============================================================================
// Vision Encoder (lightweight patch-based)
// =============================================================================

/// Patch-based vision encoder for VQA.
struct VisionEncoder {
    patch_embed: Conv2d,
    layers: Vec<VisionEncoderLayer>,
    norm: LayerNorm,
    d_model: usize,
    patch_size: usize,
}

struct VisionEncoderLayer {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
    norm2: LayerNorm,
    relu: ReLU,
}

impl VisionEncoderLayer {
    fn new(d_model: usize, nhead: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            norm1: LayerNorm::single(d_model),
            ffn1: Linear::new(d_model, d_model * 4),
            ffn2: Linear::new(d_model * 4, d_model),
            norm2: LayerNorm::single(d_model),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let attn = self.self_attn.forward(x);
        let x = self.norm1.forward(&x.add_var(&attn));
        let ffn = self
            .ffn2
            .forward(&self.relu.forward(&self.ffn1.forward(&x)));
        self.norm2.forward(&x.add_var(&ffn))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.ffn1.parameters());
        p.extend(self.ffn2.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

impl VisionEncoder {
    fn new(d_model: usize, nhead: usize, num_layers: usize, patch_size: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| VisionEncoderLayer::new(d_model, nhead))
            .collect();

        Self {
            patch_embed: Conv2d::with_options(
                3,
                d_model,
                (patch_size, patch_size),
                (patch_size, patch_size),
                (0, 0),
                true,
            ),
            layers,
            norm: LayerNorm::single(d_model),
            d_model,
            patch_size,
        }
    }

    /// Convert image to patch tokens and encode.
    fn forward(&self, image: &Variable) -> Variable {
        let shape = image.shape();
        let (n, _, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let ph = h / self.patch_size;
        let pw = w / self.patch_size;
        let seq_len = ph * pw;

        // Patch embedding
        let patches = self.patch_embed.forward(image);

        // Reshape [N, D, pH, pW] -> [N, D, seq_len] -> transpose -> [N, seq_len, D]
        let mut x = patches.reshape(&[n, self.d_model, seq_len]).transpose(1, 2);

        for layer in &self.layers {
            x = layer.forward(&x);
        }

        self.norm.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.patch_embed.parameters());
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.norm.parameters());
        p
    }
}

// =============================================================================
// VQA Model
// =============================================================================

/// Visual Question Answering model.
///
/// Combines image understanding (ViT-style) with language understanding
/// (Transformer) via cross-attention fusion.
pub struct VQAModel {
    /// Image encoder
    vision_encoder: VisionEncoder,
    /// Text encoder
    text_encoder: TextEncoder,
    /// Cross-attention: text queries attend to image features
    cross_attn: CrossAttention,
    /// Classification head
    classifier: Vec<Linear>,
    /// Model dimension
    _d_model: usize,
    /// Number of answer classes
    _num_answers: usize,
    relu: ReLU,
}

impl VQAModel {
    /// Create a VQA model.
    ///
    /// # Arguments
    /// - `vocab_size`: Size of text vocabulary
    /// - `num_answers`: Number of possible answers (classification targets)
    /// - `d_model`: Hidden dimension
    /// - `nhead`: Number of attention heads
    /// - `num_layers`: Number of transformer layers (per encoder)
    /// - `patch_size`: Image patch size
    pub fn new(
        vocab_size: usize,
        num_answers: usize,
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        patch_size: usize,
    ) -> Self {
        Self {
            vision_encoder: VisionEncoder::new(d_model, nhead, num_layers, patch_size),
            text_encoder: TextEncoder::new(vocab_size, d_model, nhead, num_layers),
            cross_attn: CrossAttention::new(d_model, nhead),
            classifier: vec![
                Linear::new(d_model, d_model),
                Linear::new(d_model, num_answers),
            ],
            _d_model: d_model,
            _num_answers: num_answers,
            relu: ReLU,
        }
    }

    /// Create a small VQA model for testing.
    pub fn small(vocab_size: usize, num_answers: usize) -> Self {
        Self::new(vocab_size, num_answers, 64, 4, 2, 8)
    }

    /// Forward pass for VQA.
    ///
    /// # Arguments
    /// - `image`: `[N, 3, H, W]` image tensor
    /// - `question_ids`: `[N, seq_len]` token IDs (as f32)
    ///
    /// # Returns
    /// `[N, num_answers]` logits over possible answers.
    pub fn forward_vqa(&self, image: &Variable, question_ids: &Variable) -> Variable {
        // Encode image -> [N, num_patches, d_model]
        let image_features = self.vision_encoder.forward(image);

        // Encode text -> [N, seq_len, d_model]
        let text_features = self.text_encoder.forward(question_ids);

        // Cross-attention: text attends to image
        let fused = self
            .cross_attn
            .cross_attention(&text_features, &image_features, None);

        // Pool: mean over sequence dimension (graph-tracked)
        let pooled = fused.mean_dim(1, false);

        // Classify
        let out = self.relu.forward(&self.classifier[0].forward(&pooled));
        self.classifier[1].forward(&out)
    }
}

impl Module for VQAModel {
    fn forward(&self, x: &Variable) -> Variable {
        // Single-input forward returns vision features only
        self.vision_encoder.forward(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.vision_encoder.parameters());
        p.extend(self.text_encoder.parameters());
        p.extend(self.cross_attn.parameters());
        for layer in &self.classifier {
            p.extend(layer.parameters());
        }
        p
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_vqa_creation() {
        let model = VQAModel::small(1000, 100);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_vqa_forward() {
        let model = VQAModel::small(100, 50);

        let image = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );

        // Question tokens (as float indices)
        let question = Variable::new(
            Tensor::from_vec(vec![1.0, 5.0, 10.0, 20.0, 3.0], &[1, 5]).unwrap(),
            false,
        );

        let logits = model.forward_vqa(&image, &question);
        assert_eq!(logits.shape(), vec![1, 50]);
    }

    #[test]
    fn test_text_encoder() {
        let enc = TextEncoder::new(100, 64, 4, 2);
        let tokens = Variable::new(
            Tensor::from_vec(vec![1.0, 5.0, 10.0], &[1, 3]).unwrap(),
            false,
        );
        let output = enc.forward(&tokens);
        assert_eq!(output.shape(), vec![1, 3, 64]);
    }

    #[test]
    fn test_vision_encoder() {
        let enc = VisionEncoder::new(64, 4, 2, 8);
        let image = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let output = enc.forward(&image);
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[2], 64); // d_model
    }
}
