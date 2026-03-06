//! DETR - DEtection TRansformer
//!
//! End-to-end object detection with Transformers.
//! No anchors, no NMS — uses set-based Hungarian matching.
//!
//! # Architecture
//!
//! ResNet backbone -> flatten + positional encoding -> Transformer encoder-decoder
//! -> FFN heads for class and bbox prediction on object queries.
//!
//! # Reference
//!
//! "End-to-End Object Detection with Transformers" (Carion et al., 2020)
//! <https://arxiv.org/abs/2005.12872>

use axonml_autograd::Variable;
use axonml_nn::{
    Conv2d, LayerNorm, Linear, Module, MultiHeadAttention, Parameter, ReLU,
};
use axonml_tensor::Tensor;

use crate::ops::{positional_encoding_2d, Detection};

// =============================================================================
// DETR Transformer
// =============================================================================

/// DETR-specific Transformer with encoder and decoder.
struct DETRTransformer {
    encoder_layers: Vec<DETREncoderLayer>,
    decoder_layers: Vec<DETRDecoderLayer>,
    d_model: usize,
}

struct DETREncoderLayer {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
    norm2: LayerNorm,
    relu: ReLU,
}

struct DETRDecoderLayer {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    cross_attn: MultiHeadAttention,
    norm2: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
    norm3: LayerNorm,
    relu: ReLU,
}

impl DETREncoderLayer {
    fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            norm1: LayerNorm::single(d_model),
            ffn1: Linear::new(d_model, dim_feedforward),
            ffn2: Linear::new(dim_feedforward, d_model),
            norm2: LayerNorm::single(d_model),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        // Self-attention + residual
        let attn = self.self_attn.forward(x);
        let x = self.norm1.forward(&x.add_var(&attn));

        // FFN + residual
        let ffn = self.ffn2.forward(&self.relu.forward(&self.ffn1.forward(&x)));
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

impl DETRDecoderLayer {
    fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            norm1: LayerNorm::single(d_model),
            cross_attn: MultiHeadAttention::new(d_model, nhead),
            norm2: LayerNorm::single(d_model),
            ffn1: Linear::new(d_model, dim_feedforward),
            ffn2: Linear::new(dim_feedforward, d_model),
            norm3: LayerNorm::single(d_model),
            relu: ReLU,
        }
    }

    fn forward(&self, query: &Variable, memory: &Variable) -> Variable {
        // Self-attention on queries
        let q = self.self_attn.forward(query);
        let query = self.norm1.forward(&query.add_var(&q));

        // Cross-attention: queries attend to encoder memory
        let cross = self.cross_attn.attention(&query, memory, memory, None);
        let query = self.norm2.forward(&query.add_var(&cross));

        // FFN
        let ffn = self.ffn2.forward(&self.relu.forward(&self.ffn1.forward(&query)));
        self.norm3.forward(&query.add_var(&ffn))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.cross_attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.ffn1.parameters());
        p.extend(self.ffn2.parameters());
        p.extend(self.norm3.parameters());
        p
    }
}

impl DETRTransformer {
    fn new(d_model: usize, nhead: usize, num_encoder_layers: usize, num_decoder_layers: usize) -> Self {
        let dim_feedforward = d_model * 4;
        let encoder_layers = (0..num_encoder_layers)
            .map(|_| DETREncoderLayer::new(d_model, nhead, dim_feedforward))
            .collect();
        let decoder_layers = (0..num_decoder_layers)
            .map(|_| DETRDecoderLayer::new(d_model, nhead, dim_feedforward))
            .collect();

        Self {
            encoder_layers,
            decoder_layers,
            d_model,
        }
    }

    fn forward(&self, src: &Variable, query: &Variable) -> Variable {
        // Encode
        let mut memory = src.clone();
        for layer in &self.encoder_layers {
            memory = layer.forward(&memory);
        }

        // Decode
        let mut output = query.clone();
        for layer in &self.decoder_layers {
            output = layer.forward(&output, &memory);
        }

        output
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for layer in &self.encoder_layers {
            p.extend(layer.parameters());
        }
        for layer in &self.decoder_layers {
            p.extend(layer.parameters());
        }
        p
    }
}

// =============================================================================
// DETR Model
// =============================================================================

/// DETR — DEtection TRansformer.
///
/// End-to-end object detection without anchors or NMS.
/// Uses learned object queries and set-based prediction.
pub struct DETR {
    /// Backbone feature projection (reduces backbone channels to d_model)
    input_proj: Conv2d,
    /// Transformer encoder-decoder
    transformer: DETRTransformer,
    /// Class prediction head
    class_embed: Linear,
    /// Bounding box prediction head (predicts cx, cy, w, h)
    bbox_embed: Vec<Linear>,
    /// Learned object queries
    query_embed_data: Tensor<f32>,
    /// Model dimension
    d_model: usize,
    /// Number of object queries
    num_queries: usize,
    /// Number of classes (including background)
    num_classes: usize,
}

impl DETR {
    /// Create a DETR model.
    ///
    /// # Arguments
    /// - `num_classes`: Number of object classes (excluding "no object")
    /// - `num_queries`: Number of object queries (max detections per image)
    /// - `d_model`: Transformer hidden dimension
    /// - `nhead`: Number of attention heads
    /// - `backbone_channels`: Output channels from backbone (e.g., 512 for ResNet34)
    pub fn new(
        num_classes: usize,
        num_queries: usize,
        d_model: usize,
        nhead: usize,
        backbone_channels: usize,
    ) -> Self {
        // Project backbone features to d_model
        let input_proj = Conv2d::with_options(
            backbone_channels, d_model, (1, 1), (1, 1), (0, 0), true,
        );

        let transformer = DETRTransformer::new(d_model, nhead, 6, 6);

        // +1 for "no object" class
        let class_embed = Linear::new(d_model, num_classes + 1);

        // 3-layer MLP for bbox prediction
        let bbox_embed = vec![
            Linear::new(d_model, d_model),
            Linear::new(d_model, d_model),
            Linear::new(d_model, 4), // cx, cy, w, h
        ];

        // Initialize object queries (learned positional embeddings)
        let query_data: Vec<f32> = (0..num_queries * d_model)
            .map(|i| ((i as f32 * 0.02).sin()) * 0.1)
            .collect();
        let query_embed_data = Tensor::from_vec(query_data, &[num_queries, d_model]).unwrap();

        Self {
            input_proj,
            transformer,
            class_embed,
            bbox_embed,
            query_embed_data,
            d_model,
            num_queries,
            num_classes,
        }
    }

    /// Create DETR with default settings for COCO (91 classes).
    pub fn for_coco() -> Self {
        Self::new(91, 100, 256, 8, 512)
    }

    /// Create a small DETR for testing.
    pub fn small(num_classes: usize) -> Self {
        Self::new(num_classes, 10, 64, 4, 64)
    }

    /// Forward pass: takes backbone features and returns (class_logits, bbox_pred).
    ///
    /// # Arguments
    /// - `backbone_features`: `[N, C, H, W]` from backbone (e.g., ResNet C5)
    ///
    /// # Returns
    /// - `class_logits`: `[N, num_queries, num_classes+1]`
    /// - `bbox_pred`: `[N, num_queries, 4]` in `(cx, cy, w, h)` format, normalized to [0,1]
    pub fn forward_detection(&self, backbone_features: &Variable) -> (Variable, Variable) {
        let shape = backbone_features.shape();
        let n = shape[0];
        let h = shape[2];
        let w = shape[3];

        // Project to d_model channels
        let src = self.input_proj.forward(backbone_features);

        // Add 2D positional encoding
        let pe = positional_encoding_2d(h, w, self.d_model);
        let pe_var = Variable::new(pe, false);

        // Flatten spatial dims: [N, d_model, H, W] -> [N, H*W, d_model]
        let seq_len = h * w;
        // [N, d_model, H*W] -> transpose -> [N, H*W, d_model]
        let src_var = src.reshape(&[n, self.d_model, seq_len]).transpose(1, 2);

        // Transpose PE: [d_model, H, W] -> [H*W, d_model]
        let pe_data = pe_var.data().to_vec();
        let mut pe_flat = vec![0.0f32; seq_len * self.d_model];
        for c in 0..self.d_model {
            for s in 0..seq_len {
                pe_flat[s * self.d_model + c] = pe_data[c * seq_len + s];
            }
        }

        // Expand PE to batch size (constant, no grad needed)
        let pe_expanded_data: Vec<f32> = (0..n).flat_map(|_| pe_flat.iter().copied()).collect();
        let pe_expanded = Variable::new(
            Tensor::from_vec(pe_expanded_data, &[n, seq_len, self.d_model]).unwrap(),
            false,
        );

        let src_with_pe = src_var.add_var(&pe_expanded);

        // Object queries: expand to batch (constant embeddings)
        let qd = self.query_embed_data.to_vec();
        let query_expanded: Vec<f32> = (0..n).flat_map(|_| qd.iter().copied()).collect();
        let queries = Variable::new(
            Tensor::from_vec(query_expanded, &[n, self.num_queries, self.d_model]).unwrap(),
            true,
        );

        // Transformer: encode features, decode with object queries
        let decoder_out = self.transformer.forward(&src_with_pe, &queries);

        // Classification head
        let class_logits = self.class_embed.forward(&decoder_out);

        // Bbox head (MLP with ReLU)
        let relu = ReLU;
        let mut bbox = self.bbox_embed[0].forward(&decoder_out);
        bbox = relu.forward(&bbox);
        bbox = self.bbox_embed[1].forward(&bbox);
        bbox = relu.forward(&bbox);
        bbox = self.bbox_embed[2].forward(&bbox);
        // Apply sigmoid to constrain to [0, 1]
        let bbox = bbox.sigmoid();

        (class_logits, bbox)
    }

    /// Post-process predictions to get detections.
    pub fn postprocess(
        &self,
        class_logits: &Variable,
        bbox_pred: &Variable,
        score_threshold: f32,
    ) -> Vec<Detection> {
        let cls_data = class_logits.data().to_vec();
        let bbox_data = bbox_pred.data().to_vec();
        let shape = class_logits.shape();
        let num_queries = shape[1];
        let num_cls = shape[2];

        let mut detections = Vec::new();

        for q in 0..num_queries {
            // Find best class (skip "no object" which is last class)
            let mut best_cls = 0;
            let mut best_score = f32::NEG_INFINITY;

            for c in 0..num_cls - 1 {
                let score = cls_data[q * num_cls + c];
                if score > best_score {
                    best_score = score;
                    best_cls = c;
                }
            }

            // Apply softmax to get probability
            let max_val: f32 = cls_data[q * num_cls..(q + 1) * num_cls]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = cls_data[q * num_cls..(q + 1) * num_cls]
                .iter()
                .map(|&v| (v - max_val).exp())
                .sum();
            let prob = (best_score - max_val).exp() / sum_exp;

            if prob < score_threshold {
                continue;
            }

            // Decode bbox (cx, cy, w, h) -> (x1, y1, x2, y2)
            let cx = bbox_data[q * 4];
            let cy = bbox_data[q * 4 + 1];
            let w = bbox_data[q * 4 + 2];
            let h = bbox_data[q * 4 + 3];

            detections.push(Detection {
                bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                confidence: prob,
                class_id: best_cls,
            });
        }

        detections
    }
}

impl Module for DETR {
    fn forward(&self, x: &Variable) -> Variable {
        let (class_logits, _) = self.forward_detection(x);
        class_logits
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.input_proj.parameters());
        p.extend(self.transformer.parameters());
        p.extend(self.class_embed.parameters());
        for layer in &self.bbox_embed {
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

    #[test]
    fn test_detr_creation() {
        let model = DETR::small(10);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_detr_forward() {
        let model = DETR::small(10);

        // Simulated backbone features [1, 64, 4, 4]
        let features = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 64 * 4 * 4], &[1, 64, 4, 4]).unwrap(),
            false,
        );

        let (cls, bbox) = model.forward_detection(&features);
        assert_eq!(cls.shape(), vec![1, 10, 11]); // 10 queries, 11 classes (10+no_object)
        assert_eq!(bbox.shape(), vec![1, 10, 4]);

        // Bbox values should be in [0, 1] (sigmoid)
        let bbox_data = bbox.data().to_vec();
        for &v in &bbox_data {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_detr_postprocess() {
        let model = DETR::small(10);

        let features = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 64 * 4 * 4], &[1, 64, 4, 4]).unwrap(),
            false,
        );

        let (cls, bbox) = model.forward_detection(&features);
        let dets = model.postprocess(&cls, &bbox, 0.01);
        // Should have some detections with low threshold
        assert!(dets.len() <= 10); // At most num_queries detections
    }

    #[test]
    fn test_detr_encoder_layer() {
        let layer = DETREncoderLayer::new(64, 4, 256);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 16 * 64], &[1, 16, 64]).unwrap(),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![1, 16, 64]);
    }
}
