//! Depth Estimation Models
//!
//! # Models
//!
//! - **DPT**: Dense Prediction Transformer — ViT backbone with multi-scale
//!   reassembly for monocular depth estimation (server-grade).
//!
//! - **FastDepth**: Lightweight encoder-decoder with depthwise separable
//!   convolutions for real-time depth on edge devices.
//!
//! # References
//!
//! - DPT: "Vision Transformers for Dense Prediction" (Ranftl et al., 2021)
//!   <https://arxiv.org/abs/2103.13413>
//! - FastDepth: "FastDepth: Fast Monocular Depth Estimation on Embedded Systems"
//!   (Wofk et al., 2019) <https://arxiv.org/abs/1903.03273>

use axonml_autograd::Variable;
use axonml_nn::{
    BatchNorm2d, Conv2d, ConvTranspose2d, Linear, Module, Parameter, ReLU,
};
use axonml_tensor::Tensor;

use crate::ops::{interpolate_var, DepthMap, InterpolateMode};

// =============================================================================
// DPT (Dense Prediction Transformer)
// =============================================================================

/// DPT — Dense Prediction Transformer for monocular depth estimation.
///
/// Uses a ViT-like backbone to produce multi-scale features, then
/// reassembles them into a dense depth prediction.
pub struct DPT {
    /// Patch embedding: project image patches to tokens
    patch_embed: Conv2d,
    /// Transformer encoder layers
    encoder_layers: Vec<DPTEncoderLayer>,
    /// Reassembly modules for multi-scale fusion
    reassemble: Vec<DPTReassemble>,
    /// Fusion modules
    fusion: Vec<DPTFusion>,
    /// Final depth head
    depth_head: Conv2d,
    /// Model dimension
    d_model: usize,
    /// Patch size
    patch_size: usize,
    /// Number of encoder layers
    num_layers: usize,
}

struct DPTEncoderLayer {
    qkv: Linear,
    proj: Linear,
    ffn1: Linear,
    ffn2: Linear,
    norm1: axonml_nn::LayerNorm,
    norm2: axonml_nn::LayerNorm,
    _d_model: usize,
    _num_heads: usize,
}

impl DPTEncoderLayer {
    fn new(d_model: usize, num_heads: usize) -> Self {
        Self {
            qkv: Linear::new(d_model, d_model * 3),
            proj: Linear::new(d_model * 3, d_model),
            ffn1: Linear::new(d_model, d_model * 4),
            ffn2: Linear::new(d_model * 4, d_model),
            norm1: axonml_nn::LayerNorm::single(d_model),
            norm2: axonml_nn::LayerNorm::single(d_model),
            _d_model: d_model,
            _num_heads: num_heads,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        // Simplified self-attention + FFN
        let normed = self.norm1.forward(x);
        let qkv = self.qkv.forward(&normed);
        let attn = self.proj.forward(&qkv.relu()); // Simplified attention
        let x = x.add_var(&attn);

        let normed = self.norm2.forward(&x);
        let ffn = self.ffn2.forward(&self.ffn1.forward(&normed).relu());
        x.add_var(&ffn)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.qkv.parameters());
        p.extend(self.proj.parameters());
        p.extend(self.ffn1.parameters());
        p.extend(self.ffn2.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

/// Reassemble tokens back to spatial feature maps.
struct DPTReassemble {
    proj: Linear,
    target_channels: usize,
}

impl DPTReassemble {
    fn new(d_model: usize, target_channels: usize) -> Self {
        Self {
            proj: Linear::new(d_model, target_channels),
            target_channels,
        }
    }

    /// Reshape tokens [N, seq_len, D] -> [N, C, H, W].
    fn forward(&self, tokens: &Variable, h: usize, w: usize) -> Variable {
        let projected = self.proj.forward(tokens);
        let n = projected.shape()[0];
        let c = self.target_channels;

        // [N, H*W, C] -> [N, H, W, C] -> transpose to [N, C, H, W]
        let reshaped = projected.reshape(&[n, h, w, c]);
        // Transpose: (0,1,2,3) NHWC -> (0,3,1,2) NCHW
        reshaped.transpose(1, 3).transpose(2, 3)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.proj.parameters()
    }
}

/// Fusion module: progressively merge feature maps.
struct DPTFusion {
    conv1: Conv2d,
    conv2: Conv2d,
    bn: BatchNorm2d,
    relu: ReLU,
}

impl DPTFusion {
    fn new(channels: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
            conv2: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
            bn: BatchNorm2d::new(channels),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable, residual: Option<&Variable>) -> Variable {
        let out = match residual {
            Some(r) => x.add_var(r),
            None => x.clone(),
        };
        let out = self.relu.forward(&self.bn.forward(&self.conv1.forward(&out)));
        self.conv2.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn.parameters());
        p
    }
}

impl DPT {
    /// Create DPT with specified configuration.
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize, patch_size: usize) -> Self {
        let fusion_channels = 256;

        let encoder_layers = (0..num_layers)
            .map(|_| DPTEncoderLayer::new(d_model, num_heads))
            .collect();

        // Reassemble at 4 different depths
        let reassemble = vec![
            DPTReassemble::new(d_model, fusion_channels),
            DPTReassemble::new(d_model, fusion_channels),
            DPTReassemble::new(d_model, fusion_channels),
            DPTReassemble::new(d_model, fusion_channels),
        ];

        let fusion = vec![
            DPTFusion::new(fusion_channels),
            DPTFusion::new(fusion_channels),
            DPTFusion::new(fusion_channels),
            DPTFusion::new(fusion_channels),
        ];

        Self {
            patch_embed: Conv2d::with_options(
                3, d_model, (patch_size, patch_size),
                (patch_size, patch_size), (0, 0), true,
            ),
            encoder_layers,
            reassemble,
            fusion,
            depth_head: Conv2d::with_options(fusion_channels, 1, (3, 3), (1, 1), (1, 1), true),
            d_model,
            patch_size,
            num_layers,
        }
    }

    /// Create a small DPT for testing.
    pub fn small() -> Self {
        Self::new(64, 4, 4, 8)
    }

    /// Create DPT-Base.
    pub fn base() -> Self {
        Self::new(768, 12, 12, 16)
    }

    /// Estimate depth from an RGB image.
    ///
    /// # Arguments
    /// - `image`: `[N, 3, H, W]` normalized RGB image
    ///
    /// # Returns
    /// `DepthMap` with relative depth values.
    pub fn estimate_depth(&self, image: &Variable) -> DepthMap {
        let depth_var = self.forward(image);
        let data = depth_var.data().to_vec();
        let shape = depth_var.shape();
        let h = shape[2];
        let w = shape[3];

        let min_depth = data.iter().copied().fold(f32::MAX, f32::min);
        let max_depth = data.iter().copied().fold(f32::MIN, f32::max);

        // Extract single-channel depth map
        let depth_data: Vec<f32> = data[..h * w].to_vec();
        let depth = Tensor::from_vec(depth_data, &[h, w]).unwrap();

        DepthMap {
            depth,
            min_depth,
            max_depth,
        }
    }
}

impl Module for DPT {
    fn forward(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        let (n, _, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let ph = h / self.patch_size;
        let pw = w / self.patch_size;

        // Patch embedding: [N, 3, H, W] -> [N, d_model, pH, pW]
        let patches = self.patch_embed.forward(x);

        // Flatten to tokens: [N, d_model, pH, pW] -> [N, pH*pW, d_model]
        let seq_len = ph * pw;
        // [N, d_model, pH, pW] -> [N, d_model, pH*pW] -> transpose -> [N, pH*pW, d_model]
        let mut tokens = patches
            .reshape(&[n, self.d_model, seq_len])
            .transpose(1, 2);

        // Extract features at 4 depths
        let quarter = self.num_layers / 4;
        let mut layer_features = Vec::new();

        for (i, layer) in self.encoder_layers.iter().enumerate() {
            tokens = layer.forward(&tokens);
            if (i + 1) % quarter == 0 {
                layer_features.push(tokens.clone());
            }
        }

        // Ensure we have 4 features
        while layer_features.len() < 4 {
            layer_features.push(tokens.clone());
        }

        // Reassemble each level back to spatial
        let features: Vec<Variable> = layer_features
            .iter()
            .enumerate()
            .map(|(i, feat)| self.reassemble[i].forward(feat, ph, pw))
            .collect();

        // Progressive fusion (bottom-up) — use interpolate_var for graph tracking
        let mut fused = self.fusion[3].forward(&features[3], None);
        let fused_up = interpolate_var(&fused, ph, pw, InterpolateMode::Bilinear);
        fused = self.fusion[2].forward(&features[2], Some(&fused_up));
        let fused_up = interpolate_var(&fused, ph, pw, InterpolateMode::Bilinear);
        fused = self.fusion[1].forward(&features[1], Some(&fused_up));
        let fused_up = interpolate_var(&fused, ph, pw, InterpolateMode::Bilinear);
        fused = self.fusion[0].forward(&features[0], Some(&fused_up));

        // Depth prediction
        let depth = self.depth_head.forward(&fused);

        // Upsample to original resolution (graph-tracked)
        interpolate_var(&depth, h, w, InterpolateMode::Bilinear)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.patch_embed.parameters());
        for layer in &self.encoder_layers {
            p.extend(layer.parameters());
        }
        for r in &self.reassemble {
            p.extend(r.parameters());
        }
        for f in &self.fusion {
            p.extend(f.parameters());
        }
        p.extend(self.depth_head.parameters());
        p
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// FastDepth (Edge)
// =============================================================================

/// FastDepth — lightweight monocular depth estimation for edge devices.
///
/// Encoder-decoder architecture with depthwise separable convolutions.
/// <4M parameters, designed for real-time inference on embedded GPUs.
pub struct FastDepth {
    // Encoder (MobileNet-style)
    enc_conv1: Conv2d,
    enc_bn1: BatchNorm2d,
    enc_dw1: Conv2d,
    enc_pw1: Conv2d,
    enc_bn2: BatchNorm2d,
    enc_dw2: Conv2d,
    enc_pw2: Conv2d,
    enc_bn3: BatchNorm2d,
    enc_dw3: Conv2d,
    enc_pw3: Conv2d,
    enc_bn4: BatchNorm2d,
    // Decoder (transposed convolutions)
    dec1: ConvTranspose2d,
    dec_bn1: BatchNorm2d,
    dec2: ConvTranspose2d,
    dec_bn2: BatchNorm2d,
    dec3: ConvTranspose2d,
    dec_bn3: BatchNorm2d,
    // Depth prediction
    depth_conv: Conv2d,
    relu: ReLU,
}

impl FastDepth {
    /// Create FastDepth for 224x224 input.
    pub fn new() -> Self {
        Self {
            // Encoder
            enc_conv1: Conv2d::with_options(3, 32, (3, 3), (2, 2), (1, 1), true),
            enc_bn1: BatchNorm2d::new(32),
            enc_dw1: Conv2d::with_groups(32, 32, (3, 3), (2, 2), (1, 1), true, 32),
            enc_pw1: Conv2d::with_options(32, 64, (1, 1), (1, 1), (0, 0), true),
            enc_bn2: BatchNorm2d::new(64),
            enc_dw2: Conv2d::with_groups(64, 64, (3, 3), (2, 2), (1, 1), true, 64),
            enc_pw2: Conv2d::with_options(64, 128, (1, 1), (1, 1), (0, 0), true),
            enc_bn3: BatchNorm2d::new(128),
            enc_dw3: Conv2d::with_groups(128, 128, (3, 3), (2, 2), (1, 1), true, 128),
            enc_pw3: Conv2d::with_options(128, 256, (1, 1), (1, 1), (0, 0), true),
            enc_bn4: BatchNorm2d::new(256),
            // Decoder
            dec1: ConvTranspose2d::with_options(256, 128, (4, 4), (2, 2), (1, 1), (0, 0), true),
            dec_bn1: BatchNorm2d::new(128),
            dec2: ConvTranspose2d::with_options(128, 64, (4, 4), (2, 2), (1, 1), (0, 0), true),
            dec_bn2: BatchNorm2d::new(64),
            dec3: ConvTranspose2d::with_options(64, 32, (4, 4), (2, 2), (1, 1), (0, 0), true),
            dec_bn3: BatchNorm2d::new(32),
            depth_conv: Conv2d::with_options(32, 1, (3, 3), (1, 1), (1, 1), true),
            relu: ReLU,
        }
    }

    /// Estimate depth from an image.
    pub fn estimate_depth(&self, image: &Variable) -> DepthMap {
        let depth_var = self.forward(image);
        let data = depth_var.data().to_vec();
        let shape = depth_var.shape();
        let h = shape[2];
        let w = shape[3];

        let min_depth = data.iter().copied().fold(f32::MAX, f32::min);
        let max_depth = data.iter().copied().fold(f32::MIN, f32::max);

        let depth_data: Vec<f32> = data[..h * w].to_vec();
        let depth = Tensor::from_vec(depth_data, &[h, w]).unwrap();

        DepthMap {
            depth,
            min_depth,
            max_depth,
        }
    }
}

impl Module for FastDepth {
    fn forward(&self, x: &Variable) -> Variable {
        // Encoder
        let e1 = self.relu.forward(&self.enc_bn1.forward(&self.enc_conv1.forward(x)));
        let e2 = self.relu.forward(&self.enc_bn2.forward(
            &self.enc_pw1.forward(&self.enc_dw1.forward(&e1)),
        ));
        let e3 = self.relu.forward(&self.enc_bn3.forward(
            &self.enc_pw2.forward(&self.enc_dw2.forward(&e2)),
        ));
        let e4 = self.relu.forward(&self.enc_bn4.forward(
            &self.enc_pw3.forward(&self.enc_dw3.forward(&e3)),
        ));

        // Decoder
        let d1 = self.relu.forward(&self.dec_bn1.forward(&self.dec1.forward(&e4)));
        let d2 = self.relu.forward(&self.dec_bn2.forward(&self.dec2.forward(&d1)));
        let d3 = self.relu.forward(&self.dec_bn3.forward(&self.dec3.forward(&d2)));

        // Depth prediction (ReLU to ensure positive depth)
        self.relu.forward(&self.depth_conv.forward(&d3))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.enc_conv1.parameters());
        p.extend(self.enc_bn1.parameters());
        p.extend(self.enc_dw1.parameters());
        p.extend(self.enc_pw1.parameters());
        p.extend(self.enc_bn2.parameters());
        p.extend(self.enc_dw2.parameters());
        p.extend(self.enc_pw2.parameters());
        p.extend(self.enc_bn3.parameters());
        p.extend(self.enc_dw3.parameters());
        p.extend(self.enc_pw3.parameters());
        p.extend(self.enc_bn4.parameters());
        p.extend(self.dec1.parameters());
        p.extend(self.dec_bn1.parameters());
        p.extend(self.dec2.parameters());
        p.extend(self.dec_bn2.parameters());
        p.extend(self.dec3.parameters());
        p.extend(self.dec_bn3.parameters());
        p.extend(self.depth_conv.parameters());
        p
    }

    fn train(&mut self) {
        self.enc_bn1.train();
        self.enc_bn2.train();
        self.enc_bn3.train();
        self.enc_bn4.train();
        self.dec_bn1.train();
        self.dec_bn2.train();
        self.dec_bn3.train();
    }

    fn eval(&mut self) {
        self.enc_bn1.eval();
        self.enc_bn2.eval();
        self.enc_bn3.eval();
        self.enc_bn4.eval();
        self.dec_bn1.eval();
        self.dec_bn2.eval();
        self.dec_bn3.eval();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpt_small_creation() {
        let model = DPT::small();
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_dpt_small_forward() {
        let model = DPT::small();
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        // Output should match input spatial dims
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 1); // single-channel depth
        assert_eq!(output.shape()[2], 32);
        assert_eq!(output.shape()[3], 32);
    }

    #[test]
    fn test_fastdepth_creation() {
        let model = FastDepth::new();
        let params = model.parameters();
        assert!(!params.is_empty());

        // Should be lightweight
        let total: usize = params.iter().map(|p| p.variable().data().to_vec().len()).sum();
        assert!(total < 4_000_000);
    }

    #[test]
    fn test_fastdepth_forward() {
        let model = FastDepth::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 1); // single-channel depth

        // All depth values should be non-negative (ReLU output)
        let data = output.data().to_vec();
        for &v in &data {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_dpt_estimate_depth() {
        let model = DPT::small();
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let depth_map = model.estimate_depth(&input);
        assert_eq!(depth_map.depth.shape().len(), 2);
    }
}
