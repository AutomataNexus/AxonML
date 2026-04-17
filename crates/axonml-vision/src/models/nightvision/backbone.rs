//! NightVision Backbone — CSP-based Thermal Feature Extraction
//!
//! Cross-Stage Partial (CSP) backbone tuned for infrared imagery that
//! produces the multi-scale feature maps (P3, P4, P5) consumed by the
//! NightVision FPN neck and detection head. The building blocks are:
//!
//! - `silu` — SiLU/Swish activation (`x * sigmoid(x)`).
//! - `ConvBNSiLU` — `Conv2d` + `BatchNorm2d` + SiLU with parameter, named-
//!   parameter, and train/eval plumbing.
//! - `Bottleneck` — 1×1 → 3×3 residual bottleneck with an optional skip
//!   connection (enabled only when `in_ch == out_ch`).
//! - `CSPBlock` — Cross-Stage Partial block: stride-2 downsample, channel
//!   split into two 1×1 branches, a bottleneck stack on branch 1,
//!   concatenation, and a 1×1 merge.
//! - `ThermalBackbone` — top-level wrapper with an adaptive 1→3 channel
//!   adapter for single-channel IR, a stride-2 stem, and three CSP stages
//!   emitting 64/128/256-channel features at 1/4, 1/8, and 1/16 resolution.
//!
//! Unit tests validate output shapes for each component and verify that the
//! 1-channel backbone has strictly more parameters than the 3-channel one
//! thanks to the channel adapter.
//!
//! # File
//! `crates/axonml-vision/src/models/nightvision/backbone.rs`
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

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Module, Parameter};

// =============================================================================
// SiLU Activation (Swish)
// =============================================================================

/// SiLU / Swish activation: x * sigmoid(x)
/// More effective than ReLU for detection networks.
fn silu(x: &Variable) -> Variable {
    let sig = x.sigmoid();
    x.mul_var(&sig)
}

// =============================================================================
// ConvBNSiLU — Conv2d + BatchNorm + SiLU
// =============================================================================

/// Convolution + BatchNorm + SiLU activation block.
pub struct ConvBNSiLU {
    conv: Conv2d,
    bn: BatchNorm2d,
}

impl ConvBNSiLU {
    /// Create a new ConvBNSiLU block.
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize) -> Self {
        Self {
            conv: Conv2d::with_options(
                in_ch,
                out_ch,
                (kernel, kernel),
                (stride, stride),
                (padding, padding),
                true,
            ),
            bn: BatchNorm2d::new(out_ch),
        }
    }

    /// Forward pass through conv, batchnorm, and SiLU activation.
    pub fn forward(&self, x: &Variable) -> Variable {
        silu(&self.bn.forward(&self.conv.forward(x)))
    }

    /// Returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.conv.parameters();
        p.extend(self.bn.parameters());
        p
    }

    /// Returns named parameters with the given prefix.
    pub fn named_parameters(&self, prefix: &str) -> HashMap<String, Parameter> {
        let mut p = HashMap::new();
        for (k, v) in self.conv.named_parameters() {
            p.insert(format!("{}.conv.{}", prefix, k), v);
        }
        for (k, v) in self.bn.named_parameters() {
            p.insert(format!("{}.bn.{}", prefix, k), v);
        }
        p
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, training: bool) {
        self.bn.set_training(training);
    }
}

// =============================================================================
// Bottleneck — Residual bottleneck block
// =============================================================================

/// Residual bottleneck block with optional skip connection.
pub struct Bottleneck {
    cv1: ConvBNSiLU,
    cv2: ConvBNSiLU,
    shortcut: bool,
}

impl Bottleneck {
    /// Create a new bottleneck block.
    pub fn new(in_ch: usize, out_ch: usize, shortcut: bool) -> Self {
        let hidden = out_ch; // No expansion for simplicity
        Self {
            cv1: ConvBNSiLU::new(in_ch, hidden, 1, 1, 0),
            cv2: ConvBNSiLU::new(hidden, out_ch, 3, 1, 1),
            shortcut: shortcut && in_ch == out_ch,
        }
    }

    /// Forward pass with optional residual connection.
    pub fn forward(&self, x: &Variable) -> Variable {
        let out = self.cv2.forward(&self.cv1.forward(x));
        if self.shortcut { out.add_var(x) } else { out }
    }

    /// Returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.cv1.parameters();
        p.extend(self.cv2.parameters());
        p
    }

    /// Returns named parameters with the given prefix.
    pub fn named_parameters(&self, prefix: &str) -> HashMap<String, Parameter> {
        let mut p = self.cv1.named_parameters(&format!("{}.cv1", prefix));
        p.extend(self.cv2.named_parameters(&format!("{}.cv2", prefix)));
        p
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, training: bool) {
        self.cv1.set_training(training);
        self.cv2.set_training(training);
    }
}

// =============================================================================
// CSPBlock — Cross-Stage Partial block
// =============================================================================

/// Cross-Stage Partial block: splits channels, processes one branch through
/// bottlenecks, concatenates with the other branch.
pub struct CSPBlock {
    /// Downsample: stride-2 conv
    downsample: ConvBNSiLU,
    /// Split branch 1: 1×1 conv to half channels
    cv1: ConvBNSiLU,
    /// Split branch 2: 1×1 conv to half channels
    cv2: ConvBNSiLU,
    /// Bottleneck stack on branch 1
    bottlenecks: Vec<Bottleneck>,
    /// Merge: 1×1 conv to combine concatenated branches
    cv3: ConvBNSiLU,
    out_ch: usize,
}

impl CSPBlock {
    /// Create a new CSP block with the given channel sizes and bottleneck count.
    pub fn new(in_ch: usize, out_ch: usize, n_bottlenecks: usize) -> Self {
        let half = out_ch / 2;
        Self {
            downsample: ConvBNSiLU::new(in_ch, out_ch, 3, 2, 1),
            cv1: ConvBNSiLU::new(out_ch, half, 1, 1, 0),
            cv2: ConvBNSiLU::new(out_ch, half, 1, 1, 0),
            bottlenecks: (0..n_bottlenecks)
                .map(|_| Bottleneck::new(half, half, true))
                .collect(),
            cv3: ConvBNSiLU::new(half * 2, out_ch, 1, 1, 0),
            out_ch,
        }
    }

    /// Forward pass. Returns the output feature map.
    pub fn forward(&self, x: &Variable) -> Variable {
        let x = self.downsample.forward(x);

        // Branch 1: bottleneck stack
        let mut b1 = self.cv1.forward(&x);
        for bottleneck in &self.bottlenecks {
            b1 = bottleneck.forward(&b1);
        }

        // Branch 2: skip
        let b2 = self.cv2.forward(&x);

        // Concatenate along channel dim
        let cat = Variable::cat(&[&b1, &b2], 1);
        self.cv3.forward(&cat)
    }

    /// Returns the output channel count.
    pub fn out_channels(&self) -> usize {
        self.out_ch
    }

    /// Returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.downsample.parameters();
        p.extend(self.cv1.parameters());
        p.extend(self.cv2.parameters());
        for b in &self.bottlenecks {
            p.extend(b.parameters());
        }
        p.extend(self.cv3.parameters());
        p
    }

    /// Returns named parameters with the given prefix.
    pub fn named_parameters(&self, prefix: &str) -> HashMap<String, Parameter> {
        let mut p = self
            .downsample
            .named_parameters(&format!("{}.down", prefix));
        p.extend(self.cv1.named_parameters(&format!("{}.cv1", prefix)));
        p.extend(self.cv2.named_parameters(&format!("{}.cv2", prefix)));
        for (i, b) in self.bottlenecks.iter().enumerate() {
            p.extend(b.named_parameters(&format!("{}.btn{}", prefix, i)));
        }
        p.extend(self.cv3.named_parameters(&format!("{}.cv3", prefix)));
        p
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, training: bool) {
        self.downsample.set_training(training);
        self.cv1.set_training(training);
        self.cv2.set_training(training);
        for b in &mut self.bottlenecks {
            b.set_training(training);
        }
        self.cv3.set_training(training);
    }
}

// =============================================================================
// ThermalBackbone — Full CSP backbone for IR imagery
// =============================================================================

/// CSP backbone with adaptive thermal input stem.
///
/// Produces multi-scale feature maps at 3 levels (P3, P4, P5)
/// for the Feature Pyramid Network.
pub struct ThermalBackbone {
    /// Adaptive stem: handles 1-ch or 3-ch input → 32 channels
    stem: ConvBNSiLU,
    /// Optional 1→3 channel adapter for single-channel IR
    ch_adapter: Option<ConvBNSiLU>,
    /// Stage 1: 32 → 64 (P3 scale, 1/4 resolution)
    stage1: CSPBlock,
    /// Stage 2: 64 → 128 (P4 scale, 1/8 resolution)
    stage2: CSPBlock,
    /// Stage 3: 128 → 256 (P5 scale, 1/16 resolution)
    stage3: CSPBlock,
}

impl ThermalBackbone {
    /// Create a backbone for the given input channels (1 for thermal, 3 for multi-band).
    pub fn new(in_channels: usize) -> Self {
        let ch_adapter = if in_channels == 1 {
            Some(ConvBNSiLU::new(1, 3, 1, 1, 0))
        } else {
            None
        };

        Self {
            stem: ConvBNSiLU::new(3, 32, 3, 2, 1), // Always 3→32 after adapter
            ch_adapter,
            stage1: CSPBlock::new(32, 64, 1),
            stage2: CSPBlock::new(64, 128, 2),
            stage3: CSPBlock::new(128, 256, 2),
        }
    }

    /// Forward pass returning multi-scale features (p3, p4, p5).
    pub fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        // Adapt single-channel to 3-channel if needed
        let x = if let Some(ref adapter) = self.ch_adapter {
            adapter.forward(x)
        } else {
            x.clone()
        };

        let x = self.stem.forward(&x); // [B, 32, H/2, W/2]
        let p3 = self.stage1.forward(&x); // [B, 64, H/4, W/4]
        let p4 = self.stage2.forward(&p3); // [B, 128, H/8, W/8]
        let p5 = self.stage3.forward(&p4); // [B, 256, H/16, W/16]

        (p3, p4, p5)
    }

    /// Returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        if let Some(ref adapter) = self.ch_adapter {
            p.extend(adapter.parameters());
        }
        p.extend(self.stem.parameters());
        p.extend(self.stage1.parameters());
        p.extend(self.stage2.parameters());
        p.extend(self.stage3.parameters());
        p
    }

    /// Returns named parameters.
    pub fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut p = HashMap::new();
        if let Some(ref adapter) = self.ch_adapter {
            p.extend(adapter.named_parameters("ch_adapter"));
        }
        p.extend(self.stem.named_parameters("stem"));
        p.extend(self.stage1.named_parameters("stage1"));
        p.extend(self.stage2.named_parameters("stage2"));
        p.extend(self.stage3.named_parameters("stage3"));
        p
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, training: bool) {
        if let Some(ref mut adapter) = self.ch_adapter {
            adapter.set_training(training);
        }
        self.stem.set_training(training);
        self.stage1.set_training(training);
        self.stage2.set_training(training);
        self.stage3.set_training(training);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_conv_bn_silu_shape() {
        let block = ConvBNSiLU::new(3, 32, 3, 1, 1);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 3 * 8 * 8], &[2, 3, 8, 8]).unwrap(),
            false,
        );
        let y = block.forward(&x);
        assert_eq!(y.data().shape(), &[2, 32, 8, 8]);
    }

    #[test]
    fn test_conv_bn_silu_params() {
        let block = ConvBNSiLU::new(3, 32, 3, 1, 1);
        assert!(!block.parameters().is_empty());
    }

    #[test]
    fn test_bottleneck_shortcut() {
        let block = Bottleneck::new(16, 16, true);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 16 * 4 * 4], &[1, 16, 4, 4]).unwrap(),
            false,
        );
        let y = block.forward(&x);
        assert_eq!(y.data().shape(), &[1, 16, 4, 4]);
    }

    #[test]
    fn test_bottleneck_no_shortcut() {
        let block = Bottleneck::new(16, 32, false);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 16 * 4 * 4], &[1, 16, 4, 4]).unwrap(),
            false,
        );
        let y = block.forward(&x);
        assert_eq!(y.data().shape(), &[1, 32, 4, 4]);
    }

    #[test]
    fn test_csp_block_shape() {
        let block = CSPBlock::new(32, 64, 1);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 32 * 8 * 8], &[1, 32, 8, 8]).unwrap(),
            false,
        );
        let y = block.forward(&x);
        // CSP with stride-2 downsample: spatial dims halved
        assert_eq!(y.data().shape()[0], 1);
        assert_eq!(y.data().shape()[1], 64);
        assert_eq!(y.data().shape()[2], 4);
        assert_eq!(y.data().shape()[3], 4);
    }

    #[test]
    fn test_thermal_backbone_3ch() {
        let backbone = ThermalBackbone::new(3);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let (p3, p4, p5) = backbone.forward(&x);
        // P3: 64ch, H/4, W/4
        assert_eq!(p3.data().shape()[1], 64);
        assert_eq!(p3.data().shape()[2], 16);
        // P4: 128ch, H/8, W/8
        assert_eq!(p4.data().shape()[1], 128);
        assert_eq!(p4.data().shape()[2], 8);
        // P5: 256ch, H/16, W/16
        assert_eq!(p5.data().shape()[1], 256);
        assert_eq!(p5.data().shape()[2], 4);
    }

    #[test]
    fn test_thermal_backbone_1ch() {
        let backbone = ThermalBackbone::new(1);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 32 * 32], &[1, 1, 32, 32]).unwrap(),
            false,
        );
        let (p3, p4, p5) = backbone.forward(&x);
        assert_eq!(p3.data().shape()[1], 64);
        assert_eq!(p4.data().shape()[1], 128);
        assert_eq!(p5.data().shape()[1], 256);
    }

    #[test]
    fn test_thermal_backbone_params() {
        let backbone = ThermalBackbone::new(1);
        let params = backbone.parameters();
        assert!(!params.is_empty());
        // 1-ch backbone should have more params (channel adapter)
        let backbone3 = ThermalBackbone::new(3);
        assert!(backbone.parameters().len() > backbone3.parameters().len());
    }
}
