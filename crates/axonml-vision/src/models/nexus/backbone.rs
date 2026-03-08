//! Nexus Backbone — Dual-Pathway Feature Extraction
//!
//! Implements the neuroscience-inspired dual-pathway architecture:
//! - **Ventral ("what")**: Fine-grained identity features via InvertedResidual blocks
//! - **Dorsal ("where")**: Wide-receptive-field spatial features via 5×5 convolutions
//!
//! Both pathways share a common stem and produce multi-scale features.
//!
//! @version 0.1.0

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Module, Parameter, ReLU};

// =============================================================================
// Shared Stem
// =============================================================================

/// Shared stem: initial feature extraction shared by both pathways.
///
/// Conv(3→32, 3×3, stride=2) → BN → ReLU → Conv(32→64, 3×3, stride=2) → BN → ReLU
/// Output: [B, 64, H/4, W/4]
pub struct SharedStem {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    relu: ReLU,
}

impl SharedStem {
    /// Create shared stem: 3 → 64 channels with 4× spatial reduction.
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::with_options(3, 32, (3, 3), (2, 2), (1, 1), true),
            bn1: BatchNorm2d::new(32),
            conv2: Conv2d::with_options(32, 64, (3, 3), (2, 2), (1, 1), true),
            bn2: BatchNorm2d::new(64),
            relu: ReLU,
        }
    }

    /// Forward: [B, 3, H, W] → [B, 64, H/4, W/4].
    pub fn forward(&self, x: &Variable) -> Variable {
        let out = self.relu.forward(&self.bn1.forward(&self.conv1.forward(x)));
        self.relu.forward(&self.bn2.forward(&self.conv2.forward(&out)))
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
    }

    pub fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
    }
}

impl Default for SharedStem {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Inverted Residual Block (Ventral Pathway)
// =============================================================================

/// MobileNetV2-style inverted residual block.
///
/// Expand → Depthwise → Project pattern with residual connection.
/// Used in the ventral pathway for fine-grained feature extraction.
pub struct InvertedResidualBlock {
    /// Expansion 1×1 conv.
    expand: Conv2d,
    expand_bn: BatchNorm2d,
    /// Depthwise 3×3 conv.
    dw: Conv2d,
    dw_bn: BatchNorm2d,
    /// Projection 1×1 conv.
    project: Conv2d,
    project_bn: BatchNorm2d,
    /// Residual projection (if channels/stride mismatch).
    shortcut: Option<(Conv2d, BatchNorm2d)>,
    relu: ReLU,
    use_residual: bool,
}

impl InvertedResidualBlock {
    /// Create an inverted residual block.
    ///
    /// - `expand_ratio`: Channel expansion factor (typically 2-6).
    pub fn new(in_ch: usize, out_ch: usize, stride: usize, expand_ratio: usize) -> Self {
        let mid_ch = in_ch * expand_ratio;

        let expand = Conv2d::with_options(in_ch, mid_ch, (1, 1), (1, 1), (0, 0), true);
        let expand_bn = BatchNorm2d::new(mid_ch);

        let dw = Conv2d::with_groups(mid_ch, mid_ch, (3, 3), (stride, stride), (1, 1), true, mid_ch);
        let dw_bn = BatchNorm2d::new(mid_ch);

        let project = Conv2d::with_options(mid_ch, out_ch, (1, 1), (1, 1), (0, 0), true);
        let project_bn = BatchNorm2d::new(out_ch);

        let use_residual = stride == 1 && in_ch == out_ch;
        let shortcut = if !use_residual && stride != 1 {
            None // No shortcut for stride > 1 with different channels
        } else if !use_residual {
            Some((
                Conv2d::with_options(in_ch, out_ch, (1, 1), (stride, stride), (0, 0), true),
                BatchNorm2d::new(out_ch),
            ))
        } else {
            None
        };

        Self {
            expand,
            expand_bn,
            dw,
            dw_bn,
            project,
            project_bn,
            shortcut,
            relu: ReLU,
            use_residual,
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        let out = self.relu.forward(&self.expand_bn.forward(&self.expand.forward(x)));
        let out = self.relu.forward(&self.dw_bn.forward(&self.dw.forward(&out)));
        let out = self.project_bn.forward(&self.project.forward(&out));

        if self.use_residual {
            out.add_var(x)
        } else if let Some((ref conv, ref bn)) = self.shortcut {
            out.add_var(&bn.forward(&conv.forward(x)))
        } else {
            out
        }
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.expand.parameters());
        p.extend(self.expand_bn.parameters());
        p.extend(self.dw.parameters());
        p.extend(self.dw_bn.parameters());
        p.extend(self.project.parameters());
        p.extend(self.project_bn.parameters());
        if let Some((ref c, ref bn)) = self.shortcut {
            p.extend(c.parameters());
            p.extend(bn.parameters());
        }
        p
    }

    pub fn eval(&mut self) {
        self.expand_bn.eval();
        self.dw_bn.eval();
        self.project_bn.eval();
        if let Some((_, ref mut bn)) = self.shortcut {
            bn.eval();
        }
    }

    pub fn train(&mut self) {
        self.expand_bn.train();
        self.dw_bn.train();
        self.project_bn.train();
        if let Some((_, ref mut bn)) = self.shortcut {
            bn.train();
        }
    }
}

// =============================================================================
// Ventral Pathway ("What" Stream)
// =============================================================================

/// Ventral pathway: fine-grained identity features.
///
/// 3 stages of InvertedResidual blocks with increasing channels:
/// - V1: [B, 96, H/8, W/8]
/// - V2: [B, 128, H/16, W/16]
/// - V3: [B, 192, H/32, W/32]
pub struct VentralPathway {
    stage1: Vec<InvertedResidualBlock>,
    stage2: Vec<InvertedResidualBlock>,
    stage3: Vec<InvertedResidualBlock>,
}

impl VentralPathway {
    /// Create ventral pathway from 64-channel stem output.
    pub fn new() -> Self {
        Self {
            stage1: vec![
                InvertedResidualBlock::new(64, 96, 2, 2),
                InvertedResidualBlock::new(96, 96, 1, 2),
            ],
            stage2: vec![
                InvertedResidualBlock::new(96, 128, 2, 2),
                InvertedResidualBlock::new(128, 128, 1, 2),
            ],
            stage3: vec![
                InvertedResidualBlock::new(128, 192, 2, 2),
                InvertedResidualBlock::new(192, 192, 1, 2),
            ],
        }
    }

    /// Forward: [B, 64, H/4, W/4] → (V1, V2, V3).
    pub fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let mut out = x.clone();
        for block in &self.stage1 {
            out = block.forward(&out);
        }
        let v1 = out.clone();

        for block in &self.stage2 {
            out = block.forward(&out);
        }
        let v2 = out.clone();

        for block in &self.stage3 {
            out = block.forward(&out);
        }
        let v3 = out;

        (v1, v2, v3)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for block in &self.stage1 {
            p.extend(block.parameters());
        }
        for block in &self.stage2 {
            p.extend(block.parameters());
        }
        for block in &self.stage3 {
            p.extend(block.parameters());
        }
        p
    }

    pub fn eval(&mut self) {
        for b in &mut self.stage1 { b.eval(); }
        for b in &mut self.stage2 { b.eval(); }
        for b in &mut self.stage3 { b.eval(); }
    }

    pub fn train(&mut self) {
        for b in &mut self.stage1 { b.train(); }
        for b in &mut self.stage2 { b.train(); }
        for b in &mut self.stage3 { b.train(); }
    }
}

impl Default for VentralPathway {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Dorsal Pathway ("Where" Stream)
// =============================================================================

/// Dorsal pathway: wide-receptive-field spatial features.
///
/// Uses 5×5 convolutions with fewer channels for efficient spatial processing.
/// - D1: [B, 48, H/8, W/8]
/// - D2: [B, 64, H/16, W/16]
/// - D3: [B, 96, H/32, W/32]
pub struct DorsalPathway {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    relu: ReLU,
}

impl DorsalPathway {
    /// Create dorsal pathway from 64-channel stem output.
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::with_options(64, 48, (5, 5), (2, 2), (2, 2), true),
            bn1: BatchNorm2d::new(48),
            conv2: Conv2d::with_options(48, 64, (5, 5), (2, 2), (2, 2), true),
            bn2: BatchNorm2d::new(64),
            conv3: Conv2d::with_options(64, 96, (5, 5), (2, 2), (2, 2), true),
            bn3: BatchNorm2d::new(96),
            relu: ReLU,
        }
    }

    /// Forward: [B, 64, H/4, W/4] → (D1, D2, D3).
    pub fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let d1 = self.relu.forward(&self.bn1.forward(&self.conv1.forward(x)));
        let d2 = self.relu.forward(&self.bn2.forward(&self.conv2.forward(&d1)));
        let d3 = self.relu.forward(&self.bn3.forward(&self.conv3.forward(&d2)));
        (d1, d2, d3)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.bn3.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        self.bn3.eval();
    }

    pub fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        self.bn3.train();
    }
}

impl Default for DorsalPathway {
    fn default() -> Self {
        Self::new()
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
    fn test_shared_stem() {
        let stem = SharedStem::new();
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 320 * 320], &[1, 3, 320, 320]).unwrap(),
            false,
        );
        let out = stem.forward(&x);
        assert_eq!(out.shape(), vec![1, 64, 80, 80]);
    }

    #[test]
    fn test_inverted_residual_same() {
        let block = InvertedResidualBlock::new(96, 96, 1, 2);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 96 * 8 * 8], &[1, 96, 8, 8]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.shape(), vec![1, 96, 8, 8]);
    }

    #[test]
    fn test_inverted_residual_downsample() {
        let block = InvertedResidualBlock::new(64, 96, 2, 2);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 16 * 16], &[1, 64, 16, 16]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.shape(), vec![1, 96, 8, 8]);
    }

    #[test]
    fn test_ventral_pathway() {
        let ventral = VentralPathway::new();
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 80 * 80], &[1, 64, 80, 80]).unwrap(),
            false,
        );
        let (v1, v2, v3) = ventral.forward(&x);
        assert_eq!(v1.shape(), vec![1, 96, 40, 40]);
        assert_eq!(v2.shape(), vec![1, 128, 20, 20]);
        assert_eq!(v3.shape(), vec![1, 192, 10, 10]);
    }

    #[test]
    fn test_dorsal_pathway() {
        let dorsal = DorsalPathway::new();
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 80 * 80], &[1, 64, 80, 80]).unwrap(),
            false,
        );
        let (d1, d2, d3) = dorsal.forward(&x);
        assert_eq!(d1.shape(), vec![1, 48, 40, 40]);
        assert_eq!(d2.shape(), vec![1, 64, 20, 20]);
        assert_eq!(d3.shape(), vec![1, 96, 10, 10]);
    }

    #[test]
    fn test_stem_param_count() {
        let stem = SharedStem::new();
        let total: usize = stem.parameters().iter().map(|p| p.numel()).sum();
        assert!(total > 1000);
        assert!(total < 50_000);
    }

    #[test]
    fn test_ventral_param_count() {
        let ventral = VentralPathway::new();
        let total: usize = ventral.parameters().iter().map(|p| p.numel()).sum();
        assert!(total > 50_000);
        assert!(total < 500_000);
    }

    #[test]
    fn test_dorsal_param_count() {
        let dorsal = DorsalPathway::new();
        let total: usize = dorsal.parameters().iter().map(|p| p.numel()).sum();
        // Dorsal uses wide convs with fewer channels — should be smaller than ventral
        assert!(total > 10_000);
        assert!(total < 350_000);
    }
}
