//! Phantom Backbone — BlazeBlock-based Lightweight Feature Extractor
//!
//! # File
//! `crates/axonml-vision/src/models/phantom/backbone.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Module, Parameter, ReLU};

// =============================================================================
// BlazeBlock (Depthwise Separable with Residual)
// =============================================================================

/// Depthwise separable convolution with residual connection.
/// Based on BlazeFace architecture, optimized for mobile face detection.
struct PhantomBlazeBlock {
    dw_conv: Conv2d,
    dw_bn: BatchNorm2d,
    pw_conv: Conv2d,
    pw_bn: BatchNorm2d,
    project: Option<(Conv2d, BatchNorm2d)>,
    relu: ReLU,
    _stride: usize,
}

impl PhantomBlazeBlock {
    fn new(in_ch: usize, out_ch: usize, stride: usize) -> Self {
        let dw_conv =
            Conv2d::with_groups(in_ch, in_ch, (3, 3), (stride, stride), (1, 1), true, in_ch);
        let dw_bn = BatchNorm2d::new(in_ch);
        let pw_conv = Conv2d::with_options(in_ch, out_ch, (1, 1), (1, 1), (0, 0), true);
        let pw_bn = BatchNorm2d::new(out_ch);

        let project = if in_ch != out_ch || stride != 1 {
            Some((
                Conv2d::with_options(in_ch, out_ch, (1, 1), (stride, stride), (0, 0), true),
                BatchNorm2d::new(out_ch),
            ))
        } else {
            None
        };

        Self {
            dw_conv,
            dw_bn,
            pw_conv,
            pw_bn,
            project,
            relu: ReLU,
            _stride: stride,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let identity = if let Some((ref proj_conv, ref proj_bn)) = self.project {
            proj_bn.forward(&proj_conv.forward(x))
        } else {
            x.clone()
        };

        let out = self
            .relu
            .forward(&self.dw_bn.forward(&self.dw_conv.forward(x)));
        let out = self.pw_bn.forward(&self.pw_conv.forward(&out));
        self.relu.forward(&out.add_var(&identity))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.dw_conv.parameters());
        params.extend(self.dw_bn.parameters());
        params.extend(self.pw_conv.parameters());
        params.extend(self.pw_bn.parameters());
        if let Some((ref c, ref bn)) = self.project {
            params.extend(c.parameters());
            params.extend(bn.parameters());
        }
        params
    }

    fn train(&mut self) {
        self.dw_bn.train();
        self.pw_bn.train();
        if let Some((_, ref mut bn)) = self.project {
            bn.train();
        }
    }

    fn eval(&mut self) {
        self.dw_bn.eval();
        self.pw_bn.eval();
        if let Some((_, ref mut bn)) = self.project {
            bn.eval();
        }
    }
}

// =============================================================================
// Event Feature Extractor
// =============================================================================

/// Lightweight feature extractor for event-active patches.
///
/// Takes a 48×48 patch (RGB + motion = 4 channels) and produces
/// a compact [32]-dimensional feature vector.
pub struct EventFeatureExtractor {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    relu: ReLU,
}

impl Default for EventFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl EventFeatureExtractor {
    /// Create event feature extractor: 4ch input → 32-dim output.
    pub fn new() -> Self {
        // Depthwise separable blocks for efficiency
        let conv1 = Conv2d::with_options(4, 16, (3, 3), (2, 2), (1, 1), true);
        let bn1 = BatchNorm2d::new(16);
        let conv2 = Conv2d::with_options(16, 32, (3, 3), (2, 2), (1, 1), true);
        let bn2 = BatchNorm2d::new(32);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            relu: ReLU,
        }
    }

    /// Forward: [B, 4, 48, 48] → [B, 32].
    pub fn forward(&self, x: &Variable) -> Variable {
        let out = self.relu.forward(&self.bn1.forward(&self.conv1.forward(x)));
        let out = self
            .relu
            .forward(&self.bn2.forward(&self.conv2.forward(&out)));

        // Global average pooling → [B, 32]
        let shape = out.shape();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // Global average pooling: [B, C, H, W] -> [B, C, H*W] -> mean(dim=2) -> [B, C]
        out.reshape(&[b, c, h * w]).mean_dim(2, false)
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p
    }

    /// Set to eval mode.
    pub fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
    }

    /// Set to train mode.
    pub fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
    }
}

// =============================================================================
// Phantom Backbone
// =============================================================================

/// Lightweight BlazeBlock-based backbone for Phantom face detection.
///
/// Architecture:
/// - Stem: Conv(3→16, 3×3, stride=2) + BN + ReLU → [B, 16, H/2, W/2]
/// - Stage 1: 2× BlazeBlock(16→24, s=1 then 24→24, s=1) → P1
/// - Stage 2: BlazeBlock(24→32, s=2) + BlazeBlock(32→32, s=1) → P2
/// - Stage 3: BlazeBlock(32→48, s=2) + BlazeBlock(48→48, s=1) → P3
///
/// Returns multi-scale features [P1, P2, P3] for anchor-free detection.
pub struct PhantomBackbone {
    stem_conv: Conv2d,
    stem_bn: BatchNorm2d,
    stage1: Vec<PhantomBlazeBlock>,
    stage2: Vec<PhantomBlazeBlock>,
    stage3: Vec<PhantomBlazeBlock>,
    relu: ReLU,
    /// Cached features from last full backbone run.
    cached_features: Option<Vec<Variable>>,
    /// Frame counter for periodic full backbone refresh.
    frame_count: u32,
    /// How often to run full backbone (in frames).
    pub refresh_interval: u32,
}

impl PhantomBackbone {
    /// Create a new Phantom backbone.
    pub fn new() -> Self {
        let stem_conv = Conv2d::with_options(3, 16, (3, 3), (2, 2), (1, 1), true);
        let stem_bn = BatchNorm2d::new(16);

        let stage1 = vec![
            PhantomBlazeBlock::new(16, 24, 1),
            PhantomBlazeBlock::new(24, 24, 1),
        ];

        let stage2 = vec![
            PhantomBlazeBlock::new(24, 32, 2),
            PhantomBlazeBlock::new(32, 32, 1),
        ];

        let stage3 = vec![
            PhantomBlazeBlock::new(32, 48, 2),
            PhantomBlazeBlock::new(48, 48, 1),
        ];

        Self {
            stem_conv,
            stem_bn,
            stage1,
            stage2,
            stage3,
            relu: ReLU,
            cached_features: None,
            frame_count: 0,
            refresh_interval: 30,
        }
    }

    /// Run full backbone forward pass and cache features.
    ///
    /// Input: [B, 3, H, W]
    /// Returns: [P1=[B,24,H/2,W/2], P2=[B,32,H/4,W/4], P3=[B,48,H/8,W/8]]
    pub fn forward_full(&mut self, x: &Variable) -> Vec<Variable> {
        let mut out = self
            .relu
            .forward(&self.stem_bn.forward(&self.stem_conv.forward(x)));

        for block in &self.stage1 {
            out = block.forward(&out);
        }
        let p1 = out.clone();

        for block in &self.stage2 {
            out = block.forward(&out);
        }
        let p2 = out.clone();

        for block in &self.stage3 {
            out = block.forward(&out);
        }
        let p3 = out;

        let features = vec![p1, p2, p3];
        self.cached_features = Some(features.clone());
        self.frame_count = 0;
        features
    }

    /// Get cached features, or run full backbone if cache is stale.
    ///
    /// Returns (features, was_full_run).
    pub fn get_features(&mut self, frame: &Variable) -> (Vec<Variable>, bool) {
        self.frame_count += 1;

        if self.cached_features.is_none() || self.frame_count >= self.refresh_interval {
            let features = self.forward_full(frame);
            (features, true)
        } else {
            (self.cached_features.clone().unwrap(), false)
        }
    }

    /// Force a cache refresh on next call.
    pub fn invalidate_cache(&mut self) {
        self.cached_features = None;
    }

    /// Check if cached features are available.
    pub fn has_cache(&self) -> bool {
        self.cached_features.is_some()
    }

    /// Get parameters for optimization.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.stem_conv.parameters());
        params.extend(self.stem_bn.parameters());
        for block in &self.stage1 {
            params.extend(block.parameters());
        }
        for block in &self.stage2 {
            params.extend(block.parameters());
        }
        for block in &self.stage3 {
            params.extend(block.parameters());
        }
        params
    }

    /// Set to eval mode.
    pub fn eval(&mut self) {
        self.stem_bn.eval();
        for block in &mut self.stage1 {
            block.eval();
        }
        for block in &mut self.stage2 {
            block.eval();
        }
        for block in &mut self.stage3 {
            block.eval();
        }
    }

    /// Set to train mode.
    pub fn train(&mut self) {
        self.stem_bn.train();
        for block in &mut self.stage1 {
            block.train();
        }
        for block in &mut self.stage2 {
            block.train();
        }
        for block in &mut self.stage3 {
            block.train();
        }
    }
}

impl Default for PhantomBackbone {
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
    fn test_blaze_block_same_channels() {
        let block = PhantomBlazeBlock::new(24, 24, 1);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 24 * 16 * 16], &[1, 24, 16, 16]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.shape(), vec![1, 24, 16, 16]);
    }

    #[test]
    fn test_blaze_block_downsample() {
        let block = PhantomBlazeBlock::new(24, 32, 2);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 24 * 16 * 16], &[1, 24, 16, 16]).unwrap(),
            false,
        );
        let out = block.forward(&x);
        assert_eq!(out.shape(), vec![1, 32, 8, 8]);
    }

    #[test]
    fn test_event_feature_extractor() {
        let efe = EventFeatureExtractor::new();
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 4 * 48 * 48], &[1, 4, 48, 48]).unwrap(),
            false,
        );
        let out = efe.forward(&x);
        assert_eq!(out.shape(), vec![1, 32]);
    }

    #[test]
    fn test_phantom_backbone_forward() {
        let mut backbone = PhantomBackbone::new();
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        let features = backbone.forward_full(&x);

        assert_eq!(features.len(), 3);
        assert_eq!(features[0].shape(), vec![1, 24, 64, 64]); // P1
        assert_eq!(features[1].shape(), vec![1, 32, 32, 32]); // P2
        assert_eq!(features[2].shape(), vec![1, 48, 16, 16]); // P3
    }

    #[test]
    fn test_phantom_backbone_caching() {
        let mut backbone = PhantomBackbone::new();
        backbone.refresh_interval = 5;

        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        // First call: full run
        let (_, was_full) = backbone.get_features(&x);
        assert!(was_full);

        // Second call: cached
        let (_, was_full) = backbone.get_features(&x);
        assert!(!was_full);

        assert!(backbone.has_cache());
    }

    #[test]
    fn test_phantom_backbone_param_count() {
        let backbone = PhantomBackbone::new();
        let total: usize = backbone.parameters().iter().map(|p| p.numel()).sum();
        // Should be lightweight (~20-30K params for backbone alone)
        assert!(total < 100_000, "Backbone too large: {total} params");
        assert!(total > 5_000, "Backbone too small: {total} params");
    }
}
