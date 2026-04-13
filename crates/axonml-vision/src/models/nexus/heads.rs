//! Nexus Detection Heads — Proposals, Classification, Uncertainty
//!
//! # File
//! `crates/axonml-vision/src/models/nexus/heads.rs`
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

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Linear, Module, Parameter};

// =============================================================================
// Proposal Head (Anchor-Free)
// =============================================================================

/// Anchor-free proposal head per spatial location.
///
/// For each location in the feature map, predicts:
/// - Classification score (object vs background)
/// - Bounding box (center offset + size)
/// - Centerness (down-weight proposals far from object center)
pub struct ProposalHead {
    conv: Conv2d,
    bn: BatchNorm2d,
    cls_conv: Conv2d,
    bbox_conv: Conv2d,
    center_conv: Conv2d,
}

impl ProposalHead {
    /// Create a proposal head for the given feature channels.
    pub fn new(in_channels: usize) -> Self {
        Self {
            conv: Conv2d::with_options(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true),
            bn: BatchNorm2d::new(in_channels),
            cls_conv: Conv2d::with_options(in_channels, 1, (1, 1), (1, 1), (0, 0), true),
            bbox_conv: Conv2d::with_options(in_channels, 4, (1, 1), (1, 1), (0, 0), true),
            center_conv: Conv2d::with_options(in_channels, 1, (1, 1), (1, 1), (0, 0), true),
        }
    }

    /// Forward: [B, C, H, W] → (cls [B,1,H,W], bbox [B,4,H,W], center [B,1,H,W]).
    pub fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let feat = self.bn.forward(&self.conv.forward(x)).relu();
        let cls = self.cls_conv.forward(&feat);
        let bbox = self.bbox_conv.forward(&feat);
        let center = self.center_conv.forward(&feat);
        (cls, bbox, center)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv.parameters());
        p.extend(self.bn.parameters());
        p.extend(self.cls_conv.parameters());
        p.extend(self.bbox_conv.parameters());
        p.extend(self.center_conv.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.bn.eval();
    }

    pub fn train(&mut self) {
        self.bn.train();
    }
}

// =============================================================================
// Classification Head
// =============================================================================

/// Classification head: predicts class from GRU hidden state + ventral ROI.
///
/// Input: concatenated GRU state [hidden_size] + ventral ROI [roi_dim]
/// Output: class logits [num_classes]
pub struct ClassHead {
    fc1: Linear,
    fc2: Linear,
}

impl ClassHead {
    /// Create a classification head.
    ///
    /// - `hidden_size`: GRU state dimension.
    /// - `roi_dim`: Flattened ventral ROI feature dimension.
    /// - `num_classes`: Number of object classes.
    pub fn new(hidden_size: usize, roi_dim: usize, num_classes: usize) -> Self {
        Self {
            fc1: Linear::new(hidden_size + roi_dim, 128),
            fc2: Linear::new(128, num_classes),
        }
    }

    /// Forward: [B, hidden_size + roi_dim] → [B, num_classes].
    pub fn forward(&self, x: &Variable) -> Variable {
        let h = self.fc1.forward(x).relu();
        self.fc2.forward(&h)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.fc1.parameters());
        p.extend(self.fc2.parameters());
        p
    }
}

// =============================================================================
// Uncertainty Bounding Box Head
// =============================================================================

/// Uncertainty-aware bounding box prediction head.
///
/// Outputs both mean and log-variance for each bbox coordinate,
/// enabling aleatoric uncertainty quantification.
///
/// For each object:
/// - bbox_mean [4]: predicted (x1, y1, x2, y2)
/// - bbox_log_var [4]: log-variance of each coordinate
///
/// During inference, variance = exp(log_var) indicates localization confidence.
pub struct UncertaintyBBoxHead {
    shared: Linear,
    mean_head: Linear,
    logvar_head: Linear,
}

impl UncertaintyBBoxHead {
    /// Create with given hidden dimension.
    pub fn new(hidden_size: usize) -> Self {
        Self {
            shared: Linear::new(hidden_size, 64),
            mean_head: Linear::new(64, 4),
            logvar_head: Linear::new(64, 4),
        }
    }

    /// Forward: [B, hidden_size] → (mean [B, 4], log_var [B, 4]).
    pub fn forward(&self, x: &Variable) -> (Variable, Variable) {
        let h = self.shared.forward(x).relu();
        let mean = self.mean_head.forward(&h);
        let log_var = self.logvar_head.forward(&h);
        (mean, log_var)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.shared.parameters());
        p.extend(self.mean_head.parameters());
        p.extend(self.logvar_head.parameters());
        p
    }
}

// =============================================================================
// Temporal Predictor
// =============================================================================

/// Temporal predictor: generates predicted features for next frame.
///
/// Uses current fused features to predict what the next frame will look like.
/// This prediction feeds into the predictive coding module.
pub struct TemporalPredictor {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
}

impl TemporalPredictor {
    /// Create a temporal predictor for the given feature channels.
    pub fn new(channels: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
            bn1: BatchNorm2d::new(channels),
            conv2: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
        }
    }

    /// Predict next frame's features: [B, C, H, W] → [B, C, H, W].
    pub fn forward(&self, x: &Variable) -> Variable {
        let h = self.bn1.forward(&self.conv1.forward(x)).relu();
        self.conv2.forward(&h)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.bn1.eval();
    }

    pub fn train(&mut self) {
        self.bn1.train();
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
    fn test_proposal_head() {
        let head = ProposalHead::new(96);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 96 * 10 * 10], &[1, 96, 10, 10]).unwrap(),
            false,
        );
        let (cls, bbox, center) = head.forward(&x);
        assert_eq!(cls.shape(), vec![1, 1, 10, 10]);
        assert_eq!(bbox.shape(), vec![1, 4, 10, 10]);
        assert_eq!(center.shape(), vec![1, 1, 10, 10]);
    }

    #[test]
    fn test_class_head() {
        let head = ClassHead::new(64, 288, 20);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 64 + 288], &[1, 352]).unwrap(),
            false,
        );
        let out = head.forward(&x);
        assert_eq!(out.shape(), vec![1, 20]);
    }

    #[test]
    fn test_uncertainty_bbox_head() {
        let head = UncertaintyBBoxHead::new(64);
        let x = Variable::new(Tensor::from_vec(vec![0.1; 64], &[1, 64]).unwrap(), false);
        let (mean, log_var) = head.forward(&x);
        assert_eq!(mean.shape(), vec![1, 4]);
        assert_eq!(log_var.shape(), vec![1, 4]);

        // log_var should be finite
        assert!(log_var.data().to_vec().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_temporal_predictor() {
        let pred = TemporalPredictor::new(96);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 96 * 10 * 10], &[1, 96, 10, 10]).unwrap(),
            false,
        );
        let out = pred.forward(&x);
        assert_eq!(out.shape(), vec![1, 96, 10, 10]);
    }

    #[test]
    fn test_proposal_head_params() {
        let head = ProposalHead::new(96);
        let total: usize = head.parameters().iter().map(|p| p.numel()).sum();
        assert!(total > 5_000);
        assert!(total < 200_000);
    }
}
