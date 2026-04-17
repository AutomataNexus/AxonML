//! Pathway Fusion — Attention-Gated Cross-Pathway Merge
//!
//! Implements `MultiScaleFusion` which merges detail and context pathway features
//! at each scale level using learned attention gates. Attention weights are
//! computed from concatenated features via Conv2d + BatchNorm + sigmoid, then
//! applied element-wise to blend the two pathways. Produces fused multi-scale
//! feature maps for the detection heads.
//!
//! # File
//! `crates/axonml-vision/src/models/nexus/fusion.rs`
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

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// Pathway Fusion
// =============================================================================

/// Attention-gated fusion of ventral and dorsal pathway features.
///
/// At each scale, produces a single fused feature map by learning
/// a spatial gate that blends identity and spatial features.
pub struct PathwayFusion {
    /// Gate convolution: concatenated input → gate map.
    gate_conv: Conv2d,
    gate_bn: BatchNorm2d,
    /// Output projection: fused features → unified channels.
    out_conv: Conv2d,
    out_bn: BatchNorm2d,
    /// Output channels.
    _out_channels: usize,
}

impl PathwayFusion {
    /// Create a pathway fusion module.
    ///
    /// - `ventral_ch`: Ventral pathway channels at this scale.
    /// - `dorsal_ch`: Dorsal pathway channels at this scale.
    /// - `out_ch`: Output channels after fusion.
    pub fn new(ventral_ch: usize, dorsal_ch: usize, out_ch: usize) -> Self {
        let total_in = ventral_ch + dorsal_ch;
        Self {
            gate_conv: Conv2d::with_options(total_in, 1, (1, 1), (1, 1), (0, 0), true),
            gate_bn: BatchNorm2d::new(1),
            out_conv: Conv2d::with_options(total_in, out_ch, (1, 1), (1, 1), (0, 0), true),
            out_bn: BatchNorm2d::new(out_ch),
            _out_channels: out_ch,
        }
    }

    /// Fuse ventral and dorsal features.
    ///
    /// # Arguments
    /// - `ventral`: Ventral features [B, V_ch, H, W]
    /// - `dorsal`: Dorsal features [B, D_ch, H, W] (same spatial dims)
    ///
    /// # Returns
    /// Fused features [B, out_ch, H, W]
    pub fn forward(&self, ventral: &Variable, dorsal: &Variable) -> Variable {
        let v_shape = ventral.shape();
        let d_shape = dorsal.shape();
        let (b, v_ch) = (v_shape[0], v_shape[1]);
        let d_ch = d_shape[1];
        let (h, w) = (v_shape[2], v_shape[3]);

        // Concatenate along channel dim (graph-tracked)
        let concatenated = Variable::cat(&[ventral, dorsal], 1);

        // Compute gate: σ(Conv(cat))
        let gate_raw = self.gate_bn.forward(&self.gate_conv.forward(&concatenated));
        let gate = gate_raw.sigmoid(); // [B, 1, H, W]

        // Gated blend using graph-tracked ops: gate * V + (1 - gate) * D
        // gate broadcasts over channels via expand
        let gate_v = gate.expand(&[b, v_ch, h, w]);
        let gate_d = gate.expand(&[b, d_ch, h, w]);
        let one = Variable::new(Tensor::ones(&[1]), false);
        let inv_gate_d = one.sub_var(&gate_d);
        let gated_v = ventral.mul_var(&gate_v);
        let gated_d = dorsal.mul_var(&inv_gate_d);
        let blended_var = Variable::cat(&[&gated_v, &gated_d], 1);

        // Project to output channels
        let out = self.out_bn.forward(&self.out_conv.forward(&blended_var));
        out.relu()
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.gate_conv.parameters());
        p.extend(self.gate_bn.parameters());
        p.extend(self.out_conv.parameters());
        p.extend(self.out_bn.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.gate_bn.eval();
        self.out_bn.eval();
    }

    pub fn train(&mut self) {
        self.gate_bn.train();
        self.out_bn.train();
    }
}

// =============================================================================
// Multi-Scale Fusion
// =============================================================================

/// Multi-scale pathway fusion across all 3 pyramid levels.
pub struct MultiScaleFusion {
    /// Fusion module for scale 1 (V=96, D=48 → 96).
    pub scale1: PathwayFusion,
    /// Fusion module for scale 2 (V=128, D=64 → 96).
    pub scale2: PathwayFusion,
    /// Fusion module for scale 3 (V=192, D=96 → 96).
    pub scale3: PathwayFusion,
}

impl MultiScaleFusion {
    /// Create fusion modules with standard Nexus channel counts.
    pub fn new() -> Self {
        Self {
            scale1: PathwayFusion::new(96, 48, 96),
            scale2: PathwayFusion::new(128, 64, 96),
            scale3: PathwayFusion::new(192, 96, 96),
        }
    }

    /// Fuse all three scales.
    ///
    /// Returns (F1=[B,96,H1,W1], F2=[B,96,H2,W2], F3=[B,96,H3,W3]).
    pub fn forward(
        &self,
        ventral: (&Variable, &Variable, &Variable),
        dorsal: (&Variable, &Variable, &Variable),
    ) -> (Variable, Variable, Variable) {
        let f1 = self.scale1.forward(ventral.0, dorsal.0);
        let f2 = self.scale2.forward(ventral.1, dorsal.1);
        let f3 = self.scale3.forward(ventral.2, dorsal.2);
        (f1, f2, f3)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.scale1.parameters());
        p.extend(self.scale2.parameters());
        p.extend(self.scale3.parameters());
        p
    }

    pub fn eval(&mut self) {
        self.scale1.eval();
        self.scale2.eval();
        self.scale3.eval();
    }

    pub fn train(&mut self) {
        self.scale1.train();
        self.scale2.train();
        self.scale3.train();
    }
}

impl Default for MultiScaleFusion {
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

    #[test]
    fn test_pathway_fusion_shapes() {
        let fusion = PathwayFusion::new(96, 48, 96);
        let v = Variable::new(
            Tensor::from_vec(vec![0.1; 96 * 10 * 10], &[1, 96, 10, 10]).unwrap(),
            false,
        );
        let d = Variable::new(
            Tensor::from_vec(vec![0.1; 48 * 10 * 10], &[1, 48, 10, 10]).unwrap(),
            false,
        );
        let out = fusion.forward(&v, &d);
        assert_eq!(out.shape(), vec![1, 96, 10, 10]);
    }

    #[test]
    fn test_multi_scale_fusion() {
        let fusion = MultiScaleFusion::new();

        let v1 = Variable::new(
            Tensor::from_vec(vec![0.1; 96 * 40 * 40], &[1, 96, 40, 40]).unwrap(),
            false,
        );
        let v2 = Variable::new(
            Tensor::from_vec(vec![0.1; 128 * 20 * 20], &[1, 128, 20, 20]).unwrap(),
            false,
        );
        let v3 = Variable::new(
            Tensor::from_vec(vec![0.1; 192 * 10 * 10], &[1, 192, 10, 10]).unwrap(),
            false,
        );

        let d1 = Variable::new(
            Tensor::from_vec(vec![0.1; 48 * 40 * 40], &[1, 48, 40, 40]).unwrap(),
            false,
        );
        let d2 = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 20 * 20], &[1, 64, 20, 20]).unwrap(),
            false,
        );
        let d3 = Variable::new(
            Tensor::from_vec(vec![0.1; 96 * 10 * 10], &[1, 96, 10, 10]).unwrap(),
            false,
        );

        let (f1, f2, f3) = fusion.forward((&v1, &v2, &v3), (&d1, &d2, &d3));
        assert_eq!(f1.shape(), vec![1, 96, 40, 40]);
        assert_eq!(f2.shape(), vec![1, 96, 20, 20]);
        assert_eq!(f3.shape(), vec![1, 96, 10, 10]);
    }

    #[test]
    fn test_fusion_gate_bounded() {
        let fusion = PathwayFusion::new(16, 8, 16);
        let v = Variable::new(
            Tensor::from_vec(vec![1.0; 16 * 4 * 4], &[1, 16, 4, 4]).unwrap(),
            false,
        );
        let d = Variable::new(
            Tensor::from_vec(vec![0.5; 8 * 4 * 4], &[1, 8, 4, 4]).unwrap(),
            false,
        );
        let out = fusion.forward(&v, &d);

        // Output should be finite
        let data = out.data().to_vec();
        assert!(data.iter().all(|v| v.is_finite()));
    }
}
