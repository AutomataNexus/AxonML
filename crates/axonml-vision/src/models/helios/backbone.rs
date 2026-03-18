//! CSPDarknet Backbone for Helios
//!
//! # File
//! `crates/axonml-vision/src/models/helios/backbone.rs`
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
use axonml_nn::{BatchNorm2d, Conv2d, MaxPool2d, Module, Parameter};

use super::HeliosConfig;

// =============================================================================
// CBS — Conv + BatchNorm + SiLU
// =============================================================================

/// Conv-BN-SiLU fundamental building block.
pub struct CBS {
    conv: Conv2d,
    bn: BatchNorm2d,
}

impl CBS {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            conv: Conv2d::with_options(in_ch, out_ch, kernel, stride, padding, false),
            bn: BatchNorm2d::new(out_ch),
        }
    }

    /// 1x1 convolution.
    pub fn pointwise(in_ch: usize, out_ch: usize) -> Self {
        Self::new(in_ch, out_ch, (1, 1), (1, 1), (0, 0))
    }

    /// 3x3 convolution with stride and padding=1.
    pub fn conv3x3(in_ch: usize, out_ch: usize, stride: usize) -> Self {
        Self::new(in_ch, out_ch, (3, 3), (stride, stride), (1, 1))
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        self.bn.forward(&self.conv.forward(x)).silu()
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.conv.parameters();
        p.extend(self.bn.parameters());
        p
    }
}

// =============================================================================
// Bottleneck
// =============================================================================

/// Bottleneck block with optional residual connection.
struct Bottleneck {
    cv1: CBS,
    cv2: CBS,
    use_residual: bool,
}

impl Bottleneck {
    fn new(in_ch: usize, out_ch: usize, shortcut: bool) -> Self {
        Self {
            cv1: CBS::conv3x3(in_ch, out_ch, 1),
            cv2: CBS::conv3x3(out_ch, out_ch, 1),
            use_residual: shortcut && in_ch == out_ch,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let out = self.cv2.forward(&self.cv1.forward(x));
        if self.use_residual {
            x.add_var(&out)
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.cv1.parameters();
        p.extend(self.cv2.parameters());
        p
    }
}

// =============================================================================
// C2f — Cross Stage Partial with 2 convolutions
// =============================================================================

/// C2f: CSP Bottleneck with 2 convolutions.
/// Splits channels, processes through bottleneck stack, concatenates.
pub struct C2f {
    cv1: CBS,
    cv2: CBS,
    bottlenecks: Vec<Bottleneck>,
    hidden_ch: usize,
}

impl C2f {
    pub fn new(in_ch: usize, out_ch: usize, num_bottlenecks: usize, shortcut: bool) -> Self {
        let hidden = out_ch;
        // cv1 expands to 2*hidden for split
        let cv1 = CBS::pointwise(in_ch, 2 * hidden);
        // After concat: (num_bottlenecks + 2) * hidden channels
        let concat_ch = (num_bottlenecks + 2) * hidden;
        let cv2 = CBS::pointwise(concat_ch, out_ch);

        let bottlenecks = (0..num_bottlenecks)
            .map(|_| Bottleneck::new(hidden, hidden, shortcut))
            .collect();

        Self {
            cv1,
            cv2,
            bottlenecks,
            hidden_ch: hidden,
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        // Split into two halves
        let y = self.cv1.forward(x);
        let y1 = y.narrow(1, 0, self.hidden_ch);
        let mut y2 = y.narrow(1, self.hidden_ch, self.hidden_ch);

        let mut chunks: Vec<Variable> = vec![y1, y2.clone()];

        for bottleneck in &self.bottlenecks {
            y2 = bottleneck.forward(&y2);
            chunks.push(y2.clone());
        }

        let refs: Vec<&Variable> = chunks.iter().collect();
        let cat = Variable::cat(&refs, 1);
        self.cv2.forward(&cat)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.cv1.parameters();
        p.extend(self.cv2.parameters());
        for b in &self.bottlenecks {
            p.extend(b.parameters());
        }
        p
    }
}

// =============================================================================
// SPPF — Spatial Pyramid Pooling Fast
// =============================================================================

/// SPPF: Sequential 5x5 max pools for multi-scale receptive field.
struct SPPF {
    cv1: CBS,
    cv2: CBS,
    pool: MaxPool2d,
}

impl SPPF {
    fn new(in_ch: usize, out_ch: usize) -> Self {
        let hidden = in_ch / 2;
        Self {
            cv1: CBS::pointwise(in_ch, hidden),
            cv2: CBS::pointwise(hidden * 4, out_ch),
            pool: MaxPool2d::with_options((5, 5), (1, 1), (2, 2)),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let x = self.cv1.forward(x);
        let y1 = self.pool.forward(&x);
        let y2 = self.pool.forward(&y1);
        let y3 = self.pool.forward(&y2);

        let cat = Variable::cat(&[&x, &y1, &y2, &y3], 1);
        self.cv2.forward(&cat)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.cv1.parameters();
        p.extend(self.cv2.parameters());
        p
    }
}

// =============================================================================
// CSPDarknet
// =============================================================================

/// CSPDarknet backbone.
///
/// 5-stage feature extractor with C2f blocks.
/// Returns (P3, P4, P5) at strides (8, 16, 32).
pub struct CSPDarknet {
    stem: CBS,
    stage1_down: CBS,
    stage1_c2f: C2f,
    stage2_down: CBS,
    stage2_c2f: C2f,
    stage3_down: CBS,
    stage3_c2f: C2f,
    stage4_down: CBS,
    stage4_c2f: C2f,
    stage4_sppf: SPPF,
    /// Output channels for [P3, P4, P5].
    pub out_channels: [usize; 3],
}

impl CSPDarknet {
    pub fn new(config: &HeliosConfig) -> Self {
        let ch = config.stage_channels();
        let depths = config.stage_depths();

        Self {
            stem: CBS::conv3x3(3, ch[0], 2),            // /2
            stage1_down: CBS::conv3x3(ch[0], ch[1], 2), // /4
            stage1_c2f: C2f::new(ch[1], ch[1], depths[0], true),
            stage2_down: CBS::conv3x3(ch[1], ch[2], 2), // /8  -> P3
            stage2_c2f: C2f::new(ch[2], ch[2], depths[1], true),
            stage3_down: CBS::conv3x3(ch[2], ch[3], 2), // /16 -> P4
            stage3_c2f: C2f::new(ch[3], ch[3], depths[2], true),
            stage4_down: CBS::conv3x3(ch[3], ch[4], 2), // /32 -> P5
            stage4_c2f: C2f::new(ch[4], ch[4], depths[3], true),
            stage4_sppf: SPPF::new(ch[4], ch[4]),
            out_channels: [ch[2], ch[3], ch[4]],
        }
    }

    /// Forward pass returns (P3, P4, P5).
    pub fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let x = self.stem.forward(x);

        let x = self.stage1_down.forward(&x);
        let x = self.stage1_c2f.forward(&x);

        let x = self.stage2_down.forward(&x);
        let p3 = self.stage2_c2f.forward(&x); // P3: /8

        let x = self.stage3_down.forward(&p3);
        let p4 = self.stage3_c2f.forward(&x); // P4: /16

        let x = self.stage4_down.forward(&p4);
        let x = self.stage4_c2f.forward(&x);
        let p5 = self.stage4_sppf.forward(&x); // P5: /32

        (p3, p4, p5)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        p.extend(self.stage1_down.parameters());
        p.extend(self.stage1_c2f.parameters());
        p.extend(self.stage2_down.parameters());
        p.extend(self.stage2_c2f.parameters());
        p.extend(self.stage3_down.parameters());
        p.extend(self.stage3_c2f.parameters());
        p.extend(self.stage4_down.parameters());
        p.extend(self.stage4_c2f.parameters());
        p.extend(self.stage4_sppf.parameters());
        p
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
    fn test_cbs_forward() {
        let cbs = CBS::conv3x3(3, 16, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let out = cbs.forward(&input);
        assert_eq!(out.shape(), vec![1, 16, 16, 16]);
    }

    #[test]
    fn test_c2f_forward() {
        let c2f = C2f::new(32, 32, 2, true);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 32 * 8 * 8], &[1, 32, 8, 8]).unwrap(),
            false,
        );
        let out = c2f.forward(&input);
        assert_eq!(out.shape(), vec![1, 32, 8, 8]);
    }

    #[test]
    fn test_backbone_nano() {
        let cfg = HeliosConfig::nano(80);
        let backbone = CSPDarknet::new(&cfg);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        let (p3, p4, p5) = backbone.forward(&input);

        // P3 at stride 8: 64/8 = 8
        assert_eq!(p3.shape(), vec![1, 64, 8, 8]);
        // P4 at stride 16: 64/16 = 4
        assert_eq!(p4.shape(), vec![1, 128, 4, 4]);
        // P5 at stride 32: 64/32 = 2
        assert_eq!(p5.shape(), vec![1, 256, 2, 2]);
    }

    #[test]
    fn test_backbone_params() {
        let cfg = HeliosConfig::nano(80);
        let backbone = CSPDarknet::new(&cfg);
        let params = backbone.parameters();
        let total: usize = params
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        assert!(
            total > 10000,
            "Backbone should have significant params, got {total}"
        );
    }
}
