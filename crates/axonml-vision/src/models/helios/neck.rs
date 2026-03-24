//! PANet Neck for Helios
//!
//! # File
//! `crates/axonml-vision/src/models/helios/neck.rs`
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
use axonml_nn::Parameter;

use super::backbone::{C2f, CBS};
use super::HeliosConfig;
use crate::ops::InterpolateMode;

// =============================================================================
// PANet
// =============================================================================

/// PANet: bidirectional feature pyramid with C2f fusion blocks.
pub struct PANet {
    // Top-down path (P5 -> P4 -> P3)
    lateral_p5: CBS,
    td_c2f_p4: C2f,
    lateral_p4: CBS,
    td_c2f_p3: C2f,

    // Bottom-up path (P3 -> P4 -> P5)
    bu_down_p3: CBS,
    bu_c2f_p4: C2f,
    bu_down_p4: CBS,
    bu_c2f_p5: C2f,

    /// Output channels for [P3, P4, P5].
    pub out_channels: [usize; 3],
}

impl PANet {
    pub fn new(in_channels: [usize; 3], config: &HeliosConfig) -> Self {
        let [ch3, ch4, ch5] = in_channels;
        let depths = config.stage_depths();
        let neck_depth = depths[0].max(1);

        // Top-down: P5 -> P4 -> P3
        let lateral_p5 = CBS::pointwise(ch5, ch4);
        // After concat(up(P5'), P4): ch4 + ch4 = 2*ch4
        let td_c2f_p4 = C2f::new(2 * ch4, ch4, neck_depth, false);
        let lateral_p4 = CBS::pointwise(ch4, ch3);
        // After concat(up(P4'), P3): ch3 + ch3 = 2*ch3
        let td_c2f_p3 = C2f::new(2 * ch3, ch3, neck_depth, false);

        // Bottom-up: P3 -> P4 -> P5
        let bu_down_p3 = CBS::conv3x3(ch3, ch3, 2);
        // After concat(down(P3'), P4_td): ch3 + ch4
        let bu_c2f_p4 = C2f::new(ch3 + ch4, ch4, neck_depth, false);
        let bu_down_p4 = CBS::conv3x3(ch4, ch4, 2);
        // After concat(down(P4'), P5_lat): ch4 + ch4
        let bu_c2f_p5 = C2f::new(ch4 + ch4, ch5, neck_depth, false);

        Self {
            lateral_p5,
            td_c2f_p4,
            lateral_p4,
            td_c2f_p3,
            bu_down_p3,
            bu_c2f_p4,
            bu_down_p4,
            bu_c2f_p5,
            out_channels: [ch3, ch4, ch5],
        }
    }

    /// Forward pass: (P3, P4, P5) from backbone -> fused (P3', P4', P5').
    pub fn forward(
        &self,
        p3: &Variable,
        p4: &Variable,
        p5: &Variable,
    ) -> (Variable, Variable, Variable) {
        let p4_shape = p4.shape();
        let p3_shape = p3.shape();

        // Top-down: P5 -> P4
        let p5_lat = self.lateral_p5.forward(p5);
        let p5_up = crate::ops::interpolate_var(
            &p5_lat,
            p4_shape[2],
            p4_shape[3],
            InterpolateMode::Nearest,
        );
        let p4_cat = Variable::cat(&[&p5_up, p4], 1);
        let p4_td = self.td_c2f_p4.forward(&p4_cat);

        // Top-down: P4 -> P3
        let p4_lat = self.lateral_p4.forward(&p4_td);
        let p4_up = crate::ops::interpolate_var(
            &p4_lat,
            p3_shape[2],
            p3_shape[3],
            InterpolateMode::Nearest,
        );
        let p3_cat = Variable::cat(&[&p4_up, p3], 1);
        let p3_td = self.td_c2f_p3.forward(&p3_cat);

        // Bottom-up: P3 -> P4
        let p3_down = self.bu_down_p3.forward(&p3_td);
        let p4_cat2 = Variable::cat(&[&p3_down, &p4_td], 1);
        let p4_bu = self.bu_c2f_p4.forward(&p4_cat2);

        // Bottom-up: P4 -> P5
        let p4_down = self.bu_down_p4.forward(&p4_bu);
        let p5_cat2 = Variable::cat(&[&p4_down, &p5_lat], 1);
        let p5_bu = self.bu_c2f_p5.forward(&p5_cat2);

        (p3_td, p4_bu, p5_bu)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.lateral_p5.parameters());
        p.extend(self.td_c2f_p4.parameters());
        p.extend(self.lateral_p4.parameters());
        p.extend(self.td_c2f_p3.parameters());
        p.extend(self.bu_down_p3.parameters());
        p.extend(self.bu_c2f_p4.parameters());
        p.extend(self.bu_down_p4.parameters());
        p.extend(self.bu_c2f_p5.parameters());
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
    fn test_panet_nano() {
        let cfg = HeliosConfig::nano(80);
        let backbone = super::super::backbone::CSPDarknet::new(&cfg);
        let neck = PANet::new(backbone.out_channels, &cfg);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        let (p3, p4, p5) = backbone.forward(&input);
        let (n3, n4, n5) = neck.forward(&p3, &p4, &p5);

        // Should preserve spatial dimensions and output channels
        assert_eq!(n3.shape()[1], 64); // ch[2]
        assert_eq!(n3.shape()[2], 8); // 64/8
        assert_eq!(n4.shape()[1], 128); // ch[3]
        assert_eq!(n4.shape()[2], 4); // 64/16
        assert_eq!(n5.shape()[1], 256); // ch[4]
        assert_eq!(n5.shape()[2], 2); // 64/32
    }
}
