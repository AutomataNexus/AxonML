//! NightVision Neck — Thermal Feature Pyramid Network
//!
//! Top-down + bottom-up FPN for fusing the multi-scale features produced by
//! `ThermalBackbone`. Implements `ThermalFPN`, which takes the P3/P4/P5
//! feature maps from the backbone (default 64/128/256 channels) and projects
//! each through 1×1 lateral `ConvBNSiLU` layers to a common output channel
//! count (default 128). A top-down pass upsamples deeper features (via
//! `interpolate_var` with `InterpolateMode::Nearest`) and fuses them into
//! shallower levels, followed by a bottom-up pass using stride-2
//! `ConvBNSiLU` layers to re-inject high-resolution detail. The end result
//! is a triplet `(fpn3, fpn4, fpn5)` of equal-channel, multi-scale feature
//! maps ready for the detection head.
//!
//! Also provides:
//! - `ThermalFPN::default_config` — the 64/128/256 → 128 default.
//! - `parameters`, `named_parameters`, and `set_training` for optimizer and
//!   checkpoint integration.
//! - Tests verifying output shape and channel count.
//!
//! # File
//! `crates/axonml-vision/src/models/nightvision/neck.rs`
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
use axonml_nn::Parameter;

use super::backbone::ConvBNSiLU;
use crate::ops::{InterpolateMode, interpolate_var};

// =============================================================================
// ThermalFPN — Feature Pyramid Network for IR
// =============================================================================

/// Feature Pyramid Network optimized for thermal imagery.
///
/// Takes P3 (64ch), P4 (128ch), P5 (256ch) from backbone and
/// produces fused multi-scale features all at 128 channels.
pub struct ThermalFPN {
    // Lateral connections (1×1 conv to unify channel count)
    lateral_p5: ConvBNSiLU,
    lateral_p4: ConvBNSiLU,
    lateral_p3: ConvBNSiLU,

    // Top-down fusion convs (smooth after upsampling + add)
    td_p4: ConvBNSiLU,
    td_p3: ConvBNSiLU,

    // Bottom-up fusion convs (stride-2 downsample + add)
    bu_p4: ConvBNSiLU,
    bu_p5: ConvBNSiLU,

    /// Output channels for all FPN levels
    out_channels: usize,
}

impl ThermalFPN {
    /// Create a new ThermalFPN with the given input and output channel sizes.
    pub fn new(p3_ch: usize, p4_ch: usize, p5_ch: usize, out_ch: usize) -> Self {
        Self {
            // Laterals: reduce each level to out_ch
            lateral_p5: ConvBNSiLU::new(p5_ch, out_ch, 1, 1, 0),
            lateral_p4: ConvBNSiLU::new(p4_ch, out_ch, 1, 1, 0),
            lateral_p3: ConvBNSiLU::new(p3_ch, out_ch, 1, 1, 0),

            // Top-down: smooth after upsample + addition
            td_p4: ConvBNSiLU::new(out_ch, out_ch, 3, 1, 1),
            td_p3: ConvBNSiLU::new(out_ch, out_ch, 3, 1, 1),

            // Bottom-up: downsample + addition
            bu_p4: ConvBNSiLU::new(out_ch, out_ch, 3, 2, 1),
            bu_p5: ConvBNSiLU::new(out_ch, out_ch, 3, 2, 1),

            out_channels: out_ch,
        }
    }

    /// Default FPN for backbone channel sizes 64/128/256 → 128 output.
    pub fn default_config() -> Self {
        Self::new(64, 128, 256, 128)
    }

    /// Returns the output channel count for all FPN levels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Forward: (p3, p4, p5) → (fpn3, fpn4, fpn5)
    pub fn forward(
        &self,
        p3: &Variable,
        p4: &Variable,
        p5: &Variable,
    ) -> (Variable, Variable, Variable) {
        // Lateral projections
        let l5 = self.lateral_p5.forward(p5);
        let l4 = self.lateral_p4.forward(p4);
        let l3 = self.lateral_p3.forward(p3);

        // Top-down: upsample P5 → add to P4
        let p4_shape = l4.shape();
        let up5 = interpolate_var(&l5, p4_shape[2], p4_shape[3], InterpolateMode::Nearest);
        let td4 = self.td_p4.forward(&up5.add_var(&l4));

        // Top-down: upsample P4 → add to P3
        let p3_shape = l3.shape();
        let up4 = interpolate_var(&td4, p3_shape[2], p3_shape[3], InterpolateMode::Nearest);
        let td3 = self.td_p3.forward(&up4.add_var(&l3));

        // Bottom-up: downsample P3 → add to P4
        let bu4 = self.bu_p4.forward(&td3).add_var(&td4);

        // Bottom-up: downsample P4 → add to P5
        let bu5 = self.bu_p5.forward(&bu4).add_var(&l5);

        (td3, bu4, bu5)
    }

    /// Returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.lateral_p5.parameters());
        p.extend(self.lateral_p4.parameters());
        p.extend(self.lateral_p3.parameters());
        p.extend(self.td_p4.parameters());
        p.extend(self.td_p3.parameters());
        p.extend(self.bu_p4.parameters());
        p.extend(self.bu_p5.parameters());
        p
    }

    /// Returns named parameters.
    pub fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut p = HashMap::new();
        p.extend(self.lateral_p5.named_parameters("lat_p5"));
        p.extend(self.lateral_p4.named_parameters("lat_p4"));
        p.extend(self.lateral_p3.named_parameters("lat_p3"));
        p.extend(self.td_p4.named_parameters("td_p4"));
        p.extend(self.td_p3.named_parameters("td_p3"));
        p.extend(self.bu_p4.named_parameters("bu_p4"));
        p.extend(self.bu_p5.named_parameters("bu_p5"));
        p
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, training: bool) {
        self.lateral_p5.set_training(training);
        self.lateral_p4.set_training(training);
        self.lateral_p3.set_training(training);
        self.td_p4.set_training(training);
        self.td_p3.set_training(training);
        self.bu_p4.set_training(training);
        self.bu_p5.set_training(training);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_autograd::Variable;
    use axonml_tensor::Tensor;

    #[test]
    fn test_thermal_fpn_output_shapes() {
        let fpn = ThermalFPN::default_config(); // 64/128/256 → 128

        let p3 = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 16 * 16], &[1, 64, 16, 16]).unwrap(),
            false,
        );
        let p4 = Variable::new(
            Tensor::from_vec(vec![0.1; 128 * 8 * 8], &[1, 128, 8, 8]).unwrap(),
            false,
        );
        let p5 = Variable::new(
            Tensor::from_vec(vec![0.1; 256 * 4 * 4], &[1, 256, 4, 4]).unwrap(),
            false,
        );

        let (fpn3, fpn4, fpn5) = fpn.forward(&p3, &p4, &p5);

        // All FPN outputs should have 128 channels
        assert_eq!(fpn3.data().shape()[1], 128);
        assert_eq!(fpn4.data().shape()[1], 128);
        assert_eq!(fpn5.data().shape()[1], 128);

        // Spatial dims should match input levels
        assert_eq!(fpn3.data().shape()[2], 16);
        assert_eq!(fpn4.data().shape()[2], 8);
        assert_eq!(fpn5.data().shape()[2], 4);
    }

    #[test]
    fn test_thermal_fpn_out_channels() {
        let fpn = ThermalFPN::default_config();
        assert_eq!(fpn.out_channels(), 128);
    }

    #[test]
    fn test_thermal_fpn_params() {
        let fpn = ThermalFPN::default_config();
        assert!(!fpn.parameters().is_empty());
        assert!(!fpn.named_parameters().is_empty());
    }
}
