//! NightVision Detection Head — Decoupled Classification + Regression
//!
//! YOLOX-style decoupled detection head for a single FPN level. Implements
//! `DecoupledHead`, which routes features through a shared 1×1 `ConvBNSiLU`
//! stem and then into four branches:
//!
//! - Classification: `cls_conv` (`ConvBNSiLU` 3×3) followed by a 1×1
//!   `cls_pred` `Conv2d` producing `[B, num_classes, H, W]` logits.
//! - Regression: `reg_conv` (`ConvBNSiLU` 3×3) feeding `reg_pred` (1×1
//!   `Conv2d`) to produce `[B, 4, H, W]` bbox outputs (x, y, w, h).
//! - Objectness: `obj_pred` (1×1 `Conv2d`) sharing the regression features
//!   for a `[B, 1, H, W]` objectness map.
//! - Optional domain: `domain_pred` (1×1 `Conv2d`) branches off the
//!   classification features when `num_domains > 0` for multi-domain
//!   deployment and is otherwise `None`.
//!
//! Decoupled classification/regression prevents task interference (a key
//! finding from YOLOX). The struct exposes `num_classes` and `num_domains`
//! as public fields, and provides `parameters`, `named_parameters`, and
//! `set_training` (only propagated to the BN-bearing sub-blocks). Tests
//! cover the 0-domain and 3-domain configurations and verify that the
//! domain tensor is present or absent as expected.
//!
//! # File
//! `crates/axonml-vision/src/models/nightvision/head.rs`
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
use axonml_nn::{Conv2d, Module, Parameter};

use super::backbone::ConvBNSiLU;

// =============================================================================
// DecoupledHead — Per-scale detection head
// =============================================================================

/// Decoupled detection head for one FPN level.
///
/// Separate branches for classification and regression prevent
/// task interference (a key insight from YOLOX).
pub struct DecoupledHead {
    // Shared stem
    stem: ConvBNSiLU,

    // Classification branch
    cls_conv: ConvBNSiLU,
    cls_pred: Conv2d, // → [B, num_classes, H, W]

    // Regression branch
    reg_conv: ConvBNSiLU,
    reg_pred: Conv2d, // → [B, 4, H, W] (x, y, w, h)

    // Objectness branch (shared with regression)
    obj_pred: Conv2d, // → [B, 1, H, W]

    // Optional domain branch
    domain_pred: Option<Conv2d>, // → [B, num_domains, H, W]

    /// Number of object classes.
    pub num_classes: usize,
    /// Number of domain labels (0 = no domain head).
    pub num_domains: usize,
}

impl DecoupledHead {
    /// Create a new decoupled detection head.
    pub fn new(in_ch: usize, num_classes: usize, num_domains: usize) -> Self {
        let hidden = in_ch;

        Self {
            stem: ConvBNSiLU::new(in_ch, hidden, 1, 1, 0),

            cls_conv: ConvBNSiLU::new(hidden, hidden, 3, 1, 1),
            cls_pred: Conv2d::with_options(hidden, num_classes, (1, 1), (1, 1), (0, 0), true),

            reg_conv: ConvBNSiLU::new(hidden, hidden, 3, 1, 1),
            reg_pred: Conv2d::with_options(hidden, 4, (1, 1), (1, 1), (0, 0), true),

            obj_pred: Conv2d::with_options(hidden, 1, (1, 1), (1, 1), (0, 0), true),

            domain_pred: if num_domains > 0 {
                Some(Conv2d::with_options(
                    hidden,
                    num_domains,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    true,
                ))
            } else {
                None
            },

            num_classes,
            num_domains,
        }
    }

    /// Forward pass for one FPN level.
    ///
    /// Returns (cls_logits, bbox_pred, obj_logits, domain_logits)
    /// - cls_logits: [B, num_classes, H, W]
    /// - bbox_pred: [B, 4, H, W]
    /// - obj_logits: [B, 1, H, W]
    /// - domain_logits: [B, num_domains, H, W] or None
    pub fn forward(&self, x: &Variable) -> (Variable, Variable, Variable, Option<Variable>) {
        let stem_out = self.stem.forward(x);

        // Classification branch
        let cls_feat = self.cls_conv.forward(&stem_out);
        let cls_out = self.cls_pred.forward(&cls_feat);

        // Regression branch
        let reg_feat = self.reg_conv.forward(&stem_out);
        let reg_out = self.reg_pred.forward(&reg_feat);
        let obj_out = self.obj_pred.forward(&reg_feat);

        // Domain branch
        let domain_out = self.domain_pred.as_ref().map(|dp| dp.forward(&cls_feat));

        (cls_out, reg_out, obj_out, domain_out)
    }

    /// Returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.stem.parameters();
        p.extend(self.cls_conv.parameters());
        p.extend(self.cls_pred.parameters());
        p.extend(self.reg_conv.parameters());
        p.extend(self.reg_pred.parameters());
        p.extend(self.obj_pred.parameters());
        if let Some(ref dp) = self.domain_pred {
            p.extend(dp.parameters());
        }
        p
    }

    /// Returns named parameters with the given prefix.
    pub fn named_parameters(&self, prefix: &str) -> HashMap<String, Parameter> {
        let mut p = self.stem.named_parameters(&format!("{}.stem", prefix));
        p.extend(
            self.cls_conv
                .named_parameters(&format!("{}.cls_conv", prefix)),
        );
        for (k, v) in self.cls_pred.named_parameters() {
            p.insert(format!("{}.cls_pred.{}", prefix, k), v);
        }
        p.extend(
            self.reg_conv
                .named_parameters(&format!("{}.reg_conv", prefix)),
        );
        for (k, v) in self.reg_pred.named_parameters() {
            p.insert(format!("{}.reg_pred.{}", prefix, k), v);
        }
        for (k, v) in self.obj_pred.named_parameters() {
            p.insert(format!("{}.obj_pred.{}", prefix, k), v);
        }
        if let Some(ref dp) = self.domain_pred {
            for (k, v) in dp.named_parameters() {
                p.insert(format!("{}.domain_pred.{}", prefix, k), v);
            }
        }
        p
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, training: bool) {
        self.stem.set_training(training);
        self.cls_conv.set_training(training);
        self.reg_conv.set_training(training);
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
    fn test_decoupled_head_shapes() {
        let head = DecoupledHead::new(128, 10, 0);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 128 * 8 * 8], &[1, 128, 8, 8]).unwrap(),
            false,
        );

        let (cls, reg, obj, domain) = head.forward(&x);

        assert_eq!(cls.data().shape(), &[1, 10, 8, 8]); // num_classes=10
        assert_eq!(reg.data().shape(), &[1, 4, 8, 8]); // bbox: x,y,w,h
        assert_eq!(obj.data().shape(), &[1, 1, 8, 8]); // objectness
        assert!(domain.is_none()); // num_domains=0
    }

    #[test]
    fn test_decoupled_head_with_domain() {
        let head = DecoupledHead::new(64, 5, 3);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 64 * 4 * 4], &[2, 64, 4, 4]).unwrap(),
            false,
        );

        let (cls, reg, obj, domain) = head.forward(&x);

        assert_eq!(cls.data().shape(), &[2, 5, 4, 4]);
        assert_eq!(reg.data().shape(), &[2, 4, 4, 4]);
        assert_eq!(obj.data().shape(), &[2, 1, 4, 4]);
        assert!(domain.is_some());
        assert_eq!(domain.unwrap().data().shape(), &[2, 3, 4, 4]); // 3 domains
    }

    #[test]
    fn test_decoupled_head_config_fields() {
        let head = DecoupledHead::new(128, 20, 4);
        assert_eq!(head.num_classes, 20);
        assert_eq!(head.num_domains, 4);
    }

    #[test]
    fn test_decoupled_head_params() {
        let head = DecoupledHead::new(64, 5, 0);
        assert!(!head.parameters().is_empty());
    }
}
