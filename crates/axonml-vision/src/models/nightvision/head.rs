//! NightVision Detection Head — Decoupled Classification + Regression
//!
//! YOLOX-style decoupled head: separate branches for classification,
//! bounding box regression, and objectness scoring.
//! Optional domain classification head for multi-domain deployment.

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

    #[allow(dead_code)]
    num_classes: usize,
    #[allow(dead_code)]
    num_domains: usize,
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
    pub fn forward(
        &self,
        x: &Variable,
    ) -> (Variable, Variable, Variable, Option<Variable>) {
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
        p.extend(self.cls_conv.named_parameters(&format!("{}.cls_conv", prefix)));
        for (k, v) in self.cls_pred.named_parameters() {
            p.insert(format!("{}.cls_pred.{}", prefix, k), v);
        }
        p.extend(self.reg_conv.named_parameters(&format!("{}.reg_conv", prefix)));
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
