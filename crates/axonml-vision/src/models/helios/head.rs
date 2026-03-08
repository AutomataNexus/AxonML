//! Decoupled Detection Head for Helios
//!
//! Anchor-free decoupled head with separate classification and regression branches.
//! Uses Distribution Focal Loss (DFL) for bounding box regression.
//!
//! @version 0.1.0

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_nn::{Conv2d, Module, Parameter};
use axonml_tensor::Tensor;

use super::backbone::CBS;

// =============================================================================
// Head Branch
// =============================================================================

/// A single detection head branch (cls or reg).
struct HeadBranch {
    conv1: CBS,
    conv2: CBS,
    out_conv: Conv2d,
}

impl HeadBranch {
    fn new(in_ch: usize, hidden_ch: usize, out_ch: usize) -> Self {
        Self {
            conv1: CBS::conv3x3(in_ch, hidden_ch, 1),
            conv2: CBS::conv3x3(hidden_ch, hidden_ch, 1),
            out_conv: Conv2d::with_options(hidden_ch, out_ch, (1, 1), (1, 1), (0, 0), true),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let out = self.conv1.forward(x);
        let out = self.conv2.forward(&out);
        self.out_conv.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.conv1.parameters();
        p.extend(self.conv2.parameters());
        p.extend(self.out_conv.parameters());
        p
    }
}

// =============================================================================
// DFL — Distribution Focal Loss Decode
// =============================================================================

/// DFL: Converts per-bin distributions into a single regression value.
/// project = [0, 1, 2, ..., reg_max-1], applied as weighted sum after softmax.
pub struct DFL {
    reg_max: usize,
    project: Vec<f32>,
}

impl DFL {
    pub fn new(reg_max: usize) -> Self {
        let project: Vec<f32> = (0..reg_max).map(|i| i as f32).collect();
        Self { reg_max, project }
    }

    /// Decode DFL distribution to regression values.
    /// Input: [N, 4*reg_max, H, W] -> Output: [N, 4, H, W]
    pub fn forward(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        let n = shape[0];
        let h = shape[2];
        let w = shape[3];

        // Reshape to [N*4, reg_max, H*W]
        let x = x.reshape(&[n * 4, self.reg_max, h * w]);

        // Softmax over reg_max dimension (dim=1)
        let x = x.softmax(1);

        // Weighted sum: multiply by project vector and sum
        // project: [reg_max] -> broadcast to [1, reg_max, 1]
        let proj_var = Variable::new(
            Tensor::from_vec(self.project.clone(), &[1, self.reg_max, 1]).unwrap(),
            false,
        );
        let weighted = x.mul_var(&proj_var);

        // Sum over reg_max dimension
        let summed = weighted.sum_dim(1); // [N*4, H*W]

        // Reshape back to [N, 4, H, W]
        summed.reshape(&[n, 4, h, w])
    }
}

// =============================================================================
// HeliosHead
// =============================================================================

/// Decoupled detection head for Helios.
///
/// Per-scale separate classification and regression branches with DFL decode.
pub struct HeliosHead {
    cls_branches: Vec<HeadBranch>,
    reg_branches: Vec<HeadBranch>,
    dfl: DFL,
    num_classes: usize,
    reg_max: usize,
}

impl HeliosHead {
    pub fn new(in_channels: &[usize], num_classes: usize, reg_max: usize) -> Self {
        let mut cls_branches = Vec::new();
        let mut reg_branches = Vec::new();

        for &in_ch in in_channels {
            // Hidden channels: at least 64, at most in_ch
            let hidden = in_ch.max(64);
            cls_branches.push(HeadBranch::new(in_ch, hidden, num_classes));
            reg_branches.push(HeadBranch::new(in_ch, hidden, 4 * reg_max));
        }

        Self {
            cls_branches,
            reg_branches,
            dfl: DFL::new(reg_max),
            num_classes,
            reg_max,
        }
    }

    /// Forward for a single scale.
    /// Returns (cls_logits [N, num_classes, H, W], bbox_dfl [N, 4*reg_max, H, W]).
    pub fn forward_single(&self, x: &Variable, scale_idx: usize) -> (Variable, Variable) {
        let cls = self.cls_branches[scale_idx].forward(x);
        let bbox = self.reg_branches[scale_idx].forward(x);
        (cls, bbox)
    }

    /// Decode DFL distribution to distance values.
    /// Input: [N, 4*reg_max, H, W] -> [N, 4, H, W]
    pub fn dfl_decode(&self, bbox_dfl: &Variable) -> Variable {
        self.dfl.forward(bbox_dfl)
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn reg_max(&self) -> usize {
        self.reg_max
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for b in &self.cls_branches {
            p.extend(b.parameters());
        }
        for b in &self.reg_branches {
            p.extend(b.parameters());
        }
        p
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_branch() {
        let branch = HeadBranch::new(64, 64, 80);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 64 * 8 * 8], &[1, 64, 8, 8]).unwrap(),
            false,
        );
        let out = branch.forward(&input);
        assert_eq!(out.shape(), vec![1, 80, 8, 8]);
    }

    #[test]
    fn test_dfl_decode() {
        let dfl = DFL::new(16);
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 64 * 4 * 4], &[1, 64, 4, 4]).unwrap(),
            false,
        );
        let out = dfl.forward(&input);
        assert_eq!(out.shape(), vec![1, 4, 4, 4]);
    }

    #[test]
    fn test_helios_head() {
        let head = HeliosHead::new(&[64, 128, 256], 80, 16);

        let feat3 = Variable::new(
            Tensor::from_vec(vec![0.5; 64 * 8 * 8], &[1, 64, 8, 8]).unwrap(),
            false,
        );
        let (cls, bbox) = head.forward_single(&feat3, 0);
        assert_eq!(cls.shape(), vec![1, 80, 8, 8]);
        assert_eq!(bbox.shape(), vec![1, 64, 8, 8]); // 4*16
    }
}
