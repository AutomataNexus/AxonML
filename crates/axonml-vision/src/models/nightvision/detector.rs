//! NightVision Detector — Multi-Domain Infrared Object Detection
//!
//! Complete detection model: backbone + FPN neck + decoupled head.
//! Configurable for wildlife, human, and interstellar detection.

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_nn::{Module, Parameter};

use super::backbone::ThermalBackbone;
use super::head::DecoupledHead;
use super::neck::ThermalFPN;

// =============================================================================
// Thermal Domain
// =============================================================================

/// Detection domain for NightVision.
///
/// Each domain has different thermal characteristics:
/// - Wildlife: warm-blooded animals against cooler backgrounds
/// - Human: body heat signatures, upright posture priors
/// - Interstellar: point sources, nebulae, thermal emissions against cold space
/// - Vehicle: engine heat, tire friction signatures
/// - General: domain-agnostic detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalDomain {
    /// Wildlife detection — animals, birds, insects (thermal contrast)
    Wildlife,
    /// Human detection — people, search & rescue, perimeter security
    Human,
    /// Interstellar — astronomical thermal sources, space debris
    Interstellar,
    /// Vehicle — cars, drones, aircraft (engine heat)
    Vehicle,
    /// General — domain-agnostic
    General,
}

impl ThermalDomain {
    /// Returns the integer index for this domain.
    pub fn index(&self) -> usize {
        match self {
            Self::Wildlife => 0,
            Self::Human => 1,
            Self::Interstellar => 2,
            Self::Vehicle => 3,
            Self::General => 4,
        }
    }

    /// Creates a domain from an integer index.
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Wildlife,
            1 => Self::Human,
            2 => Self::Interstellar,
            3 => Self::Vehicle,
            _ => Self::General,
        }
    }

    /// Returns the string name of this domain.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wildlife => "wildlife",
            Self::Human => "human",
            Self::Interstellar => "interstellar",
            Self::Vehicle => "vehicle",
            Self::General => "general",
        }
    }

    /// Returns the total number of thermal domains.
    pub fn count() -> usize {
        5
    }
}

// =============================================================================
// NightVision Config
// =============================================================================

/// Configuration for NightVision detector.
#[derive(Debug, Clone)]
pub struct NightVisionConfig {
    /// Input channels (1 = single-channel thermal, 3 = multi-band / pseudo-color)
    pub in_channels: usize,
    /// Number of object classes to detect
    pub num_classes: usize,
    /// Number of thermal domains (0 = no domain head)
    pub num_domains: usize,
    /// FPN output channels
    pub fpn_channels: usize,
    /// Input image size (square, for reference — model handles any size)
    pub img_size: usize,
}

impl NightVisionConfig {
    /// Wildlife detection: single-channel thermal, common animal classes.
    pub fn wildlife(num_species: usize) -> Self {
        Self {
            in_channels: 1,
            num_classes: num_species,
            num_domains: 0,
            fpn_channels: 128,
            img_size: 320,
        }
    }

    /// Human detection: single-channel thermal.
    pub fn human() -> Self {
        Self {
            in_channels: 1,
            num_classes: 1, // person
            num_domains: 0,
            fpn_channels: 128,
            img_size: 320,
        }
    }

    /// Interstellar detection: single or multi-band IR.
    pub fn interstellar(num_classes: usize, bands: usize) -> Self {
        Self {
            in_channels: bands,
            num_classes,
            num_domains: 0,
            fpn_channels: 128,
            img_size: 512,
        }
    }

    /// Multi-domain: wildlife + human + interstellar + vehicle with domain tags.
    pub fn multi_domain(num_classes: usize) -> Self {
        Self {
            in_channels: 1,
            num_classes,
            num_domains: ThermalDomain::count(),
            fpn_channels: 128,
            img_size: 320,
        }
    }

    /// Compact model for edge deployment.
    pub fn edge(num_classes: usize) -> Self {
        Self {
            in_channels: 1,
            num_classes,
            num_domains: 0,
            fpn_channels: 64,
            img_size: 256,
        }
    }
}

impl Default for NightVisionConfig {
    fn default() -> Self {
        Self::multi_domain(10)
    }
}

// =============================================================================
// NightVision — Full Detector
// =============================================================================

/// NightVision: Multi-domain infrared object detection.
///
/// A YOLOX-inspired detector adapted for thermal imagery with:
/// - CSP backbone with thermal-adaptive stem (handles 1-ch or 3-ch IR)
/// - Feature Pyramid Network for multi-scale detection
/// - Decoupled detection heads (classification, bbox, objectness)
/// - Optional domain classification head
///
/// # Usage
///
/// ```ignore
/// use axonml_vision::models::nightvision::{NightVision, NightVisionConfig};
///
/// // Wildlife detection (thermal camera)
/// let model = NightVision::new(NightVisionConfig::wildlife(20));
///
/// // Human detection (search & rescue)
/// let model = NightVision::new(NightVisionConfig::human());
///
/// // Multi-domain (all targets)
/// let model = NightVision::new(NightVisionConfig::multi_domain(50));
///
/// // Forward pass
/// let (cls, bbox, obj, domain) = model.forward_detection(&ir_image);
/// ```
pub struct NightVision {
    backbone: ThermalBackbone,
    neck: ThermalFPN,
    head_p3: DecoupledHead,
    head_p4: DecoupledHead,
    head_p5: DecoupledHead,
    config: NightVisionConfig,
}

impl NightVision {
    /// Create a new NightVision detector with the given configuration.
    pub fn new(config: NightVisionConfig) -> Self {
        let fpn_ch = config.fpn_channels;

        Self {
            backbone: ThermalBackbone::new(config.in_channels),
            neck: ThermalFPN::new(64, 128, 256, fpn_ch),
            head_p3: DecoupledHead::new(fpn_ch, config.num_classes, config.num_domains),
            head_p4: DecoupledHead::new(fpn_ch, config.num_classes, config.num_domains),
            head_p5: DecoupledHead::new(fpn_ch, config.num_classes, config.num_domains),
            config,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &NightVisionConfig {
        &self.config
    }

    /// Full detection forward pass.
    ///
    /// Input: [B, C, H, W] thermal image
    /// Returns per-scale outputs: Vec of (cls, bbox, obj, domain) per FPN level.
    pub fn forward_detection(
        &self,
        x: &Variable,
    ) -> Vec<(Variable, Variable, Variable, Option<Variable>)> {
        // Backbone: extract multi-scale features
        let (p3, p4, p5) = self.backbone.forward(x);

        // Neck: fuse features
        let (fpn3, fpn4, fpn5) = self.neck.forward(&p3, &p4, &p5);

        // Heads: per-scale detection
        let out3 = self.head_p3.forward(&fpn3);
        let out4 = self.head_p4.forward(&fpn4);
        let out5 = self.head_p5.forward(&fpn5);

        vec![out3, out4, out5]
    }

    /// Flatten multi-scale outputs into single tensors.
    ///
    /// Returns (all_cls, all_bbox, all_obj) flattened across spatial locations and scales.
    pub fn forward_flat(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let outputs = self.forward_detection(x);
        let batch = x.shape()[0];

        let mut all_cls = Vec::new();
        let mut all_bbox = Vec::new();
        let mut all_obj = Vec::new();

        for (cls, bbox, obj, _) in &outputs {
            let cls_shape = cls.shape();
            let h = cls_shape[2];
            let w = cls_shape[3];
            let n_anchors = h * w;

            // Reshape: [B, C, H, W] → [B, H*W, C]
            let cls_flat = cls
                .reshape(&[batch, self.config.num_classes, n_anchors])
                .transpose(1, 2); // [B, H*W, num_classes]
            let bbox_flat = bbox
                .reshape(&[batch, 4, n_anchors])
                .transpose(1, 2); // [B, H*W, 4]
            let obj_flat = obj
                .reshape(&[batch, 1, n_anchors])
                .transpose(1, 2); // [B, H*W, 1]

            all_cls.push(cls_flat);
            all_bbox.push(bbox_flat);
            all_obj.push(obj_flat);
        }

        // Concatenate across scales
        let cls_refs: Vec<&Variable> = all_cls.iter().collect();
        let bbox_refs: Vec<&Variable> = all_bbox.iter().collect();
        let obj_refs: Vec<&Variable> = all_obj.iter().collect();

        (
            Variable::cat(&cls_refs, 1),  // [B, total_anchors, num_classes]
            Variable::cat(&bbox_refs, 1), // [B, total_anchors, 4]
            Variable::cat(&obj_refs, 1),  // [B, total_anchors, 1]
        )
    }
}

impl Module for NightVision {
    /// Forward pass returning flattened class logits.
    /// For full detection outputs, use `forward_detection()` or `forward_flat()`.
    fn forward(&self, input: &Variable) -> Variable {
        let (cls, _bbox, _obj) = self.forward_flat(input);
        cls
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.backbone.parameters();
        p.extend(self.neck.parameters());
        p.extend(self.head_p3.parameters());
        p.extend(self.head_p4.parameters());
        p.extend(self.head_p5.parameters());
        p
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut p = self.backbone.named_parameters();
        p.extend(self.neck.named_parameters());
        p.extend(self.head_p3.named_parameters("head_p3"));
        p.extend(self.head_p4.named_parameters("head_p4"));
        p.extend(self.head_p5.named_parameters("head_p5"));
        p
    }

    fn name(&self) -> &'static str {
        "NightVision"
    }

    fn set_training(&mut self, training: bool) {
        self.backbone.set_training(training);
        self.neck.set_training(training);
        self.head_p3.set_training(training);
        self.head_p4.set_training(training);
        self.head_p5.set_training(training);
    }
}

// =============================================================================
// NightVisionLoss
// =============================================================================

/// Combined detection loss for NightVision.
///
/// Components:
/// - Classification: BCE with logits (per-class)
/// - Bounding box: CIoU loss (center + size regression)
/// - Objectness: BCE with logits (foreground/background)
/// - Domain: CrossEntropy (optional, multi-domain mode)
pub struct NightVisionLoss {
    /// Classification loss weight.
    pub cls_weight: f32,
    /// Bounding box regression loss weight.
    pub bbox_weight: f32,
    /// Objectness loss weight.
    pub obj_weight: f32,
    /// Domain classification loss weight.
    pub domain_weight: f32,
}

impl Default for NightVisionLoss {
    fn default() -> Self {
        Self {
            cls_weight: 1.0,
            bbox_weight: 5.0,
            obj_weight: 1.0,
            domain_weight: 0.5,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nightvision_wildlife_forward() {
        let config = NightVisionConfig::wildlife(10);
        let model = NightVision::new(config);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );

        let outputs = model.forward_detection(&input);
        assert_eq!(outputs.len(), 3); // 3 FPN levels

        for (cls, bbox, obj, domain) in &outputs {
            assert_eq!(cls.shape()[0], 1);
            assert_eq!(cls.shape()[1], 10); // num_classes
            assert_eq!(bbox.shape()[1], 4);
            assert_eq!(obj.shape()[1], 1);
            assert!(domain.is_none());
        }
    }

    #[test]
    fn test_nightvision_human_forward() {
        let config = NightVisionConfig::human();
        let model = NightVision::new(config);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 64 * 64], &[1, 1, 64, 64]).unwrap(),
            false,
        );

        let (cls, bbox, obj) = model.forward_flat(&input);
        assert_eq!(cls.shape()[0], 1);
        assert_eq!(cls.shape()[2], 1); // 1 class (person)
        assert_eq!(bbox.shape()[2], 4);
    }

    #[test]
    fn test_nightvision_multi_domain() {
        let config = NightVisionConfig::multi_domain(5);
        let model = NightVision::new(config);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 64 * 64], &[1, 1, 64, 64]).unwrap(),
            false,
        );

        let outputs = model.forward_detection(&input);
        for (_, _, _, domain) in &outputs {
            assert!(domain.is_some());
            let d = domain.as_ref().unwrap();
            assert_eq!(d.shape()[1], 5); // 5 domains
        }
    }

    #[test]
    fn test_nightvision_param_count() {
        let model = NightVision::new(NightVisionConfig::wildlife(10));
        let count: usize = model.parameters().iter().map(|p| p.numel()).sum();
        println!("NightVision(wildlife, 10 classes): {} params", count);
        assert!(count > 100_000);
        assert!(count < 5_000_000);
    }

    #[test]
    fn test_nightvision_interstellar() {
        let config = NightVisionConfig::interstellar(3, 3); // 3-band IR, 3 classes
        let model = NightVision::new(config);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        let (cls, bbox, obj) = model.forward_flat(&input);
        assert_eq!(cls.shape()[2], 3);
    }
}
