//! Vision Models
//!
//! # File
//! `crates/axonml-vision/src/models/mod.rs`
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

pub mod aegis3d;
pub mod anomaly;
pub mod biometric;
pub mod blazeface;
pub mod depth;
pub mod detr;
pub mod fpn;
pub mod helios;
pub mod lenet;
pub mod nanodet;
pub mod nexus;
pub mod phantom;
pub mod resnet;
pub mod retinaface;
pub mod transformer;
pub mod vgg;
pub mod vqa;

// Classification
pub use lenet::{LeNet, SimpleCNN, MLP};
pub use resnet::{resnet18, resnet34, BasicBlock, Bottleneck, ResNet};
pub use transformer::{
    vit_base, vit_large, PositionalEncoding, Transformer, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, VisionTransformer,
};
pub use vgg::{vgg11, vgg13, vgg16, vgg19, VggClassifier, VggFeatures, VGG};

// Infrastructure
pub use fpn::FPN;

// Detection
pub use blazeface::BlazeFace;
pub use detr::DETR;
pub use helios::{CIoULoss, Helios, HeliosLoss, TaskAlignedAssigner};
pub use nanodet::NanoDet;
pub use retinaface::RetinaFace;

// Anomaly Detection
pub use anomaly::{PatchCore, StudentTeacher};

// Depth Estimation
pub use depth::{FastDepth, DPT};

// Visual Question Answering
pub use vqa::VQAModel;

// 3D Reconstruction
pub use aegis3d::Aegis3D;

// Novel Detection Architectures
pub use nexus::Nexus;
pub use phantom::Phantom;

// Biometric Identity (Aegis Identity)
pub use biometric::{
    AegisIdentity, ArgusIris, AriadneFingerprint, EchoSpeaker, IdentityBank, MnemosyneIdentity,
    ThemisFusion,
};

// =============================================================================
// Smoke Tests — Detection Model Param Counts & Output Shapes
// =============================================================================

#[cfg(test)]
mod detection_smoke_tests {
    use axonml_autograd::Variable;
    use axonml_nn::Module;
    use axonml_tensor::Tensor;

    fn param_count(model: &dyn Module) -> usize {
        model
            .parameters()
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum()
    }

    #[test]
    fn smoke_nanodet_param_count_and_shape() {
        let model = super::NanoDet::new(80);
        let count = param_count(&model);
        eprintln!("NanoDet(80): {} params", count);
        assert!(count > 0 && count < 1_000_000, "NanoDet: {} params", count);

        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        let out = model.forward(&input);
        eprintln!("NanoDet output shape: {:?}", out.shape());
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 80);
    }

    #[test]
    fn smoke_retinaface_param_count_and_shape() {
        let model = super::RetinaFace::new();
        let count = param_count(&model);
        eprintln!("RetinaFace: {} params", count);
        assert!(count > 10_000, "RetinaFace: {} params", count);

        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        let out = model.forward(&input);
        eprintln!("RetinaFace output shape: {:?}", out.shape());
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 4); // 2 anchors * 2 classes
    }

    #[test]
    fn smoke_detr_param_count_and_shape() {
        let model = super::DETR::small(10);
        let count = param_count(&model);
        eprintln!("DETR(small, 10 classes): {} params", count);
        assert!(count > 10_000, "DETR: {} params", count);

        let features = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 64 * 4 * 4], &[1, 64, 4, 4]).unwrap(),
            false,
        );
        let (cls, bbox) = model.forward_detection(&features);
        eprintln!("DETR cls shape: {:?}, bbox shape: {:?}", cls.shape(), bbox.shape());
        assert_eq!(cls.shape(), vec![1, 10, 11]);
        assert_eq!(bbox.shape(), vec![1, 10, 4]);
    }

    #[test]
    fn smoke_dpt_param_count_and_shape() {
        let model = super::DPT::small();
        let count = param_count(&model);
        eprintln!("DPT(small): {} params", count);
        assert!(count > 10_000, "DPT: {} params", count);

        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let out = model.forward(&input);
        eprintln!("DPT output shape: {:?}", out.shape());
        assert_eq!(out.shape(), vec![1, 1, 32, 32]);
    }

    #[test]
    fn smoke_fastdepth_param_count_and_shape() {
        let model = super::FastDepth::new();
        let count = param_count(&model);
        eprintln!("FastDepth: {} params", count);
        assert!(count > 10_000 && count < 4_000_000, "FastDepth: {} params", count);

        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let out = model.forward(&input);
        eprintln!("FastDepth output shape: {:?}", out.shape());
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 1);
    }

    #[test]
    fn smoke_vqa_param_count_and_shape() {
        let model = super::VQAModel::small(100, 50);
        let count = param_count(&model);
        eprintln!("VQA(small, 100 vocab, 50 answers): {} params", count);
        assert!(count > 1_000, "VQA: {} params", count);

        let image = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let question = Variable::new(
            Tensor::from_vec(vec![1.0, 5.0, 10.0, 20.0, 3.0], &[1, 5]).unwrap(),
            false,
        );
        let logits = model.forward_vqa(&image, &question);
        eprintln!("VQA output shape: {:?}", logits.shape());
        assert_eq!(logits.shape(), vec![1, 50]);
    }

    #[test]
    fn smoke_patchcore_param_count_and_forward() {
        let model = super::PatchCore::default_rgb();
        let count = param_count(&model);
        eprintln!("PatchCore(RGB, 256): {} params", count);
        assert!(count > 1_000, "PatchCore: {} params", count);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let out = model.forward(&input);
        eprintln!("PatchCore feature shape: {:?}", out.shape());
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 256);
    }

    #[test]
    fn smoke_student_teacher_param_count_and_forward() {
        let model = super::StudentTeacher::default_rgb();
        let count = param_count(&model);
        eprintln!("StudentTeacher(RGB, 128): {} params (student only)", count);
        assert!(count > 1_000, "StudentTeacher: {} params", count);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 1 * 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let out = model.forward(&input);
        eprintln!("StudentTeacher output shape: {:?}", out.shape());
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 128);
    }
}
