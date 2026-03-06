//! Vision Models
//!
//! Provides pre-defined neural network architectures for computer vision.
//!
//! # Available Models
//!
//! ## Classification
//! - **`LeNet`**: Classic architecture for MNIST
//! - **`SimpleCNN`**: Flexible CNN for quick experiments
//! - **`ResNet`**: Deep residual networks (`ResNet18`, `ResNet34`)
//! - **VGG**: Very deep networks (VGG11, VGG13, VGG16, VGG19)
//! - **`VisionTransformer`**: Attention-based image classification (`ViT`)
//!
//! ## Detection
//! - **`Helios`**: YOLO-competitive anchor-free detector (Nano 3M → XLarge 68M)
//! - **`RetinaFace`**: Server-grade face detection with landmarks
//! - **`BlazeFace`**: Edge face detection (~100K params)
//! - **DETR**: End-to-end object detection with Transformers
//! - **`NanoDet`**: Edge anchor-free object detection (<1M params)
//!
//! ## Anomaly Detection
//! - **`PatchCore`**: Feature-based anomaly detection (no anomaly training data needed)
//! - **`StudentTeacher`**: Lightweight edge anomaly detection
//!
//! ## Depth Estimation
//! - **DPT**: Dense Prediction Transformer for monocular depth
//! - **`FastDepth`**: Lightweight edge depth estimation
//!
//! ## Visual Question Answering
//! - **`VQAModel`**: Image + text -> answer classification
//!
//! ## 3D Reconstruction
//! - **`Aegis3D`**: Octree-adaptive neural implicit surface reconstruction
//!   - Progressive LOD, depth-guided init, OBJ/STL mesh export
//!
//! ## Biometric Identity (Aegis Identity)
//! - **Mnemosyne**: Face identity via temporal crystallization (~115K params)
//! - **Ariadne**: Fingerprint via ridge event fields (~65K params)
//! - **Echo**: Voice via predictive speaker residuals (~68K params)
//! - **Argus**: Iris via radial phase encoding (~65K params)
//! - **Themis**: Multimodal belief propagation fusion (~49K params)
//! - **AegisIdentity**: Unified biometric API (enroll/verify/identify)
//!
//! ## Infrastructure
//! - **FPN**: Feature Pyramid Network (shared by detection models)
//!
//! @version 0.2.0
//! @author `AutomataNexus` Development Team

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
pub mod retinaface;
pub mod resnet;
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
pub use helios::{Helios, HeliosLoss, CIoULoss, TaskAlignedAssigner};
pub use nanodet::NanoDet;
pub use retinaface::RetinaFace;

// Anomaly Detection
pub use anomaly::{PatchCore, StudentTeacher};

// Depth Estimation
pub use depth::{DPT, FastDepth};

// Visual Question Answering
pub use vqa::VQAModel;

// 3D Reconstruction
pub use aegis3d::Aegis3D;

// Novel Detection Architectures
pub use nexus::Nexus;
pub use phantom::Phantom;

// Biometric Identity (Aegis Identity)
pub use biometric::{
    AegisIdentity, AriadneFingerprint, ArgusIris, EchoSpeaker, IdentityBank,
    MnemosyneIdentity, ThemisFusion,
};
