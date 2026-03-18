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
