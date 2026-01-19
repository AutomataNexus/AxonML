//! Vision Models
//!
//! Provides pre-defined neural network architectures for computer vision.
//!
//! # Available Models
//!
//! - **`LeNet`**: Classic architecture for MNIST
//! - **`SimpleCNN`**: Flexible CNN for quick experiments
//! - **`ResNet`**: Deep residual networks (`ResNet18`, `ResNet34`)
//! - **VGG**: Very deep networks (VGG11, VGG13, VGG16, VGG19)
//! - **Transformer**: Attention-based models (`ViT`)
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

pub mod lenet;
pub mod resnet;
pub mod transformer;
pub mod vgg;

pub use lenet::{LeNet, SimpleCNN, MLP};
pub use resnet::{resnet18, resnet34, BasicBlock, Bottleneck, ResNet};
pub use transformer::{
    vit_base, vit_large, PositionalEncoding, Transformer, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, VisionTransformer,
};
pub use vgg::{vgg11, vgg13, vgg16, vgg19, VggClassifier, VggFeatures, VGG};
