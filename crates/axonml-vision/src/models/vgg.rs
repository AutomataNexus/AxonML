//! VGG - Very Deep Convolutional Networks
//!
//! Implementation of VGG architectures for image classification.
//!
//! # Supported Variants
//!
//! - VGG11: 11 layers (~133M parameters)
//! - VGG13: 13 layers (~133M parameters)
//! - VGG16: 16 layers (~138M parameters)
//! - VGG19: 19 layers (~144M parameters)
//!
//! All variants available with or without batch normalization.
//!
//! # Reference
//!
//! "Very Deep Convolutional Networks for Large-Scale Image Recognition"
//! (Simonyan & Zisserman, 2014)
//! <https://arxiv.org/abs/1409.1556>

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, Parameter, ReLU};
use axonml_tensor::Tensor;

// =============================================================================
// Helper Functions
// =============================================================================

/// Flatten a tensor from [N, C, H, W] to [N, C*H*W].
fn flatten(input: &Variable) -> Variable {
    let data = input.data();
    let shape = data.shape();

    if shape.len() <= 2 {
        return input.clone();
    }

    let batch_size = shape[0];
    let features: usize = shape[1..].iter().product();

    Variable::new(
        Tensor::from_vec(data.to_vec(), &[batch_size, features]).unwrap(),
        input.requires_grad(),
    )
}

// =============================================================================
// VGG Configuration
// =============================================================================

/// VGG layer configuration.
#[derive(Debug, Clone, Copy)]
pub enum VggLayer {
    /// Convolutional layer with output channels.
    Conv(usize),
    /// Max pooling layer.
    MaxPool,
}

/// Get VGG11 configuration.
#[must_use]
pub fn vgg11_config() -> Vec<VggLayer> {
    use VggLayer::{Conv, MaxPool};
    vec![
        Conv(64),
        MaxPool,
        Conv(128),
        MaxPool,
        Conv(256),
        Conv(256),
        MaxPool,
        Conv(512),
        Conv(512),
        MaxPool,
        Conv(512),
        Conv(512),
        MaxPool,
    ]
}

/// Get VGG13 configuration.
#[must_use]
pub fn vgg13_config() -> Vec<VggLayer> {
    use VggLayer::{Conv, MaxPool};
    vec![
        Conv(64),
        Conv(64),
        MaxPool,
        Conv(128),
        Conv(128),
        MaxPool,
        Conv(256),
        Conv(256),
        MaxPool,
        Conv(512),
        Conv(512),
        MaxPool,
        Conv(512),
        Conv(512),
        MaxPool,
    ]
}

/// Get VGG16 configuration.
#[must_use]
pub fn vgg16_config() -> Vec<VggLayer> {
    use VggLayer::{Conv, MaxPool};
    vec![
        Conv(64),
        Conv(64),
        MaxPool,
        Conv(128),
        Conv(128),
        MaxPool,
        Conv(256),
        Conv(256),
        Conv(256),
        MaxPool,
        Conv(512),
        Conv(512),
        Conv(512),
        MaxPool,
        Conv(512),
        Conv(512),
        Conv(512),
        MaxPool,
    ]
}

/// Get VGG19 configuration.
#[must_use]
pub fn vgg19_config() -> Vec<VggLayer> {
    use VggLayer::{Conv, MaxPool};
    vec![
        Conv(64),
        Conv(64),
        MaxPool,
        Conv(128),
        Conv(128),
        MaxPool,
        Conv(256),
        Conv(256),
        Conv(256),
        Conv(256),
        MaxPool,
        Conv(512),
        Conv(512),
        Conv(512),
        Conv(512),
        MaxPool,
        Conv(512),
        Conv(512),
        Conv(512),
        Conv(512),
        MaxPool,
    ]
}

// =============================================================================
// VGG Feature Extractor
// =============================================================================

/// VGG feature extraction layers.
pub struct VggFeatures {
    layers: Vec<VggFeatureLayer>,
}

enum VggFeatureLayer {
    Conv(Conv2d),
    BatchNorm(BatchNorm2d),
    ReLU(ReLU),
    MaxPool(MaxPool2d),
}

impl VggFeatures {
    /// Create VGG feature layers from configuration.
    #[must_use]
    pub fn new(config: &[VggLayer], batch_norm: bool) -> Self {
        let mut layers = Vec::new();
        let mut in_channels = 3;

        for &layer in config {
            match layer {
                VggLayer::Conv(out_channels) => {
                    layers.push(VggFeatureLayer::Conv(Conv2d::with_options(
                        in_channels,
                        out_channels,
                        (3, 3),
                        (1, 1),
                        (1, 1),
                        true,
                    )));
                    if batch_norm {
                        layers.push(VggFeatureLayer::BatchNorm(BatchNorm2d::new(out_channels)));
                    }
                    layers.push(VggFeatureLayer::ReLU(ReLU));
                    in_channels = out_channels;
                }
                VggLayer::MaxPool => {
                    layers.push(VggFeatureLayer::MaxPool(MaxPool2d::with_options(
                        (2, 2),
                        (2, 2),
                        (0, 0),
                    )));
                }
            }
        }

        Self { layers }
    }
}

impl Module for VggFeatures {
    fn forward(&self, x: &Variable) -> Variable {
        let mut out = x.clone();
        for layer in &self.layers {
            out = match layer {
                VggFeatureLayer::Conv(conv) => conv.forward(&out),
                VggFeatureLayer::BatchNorm(bn) => bn.forward(&out),
                VggFeatureLayer::ReLU(relu) => relu.forward(&out),
                VggFeatureLayer::MaxPool(pool) => pool.forward(&out),
            };
        }
        out
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for layer in &self.layers {
            match layer {
                VggFeatureLayer::Conv(conv) => params.extend(conv.parameters()),
                VggFeatureLayer::BatchNorm(bn) => params.extend(bn.parameters()),
                _ => {}
            }
        }
        params
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            if let VggFeatureLayer::BatchNorm(bn) = layer {
                bn.train();
            }
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            if let VggFeatureLayer::BatchNorm(bn) = layer {
                bn.eval();
            }
        }
    }

    fn is_training(&self) -> bool {
        for layer in &self.layers {
            if let VggFeatureLayer::BatchNorm(bn) = layer {
                return bn.is_training();
            }
        }
        true
    }
}

// =============================================================================
// VGG Classifier
// =============================================================================

/// VGG classifier head.
pub struct VggClassifier {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    relu: ReLU,
    dropout: Dropout,
}

impl VggClassifier {
    /// Create classifier for VGG (assuming 7x7 feature maps).
    #[must_use]
    pub fn new(num_classes: usize) -> Self {
        Self {
            fc1: Linear::new(512 * 7 * 7, 4096),
            fc2: Linear::new(4096, 4096),
            fc3: Linear::new(4096, num_classes),
            relu: ReLU,
            dropout: Dropout::new(0.5),
        }
    }

    /// Create classifier with custom input size.
    #[must_use]
    pub fn with_input_size(input_features: usize, num_classes: usize) -> Self {
        Self {
            fc1: Linear::new(input_features, 4096),
            fc2: Linear::new(4096, 4096),
            fc3: Linear::new(4096, num_classes),
            relu: ReLU,
            dropout: Dropout::new(0.5),
        }
    }
}

impl Module for VggClassifier {
    fn forward(&self, x: &Variable) -> Variable {
        let out = self.fc1.forward(x);
        let out = self.relu.forward(&out);
        let out = self.dropout.forward(&out);

        let out = self.fc2.forward(&out);
        let out = self.relu.forward(&out);
        let out = self.dropout.forward(&out);

        self.fc3.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.dropout.is_training()
    }
}

// =============================================================================
// VGG Model
// =============================================================================

/// VGG model for image classification.
pub struct VGG {
    features: VggFeatures,
    classifier: VggClassifier,
}

impl VGG {
    /// Create VGG with custom configuration.
    #[must_use]
    pub fn new(config: &[VggLayer], num_classes: usize, batch_norm: bool) -> Self {
        Self {
            features: VggFeatures::new(config, batch_norm),
            classifier: VggClassifier::new(num_classes),
        }
    }

    /// Create VGG11.
    #[must_use]
    pub fn vgg11(num_classes: usize) -> Self {
        Self::new(&vgg11_config(), num_classes, false)
    }

    /// Create VGG11 with batch normalization.
    #[must_use]
    pub fn vgg11_bn(num_classes: usize) -> Self {
        Self::new(&vgg11_config(), num_classes, true)
    }

    /// Create VGG13.
    #[must_use]
    pub fn vgg13(num_classes: usize) -> Self {
        Self::new(&vgg13_config(), num_classes, false)
    }

    /// Create VGG13 with batch normalization.
    #[must_use]
    pub fn vgg13_bn(num_classes: usize) -> Self {
        Self::new(&vgg13_config(), num_classes, true)
    }

    /// Create VGG16.
    #[must_use]
    pub fn vgg16(num_classes: usize) -> Self {
        Self::new(&vgg16_config(), num_classes, false)
    }

    /// Create VGG16 with batch normalization.
    #[must_use]
    pub fn vgg16_bn(num_classes: usize) -> Self {
        Self::new(&vgg16_config(), num_classes, true)
    }

    /// Create VGG19.
    #[must_use]
    pub fn vgg19(num_classes: usize) -> Self {
        Self::new(&vgg19_config(), num_classes, false)
    }

    /// Create VGG19 with batch normalization.
    #[must_use]
    pub fn vgg19_bn(num_classes: usize) -> Self {
        Self::new(&vgg19_config(), num_classes, true)
    }
}

impl Module for VGG {
    fn forward(&self, x: &Variable) -> Variable {
        let out = self.features.forward(x);

        // Flatten: [batch, 512, 7, 7] -> [batch, 512*7*7]
        let out = flatten(&out);

        self.classifier.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.features.parameters());
        params.extend(self.classifier.parameters());
        params
    }

    fn train(&mut self) {
        self.features.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.features.eval();
        self.classifier.eval();
    }

    fn is_training(&self) -> bool {
        self.features.is_training()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create VGG11 for `ImageNet` (1000 classes).
#[must_use]
pub fn vgg11() -> VGG {
    VGG::vgg11(1000)
}

/// Create VGG13 for `ImageNet` (1000 classes).
#[must_use]
pub fn vgg13() -> VGG {
    VGG::vgg13(1000)
}

/// Create VGG16 for `ImageNet` (1000 classes).
#[must_use]
pub fn vgg16() -> VGG {
    VGG::vgg16(1000)
}

/// Create VGG19 for `ImageNet` (1000 classes).
#[must_use]
pub fn vgg19() -> VGG {
    VGG::vgg19(1000)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgg_features() {
        let config = vec![VggLayer::Conv(64), VggLayer::MaxPool];
        let features = VggFeatures::new(&config, false);

        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );

        let output = features.forward(&input);
        // After one conv and one maxpool
        assert_eq!(output.data().shape()[1], 64);
        assert_eq!(output.data().shape()[2], 16); // 32 / 2
    }

    #[test]
    fn test_vgg11_creation() {
        let model = VGG::vgg11(10);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_vgg11_bn_creation() {
        let model = VGG::vgg11_bn(10);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_vgg16_creation() {
        let model = VGG::vgg16(1000);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_vgg_forward_small() {
        // Use small input for quick test
        let config = vec![VggLayer::Conv(64), VggLayer::MaxPool];
        let features = VggFeatures::new(&config, false);

        // Custom small classifier
        let classifier = VggClassifier::with_input_size(64 * 16 * 16, 10);

        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );

        let out = features.forward(&input);
        let out = flatten(&out);
        let out = classifier.forward(&out);

        assert_eq!(out.data().shape(), &[1, 10]);
    }

    #[test]
    fn test_vgg_train_eval_mode() {
        let mut model = VGG::vgg11_bn(10);

        model.train();
        assert!(model.is_training());

        model.eval();
        // Note: eval mode may not change is_training for all layers
    }
}
