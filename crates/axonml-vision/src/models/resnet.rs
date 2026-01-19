//! `ResNet` - Deep Residual Networks
//!
//! Implementation of `ResNet` architectures for image classification.
//!
//! # Supported Variants
//!
//! - `ResNet18`: 18 layers, ~11M parameters
//! - `ResNet34`: 34 layers, ~21M parameters
//! - `ResNet50`: 50 layers, ~23M parameters
//! - `ResNet101`: 101 layers, ~42M parameters
//! - `ResNet152`: 152 layers, ~58M parameters
//!
//! # Reference
//!
//! "Deep Residual Learning for Image Recognition" (He et al., 2015)
//! <https://arxiv.org/abs/1512.03385>

use axonml_autograd::Variable;
use axonml_nn::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, Parameter, ReLU,
};
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
// Basic Block (for ResNet18, ResNet34)
// =============================================================================

/// Basic residual block for ResNet18/34.
///
/// Structure: conv3x3 -> BN -> `ReLU` -> conv3x3 -> BN + skip -> `ReLU`
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
    relu: ReLU,
}

impl BasicBlock {
    /// Expansion factor for this block type.
    pub const EXPANSION: usize = 1;

    /// Create a new `BasicBlock`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: Option<(Conv2d, BatchNorm2d)>,
    ) -> Self {
        Self {
            conv1: Conv2d::with_options(
                in_channels,
                out_channels,
                (3, 3),
                (stride, stride),
                (1, 1),
                true,
            ),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::with_options(out_channels, out_channels, (3, 3), (1, 1), (1, 1), true),
            bn2: BatchNorm2d::new(out_channels),
            downsample,
            relu: ReLU,
        }
    }
}

impl Module for BasicBlock {
    fn forward(&self, x: &Variable) -> Variable {
        let identity = x.clone();

        let out = self.conv1.forward(x);
        let out = self.bn1.forward(&out);
        let out = self.relu.forward(&out);

        let out = self.conv2.forward(&out);
        let out = self.bn2.forward(&out);

        let identity = match &self.downsample {
            Some((conv, bn)) => {
                let ds = conv.forward(&identity);
                bn.forward(&ds)
            }
            None => identity,
        };

        // Residual connection: out = out + identity
        let out = out.add_var(&identity);
        self.relu.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some((conv, bn)) = &self.downsample {
            params.extend(conv.parameters());
            params.extend(bn.parameters());
        }
        params
    }

    fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        if let Some((_, bn)) = &mut self.downsample {
            bn.train();
        }
    }

    fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        if let Some((_, bn)) = &mut self.downsample {
            bn.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.bn1.is_training()
    }
}

// =============================================================================
// Bottleneck Block (for ResNet50, ResNet101, ResNet152)
// =============================================================================

/// Bottleneck residual block for ResNet50/101/152.
///
/// Structure: conv1x1 -> BN -> `ReLU` -> conv3x3 -> BN -> `ReLU` -> conv1x1 -> BN + skip -> `ReLU`
pub struct Bottleneck {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
    relu: ReLU,
}

impl Bottleneck {
    /// Expansion factor for this block type.
    pub const EXPANSION: usize = 4;

    /// Create a new Bottleneck block.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: Option<(Conv2d, BatchNorm2d)>,
    ) -> Self {
        let width = out_channels;

        Self {
            // 1x1 conv to reduce channels
            conv1: Conv2d::with_options(in_channels, width, (1, 1), (1, 1), (0, 0), true),
            bn1: BatchNorm2d::new(width),
            // 3x3 conv
            conv2: Conv2d::with_options(width, width, (3, 3), (stride, stride), (1, 1), true),
            bn2: BatchNorm2d::new(width),
            // 1x1 conv to expand channels
            conv3: Conv2d::with_options(
                width,
                out_channels * Self::EXPANSION,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            ),
            bn3: BatchNorm2d::new(out_channels * Self::EXPANSION),
            downsample,
            relu: ReLU,
        }
    }
}

impl Module for Bottleneck {
    fn forward(&self, x: &Variable) -> Variable {
        let identity = x.clone();

        let out = self.conv1.forward(x);
        let out = self.bn1.forward(&out);
        let out = self.relu.forward(&out);

        let out = self.conv2.forward(&out);
        let out = self.bn2.forward(&out);
        let out = self.relu.forward(&out);

        let out = self.conv3.forward(&out);
        let out = self.bn3.forward(&out);

        let identity = match &self.downsample {
            Some((conv, bn)) => {
                let ds = conv.forward(&identity);
                bn.forward(&ds)
            }
            None => identity,
        };

        let out = out.add_var(&identity);
        self.relu.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());
        if let Some((conv, bn)) = &self.downsample {
            params.extend(conv.parameters());
            params.extend(bn.parameters());
        }
        params
    }

    fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        self.bn3.train();
        if let Some((_, bn)) = &mut self.downsample {
            bn.train();
        }
    }

    fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        self.bn3.eval();
        if let Some((_, bn)) = &mut self.downsample {
            bn.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.bn1.is_training()
    }
}

// =============================================================================
// ResNet
// =============================================================================

/// `ResNet` model for image classification.
pub struct ResNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
}

impl ResNet {
    /// Create `ResNet18`.
    #[must_use] pub fn resnet18(num_classes: usize) -> Self {
        Self::new_basic(&[2, 2, 2, 2], num_classes)
    }

    /// Create `ResNet34`.
    #[must_use] pub fn resnet34(num_classes: usize) -> Self {
        Self::new_basic(&[3, 4, 6, 3], num_classes)
    }

    /// Create `ResNet` with `BasicBlock`.
    fn new_basic(layers: &[usize; 4], num_classes: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(3, 64, (7, 7), (2, 2), (3, 3), true),
            bn1: BatchNorm2d::new(64),
            relu: ReLU,
            maxpool: MaxPool2d::with_options((3, 3), (2, 2), (1, 1)),
            layer1: Self::make_basic_layer(64, 64, layers[0], 1),
            layer2: Self::make_basic_layer(64, 128, layers[1], 2),
            layer3: Self::make_basic_layer(128, 256, layers[2], 2),
            layer4: Self::make_basic_layer(256, 512, layers[3], 2),
            avgpool: AdaptiveAvgPool2d::new((1, 1)),
            fc: Linear::new(512 * BasicBlock::EXPANSION, num_classes),
        }
    }

    fn make_basic_layer(
        in_channels: usize,
        out_channels: usize,
        blocks: usize,
        stride: usize,
    ) -> Vec<BasicBlock> {
        let mut layers = Vec::new();

        // First block may have stride and downsample
        let downsample = if stride != 1 || in_channels != out_channels {
            Some((
                Conv2d::with_options(
                    in_channels,
                    out_channels,
                    (1, 1),
                    (stride, stride),
                    (0, 0),
                    false,
                ),
                BatchNorm2d::new(out_channels),
            ))
        } else {
            None
        };

        layers.push(BasicBlock::new(
            in_channels,
            out_channels,
            stride,
            downsample,
        ));

        // Remaining blocks
        for _ in 1..blocks {
            layers.push(BasicBlock::new(out_channels, out_channels, 1, None));
        }

        layers
    }
}

impl Module for ResNet {
    fn forward(&self, x: &Variable) -> Variable {
        // Initial conv layer
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = self.relu.forward(&out);
        out = self.maxpool.forward(&out);

        // Residual layers
        for block in &self.layer1 {
            out = block.forward(&out);
        }
        for block in &self.layer2 {
            out = block.forward(&out);
        }
        for block in &self.layer3 {
            out = block.forward(&out);
        }
        for block in &self.layer4 {
            out = block.forward(&out);
        }

        // Classification head
        out = self.avgpool.forward(&out);
        // Flatten: [batch, channels, 1, 1] -> [batch, channels]
        out = flatten(&out);

        self.fc.forward(&out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        for block in &self.layer1 {
            params.extend(block.parameters());
        }
        for block in &self.layer2 {
            params.extend(block.parameters());
        }
        for block in &self.layer3 {
            params.extend(block.parameters());
        }
        for block in &self.layer4 {
            params.extend(block.parameters());
        }
        params.extend(self.fc.parameters());
        params
    }

    fn train(&mut self) {
        self.bn1.train();
        for block in &mut self.layer1 {
            block.train();
        }
        for block in &mut self.layer2 {
            block.train();
        }
        for block in &mut self.layer3 {
            block.train();
        }
        for block in &mut self.layer4 {
            block.train();
        }
    }

    fn eval(&mut self) {
        self.bn1.eval();
        for block in &mut self.layer1 {
            block.eval();
        }
        for block in &mut self.layer2 {
            block.eval();
        }
        for block in &mut self.layer3 {
            block.eval();
        }
        for block in &mut self.layer4 {
            block.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.bn1.is_training()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create `ResNet18` for `ImageNet` (1000 classes).
#[must_use] pub fn resnet18() -> ResNet {
    ResNet::resnet18(1000)
}

/// Create `ResNet34` for `ImageNet` (1000 classes).
#[must_use] pub fn resnet34() -> ResNet {
    ResNet::resnet34(1000)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block() {
        let block = BasicBlock::new(64, 64, 1, None);

        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 64 * 8 * 8], &[1, 64, 8, 8]).unwrap(),
            false,
        );

        let output = block.forward(&input);
        assert_eq!(output.data().shape(), &[1, 64, 8, 8]);
    }

    #[test]
    fn test_basic_block_with_downsample() {
        let downsample = (
            Conv2d::with_options(64, 128, (1, 1), (2, 2), (0, 0), false),
            BatchNorm2d::new(128),
        );

        let block = BasicBlock::new(64, 128, 2, Some(downsample));

        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 64 * 8 * 8], &[1, 64, 8, 8]).unwrap(),
            false,
        );

        let output = block.forward(&input);
        assert_eq!(output.data().shape(), &[1, 128, 4, 4]);
    }

    #[test]
    fn test_resnet18_creation() {
        let model = ResNet::resnet18(10);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_resnet18_forward_small() {
        let model = ResNet::resnet18(10);

        // Small input for quick test
        let input = Variable::new(
            Tensor::from_vec(vec![0.0; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );

        let output = model.forward(&input);
        assert_eq!(output.data().shape()[0], 1);
        assert_eq!(output.data().shape()[1], 10);
    }
}
