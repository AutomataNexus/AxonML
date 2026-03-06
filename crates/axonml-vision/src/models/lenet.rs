//! `LeNet` - Classic CNN Architecture
//!
//! Implementation of LeNet-5, one of the earliest successful CNNs.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::any::Any;

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_nn::{Conv2d, Linear, Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// LeNet-5
// =============================================================================

/// LeNet-5 architecture for MNIST digit classification.
///
/// Architecture:
/// - Conv2d(1, 6, 5) -> `ReLU` -> MaxPool2d(2)
/// - Conv2d(6, 16, 5) -> `ReLU` -> MaxPool2d(2)
/// - Flatten
/// - Linear(256, 120) -> `ReLU`
/// - Linear(120, 84) -> `ReLU`
/// - Linear(84, 10)
pub struct LeNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl LeNet {
    /// Creates a new LeNet-5 for MNIST (28x28 input, 10 classes).
    #[must_use]
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 6, 5),       // 28x28 -> 24x24
            conv2: Conv2d::new(6, 16, 5),      // 12x12 -> 8x8 (after pool)
            fc1: Linear::new(16 * 4 * 4, 120), // After 2 pools: 8x8 -> 4x4
            fc2: Linear::new(120, 84),
            fc3: Linear::new(84, 10),
        }
    }

    /// Creates a `LeNet` for CIFAR-10 (32x32 input, 10 classes).
    #[must_use]
    pub fn for_cifar10() -> Self {
        Self {
            conv1: Conv2d::new(3, 6, 5),       // 32x32 -> 28x28
            conv2: Conv2d::new(6, 16, 5),      // 14x14 -> 10x10 (after pool)
            fc1: Linear::new(16 * 5 * 5, 120), // After 2 pools: 10x10 -> 5x5
            fc2: Linear::new(120, 84),
            fc3: Linear::new(84, 10),
        }
    }

    /// Max pooling 2x2 operation.
    fn max_pool2d(&self, input: &Variable, kernel_size: usize) -> Variable {
        let data = input.data();
        let shape = data.shape();

        if shape.len() == 4 {
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let out_h = h / kernel_size;
            let out_w = w / kernel_size;

            let data_vec = data.to_vec();
            let out_size = n * c * out_h * out_w;
            let mut result = vec![0.0f32; out_size];
            let mut max_indices = vec![0usize; out_size];

            for batch in 0..n {
                for ch in 0..c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut max_val = f32::NEG_INFINITY;
                            let mut max_idx = 0usize;
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    let ih = oh * kernel_size + kh;
                                    let iw = ow * kernel_size + kw;
                                    let idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                    if data_vec[idx] > max_val {
                                        max_val = data_vec[idx];
                                        max_idx = idx;
                                    }
                                }
                            }
                            let out_idx =
                                batch * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow;
                            result[out_idx] = max_val;
                            max_indices[out_idx] = max_idx;
                        }
                    }
                }
            }

            let output_tensor = Tensor::from_vec(result, &[n, c, out_h, out_w]).unwrap();
            if input.requires_grad() && is_grad_enabled() {
                let grad_fn = GradFn::new(MaxPool2dBackward {
                    next_fns: vec![input.grad_fn().cloned()],
                    max_indices,
                    input_shape: shape.to_vec(),
                });
                Variable::from_operation(output_tensor, grad_fn, true)
            } else {
                Variable::new(output_tensor, false)
            }
        } else if shape.len() == 3 {
            // Single image without batch
            let (c, h, w) = (shape[0], shape[1], shape[2]);
            let out_h = h / kernel_size;
            let out_w = w / kernel_size;

            let data_vec = data.to_vec();
            let out_size = c * out_h * out_w;
            let mut result = vec![0.0f32; out_size];
            let mut max_indices = vec![0usize; out_size];

            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0usize;
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * kernel_size + kh;
                                let iw = ow * kernel_size + kw;
                                let idx = ch * h * w + ih * w + iw;
                                if data_vec[idx] > max_val {
                                    max_val = data_vec[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        let out_idx = ch * out_h * out_w + oh * out_w + ow;
                        result[out_idx] = max_val;
                        max_indices[out_idx] = max_idx;
                    }
                }
            }

            let output_tensor = Tensor::from_vec(result, &[c, out_h, out_w]).unwrap();
            if input.requires_grad() && is_grad_enabled() {
                let grad_fn = GradFn::new(MaxPool2dBackward {
                    next_fns: vec![input.grad_fn().cloned()],
                    max_indices,
                    input_shape: shape.to_vec(),
                });
                Variable::from_operation(output_tensor, grad_fn, true)
            } else {
                Variable::new(output_tensor, false)
            }
        } else {
            input.clone()
        }
    }

    /// Flattens a tensor to 2D (batch, features).
    /// Uses Variable::reshape() to preserve the autograd graph.
    fn flatten(&self, input: &Variable) -> Variable {
        let shape = input.shape();

        if shape.len() <= 2 {
            return input.clone();
        }

        let batch_size = shape[0];
        let features: usize = shape[1..].iter().product();

        input.reshape(&[batch_size, features])
    }
}

impl Default for LeNet {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LeNet {
    fn forward(&self, input: &Variable) -> Variable {
        // Conv1 -> ReLU -> Pool
        let x = self.conv1.forward(input);
        let x = x.relu();
        let x = self.max_pool2d(&x, 2);

        // Conv2 -> ReLU -> Pool
        let x = self.conv2.forward(&x);
        let x = x.relu();
        let x = self.max_pool2d(&x, 2);

        // Flatten
        let x = self.flatten(&x);

        // FC layers
        let x = self.fc1.forward(&x);
        let x = x.relu();
        let x = self.fc2.forward(&x);
        let x = x.relu();
        self.fc3.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    fn train(&mut self) {
        // LeNet has no training-mode-specific behavior
    }

    fn eval(&mut self) {
        // LeNet has no eval-mode-specific behavior
    }
}

// =============================================================================
// SimpleCNN
// =============================================================================

/// A simple CNN for quick experiments.
pub struct SimpleCNN {
    conv1: Conv2d,
    fc1: Linear,
    fc2: Linear,
    input_channels: usize,
    num_classes: usize,
}

impl SimpleCNN {
    /// Creates a new `SimpleCNN`.
    /// Note: Conv2d with kernel 3 and no padding: 28-3+1=26, after pool: 13
    #[must_use]
    pub fn new(input_channels: usize, num_classes: usize) -> Self {
        Self {
            conv1: Conv2d::new(input_channels, 32, 3),
            fc1: Linear::new(32 * 13 * 13, 128), // 28x28 -> 26x26 (conv) -> 13x13 (pool)
            fc2: Linear::new(128, num_classes),
            input_channels,
            num_classes,
        }
    }

    /// Creates a `SimpleCNN` for MNIST.
    #[must_use]
    pub fn for_mnist() -> Self {
        Self::new(1, 10)
    }

    /// Creates a `SimpleCNN` for CIFAR-10.
    #[must_use]
    pub fn for_cifar10() -> Self {
        // 32x32 -> 30x30 (conv with k=3) -> 15x15 (pool)
        Self {
            conv1: Conv2d::new(3, 32, 3),
            fc1: Linear::new(32 * 15 * 15, 128),
            fc2: Linear::new(128, 10),
            input_channels: 3,
            num_classes: 10,
        }
    }

    /// Returns the number of input channels.
    #[must_use]
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Returns the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn max_pool2d(&self, input: &Variable, kernel_size: usize) -> Variable {
        let data = input.data();
        let shape = data.shape();

        if shape.len() != 4 {
            return input.clone();
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_h = h / kernel_size;
        let out_w = w / kernel_size;

        let data_vec = data.to_vec();
        let mut result = vec![0.0f32; n * c * out_h * out_w];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * kernel_size + kh;
                                let iw = ow * kernel_size + kw;
                                let idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                max_val = max_val.max(data_vec[idx]);
                            }
                        }
                        let out_idx =
                            batch * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow;
                        result[out_idx] = max_val;
                    }
                }
            }
        }

        Variable::new(
            Tensor::from_vec(result, &[n, c, out_h, out_w]).unwrap(),
            input.requires_grad(),
        )
    }

    fn flatten(&self, input: &Variable) -> Variable {
        let shape = input.shape();

        if shape.len() <= 2 {
            return input.clone();
        }

        let batch_size = shape[0];
        let features: usize = shape[1..].iter().product();

        input.reshape(&[batch_size, features])
    }
}

impl Module for SimpleCNN {
    fn forward(&self, input: &Variable) -> Variable {
        let x = self.conv1.forward(input);
        let x = x.relu();
        let x = self.max_pool2d(&x, 2);
        let x = self.flatten(&x);
        let x = self.fc1.forward(&x);
        let x = x.relu();
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// MLP for classification
// =============================================================================

/// A simple MLP for classification (flattened input).
pub struct MLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MLP {
    /// Creates a new MLP.
    #[must_use]
    pub fn new(input_size: usize, hidden_size: usize, num_classes: usize) -> Self {
        Self {
            fc1: Linear::new(input_size, hidden_size),
            fc2: Linear::new(hidden_size, hidden_size / 2),
            fc3: Linear::new(hidden_size / 2, num_classes),
        }
    }

    /// Creates an MLP for MNIST (784 -> 256 -> 128 -> 10).
    #[must_use]
    pub fn for_mnist() -> Self {
        Self::new(784, 256, 10)
    }

    /// Creates an MLP for CIFAR-10 (3072 -> 512 -> 256 -> 10).
    #[must_use]
    pub fn for_cifar10() -> Self {
        Self::new(3072, 512, 10)
    }
}

impl Module for MLP {
    fn forward(&self, input: &Variable) -> Variable {
        // Flatten if needed
        let data = input.data();
        let shape = data.shape();
        let x = if shape.len() > 2 {
            let batch = shape[0];
            let features: usize = shape[1..].iter().product();
            Variable::new(
                Tensor::from_vec(data.to_vec(), &[batch, features]).unwrap(),
                input.requires_grad(),
            )
        } else if shape.len() == 1 {
            // Add batch dimension
            Variable::new(
                Tensor::from_vec(data.to_vec(), &[1, shape[0]]).unwrap(),
                input.requires_grad(),
            )
        } else {
            input.clone()
        };

        let x = self.fc1.forward(&x);
        let x = x.relu();
        let x = self.fc2.forward(&x);
        let x = x.relu();
        self.fc3.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// MaxPool2dBackward
// =============================================================================

/// Gradient function for MaxPool2d.
///
/// Backward pass routes gradient only to the max element in each pooling window.
#[derive(Debug)]
struct MaxPool2dBackward {
    next_fns: Vec<Option<GradFn>>,
    max_indices: Vec<usize>,
    input_shape: Vec<usize>,
}

impl GradientFunction for MaxPool2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let g_vec = grad_output.to_vec();
        let input_size: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; input_size];

        for (i, &idx) in self.max_indices.iter().enumerate() {
            if i < g_vec.len() {
                grad_input[idx] += g_vec[i];
            }
        }

        let gi = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "MaxPool2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lenet_creation() {
        let model = LeNet::new();
        let params = model.parameters();

        // Should have parameters from 2 conv + 3 fc layers
        assert!(!params.is_empty());
    }

    #[test]
    fn test_lenet_forward() {
        let model = LeNet::new();

        // Create a batch of 2 MNIST images
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 28 * 28], &[2, 1, 28, 28]).unwrap(),
            false,
        );

        let output = model.forward(&input);
        assert_eq!(output.data().shape(), &[2, 10]);
    }

    #[test]
    fn test_simple_cnn_mnist() {
        let model = SimpleCNN::for_mnist();

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 28 * 28], &[2, 1, 28, 28]).unwrap(),
            false,
        );

        let output = model.forward(&input);
        assert_eq!(output.data().shape(), &[2, 10]);
    }

    #[test]
    fn test_mlp_mnist() {
        let model = MLP::for_mnist();

        // Flattened MNIST input
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 784], &[2, 784]).unwrap(),
            false,
        );

        let output = model.forward(&input);
        assert_eq!(output.data().shape(), &[2, 10]);
    }

    #[test]
    fn test_mlp_auto_flatten() {
        let model = MLP::for_mnist();

        // 4D input (like image)
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 28 * 28], &[2, 1, 28, 28]).unwrap(),
            false,
        );

        let output = model.forward(&input);
        assert_eq!(output.data().shape(), &[2, 10]);
    }

    #[test]
    fn test_lenet_parameter_count() {
        let model = LeNet::new();
        let params = model.parameters();

        // Count total parameters
        let total: usize = params
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();

        // LeNet-5 should have around 44k parameters for MNIST
        assert!(total > 40000 && total < 100000);
    }
}
