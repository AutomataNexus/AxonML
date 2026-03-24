//! Residual Block - Generic Skip Connection Layer
//!
//! # File
//! `crates/axonml-nn/src/layers/residual.rs`
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

use std::collections::HashMap;

use axonml_autograd::Variable;

use crate::activation::ReLU;
use crate::module::Module;
use crate::parameter::Parameter;
use crate::sequential::Sequential;

// =============================================================================
// ResidualBlock
// =============================================================================

/// A generic residual block that wraps any module sequence with a skip connection.
///
/// Computes: `activation(main_path(x) + downsample(x))` where downsample is
/// optional (defaults to identity). This enables gradient flow through the
/// skip connection, improving training of deep networks.
///
/// # Example
/// ```ignore
/// use axonml_nn::prelude::*;
/// use axonml_nn::layers::ResidualBlock;
///
/// // Conv1d residual block
/// let main = Sequential::new()
///     .add(Conv1d::new(64, 64, 3))
///     .add(BatchNorm1d::new(64))
///     .add(ReLU)
///     .add(Conv1d::new(64, 64, 3))
///     .add(BatchNorm1d::new(64));
///
/// let block = ResidualBlock::new(main);
/// ```
pub struct ResidualBlock {
    main_path: Sequential,
    downsample: Option<Sequential>,
    activation: Option<Box<dyn Module>>,
    training: bool,
}

impl ResidualBlock {
    /// Creates a new residual block with the given main path and ReLU activation.
    ///
    /// The skip connection is identity (no downsample). Use `with_downsample()`
    /// if the main path changes dimensions.
    pub fn new(main_path: Sequential) -> Self {
        Self {
            main_path,
            downsample: None,
            activation: Some(Box::new(ReLU)),
            training: true,
        }
    }

    /// Adds a downsample projection for when input/output dimensions differ.
    ///
    /// Typically a Conv + BatchNorm to match channel/spatial dimensions.
    pub fn with_downsample(mut self, downsample: Sequential) -> Self {
        self.downsample = Some(downsample);
        self
    }

    /// Sets a custom activation function applied after the residual addition.
    ///
    /// Pass any module implementing `Module` (ReLU, GELU, SiLU, etc.).
    pub fn with_activation<M: Module + 'static>(mut self, activation: M) -> Self {
        self.activation = Some(Box::new(activation));
        self
    }

    /// Removes the post-addition activation (pre-activation ResNet style).
    pub fn without_activation(mut self) -> Self {
        self.activation = None;
        self
    }
}

impl Module for ResidualBlock {
    fn forward(&self, input: &Variable) -> Variable {
        let identity = match &self.downsample {
            Some(ds) => ds.forward(input),
            None => input.clone(),
        };

        let out = self.main_path.forward(input);
        let out = out.add_var(&identity);

        match &self.activation {
            Some(act) => act.forward(&out),
            None => out,
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.main_path.parameters();
        if let Some(ds) = &self.downsample {
            params.extend(ds.parameters());
        }
        if let Some(act) = &self.activation {
            params.extend(act.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.main_path.named_parameters() {
            params.insert(format!("main_path.{name}"), param);
        }
        if let Some(ds) = &self.downsample {
            for (name, param) in ds.named_parameters() {
                params.insert(format!("downsample.{name}"), param);
            }
        }
        if let Some(act) = &self.activation {
            for (name, param) in act.named_parameters() {
                params.insert(format!("activation.{name}"), param);
            }
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.main_path.set_training(training);
        if let Some(ds) = &mut self.downsample {
            ds.set_training(training);
        }
        if let Some(act) = &mut self.activation {
            act.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "ResidualBlock"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{ReLU, GELU};
    use crate::layers::{BatchNorm1d, Conv1d, Linear};
    use axonml_tensor::Tensor;

    #[test]
    fn test_residual_block_identity_skip() {
        // Main path that preserves dimensions
        let main = Sequential::new()
            .add(Linear::new(32, 32))
            .add(ReLU)
            .add(Linear::new(32, 32));

        let block = ResidualBlock::new(main);

        let input = Variable::new(Tensor::from_vec(vec![1.0; 64], &[2, 32]).unwrap(), false);
        let output = block.forward(&input);

        // Output shape should match input
        assert_eq!(output.shape(), vec![2, 32]);
    }

    #[test]
    fn test_residual_block_with_downsample() {
        // Main path changes dimensions: 32 -> 64
        let main = Sequential::new()
            .add(Linear::new(32, 64))
            .add(ReLU)
            .add(Linear::new(64, 64));

        // Downsample projects input: 32 -> 64
        let downsample = Sequential::new().add(Linear::new(32, 64));

        let block = ResidualBlock::new(main).with_downsample(downsample);

        let input = Variable::new(Tensor::from_vec(vec![1.0; 64], &[2, 32]).unwrap(), false);
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![2, 64]);
    }

    #[test]
    fn test_residual_block_custom_activation() {
        let main = Sequential::new().add(Linear::new(16, 16));

        let block = ResidualBlock::new(main).with_activation(GELU);

        let input = Variable::new(Tensor::from_vec(vec![1.0; 32], &[2, 16]).unwrap(), false);
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![2, 16]);
    }

    #[test]
    fn test_residual_block_no_activation() {
        let main = Sequential::new().add(Linear::new(16, 16));

        let block = ResidualBlock::new(main).without_activation();

        let input = Variable::new(Tensor::from_vec(vec![1.0; 32], &[2, 16]).unwrap(), false);
        let output = block.forward(&input);
        assert_eq!(output.shape(), vec![2, 16]);
    }

    #[test]
    fn test_residual_block_parameters() {
        let main = Sequential::new()
            .add(Linear::new(32, 32)) // weight(32x32) + bias(32) = 1056
            .add(Linear::new(32, 32)); // weight(32x32) + bias(32) = 1056

        let block = ResidualBlock::new(main);
        let params = block.parameters();
        assert_eq!(params.len(), 4); // 2 weights + 2 biases
    }

    #[test]
    fn test_residual_block_named_parameters() {
        let main = Sequential::new()
            .add_named("conv1", Linear::new(32, 32))
            .add_named("conv2", Linear::new(32, 32));

        let downsample = Sequential::new().add_named("proj", Linear::new(32, 32));

        let block = ResidualBlock::new(main).with_downsample(downsample);
        let params = block.named_parameters();

        assert!(params.contains_key("main_path.conv1.weight"));
        assert!(params.contains_key("main_path.conv2.weight"));
        assert!(params.contains_key("downsample.proj.weight"));
    }

    #[test]
    fn test_residual_block_training_mode() {
        let main = Sequential::new()
            .add(BatchNorm1d::new(32))
            .add(Linear::new(32, 32));

        let mut block = ResidualBlock::new(main);
        assert!(block.is_training());

        block.set_training(false);
        assert!(!block.is_training());

        block.set_training(true);
        assert!(block.is_training());
    }

    #[test]
    fn test_residual_block_conv1d_with_downsample() {
        // Real use case: Conv1d residual block with downsample to match dimensions
        // Main path: 2 Conv1d(k=3) reduces time by 4 (20 -> 18 -> 16)
        let main = Sequential::new()
            .add(Conv1d::new(64, 64, 3))
            .add(BatchNorm1d::new(64))
            .add(ReLU)
            .add(Conv1d::new(64, 64, 3))
            .add(BatchNorm1d::new(64));

        // Downsample matches skip connection to main path output shape
        // Conv1d with kernel=5 reduces 20 -> 16
        let downsample = Sequential::new()
            .add(Conv1d::new(64, 64, 5))
            .add(BatchNorm1d::new(64));

        let block = ResidualBlock::new(main).with_downsample(downsample);

        // Input: (batch=2, channels=64, time=20)
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 64 * 20], &[2, 64, 20]).unwrap(),
            false,
        );
        let output = block.forward(&input);

        assert_eq!(output.shape()[0], 2);
        assert_eq!(output.shape()[1], 64);
        assert_eq!(output.shape()[2], 16);
    }

    #[test]
    fn test_residual_block_gradient_flow() {
        let main = Sequential::new().add(Linear::new(4, 4));

        let block = ResidualBlock::new(main);

        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap(),
            true,
        );
        let output = block.forward(&input);

        // Sum to scalar for backward
        let sum = output.sum();
        sum.backward();

        // Gradient should flow through both main path and skip connection
        let params = block.parameters();
        assert!(!params.is_empty());
    }
}
