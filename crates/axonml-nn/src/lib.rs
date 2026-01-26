//! axonml-nn - Neural Network Module Library
//!
//! Provides neural network layers, activation functions, loss functions,
//! and utilities for building deep learning models in Axonml.
//!
//! # Key Components
//!
//! - **Module trait**: Core interface for all neural network modules
//! - **Parameter**: Wrapper for learnable parameters
//! - **Sequential**: Container for chaining modules
//! - **Layers**: Linear, Conv, RNN, LSTM, Attention, etc.
//! - **Activations**: ReLU, Sigmoid, Tanh, GELU, etc.
//! - **Loss Functions**: MSE, CrossEntropy, BCE, etc.
//! - **Initialization**: Xavier, Kaiming, orthogonal, etc.
//! - **Functional API**: Stateless operations
//!
//! # Example
//!
//! ```ignore
//! use axonml_nn::prelude::*;
//!
//! // Build a simple MLP
//! let model = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(ReLU)
//!     .add(Linear::new(256, 10));
//!
//! // Forward pass
//! let output = model.forward(&input);
//!
//! // Compute loss
//! let loss = CrossEntropyLoss::new().compute(&output, &target);
//!
//! // Backward pass
//! loss.backward();
//! ```
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// ML/tensor-specific allowances
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unused_self)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::single_match_else)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::format_push_string)]
#![allow(clippy::erasing_op)]
#![allow(clippy::type_repetition_in_bounds)]
#![allow(clippy::iter_without_into_iter)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::use_debug)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::panic)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::ref_option)]
#![allow(clippy::multiple_bound_locations)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::manual_assert)]
#![allow(clippy::unnecessary_debug_formatting)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod activation;
pub mod functional;
pub mod init;
pub mod layers;
pub mod loss;
pub mod module;
pub mod parameter;
pub mod sequential;

// =============================================================================
// Re-exports
// =============================================================================

pub use module::{Module, ModuleList};
pub use parameter::Parameter;
pub use sequential::Sequential;

// Layer re-exports
pub use layers::{
    AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, Dropout,
    Embedding, GRUCell, GroupNorm, InstanceNorm2d, LSTMCell, LayerNorm, Linear, MaxPool1d,
    MaxPool2d, MultiHeadAttention, RNNCell, GRU, LSTM, RNN,
};

// Activation re-exports
pub use activation::{
    Identity, LeakyReLU, LogSoftmax, ReLU, SiLU, Sigmoid, Softmax, Tanh, ELU, GELU,
};

// Loss re-exports
pub use loss::{
    BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, NLLLoss, Reduction, SmoothL1Loss,
};

// Init re-exports
pub use init::{
    constant, diag, eye, glorot_normal, glorot_uniform, he_normal, he_uniform, kaiming_normal,
    kaiming_uniform, normal, ones, orthogonal, randn, sparse, uniform, uniform_range,
    xavier_normal, xavier_uniform, zeros, InitMode,
};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for neural network development.
pub mod prelude {
    pub use crate::{
        // Functional
        functional,
        AdaptiveAvgPool2d,
        AvgPool1d,
        AvgPool2d,
        BCELoss,
        BatchNorm1d,
        BatchNorm2d,
        Conv1d,
        Conv2d,
        CrossEntropyLoss,
        Dropout,
        Embedding,
        GroupNorm,
        Identity,
        InstanceNorm2d,
        L1Loss,
        LayerNorm,
        LeakyReLU,
        // Layers
        Linear,
        MSELoss,
        MaxPool1d,
        MaxPool2d,
        // Core traits and types
        Module,
        ModuleList,
        MultiHeadAttention,
        NLLLoss,
        Parameter,
        // Activations
        ReLU,
        // Loss functions
        Reduction,
        Sequential,
        SiLU,
        Sigmoid,
        Softmax,
        Tanh,
        ELU,
        GELU,
        GRU,
        LSTM,
        RNN,
    };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_autograd::Variable;
    use axonml_tensor::Tensor;

    #[test]
    fn test_simple_mlp() {
        let model = Sequential::new()
            .add(Linear::new(10, 5))
            .add(ReLU)
            .add(Linear::new(5, 2));

        let input = Variable::new(Tensor::from_vec(vec![1.0; 20], &[2, 10]).unwrap(), false);
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![2, 2]);
    }

    #[test]
    fn test_module_parameters() {
        let model = Sequential::new()
            .add(Linear::new(10, 5))
            .add(Linear::new(5, 2));

        let params = model.parameters();
        // 2 Linear layers with weight + bias each = 4 parameters
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_conv_model() {
        let model = Sequential::new()
            .add(Conv2d::new(1, 16, 3))
            .add(ReLU)
            .add(MaxPool2d::new(2));

        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 784], &[1, 1, 28, 28]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        // Conv2d: 28 -> 26, MaxPool2d: 26 -> 13
        assert_eq!(output.shape(), vec![1, 16, 13, 13]);
    }

    #[test]
    fn test_loss_computation() {
        let pred = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
            true,
        );
        let target = Variable::new(Tensor::from_vec(vec![0.0, 2.0], &[2]).unwrap(), false);

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(&pred, &target);
        assert!(loss.numel() == 1);
    }

    #[test]
    fn test_embedding_model() {
        let emb = Embedding::new(100, 32);
        let indices = Variable::new(
            Tensor::from_vec(vec![0.0, 5.0, 10.0, 15.0], &[2, 2]).unwrap(),
            false,
        );
        let output = emb.forward(&indices);
        assert_eq!(output.shape(), vec![2, 2, 32]);
    }

    #[test]
    fn test_rnn_model() {
        let rnn = LSTM::new(10, 20, 1);
        let input = Variable::new(Tensor::from_vec(vec![1.0; 60], &[2, 3, 10]).unwrap(), false);
        let output = rnn.forward(&input);
        assert_eq!(output.shape(), vec![2, 3, 20]);
    }

    #[test]
    fn test_attention_model() {
        let attn = MultiHeadAttention::new(64, 4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 640], &[2, 5, 64]).unwrap(),
            false,
        );
        let output = attn.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }
}
