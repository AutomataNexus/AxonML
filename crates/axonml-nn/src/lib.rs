//! Neural network building blocks for AxonML.
//!
//! `Module` trait (forward, parameters, train/eval, zero_grad), `Parameter`
//! (named gradient-tracked weight), `Sequential` container, 40+ layer types
//! in `layers` (Linear, Conv1d/2d, ConvTranspose2d, MaxPool/AvgPool/
//! AdaptiveAvgPool, BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d,
//! RMSNorm, Dropout/Dropout2d, RNN/LSTM/GRU + cell variants, MultiHead/
//! Cross/DifferentialAttention, Embedding, TernaryLinear, Transformer
//! encoder/decoder, Seq2SeqTransformer, ResidualBlock, MoE, GCN/GAT,
//! FFT/STFT), activations (ReLU, Sigmoid, Tanh, GELU, SiLU, ELU,
//! LeakyReLU, Mish, Softmax, LogSoftmax), losses (MSE, CrossEntropy, BCE,
//! BCEWithLogits, L1, SmoothL1, NLL, CTC, Focal, Triplet, ArcFace),
//! initialization (Xavier, Kaiming, Glorot, He, orthogonal, sparse),
//! differentiable structured sparsity (SparseLinear, GroupSparsity,
//! LotteryTicket), and `functional` helpers.
//!
//! # File
//! `crates/axonml-nn/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

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
    AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, BatchNorm1d, BatchNorm2d, Conv1d, Conv2d,
    ConvTranspose2d, CrossAttention, DifferentialAttention, Dropout, Embedding, Expert, FFT1d,
    GATConv, GCNConv, GRU, GRUCell, GroupNorm, GroupSparsity, InstanceNorm2d, LSTM, LSTMCell,
    LayerNorm, Linear, LotteryTicket, MaxPool1d, MaxPool2d, MoELayer, MoERouter,
    MultiHeadAttention, PackedTernaryWeights, RNN, RNNCell, ResidualBlock, STFT,
    Seq2SeqTransformer, SparseLinear, TernaryLinear, TransformerDecoder, TransformerDecoderLayer,
    TransformerEncoder, TransformerEncoderLayer,
};

// Activation re-exports
pub use activation::{
    ELU, Flatten, GELU, Identity, LeakyReLU, LogSoftmax, ReLU, SiLU, Sigmoid, Softmax, Tanh,
};

// Loss re-exports
pub use loss::{
    BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, NLLLoss, Reduction, SmoothL1Loss,
};

// Init re-exports
pub use init::{
    InitMode, constant, diag, eye, glorot_normal, glorot_uniform, he_normal, he_uniform,
    kaiming_normal, kaiming_uniform, normal, ones, orthogonal, randn, sparse, uniform,
    uniform_range, xavier_normal, xavier_uniform, zeros,
};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for neural network development.
pub mod prelude {
    pub use crate::{
        AdaptiveAvgPool2d,
        AvgPool1d,
        AvgPool2d,
        BCELoss,
        BatchNorm1d,
        BatchNorm2d,
        Conv1d,
        Conv2d,
        CrossAttention,
        CrossEntropyLoss,
        Dropout,
        ELU,
        Embedding,
        GELU,
        GRU,
        GroupNorm,
        Identity,
        InstanceNorm2d,
        L1Loss,
        LSTM,
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
        RNN,
        // Activations
        ReLU,
        // Loss functions
        Reduction,
        Seq2SeqTransformer,
        Sequential,
        SiLU,
        Sigmoid,
        Softmax,
        Tanh,
        TransformerDecoder,
        TransformerDecoderLayer,
        TransformerEncoder,
        TransformerEncoderLayer,
        // Functional
        functional,
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

        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 20], &[2, 10]).expect("tensor creation failed"),
            false,
        );
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
            Tensor::from_vec(vec![1.0; 784], &[1, 1, 28, 28]).expect("tensor creation failed"),
            false,
        );
        let output = model.forward(&input);
        // Conv2d: 28 -> 26, MaxPool2d: 26 -> 13
        assert_eq!(output.shape(), vec![1, 16, 13, 13]);
    }

    #[test]
    fn test_loss_computation() {
        let pred = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
                .expect("tensor creation failed"),
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(vec![0.0, 2.0], &[2]).expect("tensor creation failed"),
            false,
        );

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(&pred, &target);
        assert!(loss.numel() == 1);
    }

    #[test]
    fn test_embedding_model() {
        let emb = Embedding::new(100, 32);
        let indices = Variable::new(
            Tensor::from_vec(vec![0.0, 5.0, 10.0, 15.0], &[2, 2]).expect("tensor creation failed"),
            false,
        );
        let output = emb.forward(&indices);
        assert_eq!(output.shape(), vec![2, 2, 32]);
    }

    #[test]
    fn test_rnn_model() {
        let rnn = LSTM::new(10, 20, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 60], &[2, 3, 10]).expect("tensor creation failed"),
            false,
        );
        let output = rnn.forward(&input);
        assert_eq!(output.shape(), vec![2, 3, 20]);
    }

    #[test]
    fn test_attention_model() {
        let attn = MultiHeadAttention::new(64, 4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 640], &[2, 5, 64]).expect("tensor creation failed"),
            false,
        );
        let output = attn.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 64]);
    }
}
