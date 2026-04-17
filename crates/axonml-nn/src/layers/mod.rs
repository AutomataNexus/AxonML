//! Neural network layer modules — 15 submodules re-exported here.
//!
//! `linear`, `conv` (Conv1d/2d/Transpose2d), `norm` (BatchNorm1d/2d,
//! LayerNorm, GroupNorm, InstanceNorm2d, RMSNorm), `dropout` (Dropout/
//! Dropout2d), `rnn` (RNN/LSTM/GRU + cell variants), `attention`
//! (MultiHead, Cross, fused scaled-dot-product), `diff_attention`
//! (DifferentialAttention), `embedding`, `ternary` (TernaryLinear for
//! 1.58-bit weights), `pooling` (MaxPool/AvgPool/AdaptiveAvgPool),
//! `residual` (ResidualBlock), `moe` (MixtureOfExperts with top-k gating),
//! `graph` (GCN/GAT), `fft` (FFT/STFT), `sparse` (SparseLinear,
//! GroupSparsity, LotteryTicket), `transformer` (encoder/decoder stacks,
//! Seq2SeqTransformer).
//!
//! # File
//! `crates/axonml-nn/src/layers/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Sub-Modules
// =============================================================================

pub mod attention;
pub mod conv;
pub mod diff_attention;
pub mod dropout;
pub mod embedding;
pub mod fft;
pub mod graph;
pub mod linear;
pub mod moe;
pub mod norm;
pub mod pooling;
pub mod residual;
pub mod rnn;
pub mod sparse;
pub mod ternary;
pub mod transformer;

// =============================================================================
// Re-Exports
// =============================================================================

pub use attention::{CrossAttention, MultiHeadAttention, scaled_dot_product_attention_fused};
pub use conv::{Conv1d, Conv2d, ConvTranspose2d};
pub use diff_attention::DifferentialAttention;
pub use dropout::{AlphaDropout, Dropout, Dropout2d};
pub use embedding::Embedding;
pub use fft::{FFT1d, STFT};
pub use graph::{GATConv, GCNConv};
pub use linear::Linear;
pub use moe::{Expert, MoELayer, MoERouter};
pub use norm::{BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm2d, LayerNorm};
pub use pooling::{AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d};
pub use residual::ResidualBlock;
pub use rnn::{GRU, GRUCell, LSTM, LSTMCell, RNN, RNNCell};
pub use sparse::{GroupSparsity, LotteryTicket, SparseLinear};
pub use ternary::{PackedTernaryWeights, TernaryLinear};
pub use transformer::{
    Seq2SeqTransformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer,
};
