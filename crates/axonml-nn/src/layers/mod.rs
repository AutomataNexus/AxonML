//! Neural Network Layers
//!
//! # File
//! `crates/axonml-nn/src/layers/mod.rs`
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

pub mod attention;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod fft;
pub mod graph;
pub mod linear;
pub mod norm;
pub mod pooling;
pub mod residual;
pub mod rnn;
pub mod sparse;
pub mod transformer;

// Re-exports
pub use attention::{scaled_dot_product_attention_fused, CrossAttention, MultiHeadAttention};
pub use conv::{Conv1d, Conv2d, ConvTranspose2d};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use fft::{FFT1d, STFT};
pub use graph::{GATConv, GCNConv};
pub use linear::Linear;
pub use norm::{BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm2d, LayerNorm};
pub use pooling::{AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d};
pub use residual::ResidualBlock;
pub use rnn::{GRUCell, LSTMCell, RNNCell, GRU, LSTM, RNN};
pub use sparse::{GroupSparsity, LotteryTicket, SparseLinear};
pub use transformer::{
    Seq2SeqTransformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer,
};
