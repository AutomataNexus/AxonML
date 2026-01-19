//! Neural Network Layers
//!
//! Contains all the standard neural network layer implementations.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

pub mod attention;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod norm;
pub mod pooling;
pub mod rnn;

// Re-exports
pub use attention::MultiHeadAttention;
pub use conv::{Conv1d, Conv2d};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use norm::{BatchNorm1d, BatchNorm2d, LayerNorm};
pub use pooling::{AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d};
pub use rnn::{GRUCell, LSTMCell, RNNCell, GRU, LSTM, RNN};
