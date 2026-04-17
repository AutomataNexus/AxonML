//! Gradient function implementations for all differentiable operations.
//!
//! Seven submodules, each containing `*Backward` structs implementing
//! `GradientFunction`: `basic` (add, sub, mul, div, neg, scalar ops, reshape,
//! transpose, narrow, select, unsqueeze, expand, cat, clamp, exp, log, sqrt,
//! pow, sum, mean, sum_dim, mean_dim, var_dim), `activation` (relu, sigmoid,
//! tanh, gelu, silu, elu, leaky_relu, softmax, log_softmax), `linalg`
//! (matmul), `loss` (mse, cross_entropy, bce, bce_with_logits, l1, smooth_l1,
//! nll), `conv` (conv1d, conv2d with BLAS-accelerated backward), `rnn`
//! (lstm, gru, rnn cell backward), `attention` (multi-head attention backward).
//!
//! # File
//! `crates/axonml-autograd/src/functions/mod.rs`
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

mod activation;
mod attention;
mod basic;
mod conv;
mod linalg;
mod loss;
mod rnn;

// =============================================================================
// Re-Exports
// =============================================================================

pub use activation::*;
pub use attention::*;
pub use basic::*;
pub use conv::*;
pub use linalg::*;
pub use loss::*;
pub use rnn::*;
