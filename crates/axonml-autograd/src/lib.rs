//! Axonml Autograd - Automatic Differentiation Engine
//!
//! Provides reverse-mode automatic differentiation for computing gradients
//! of tensor operations. This is the foundation for training neural networks
//! using gradient descent optimization.
//!
//! # Key Features
//!
//! - **Dynamic Computational Graph** - Build graph during forward pass
//! - **Reverse-mode Autodiff** - Efficient backpropagation
//! - **Gradient Accumulation** - Support for gradient accumulation across batches
//! - **No-grad Context** - Disable gradient tracking for inference
//! - **Automatic Mixed Precision (AMP)** - F16 autocast for faster training
//! - **Gradient Checkpointing** - Trade compute for memory on large models
//!
//! # Basic Example
//!
//! ```rust,ignore
//! use axonml_autograd::{Variable, no_grad};
//!
//! // Create variables with gradient tracking
//! let x = Variable::new(tensor, true);  // requires_grad = true
//! let w = Variable::new(weights, true);
//!
//! // Forward pass builds computational graph
//! let y = x.matmul(&w);
//! let loss = y.mse_loss(&target);
//!
//! // Backward pass computes gradients
//! loss.backward();
//!
//! // Access gradients
//! println!("dL/dw = {:?}", w.grad());
//! ```
//!
//! # Mixed Precision Training
//!
//! ```rust,ignore
//! use axonml_autograd::amp::{autocast, AutocastGuard};
//! use axonml_core::DType;
//!
//! // Enable F16 autocast for forward pass
//! let output = autocast(DType::F16, || {
//!     model.forward(&input)
//! });
//!
//! // Or use RAII guard
//! {
//!     let _guard = AutocastGuard::new(DType::F16);
//!     let output = model.forward(&input);
//! }
//! ```
//!
//! # Gradient Checkpointing
//!
//! ```rust,ignore
//! use axonml_autograd::checkpoint::{checkpoint, checkpoint_sequential};
//!
//! // Checkpoint a single function - recomputes during backward
//! let output = checkpoint(|x| heavy_computation(x), &input);
//!
//! // Checkpoint sequential layers in segments
//! let output = checkpoint_sequential(24, 4, &input, |layer_idx, x| {
//!     layers[layer_idx].forward(x)
//! });
//! ```
//!
//! @version 0.2.6
//! @author `AutomataNexus` Development Team

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
// Modules
// =============================================================================

pub mod amp;
pub mod backward;
pub mod checkpoint;
pub mod functions;
pub mod grad_fn;
pub mod graph;
pub mod no_grad;
pub mod variable;

// =============================================================================
// Re-exports
// =============================================================================

pub use amp::{
    autocast, autocast_dtype, disable_autocast, is_autocast_enabled, AutocastGuard, AutocastPolicy,
};
pub use backward::backward;
pub use checkpoint::{checkpoint, checkpoint_sequential};
pub use grad_fn::{GradFn, GradientFunction};
pub use graph::{ComputationGraph, GraphNode};
pub use no_grad::{no_grad, NoGradGuard};
pub use variable::Variable;

// =============================================================================
// Prelude
// =============================================================================

/// Convenient imports for common autograd usage.
pub mod prelude {
    pub use crate::backward::backward;
    pub use crate::no_grad::{no_grad, NoGradGuard};
    pub use crate::variable::Variable;
}
