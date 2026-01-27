//! Axonml Tensor - N-Dimensional Array for Machine Learning
//!
//! This crate provides the core `Tensor` type that serves as the foundation
//! for all machine learning operations in Axonml. Tensors are multi-dimensional
//! arrays with support for automatic broadcasting, device placement, and
//! efficient memory sharing through views.
//!
//! # Key Features
//! - N-dimensional tensor with arbitrary shape
//! - Automatic broadcasting for operations
//! - Efficient views and slicing (zero-copy where possible)
//! - Device-agnostic (CPU, CUDA, Vulkan, etc.)
//! - Generic over data type (f32, f64, i32, etc.)
//!
//! # Example
//! ```rust
//! use axonml_tensor::{zeros, ones, Tensor};
//!
//! // Create tensors using factory functions
//! let a = zeros::<f32>(&[2, 3]);
//! let b = ones::<f32>(&[2, 3]);
//!
//! // Arithmetic operations
//! let c = a.add(&b).unwrap();
//! let d = c.mul_scalar(2.0);
//!
//! // Reductions
//! let sum = d.sum();
//! let mean = d.mean();
//! ```
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

#![cfg_attr(not(feature = "std"), no_std)]
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

pub mod creation;
pub mod ops;
pub mod shape;
pub mod sparse;
pub mod tensor;
pub mod view;

// =============================================================================
// Re-exports
// =============================================================================

pub use axonml_core::{DType, Device, Error, Result};
pub use creation::*;
pub use shape::{Shape, Strides};
pub use tensor::Tensor;

// =============================================================================
// Prelude
// =============================================================================

/// Convenient imports for common usage.
pub mod prelude {
    pub use crate::shape::{Shape, Strides};
    pub use crate::tensor::Tensor;
    pub use crate::{arange, full, linspace, ones, rand, randn, zeros};
    pub use axonml_core::{DType, Device, Error, Result};
}
