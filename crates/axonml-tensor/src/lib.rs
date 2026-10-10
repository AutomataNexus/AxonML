//! N-dimensional tensor library for AxonML.
//!
//! Exports `Tensor<T>` (generic over Scalar types) with NumPy-style
//! broadcasting, strided zero-copy views, CPU + CUDA GPU matmul (with GEMV
//! fast path for m=1 decode), quantized matmul dispatch (Q4_K/Q6_K in-shader
//! dequant via `cuda_ops`), lazy tensors with algebraic optimization, sparse
//! COO tensors, factory functions (zeros/ones/randn/arange/linspace/eye/full),
//! and shape/stride utilities. Re-exports `Device`, `DType`, `Error`, `Result`
//! from `axonml-core`.
//!
//! Modules: `tensor`, `shape`, `creation`, `ops`, `view`, `cuda_ops`, `lazy`,
//! `sparse`.
//!
//! # File
//! `crates/axonml-tensor/src/lib.rs`
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

// Safe Rust everywhere except the CUDA backend, whose cudarc calls are unsafe by
// construction (audited in 05ca73d); the ban is a hard forbid on every non-CUDA build.
#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
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

#[cfg(feature = "cuda")]
pub use cuda_ops::ReplayGraph;
/// The CUDA timing event handed out by [`Tensor::event_record`]; re-exported so
/// profiling code downstream can name it without depending on cudarc itself.
#[cfg(feature = "cuda")]
pub use cudarc::driver::CudaEvent;
/// An instantiated CUDA graph from [`Tensor::graph_end_capture`]: owns the graph
/// and its exec, destroys both on drop, replays with [`Tensor::graph_launch`].
#[cfg(feature = "cuda")]
pub use cudarc::driver::CudaGraph;

// =============================================================================
// Modules
// =============================================================================

extern crate alloc;

/// `alloc` types that the std prelude provides for free; imported by modules
/// only when building without `std`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
pub(crate) mod alloc_prelude {
    pub use alloc::borrow::ToOwned;
    pub use alloc::boxed::Box;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec::Vec;
    pub use alloc::{format, vec};
}

pub mod creation;
#[cfg(feature = "cuda")]
pub mod cuda_ops;
pub mod fused_chain;
pub mod lazy;
pub mod ops;
#[cfg(feature = "std")]
pub mod rng;
pub mod shape;
pub mod sparse;
pub mod tensor;
pub mod view;

// =============================================================================
// Re-exports
// =============================================================================

#[cfg(feature = "cuda")]
pub use axonml_core::backends::cuda::{
    HostKeep, capture_host_arena_begin, capture_host_arena_take,
};
#[cfg(feature = "cuda")]
pub use axonml_core::backends::cuda_pool::{
    CapturePen, set_pool_uncapped, with_capture_pen, with_driver_alloc, with_pool_uncapped,
};
pub use axonml_core::{DType, Device, Error, Result};
pub use creation::*;
pub use lazy::{LazyOp, LazyTensor};
pub use shape::{Shape, Strides};
pub use tensor::Tensor;

// =============================================================================
// Prelude
// =============================================================================

/// Convenient imports for common usage.
pub mod prelude {
    pub use crate::shape::{Shape, Strides};
    pub use crate::tensor::Tensor;
    pub use crate::{arange, full, linspace, ones, zeros};
    #[cfg(feature = "std")]
    pub use crate::{rand, randn};
    pub use axonml_core::{DType, Device, Error, Result};
}
