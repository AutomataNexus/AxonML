//! Foundation layer for the AxonML deep learning framework.
//!
//! Provides the `Device` enum (CPU, CUDA, Vulkan, Metal, WebGPU) with runtime
//! capability queries, the `Scalar`/`Numeric`/`Float` trait hierarchy for
//! generic type-safe dispatch, reference-counted `Storage<T>` with pooled GPU
//! allocations, and five compute backends: CPU (rayon-parallel GEMM/GEMV via
//! matrixmultiply), CUDA (cuBLAS + 15 custom PTX kernel modules covering
//! elementwise ops, activations, attention, Q4_K/Q6_K dequant-in-shader
//! matmul, softmax, layernorm, RMSNorm, transpose, and embedding gather),
//! Vulkan (ash + gpu-allocator, full buffer/pipeline/dispatch), Metal
//! (full buffer/pipeline/dispatch on Apple Silicon), and WebGPU (wgpu,
//! full buffer/pipeline/dispatch for browser targets).
//!
//! # File
//! `crates/axonml-core/src/lib.rs`
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

pub mod allocator;
pub mod backends;
pub mod device;
pub mod dtype;
pub mod error;
pub mod storage;

// =============================================================================
// Re-exports
// =============================================================================

pub use allocator::{Allocator, DefaultAllocator};
pub use device::Device;
pub use dtype::{DType, Float, Numeric, Scalar};
pub use error::{Error, Result};
pub use storage::Storage;

// =============================================================================
// Prelude
// =============================================================================

/// Convenient imports for common usage.
pub mod prelude {
    pub use crate::device::Device;
    pub use crate::dtype::{DType, Float, Numeric, Scalar};
    pub use crate::error::{Error, Result};
    pub use crate::storage::Storage;
}
