//! Axonml Core - Foundation Layer for the Axonml ML Framework
//!
//! This crate provides the core abstractions that underpin the entire Axonml
//! machine learning framework. It handles device management, memory storage,
//! data types, and backend implementations.
//!
//! # Key Features
//! - Device abstraction (CPU, CUDA, Vulkan, Metal, WebGPU)
//! - Type-safe data type system (f32, f64, f16, i32, i64, bool)
//! - Efficient memory storage with reference counting
//! - Pluggable backend architecture
//!
//! # Example
//! ```rust
//! use axonml_core::{Device, DType, Storage};
//!
//! let device = Device::Cpu;
//! let storage = Storage::<f32>::zeros(1024, device);
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
