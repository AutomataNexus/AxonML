//! axonml-optim - Optimization Algorithms
//!
//! Provides optimizers for training neural networks, including:
//! - SGD with momentum and Nesterov acceleration
//! - Adam and `AdamW`
//! - `RMSprop`
//! - Learning rate schedulers
//!
//! # Example
//!
//! ```ignore
//! use axonml_optim::prelude::*;
//! use axonml_nn::{Linear, Module, Sequential};
//!
//! // Create model
//! let model = Sequential::new()
//!     .add(Linear::new(784, 128))
//!     .add(Linear::new(128, 10));
//!
//! // Create optimizer
//! let mut optimizer = Adam::new(model.parameters(), 0.001);
//!
//! // Training loop
//! for epoch in 0..100 {
//!     let output = model.forward(&input);
//!     let loss = compute_loss(&output, &target);
//!
//!     optimizer.zero_grad();
//!     loss.backward();
//!     optimizer.step();
//! }
//! ```
//!
//! @version 0.1.0
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
// Module Declarations
// =============================================================================

pub mod adam;
pub mod grad_scaler;
pub mod lamb;
pub mod lr_scheduler;
pub mod optimizer;
pub mod rmsprop;
pub mod sgd;

// =============================================================================
// Re-exports
// =============================================================================

pub use adam::{Adam, AdamW};
pub use grad_scaler::{GradScaler, GradScalerState};
pub use lamb::LAMB;
pub use lr_scheduler::{
    CosineAnnealingLR, ExponentialLR, LRScheduler, MultiStepLR, OneCycleLR, ReduceLROnPlateau,
    StepLR, WarmupLR,
};
pub use optimizer::Optimizer;
pub use rmsprop::RMSprop;
pub use sgd::SGD;

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for optimization.
pub mod prelude {
    pub use crate::{
        Adam, AdamW, CosineAnnealingLR, ExponentialLR, GradScaler, LRScheduler, LAMB, MultiStepLR,
        OneCycleLR, Optimizer, RMSprop, ReduceLROnPlateau, StepLR, WarmupLR, SGD,
    };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_autograd::Variable;
    use axonml_nn::{Linear, MSELoss, Module, ReLU, Sequential};
    use axonml_tensor::Tensor;

    #[test]
    fn test_sgd_optimization() {
        let model = Sequential::new()
            .add(Linear::new(2, 4))
            .add(ReLU)
            .add(Linear::new(4, 1));

        let mut optimizer = SGD::new(model.parameters(), 0.01);
        let loss_fn = MSELoss::new();

        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            false,
        );
        let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap(), false);

        let initial_loss = loss_fn.compute(&model.forward(&input), &target);
        let initial_loss_val = initial_loss.data().to_vec()[0];

        // Run a few optimization steps
        for _ in 0..10 {
            optimizer.zero_grad();
            let output = model.forward(&input);
            let loss = loss_fn.compute(&output, &target);
            loss.backward();
            optimizer.step();
        }

        let final_loss = loss_fn.compute(&model.forward(&input), &target);
        let final_loss_val = final_loss.data().to_vec()[0];

        // Loss should decrease
        assert!(final_loss_val <= initial_loss_val);
    }

    #[test]
    fn test_adam_optimization() {
        let model = Sequential::new()
            .add(Linear::new(2, 4))
            .add(ReLU)
            .add(Linear::new(4, 1));

        let mut optimizer = Adam::new(model.parameters(), 0.01);
        let loss_fn = MSELoss::new();

        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            false,
        );
        let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap(), false);

        // Run optimization
        for _ in 0..20 {
            optimizer.zero_grad();
            let output = model.forward(&input);
            let loss = loss_fn.compute(&output, &target);
            loss.backward();
            optimizer.step();
        }

        // Just verify it runs without error
        let final_output = model.forward(&input);
        assert_eq!(final_output.shape(), vec![2, 1]);
    }

    #[test]
    fn test_lr_scheduler() {
        let model = Linear::new(10, 5);
        let mut optimizer = SGD::new(model.parameters(), 0.1);
        let mut scheduler = StepLR::new(&optimizer, 10, 0.1);

        assert!((optimizer.get_lr() - 0.1).abs() < 1e-6);

        for _ in 0..10 {
            scheduler.step(&mut optimizer);
        }

        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);
    }
}
