//! Axonml Vision - Computer Vision Utilities
//!
//! This crate provides computer vision functionality for the Axonml ML framework:
//!
//! - **Transforms**: Image-specific data augmentation and preprocessing
//! - **Datasets**: Loaders for common vision datasets (MNIST, CIFAR)
//! - **Models**: Pre-defined neural network architectures (`LeNet`, MLP)
//!
//! # Example
//!
//! ```ignore
//! use axonml_vision::prelude::*;
//!
//! // Load synthetic MNIST data
//! let train_data = SyntheticMNIST::train();
//! let test_data = SyntheticMNIST::test();
//!
//! // Create a LeNet model
//! let model = LeNet::new();
//!
//! // Apply image transforms
//! let transform = Compose::empty()
//!     .add(ImageNormalize::mnist())
//!     .add(RandomHorizontalFlip::new());
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

pub mod datasets;
pub mod hub;
pub mod models;
pub mod transforms;

// =============================================================================
// Re-exports
// =============================================================================

pub use transforms::{
    CenterCrop, ColorJitter, Grayscale, ImageNormalize, Pad, RandomHorizontalFlip, RandomRotation,
    RandomVerticalFlip, Resize, ToTensorImage,
};

pub use datasets::{FashionMNIST, SyntheticCIFAR, SyntheticMNIST, CIFAR10, CIFAR100, MNIST};

pub use models::{LeNet, SimpleCNN, MLP};

pub use hub::{
    cache_dir, download_weights, is_cached, list_models, load_state_dict, model_info,
    model_registry, HubError, HubResult, PretrainedModel, StateDict,
};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for computer vision tasks.
pub mod prelude {
    pub use crate::{
        CenterCrop,
        ColorJitter,
        FashionMNIST,
        Grayscale,
        ImageNormalize,
        // Models
        LeNet,
        Pad,
        RandomHorizontalFlip,
        RandomRotation,
        RandomVerticalFlip,
        // Transforms
        Resize,
        SimpleCNN,
        SyntheticCIFAR,
        SyntheticMNIST,
        ToTensorImage,
        CIFAR10,
        CIFAR100,
        MLP,
        // Datasets
        MNIST,
    };

    // Re-export useful items from dependencies
    pub use axonml_autograd::Variable;
    pub use axonml_data::{Compose, DataLoader, Dataset, Transform};
    pub use axonml_nn::Module;
    pub use axonml_tensor::Tensor;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_data::{Compose, Dataset, Transform};
    use axonml_tensor::Tensor;

    #[test]
    fn test_synthetic_mnist_with_transforms() {
        let dataset = SyntheticMNIST::small();
        let normalize = ImageNormalize::mnist();

        let (image, label) = dataset.get(0).unwrap();
        let normalized = normalize.apply(&image);

        assert_eq!(normalized.shape(), &[1, 28, 28]);
        assert_eq!(label.shape(), &[10]);
    }

    #[test]
    fn test_synthetic_cifar_with_transforms() {
        let dataset = SyntheticCIFAR::small();
        let normalize = ImageNormalize::cifar10();

        let (image, label) = dataset.get(0).unwrap();
        let normalized = normalize.apply(&image);

        assert_eq!(normalized.shape(), &[3, 32, 32]);
        assert_eq!(label.shape(), &[10]);
    }

    #[test]
    fn test_transform_pipeline() {
        let transform = Compose::empty()
            .add(Resize::new(32, 32))
            .add(RandomHorizontalFlip::with_probability(0.0)) // No flip for determinism
            .add(ImageNormalize::new(vec![0.5], vec![0.5]));

        let input = Tensor::from_vec(vec![0.5; 28 * 28], &[1, 28, 28]).unwrap();
        let output = transform.apply(&input);

        assert_eq!(output.shape(), &[1, 32, 32]);
    }

    #[test]
    fn test_lenet_with_synthetic_data() {
        use axonml_autograd::Variable;
        use axonml_nn::Module;

        let dataset = SyntheticMNIST::small();
        let model = LeNet::new();

        // Get a sample and run forward pass
        let (image, _label) = dataset.get(0).unwrap();

        // Add batch dimension
        let batched = Tensor::from_vec(image.to_vec(), &[1, 1, 28, 28]).unwrap();

        let input = Variable::new(batched, false);
        let output = model.forward(&input);

        assert_eq!(output.data().shape(), &[1, 10]);
    }

    #[test]
    fn test_mlp_with_synthetic_data() {
        use axonml_autograd::Variable;
        use axonml_nn::Module;

        let dataset = SyntheticMNIST::small();
        let model = MLP::for_mnist();

        let (image, _) = dataset.get(0).unwrap();

        // MLP expects flattened input with batch dimension
        let flattened = Tensor::from_vec(image.to_vec(), &[1, 784]).unwrap();

        let input = Variable::new(flattened, false);
        let output = model.forward(&input);

        assert_eq!(output.data().shape(), &[1, 10]);
    }

    #[test]
    fn test_resize_and_crop_pipeline() {
        let transform = Compose::empty()
            .add(Resize::new(64, 64))
            .add(CenterCrop::new(32, 32));

        let input = Tensor::from_vec(vec![0.5; 3 * 28 * 28], &[3, 28, 28]).unwrap();
        let output = transform.apply(&input);

        assert_eq!(output.shape(), &[3, 32, 32]);
    }

    #[test]
    fn test_grayscale_transform() {
        let transform = Grayscale::new();
        let input = Tensor::from_vec(vec![0.5; 3 * 32 * 32], &[3, 32, 32]).unwrap();
        let output = transform.apply(&input);

        assert_eq!(output.shape(), &[1, 32, 32]);
    }

    #[test]
    fn test_full_training_pipeline() {
        use axonml_autograd::Variable;
        use axonml_data::DataLoader;
        use axonml_nn::Module;

        // Create dataset
        let dataset = SyntheticMNIST::new(32);

        // Create dataloader
        let loader = DataLoader::new(dataset, 8);

        // Create model
        let model = MLP::for_mnist();

        // Process one batch
        let mut processed_batches = 0;
        for batch in loader.iter().take(1) {
            // Flatten images for MLP
            let batch_data = batch.data.to_vec();
            let batch_size = batch.data.shape()[0];
            let features: usize = batch.data.shape()[1..].iter().product();

            let flattened = Tensor::from_vec(batch_data, &[batch_size, features]).unwrap();
            let input = Variable::new(flattened, false);

            let output = model.forward(&input);
            assert_eq!(output.data().shape()[0], batch_size);
            assert_eq!(output.data().shape()[1], 10);

            processed_batches += 1;
        }

        assert_eq!(processed_batches, 1);
    }
}
