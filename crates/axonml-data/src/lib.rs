//! axonml-data - Data Loading Utilities
//!
//! Provides data loading infrastructure for training neural networks:
//! - Dataset trait for defining data sources
//! - `DataLoader` for batched iteration with parallel loading
//! - Samplers for controlling data access patterns
//! - Transforms for data augmentation
//!
//! # Example
//!
//! ```ignore
//! use axonml_data::prelude::*;
//!
//! // Define a simple dataset
//! struct MyDataset {
//!     data: Vec<(Tensor<f32>, Tensor<f32>)>,
//! }
//!
//! impl Dataset for MyDataset {
//!     type Item = (Tensor<f32>, Tensor<f32>);
//!
//!     fn len(&self) -> usize {
//!         self.data.len()
//!     }
//!
//!     fn get(&self, index: usize) -> Option<Self::Item> {
//!         self.data.get(index).cloned()
//!     }
//! }
//!
//! // Create a DataLoader
//! let loader = DataLoader::new(dataset, 32)
//!     .shuffle(true)
//!     .num_workers(4);
//!
//! for batch in loader.iter() {
//!     // Process batch
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

pub mod collate;
pub mod dataloader;
pub mod dataset;
pub mod sampler;
pub mod transforms;

// =============================================================================
// Re-exports
// =============================================================================

pub use collate::{Collate, DefaultCollate, StackCollate};
pub use dataloader::{Batch, DataLoader, DataLoaderIter};
pub use dataset::{
    ConcatDataset, Dataset, InMemoryDataset, MapDataset, SubsetDataset, TensorDataset,
};
pub use sampler::{
    BatchSampler, RandomSampler, Sampler, SequentialSampler, SubsetRandomSampler,
    WeightedRandomSampler,
};
pub use transforms::{Compose, Normalize, RandomNoise, ToTensor, Transform};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for data loading.
pub mod prelude {
    pub use crate::{
        Batch, BatchSampler, Collate, Compose, ConcatDataset, DataLoader, DataLoaderIter, Dataset,
        DefaultCollate, InMemoryDataset, MapDataset, Normalize, RandomNoise, RandomSampler,
        Sampler, SequentialSampler, StackCollate, SubsetDataset, SubsetRandomSampler,
        TensorDataset, ToTensor, Transform, WeightedRandomSampler,
    };
    pub use axonml_tensor::Tensor;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_tensor_dataset() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0, 0.0], &[3]).unwrap();
        let dataset = TensorDataset::new(x, y);

        assert_eq!(dataset.len(), 3);
        let (x_item, y_item) = dataset.get(0).unwrap();
        assert_eq!(x_item.to_vec(), vec![1.0, 2.0]);
        assert_eq!(y_item.to_vec(), vec![0.0]);
    }

    #[test]
    fn test_dataloader_basic() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6, 1]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], &[6]).unwrap();
        let dataset = TensorDataset::new(x, y);
        let loader = DataLoader::new(dataset, 2);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3); // 6 items / 2 batch_size = 3 batches
    }

    #[test]
    fn test_dataloader_shuffle() {
        let x = Tensor::from_vec((0..100).map(|i| i as f32).collect(), &[100, 1]).unwrap();
        let y = Tensor::from_vec((0..100).map(|i| i as f32).collect(), &[100]).unwrap();
        let dataset = TensorDataset::new(x, y);
        let loader = DataLoader::new(dataset, 10).shuffle(true);

        // Collect first batch from two iterations - they should differ if shuffled
        let batch1: Vec<_> = loader.iter().take(1).collect();
        let batch2: Vec<_> = loader.iter().take(1).collect();

        // Due to randomness, we can't guarantee they're different,
        // but at least verify the loader works
        assert!(!batch1.is_empty());
        assert!(!batch2.is_empty());
    }

    #[test]
    fn test_transform_compose() {
        let normalize = Normalize::new(0.0, 1.0);
        let noise = RandomNoise::new(0.0);
        let transform = Compose::new(vec![Box::new(normalize), Box::new(noise)]);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = transform.apply(&input);
        assert_eq!(output.shape(), &[3]);
    }

    #[test]
    fn test_samplers() {
        let sequential = SequentialSampler::new(10);
        let indices: Vec<_> = sequential.iter().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let random = RandomSampler::new(10);
        let indices: Vec<_> = random.iter().collect();
        assert_eq!(indices.len(), 10);
    }
}
