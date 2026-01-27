//! Axonml Distributed - Distributed Training Utilities
//!
//! Comprehensive distributed training support for scaling ML workloads across
//! multiple GPUs and machines. Provides PyTorch-equivalent functionality.
//!
//! # Features
//!
//! ## Data Parallelism
//! - **DDP** - `DistributedDataParallel` for gradient synchronization across replicas
//! - **FSDP** - Fully Sharded Data Parallel with ZeRO-2 and ZeRO-3 optimizations
//!
//! ## Model Parallelism
//! - **Pipeline Parallelism** - Split model across devices with microbatching (GPipe-style)
//! - **Tensor Parallelism** - Layer-wise model sharding for large models
//!
//! ## Communication
//! - **Collective Operations**: all-reduce, all-gather, broadcast, reduce-scatter, barrier
//! - **Point-to-Point**: send, recv for direct tensor communication
//! - **Process Groups**: Flexible grouping for hierarchical parallelism
//!
//! ## Backends
//! - Mock backend for testing without real hardware
//! - Extensible Backend trait for NCCL, Gloo, MPI integration
//!
//! # DDP Example
//!
//! ```ignore
//! use axonml_distributed::prelude::*;
//! use axonml_nn::Linear;
//!
//! let world = World::mock();
//! let model = Linear::new(10, 5);
//! let ddp_model = DistributedDataParallel::new(model, world.default_group().clone());
//!
//! // Forward pass
//! let output = ddp_model.forward(&input);
//! loss.backward();
//!
//! // Gradient sync happens automatically or manually:
//! ddp_model.sync_gradients();
//! ```
//!
//! # FSDP Example (ZeRO-3)
//!
//! ```ignore
//! use axonml_distributed::{FSDP, FSDPConfig, ShardingStrategy};
//!
//! let config = FSDPConfig {
//!     sharding_strategy: ShardingStrategy::FullShard, // ZeRO-3
//!     cpu_offload: true,
//!     ..Default::default()
//! };
//!
//! let fsdp_model = FSDP::new(model, process_group, config);
//! let output = fsdp_model.forward(&input);
//! ```
//!
//! # Pipeline Parallelism Example
//!
//! ```ignore
//! use axonml_distributed::{PipelineParallel, PipelineConfig, PipelineSchedule};
//!
//! let config = PipelineConfig {
//!     num_stages: 4,
//!     num_microbatches: 8,
//!     schedule: PipelineSchedule::GPipe,
//!     ..Default::default()
//! };
//!
//! let pipeline = PipelineParallel::new(stages, process_group, config);
//! let output = pipeline.forward(&input);
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

pub mod backend;
pub mod comm;
pub mod ddp;
pub mod fsdp;
pub mod pipeline;
pub mod process_group;

// =============================================================================
// Re-exports
// =============================================================================

pub use backend::{Backend, MockBackend, ReduceOp};
pub use comm::{
    all_gather, all_reduce_max, all_reduce_mean, all_reduce_min, all_reduce_product,
    all_reduce_sum, barrier, broadcast, broadcast_from, gather_tensor, is_main_process, rank,
    reduce_scatter_mean, reduce_scatter_sum, scatter_tensor, sync_gradient, sync_gradients,
    world_size,
};
pub use ddp::{DistributedDataParallel, GradSyncStrategy, GradientBucket, GradientSynchronizer};
pub use fsdp::{
    CPUOffload, ColumnParallelLinear, FSDPMemoryStats, FullyShardedDataParallel, RowParallelLinear,
    ShardingStrategy,
};
pub use pipeline::{Pipeline, PipelineMemoryStats, PipelineSchedule, PipelineStage};
pub use process_group::{ProcessGroup, World};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for distributed training.
pub mod prelude {
    pub use crate::{
        all_gather,
        all_reduce_max,
        all_reduce_mean,
        all_reduce_min,
        all_reduce_product,
        // Communication
        all_reduce_sum,
        barrier,
        broadcast,
        broadcast_from,
        gather_tensor,
        is_main_process,
        rank,
        reduce_scatter_mean,
        reduce_scatter_sum,
        scatter_tensor,
        sync_gradient,
        sync_gradients,
        world_size,
        // Backend
        Backend,
        CPUOffload,
        // FSDP
        ColumnParallelLinear,
        // DDP
        DistributedDataParallel,
        FullyShardedDataParallel,
        GradSyncStrategy,
        GradientBucket,
        GradientSynchronizer,
        MockBackend,
        // Process groups
        ProcessGroup,
        ReduceOp,
        RowParallelLinear,
        ShardingStrategy,
        World,
    };

    pub use axonml_autograd::Variable;
    pub use axonml_nn::Module;
    pub use axonml_tensor::Tensor;
}

// =============================================================================
// Type Aliases
// =============================================================================

/// Type alias for `DistributedDataParallel`.
pub type DDP<M> = DistributedDataParallel<M>;

/// Type alias for `FullyShardedDataParallel`.
pub type FSDP<M> = FullyShardedDataParallel<M>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_autograd::Variable;
    use axonml_nn::{Linear, Module};
    use axonml_tensor::Tensor;
    use std::sync::Arc;

    #[test]
    fn test_full_distributed_workflow() {
        // Initialize world
        let world = World::mock();
        assert_eq!(world.rank(), 0);
        assert!(world.is_main());

        // Create model and wrap in DDP
        let model = Linear::new(10, 5);
        let mut ddp = DDP::new(model, world.default_group().clone());

        // Forward pass
        let input = Variable::new(Tensor::from_vec(vec![1.0; 10], &[1, 10]).unwrap(), false);
        let output = ddp.forward(&input);

        assert_eq!(output.data().shape(), &[1, 5]);

        // Train mode
        ddp.train();
        assert!(ddp.is_training());

        // Sync parameters
        ddp.sync_parameters();

        // Sync gradients
        ddp.sync_gradients();
    }

    #[test]
    fn test_multiple_backends() {
        let backends = MockBackend::create_world(4);

        // All backends should have consistent world view
        for (i, backend) in backends.iter().enumerate() {
            assert_eq!(backend.rank(), i);
            assert_eq!(backend.world_size(), 4);
        }
    }

    #[test]
    fn test_process_group_creation() {
        let backends = MockBackend::create_world(2);
        let pg = ProcessGroup::new(Arc::new(backends.into_iter().next().unwrap()));

        assert_eq!(pg.size(), 2);
        assert_eq!(pg.ranks().len(), 2);
    }

    #[test]
    fn test_communication_functions() {
        let pg = ProcessGroup::mock();

        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        // All reduce
        all_reduce_sum(&mut tensor, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);

        // Broadcast
        broadcast(&mut tensor, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);

        // All gather
        let gathered = all_gather(&tensor, &pg);
        assert_eq!(gathered.shape(), &[1, 3]);

        // Barrier
        barrier(&pg);
    }

    #[test]
    fn test_gradient_bucket_workflow() {
        let mut bucket = GradientBucket::new(1000);

        // Add gradients
        let grad1 = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        let grad2 = Tensor::from_vec(vec![0.3, 0.4, 0.5], &[3]).unwrap();

        bucket.add(&grad1);
        bucket.add(&grad2);

        assert_eq!(bucket.size(), 5);

        // Extract
        let tensors = bucket.extract();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].to_vec(), vec![0.1, 0.2]);
        assert_eq!(tensors[1].to_vec(), vec![0.3, 0.4, 0.5]);

        // Clear
        bucket.clear();
        assert!(bucket.is_empty());
    }

    #[test]
    fn test_gradient_synchronizer_workflow() {
        let mut sync = GradientSynchronizer::new(GradSyncStrategy::Synchronous, 1000);
        sync.prepare(10);

        // Add gradients
        let grad = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        sync.add_gradient(0, &grad);

        // Sync
        let pg = ProcessGroup::mock();
        sync.sync_all(&pg);

        // Clear
        sync.clear();
    }

    #[test]
    fn test_world_default_group() {
        let world = World::mock();
        let group = world.default_group();

        assert_eq!(group.rank(), 0);
        assert_eq!(group.world_size(), 1);
    }

    #[test]
    fn test_world_new_subgroup() {
        let world = World::mock();
        let subgroup = world.new_group(vec![0]);

        assert_eq!(subgroup.size(), 1);
        assert!(subgroup.contains(0));
    }

    #[test]
    fn test_ddp_builder_pattern() {
        let model = Linear::new(10, 5);
        let pg = ProcessGroup::mock();

        let ddp = DDP::new(model, pg)
            .broadcast_buffers(false)
            .gradient_as_bucket_view(false);

        // Linear defaults to training mode, DDP wraps it
        assert!(ddp.is_training());
    }

    #[test]
    fn test_reduce_op_all_variants() {
        let op_sum = ReduceOp::Sum;
        let op_prod = ReduceOp::Product;
        let op_min = ReduceOp::Min;
        let op_max = ReduceOp::Max;
        let op_avg = ReduceOp::Average;

        assert_eq!(op_sum.apply_f32(1.0, 2.0), 3.0);
        assert_eq!(op_prod.apply_f32(2.0, 3.0), 6.0);
        assert_eq!(op_min.apply_f32(2.0, 3.0), 2.0);
        assert_eq!(op_max.apply_f32(2.0, 3.0), 3.0);
        assert_eq!(op_avg.apply_f32(2.0, 4.0), 3.0);
    }
}
