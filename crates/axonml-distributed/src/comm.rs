//! Communication - High-level Communication Utilities
//!
//! Provides high-level functions for common distributed communication patterns.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use crate::backend::ReduceOp;
use crate::process_group::ProcessGroup;
use axonml_tensor::Tensor;

// =============================================================================
// All-Reduce Operations
// =============================================================================

/// Performs all-reduce sum on a tensor.
pub fn all_reduce_sum(tensor: &mut Tensor<f32>, pg: &ProcessGroup) {
    pg.all_reduce_tensor(tensor, ReduceOp::Sum);
}

/// Performs all-reduce mean on a tensor.
pub fn all_reduce_mean(tensor: &mut Tensor<f32>, pg: &ProcessGroup) {
    pg.all_reduce_tensor(tensor, ReduceOp::Average);
}

/// Performs all-reduce min on a tensor.
pub fn all_reduce_min(tensor: &mut Tensor<f32>, pg: &ProcessGroup) {
    pg.all_reduce_tensor(tensor, ReduceOp::Min);
}

/// Performs all-reduce max on a tensor.
pub fn all_reduce_max(tensor: &mut Tensor<f32>, pg: &ProcessGroup) {
    pg.all_reduce_tensor(tensor, ReduceOp::Max);
}

/// Performs all-reduce product on a tensor.
pub fn all_reduce_product(tensor: &mut Tensor<f32>, pg: &ProcessGroup) {
    pg.all_reduce_tensor(tensor, ReduceOp::Product);
}

// =============================================================================
// Broadcast Operations
// =============================================================================

/// Broadcasts a tensor from the root rank (0).
pub fn broadcast(tensor: &mut Tensor<f32>, pg: &ProcessGroup) {
    broadcast_from(tensor, 0, pg);
}

/// Broadcasts a tensor from a specific rank.
pub fn broadcast_from(tensor: &mut Tensor<f32>, src: usize, pg: &ProcessGroup) {
    pg.broadcast_tensor(tensor, src);
}

// =============================================================================
// Gather Operations
// =============================================================================

/// All-gathers a tensor across all ranks.
#[must_use] pub fn all_gather(tensor: &Tensor<f32>, pg: &ProcessGroup) -> Tensor<f32> {
    pg.all_gather_tensor(tensor)
}

// =============================================================================
// Reduce-Scatter Operations
// =============================================================================

/// Reduce-scatters a tensor with sum.
#[must_use] pub fn reduce_scatter_sum(tensor: &Tensor<f32>, pg: &ProcessGroup) -> Tensor<f32> {
    pg.reduce_scatter_tensor(tensor, ReduceOp::Sum)
}

/// Reduce-scatters a tensor with mean.
#[must_use] pub fn reduce_scatter_mean(tensor: &Tensor<f32>, pg: &ProcessGroup) -> Tensor<f32> {
    pg.reduce_scatter_tensor(tensor, ReduceOp::Average)
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Synchronizes all processes.
pub fn barrier(pg: &ProcessGroup) {
    pg.barrier();
}

/// Checks if this is the main process (rank 0).
#[must_use] pub fn is_main_process(pg: &ProcessGroup) -> bool {
    pg.rank() == 0
}

/// Returns the world size.
#[must_use] pub fn world_size(pg: &ProcessGroup) -> usize {
    pg.world_size()
}

/// Returns the current rank.
#[must_use] pub fn rank(pg: &ProcessGroup) -> usize {
    pg.rank()
}

// =============================================================================
// Model Parallel Utilities
// =============================================================================

/// Splits a tensor along a dimension for model parallelism.
#[must_use] pub fn scatter_tensor(tensor: &Tensor<f32>, dim: usize, pg: &ProcessGroup) -> Tensor<f32> {
    let shape = tensor.shape();
    if dim >= shape.len() {
        return tensor.clone();
    }

    let world_size = pg.world_size();
    let rank = pg.rank();
    let dim_size = shape[dim];

    if dim_size % world_size != 0 {
        return tensor.clone();
    }

    let chunk_size = dim_size / world_size;
    let start = rank * chunk_size;
    let end = start + chunk_size;

    // For 1D tensors along dim 0
    if shape.len() == 1 && dim == 0 {
        let data = tensor.to_vec();
        let chunk = data[start..end].to_vec();
        return Tensor::from_vec(chunk, &[chunk_size]).unwrap();
    }

    // For 2D tensors along dim 0
    if shape.len() == 2 && dim == 0 {
        let data = tensor.to_vec();
        let cols = shape[1];
        let mut chunk = Vec::with_capacity(chunk_size * cols);
        for row in start..end {
            let row_start = row * cols;
            let row_end = row_start + cols;
            chunk.extend_from_slice(&data[row_start..row_end]);
        }
        return Tensor::from_vec(chunk, &[chunk_size, cols]).unwrap();
    }

    tensor.clone()
}

/// Gathers scattered tensor chunks back together.
#[must_use] pub fn gather_tensor(tensor: &Tensor<f32>, dim: usize, pg: &ProcessGroup) -> Tensor<f32> {
    let gathered = pg.all_gather_tensor(tensor);

    // Reshape gathered tensor
    let world_size = pg.world_size();
    let shape = tensor.shape();

    if shape.len() == 1 && dim == 0 {
        // Flatten [world_size, chunk_size] to [total_size]
        let data = gathered.to_vec();
        return Tensor::from_vec(data, &[shape[0] * world_size]).unwrap();
    }

    gathered
}

// =============================================================================
// Gradient Synchronization
// =============================================================================

/// Synchronizes gradients by averaging across all processes.
pub fn sync_gradients(gradients: &mut [Tensor<f32>], pg: &ProcessGroup) {
    for grad in gradients.iter_mut() {
        all_reduce_mean(grad, pg);
    }
}

/// Synchronizes a single gradient tensor.
pub fn sync_gradient(gradient: &mut Tensor<f32>, pg: &ProcessGroup) {
    all_reduce_mean(gradient, pg);
}

// =============================================================================
// Ring Communication Pattern
// =============================================================================

/// Ring all-reduce implementation (educational).
/// In practice, the backend handles this more efficiently.
pub fn ring_all_reduce(data: &mut [f32], pg: &ProcessGroup, op: ReduceOp) {
    let world_size = pg.world_size();
    if world_size == 1 {
        return;
    }

    // Use backend's all-reduce
    pg.backend().all_reduce(data, op);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_reduce_sum() {
        let pg = ProcessGroup::mock();
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        all_reduce_sum(&mut tensor, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_all_reduce_mean() {
        let pg = ProcessGroup::mock();
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        all_reduce_mean(&mut tensor, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_all_reduce_min() {
        let pg = ProcessGroup::mock();
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        all_reduce_min(&mut tensor, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_all_reduce_max() {
        let pg = ProcessGroup::mock();
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        all_reduce_max(&mut tensor, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast() {
        let pg = ProcessGroup::mock();
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        broadcast(&mut tensor, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_broadcast_from() {
        let pg = ProcessGroup::mock();
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        broadcast_from(&mut tensor, 0, &pg);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_all_gather() {
        let pg = ProcessGroup::mock();
        let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        let gathered = all_gather(&tensor, &pg);
        assert_eq!(gathered.shape(), &[1, 2]);
    }

    #[test]
    fn test_reduce_scatter_sum() {
        let pg = ProcessGroup::mock();
        let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        let scattered = reduce_scatter_sum(&tensor, &pg);
        assert_eq!(scattered.shape(), &[2]);
    }

    #[test]
    fn test_barrier() {
        let pg = ProcessGroup::mock();
        barrier(&pg); // Should not deadlock
    }

    #[test]
    fn test_is_main_process() {
        let pg = ProcessGroup::mock();
        assert!(is_main_process(&pg));
    }

    #[test]
    fn test_world_size() {
        let pg = ProcessGroup::mock();
        assert_eq!(world_size(&pg), 1);
    }

    #[test]
    fn test_rank() {
        let pg = ProcessGroup::mock();
        assert_eq!(rank(&pg), 0);
    }

    #[test]
    fn test_scatter_tensor_1d() {
        let pg = ProcessGroup::mock();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let scattered = scatter_tensor(&tensor, 0, &pg);
        // With world_size=1, should return full tensor
        assert_eq!(scattered.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gather_tensor() {
        let pg = ProcessGroup::mock();
        let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        let gathered = gather_tensor(&tensor, 0, &pg);
        assert_eq!(gathered.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_sync_gradients() {
        let pg = ProcessGroup::mock();
        let mut grads = vec![
            Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
        ];

        sync_gradients(&mut grads, &pg);

        assert_eq!(grads[0].to_vec(), vec![1.0, 2.0]);
        assert_eq!(grads[1].to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_sync_gradient() {
        let pg = ProcessGroup::mock();
        let mut grad = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        sync_gradient(&mut grad, &pg);
        assert_eq!(grad.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ring_all_reduce() {
        let pg = ProcessGroup::mock();
        let mut data = vec![1.0, 2.0, 3.0];

        ring_all_reduce(&mut data, &pg, ReduceOp::Sum);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
