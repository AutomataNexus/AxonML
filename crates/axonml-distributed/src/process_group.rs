//! `ProcessGroup` - Process Group Abstraction
//!
//! Provides a high-level abstraction for managing groups of processes
//! in distributed training.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use crate::backend::{Backend, MockBackend, ReduceOp};
use axonml_tensor::Tensor;
use std::sync::Arc;

// =============================================================================
// ProcessGroup
// =============================================================================

/// A group of processes that can communicate with each other.
pub struct ProcessGroup {
    backend: Arc<dyn Backend>,
    ranks: Vec<usize>,
}

impl ProcessGroup {
    /// Creates a new process group with all ranks.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        let world_size = backend.world_size();
        Self {
            backend,
            ranks: (0..world_size).collect(),
        }
    }

    /// Creates a process group with specific ranks.
    pub fn with_ranks(backend: Arc<dyn Backend>, ranks: Vec<usize>) -> Self {
        Self { backend, ranks }
    }

    /// Creates a mock process group for testing.
    #[must_use] pub fn mock() -> Self {
        Self::new(Arc::new(MockBackend::single()))
    }

    /// Returns the backend.
    #[must_use] pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Returns the rank of this process.
    #[must_use] pub fn rank(&self) -> usize {
        self.backend.rank()
    }

    /// Returns the world size.
    #[must_use] pub fn world_size(&self) -> usize {
        self.backend.world_size()
    }

    /// Returns the number of processes in this group.
    #[must_use] pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// Returns the ranks in this group.
    #[must_use] pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    /// Checks if this process is part of the group.
    #[must_use] pub fn contains(&self, rank: usize) -> bool {
        self.ranks.contains(&rank)
    }

    /// Synchronizes all processes in the group.
    pub fn barrier(&self) {
        self.backend.barrier();
    }

    /// Performs all-reduce on a tensor.
    pub fn all_reduce_tensor(&self, tensor: &mut Tensor<f32>, op: ReduceOp) {
        let mut data = tensor.to_vec();
        self.backend.all_reduce(&mut data, op);
        *tensor = Tensor::from_vec(data, tensor.shape()).unwrap();
    }

    /// Broadcasts a tensor from a source rank.
    pub fn broadcast_tensor(&self, tensor: &mut Tensor<f32>, src: usize) {
        let mut data = tensor.to_vec();
        self.backend.broadcast(&mut data, src);
        *tensor = Tensor::from_vec(data, tensor.shape()).unwrap();
    }

    /// Performs all-gather on tensors.
    #[must_use] pub fn all_gather_tensor(&self, send_tensor: &Tensor<f32>) -> Tensor<f32> {
        let send_data = send_tensor.to_vec();
        let mut recv_data = vec![0.0; send_data.len() * self.world_size()];
        self.backend.all_gather(&send_data, &mut recv_data);

        // Output shape: [world_size, ...original_shape]
        let mut new_shape = vec![self.world_size()];
        new_shape.extend(send_tensor.shape());
        Tensor::from_vec(recv_data, &new_shape).unwrap()
    }

    /// Performs reduce-scatter on a tensor.
    #[must_use] pub fn reduce_scatter_tensor(&self, send_tensor: &Tensor<f32>, op: ReduceOp) -> Tensor<f32> {
        let send_data = send_tensor.to_vec();
        let chunk_size = send_data.len() / self.world_size();
        let mut recv_data = vec![0.0; chunk_size];
        self.backend.reduce_scatter(&send_data, &mut recv_data, op);

        // Output shape: reduced original shape
        let original_shape = send_tensor.shape();
        let mut new_shape = original_shape.to_vec();
        if !new_shape.is_empty() {
            new_shape[0] /= self.world_size();
        }
        Tensor::from_vec(recv_data, &new_shape).unwrap()
    }
}

// =============================================================================
// World
// =============================================================================

/// Global distributed world.
pub struct World {
    default_group: ProcessGroup,
}

impl World {
    /// Initializes the distributed world.
    pub fn init(backend: Arc<dyn Backend>) -> Self {
        Self {
            default_group: ProcessGroup::new(backend),
        }
    }

    /// Creates a mock world for testing.
    #[must_use] pub fn mock() -> Self {
        Self {
            default_group: ProcessGroup::mock(),
        }
    }

    /// Returns the default process group.
    #[must_use] pub fn default_group(&self) -> &ProcessGroup {
        &self.default_group
    }

    /// Returns the rank of this process.
    #[must_use] pub fn rank(&self) -> usize {
        self.default_group.rank()
    }

    /// Returns the world size.
    #[must_use] pub fn world_size(&self) -> usize {
        self.default_group.world_size()
    }

    /// Checks if this is the main process (rank 0).
    #[must_use] pub fn is_main(&self) -> bool {
        self.rank() == 0
    }

    /// Synchronizes all processes.
    pub fn barrier(&self) {
        self.default_group.barrier();
    }

    /// Creates a new process group with specific ranks.
    #[must_use] pub fn new_group(&self, ranks: Vec<usize>) -> ProcessGroup {
        ProcessGroup::with_ranks(Arc::clone(&self.default_group.backend), ranks)
    }
}

impl Clone for ProcessGroup {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            ranks: self.ranks.clone(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_group_mock() {
        let pg = ProcessGroup::mock();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);
        assert_eq!(pg.size(), 1);
    }

    #[test]
    fn test_process_group_contains() {
        let pg = ProcessGroup::mock();
        assert!(pg.contains(0));
        assert!(!pg.contains(1));
    }

    #[test]
    fn test_world_mock() {
        let world = World::mock();
        assert_eq!(world.rank(), 0);
        assert_eq!(world.world_size(), 1);
        assert!(world.is_main());
    }

    #[test]
    fn test_world_new_group() {
        let world = World::mock();
        let group = world.new_group(vec![0]);
        assert_eq!(group.size(), 1);
    }

    #[test]
    fn test_process_group_all_reduce_tensor() {
        let backends = MockBackend::create_world(2);
        let pg0 = ProcessGroup::new(Arc::new(backends.into_iter().next().unwrap()));

        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        pg0.all_reduce_tensor(&mut tensor, ReduceOp::Sum);

        // Single rank, values unchanged
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_process_group_broadcast_tensor() {
        let pg = ProcessGroup::mock();

        let mut tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        pg.broadcast_tensor(&mut tensor, 0);

        assert_eq!(tensor.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_process_group_all_gather_tensor() {
        let pg = ProcessGroup::mock();

        let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let gathered = pg.all_gather_tensor(&tensor);

        assert_eq!(gathered.shape(), &[1, 2]);
    }

    #[test]
    fn test_process_group_barrier() {
        let pg = ProcessGroup::mock();
        pg.barrier(); // Should not deadlock
    }

    #[test]
    fn test_world_barrier() {
        let world = World::mock();
        world.barrier(); // Should not deadlock
    }

    #[test]
    fn test_process_group_clone() {
        let pg = ProcessGroup::mock();
        let pg2 = pg.clone();
        assert_eq!(pg.rank(), pg2.rank());
        assert_eq!(pg.world_size(), pg2.world_size());
    }
}
