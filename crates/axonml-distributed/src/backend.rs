//! Backend - Communication Backend Abstractions
//!
//! Provides backend trait and implementations for distributed communication.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// =============================================================================
// Reduce Operations
// =============================================================================

/// Reduction operation for collective communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all values.
    Sum,
    /// Compute product of all values.
    Product,
    /// Find minimum value.
    Min,
    /// Find maximum value.
    Max,
    /// Compute average of all values.
    Average,
}

impl ReduceOp {
    /// Applies the reduction operation to two f32 values.
    #[must_use]
    pub fn apply_f32(&self, a: f32, b: f32) -> f32 {
        match self {
            ReduceOp::Sum => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Average => (a + b) / 2.0,
        }
    }

    /// Applies the reduction operation to slices.
    #[must_use]
    pub fn reduce_slices(&self, slices: &[Vec<f32>]) -> Vec<f32> {
        if slices.is_empty() {
            return Vec::new();
        }

        let len = slices[0].len();
        let mut result = slices[0].clone();

        for slice in slices.iter().skip(1) {
            for (i, &val) in slice.iter().enumerate() {
                if i < len {
                    result[i] = self.apply_f32(result[i], val);
                }
            }
        }

        // For average, we need to divide by count (already averaged pairwise above)
        if *self == ReduceOp::Average && slices.len() > 1 {
            // Re-compute as actual average
            result = vec![0.0; len];
            for slice in slices {
                for (i, &val) in slice.iter().enumerate() {
                    if i < len {
                        result[i] += val;
                    }
                }
            }
            let count = slices.len() as f32;
            for val in &mut result {
                *val /= count;
            }
        }

        result
    }
}

// =============================================================================
// Backend Trait
// =============================================================================

/// Trait for distributed communication backends.
pub trait Backend: Send + Sync {
    /// Returns the name of the backend.
    fn name(&self) -> &str;

    /// Returns the rank of this process.
    fn rank(&self) -> usize;

    /// Returns the total world size.
    fn world_size(&self) -> usize;

    /// Performs all-reduce operation.
    fn all_reduce(&self, data: &mut [f32], op: ReduceOp);

    /// Broadcasts data from a source rank.
    fn broadcast(&self, data: &mut [f32], src: usize);

    /// Performs all-gather operation.
    fn all_gather(&self, send_data: &[f32], recv_data: &mut [f32]);

    /// Performs reduce-scatter operation.
    fn reduce_scatter(&self, send_data: &[f32], recv_data: &mut [f32], op: ReduceOp);

    /// Performs gather operation.
    fn gather(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize);

    /// Performs scatter operation.
    fn scatter(&self, send_data: &[f32], recv_data: &mut [f32], src: usize);

    /// Performs reduce operation (result only on dst rank).
    fn reduce(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize, op: ReduceOp);

    /// Synchronizes all processes.
    fn barrier(&self);

    /// Sends data to a specific rank.
    fn send(&self, data: &[f32], dst: usize, tag: usize);

    /// Receives data from a specific rank.
    fn recv(&self, data: &mut [f32], src: usize, tag: usize);
}

// =============================================================================
// SharedState for Mock Backend
// =============================================================================

/// Shared state for mock distributed communication.
#[derive(Debug)]
struct SharedState {
    /// Data buffers for each rank.
    buffers: HashMap<usize, Vec<f32>>,
    /// Barrier counter.
    barrier_count: usize,
    /// Message queue for send/recv operations.
    messages: HashMap<(usize, usize, usize), Vec<f32>>, // (src, dst, tag) -> data
}

// =============================================================================
// Mock Backend
// =============================================================================

/// A mock backend for testing distributed operations in a single process.
/// Simulates distributed communication without actual network operations.
pub struct MockBackend {
    rank: usize,
    world_size: usize,
    state: Arc<Mutex<SharedState>>,
}

impl MockBackend {
    /// Creates a collection of mock backends for testing.
    #[must_use]
    pub fn create_world(world_size: usize) -> Vec<Self> {
        let state = Arc::new(Mutex::new(SharedState {
            buffers: HashMap::new(),
            barrier_count: 0,
            messages: HashMap::new(),
        }));

        (0..world_size)
            .map(|rank| MockBackend {
                rank,
                world_size,
                state: Arc::clone(&state),
            })
            .collect()
    }

    /// Creates a single mock backend (rank 0, world size 1).
    #[must_use]
    pub fn single() -> Self {
        MockBackend::create_world(1).pop().unwrap()
    }
}

impl Backend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) {
        let mut state = self.state.lock().unwrap();

        // Store this rank's data
        state.buffers.insert(self.rank, data.to_vec());

        // Check if all ranks have submitted
        if state.buffers.len() == self.world_size {
            // Perform reduction
            let all_data: Vec<Vec<f32>> = (0..self.world_size)
                .map(|r| state.buffers.get(&r).cloned().unwrap_or_default())
                .collect();

            let reduced = op.reduce_slices(&all_data);

            // Update all buffers with result
            for r in 0..self.world_size {
                state.buffers.insert(r, reduced.clone());
            }
        }

        // Get result for this rank
        if let Some(result) = state.buffers.get(&self.rank) {
            for (i, &val) in result.iter().enumerate() {
                if i < data.len() {
                    data[i] = val;
                }
            }
        }

        // Clear buffers if this is the last rank to read
        if state.buffers.len() == self.world_size {
            state.buffers.clear();
        }
    }

    fn broadcast(&self, data: &mut [f32], src: usize) {
        let mut state = self.state.lock().unwrap();

        if self.rank == src {
            // Source rank stores its data
            state.buffers.insert(0, data.to_vec());
        }

        // Get broadcast data
        if let Some(src_data) = state.buffers.get(&0) {
            for (i, &val) in src_data.iter().enumerate() {
                if i < data.len() {
                    data[i] = val;
                }
            }
        }
    }

    fn all_gather(&self, send_data: &[f32], recv_data: &mut [f32]) {
        let mut state = self.state.lock().unwrap();

        // Store this rank's data
        state.buffers.insert(self.rank, send_data.to_vec());

        // Check if all ranks have submitted
        if state.buffers.len() == self.world_size {
            // Concatenate all data in rank order
            let chunk_size = send_data.len();
            for r in 0..self.world_size {
                if let Some(data) = state.buffers.get(&r) {
                    let start = r * chunk_size;
                    for (i, &val) in data.iter().enumerate() {
                        if start + i < recv_data.len() {
                            recv_data[start + i] = val;
                        }
                    }
                }
            }
        }
    }

    fn reduce_scatter(&self, send_data: &[f32], recv_data: &mut [f32], op: ReduceOp) {
        let mut state = self.state.lock().unwrap();

        // Store this rank's data
        state.buffers.insert(self.rank, send_data.to_vec());

        // Check if all ranks have submitted
        if state.buffers.len() == self.world_size {
            // First reduce all data
            let all_data: Vec<Vec<f32>> = (0..self.world_size)
                .map(|r| state.buffers.get(&r).cloned().unwrap_or_default())
                .collect();

            let reduced = op.reduce_slices(&all_data);

            // Scatter to each rank
            let chunk_size = recv_data.len();
            let start = self.rank * chunk_size;
            let end = (start + chunk_size).min(reduced.len());

            for (i, &val) in reduced[start..end].iter().enumerate() {
                if i < recv_data.len() {
                    recv_data[i] = val;
                }
            }
        }
    }

    fn gather(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize) {
        let mut state = self.state.lock().unwrap();

        // Store this rank's data
        state.buffers.insert(self.rank, send_data.to_vec());

        // Only destination rank collects
        if self.rank == dst && state.buffers.len() == self.world_size {
            let chunk_size = send_data.len();
            for r in 0..self.world_size {
                if let Some(data) = state.buffers.get(&r) {
                    let start = r * chunk_size;
                    for (i, &val) in data.iter().enumerate() {
                        if start + i < recv_data.len() {
                            recv_data[start + i] = val;
                        }
                    }
                }
            }
        }
    }

    fn scatter(&self, send_data: &[f32], recv_data: &mut [f32], src: usize) {
        let state = self.state.lock().unwrap();

        // Only source rank has full data
        if self.rank == src {
            // Scatter to all (in mock, we copy our portion)
            let chunk_size = recv_data.len();
            let start = self.rank * chunk_size;
            let end = (start + chunk_size).min(send_data.len());

            for (i, &val) in send_data[start..end].iter().enumerate() {
                recv_data[i] = val;
            }
        }
        drop(state);

        // Others would receive via message passing in real impl
    }

    fn reduce(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize, op: ReduceOp) {
        let mut state = self.state.lock().unwrap();

        // Store this rank's data
        state.buffers.insert(self.rank, send_data.to_vec());

        // Only destination rank reduces
        if self.rank == dst && state.buffers.len() == self.world_size {
            let all_data: Vec<Vec<f32>> = (0..self.world_size)
                .map(|r| state.buffers.get(&r).cloned().unwrap_or_default())
                .collect();

            let reduced = op.reduce_slices(&all_data);

            for (i, &val) in reduced.iter().enumerate() {
                if i < recv_data.len() {
                    recv_data[i] = val;
                }
            }
        }
    }

    fn barrier(&self) {
        let mut state = self.state.lock().unwrap();
        state.barrier_count += 1;

        // Reset when all have arrived
        if state.barrier_count == self.world_size {
            state.barrier_count = 0;
        }
    }

    fn send(&self, data: &[f32], dst: usize, tag: usize) {
        let mut state = self.state.lock().unwrap();
        state.messages.insert((self.rank, dst, tag), data.to_vec());
    }

    fn recv(&self, data: &mut [f32], src: usize, tag: usize) {
        let mut state = self.state.lock().unwrap();
        if let Some(msg) = state.messages.remove(&(src, self.rank, tag)) {
            for (i, &val) in msg.iter().enumerate() {
                if i < data.len() {
                    data[i] = val;
                }
            }
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
    fn test_reduce_op_sum() {
        let op = ReduceOp::Sum;
        assert_eq!(op.apply_f32(1.0, 2.0), 3.0);
    }

    #[test]
    fn test_reduce_op_product() {
        let op = ReduceOp::Product;
        assert_eq!(op.apply_f32(2.0, 3.0), 6.0);
    }

    #[test]
    fn test_reduce_op_min() {
        let op = ReduceOp::Min;
        assert_eq!(op.apply_f32(2.0, 3.0), 2.0);
    }

    #[test]
    fn test_reduce_op_max() {
        let op = ReduceOp::Max;
        assert_eq!(op.apply_f32(2.0, 3.0), 3.0);
    }

    #[test]
    fn test_reduce_slices_sum() {
        let op = ReduceOp::Sum;
        let slices = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = op.reduce_slices(&slices);
        assert_eq!(result, vec![9.0, 12.0]);
    }

    #[test]
    fn test_reduce_slices_average() {
        let op = ReduceOp::Average;
        let slices = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = op.reduce_slices(&slices);
        assert_eq!(result, vec![2.0, 3.0]);
    }

    #[test]
    fn test_mock_backend_single() {
        let backend = MockBackend::single();
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.world_size(), 1);
        assert_eq!(backend.name(), "mock");
    }

    #[test]
    fn test_mock_backend_world() {
        let backends = MockBackend::create_world(4);
        assert_eq!(backends.len(), 4);

        for (i, b) in backends.iter().enumerate() {
            assert_eq!(b.rank(), i);
            assert_eq!(b.world_size(), 4);
        }
    }

    #[test]
    fn test_mock_all_reduce() {
        // Note: In a real distributed system, all_reduce would be called from different
        // processes simultaneously. The mock backend simulates a single process,
        // so values remain unchanged when called sequentially from same thread.
        let backend = MockBackend::single();

        let mut data = vec![1.0, 2.0];
        backend.all_reduce(&mut data, ReduceOp::Sum);

        // With single rank, values remain the same
        assert_eq!(data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_mock_broadcast() {
        let backends = MockBackend::create_world(2);

        let mut data0 = vec![1.0, 2.0, 3.0];
        let mut data1 = vec![0.0, 0.0, 0.0];

        // Broadcast from rank 0
        backends[0].broadcast(&mut data0, 0);
        backends[1].broadcast(&mut data1, 0);

        assert_eq!(data0, vec![1.0, 2.0, 3.0]);
        assert_eq!(data1, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mock_send_recv() {
        let backends = MockBackend::create_world(2);

        // Send from rank 0 to rank 1
        let send_data = vec![1.0, 2.0, 3.0];
        backends[0].send(&send_data, 1, 0);

        // Receive on rank 1
        let mut recv_data = vec![0.0, 0.0, 0.0];
        backends[1].recv(&mut recv_data, 0, 0);

        assert_eq!(recv_data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mock_barrier() {
        let backends = MockBackend::create_world(2);

        // Both call barrier
        backends[0].barrier();
        backends[1].barrier();

        // Should not deadlock
    }
}
