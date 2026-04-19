//! Backend - Communication Backend Abstractions
//!
//! # File
//! `crates/axonml-distributed/src/backend.rs`
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

use std::collections::HashMap;
use std::sync::{Arc, Barrier, Mutex};

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
            ReduceOp::Average => f32::midpoint(a, b),
        }
    }

    /// Applies the reduction operation to slices.
    #[must_use]
    pub fn reduce_slices(&self, slices: &[Vec<f32>]) -> Vec<f32> {
        if slices.is_empty() {
            return Vec::new();
        }

        let len = slices[0].len();

        // Average gets its own path to avoid incorrect pairwise midpoint
        if *self == ReduceOp::Average {
            let mut result = vec![0.0f32; len];
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
            return result;
        }

        // All other ops: pairwise reduction
        let mut result = slices[0].clone();
        for slice in slices.iter().skip(1) {
            for (i, &val) in slice.iter().enumerate() {
                if i < len {
                    result[i] = self.apply_f32(result[i], val);
                }
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
///
/// The MockBackend simulates an N-rank collective by running each rank on
/// its own OS thread. Every collective op is a two-phase cycle:
///
///   Phase 1 — *submit* : each rank writes its contribution into
///                        `buffers`, then meets `submit_barrier`.
///   Phase 2 — *read*   : each rank reads the collective result out of
///                        `buffers`, then meets `read_barrier`.
///
/// The `read_barrier` is essential: without it, the first rank out of
/// submit phase can call the NEXT collective op and overwrite `buffers`
/// before slower ranks have had a chance to read. `std::sync::Barrier`
/// is reusable — it auto-resets once all `world_size` threads reach it,
/// which makes it the natural building block for this cycle.
struct SharedState {
    /// Data buffers for each rank, populated per-round by the submit phase.
    /// The sentinel key `usize::MAX` is used by reductions/scatter to
    /// store the full reduced / scatter-source vector.
    buffers: HashMap<usize, Vec<f32>>,
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
    /// Meets after every rank has written its contribution into `state`.
    submit_barrier: Arc<Barrier>,
    /// Meets after every rank has read the result out of `state`. Ensures
    /// no rank can start the next collective (and clobber buffers) before
    /// everyone has read the current one.
    read_barrier: Arc<Barrier>,
    /// Separate barrier for explicit `barrier()` calls, decoupled from
    /// the collective cycle.
    explicit_barrier: Arc<Barrier>,
}

impl MockBackend {
    /// Creates a collection of mock backends for testing.
    #[must_use]
    pub fn create_world(world_size: usize) -> Vec<Self> {
        let state = Arc::new(Mutex::new(SharedState {
            buffers: HashMap::new(),
            messages: HashMap::new(),
        }));
        let submit_barrier = Arc::new(Barrier::new(world_size));
        let read_barrier = Arc::new(Barrier::new(world_size));
        let explicit_barrier = Arc::new(Barrier::new(world_size));

        (0..world_size)
            .map(|rank| MockBackend {
                rank,
                world_size,
                state: Arc::clone(&state),
                submit_barrier: Arc::clone(&submit_barrier),
                read_barrier: Arc::clone(&read_barrier),
                explicit_barrier: Arc::clone(&explicit_barrier),
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
        // --- Submit phase ---
        {
            let mut state = self.state.lock().unwrap();
            state.buffers.insert(self.rank, data.to_vec());
        }
        self.submit_barrier.wait();

        // All ranks have submitted. Any one rank could compute the
        // reduction, but we have every rank do it from its own lock so
        // the result is available without an extra broadcast. The
        // `state` lock serializes so in practice only one rank's
        // reduction actually writes to `buffers[self.rank]` — all ranks
        // read the same `buffers[rank]` they wrote.
        {
            let state = self.state.lock().unwrap();
            let all_data: Vec<Vec<f32>> = (0..self.world_size)
                .map(|r| state.buffers.get(&r).cloned().unwrap_or_default())
                .collect();
            let reduced = op.reduce_slices(&all_data);
            for (i, &val) in reduced.iter().enumerate() {
                if i < data.len() {
                    data[i] = val;
                }
            }
        }

        // --- Read-complete phase: ensure nobody starts the next op
        // before every rank has read the buffers from this one. ---
        self.read_barrier.wait();
        // Rank 0 clears buffers for the next cycle (safe because
        // read_barrier guarantees everyone has already read).
        if self.rank == 0 {
            let mut state = self.state.lock().unwrap();
            state.buffers.clear();
        }
    }

    fn broadcast(&self, data: &mut [f32], src: usize) {
        // --- Submit phase: src writes its data ---
        if self.rank == src {
            let mut state = self.state.lock().unwrap();
            state.buffers.insert(src, data.to_vec());
        }
        self.submit_barrier.wait();

        // --- Read phase: every rank reads buffers[src] ---
        {
            let state = self.state.lock().unwrap();
            if let Some(src_data) = state.buffers.get(&src) {
                for (i, &val) in src_data.iter().enumerate() {
                    if i < data.len() {
                        data[i] = val;
                    }
                }
            }
        }

        self.read_barrier.wait();
        if self.rank == 0 {
            let mut state = self.state.lock().unwrap();
            state.buffers.clear();
        }
    }

    fn all_gather(&self, send_data: &[f32], recv_data: &mut [f32]) {
        {
            let mut state = self.state.lock().unwrap();
            state.buffers.insert(self.rank, send_data.to_vec());
        }
        self.submit_barrier.wait();

        // Every rank reconstructs the full concat-in-rank-order result.
        {
            let state = self.state.lock().unwrap();
            let chunk_size = send_data.len();
            for r in 0..self.world_size {
                if let Some(d) = state.buffers.get(&r) {
                    let start = r * chunk_size;
                    for (i, &val) in d.iter().enumerate() {
                        if start + i < recv_data.len() {
                            recv_data[start + i] = val;
                        }
                    }
                }
            }
        }

        self.read_barrier.wait();
        if self.rank == 0 {
            let mut state = self.state.lock().unwrap();
            state.buffers.clear();
        }
    }

    fn reduce_scatter(&self, send_data: &[f32], recv_data: &mut [f32], op: ReduceOp) {
        {
            let mut state = self.state.lock().unwrap();
            state.buffers.insert(self.rank, send_data.to_vec());
        }
        self.submit_barrier.wait();

        {
            let state = self.state.lock().unwrap();
            let all_data: Vec<Vec<f32>> = (0..self.world_size)
                .map(|r| state.buffers.get(&r).cloned().unwrap_or_default())
                .collect();
            let reduced = op.reduce_slices(&all_data);
            let chunk_size = recv_data.len();
            let start = self.rank * chunk_size;
            let end = (start + chunk_size).min(reduced.len());
            for (i, &val) in reduced[start..end].iter().enumerate() {
                if i < recv_data.len() {
                    recv_data[i] = val;
                }
            }
        }

        self.read_barrier.wait();
        if self.rank == 0 {
            let mut state = self.state.lock().unwrap();
            state.buffers.clear();
        }
    }

    fn gather(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize) {
        {
            let mut state = self.state.lock().unwrap();
            state.buffers.insert(self.rank, send_data.to_vec());
        }
        self.submit_barrier.wait();

        if self.rank == dst {
            let state = self.state.lock().unwrap();
            let chunk_size = send_data.len();
            for r in 0..self.world_size {
                if let Some(d) = state.buffers.get(&r) {
                    let start = r * chunk_size;
                    for (i, &val) in d.iter().enumerate() {
                        if start + i < recv_data.len() {
                            recv_data[start + i] = val;
                        }
                    }
                }
            }
        }

        self.read_barrier.wait();
        if self.rank == 0 {
            let mut state = self.state.lock().unwrap();
            state.buffers.clear();
        }
    }

    fn scatter(&self, send_data: &[f32], recv_data: &mut [f32], src: usize) {
        // Only `src` provides the full vector — stash under a sentinel
        // key so every other rank can slice out its chunk.
        if self.rank == src {
            let mut state = self.state.lock().unwrap();
            state.buffers.insert(usize::MAX, send_data.to_vec());
        }
        self.submit_barrier.wait();

        {
            let state = self.state.lock().unwrap();
            if let Some(full) = state.buffers.get(&usize::MAX) {
                let chunk_size = recv_data.len();
                let start = self.rank * chunk_size;
                let end = (start + chunk_size).min(full.len());
                for (i, &val) in full[start..end].iter().enumerate() {
                    if i < recv_data.len() {
                        recv_data[i] = val;
                    }
                }
            }
        }

        self.read_barrier.wait();
        if self.rank == 0 {
            let mut state = self.state.lock().unwrap();
            state.buffers.clear();
        }
    }

    fn reduce(&self, send_data: &[f32], recv_data: &mut [f32], dst: usize, op: ReduceOp) {
        {
            let mut state = self.state.lock().unwrap();
            state.buffers.insert(self.rank, send_data.to_vec());
        }
        self.submit_barrier.wait();

        if self.rank == dst {
            let state = self.state.lock().unwrap();
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

        self.read_barrier.wait();
        if self.rank == 0 {
            let mut state = self.state.lock().unwrap();
            state.buffers.clear();
        }
    }

    fn barrier(&self) {
        self.explicit_barrier.wait();
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
        // Collective ops on the Condvar-based MockBackend require every
        // rank to arrive concurrently — running them sequentially on one
        // thread deadlocks. Spawn one thread per rank and join.
        use std::thread;

        let backends = MockBackend::create_world(2);
        let handles: Vec<_> = backends
            .into_iter()
            .enumerate()
            .map(|(rank, backend)| {
                thread::spawn(move || {
                    let mut data = if rank == 0 {
                        vec![1.0, 2.0, 3.0]
                    } else {
                        vec![0.0, 0.0, 0.0]
                    };
                    backend.broadcast(&mut data, 0);
                    data
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_mock_all_reduce_sum_world_of_three() {
        use std::thread;
        let backends = MockBackend::create_world(3);
        let handles: Vec<_> = backends
            .into_iter()
            .enumerate()
            .map(|(rank, backend)| {
                thread::spawn(move || {
                    let mut data = vec![(rank + 1) as f32; 4];
                    backend.all_reduce(&mut data, ReduceOp::Sum);
                    data
                })
            })
            .collect();
        for h in handles {
            // 1 + 2 + 3 = 6 on every element on every rank.
            for v in h.join().unwrap() {
                assert!((v - 6.0).abs() < 1e-6, "sum: got {v}, want 6.0");
            }
        }
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
        use std::thread;
        let backends = MockBackend::create_world(2);
        let handles: Vec<_> = backends
            .into_iter()
            .map(|backend| thread::spawn(move || backend.barrier()))
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
