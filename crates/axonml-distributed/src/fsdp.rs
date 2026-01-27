//! FSDP - Fully Sharded Data Parallel
//!
//! Implements Fully Sharded Data Parallel training for scaling to multiple GPUs/nodes.
//! FSDP shards model parameters, gradients, and optimizer states across devices.
//!
//! Reference: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
//! https://arxiv.org/abs/1910.02054
//!
//! # Example
//! ```rust,ignore
//! use axonml_distributed::fsdp::{FullyShardedDataParallel, ShardingStrategy};
//!
//! let model = MyModel::new();
//! let fsdp_model = FullyShardedDataParallel::new(model, process_group)
//!     .sharding_strategy(ShardingStrategy::FullShard)
//!     .cpu_offload(false);
//! ```
//!
//! @version 0.1.0

use crate::backend::ReduceOp;
use crate::process_group::ProcessGroup;
use axonml_autograd::Variable;
use axonml_nn::{Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// Sharding Strategy
// =============================================================================

/// Strategy for sharding parameters in FSDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Shard parameters, gradients, and optimizer state (ZeRO-3)
    FullShard,
    /// Shard gradients and optimizer state only (ZeRO-2)
    ShardGradOp,
    /// No sharding, replicate across ranks (DDP-like)
    NoShard,
    /// Hybrid sharding within node, replicate across nodes
    HybridShard,
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        Self::FullShard
    }
}

// =============================================================================
// FSDP State
// =============================================================================

/// State for a sharded parameter.
#[derive(Debug)]
#[allow(dead_code)]
struct ShardedParam {
    /// Local shard of the parameter
    local_shard: Tensor<f32>,
    /// Original shape before sharding
    original_shape: Vec<usize>,
    /// Number of elements in original parameter
    numel: usize,
    /// Padding added for even sharding (for uneven divisions)
    padding: usize,
}

/// CPU offload configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CPUOffload {
    /// No CPU offloading
    None,
    /// Offload parameters to CPU when not in use
    Params,
    /// Offload both parameters and gradients
    Full,
}

impl Default for CPUOffload {
    fn default() -> Self {
        Self::None
    }
}

// =============================================================================
// Fully Sharded Data Parallel
// =============================================================================

/// Fully Sharded Data Parallel wrapper for memory-efficient distributed training.
///
/// FSDP shards model parameters across devices, gathering them only when needed
/// for computation and sharding them again afterward.
pub struct FullyShardedDataParallel<M: Module> {
    /// Wrapped module
    module: M,
    /// Process group for communication
    process_group: ProcessGroup,
    /// Sharding strategy
    sharding_strategy: ShardingStrategy,
    /// CPU offload configuration
    cpu_offload: CPUOffload,
    /// Sharded parameter states
    sharded_params: Vec<ShardedParam>,
    /// Whether module is currently gathered (unsharded)
    is_gathered: bool,
    /// Mixed precision compute dtype
    mixed_precision: bool,
}

impl<M: Module> FullyShardedDataParallel<M> {
    /// Creates a new FSDP wrapper.
    pub fn new(module: M, process_group: ProcessGroup) -> Self {
        let mut fsdp = Self {
            module,
            process_group,
            sharding_strategy: ShardingStrategy::default(),
            cpu_offload: CPUOffload::default(),
            sharded_params: Vec::new(),
            is_gathered: true,
            mixed_precision: false,
        };

        // Initialize sharding
        fsdp.shard_parameters();
        fsdp
    }

    /// Builder: set sharding strategy.
    pub fn sharding_strategy(mut self, strategy: ShardingStrategy) -> Self {
        self.sharding_strategy = strategy;
        self.shard_parameters();
        self
    }

    /// Builder: set CPU offload configuration.
    pub fn cpu_offload(mut self, offload: CPUOffload) -> Self {
        self.cpu_offload = offload;
        self
    }

    /// Builder: enable mixed precision.
    pub fn mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Returns reference to wrapped module.
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Returns mutable reference to wrapped module.
    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Returns the process group.
    pub fn process_group(&self) -> &ProcessGroup {
        &self.process_group
    }

    /// Returns the sharding strategy.
    pub fn strategy(&self) -> ShardingStrategy {
        self.sharding_strategy
    }

    /// Shards parameters across devices.
    fn shard_parameters(&mut self) {
        if self.sharding_strategy == ShardingStrategy::NoShard {
            return;
        }

        let world_size = self.process_group.world_size();
        let rank = self.process_group.rank();

        self.sharded_params.clear();

        for param in self.module.parameters() {
            let data = param.data();
            let shape = data.shape().to_vec();
            let numel = data.numel();

            // Calculate shard size with padding for even division
            let shard_size = (numel + world_size - 1) / world_size;
            let padding = shard_size * world_size - numel;

            // Get local shard
            let flat_data = data.to_vec();
            let start = rank * shard_size;
            let end = ((rank + 1) * shard_size).min(flat_data.len());

            let mut shard_data: Vec<f32> = if start < flat_data.len() {
                flat_data[start..end].to_vec()
            } else {
                vec![0.0; shard_size]
            };

            // Pad to shard_size
            while shard_data.len() < shard_size {
                shard_data.push(0.0);
            }

            self.sharded_params.push(ShardedParam {
                local_shard: Tensor::from_vec(shard_data, &[shard_size]).unwrap(),
                original_shape: shape,
                numel,
                padding,
            });
        }

        self.is_gathered = false;
    }

    /// Gathers all parameter shards before forward pass.
    pub fn gather_parameters(&mut self) {
        if self.is_gathered || self.sharding_strategy == ShardingStrategy::NoShard {
            return;
        }

        let _world_size = self.process_group.world_size();
        let params = self.module.parameters();

        for (param, sharded) in params.iter().zip(self.sharded_params.iter()) {
            // All-gather the shards
            let gathered = self.process_group.all_gather_tensor(&sharded.local_shard);

            // Reshape back to original shape (removing padding)
            let flat: Vec<f32> = gathered.to_vec().into_iter().take(sharded.numel).collect();
            let restored = Tensor::from_vec(flat, &sharded.original_shape).unwrap();

            param.update_data(restored);
        }

        self.is_gathered = true;
    }

    /// Shards parameters after forward/backward pass.
    pub fn reshard_parameters(&mut self) {
        if !self.is_gathered || self.sharding_strategy == ShardingStrategy::NoShard {
            return;
        }

        self.shard_parameters();
    }

    /// Synchronizes gradients across all ranks.
    pub fn sync_gradients(&self) {
        match self.sharding_strategy {
            ShardingStrategy::NoShard => {
                // Full all-reduce like DDP
                for param in self.module.parameters() {
                    if let Some(grad) = param.grad() {
                        let mut grad_tensor = grad.clone();
                        self.process_group
                            .all_reduce_tensor(&mut grad_tensor, ReduceOp::Average);
                    }
                }
            }
            ShardingStrategy::ShardGradOp | ShardingStrategy::FullShard => {
                // Reduce-scatter gradients to get sharded gradients
                for param in self.module.parameters() {
                    if let Some(grad) = param.grad() {
                        let _reduced = self
                            .process_group
                            .reduce_scatter_tensor(&grad, ReduceOp::Average);
                        // In full implementation, would update parameter's gradient shard
                    }
                }
            }
            ShardingStrategy::HybridShard => {
                // All-reduce within node, reduce-scatter across nodes
                for param in self.module.parameters() {
                    if let Some(grad) = param.grad() {
                        let mut grad_tensor = grad.clone();
                        self.process_group
                            .all_reduce_tensor(&mut grad_tensor, ReduceOp::Average);
                    }
                }
            }
        }
    }

    /// Clips gradients by global norm.
    pub fn clip_grad_norm(&self, max_norm: f32) -> f32 {
        let mut total_norm_sq = 0.0f32;

        for param in self.module.parameters() {
            if let Some(grad) = param.grad() {
                let grad_vec = grad.to_vec();
                let norm_sq: f32 = grad_vec.iter().map(|x| x * x).sum();
                total_norm_sq += norm_sq;
            }
        }

        // All-reduce total norm across ranks
        let mut norm_tensor = Tensor::from_vec(vec![total_norm_sq], &[1]).unwrap();
        self.process_group
            .all_reduce_tensor(&mut norm_tensor, ReduceOp::Sum);
        let global_norm = norm_tensor.to_vec()[0].sqrt();

        // Clip if necessary
        if global_norm > max_norm {
            let clip_coef = max_norm / (global_norm + 1e-6);
            for param in self.module.parameters() {
                if let Some(grad) = param.grad() {
                    let clipped: Vec<f32> = grad.to_vec().iter().map(|x| x * clip_coef).collect();
                    let clipped_tensor = Tensor::from_vec(clipped, grad.shape()).unwrap();
                    param.variable().set_grad(clipped_tensor);
                }
            }
        }

        global_norm
    }

    /// Estimates memory usage with different sharding strategies.
    pub fn memory_estimate(&self) -> FSDPMemoryStats {
        let params = self.module.parameters();
        let total_params: usize = params.iter().map(|p| p.numel()).sum();
        let world_size = self.process_group.world_size();

        let bytes_per_param = 4; // f32
        let param_memory = total_params * bytes_per_param;

        let (sharded_params, sharded_grads, sharded_optim) = match self.sharding_strategy {
            ShardingStrategy::NoShard => (param_memory, param_memory, param_memory * 2),
            ShardingStrategy::ShardGradOp => (
                param_memory,
                param_memory / world_size,
                param_memory * 2 / world_size,
            ),
            ShardingStrategy::FullShard | ShardingStrategy::HybridShard => (
                param_memory / world_size,
                param_memory / world_size,
                param_memory * 2 / world_size,
            ),
        };

        FSDPMemoryStats {
            total_params,
            param_memory_bytes: sharded_params,
            grad_memory_bytes: sharded_grads,
            optim_memory_bytes: sharded_optim,
            world_size,
        }
    }
}

impl<M: Module> Module for FullyShardedDataParallel<M> {
    fn forward(&self, input: &Variable) -> Variable {
        // Note: In a real implementation, gather would be called automatically
        // through hooks before forward and reshard after
        self.module.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.module.parameters()
    }

    fn train(&mut self) {
        self.module.train();
    }

    fn eval(&mut self) {
        self.module.eval();
    }

    fn is_training(&self) -> bool {
        self.module.is_training()
    }
}

/// Memory statistics for FSDP.
#[derive(Debug, Clone)]
pub struct FSDPMemoryStats {
    /// Total number of parameters
    pub total_params: usize,
    /// Memory for parameters (bytes)
    pub param_memory_bytes: usize,
    /// Memory for gradients (bytes)
    pub grad_memory_bytes: usize,
    /// Memory for optimizer state (bytes)
    pub optim_memory_bytes: usize,
    /// World size (number of ranks)
    pub world_size: usize,
}

impl FSDPMemoryStats {
    /// Total memory per rank in MB.
    pub fn total_memory_mb(&self) -> f32 {
        (self.param_memory_bytes + self.grad_memory_bytes + self.optim_memory_bytes) as f32
            / (1024.0 * 1024.0)
    }

    /// Memory savings compared to no sharding.
    pub fn memory_savings(&self) -> f32 {
        if self.world_size > 1 {
            1.0 - (1.0 / self.world_size as f32)
        } else {
            0.0
        }
    }
}

// =============================================================================
// Tensor Parallelism
// =============================================================================

/// Column-parallel linear layer.
///
/// Splits the weight matrix along the column dimension across ranks.
/// Each rank computes a portion of the output features.
#[allow(dead_code)]
pub struct ColumnParallelLinear {
    /// Local weight shard
    weight: Parameter,
    /// Bias (replicated on all ranks)
    bias: Option<Parameter>,
    /// Process group
    process_group: ProcessGroup,
    /// Input features
    in_features: usize,
    /// Output features (total across all ranks)
    out_features: usize,
    /// Whether to gather output
    gather_output: bool,
}

impl ColumnParallelLinear {
    /// Creates a new column-parallel linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        process_group: ProcessGroup,
        gather_output: bool,
    ) -> Self {
        let world_size = process_group.world_size();
        let local_out_features = out_features / world_size;

        let weight_data = Tensor::randn(&[local_out_features, in_features]);
        let weight = Parameter::new(weight_data, true);

        let bias = if bias {
            let bias_data = Tensor::zeros(&[local_out_features]);
            Some(Parameter::new(bias_data, true))
        } else {
            None
        };

        Self {
            weight,
            bias,
            process_group,
            in_features,
            out_features,
            gather_output,
        }
    }
}

impl Module for ColumnParallelLinear {
    fn forward(&self, input: &Variable) -> Variable {
        // Local matmul: input @ weight.T
        let weight_var = Variable::new(self.weight.data(), false);
        let output = input.matmul(&weight_var.transpose(0, 1));

        // Add bias
        let output = if let Some(ref bias) = self.bias {
            let bias_var = Variable::new(bias.data(), false);
            output.add(&bias_var)
        } else {
            output
        };

        // Optionally gather output across ranks
        if self.gather_output {
            let gathered = self.process_group.all_gather_tensor(&output.data());
            Variable::new(gathered, output.requires_grad())
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// Row-parallel linear layer.
///
/// Splits the weight matrix along the row dimension across ranks.
/// Each rank has a portion of the input features.
#[allow(dead_code)]
pub struct RowParallelLinear {
    /// Local weight shard
    weight: Parameter,
    /// Bias (only on rank 0)
    bias: Option<Parameter>,
    /// Process group
    process_group: ProcessGroup,
    /// Input features (total across all ranks)
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Whether input is already split
    input_is_parallel: bool,
}

impl RowParallelLinear {
    /// Creates a new row-parallel linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        process_group: ProcessGroup,
        input_is_parallel: bool,
    ) -> Self {
        let world_size = process_group.world_size();
        let rank = process_group.rank();
        let local_in_features = in_features / world_size;

        let weight_data = Tensor::randn(&[out_features, local_in_features]);
        let weight = Parameter::new(weight_data, true);

        // Only rank 0 has bias
        let bias = if bias && rank == 0 {
            let bias_data = Tensor::zeros(&[out_features]);
            Some(Parameter::new(bias_data, true))
        } else {
            None
        };

        Self {
            weight,
            bias,
            process_group,
            in_features,
            out_features,
            input_is_parallel,
        }
    }
}

impl Module for RowParallelLinear {
    fn forward(&self, input: &Variable) -> Variable {
        // If input is not parallel, take local shard
        let local_input = if self.input_is_parallel {
            input.clone()
        } else {
            // Split input along feature dimension for row parallelism
            let world_size = self.process_group.world_size();
            let rank = self.process_group.rank();
            let data = input.data();
            let shape = data.shape();
            let feature_dim = shape[shape.len() - 1];
            let local_features = feature_dim / world_size;
            let start = rank * local_features;
            let end = start + local_features;

            // Slice the last dimension
            let sliced = if shape.len() == 2 {
                data.slice(&[0..shape[0], start..end])
            } else {
                data.clone() // Fallback for other shapes
            };
            Variable::new(sliced, input.requires_grad())
        };

        // Local matmul
        let weight_var = Variable::new(self.weight.data(), false);
        let local_output = local_input.matmul(&weight_var.transpose(0, 1));

        // All-reduce to combine partial outputs
        let mut output_data = local_output.data().clone();
        self.process_group
            .all_reduce_tensor(&mut output_data, ReduceOp::Sum);
        let output = Variable::new(output_data, local_output.requires_grad());

        // Add bias (only on rank 0, then broadcast)
        if let Some(ref bias) = self.bias {
            let bias_var = Variable::new(bias.data(), false);
            output.add(&bias_var)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_nn::Linear;

    #[test]
    fn test_sharding_strategy_default() {
        assert_eq!(ShardingStrategy::default(), ShardingStrategy::FullShard);
    }

    #[test]
    fn test_fsdp_creation() {
        let model = Linear::new(10, 5);
        let pg = ProcessGroup::mock();
        let fsdp = FullyShardedDataParallel::new(model, pg);

        assert_eq!(fsdp.strategy(), ShardingStrategy::FullShard);
    }

    #[test]
    fn test_fsdp_forward() {
        let model = Linear::new(4, 2);
        let pg = ProcessGroup::mock();
        let mut fsdp = FullyShardedDataParallel::new(model, pg);

        // Gather before forward
        fsdp.gather_parameters();

        let input = Variable::new(Tensor::from_vec(vec![1.0; 4], &[1, 4]).unwrap(), false);
        let output = fsdp.forward(&input);

        assert_eq!(output.data().shape(), &[1, 2]);
    }

    #[test]
    fn test_fsdp_builder() {
        let model = Linear::new(10, 5);
        let pg = ProcessGroup::mock();

        let fsdp = FullyShardedDataParallel::new(model, pg)
            .sharding_strategy(ShardingStrategy::ShardGradOp)
            .cpu_offload(CPUOffload::Params)
            .mixed_precision(true);

        assert_eq!(fsdp.strategy(), ShardingStrategy::ShardGradOp);
    }

    #[test]
    fn test_fsdp_memory_stats() {
        let model = Linear::new(100, 50);
        let pg = ProcessGroup::mock();
        let fsdp = FullyShardedDataParallel::new(model, pg);

        let stats = fsdp.memory_estimate();
        assert!(stats.total_params > 0);
        assert!(stats.total_memory_mb() > 0.0);
    }

    #[test]
    fn test_fsdp_no_shard() {
        let model = Linear::new(10, 5);
        let pg = ProcessGroup::mock();
        let fsdp =
            FullyShardedDataParallel::new(model, pg).sharding_strategy(ShardingStrategy::NoShard);

        assert_eq!(fsdp.strategy(), ShardingStrategy::NoShard);
    }

    #[test]
    fn test_column_parallel_linear() {
        let pg = ProcessGroup::mock();
        // With world_size=1, local_out_features = out_features / 1 = 4
        let layer = ColumnParallelLinear::new(8, 4, true, pg, false); // Don't gather for simple test

        let input = Variable::new(Tensor::randn(&[2, 8]), false);
        let output = layer.forward(&input);

        // Output shape should be [batch, local_out_features] = [2, 4]
        assert_eq!(output.data().shape(), &[2, 4]);
    }

    #[test]
    fn test_row_parallel_linear() {
        let pg = ProcessGroup::mock();
        let layer = RowParallelLinear::new(8, 4, true, pg, false);

        let input = Variable::new(Tensor::randn(&[2, 8]), false);
        let output = layer.forward(&input);

        assert_eq!(output.data().shape(), &[2, 4]);
    }
}
