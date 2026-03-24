# axonml-distributed Documentation

> Distributed training utilities for the Axonml ML framework.

## Overview

`axonml-distributed` provides tools for distributed deep learning across multiple GPUs and nodes. It includes DistributedDataParallel (DDP), communication primitives, and process group management.

## Core Concepts

### Process Groups

A process group represents a set of processes that can communicate:

```rust
use axonml_distributed::{World, ProcessGroup};

// Get world (all processes)
let world = World::new()?;
println!("Rank: {}", world.rank());
println!("World size: {}", world.world_size());

// Create sub-groups
let group = ProcessGroup::new(vec![0, 1, 2, 3]);
```

### Data Parallelism

In data parallel training:
1. Each process has a copy of the model
2. Data is split across processes
3. Gradients are synchronized after backward pass
4. All processes update with averaged gradients

## Modules

### ddp.rs

DistributedDataParallel wrapper for synchronized training.

```rust
use axonml_distributed::{DistributedDataParallel, DDP};

// Wrap model for distributed training
let model = create_model();
let ddp_model = DDP::new(model, world);

// Training loop (same as single-GPU)
for batch in train_loader.iter() {
    let output = ddp_model.forward(&batch.data);
    let loss = compute_loss(&output, &batch.targets);

    loss.backward();

    // Gradients are automatically synchronized here
    optimizer.step();
    optimizer.zero_grad();
}
```

#### DDP Builder Pattern

```rust
let ddp_model = DDP::builder(model)
    .world(world)
    .bucket_size_mb(25.0)      // Gradient bucket size
    .broadcast_buffers(true)   // Sync batch norm stats
    .build();
```

### comm.rs

High-level communication primitives.

#### All-Reduce

Reduce tensors across all processes:

```rust
use axonml_distributed::{all_reduce_sum, all_reduce_mean};

// Sum across all ranks
let local_tensor = compute_local_gradient();
let global_sum = all_reduce_sum(&local_tensor);

// Average across all ranks
let global_mean = all_reduce_mean(&local_tensor);
```

#### Broadcast

Send tensor from one rank to all others:

```rust
use axonml_distributed::broadcast;

// Broadcast from rank 0
let tensor = if world.rank() == 0 {
    initialize_weights()
} else {
    Tensor::zeros(&shape)
};

let synced = broadcast(&tensor, 0);  // src_rank = 0
```

#### Barrier

Synchronize all processes:

```rust
use axonml_distributed::barrier;

// Wait for all processes to reach this point
barrier();
```

### process_group.rs

Process group management.

```rust
use axonml_distributed::{World, ProcessGroup};

// Initialize world
let world = World::new()?;

// Create custom groups
let even_ranks = ProcessGroup::new((0..world.world_size()).step_by(2).collect());
let odd_ranks = ProcessGroup::new((1..world.world_size()).step_by(2).collect());

// Use groups for communication
all_reduce_sum_group(&tensor, &even_ranks);
```

### backend.rs

Communication backend implementations.

```rust
use axonml_distributed::{Backend, ReduceOp};

// Available backends
let backend = Backend::Gloo;   // CPU distributed
let backend = Backend::NCCL;   // GPU distributed (CUDA)
let backend = Backend::Mock;   // For testing

// Reduce operations
let op = ReduceOp::Sum;
let op = ReduceOp::Mean;
let op = ReduceOp::Max;
let op = ReduceOp::Min;
```

## Usage Examples

### Basic DDP Training

```rust
use axonml::prelude::*;

fn main() {
    // Initialize distributed environment
    let world = World::new().expect("Failed to init distributed");
    let rank = world.rank();
    let world_size = world.world_size();

    println!("Process {}/{}", rank, world_size);

    // Create model (same on all ranks)
    let model = create_model();

    // Wrap with DDP
    let ddp_model = DDP::new(model, world.clone());

    // Create distributed data loader
    // Each rank gets a different shard of data
    let dataset = load_dataset();
    let sampler = DistributedSampler::new(&dataset, world_size, rank);
    let loader = DataLoader::with_sampler(dataset, 32, sampler);

    // Optimizer
    let mut optimizer = Adam::new(ddp_model.parameters(), 0.001);

    // Training loop
    for epoch in 0..epochs {
        for batch in loader.iter() {
            let output = ddp_model.forward(&batch.data);
            let loss = compute_loss(&output, &batch.targets);

            loss.backward();
            // Gradients synchronized automatically by DDP

            optimizer.step();
            optimizer.zero_grad();
        }

        // Sync metrics across ranks
        let local_loss = compute_epoch_loss();
        let global_loss = all_reduce_mean(&local_loss);

        if rank == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, global_loss);
        }
    }

    // Save checkpoint from rank 0 only
    if rank == 0 {
        save_checkpoint(&ddp_model.module());
    }
}
```

### Multi-Node Setup

```bash
# Node 0 (master)
MASTER_ADDR=192.168.1.1 MASTER_PORT=29500 \
WORLD_SIZE=8 RANK=0 \
cargo run --release

# Node 1
MASTER_ADDR=192.168.1.1 MASTER_PORT=29500 \
WORLD_SIZE=8 RANK=4 \
cargo run --release
```

### Gradient Accumulation with DDP

```rust
let accumulation_steps = 4;

for (i, batch) in loader.iter().enumerate() {
    let output = ddp_model.forward(&batch.data);
    let loss = compute_loss(&output, &batch.targets) / accumulation_steps;

    loss.backward();

    if (i + 1) % accumulation_steps == 0 {
        // Sync gradients only when updating
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

### Model Checkpointing

```rust
// Save (only on rank 0)
if world.rank() == 0 {
    let model = ddp_model.module();
    save_model(model, "checkpoint.pt")?;
}
barrier();  // Wait for save to complete

// Load (all ranks)
let model = load_model("checkpoint.pt")?;
let ddp_model = DDP::new(model, world);
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MASTER_ADDR` | IP address of rank 0 |
| `MASTER_PORT` | Port for communication |
| `WORLD_SIZE` | Total number of processes |
| `RANK` | Global rank of this process |
| `LOCAL_RANK` | Local rank (for multi-GPU per node) |

## Best Practices

1. **Same seed** - Use same random seed on all ranks for reproducibility
2. **Rank 0 I/O** - Only do file I/O from rank 0
3. **Barrier before load** - Sync before loading checkpoints
4. **Gradient accumulation** - Accumulate before sync for efficiency
5. **Mixed precision** - Combine with AMP for better performance

## Related Modules

- [Neural Networks](../nn/README.md) - Models to distribute
- [Optimizers](../optim/README.md) - Parameter updates
- [Data](../data/README.md) - Distributed data loading

@version 0.1.0
@author AutomataNexus Development Team
