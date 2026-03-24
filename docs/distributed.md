---
layout: default
title: Distributed Training
nav_order: 6
description: "Multi-GPU and distributed training with AxonML"
---

# Distributed Training
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AxonML provides comprehensive distributed training support:

| Strategy | Description | Memory | Communication |
|:---------|:------------|:-------|:--------------|
| **DDP** | Data parallelism with gradient sync | Full model per GPU | All-reduce gradients |
| **FSDP** | Fully sharded data parallel | Sharded across GPUs | All-gather params |
| **Pipeline** | Model split across stages | Partitioned model | Point-to-point |

## Data Distributed Parallel (DDP)

### Basic Usage

```rust
use axonml::distributed::{World, DDP};

fn main() {
    // Initialize distributed world
    let world = World::init();
    let rank = world.rank();
    let world_size = world.world_size();

    println!("Process {}/{}", rank, world_size);

    // Create model
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));

    // Wrap in DDP
    let mut ddp_model = DDP::new(model, world.default_group().clone());

    // Broadcast initial parameters from rank 0
    ddp_model.sync_parameters();

    // Training loop (same as single GPU)
    let mut optimizer = Adam::new(ddp_model.parameters(), 0.001);

    for epoch in 0..10 {
        for (inputs, targets) in train_loader.iter() {
            let output = ddp_model.forward(&inputs);
            let loss = loss_fn.compute(&output, &targets);

            optimizer.zero_grad();
            loss.backward();

            // Gradient sync happens automatically
            ddp_model.sync_gradients();

            optimizer.step();
        }
    }
}
```

### Launch Script

```bash
# Launch with 4 GPUs
torchrun --nproc_per_node=4 train.py

# Or with MPI
mpirun -np 4 ./target/release/train
```

## Fully Sharded Data Parallel (FSDP)

FSDP implements ZeRO optimization stages:

| Stage | Sharded | Memory Reduction |
|:------|:--------|:-----------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~Nx (N = world size) |

### Basic FSDP

```rust
use axonml::distributed::{FSDP, ShardingStrategy, CPUOffload};

let model = create_large_model();

// Wrap in FSDP (ZeRO-3 by default)
let fsdp_model = FSDP::new(model, world.default_group().clone())
    .sharding_strategy(ShardingStrategy::FullShard)  // ZeRO-3
    .cpu_offload(CPUOffload::None);

// Training is the same
let output = fsdp_model.forward(&input);
```

### Sharding Strategies

```rust
use axonml::distributed::ShardingStrategy;

// ZeRO-2: Shard optimizer + gradients
let fsdp = FSDP::new(model, pg)
    .sharding_strategy(ShardingStrategy::ShardGradOp);

// ZeRO-3: Full sharding
let fsdp = FSDP::new(model, pg)
    .sharding_strategy(ShardingStrategy::FullShard);

// No sharding (like DDP)
let fsdp = FSDP::new(model, pg)
    .sharding_strategy(ShardingStrategy::NoShard);
```

### CPU Offloading

```rust
use axonml::distributed::CPUOffload;

// Offload parameters and gradients to CPU
let fsdp = FSDP::new(model, pg)
    .cpu_offload(CPUOffload::Full);

// Offload only optimizer states
let fsdp = FSDP::new(model, pg)
    .cpu_offload(CPUOffload::OptimizerStates);
```

## Pipeline Parallelism

Split model across GPUs by layers:

```rust
use axonml::distributed::{Pipeline, PipelineSchedule, PipelineStage};

// Define stages
let stage0 = Sequential::new()
    .add(Linear::new(784, 1024))
    .add(ReLU);

let stage1 = Sequential::new()
    .add(Linear::new(1024, 1024))
    .add(ReLU);

let stage2 = Sequential::new()
    .add(Linear::new(1024, 10));

// Create pipeline
let pipeline = Pipeline::new(vec![
    PipelineStage::new(stage0, Device::CUDA(0)),
    PipelineStage::new(stage1, Device::CUDA(1)),
    PipelineStage::new(stage2, Device::CUDA(2)),
])
.num_microbatches(8)
.schedule(PipelineSchedule::GPipe);

// Forward pass handles micro-batching
let output = pipeline.forward(&input);
```

### Pipeline Schedules

```rust
use axonml::distributed::PipelineSchedule;

// GPipe: All forward, then all backward
let pipeline = pipeline.schedule(PipelineSchedule::GPipe);

// 1F1B: Interleaved forward/backward
let pipeline = pipeline.schedule(PipelineSchedule::Interleaved1F1B);
```

## Communication Primitives

### Collective Operations

```rust
use axonml::distributed::*;

let pg = ProcessGroup::mock();
let mut tensor = Tensor::randn(&[100]);

// All-reduce (sum)
all_reduce_sum(&mut tensor, &pg);

// All-reduce (mean)
all_reduce_mean(&mut tensor, &pg);

// All-gather
let gathered = all_gather(&tensor, &pg);

// Broadcast from rank 0
broadcast(&mut tensor, &pg);

// Reduce-scatter
let scattered = reduce_scatter_sum(&tensor, &pg);

// Barrier (synchronization)
barrier(&pg);
```

### Point-to-Point

```rust
use axonml::distributed::{send, recv};

if rank == 0 {
    send(&tensor, 1, &pg);  // Send to rank 1
} else if rank == 1 {
    let received = recv(0, &pg);  // Receive from rank 0
}
```

## Process Groups

```rust
use axonml::distributed::{World, ProcessGroup};

let world = World::init();

// Default group (all ranks)
let default_pg = world.default_group();

// Create subgroup
let ranks = vec![0, 1];  // Only ranks 0 and 1
let subgroup = world.new_group(ranks);

// Check membership
if subgroup.contains(world.rank()) {
    // Do something for this subgroup
}
```

## Tensor Parallelism

For very large layers:

```rust
use axonml::distributed::{ColumnParallelLinear, RowParallelLinear};

// Column parallel: split output features
let col_linear = ColumnParallelLinear::new(1024, 4096, &pg);

// Row parallel: split input features
let row_linear = RowParallelLinear::new(4096, 1024, &pg);

// Typical usage in transformer MLP:
// x -> ColumnParallel -> GELU -> RowParallel -> output
let h = col_linear.forward(&x);
let h = h.gelu();
let output = row_linear.forward(&h);
```

## Hybrid Parallelism

Combine multiple strategies:

```rust
// 8 GPUs: 2-way tensor parallel × 4-way data parallel
let tp_group = world.new_group(vec![0, 1]);  // GPUs 0-1
let dp_group = world.new_group(vec![0, 2, 4, 6]);  // One from each TP group

// Apply tensor parallelism to large layers
let attn = TensorParallelAttention::new(hidden_size, num_heads, &tp_group);

// Wrap entire model in DDP for data parallelism
let ddp_model = DDP::new(model, dp_group);
```

## Memory Stats

```rust
use axonml::distributed::FSDPMemoryStats;

let stats = fsdp_model.memory_stats();
println!("Peak memory: {} MB", stats.peak_memory_mb);
println!("Current allocation: {} MB", stats.current_allocation_mb);
```

## Best Practices

1. **Start with DDP** - Simplest, works well for most cases
2. **Use FSDP for large models** - When model doesn't fit on single GPU
3. **Pipeline for very deep models** - Transformers with 100+ layers
4. **Match batch size to world size** - Effective batch = local_batch × world_size
5. **Gradient accumulation** - For very large effective batch sizes
6. **Mixed precision** - Combine with AMP for memory savings
