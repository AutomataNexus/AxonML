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

`axonml-distributed` provides data, model, pipeline, and tensor parallelism for AxonML:

| Strategy | Type | Description |
|:---------|:-----|:------------|
| **DDP** | `DistributedDataParallel<M>` (alias `DDP<M>`) | Gradient bucketing + all-reduce |
| **FSDP** | `FullyShardedDataParallel<M>` (alias `FSDP<M>`) | ZeRO-2 / ZeRO-3 + HybridShard + CPU offload |
| **Pipeline** | `Pipeline` | GPipe / 1F1B / Interleaved microbatch schedules |
| **Tensor** | `ColumnParallelLinear`, `RowParallelLinear` | Split Linear layers across ranks |

Two collective backends: `MockBackend` (shared-state in-process, for deterministic tests) and `NcclBackend` (dynamic libcudart / libnccl loading, multi-node via `NcclUniqueId`, gated behind the `nccl` feature).

## Data Distributed Parallel (DDP)

```rust
use axonml_distributed::{World, ProcessGroup, DistributedDataParallel, DDP};
use axonml_nn::{Sequential, Linear, ReLU};
use axonml_optim::{Adam, Optimizer};

// Bring up the world
let world = World::init();
let pg: ProcessGroup = world.default_group().clone();

// Build model and wrap in DDP (DDP<M> is an alias for DistributedDataParallel<M>)
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 10));

let mut ddp = DistributedDataParallel::new(model, pg);

// Broadcast initial parameters from rank 0 so every rank starts at the same state
ddp.sync_parameters();

let mut optimizer = Adam::new(ddp.parameters(), 0.001);

// Per-step:
let output = ddp.forward(&inputs);
let loss = loss_fn.compute(&output, &targets);

optimizer.zero_grad();
loss.backward();
ddp.sync_gradients();                              // all-reduce gradients
optimizer.step();
```

### Gradient Bucketing

DDP exposes `GradientBucket`, `GradientSynchronizer`, and `GradSyncStrategy` if you want finer control over bucket size / overlap-with-backward.

### Launching Multi-Rank Jobs

AxonML itself is the Rust process; rank orchestration is up to the launcher (MPI / srun / shell). With the `nccl` feature enabled, `NcclBackend::init_from_unique_id` handles cross-node bootstrap.

## Fully Sharded Data Parallel (FSDP)

FSDP implements ZeRO-style sharding over a `ProcessGroup`.

```rust
use axonml_distributed::{
    FullyShardedDataParallel, FSDP, ShardingStrategy, CPUOffload, FSDPMemoryStats, World,
};

let world = World::init();
let pg = world.default_group().clone();
let model = build_large_model();

let fsdp = FullyShardedDataParallel::new(model, pg)
    .sharding_strategy(ShardingStrategy::FullShard);    // ZeRO-3
```

### Sharding Strategies

```rust
use axonml_distributed::ShardingStrategy;

ShardingStrategy::NoShard;           // Replicate (like DDP)
ShardingStrategy::ShardGradOp;       // Shard optimizer states + gradients (ZeRO-2)
ShardingStrategy::FullShard;         // Shard parameters + gradients + optimizer (ZeRO-3)
ShardingStrategy::HybridShard;       // Shard within node, replicate across nodes
```

### CPU Offloading

```rust
use axonml_distributed::CPUOffload;

let fsdp = FullyShardedDataParallel::new(model, pg).cpu_offload(CPUOffload::Full);
```

### Memory Stats

```rust
use axonml_distributed::FSDPMemoryStats;

let stats: FSDPMemoryStats = fsdp.memory_stats();
```

`FSDPMemoryStats` is also used to estimate peak memory under different sharding strategies before picking one.

## Pipeline Parallelism

`Pipeline` splits a model across stages, each on its own device, and schedules microbatches.

```rust
use axonml_distributed::{Pipeline, PipelineStage, PipelineSchedule, PipelineMemoryStats};
use axonml_core::Device;

// PipelineStage::new(module, stage_id, device_rank)
let stage0 = PipelineStage::new(stage0_module, 0, 0);
let stage1 = PipelineStage::new(stage1_module, 1, 1);
let stage2 = PipelineStage::new(stage2_module, 2, 2);

let pipeline = Pipeline::new(vec![stage0, stage1, stage2]);
// schedule / microbatch count are configured via Pipeline's builder surface —
// see PipelineSchedule::{GPipe, Interleaved1F1B}
```

`PipelineSchedule` variants: GPipe (all-forward-then-all-backward), 1F1B (interleaved), and an interleaved 1F1B scheduler for deep transformers.

## Tensor Parallelism

For large Linear layers, use column / row parallelism — the classic transformer MLP pattern is `ColumnParallelLinear → GELU → RowParallelLinear`:

```rust
use axonml_distributed::{ColumnParallelLinear, RowParallelLinear};

let col = ColumnParallelLinear::new(1024, 4096, &pg);
let row = RowParallelLinear::new(4096, 1024, &pg);
```

## Communication Primitives

The `axonml_distributed::comm` module (re-exported at crate root) exposes the standard collectives and rank helpers:

```rust
use axonml_distributed::{
    all_reduce_sum, all_reduce_mean, all_reduce_min, all_reduce_max, all_reduce_product,
    broadcast, broadcast_from,
    all_gather, gather_tensor, scatter_tensor,
    reduce_scatter_sum, reduce_scatter_mean,
    barrier, rank, world_size, is_main_process,
    sync_gradient, sync_gradients,
    Backend, MockBackend, ReduceOp,
};
```

Send/recv point-to-point is provided by the concrete backend (`Backend::send` / `Backend::recv`) — not as free functions.

## Process Groups

```rust
use axonml_distributed::{World, ProcessGroup};

let world = World::init();
let default_pg = world.default_group();

// Create a sub-group (e.g. tensor-parallel group of ranks 0..2)
let tp_ranks = vec![0, 1];
let tp_group = world.new_group(tp_ranks);
```

## Hybrid Parallelism

Combine strategies by composing groups. A common 3-D parallel recipe for transformers:

```rust
// 8 GPUs: 2-way TP × 2-way PP × 2-way DP
let tp_group = world.new_group(vec![0, 1]);
let pp_group = world.new_group(vec![0, 2]);
let dp_group = world.new_group(vec![0, 4]);

let attn = tensor_parallel_attention(hidden, num_heads, &tp_group);
let ddp = DistributedDataParallel::new(model_stage, dp_group);
```

## Best Practices

1. **Start with DDP** — Simplest; works when the model fits on one GPU.
2. **FSDP when the model doesn't fit** — Start with `ShardGradOp` (ZeRO-2), escalate to `FullShard` (ZeRO-3) if needed.
3. **Pipeline for very deep models** — Transformers with 100+ layers.
4. **Tensor parallel for giant matmuls** — e.g. 4k+ hidden size with few tokens per rank.
5. **Effective batch** — `local_batch × dp_world_size`.
6. **Mix with AMP** — Combine autocast + GradScaler for memory savings.
7. **Test with MockBackend** — Write deterministic collective tests before wiring up NCCL.

---

*Last updated: 2026-04-16 (v0.6.1)*
