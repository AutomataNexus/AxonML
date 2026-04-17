# axonml-distributed Documentation

> Distributed training for the AxonML ML framework.

## Overview

`axonml-distributed` implements data, pipeline, and tensor parallelism for
multi-GPU / multi-node training: `DistributedDataParallel` (DDP) with
gradient bucketing, `FullyShardedDataParallel` (FSDP — ZeRO-2 / ZeRO-3 +
HybridShard + CPU offload), `Pipeline` (GPipe / 1F1B / interleaved
microbatch scheduling), tensor-parallel linear layers (`ColumnParallelLinear`,
`RowParallelLinear`), a full set of collective operations, and `ProcessGroup`
/ `World` abstractions. Backends: `NcclBackend` (real multi-GPU / multi-node,
dynamic `libcudart` + `libnccl` loading) and `MockBackend` (shared-state
in-process simulation for deterministic testing).

## Core Concepts

### World and process groups

```rust
use axonml_distributed::{World, ProcessGroup};

let world = World::new()?;             // real world (from env)
// let world = World::mock();          // in-process mock for tests

println!("rank {} / world_size {}", world.rank(), world.world_size());

let pg_all = world.default_group();
let pg_even = world.new_group(vec![0, 2, 4, 6]);
```

### Data parallelism (DDP)

1. Each rank holds a full copy of the model.
2. Data is sharded across ranks.
3. Gradients are bucketed and all-reduced during backward.
4. All ranks apply the same averaged-gradient update.

## Modules

### `ddp` — `DistributedDataParallel` (alias `DDP<M>`)

```rust
use axonml_distributed::{DDP, GradSyncStrategy};

let ddp = DDP::new(model, world.default_group().clone())
    .broadcast_buffers(false)
    .gradient_as_bucket_view(false);

for batch in train_loader.iter() {
    let output = ddp.forward(&batch.data);
    let loss = compute_loss(&output, &batch.targets);
    loss.backward();           // gradients synced via buckets
    optimizer.step();
    optimizer.zero_grad();
}
```

Related types: `GradientBucket`, `GradientSynchronizer`, `GradSyncStrategy`.

### `fsdp` — `FullyShardedDataParallel` (alias `FSDP<M>`)

Shards parameters + optimizer state across ranks (ZeRO-2 / ZeRO-3 /
HybridShard) with optional CPU offload.

```rust
use axonml_distributed::{FSDP, ShardingStrategy, CPUOffload};

let fsdp = FSDP::new(model, world.default_group().clone())
    .sharding_strategy(ShardingStrategy::FullShard)
    .cpu_offload(CPUOffload::params_and_grads());

let mem = fsdp.memory_stats(); // FSDPMemoryStats
```

Also exports `ColumnParallelLinear` and `RowParallelLinear` for tensor
parallelism inside an FSDP-style setup.

### `pipeline` — Pipeline parallelism

`Pipeline`, `PipelineStage`, `PipelineSchedule` (GPipe / 1F1B /
Interleaved), `PipelineMemoryStats`. Splits a model across devices and
drives microbatch execution.

### `comm`

Collective ops. All take `&ProcessGroup` (or `&mut Tensor`).

| Op                                                           | Purpose                                           |
|--------------------------------------------------------------|---------------------------------------------------|
| `all_reduce_sum`, `all_reduce_mean`, `all_reduce_min`, `all_reduce_max`, `all_reduce_product` | Collective reductions          |
| `broadcast`, `broadcast_from`                                | One -> all                                        |
| `all_gather`                                                 | All ranks collect every shard                     |
| `gather_tensor`, `scatter_tensor`                            | Many-to-one / one-to-many                         |
| `reduce_scatter_sum`, `reduce_scatter_mean`                  | Reduce + scatter                                  |
| `barrier`                                                    | Synchronization point                             |
| `sync_gradient`, `sync_gradients`                            | DDP-style gradient sync                           |
| `rank`, `world_size`, `is_main_process`                      | Introspection                                     |

### `backend`

`Backend` trait + `ReduceOp` enum (`Sum`, `Product`, `Min`, `Max`,
`Average`). `MockBackend::create_world(n)` returns `n` linked backends for
tests.

### `nccl_backend` *(feature = `nccl`)*

`NcclBackend` loads `libcudart` + `libnccl` at runtime. Multi-node
initialisation via `NcclUniqueId`. `NcclError` wraps NCCL failure codes.

### `process_group`

`World` — global process group provider (`::new()` from environment,
`::mock()` for tests). `ProcessGroup` — subset of ranks sharing a backend.

## Usage

### Basic DDP training

```rust
use axonml::prelude::*;
use axonml_distributed::prelude::*;

let world = World::new().expect("failed to init distributed");
let rank = world.rank();
let world_size = world.world_size();

let model = create_model();
let mut ddp = DDP::new(model, world.default_group().clone());

let dataset = load_dataset();
// DistributedSampler / similar shard mechanism — use rank + world_size

let mut optimizer = Adam::new(ddp.parameters(), 0.001);

for epoch in 0..epochs {
    for batch in loader.iter() {
        let output = ddp.forward(&batch.data);
        let loss = compute_loss(&output, &batch.targets);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    let mut local = compute_epoch_loss();
    all_reduce_mean(&mut local, &world.default_group());
    if rank == 0 {
        println!("epoch {}: loss = {:.4}", epoch, local.to_vec()[0]);
    }
}
```

### Multi-node launch

Set the standard environment variables on each node:

```bash
# Node 0 (master)
MASTER_ADDR=192.168.1.1 MASTER_PORT=29500 \
WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 \
cargo run --release --features nccl

# Node 1
MASTER_ADDR=192.168.1.1 MASTER_PORT=29500 \
WORLD_SIZE=8 RANK=4 LOCAL_RANK=0 \
cargo run --release --features nccl
```

### Gradient accumulation

```rust
let accumulation_steps = 4;
for (i, batch) in loader.iter().enumerate() {
    let output = ddp.forward(&batch.data);
    let loss = compute_loss(&output, &batch.targets) / accumulation_steps as f32;
    loss.backward();

    if (i + 1) % accumulation_steps == 0 {
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

### Checkpointing

```rust
if world.rank() == 0 {
    save_model(&ddp.module(), "checkpoint.axonml")?;
}
barrier(&world.default_group());

let model = load_model("checkpoint.axonml")?;
let ddp = DDP::new(model, world.default_group().clone());
```

## Environment Variables

| Variable      | Description                                                 |
|---------------|-------------------------------------------------------------|
| `MASTER_ADDR` | IP address of rank 0                                        |
| `MASTER_PORT` | Port for rendezvous                                         |
| `WORLD_SIZE`  | Total number of processes across all nodes                  |
| `RANK`        | Global rank of this process (0-based)                       |
| `LOCAL_RANK`  | Local rank (per node, for multi-GPU selection)              |

## Feature Flags

- `nccl` — enable the real `NcclBackend` (links `libcudart` + `libnccl` at
  runtime). Without this flag, only `MockBackend` is available.

## Best Practices

1. Same seed on all ranks for reproducibility.
2. Do file IO only from rank 0; `barrier` before loading.
3. Accumulate gradients before syncing for efficiency.
4. Combine with `axonml_autograd::amp` + `axonml_optim::GradScaler` for
   mixed-precision training.

## Related Modules

- [Neural Networks](../nn/README.md) — models to distribute
- [Optimizers](../optim/README.md) — parameter updates
- [Autograd](../autograd/README.md) — AMP + gradient checkpointing

## Last updated

0.6.1 (2026-04-16)
