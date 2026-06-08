# axonml-distributed

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.6.5-green.svg" alt="Version 0.6.5">
  <img src="https://img.shields.io/badge/part%20of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

`axonml-distributed` provides distributed training primitives for AxonML:
data parallelism (`DDP`), fully sharded data parallelism (`FSDP`, ZeRO-2 /
ZeRO-3 with HybridShard and CPU offload), pipeline parallelism (`Pipeline`
with GPipe / 1F1B / Interleaved 1F1B schedules), tensor parallelism
(`ColumnParallelLinear`, `RowParallelLinear`), collective ops (all-reduce,
broadcast, gather/scatter, reduce-scatter, barrier), a pluggable `Backend`
trait, a deterministic `MockBackend` for tests, and an optional NCCL backend
behind the `nccl` feature.

## Features

- **Backend Abstraction** — `Backend` trait with `MockBackend` (in-process shared-state simulation) and optional `NcclBackend` (dynamic `libcudart` / `libnccl` loading via `libloading`)
- **Process Groups** — `ProcessGroup` / `World` abstractions with rank, world size, subgroups, default group
- **DistributedDataParallel (DDP)** — model wrapper with gradient bucketing, sync strategies (`Synchronous`, `Overlapped`, `NoSync`), parameter broadcast, buffer sync toggle, `DDP<M>` type alias
- **FullyShardedDataParallel (FSDP)** — parameter sharding with `FullShard` (ZeRO-3), `ShardGradOp` (ZeRO-2), `NoShard`, `HybridShard`; optional `CPUOffload::{None, Params, Full}`; mixed precision toggle; `gather_parameters` / `reshard_parameters`; `clip_grad_norm`; `FSDPMemoryStats` diagnostics; `FSDP<M>` type alias
- **Pipeline Parallelism** — `Pipeline` with `GPipe`, `OneFOneBSchedule` (default), `InterleavedOneFOneB`; `PipelineStage`, `PipelineMemoryStats` with `gpipe_peak_activations` and `one_f_one_b_peak_activations`
- **Tensor Parallelism** — `ColumnParallelLinear`, `RowParallelLinear`
- **Collective Operations** — `all_reduce_{sum,mean,min,max,product}`, `broadcast`, `broadcast_from`, `all_gather`, `reduce_scatter_sum`, `reduce_scatter_mean`, `gather_tensor`, `scatter_tensor`, `barrier`, `sync_gradient`, `sync_gradients`, `rank`, `world_size`, `is_main_process`
- **Reduce Operations** — `ReduceOp::{Sum, Product, Min, Max, Average}`
- **Gradient Bucketing** — `GradientBucket`, `GradientSynchronizer`, `GradSyncStrategy`

## Modules

| Module | Description |
|--------|-------------|
| `backend` | `Backend` trait, `MockBackend`, `ReduceOp` |
| `process_group` | `ProcessGroup`, `World` (with `new_group` subgroups, `default_group`, `mock` constructor) |
| `comm` | Collective ops (`all_reduce_*`, `broadcast*`, `all_gather`, `reduce_scatter_*`, `gather_tensor`, `scatter_tensor`, `barrier`, `sync_gradient(s)`, query helpers) |
| `ddp` | `DistributedDataParallel`, `GradientBucket`, `GradientSynchronizer`, `GradSyncStrategy` |
| `fsdp` | `FullyShardedDataParallel`, `ShardingStrategy`, `CPUOffload`, `FSDPMemoryStats`, `ColumnParallelLinear`, `RowParallelLinear` |
| `pipeline` | `Pipeline`, `PipelineStage`, `PipelineSchedule`, `PipelineMemoryStats` |
| `nccl_backend` (feature: `nccl`) | `NcclBackend`, `NcclUniqueId`, `NcclError`, version/device query, multi-node init |

## Features Flags

| Flag | Effect |
|------|--------|
| `nccl` | Enables the `NcclBackend` module and pulls in `libloading` for runtime NCCL discovery |

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-distributed = "0.6.5"

# Or with NCCL support:
axonml-distributed = { version = "0.6.5", features = ["nccl"] }
```

### Basic DDP Training

```rust
use axonml_distributed::prelude::*;
use axonml_nn::Linear;

// Initialize distributed world
let world = World::mock();  // Use mock for testing

// Create model and wrap in DDP
let model = Linear::new(10, 5);
let mut ddp = DistributedDataParallel::new(model, world.default_group().clone());

// Synchronize parameters from rank 0 at start of training
ddp.sync_parameters();

// Training loop
ddp.train();
for batch in data_loader.iter() {
    let output = ddp.forward(&input);
    // ... compute loss and backward ...

    // Synchronize gradients across all processes
    ddp.sync_gradients();

    // ... optimizer step ...
}
```

### DDP Builder

```rust
use axonml_distributed::prelude::*;

let ddp = DDP::new(model, pg)
    .broadcast_buffers(false)
    .gradient_as_bucket_view(false);
```

### FSDP (ZeRO-3 / ZeRO-2 / Hybrid)

```rust
use axonml_distributed::prelude::*;

let fsdp = FullyShardedDataParallel::new(model, world.default_group().clone())
    .sharding_strategy(ShardingStrategy::FullShard)   // ZeRO-3
    .cpu_offload(CPUOffload::Params)
    .mixed_precision(true);

// Gather full parameters for a forward pass, then reshard
fsdp.gather_parameters();
// ... forward / backward ...
fsdp.reshard_parameters();
fsdp.sync_gradients();

// Gradient clipping
let grad_norm = fsdp.clip_grad_norm(1.0);

// Memory accounting
let stats = fsdp.memory_estimate();
println!("FSDP total: {:.1} MB (savings {:.1}%)",
         stats.total_memory_mb(), stats.memory_savings() * 100.0);
```

### Pipeline Parallelism

```rust
use axonml_distributed::prelude::*;

let pipe = Pipeline::from_modules(stage_modules, world.default_group().clone())
    .schedule(PipelineSchedule::OneFOneBSchedule)
    .num_microbatches(8);

let output = pipe.forward(&input);

// Memory accounting
let peak = PipelineMemoryStats::one_f_one_b_peak_activations(pipe.num_stages(), 8);
```

### Tensor Parallelism

```rust
use axonml_distributed::prelude::*;

// Shard along output dimension
let col = ColumnParallelLinear::new(/* ... */);

// Shard along input dimension
let row = RowParallelLinear::new(/* ... */);
```

### Communication Primitives

```rust
use axonml_distributed::prelude::*;

let pg = ProcessGroup::mock();

// All-reduce operations
let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
all_reduce_sum(&mut tensor, &pg);
all_reduce_mean(&mut tensor, &pg);
all_reduce_max(&mut tensor, &pg);
all_reduce_min(&mut tensor, &pg);
all_reduce_product(&mut tensor, &pg);

// Broadcast from rank 0 (or from any source rank)
broadcast(&mut tensor, &pg);
broadcast_from(&mut tensor, &pg, /* src_rank */ 0);

// All-gather / reduce-scatter / gather / scatter
let gathered = all_gather(&tensor, &pg);
let scattered_sum   = reduce_scatter_sum(&tensor, &pg);
let scattered_mean  = reduce_scatter_mean(&tensor, &pg);
let _ = gather_tensor(&tensor, &pg, 0);
let _ = scatter_tensor(&tensor, &pg, 0);

// Barrier synchronization
barrier(&pg);

// Query process information
let my_rank         = rank(&pg);
let total_processes = world_size(&pg);
let is_main         = is_main_process(&pg);
```

### Gradient Synchronization

```rust
use axonml_distributed::prelude::*;

// Synchronize multiple gradients
let mut gradients = vec![
    Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap(),
    Tensor::from_vec(vec![0.3, 0.4, 0.5], &[3]).unwrap(),
];
sync_gradients(&mut gradients, &pg);

// Or synchronize a single gradient
let mut grad = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
sync_gradient(&mut grad, &pg);
```

### Gradient Bucketing

```rust
use axonml_distributed::prelude::*;

// Create gradient bucket for efficient all-reduce
let mut bucket = GradientBucket::new(1000);  // 1000 element capacity

let grad1 = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
let grad2 = Tensor::from_vec(vec![0.3, 0.4, 0.5], &[3]).unwrap();

bucket.add(&grad1);
bucket.add(&grad2);

// Extract synchronized gradients
let synced_grads = bucket.extract();
```

### Custom Synchronization Strategy

```rust
use axonml_distributed::prelude::*;

let mut sync = GradientSynchronizer::new(
    GradSyncStrategy::Synchronous,  // or Overlapped, NoSync
    25_000_000                      // ~100MB bucket size for f32
);

sync.prepare(10);  // 10 parameters

// Add gradients during backward pass
let grad = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
sync.add_gradient(0, &grad);

// Synchronize all buckets
sync.sync_all(&pg);
sync.clear();
```

### Multi-Backend Setup

```rust
use axonml_distributed::prelude::*;
use std::sync::Arc;

// Create multiple mock backends (simulates multi-process)
let backends = MockBackend::create_world(4);

// Each process creates its ProcessGroup
for backend in backends {
    let pg = ProcessGroup::new(Arc::new(backend));
    println!("Rank {} of {}", pg.rank(), pg.world_size());
}
```

### Process Subgroups

```rust
use axonml_distributed::prelude::*;

let world = World::mock();

// Create a subgroup with specific ranks
let subgroup = world.new_group(vec![0, 1]);
assert!(subgroup.contains(0));
assert_eq!(subgroup.size(), 2);
```

### NCCL Backend (feature-gated)

```rust
#[cfg(feature = "nccl")]
use axonml_distributed::{NcclBackend, NcclUniqueId};

# #[cfg(feature = "nccl")]
# fn example() -> Result<(), axonml_distributed::NcclError> {
// Multi-node: rank 0 generates the unique id and broadcasts it out-of-band.
let unique_id = NcclBackend::generate_unique_id()?;
let backend   = NcclBackend::new(unique_id, /* rank */ 0, /* world_size */ 2, /* device */ 0)?;

// Or spin up a single-node world over multiple local GPUs:
let backends = NcclBackend::create_world(&[0, 1])?;

let (major, minor, patch) = backend.nccl_version()?;
backend.synchronize()?;
# Ok(()) }
```

## Tests

```bash
cargo test -p axonml-distributed
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
