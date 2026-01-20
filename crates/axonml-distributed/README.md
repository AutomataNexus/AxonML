# axonml-distributed

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part%20of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

`axonml-distributed` provides distributed training utilities for the AxonML machine learning framework. It includes backend abstractions for communication, process group management, DistributedDataParallel (DDP) wrappers for multi-GPU training, and high-level communication primitives like all-reduce, broadcast, and gather operations.

## Features

- **Backend Abstraction** - Pluggable communication backend trait with mock implementation for testing
- **Process Groups** - Manage distributed processes with rank and world size information
- **DistributedDataParallel (DDP)** - Wrap models for automatic gradient synchronization across processes
- **Collective Operations** - All-reduce, broadcast, all-gather, reduce-scatter, and barrier primitives
- **Gradient Bucketing** - Efficient gradient accumulation and synchronization with configurable bucket sizes
- **Multiple Reduce Operations** - Sum, product, min, max, and average reduction strategies
- **Model Parallel Utilities** - Tensor scattering and gathering for model parallelism

## Modules

| Module | Description |
|--------|-------------|
| `backend` | Communication backend trait and MockBackend implementation for testing |
| `process_group` | ProcessGroup and World abstractions for managing distributed processes |
| `ddp` | DistributedDataParallel wrapper, GradientBucket, and GradientSynchronizer |
| `comm` | High-level communication utilities (all_reduce, broadcast, gather, etc.) |

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
axonml-distributed = "0.1.0"
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

### Communication Primitives

```rust
use axonml_distributed::prelude::*;

let pg = ProcessGroup::mock();

// All-reduce operations
let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
all_reduce_sum(&mut tensor, &pg);
all_reduce_mean(&mut tensor, &pg);

// Broadcast from rank 0
broadcast(&mut tensor, &pg);

// All-gather across processes
let gathered = all_gather(&tensor, &pg);

// Barrier synchronization
barrier(&pg);

// Query process information
let my_rank = rank(&pg);
let total_processes = world_size(&pg);
let is_main = is_main_process(&pg);
```

### Gradient Synchronization

```rust
use axonml_distributed::prelude::*;

let pg = ProcessGroup::mock();

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

// All-reduce the flattened bucket data
let pg = ProcessGroup::mock();
pg.backend().all_reduce(bucket.data_mut(), ReduceOp::Average);

// Extract synchronized gradients
let synced_grads = bucket.extract();
```

### Custom Synchronization Strategy

```rust
use axonml_distributed::prelude::*;

let mut sync = GradientSynchronizer::new(
    GradSyncStrategy::Synchronous,
    25_000_000  // ~100MB bucket size for f32
);

sync.prepare(10);  // 10 parameters

// Add gradients during backward pass
let grad = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
sync.add_gradient(0, &grad);

// Synchronize all buckets
let pg = ProcessGroup::mock();
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

## Tests

Run the test suite:

```bash
cargo test -p axonml-distributed
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
