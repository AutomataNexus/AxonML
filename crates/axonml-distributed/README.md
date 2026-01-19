# axonml-distributed

[![Crates.io](https://img.shields.io/crates/v/axonml-distributed.svg)](https://crates.io/crates/axonml-distributed)
[![Docs.rs](https://docs.rs/axonml-distributed/badge.svg)](https://docs.rs/axonml-distributed)
[![Downloads](https://img.shields.io/crates/d/axonml-distributed.svg)](https://crates.io/crates/axonml-distributed)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Distributed training utilities for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-distributed` provides multi-GPU and multi-node training capabilities including data parallelism, communication primitives, and process group management.

## Features

### Data Parallelism
- **DistributedDataParallel (DDP)** - Synchronous data parallel training
- **Gradient averaging** - Automatic gradient synchronization
- **Bucket gradient reduction** - Optimized communication

### Communication Primitives
- **All-reduce** - Sum/average across all processes
- **Broadcast** - Send from one to all
- **All-gather** - Gather from all to all
- **Reduce-scatter** - Reduce and scatter
- **Barrier** - Synchronization point

### Process Management
- **ProcessGroup** - Group of distributed processes
- **Rank/World size** - Process identification
- **Backend abstraction** - NCCL, Gloo, MPI

## Installation

```toml
[dependencies]
axonml-distributed = "0.1"
```

## Usage

### Basic DDP Training

```rust
use axonml_distributed::{init_process_group, DistributedDataParallel};

// Initialize distributed environment
init_process_group("nccl", "env://")?;  // Backend, init method

let rank = get_rank();
let world_size = get_world_size();
println!("Process {}/{}", rank, world_size);

// Wrap model in DDP
let model = create_model().to_device(rank);
let ddp_model = DistributedDataParallel::new(model);

// Training loop (gradients auto-synchronized)
for batch in dataloader.iter() {
    let output = ddp_model.forward(&batch.data);
    let loss = compute_loss(&output, &batch.targets);

    optimizer.zero_grad();
    loss.backward();  // Gradients synchronized here
    optimizer.step();
}
```

### Communication Primitives

```rust
use axonml_distributed::{all_reduce, broadcast, barrier, ReduceOp};

// All-reduce: sum tensor across all processes
let mut tensor = randn(&[100, 100]);
all_reduce(&mut tensor, ReduceOp::Sum);  // tensor = sum of all

// Broadcast: send from rank 0 to all
let mut tensor = if rank == 0 { randn(&[100]) } else { zeros(&[100]) };
broadcast(&mut tensor, 0);  // All ranks now have rank 0's tensor

// Barrier: wait for all processes
barrier();
```

### Distributed DataLoader

```rust
use axonml_distributed::DistributedSampler;
use axonml_data::DataLoader;

// Each process gets different subset of data
let sampler = DistributedSampler::new(dataset.len(), world_size, rank)
    .shuffle(true);

let dataloader = DataLoader::new(dataset, batch_size)
    .sampler(sampler);

for epoch in 0..num_epochs {
    sampler.set_epoch(epoch);  // Different shuffle each epoch
    for batch in dataloader.iter() {
        // Each rank processes different batch
    }
}
```

### Multi-GPU Launch

```bash
# Launch 4 GPU training
torchrun --nproc_per_node=4 train.rs

# Or with environment variables
WORLD_SIZE=4 RANK=0 LOCAL_RANK=0 ./train
WORLD_SIZE=4 RANK=1 LOCAL_RANK=1 ./train
# ...
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `init_process_group(backend, init_method)` | Initialize distributed |
| `get_rank()` | Current process rank |
| `get_world_size()` | Total number of processes |
| `get_local_rank()` | Rank on current node |
| `is_initialized()` | Check if initialized |
| `destroy_process_group()` | Cleanup |

### Communication

| Function | Description |
|----------|-------------|
| `all_reduce(tensor, op)` | Reduce across all |
| `broadcast(tensor, src)` | Broadcast from src |
| `all_gather(tensors, tensor)` | Gather to all |
| `reduce(tensor, dst, op)` | Reduce to dst |
| `scatter(tensor, tensors, src)` | Scatter from src |
| `barrier()` | Synchronize all |

### ReduceOp

| Op | Description |
|----|-------------|
| `Sum` | Element-wise sum |
| `Product` | Element-wise product |
| `Min` | Element-wise minimum |
| `Max` | Element-wise maximum |
| `Average` | Sum / world_size |

## Part of Axonml

```toml
[dependencies]
axonml = { version = "0.1", features = ["distributed"] }
```

## License

MIT OR Apache-2.0
