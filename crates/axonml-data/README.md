# axonml-data

[![Crates.io](https://img.shields.io/crates/v/axonml-data.svg)](https://crates.io/crates/axonml-data)
[![Docs.rs](https://docs.rs/axonml-data/badge.svg)](https://docs.rs/axonml-data)
[![Downloads](https://img.shields.io/crates/d/axonml-data.svg)](https://crates.io/crates/axonml-data)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Data loading utilities for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-data` provides efficient data loading infrastructure for training neural networks. It includes the `Dataset` trait, `DataLoader` for batching and shuffling, various samplers, and common data transforms. Designed for high-throughput data pipelines with parallel loading.

## Features

### Core Components
- **Dataset trait** - Interface for all datasets
- **DataLoader** - Batching, shuffling, parallel loading
- **Samplers** - Sequential, random, weighted sampling
- **Collate functions** - Custom batch assembly

### Transforms
- **ToTensor** - Convert to tensor format
- **Normalize** - Mean/std normalization
- **Compose** - Chain multiple transforms

### Parallel Loading
- **Multi-threaded prefetch** - Background data loading
- **Pin memory** - Faster GPU transfer
- **Persistent workers** - Reduced overhead

## Installation

```toml
[dependencies]
axonml-data = "0.1"
```

## Usage

### Implementing a Dataset

```rust
use axonml_data::{Dataset, Sample};
use axonml_tensor::Tensor;

struct MyDataset {
    data: Vec<Vec<f32>>,
    labels: Vec<i64>,
}

impl Dataset for MyDataset {
    type Item = Sample<f32, i64>;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Self::Item {
        let data = Tensor::from_vec(self.data[index].clone(), &[784]).unwrap();
        let label = self.labels[index];
        Sample { data, label }
    }
}
```

### Basic DataLoader

```rust
use axonml_data::{DataLoader, Dataset};

let dataset = MyDataset::new();
let dataloader = DataLoader::new(dataset, 32)  // batch_size = 32
    .shuffle(true)
    .drop_last(true);

for batch in dataloader.iter() {
    // batch.data: [32, 784]
    // batch.labels: [32]
    let output = model.forward(&batch.data);
    // ...
}
```

### Parallel Data Loading

```rust
use axonml_data::DataLoader;

let dataloader = DataLoader::new(dataset, 64)
    .shuffle(true)
    .num_workers(4)      // 4 parallel loading threads
    .prefetch_factor(2)  // 2 batches per worker
    .pin_memory(true);   // Pin memory for GPU

for epoch in 0..100 {
    for batch in dataloader.iter() {
        train_step(&batch);
    }
}
```

### Using Transforms

```rust
use axonml_data::{DataLoader, transforms::{Compose, Normalize, ToTensor}};

// Create transform pipeline
let transform = Compose::new()
    .add(ToTensor::new())
    .add(Normalize::new(&[0.5], &[0.5]));  // (x - mean) / std

let dataloader = DataLoader::new(dataset, 32)
    .transform(transform)
    .shuffle(true);
```

### Custom Samplers

```rust
use axonml_data::{DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler};

// Random sampling (default with shuffle=true)
let random_sampler = RandomSampler::new(dataset.len());

// Sequential sampling (default with shuffle=false)
let sequential_sampler = SequentialSampler::new(dataset.len());

// Weighted sampling for imbalanced datasets
let weights = vec![1.0, 2.0, 1.0, 3.0, ...];  // Per-sample weights
let weighted_sampler = WeightedRandomSampler::new(weights, num_samples, true);

let dataloader = DataLoader::new(dataset, 32)
    .sampler(weighted_sampler);
```

### Subset and Split

```rust
use axonml_data::{Dataset, Subset, random_split};

let full_dataset = MyDataset::new();  // 10000 samples

// Create subset with specific indices
let indices: Vec<usize> = (0..1000).collect();
let subset = Subset::new(&full_dataset, indices);

// Random split: 80% train, 20% test
let (train_set, test_set) = random_split(&full_dataset, &[0.8, 0.2]);
println!("Train: {}, Test: {}", train_set.len(), test_set.len());
```

### Concatenating Datasets

```rust
use axonml_data::{Dataset, ConcatDataset};

let dataset1 = MyDataset::new_part1();  // 5000 samples
let dataset2 = MyDataset::new_part2();  // 3000 samples

let combined = ConcatDataset::new(vec![dataset1, dataset2]);
println!("Combined: {} samples", combined.len());  // 8000
```

### Custom Collate Function

```rust
use axonml_data::{DataLoader, Batch};
use axonml_tensor::Tensor;

fn custom_collate(samples: Vec<Sample<f32, i64>>) -> Batch<f32, i64> {
    // Custom batching logic
    let data: Vec<_> = samples.iter().map(|s| s.data.clone()).collect();
    let labels: Vec<_> = samples.iter().map(|s| s.label).collect();

    Batch {
        data: stack_tensors(&data),
        labels: Tensor::from_vec(labels, &[samples.len()]).unwrap(),
    }
}

let dataloader = DataLoader::new(dataset, 32)
    .collate_fn(custom_collate);
```

### Iterating with Indices

```rust
use axonml_data::DataLoader;

let dataloader = DataLoader::new(dataset, 32).shuffle(true);

for (batch_idx, batch) in dataloader.iter().enumerate() {
    if batch_idx % 100 == 0 {
        println!("Batch {}: loss = {}", batch_idx, compute_loss(&batch));
    }
}
```

### Reproducible Shuffling

```rust
use axonml_data::DataLoader;

// Set seed for reproducible shuffling
let dataloader = DataLoader::new(dataset, 32)
    .shuffle(true)
    .seed(42);

// Same order every time with seed=42
```

## API Reference

### Core Traits

| Trait | Description |
|-------|-------------|
| `Dataset` | Interface for datasets with `len()` and `get(idx)` |
| `Sampler` | Interface for sampling strategies |
| `Transform` | Interface for data transformations |
| `CollateFn` | Interface for batch assembly |

### Dataset Types

| Type | Description |
|------|-------------|
| `Subset<D>` | Subset of a dataset by indices |
| `ConcatDataset<D>` | Concatenation of multiple datasets |
| `MapDataset<D, F>` | Apply function to each item |

### DataLoader Methods

| Method | Description |
|--------|-------------|
| `new(dataset, batch_size)` | Create new DataLoader |
| `shuffle(bool)` | Enable/disable shuffling |
| `drop_last(bool)` | Drop incomplete last batch |
| `num_workers(n)` | Parallel loading threads |
| `prefetch_factor(n)` | Batches to prefetch per worker |
| `pin_memory(bool)` | Pin memory for GPU transfer |
| `sampler(s)` | Use custom sampler |
| `collate_fn(f)` | Use custom collate function |
| `seed(s)` | Set random seed |

### Samplers

| Sampler | Description |
|---------|-------------|
| `SequentialSampler` | Sequential indices 0, 1, 2, ... |
| `RandomSampler` | Random permutation |
| `WeightedRandomSampler` | Weighted random sampling |
| `SubsetRandomSampler` | Random from subset |

### Transforms

| Transform | Description |
|-----------|-------------|
| `ToTensor` | Convert to tensor |
| `Normalize` | (x - mean) / std |
| `Compose` | Chain transforms |
| `Lambda` | Custom function |

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

```toml
[dependencies]
axonml = "0.1"  # Includes axonml-data
```

## License

MIT OR Apache-2.0
