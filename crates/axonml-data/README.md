# axonml-data

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-data"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-data** provides data loading infrastructure for training neural networks in the AxonML framework. It includes the `Dataset` trait, efficient `DataLoader` with batching and shuffling, various sampling strategies, and composable data transforms.

## Features

- **Dataset Trait** - Core abstraction for indexed data access with `TensorDataset`, `MapDataset`, `ConcatDataset`, and `SubsetDataset` implementations
- **DataLoader** - Efficient batched iteration with configurable batch size, shuffling, and drop-last behavior
- **Samplers** - Flexible sampling strategies including `SequentialSampler`, `RandomSampler`, `SubsetRandomSampler`, `WeightedRandomSampler`, and `BatchSampler`
- **Transforms** - Composable data augmentation with `Normalize`, `RandomNoise`, `RandomCrop`, `RandomFlip`, `Scale`, `Clamp`, and more
- **Collate Functions** - Batch assembly with `DefaultCollate` and `StackCollate` for tensor stacking
- **Generic DataLoader** - Flexible loader that works with any `Dataset` and `Collate` combination

## Modules

| Module | Description |
|--------|-------------|
| `dataset` | Core `Dataset` trait and implementations (`TensorDataset`, `MapDataset`, `ConcatDataset`, `SubsetDataset`, `InMemoryDataset`) |
| `dataloader` | `DataLoader` for batched iteration with shuffling support |
| `sampler` | Sampling strategies for controlling data access patterns |
| `transforms` | Composable data transformations for preprocessing and augmentation |
| `collate` | Batch assembly functions for combining samples into tensors |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-data = "0.1.0"
```

### Creating a Dataset

```rust
use axonml_data::prelude::*;

// From tensors
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
let y = Tensor::from_vec(vec![0.0, 1.0, 0.0], &[3]).unwrap();
let dataset = TensorDataset::new(x, y);

assert_eq!(dataset.len(), 3);
let (input, target) = dataset.get(0).unwrap();
```

### Using the DataLoader

```rust
use axonml_data::{DataLoader, TensorDataset};

let dataset = TensorDataset::new(x_data, y_data);

// Create loader with batch size 32
let loader = DataLoader::new(dataset, 32)
    .shuffle(true)
    .drop_last(false);

// Iterate over batches
for batch in loader.iter() {
    let inputs = batch.data;
    let targets = batch.targets;
    // ... process batch ...
}
```

### Implementing Custom Datasets

```rust
use axonml_data::Dataset;
use axonml_tensor::Tensor;

struct MyDataset {
    data: Vec<(Tensor<f32>, Tensor<f32>)>,
}

impl Dataset for MyDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.data.get(index).cloned()
    }
}
```

### Data Transforms

```rust
use axonml_data::{Compose, Normalize, RandomNoise, Scale, Transform};

// Compose multiple transforms
let transform = Compose::empty()
    .add(Normalize::new(0.5, 0.5))
    .add(RandomNoise::new(0.01))
    .add(Scale::new(2.0));

let output = transform.apply(&input_tensor);
```

### Using Samplers

```rust
use axonml_data::{RandomSampler, WeightedRandomSampler, BatchSampler, Sampler};

// Random sampling without replacement
let sampler = RandomSampler::new(1000);
for idx in sampler.iter() {
    // Process sample at idx
}

// Weighted sampling for imbalanced datasets
let weights = vec![1.0, 2.0, 0.5, 3.0];
let sampler = WeightedRandomSampler::new(weights, 100, true);

// Batch sampling
let base_sampler = RandomSampler::new(1000);
let batch_sampler = BatchSampler::new(base_sampler, 32, false);
for batch_indices in batch_sampler.iter() {
    // batch_indices is Vec<usize>
}
```

### Dataset Splitting

```rust
use axonml_data::{TensorDataset, SubsetDataset};

let dataset = TensorDataset::new(x_data, y_data);

// Random split: 80% train, 20% validation
let splits = SubsetDataset::random_split(dataset, &[800, 200]);
let train_dataset = &splits[0];
let val_dataset = &splits[1];
```

### Combining Datasets

```rust
use axonml_data::{TensorDataset, ConcatDataset, MapDataset};

// Concatenate datasets
let combined = ConcatDataset::new(vec![dataset1, dataset2, dataset3]);

// Apply transform to dataset
let mapped = MapDataset::new(dataset, |(x, y)| {
    (x.mul_scalar(2.0), y)
});
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-data
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
