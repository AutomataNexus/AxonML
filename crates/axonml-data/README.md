# axonml-data

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust"></a>
  <a href="https://crates.io/crates/axonml-data"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Version"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

## Overview

**axonml-data** provides data-loading infrastructure for training neural networks in the AxonML framework. It includes the `Dataset` trait, a `DataLoader` with rayon-backed parallel sample collection, a GPU prefetch iterator that overlaps host loading with device compute, sampling strategies, composable data transforms, and collate utilities.

## Features

- **Dataset trait** — `TensorDataset` (caches flat data for O(row_size) access), `MapDataset`, `ConcatDataset`, `SubsetDataset` (with `random_split`), and `InMemoryDataset<T>` for arbitrary cloneable items.
- **DataLoader** — batched iteration with `shuffle`, `drop_last`, and `num_workers` (rayon-parallel sample collection per batch when `num_workers > 0`).
- **GPU prefetch** — `DataLoader::prefetch_to_gpu(device)` returns a `GpuPrefetchIter` that streams batches from a background thread through a bounded channel (2 batches buffered) so CPU loading overlaps with GPU compute.
- **Samplers** — `SequentialSampler`, `RandomSampler` (with/without replacement), `SubsetRandomSampler`, `WeightedRandomSampler` (O(log n) per sample via cumulative-sum binary search, swap-remove without replacement), and `BatchSampler`.
- **Transforms** — `Compose`, `ToTensor`, `Normalize` (scalar, per-channel, ImageNet preset), `RandomNoise` (Box-Muller Gaussian), `RandomCrop` (1D/2D/3D/4D), `RandomFlip` (generic N-d flip along any dim), `Scale`, `Clamp`, `Flatten`, `Reshape`, `DropoutTransform` (train/eval aware), `Lambda`.
- **Collate** — `DefaultCollate` and `StackCollate` (with `with_dim` for stacking along any axis), `GenericDataLoader` for arbitrary `Dataset` + `Collate` pairings, plus `stack_tensors` and `concat_tensors` helpers.

## Modules

| Module | Description |
|--------|-------------|
| `dataset` | `Dataset` trait, `TensorDataset`, `MapDataset`, `ConcatDataset`, `SubsetDataset`, `InMemoryDataset` |
| `dataloader` | `DataLoader`, `DataLoaderIter`, `Batch`, `GpuPrefetchIter`, `GenericDataLoader`, `GenericDataLoaderIter` |
| `sampler` | `Sampler` trait, `SequentialSampler`, `RandomSampler`, `SubsetRandomSampler`, `WeightedRandomSampler`, `BatchSampler` |
| `transforms` | `Transform` trait, `Compose`, `ToTensor`, `Normalize`, `RandomNoise`, `RandomCrop`, `RandomFlip`, `Scale`, `Clamp`, `Flatten`, `Reshape`, `DropoutTransform`, `Lambda` |
| `collate` | `Collate` trait, `DefaultCollate`, `StackCollate`, `stack_tensors`, `concat_tensors` |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
axonml-data = "0.6.1"
```

### Creating a Dataset

```rust
use axonml_data::prelude::*;

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

let loader = DataLoader::new(dataset, 32)
    .shuffle(true)
    .drop_last(false)
    .num_workers(4); // rayon-parallel sample collection per batch

for batch in loader.iter() {
    let inputs = batch.data;
    let targets = batch.targets;
    // ... process batch ...
}
```

### GPU Prefetch

```rust
use axonml_core::Device;
use axonml_data::DataLoader;

let loader = DataLoader::new(dataset, 64).shuffle(true).num_workers(4);

// Background thread produces batches and transfers to GPU;
// bounded to 2 batches in flight.
for batch in loader.prefetch_to_gpu(Device::Cuda(0)) {
    // batch.data and batch.targets are already on the GPU
    let output = model.forward(&batch.data);
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

    fn len(&self) -> usize { self.data.len() }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.data.get(index).cloned()
    }
}
```

### Data Transforms

```rust
use axonml_data::{Compose, Normalize, RandomNoise, Scale, Transform};

let transform = Compose::empty()
    .add(Normalize::imagenet())          // per-channel ImageNet stats
    .add(RandomNoise::new(0.01))
    .add(Scale::new(2.0));

let output = transform.apply(&input_tensor);
```

### Using Samplers

```rust
use axonml_data::{RandomSampler, WeightedRandomSampler, BatchSampler, Sampler};

let sampler = RandomSampler::new(1000);
for idx in sampler.iter() { /* ... */ }

// Weighted sampling for class-imbalanced datasets (O(log n) per sample)
let weights = vec![1.0, 2.0, 0.5, 3.0];
let sampler = WeightedRandomSampler::new(weights, 100, true);

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

// Shuffled random split (requires Dataset: Clone)
let splits = SubsetDataset::random_split(dataset, &[800, 200]);
let train_dataset = &splits[0];
let val_dataset = &splits[1];
```

### Combining Datasets

```rust
use axonml_data::{TensorDataset, ConcatDataset, MapDataset};

let combined = ConcatDataset::new(vec![dataset1, dataset2, dataset3]);

let mapped = MapDataset::new(dataset, |(x, y)| {
    (x.mul_scalar(2.0), y)
});
```

### Generic DataLoader

Flexible loader that works with any `Dataset<Item = T>` and any `Collate<T>`:

```rust
use axonml_data::{GenericDataLoader, DefaultCollate};

let loader = GenericDataLoader::new(dataset, DefaultCollate::new(), 32)
    .shuffle(true)
    .num_workers(4);

for batch in loader.iter() { /* ... */ }
```

## Tests

```bash
cargo test -p axonml-data
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

_Last updated: 2026-04-16 (v0.6.1)_
