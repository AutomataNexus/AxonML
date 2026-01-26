//! `DataLoader` - Batched Data Iteration
//!
//! Provides efficient batched iteration over datasets with optional
//! shuffling and parallel data loading.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use crate::collate::{stack_tensors, Collate};
use crate::dataset::Dataset;
use crate::sampler::{RandomSampler, Sampler, SequentialSampler};
use axonml_tensor::Tensor;
use rayon::prelude::*;
use std::marker::PhantomData;

// =============================================================================
// Batch Type
// =============================================================================

/// A batch of data from the `DataLoader`.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Batched input data.
    pub data: Tensor<f32>,
    /// Batched targets.
    pub targets: Tensor<f32>,
    /// Number of samples in this batch.
    pub size: usize,
}

impl Batch {
    /// Creates a new Batch.
    #[must_use] pub fn new(data: Tensor<f32>, targets: Tensor<f32>) -> Self {
        let size = data.shape()[0];
        Self {
            data,
            targets,
            size,
        }
    }

    /// Returns the batch size.
    #[must_use] pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the batch is empty.
    #[must_use] pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

// =============================================================================
// DataLoader
// =============================================================================

/// `DataLoader` for batched iteration over datasets.
///
/// Provides configurable batching, shuffling, and iteration over datasets.
pub struct DataLoader<D>
where
    D: Dataset<Item = (Tensor<f32>, Tensor<f32>)>,
{
    /// The underlying dataset.
    dataset: D,
    /// Batch size.
    batch_size: usize,
    /// Whether to shuffle data each epoch.
    shuffle: bool,
    /// Whether to drop the last incomplete batch.
    drop_last: bool,
    /// Number of worker threads (for future parallel loading).
    num_workers: usize,
}

impl<D> DataLoader<D>
where
    D: Dataset<Item = (Tensor<f32>, Tensor<f32>)>,
{
    /// Creates a new `DataLoader` with the specified batch size.
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            num_workers: 0,
        }
    }

    /// Enables or disables shuffling.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets whether to drop the last incomplete batch.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Sets the number of worker threads for parallel data loading.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the number of batches.
    pub fn len(&self) -> usize {
        let total = self.dataset.len();
        if self.drop_last {
            total / self.batch_size
        } else {
            total.div_ceil(self.batch_size)
        }
    }

    /// Returns true if the `DataLoader` is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Returns the dataset length.
    pub fn dataset_len(&self) -> usize {
        self.dataset.len()
    }

    /// Creates an iterator over batches.
    pub fn iter(&self) -> DataLoaderIter<'_, D> {
        let indices: Vec<usize> = if self.shuffle {
            let sampler = RandomSampler::new(self.dataset.len());
            sampler.iter().collect()
        } else {
            let sampler = SequentialSampler::new(self.dataset.len());
            sampler.iter().collect()
        };

        DataLoaderIter {
            dataset: &self.dataset,
            indices,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            position: 0,
            num_workers: self.num_workers,
        }
    }
}

// =============================================================================
// DataLoaderIter
// =============================================================================

/// Iterator over batches from a `DataLoader`.
pub struct DataLoaderIter<'a, D>
where
    D: Dataset<Item = (Tensor<f32>, Tensor<f32>)>,
{
    dataset: &'a D,
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    position: usize,
    num_workers: usize,
}

impl<D> Iterator for DataLoaderIter<'_, D>
where
    D: Dataset<Item = (Tensor<f32>, Tensor<f32>)>,
{
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];

        // Check if this is an incomplete batch
        if batch_indices.len() < self.batch_size && self.drop_last {
            return None;
        }

        // Collect samples for this batch (parallel when num_workers > 0)
        let samples: Vec<(Tensor<f32>, Tensor<f32>)> = if self.num_workers > 0 {
            // Parallel sample collection using rayon
            batch_indices
                .par_iter()
                .filter_map(|&idx| self.dataset.get(idx))
                .collect()
        } else {
            // Sequential fallback for num_workers = 0
            batch_indices
                .iter()
                .filter_map(|&idx| self.dataset.get(idx))
                .collect()
        };

        if samples.is_empty() {
            return None;
        }

        // Separate data and targets for stacking
        let data_samples: Vec<Tensor<f32>> = samples.iter().map(|(x, _)| x.clone()).collect();
        let target_samples: Vec<Tensor<f32>> = samples.iter().map(|(_, y)| y.clone()).collect();

        // Stack samples into batches
        let data = stack_tensors(&data_samples);
        let targets = stack_tensors(&target_samples);

        self.position = end;

        Some(Batch::new(data, targets))
    }
}

impl<D> DataLoaderIter<'_, D>
where
    D: Dataset<Item = (Tensor<f32>, Tensor<f32>)>,
{
    /// Returns the number of remaining batches.
    #[must_use] pub fn remaining(&self) -> usize {
        let remaining_samples = self.indices.len().saturating_sub(self.position);
        if self.drop_last {
            remaining_samples / self.batch_size
        } else {
            remaining_samples.div_ceil(self.batch_size)
        }
    }
}

// =============================================================================
// GenericDataLoader
// =============================================================================

/// A more flexible `DataLoader` that works with any Dataset and Collate function.
pub struct GenericDataLoader<D, C, T>
where
    D: Dataset<Item = T>,
    C: Collate<T>,
    T: Send,
{
    dataset: D,
    collate_fn: C,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,
    _phantom: PhantomData<T>,
}

impl<D, C, T> GenericDataLoader<D, C, T>
where
    D: Dataset<Item = T>,
    C: Collate<T>,
    T: Send,
{
    /// Creates a new `GenericDataLoader`.
    pub fn new(dataset: D, collate_fn: C, batch_size: usize) -> Self {
        Self {
            dataset,
            collate_fn,
            batch_size,
            shuffle: false,
            drop_last: false,
            num_workers: 0,
            _phantom: PhantomData,
        }
    }

    /// Sets the number of worker threads for parallel data loading.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Enables or disables shuffling.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets whether to drop the last incomplete batch.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Returns the number of batches.
    pub fn len(&self) -> usize {
        let total = self.dataset.len();
        if self.drop_last {
            total / self.batch_size
        } else {
            total.div_ceil(self.batch_size)
        }
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Creates an iterator over batches.
    pub fn iter(&self) -> GenericDataLoaderIter<'_, D, C, T> {
        let indices: Vec<usize> = if self.shuffle {
            let sampler = RandomSampler::new(self.dataset.len());
            sampler.iter().collect()
        } else {
            (0..self.dataset.len()).collect()
        };

        GenericDataLoaderIter {
            dataset: &self.dataset,
            collate_fn: &self.collate_fn,
            indices,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            position: 0,
            num_workers: self.num_workers,
            _phantom: PhantomData,
        }
    }
}

/// Iterator for `GenericDataLoader`.
pub struct GenericDataLoaderIter<'a, D, C, T>
where
    D: Dataset<Item = T>,
    C: Collate<T>,
    T: Send,
{
    dataset: &'a D,
    collate_fn: &'a C,
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    position: usize,
    num_workers: usize,
    _phantom: PhantomData<T>,
}

impl<D, C, T> Iterator for GenericDataLoaderIter<'_, D, C, T>
where
    D: Dataset<Item = T>,
    C: Collate<T>,
    T: Send + Sync,
{
    type Item = C::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];

        if batch_indices.len() < self.batch_size && self.drop_last {
            return None;
        }

        // Collect samples (parallel when num_workers > 0)
        let samples: Vec<T> = if self.num_workers > 0 {
            batch_indices
                .par_iter()
                .filter_map(|&idx| self.dataset.get(idx))
                .collect()
        } else {
            batch_indices
                .iter()
                .filter_map(|&idx| self.dataset.get(idx))
                .collect()
        };

        if samples.is_empty() {
            return None;
        }

        self.position = end;

        Some(self.collate_fn.collate(samples))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collate::DefaultCollate;
    use crate::dataset::TensorDataset;

    fn create_test_dataset(size: usize) -> TensorDataset {
        let data: Vec<f32> = (0..size * 2).map(|i| i as f32).collect();
        let targets: Vec<f32> = (0..size).map(|i| (i % 2) as f32).collect();

        let x = Tensor::from_vec(data, &[size, 2]).unwrap();
        let y = Tensor::from_vec(targets, &[size]).unwrap();

        TensorDataset::new(x, y)
    }

    #[test]
    fn test_dataloader_basic() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset, 3);

        assert_eq!(loader.batch_size(), 3);
        assert_eq!(loader.len(), 4); // ceil(10/3) = 4

        let batches: Vec<Batch> = loader.iter().collect();
        assert_eq!(batches.len(), 4);

        // First 3 batches have size 3, last has size 1
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn test_dataloader_drop_last() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset, 3).drop_last(true);

        assert_eq!(loader.len(), 3); // floor(10/3) = 3

        let batches: Vec<Batch> = loader.iter().collect();
        assert_eq!(batches.len(), 3);

        // All batches have full size
        for batch in &batches {
            assert_eq!(batch.len(), 3);
        }
    }

    #[test]
    fn test_dataloader_shuffle() {
        let dataset = create_test_dataset(100);
        let loader = DataLoader::new(dataset, 10).shuffle(true);

        // Run multiple iterations and collect first batch data
        let batch1: Vec<Batch> = loader.iter().take(1).collect();
        let batch2: Vec<Batch> = loader.iter().take(1).collect();

        // Due to shuffling, batches should (usually) be different
        // We can't guarantee this, but the loader should work
        assert!(!batch1.is_empty());
        assert!(!batch2.is_empty());
    }

    #[test]
    fn test_dataloader_exact_batches() {
        let dataset = create_test_dataset(9);
        let loader = DataLoader::new(dataset, 3);

        let batches: Vec<Batch> = loader.iter().collect();
        assert_eq!(batches.len(), 3);

        for batch in &batches {
            assert_eq!(batch.len(), 3);
        }
    }

    #[test]
    fn test_batch_struct() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let targets = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let batch = Batch::new(data, targets);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_dataloader_empty() {
        let x = Tensor::from_vec(vec![], &[0, 2]).unwrap();
        let y = Tensor::from_vec(vec![], &[0]).unwrap();
        let dataset = TensorDataset::new(x, y);
        let loader = DataLoader::new(dataset, 3);

        assert!(loader.is_empty());
        let batches: Vec<Batch> = loader.iter().collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_dataloader_single_item() {
        let dataset = create_test_dataset(1);
        let loader = DataLoader::new(dataset, 3);

        let batches: Vec<Batch> = loader.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_dataloader_iteration_order() {
        let dataset = create_test_dataset(6);
        let loader = DataLoader::new(dataset, 2).shuffle(false);

        let batches: Vec<Batch> = loader.iter().collect();

        // Without shuffle, data should be in order
        assert_eq!(batches[0].data.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(batches[1].data.to_vec(), vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(batches[2].data.to_vec(), vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_generic_dataloader() {
        let dataset = create_test_dataset(6);
        let collate = DefaultCollate::new();
        let loader = GenericDataLoader::new(dataset, collate, 2);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_dataloader_remaining() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset, 3);

        let mut iter = loader.iter();
        assert_eq!(iter.remaining(), 4);

        iter.next();
        assert_eq!(iter.remaining(), 3);

        iter.next();
        assert_eq!(iter.remaining(), 2);
    }

    #[test]
    fn test_parallel_dataloader() {
        let dataset = create_test_dataset(100);
        let loader = DataLoader::new(dataset, 10).num_workers(4);

        let batches: Vec<Batch> = loader.iter().collect();
        assert_eq!(batches.len(), 10);

        // Verify all samples are present
        let total_samples: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total_samples, 100);
    }

    #[test]
    fn test_parallel_vs_sequential_equivalence() {
        // Create two identical datasets
        let dataset_seq = create_test_dataset(50);
        let dataset_par = create_test_dataset(50);

        // Sequential
        let loader_seq = DataLoader::new(dataset_seq, 5).num_workers(0);
        let batches_seq: Vec<Batch> = loader_seq.iter().collect();

        // Parallel
        let loader_par = DataLoader::new(dataset_par, 5).num_workers(4);
        let batches_par: Vec<Batch> = loader_par.iter().collect();

        // Same number of batches
        assert_eq!(batches_seq.len(), batches_par.len());

        // Same data (no shuffle, so order should be same)
        for i in 0..batches_seq.len() {
            assert_eq!(batches_seq[i].data.to_vec(), batches_par[i].data.to_vec());
            assert_eq!(batches_seq[i].targets.to_vec(), batches_par[i].targets.to_vec());
        }
    }

    #[test]
    fn test_parallel_dataloader_drop_last() {
        let dataset = create_test_dataset(95);
        let loader = DataLoader::new(dataset, 10)
            .drop_last(true)
            .num_workers(4);

        let batches: Vec<Batch> = loader.iter().collect();
        assert_eq!(batches.len(), 9); // 95 / 10 = 9 full batches

        for batch in &batches {
            assert_eq!(batch.len(), 10);
        }
    }

    #[test]
    fn test_parallel_generic_dataloader() {
        let dataset = create_test_dataset(60);
        let collate = DefaultCollate::new();
        let loader = GenericDataLoader::new(dataset, collate, 10).num_workers(4);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 6);
    }
}
