//! Dataset Trait - Core Data Abstraction
//!
//! Defines the Dataset trait that all data sources implement.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_tensor::Tensor;

// =============================================================================
// Dataset Trait
// =============================================================================

/// Core trait for all datasets.
///
/// A dataset provides indexed access to data items.
pub trait Dataset: Send + Sync {
    /// The type of items in the dataset.
    type Item: Send;

    /// Returns the number of items in the dataset.
    fn len(&self) -> usize;

    /// Returns true if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets an item by index.
    fn get(&self, index: usize) -> Option<Self::Item>;
}

// =============================================================================
// TensorDataset
// =============================================================================

/// A dataset wrapping tensors.
///
/// Each item is a tuple of (input, target) tensors.
pub struct TensorDataset {
    /// Input data tensor.
    data: Tensor<f32>,
    /// Target tensor.
    targets: Tensor<f32>,
    /// Number of samples.
    len: usize,
}

impl TensorDataset {
    /// Creates a new `TensorDataset` from input and target tensors.
    ///
    /// The first dimension of both tensors must match.
    #[must_use] pub fn new(data: Tensor<f32>, targets: Tensor<f32>) -> Self {
        let len = data.shape()[0];
        assert_eq!(
            len,
            targets.shape()[0],
            "Data and targets must have same first dimension"
        );
        Self { data, targets, len }
    }

    /// Creates a `TensorDataset` from just input data (no targets).
    #[must_use] pub fn from_data(data: Tensor<f32>) -> Self {
        let len = data.shape()[0];
        let targets = Tensor::from_vec(vec![0.0; len], &[len]).unwrap();
        Self { data, targets, len }
    }
}

impl Dataset for TensorDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.len {
            return None;
        }

        // Extract row from data tensor
        let data_shape = self.data.shape();
        let row_size: usize = data_shape[1..].iter().product();
        let data_vec = self.data.to_vec();
        let start = index * row_size;
        let end = start + row_size;
        let item_data = data_vec[start..end].to_vec();
        let item_shape: Vec<usize> = data_shape[1..].to_vec();
        let x = Tensor::from_vec(item_data, &item_shape).unwrap();

        // Extract target
        let target_shape = self.targets.shape();
        let target_row_size: usize = if target_shape.len() > 1 {
            target_shape[1..].iter().product()
        } else {
            1
        };
        let target_vec = self.targets.to_vec();
        let target_start = index * target_row_size;
        let target_end = target_start + target_row_size;
        let item_target = target_vec[target_start..target_end].to_vec();
        let target_item_shape: Vec<usize> = if target_shape.len() > 1 {
            target_shape[1..].to_vec()
        } else {
            vec![1]
        };
        let y = Tensor::from_vec(item_target, &target_item_shape).unwrap();

        Some((x, y))
    }
}

// =============================================================================
// MapDataset
// =============================================================================

/// A dataset that applies a transform to another dataset.
pub struct MapDataset<D, F>
where
    D: Dataset,
    F: Fn(D::Item) -> D::Item + Send + Sync,
{
    dataset: D,
    transform: F,
}

impl<D, F> MapDataset<D, F>
where
    D: Dataset,
    F: Fn(D::Item) -> D::Item + Send + Sync,
{
    /// Creates a new `MapDataset`.
    pub fn new(dataset: D, transform: F) -> Self {
        Self { dataset, transform }
    }
}

impl<D, F> Dataset for MapDataset<D, F>
where
    D: Dataset,
    F: Fn(D::Item) -> D::Item + Send + Sync,
{
    type Item = D::Item;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.dataset.get(index).map(&self.transform)
    }
}

// =============================================================================
// ConcatDataset
// =============================================================================

/// A dataset that concatenates multiple datasets.
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
}

impl<D: Dataset> ConcatDataset<D> {
    /// Creates a new `ConcatDataset` from multiple datasets.
    #[must_use] pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;
        for d in &datasets {
            total += d.len();
            cumulative_sizes.push(total);
        }
        Self {
            datasets,
            cumulative_sizes,
        }
    }

    /// Finds which dataset contains the given index.
    fn find_dataset(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.len() {
            return None;
        }

        for (i, &cum_size) in self.cumulative_sizes.iter().enumerate() {
            if index < cum_size {
                let prev_size = if i == 0 {
                    0
                } else {
                    self.cumulative_sizes[i - 1]
                };
                return Some((i, index - prev_size));
            }
        }
        None
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D> {
    type Item = D::Item;

    fn len(&self) -> usize {
        *self.cumulative_sizes.last().unwrap_or(&0)
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        let (dataset_idx, local_idx) = self.find_dataset(index)?;
        self.datasets[dataset_idx].get(local_idx)
    }
}

// =============================================================================
// SubsetDataset
// =============================================================================

/// A dataset that provides a subset of another dataset.
pub struct SubsetDataset<D: Dataset> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> SubsetDataset<D> {
    /// Creates a new `SubsetDataset` with specified indices.
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }

    /// Creates a random split of a dataset into two subsets.
    pub fn random_split(dataset: D, lengths: &[usize]) -> Vec<Self>
    where
        D: Clone,
    {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let total_len: usize = lengths.iter().sum();
        assert_eq!(
            total_len,
            dataset.len(),
            "Split lengths must sum to dataset length"
        );

        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        indices.shuffle(&mut thread_rng());

        let mut subsets = Vec::with_capacity(lengths.len());
        let mut offset = 0;
        for &len in lengths {
            let subset_indices = indices[offset..offset + len].to_vec();
            subsets.push(Self::new(dataset.clone(), subset_indices));
            offset += len;
        }
        subsets
    }
}

impl<D: Dataset> Dataset for SubsetDataset<D> {
    type Item = D::Item;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        let real_index = *self.indices.get(index)?;
        self.dataset.get(real_index)
    }
}

// =============================================================================
// InMemoryDataset
// =============================================================================

/// A simple in-memory dataset from a vector.
pub struct InMemoryDataset<T: Clone + Send> {
    items: Vec<T>,
}

impl<T: Clone + Send> InMemoryDataset<T> {
    /// Creates a new `InMemoryDataset` from a vector.
    #[must_use] pub fn new(items: Vec<T>) -> Self {
        Self { items }
    }
}

impl<T: Clone + Send + Sync> Dataset for InMemoryDataset<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.items.get(index).cloned()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dataset() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let targets = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
        let dataset = TensorDataset::new(data, targets);

        assert_eq!(dataset.len(), 3);

        let (x, y) = dataset.get(0).unwrap();
        assert_eq!(x.to_vec(), vec![1.0, 2.0]);
        assert_eq!(y.to_vec(), vec![0.0]);

        let (x, y) = dataset.get(2).unwrap();
        assert_eq!(x.to_vec(), vec![5.0, 6.0]);
        assert_eq!(y.to_vec(), vec![2.0]);

        assert!(dataset.get(3).is_none());
    }

    #[test]
    fn test_map_dataset() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4, 1]).unwrap();
        let targets = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4]).unwrap();
        let base = TensorDataset::new(data, targets);

        let mapped = MapDataset::new(base, |(x, y)| (x.mul_scalar(2.0), y));

        assert_eq!(mapped.len(), 4);
        let (x, _) = mapped.get(0).unwrap();
        assert_eq!(x.to_vec(), vec![2.0]);
    }

    #[test]
    fn test_concat_dataset() {
        let data1 = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let targets1 = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let ds1 = TensorDataset::new(data1, targets1);

        let data2 = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3, 1]).unwrap();
        let targets2 = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
        let ds2 = TensorDataset::new(data2, targets2);

        let concat = ConcatDataset::new(vec![ds1, ds2]);

        assert_eq!(concat.len(), 5);

        let (x, y) = concat.get(0).unwrap();
        assert_eq!(x.to_vec(), vec![1.0]);
        assert_eq!(y.to_vec(), vec![0.0]);

        let (x, y) = concat.get(3).unwrap();
        assert_eq!(x.to_vec(), vec![4.0]);
        assert_eq!(y.to_vec(), vec![3.0]);
    }

    #[test]
    fn test_subset_dataset() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap();
        let targets = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let base = TensorDataset::new(data, targets);

        let subset = SubsetDataset::new(base, vec![0, 2, 4]);

        assert_eq!(subset.len(), 3);

        let (x, _) = subset.get(0).unwrap();
        assert_eq!(x.to_vec(), vec![1.0]);

        let (x, _) = subset.get(1).unwrap();
        assert_eq!(x.to_vec(), vec![3.0]);

        let (x, _) = subset.get(2).unwrap();
        assert_eq!(x.to_vec(), vec![5.0]);
    }

    #[test]
    fn test_in_memory_dataset() {
        let dataset = InMemoryDataset::new(vec![1, 2, 3, 4, 5]);

        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.get(0), Some(1));
        assert_eq!(dataset.get(4), Some(5));
        assert_eq!(dataset.get(5), None);
    }
}
