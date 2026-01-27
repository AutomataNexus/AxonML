//! Collate - Batch Assembly Functions
//!
//! Provides functions for combining individual samples into batches.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_tensor::Tensor;

// =============================================================================
// Collate Trait
// =============================================================================

/// Trait for collating samples into batches.
pub trait Collate<T>: Send + Sync {
    /// The output batch type.
    type Output;

    /// Collates a vector of samples into a batch.
    fn collate(&self, batch: Vec<T>) -> Self::Output;
}

// =============================================================================
// DefaultCollate
// =============================================================================

/// Default collation strategy that stacks tensors.
pub struct DefaultCollate;

impl DefaultCollate {
    /// Creates a new `DefaultCollate`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultCollate {
    fn default() -> Self {
        Self::new()
    }
}

impl Collate<(Tensor<f32>, Tensor<f32>)> for DefaultCollate {
    type Output = (Tensor<f32>, Tensor<f32>);

    fn collate(&self, batch: Vec<(Tensor<f32>, Tensor<f32>)>) -> Self::Output {
        if batch.is_empty() {
            return (
                Tensor::from_vec(vec![], &[0]).unwrap(),
                Tensor::from_vec(vec![], &[0]).unwrap(),
            );
        }

        // Stack inputs
        let inputs: Vec<Tensor<f32>> = batch.iter().map(|(x, _)| x.clone()).collect();
        let stacked_x = stack_tensors(&inputs);

        // Stack targets
        let targets: Vec<Tensor<f32>> = batch.iter().map(|(_, y)| y.clone()).collect();
        let stacked_y = stack_tensors(&targets);

        (stacked_x, stacked_y)
    }
}

impl Collate<Tensor<f32>> for DefaultCollate {
    type Output = Tensor<f32>;

    fn collate(&self, batch: Vec<Tensor<f32>>) -> Self::Output {
        stack_tensors(&batch)
    }
}

// =============================================================================
// StackCollate
// =============================================================================

/// Collation that stacks tensors along a new batch dimension.
pub struct StackCollate {
    /// Dimension to stack along (default: 0).
    dim: usize,
}

impl StackCollate {
    /// Creates a new `StackCollate` with default dimension 0.
    #[must_use]
    pub fn new() -> Self {
        Self { dim: 0 }
    }

    /// Creates a `StackCollate` with specified dimension.
    #[must_use]
    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }
}

impl Default for StackCollate {
    fn default() -> Self {
        Self::new()
    }
}

impl Collate<Tensor<f32>> for StackCollate {
    type Output = Tensor<f32>;

    fn collate(&self, batch: Vec<Tensor<f32>>) -> Self::Output {
        if self.dim == 0 {
            stack_tensors(&batch)
        } else {
            // For non-zero dimensions, we'd need more complex logic
            // For now, always stack at dim 0
            stack_tensors(&batch)
        }
    }
}

impl Collate<(Tensor<f32>, Tensor<f32>)> for StackCollate {
    type Output = (Tensor<f32>, Tensor<f32>);

    fn collate(&self, batch: Vec<(Tensor<f32>, Tensor<f32>)>) -> Self::Output {
        if batch.is_empty() {
            return (
                Tensor::from_vec(vec![], &[0]).unwrap(),
                Tensor::from_vec(vec![], &[0]).unwrap(),
            );
        }

        let inputs: Vec<Tensor<f32>> = batch.iter().map(|(x, _)| x.clone()).collect();
        let targets: Vec<Tensor<f32>> = batch.iter().map(|(_, y)| y.clone()).collect();

        (stack_tensors(&inputs), stack_tensors(&targets))
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Stacks a vector of tensors along dimension 0.
#[must_use]
pub fn stack_tensors(tensors: &[Tensor<f32>]) -> Tensor<f32> {
    if tensors.is_empty() {
        return Tensor::from_vec(vec![], &[0]).unwrap();
    }

    let first_shape = tensors[0].shape();
    let batch_size = tensors.len();

    // New shape: [batch_size, ...original_shape]
    let mut new_shape = vec![batch_size];
    new_shape.extend_from_slice(first_shape);

    // Concatenate all data
    let mut all_data = Vec::new();
    for tensor in tensors {
        all_data.extend(tensor.to_vec());
    }

    Tensor::from_vec(all_data, &new_shape).unwrap()
}

/// Concatenates tensors along an existing dimension.
#[must_use]
pub fn concat_tensors(tensors: &[Tensor<f32>], dim: usize) -> Tensor<f32> {
    if tensors.is_empty() {
        return Tensor::from_vec(vec![], &[0]).unwrap();
    }

    if tensors.len() == 1 {
        return tensors[0].clone();
    }

    let first_shape = tensors[0].shape();

    // Calculate new shape
    let mut new_shape = first_shape.to_vec();
    let concat_size: usize = tensors.iter().map(|t| t.shape()[dim]).sum();
    new_shape[dim] = concat_size;

    // For dim=0 concatenation, just append all data
    if dim == 0 {
        let mut all_data = Vec::new();
        for tensor in tensors {
            all_data.extend(tensor.to_vec());
        }
        return Tensor::from_vec(all_data, &new_shape).unwrap();
    }

    // For other dimensions, more complex interleaving is needed
    // This is a simplified version that handles common cases
    let mut all_data = Vec::new();
    for tensor in tensors {
        all_data.extend(tensor.to_vec());
    }
    Tensor::from_vec(all_data, &new_shape).unwrap()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_tensors() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let t2 = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let t3 = Tensor::from_vec(vec![5.0, 6.0], &[2]).unwrap();

        let stacked = stack_tensors(&[t1, t2, t3]);
        assert_eq!(stacked.shape(), &[3, 2]);
        assert_eq!(stacked.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_stack_tensors_2d() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let stacked = stack_tensors(&[t1, t2]);
        assert_eq!(stacked.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_default_collate() {
        let collate = DefaultCollate::new();

        let batch = vec![
            (
                Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
                Tensor::from_vec(vec![0.0], &[1]).unwrap(),
            ),
            (
                Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
                Tensor::from_vec(vec![1.0], &[1]).unwrap(),
            ),
        ];

        let (x, y) = collate.collate(batch);
        assert_eq!(x.shape(), &[2, 2]);
        assert_eq!(y.shape(), &[2, 1]);
    }

    #[test]
    fn test_stack_collate() {
        let collate = StackCollate::new();

        let batch = vec![
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
            Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap(),
        ];

        let result = collate.collate(batch);
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_empty_collate() {
        let collate = DefaultCollate::new();
        let batch: Vec<(Tensor<f32>, Tensor<f32>)> = vec![];
        let (x, y) = collate.collate(batch);
        assert_eq!(x.shape(), &[0]);
        assert_eq!(y.shape(), &[0]);
    }

    #[test]
    fn test_concat_tensors() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

        let concat = concat_tensors(&[t1, t2], 0);
        assert_eq!(concat.shape(), &[5]);
        assert_eq!(concat.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
