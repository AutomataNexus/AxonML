//! Collate - Batch Assembly Functions
//!
//! # File
//! `crates/axonml-data/src/collate.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

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
        if batch.is_empty() {
            return Tensor::from_vec(vec![], &[0]).unwrap();
        }

        if self.dim == 0 {
            return stack_tensors(&batch);
        }

        // Stack along non-zero dimension:
        // Insert a new dimension at self.dim, then concatenate.
        // First, unsqueeze each tensor at self.dim.
        let first_shape = batch[0].shape();
        let ndim = first_shape.len();
        let dim = self.dim.min(ndim); // clamp to valid range

        // Build the shape with new dim inserted
        let mut item_shape_expanded = Vec::with_capacity(ndim + 1);
        item_shape_expanded.extend_from_slice(&first_shape[..dim]);
        item_shape_expanded.push(1);
        item_shape_expanded.extend_from_slice(&first_shape[dim..]);

        // Reshape each tensor to have size-1 at self.dim, then concat along self.dim
        let reshaped: Vec<Tensor<f32>> = batch
            .iter()
            .map(|t| Tensor::from_vec(t.to_vec(), &item_shape_expanded).unwrap())
            .collect();

        concat_tensors(&reshaped, dim)
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
    let ndim = first_shape.len();

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

    // For non-zero dims, properly interleave data.
    // outer_size = product of dims before `dim`
    // inner_size = product of dims after `dim`
    let outer_size: usize = first_shape[..dim].iter().product();
    let inner_size: usize = if dim + 1 < ndim {
        first_shape[dim + 1..].iter().product()
    } else {
        1
    };

    let total_elements: usize = new_shape.iter().product();
    let mut result = Vec::with_capacity(total_elements);

    // Precompute all tensor data
    let all_vecs: Vec<Vec<f32>> = tensors.iter().map(|t| t.to_vec()).collect();

    for o in 0..outer_size {
        // For each tensor, copy its slice along the concat dimension
        for (t_idx, tensor_data) in all_vecs.iter().enumerate() {
            let t_dim_size = tensors[t_idx].shape()[dim];
            let t_inner_stride = t_dim_size * inner_size;
            let src_offset = o * t_inner_stride;
            result.extend_from_slice(&tensor_data[src_offset..src_offset + t_inner_stride]);
        }
    }

    Tensor::from_vec(result, &new_shape).unwrap()
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
    fn test_concat_tensors_dim0() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

        let concat = concat_tensors(&[t1, t2], 0);
        assert_eq!(concat.shape(), &[5]);
        assert_eq!(concat.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concat_tensors_dim1() {
        // Two [2, 2] tensors → [2, 4]
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let concat = concat_tensors(&[t1, t2], 1);
        assert_eq!(concat.shape(), &[2, 4]);
        // Row 0: [1,2,5,6], Row 1: [3,4,7,8]
        assert_eq!(
            concat.to_vec(),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }
}
