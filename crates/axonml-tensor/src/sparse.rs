//! Sparse Tensor Support
//!
//! Provides sparse tensor representations for memory-efficient storage and
//! computation when tensors have many zero elements.
//!
//! # Formats
//! - COO (Coordinate): Best for construction and random access
//! - CSR (Compressed Sparse Row): Best for row-wise operations and matrix-vector products
//!
//! # Example
//! ```rust,ignore
//! use axonml_tensor::sparse::{SparseTensor, SparseFormat};
//!
//! // Create from COO format
//! let indices = vec![(0, 1), (1, 0), (2, 2)];
//! let values = vec![1.0, 2.0, 3.0];
//! let sparse = SparseTensor::from_coo(&indices, &values, &[3, 3]);
//!
//! // Convert to dense
//! let dense = sparse.to_dense();
//! ```
//!
//! @version 0.1.0

use crate::Tensor;

// =============================================================================
// Sparse Format
// =============================================================================

/// Sparse tensor storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Coordinate format: (row, col, value) tuples
    COO,
    /// Compressed Sparse Row format
    CSR,
    /// Compressed Sparse Column format
    CSC,
}

// =============================================================================
// COO Sparse Tensor
// =============================================================================

/// Sparse tensor in COO (Coordinate) format.
///
/// Stores non-zero elements as a list of (index, value) pairs.
/// Efficient for construction but less efficient for arithmetic.
#[derive(Debug, Clone)]
pub struct SparseCOO {
    /// Row indices of non-zero elements
    pub indices: Vec<Vec<usize>>,
    /// Values of non-zero elements
    pub values: Vec<f32>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Whether indices are sorted
    pub is_coalesced: bool,
}

impl SparseCOO {
    /// Creates a new sparse COO tensor.
    ///
    /// # Arguments
    /// * `indices` - List of index tuples, one per dimension
    /// * `values` - Non-zero values
    /// * `shape` - Shape of the tensor
    pub fn new(indices: Vec<Vec<usize>>, values: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            indices.len(),
            shape.len(),
            "indices dimensions must match shape"
        );
        if !indices.is_empty() {
            let nnz = indices[0].len();
            for idx in &indices {
                assert_eq!(idx.len(), nnz, "all index arrays must have same length");
            }
            assert_eq!(
                values.len(),
                nnz,
                "values length must match number of indices"
            );
        }

        Self {
            indices,
            values,
            shape,
            is_coalesced: false,
        }
    }

    /// Creates from a list of 2D coordinate tuples.
    pub fn from_coo_2d(coords: &[(usize, usize)], values: &[f32], shape: &[usize]) -> Self {
        assert_eq!(shape.len(), 2, "shape must be 2D");
        assert_eq!(
            coords.len(),
            values.len(),
            "coords and values must have same length"
        );

        let rows: Vec<usize> = coords.iter().map(|(r, _)| *r).collect();
        let cols: Vec<usize> = coords.iter().map(|(_, c)| *c).collect();

        Self::new(vec![rows, cols], values.to_vec(), shape.to_vec())
    }

    /// Returns number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns the density (ratio of non-zeros to total elements).
    pub fn density(&self) -> f32 {
        let total: usize = self.shape.iter().product();
        if total == 0 {
            0.0
        } else {
            self.nnz() as f32 / total as f32
        }
    }

    /// Returns the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Coalesces the sparse tensor (combines duplicate indices).
    pub fn coalesce(&mut self) {
        if self.is_coalesced || self.nnz() == 0 {
            self.is_coalesced = true;
            return;
        }

        // Create (indices, value) pairs and sort
        let mut entries: Vec<(Vec<usize>, f32)> = (0..self.nnz())
            .map(|i| {
                let idx: Vec<usize> = self.indices.iter().map(|dim| dim[i]).collect();
                (idx, self.values[i])
            })
            .collect();

        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Combine duplicates
        let mut new_indices: Vec<Vec<usize>> = vec![Vec::new(); self.shape.len()];
        let mut new_values = Vec::new();

        let mut prev_idx: Option<Vec<usize>> = None;

        for (idx, val) in entries {
            if prev_idx.as_ref() == Some(&idx) {
                // Duplicate: add to previous value
                if let Some(last) = new_values.last_mut() {
                    *last += val;
                }
            } else {
                // New index
                for (d, i) in idx.iter().enumerate() {
                    new_indices[d].push(*i);
                }
                new_values.push(val);
                prev_idx = Some(idx);
            }
        }

        self.indices = new_indices;
        self.values = new_values;
        self.is_coalesced = true;
    }

    /// Converts to dense tensor.
    pub fn to_dense(&self) -> Tensor<f32> {
        let total: usize = self.shape.iter().product();
        let mut data = vec![0.0f32; total];

        for i in 0..self.nnz() {
            let mut flat_idx = 0;
            let mut stride = 1;
            for d in (0..self.shape.len()).rev() {
                flat_idx += self.indices[d][i] * stride;
                stride *= self.shape[d];
            }
            data[flat_idx] += self.values[i];
        }

        Tensor::from_vec(data, &self.shape).unwrap()
    }

    /// Converts to CSR format (for 2D matrices).
    pub fn to_csr(&self) -> SparseCSR {
        assert_eq!(self.shape.len(), 2, "CSR only supports 2D tensors");

        let mut coo = self.clone();
        coo.coalesce();

        let nrows = self.shape[0];
        let nnz = coo.nnz();

        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Count entries per row
        for &row in &coo.indices[0] {
            row_ptr[row + 1] += 1;
        }

        // Cumulative sum
        for i in 1..=nrows {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Sort by row, then column
        let mut entries: Vec<(usize, usize, f32)> = (0..nnz)
            .map(|i| (coo.indices[0][i], coo.indices[1][i], coo.values[i]))
            .collect();
        entries.sort_by_key(|(r, c, _)| (*r, *c));

        for (_, col, val) in entries {
            col_indices.push(col);
            values.push(val);
        }

        SparseCSR {
            row_ptr,
            col_indices,
            values,
            shape: self.shape.clone(),
        }
    }
}

// =============================================================================
// CSR Sparse Tensor
// =============================================================================

/// Sparse tensor in CSR (Compressed Sparse Row) format.
///
/// Efficient for row-wise operations and sparse matrix-vector products.
#[derive(Debug, Clone)]
pub struct SparseCSR {
    /// Row pointers (length = nrows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices for each non-zero
    pub col_indices: Vec<usize>,
    /// Values for each non-zero
    pub values: Vec<f32>,
    /// Shape [nrows, ncols]
    pub shape: Vec<usize>,
}

impl SparseCSR {
    /// Creates a new CSR sparse matrix.
    pub fn new(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Vec<usize>,
    ) -> Self {
        assert_eq!(shape.len(), 2, "CSR only supports 2D tensors");
        assert_eq!(
            row_ptr.len(),
            shape[0] + 1,
            "row_ptr length must be nrows + 1"
        );
        assert_eq!(
            col_indices.len(),
            values.len(),
            "col_indices and values must match"
        );

        Self {
            row_ptr,
            col_indices,
            values,
            shape,
        }
    }

    /// Returns number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns number of rows.
    pub fn nrows(&self) -> usize {
        self.shape[0]
    }

    /// Returns number of columns.
    pub fn ncols(&self) -> usize {
        self.shape[1]
    }

    /// Returns the density.
    pub fn density(&self) -> f32 {
        let total = self.nrows() * self.ncols();
        if total == 0 {
            0.0
        } else {
            self.nnz() as f32 / total as f32
        }
    }

    /// Gets entries for a specific row.
    pub fn row(&self, row_idx: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let start = self.row_ptr[row_idx];
        let end = self.row_ptr[row_idx + 1];
        (start..end).map(move |i| (self.col_indices[i], self.values[i]))
    }

    /// Sparse matrix-vector multiplication: A @ x.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.ncols(), "vector length must match ncols");

        let mut result = vec![0.0f32; self.nrows()];

        for row in 0..self.nrows() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_indices[i];
                let val = self.values[i];
                result[row] += val * x[col];
            }
        }

        result
    }

    /// Sparse matrix-matrix multiplication: A @ B (where B is dense).
    pub fn matmul_dense(&self, b: &Tensor<f32>) -> Tensor<f32> {
        let b_shape = b.shape();
        assert_eq!(b_shape[0], self.ncols(), "inner dimensions must match");

        let m = self.nrows();
        let n = b_shape[1];
        let b_data = b.to_vec();

        let mut result = vec![0.0f32; m * n];

        for row in 0..m {
            for (col, val) in self.row(row) {
                for j in 0..n {
                    result[row * n + j] += val * b_data[col * n + j];
                }
            }
        }

        Tensor::from_vec(result, &[m, n]).unwrap()
    }

    /// Converts to dense tensor.
    pub fn to_dense(&self) -> Tensor<f32> {
        let mut data = vec![0.0f32; self.nrows() * self.ncols()];

        for row in 0..self.nrows() {
            for (col, val) in self.row(row) {
                data[row * self.ncols() + col] = val;
            }
        }

        Tensor::from_vec(data, &self.shape).unwrap()
    }

    /// Converts to COO format.
    pub fn to_coo(&self) -> SparseCOO {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());

        for row in 0..self.nrows() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for i in start..end {
                rows.push(row);
                cols.push(self.col_indices[i]);
            }
        }

        SparseCOO {
            indices: vec![rows, cols],
            values: self.values.clone(),
            shape: self.shape.clone(),
            is_coalesced: true,
        }
    }
}

// =============================================================================
// SparseTensor (Unified Interface)
// =============================================================================

/// Unified sparse tensor interface supporting multiple formats.
#[derive(Debug, Clone)]
pub enum SparseTensor {
    /// COO format
    COO(SparseCOO),
    /// CSR format
    CSR(SparseCSR),
}

impl SparseTensor {
    /// Creates a sparse tensor from COO data.
    pub fn from_coo(indices: Vec<Vec<usize>>, values: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::COO(SparseCOO::new(indices, values, shape))
    }

    /// Creates a 2D sparse tensor from coordinate list.
    pub fn from_coords(coords: &[(usize, usize)], values: &[f32], shape: &[usize]) -> Self {
        Self::COO(SparseCOO::from_coo_2d(coords, values, shape))
    }

    /// Creates from a dense tensor, keeping only non-zero elements.
    pub fn from_dense(tensor: &Tensor<f32>, threshold: f32) -> Self {
        let data = tensor.to_vec();
        let shape = tensor.shape().to_vec();

        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); shape.len()];
        let mut values = Vec::new();

        let strides: Vec<usize> = {
            let mut s = vec![1; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                s[i] = s[i + 1] * shape[i + 1];
            }
            s
        };

        for (flat_idx, &val) in data.iter().enumerate() {
            if val.abs() > threshold {
                let mut idx = flat_idx;
                for (d, &stride) in strides.iter().enumerate() {
                    indices[d].push(idx / stride);
                    idx %= stride;
                }
                values.push(val);
            }
        }

        Self::COO(SparseCOO::new(indices, values, shape))
    }

    /// Creates an identity matrix in sparse format.
    pub fn eye(n: usize) -> Self {
        let indices: Vec<usize> = (0..n).collect();
        let values = vec![1.0f32; n];
        Self::COO(SparseCOO::new(
            vec![indices.clone(), indices],
            values,
            vec![n, n],
        ))
    }

    /// Creates a sparse diagonal matrix.
    pub fn diag(values: &[f32]) -> Self {
        let n = values.len();
        let indices: Vec<usize> = (0..n).collect();
        Self::COO(SparseCOO::new(
            vec![indices.clone(), indices],
            values.to_vec(),
            vec![n, n],
        ))
    }

    /// Returns number of non-zero elements.
    pub fn nnz(&self) -> usize {
        match self {
            Self::COO(coo) => coo.nnz(),
            Self::CSR(csr) => csr.nnz(),
        }
    }

    /// Returns the shape.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::COO(coo) => &coo.shape,
            Self::CSR(csr) => &csr.shape,
        }
    }

    /// Returns the density.
    pub fn density(&self) -> f32 {
        match self {
            Self::COO(coo) => coo.density(),
            Self::CSR(csr) => csr.density(),
        }
    }

    /// Converts to dense tensor.
    pub fn to_dense(&self) -> Tensor<f32> {
        match self {
            Self::COO(coo) => coo.to_dense(),
            Self::CSR(csr) => csr.to_dense(),
        }
    }

    /// Converts to CSR format.
    pub fn to_csr(&self) -> SparseCSR {
        match self {
            Self::COO(coo) => coo.to_csr(),
            Self::CSR(csr) => csr.clone(),
        }
    }

    /// Converts to COO format.
    pub fn to_coo(&self) -> SparseCOO {
        match self {
            Self::COO(coo) => coo.clone(),
            Self::CSR(csr) => csr.to_coo(),
        }
    }

    /// Sparse matrix-vector multiplication.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::COO(coo) => coo.to_csr().matvec(x),
            Self::CSR(csr) => csr.matvec(x),
        }
    }

    /// Sparse-dense matrix multiplication.
    pub fn matmul(&self, dense: &Tensor<f32>) -> Tensor<f32> {
        match self {
            Self::COO(coo) => coo.to_csr().matmul_dense(dense),
            Self::CSR(csr) => csr.matmul_dense(dense),
        }
    }

    /// Element-wise multiplication with a scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        match self {
            Self::COO(coo) => {
                let values: Vec<f32> = coo.values.iter().map(|v| v * scalar).collect();
                Self::COO(SparseCOO::new(
                    coo.indices.clone(),
                    values,
                    coo.shape.clone(),
                ))
            }
            Self::CSR(csr) => {
                let values: Vec<f32> = csr.values.iter().map(|v| v * scalar).collect();
                Self::CSR(SparseCSR::new(
                    csr.row_ptr.clone(),
                    csr.col_indices.clone(),
                    values,
                    csr.shape.clone(),
                ))
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_coo_creation() {
        let indices = vec![vec![0, 1, 2], vec![1, 0, 2]];
        let values = vec![1.0, 2.0, 3.0];
        let sparse = SparseCOO::new(indices, values, vec![3, 3]);

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.shape(), &[3, 3]);
    }

    #[test]
    fn test_sparse_coo_to_dense() {
        let coords = vec![(0, 1), (1, 0), (2, 2)];
        let values = vec![1.0, 2.0, 3.0];
        let sparse = SparseCOO::from_coo_2d(&coords, &values, &[3, 3]);

        let dense = sparse.to_dense();
        let data = dense.to_vec();

        assert_eq!(data[0 * 3 + 1], 1.0); // (0, 1)
        assert_eq!(data[1 * 3 + 0], 2.0); // (1, 0)
        assert_eq!(data[2 * 3 + 2], 3.0); // (2, 2)
    }

    #[test]
    fn test_sparse_coo_coalesce() {
        let indices = vec![vec![0, 0, 1], vec![0, 0, 1]];
        let values = vec![1.0, 2.0, 3.0];
        let mut sparse = SparseCOO::new(indices, values, vec![2, 2]);

        sparse.coalesce();

        assert_eq!(sparse.nnz(), 2); // Duplicates combined
        let dense = sparse.to_dense();
        assert_eq!(dense.to_vec()[0], 3.0); // 1.0 + 2.0
    }

    #[test]
    fn test_sparse_csr_creation() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 0, 2];
        let values = vec![1.0, 2.0, 3.0];
        let csr = SparseCSR::new(row_ptr, col_indices, values, vec![3, 3]);

        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
    }

    #[test]
    fn test_sparse_csr_matvec() {
        // Matrix: [[1, 0], [0, 2]]
        let row_ptr = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0, 2.0];
        let csr = SparseCSR::new(row_ptr, col_indices, values, vec![2, 2]);

        let x = vec![1.0, 2.0];
        let result = csr.matvec(&x);

        assert_eq!(result, vec![1.0, 4.0]);
    }

    #[test]
    fn test_sparse_coo_to_csr() {
        let coords = vec![(0, 1), (1, 0), (2, 2)];
        let values = vec![1.0, 2.0, 3.0];
        let coo = SparseCOO::from_coo_2d(&coords, &values, &[3, 3]);

        let csr = coo.to_csr();

        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.nrows(), 3);
    }

    #[test]
    fn test_sparse_tensor_from_dense() {
        let dense = Tensor::from_vec(vec![0.0, 1.0, 0.0, 2.0], &[2, 2]).unwrap();
        let sparse = SparseTensor::from_dense(&dense, 0.0);

        assert_eq!(sparse.nnz(), 2);
    }

    #[test]
    fn test_sparse_tensor_eye() {
        let eye = SparseTensor::eye(3);
        let dense = eye.to_dense();
        let data = dense.to_vec();

        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 1.0);
        assert_eq!(data[8], 1.0);
        assert_eq!(data[1], 0.0);
    }

    #[test]
    fn test_sparse_tensor_diag() {
        let diag = SparseTensor::diag(&[1.0, 2.0, 3.0]);
        let dense = diag.to_dense();
        let data = dense.to_vec();

        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 2.0);
        assert_eq!(data[8], 3.0);
    }

    #[test]
    fn test_sparse_density() {
        let coords = vec![(0, 0), (1, 1)];
        let values = vec![1.0, 2.0];
        let sparse = SparseTensor::from_coords(&coords, &values, &[4, 4]);

        assert!((sparse.density() - 0.125).abs() < 1e-6); // 2/16
    }

    #[test]
    fn test_sparse_mul_scalar() {
        let coords = vec![(0, 0)];
        let values = vec![2.0];
        let sparse = SparseTensor::from_coords(&coords, &values, &[2, 2]);

        let scaled = sparse.mul_scalar(3.0);
        let dense = scaled.to_dense();

        assert_eq!(dense.to_vec()[0], 6.0);
    }

    #[test]
    fn test_sparse_matmul() {
        // Sparse: [[1, 0], [0, 2]]
        let coords = vec![(0, 0), (1, 1)];
        let values = vec![1.0, 2.0];
        let sparse = SparseTensor::from_coords(&coords, &values, &[2, 2]);

        // Dense: [[1, 2], [3, 4]]
        let dense = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = sparse.matmul(&dense);
        let data = result.to_vec();

        // [[1, 0], [0, 2]] @ [[1, 2], [3, 4]] = [[1, 2], [6, 8]]
        assert_eq!(data, vec![1.0, 2.0, 6.0, 8.0]);
    }
}
