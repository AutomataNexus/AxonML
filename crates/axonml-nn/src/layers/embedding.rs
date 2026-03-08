//! Embedding Layer - Lookup Table for Indices
//!
//! # File
//! `crates/axonml-nn/src/layers/embedding.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::any::Any;
use std::collections::HashMap;

use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_tensor::Tensor;

use crate::init::normal;
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// EmbeddingBackward
// =============================================================================

/// Gradient function for Embedding lookup.
///
/// Scatters the upstream gradient back into a sparse gradient of shape
/// `[num_embeddings, embedding_dim]` using the indices from the forward pass.
#[derive(Debug)]
struct EmbeddingBackward {
    next_fns: Vec<Option<GradFn>>,
    /// The indices used during the forward lookup (as usize).
    indices: Vec<usize>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl GradientFunction for EmbeddingBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU path: use CUDA scatter-add kernel
        #[cfg(feature = "cuda")]
        if grad_output.device().is_gpu() {
            let indices_u32: Vec<u32> = self.indices.iter().map(|&i| i as u32).collect();
            let grad_tensor = grad_output.embedding_scatter_add_cuda(
                &indices_u32,
                self.num_embeddings,
                self.embedding_dim,
            );
            return vec![Some(grad_tensor)];
        }

        // CPU fallback
        let grad_data = grad_output.to_vec();
        let mut weight_grad = vec![0.0f32; self.num_embeddings * self.embedding_dim];

        // Scatter-add: accumulate gradients for each index
        for (i, &idx) in self.indices.iter().enumerate() {
            if idx < self.num_embeddings {
                let src_offset = i * self.embedding_dim;
                let dst_offset = idx * self.embedding_dim;
                for d in 0..self.embedding_dim {
                    weight_grad[dst_offset + d] += grad_data[src_offset + d];
                }
            }
        }

        let grad_tensor = Tensor::from_vec(
            weight_grad,
            &[self.num_embeddings, self.embedding_dim],
        )
        .unwrap();
        vec![Some(grad_tensor)]
    }

    fn name(&self) -> &'static str {
        "EmbeddingBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Embedding
// =============================================================================

/// A simple lookup table that stores embeddings of a fixed dictionary.
///
/// This module is often used to store word embeddings and retrieve them
/// using indices.
///
/// # Shape
/// - Input: (*) - LongTensor of arbitrary shape containing indices
/// - Output: (*, H) - where H = embedding_dim
pub struct Embedding {
    /// Embedding weights of shape (num_embeddings, embedding_dim).
    pub weight: Parameter,
    /// Number of embeddings in the dictionary.
    num_embeddings: usize,
    /// Dimension of each embedding vector.
    embedding_dim: usize,
    /// Index of padding token (if any).
    padding_idx: Option<usize>,
}

impl Embedding {
    /// Creates a new Embedding layer.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self::with_options(num_embeddings, embedding_dim, None)
    }

    /// Creates an Embedding with padding index.
    pub fn with_options(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> Self {
        // Initialize weights from N(0, 1)
        let mut weight_data = normal(&[num_embeddings, embedding_dim], 0.0, 1.0);

        // Set padding index to zeros if specified
        if let Some(pad_idx) = padding_idx {
            let mut data = weight_data.to_vec();
            for i in 0..embedding_dim {
                data[pad_idx * embedding_dim + i] = 0.0;
            }
            weight_data = Tensor::from_vec(data, &[num_embeddings, embedding_dim]).unwrap();
        }

        Self {
            weight: Parameter::named("weight", weight_data, true),
            num_embeddings,
            embedding_dim,
            padding_idx,
        }
    }

    /// Creates an Embedding from pretrained weights.
    pub fn from_pretrained(weights: Tensor<f32>, freeze: bool) -> Self {
        let shape = weights.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Self {
            weight: Parameter::named("weight", weights, !freeze),
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        }
    }

    /// Returns the number of embeddings.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Looks up embeddings for the given indices.
    ///
    /// # Arguments
    /// * `indices` - Variable containing integer indices
    ///
    /// Note: In a full implementation, indices would be LongTensor.
    /// Here we use f32 and cast to usize.
    pub fn lookup(&self, indices: &Variable) -> Variable {
        let indices_data = indices.data();
        // Copy indices to CPU (small: batch_size * seq_len values)
        let indices_vec = indices_data.to_vec();
        let indices_shape = indices_data.shape().to_vec();

        // Output shape: indices_shape + [embedding_dim]
        let mut output_shape = indices_shape.clone();
        output_shape.push(self.embedding_dim);
        let output_size: usize = output_shape.iter().product();

        // Compute gather indices and validate on CPU (indices are small)
        let mut safe_indices = Vec::with_capacity(indices_vec.len());
        // Build flat gather index: for each token index, we need embedding_dim consecutive elements
        let mut gather_idx = Vec::with_capacity(output_size);

        for &idx_f in &indices_vec {
            let idx = idx_f as usize;
            let safe_idx = if idx >= self.num_embeddings {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Warning: embedding index {} out of range (max {}), using padding index 0",
                    idx,
                    self.num_embeddings - 1
                );
                0
            } else {
                idx
            };
            safe_indices.push(safe_idx);
            // Each token maps to embedding_dim elements starting at safe_idx * embedding_dim
            let base = safe_idx * self.embedding_dim;
            for d in 0..self.embedding_dim {
                gather_idx.push((base + d) as u32);
            }
        }

        let weight_data = self.weight.data();
        #[allow(unused_variables)]
        let weight_device = weight_data.device();

        // GPU path: use gather kernel to avoid copying entire weight matrix
        #[cfg(feature = "cuda")]
        let output_tensor = if weight_device.is_gpu() {
            weight_data.embedding_gather_cuda(&gather_idx, &output_shape)
        } else {
            let weight_vec = weight_data.to_vec();
            let output_data: Vec<f32> = gather_idx.iter().map(|&i| weight_vec[i as usize]).collect();
            Tensor::from_vec(output_data, &output_shape).unwrap()
        };

        #[cfg(not(feature = "cuda"))]
        let output_tensor = {
            let weight_vec = weight_data.to_vec();
            let output_data: Vec<f32> = gather_idx.iter().map(|&i| weight_vec[i as usize]).collect();
            Tensor::from_vec(output_data, &output_shape).unwrap()
        };

        if self.weight.requires_grad() {
            let grad_fn = GradFn::new(EmbeddingBackward {
                next_fns: vec![self.weight.variable().grad_fn().cloned()],
                indices: safe_indices,
                num_embeddings: self.num_embeddings,
                embedding_dim: self.embedding_dim,
            });
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Variable) -> Variable {
        self.lookup(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params
    }

    fn name(&self) -> &'static str {
        "Embedding"
    }
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedding")
            .field("num_embeddings", &self.num_embeddings)
            .field("embedding_dim", &self.embedding_dim)
            .field("padding_idx", &self.padding_idx)
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let emb = Embedding::new(1000, 128);
        assert_eq!(emb.num_embeddings(), 1000);
        assert_eq!(emb.embedding_dim(), 128);
    }

    #[test]
    fn test_embedding_lookup() {
        let emb = Embedding::new(10, 4);
        let indices = Variable::new(Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap(), false);
        let output = emb.forward(&indices);
        assert_eq!(output.shape(), vec![3, 4]);
    }

    #[test]
    fn test_embedding_batch() {
        let emb = Embedding::new(10, 4);
        let indices = Variable::new(
            Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap(),
            false,
        );
        let output = emb.forward(&indices);
        assert_eq!(output.shape(), vec![2, 3, 4]);
    }

    #[test]
    fn test_embedding_parameters() {
        let emb = Embedding::new(100, 64);
        assert_eq!(emb.parameters().len(), 1);
        assert_eq!(emb.num_parameters(), 100 * 64);
    }

    #[test]
    fn test_embedding_with_padding() {
        let emb = Embedding::with_options(10, 4, Some(0));
        // Padding index 0 should be all zeros
        let indices = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
        let output = emb.forward(&indices);
        let output_vec = output.data().to_vec();
        assert!(output_vec.iter().all(|&x| x == 0.0));
    }
}
