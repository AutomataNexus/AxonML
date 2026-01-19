//! Embedding Layer - Lookup Table for Indices
//!
//! Maps discrete indices to dense vectors.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::init::normal;
use crate::module::Module;
use crate::parameter::Parameter;

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
        let indices_vec = indices_data.to_vec();
        let indices_shape = indices_data.shape().to_vec();

        let weight_vec = self.weight.data().to_vec();

        // Output shape: indices_shape + [embedding_dim]
        let mut output_shape = indices_shape.clone();
        output_shape.push(self.embedding_dim);
        let output_size: usize = output_shape.iter().product();

        let mut output_data = vec![0.0f32; output_size];

        for (i, &idx_f) in indices_vec.iter().enumerate() {
            let idx = idx_f as usize;
            if idx >= self.num_embeddings {
                panic!(
                    "Index {} out of range for embedding with {} entries",
                    idx, self.num_embeddings
                );
            }

            for d in 0..self.embedding_dim {
                output_data[i * self.embedding_dim + d] = weight_vec[idx * self.embedding_dim + d];
            }
        }

        Variable::new(
            Tensor::from_vec(output_data, &output_shape).unwrap(),
            self.weight.requires_grad(),
        )
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
