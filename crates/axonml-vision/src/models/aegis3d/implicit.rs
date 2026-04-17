//! Neural Implicit Surfaces — SDF Networks with Fourier Positional Encoding
//!
//! Provides `FourierFeatures` for sinusoidal positional encoding of 3D coordinates,
//! `LocalSDF` for per-octree-cell signed distance function networks (3-layer MLP
//! mapping Fourier-encoded xyz to scalar distance), and `GlobalSDF` as a single
//! larger MLP baseline for whole-scene SDF. Both implement the `Module` trait for
//! forward pass and parameter collection. `LocalSDF` supports coordinate
//! normalization, single-point evaluation, and finite-difference gradient estimation
//! for eikonal regularization and surface normal computation.
//!
//! # File
//! `crates/axonml-vision/src/models/aegis3d/implicit.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#![allow(missing_docs)]

use axonml_autograd::Variable;
use axonml_nn::{Linear, Module, Parameter, ReLU};
use axonml_tensor::Tensor;

// =============================================================================
// Fourier Features
// =============================================================================

/// Fourier feature positional encoding.
///
/// Maps low-dimensional coordinates to high-dimensional features using
/// sinusoidal functions at multiple frequencies. This enables MLPs to
/// learn high-frequency geometry details.
///
/// `γ(x) = [sin(2⁰πx), cos(2⁰πx), ..., sin(2^(L-1)πx), cos(2^(L-1)πx)]`
pub struct FourierFeatures {
    /// Number of frequency bands
    pub num_frequencies: usize,
    /// Input dimension (typically 3 for xyz)
    pub input_dim: usize,
}

impl FourierFeatures {
    pub fn new(input_dim: usize, num_frequencies: usize) -> Self {
        Self {
            num_frequencies,
            input_dim,
        }
    }

    /// Output dimension = input_dim + 2 * input_dim * num_frequencies.
    pub fn output_dim(&self) -> usize {
        self.input_dim + 2 * self.input_dim * self.num_frequencies
    }

    /// Encode a batch of 3D coordinates.
    ///
    /// # Arguments
    /// - `coords`: `[N, input_dim]` coordinate tensor
    ///
    /// # Returns
    /// `[N, output_dim]` encoded features
    pub fn encode(&self, coords: &Variable) -> Variable {
        let data = coords.data().to_vec();
        let shape = coords.shape();
        let n = shape[0];
        let d = shape[1];
        let out_dim = self.output_dim();

        let mut encoded = vec![0.0f32; n * out_dim];

        for i in 0..n {
            let mut offset = 0;

            // Identity mapping
            for j in 0..d {
                encoded[i * out_dim + offset] = data[i * d + j];
                offset += 1;
            }

            // Sinusoidal encoding
            for freq in 0..self.num_frequencies {
                let scale = std::f32::consts::PI * (1 << freq) as f32;
                for j in 0..d {
                    let val = data[i * d + j] * scale;
                    encoded[i * out_dim + offset] = val.sin();
                    offset += 1;
                    encoded[i * out_dim + offset] = val.cos();
                    offset += 1;
                }
            }
        }

        Variable::new(
            Tensor::from_vec(encoded, &[n, out_dim]).unwrap(),
            coords.requires_grad(),
        )
    }
}

// =============================================================================
// Local SDF Network
// =============================================================================

/// Local SDF (Signed Distance Function) network.
///
/// A small MLP that maps encoded 3D coordinates to signed distance values.
/// Negative = inside the surface, positive = outside, zero = on the surface.
///
/// Architecture: `encoded_coords → Linear → ReLU → Linear → ReLU → Linear → SDF`
pub struct LocalSDF {
    fourier: FourierFeatures,
    layers: Vec<Linear>,
    relu: ReLU,
    /// Center of the local region this SDF covers
    pub center: [f32; 3],
    /// Size of the local region
    pub extent: f32,
}

impl LocalSDF {
    /// Create a new LocalSDF network.
    ///
    /// # Arguments
    /// - `hidden_dim`: Hidden layer width (default 64)
    /// - `num_frequencies`: Fourier encoding frequencies (default 4)
    /// - `center`: Center of the local cell in world coordinates
    /// - `extent`: Half-size of the local cell
    pub fn new(hidden_dim: usize, num_frequencies: usize, center: [f32; 3], extent: f32) -> Self {
        let fourier = FourierFeatures::new(3, num_frequencies);
        let input_dim = fourier.output_dim();

        Self {
            fourier,
            layers: vec![
                Linear::new(input_dim, hidden_dim),
                Linear::new(hidden_dim, hidden_dim),
                Linear::new(hidden_dim, 1),
            ],
            relu: ReLU,
            center,
            extent,
        }
    }

    /// Create with default parameters.
    pub fn default_at(center: [f32; 3], extent: f32) -> Self {
        Self::new(64, 4, center, extent)
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------

    /// Evaluate SDF at given world-space coordinates.
    ///
    /// Coordinates are normalized to [-1, 1] relative to the local cell.
    ///
    /// # Arguments
    /// - `coords`: `[N, 3]` world-space 3D coordinates
    ///
    /// # Returns
    /// `[N, 1]` signed distance values
    pub fn evaluate(&self, coords: &Variable) -> Variable {
        // Normalize to local coordinates
        let data = coords.data().to_vec();
        let shape = coords.shape();
        let n = shape[0];

        let mut local_coords = vec![0.0f32; n * 3];
        for i in 0..n {
            local_coords[i * 3] = (data[i * 3] - self.center[0]) / self.extent;
            local_coords[i * 3 + 1] = (data[i * 3 + 1] - self.center[1]) / self.extent;
            local_coords[i * 3 + 2] = (data[i * 3 + 2] - self.center[2]) / self.extent;
        }

        let local = Variable::new(
            Tensor::from_vec(local_coords, &[n, 3]).unwrap(),
            coords.requires_grad(),
        );

        // Fourier encode
        let encoded = self.fourier.encode(&local);

        // MLP forward
        let mut x = self.relu.forward(&self.layers[0].forward(&encoded));
        x = self.relu.forward(&self.layers[1].forward(&x));
        self.layers[2].forward(&x) // No activation on final layer (raw SDF)
    }

    /// Evaluate SDF at a single point (returns scalar).
    pub fn evaluate_single(&self, x: f32, y: f32, z: f32) -> f32 {
        let coords = Variable::new(Tensor::from_vec(vec![x, y, z], &[1, 3]).unwrap(), false);
        let result = self.evaluate(&coords);
        result.data().to_vec()[0]
    }

    // -------------------------------------------------------------------------
    // Gradient Estimation
    // -------------------------------------------------------------------------

    /// Compute the approximate gradient of SDF (∇SDF) via finite differences.
    ///
    /// Used for eikonal loss (|∇SDF| = 1) and surface normal estimation.
    pub fn gradient(&self, coords: &Variable, epsilon: f32) -> Variable {
        let data = coords.data().to_vec();
        let shape = coords.shape();
        let n = shape[0];

        let mut grad_data = vec![0.0f32; n * 3];

        for i in 0..n {
            let (x, y, z) = (data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);

            // Central differences
            let dx =
                self.evaluate_single(x + epsilon, y, z) - self.evaluate_single(x - epsilon, y, z);
            let dy =
                self.evaluate_single(x, y + epsilon, z) - self.evaluate_single(x, y - epsilon, z);
            let dz =
                self.evaluate_single(x, y, z + epsilon) - self.evaluate_single(x, y, z - epsilon);

            let scale = 1.0 / (2.0 * epsilon);
            grad_data[i * 3] = dx * scale;
            grad_data[i * 3 + 1] = dy * scale;
            grad_data[i * 3 + 2] = dz * scale;
        }

        Variable::new(
            Tensor::from_vec(grad_data, &[n, 3]).unwrap(),
            coords.requires_grad(),
        )
    }
}

// -------------------------------------------------------------------------
// Module Impl — LocalSDF
// -------------------------------------------------------------------------

impl Module for LocalSDF {
    fn forward(&self, x: &Variable) -> Variable {
        self.evaluate(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// Global SDF (for comparison/fallback)
// =============================================================================

/// Global SDF network — a single larger MLP for the entire scene.
///
/// Used as a baseline or for initial coarse reconstruction before
/// octree subdivision.
pub struct GlobalSDF {
    fourier: FourierFeatures,
    layers: Vec<Linear>,
    relu: ReLU,
}

impl GlobalSDF {
    /// Create a global SDF network.
    pub fn new(hidden_dim: usize, num_layers: usize, num_frequencies: usize) -> Self {
        let fourier = FourierFeatures::new(3, num_frequencies);
        let input_dim = fourier.output_dim();

        let mut layers = Vec::new();
        layers.push(Linear::new(input_dim, hidden_dim));
        for _ in 1..num_layers {
            layers.push(Linear::new(hidden_dim, hidden_dim));
        }
        layers.push(Linear::new(hidden_dim, 1));

        Self {
            fourier,
            layers,
            relu: ReLU,
        }
    }

    /// Default configuration: 128-wide, 4 layers, 6 frequency bands.
    pub fn default_config() -> Self {
        Self::new(128, 4, 6)
    }

    pub fn evaluate(&self, coords: &Variable) -> Variable {
        let encoded = self.fourier.encode(coords);

        let num_layers = self.layers.len();
        let mut x = encoded;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < num_layers - 1 {
                x = self.relu.forward(&x);
            }
        }
        x
    }
}

// -------------------------------------------------------------------------
// Module Impl — GlobalSDF
// -------------------------------------------------------------------------

impl Module for GlobalSDF {
    fn forward(&self, x: &Variable) -> Variable {
        self.evaluate(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fourier_features() {
        let ff = FourierFeatures::new(3, 4);
        assert_eq!(ff.output_dim(), 3 + 2 * 3 * 4); // 27

        let coords = Variable::new(
            Tensor::from_vec(vec![0.5, 0.3, 0.7, -0.1, 0.2, 0.9], &[2, 3]).unwrap(),
            false,
        );
        let encoded = ff.encode(&coords);
        assert_eq!(encoded.shape(), vec![2, 27]);
    }

    #[test]
    fn test_local_sdf() {
        let sdf = LocalSDF::default_at([0.0, 0.0, 0.0], 1.0);
        let coords = Variable::new(
            Tensor::from_vec(vec![0.1, 0.2, 0.3, -0.1, 0.0, 0.5], &[2, 3]).unwrap(),
            false,
        );
        let output = sdf.evaluate(&coords);
        assert_eq!(output.shape(), vec![2, 1]);
    }

    #[test]
    fn test_local_sdf_gradient() {
        let sdf = LocalSDF::default_at([0.0, 0.0, 0.0], 1.0);
        let coords = Variable::new(
            Tensor::from_vec(vec![0.1, 0.2, 0.3], &[1, 3]).unwrap(),
            false,
        );
        let grad = sdf.gradient(&coords, 0.001);
        assert_eq!(grad.shape(), vec![1, 3]);
    }

    #[test]
    fn test_global_sdf() {
        let sdf = GlobalSDF::default_config();
        let coords = Variable::new(
            Tensor::from_vec(vec![0.1, 0.2, 0.3, -0.5, 0.0, 0.8], &[2, 3]).unwrap(),
            false,
        );
        let output = sdf.evaluate(&coords);
        assert_eq!(output.shape(), vec![2, 1]);
    }

    #[test]
    fn test_local_sdf_parameters() {
        let sdf = LocalSDF::default_at([0.0, 0.0, 0.0], 1.0);
        let params = sdf.parameters();
        assert!(!params.is_empty());
        // 3 layers × (weight + bias) = 6 parameter tensors
        assert_eq!(params.len(), 6);
    }
}
