//! Normalization Layers - BatchNorm and LayerNorm
//!
//! Normalizes inputs to improve training stability and speed.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use axonml_autograd::Variable;
use axonml_tensor::Tensor;
use parking_lot::RwLock;

use crate::init::{ones, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// BatchNorm1d
// =============================================================================

/// Applies Batch Normalization over a 2D or 3D input.
///
/// y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
///
/// # Shape
/// - Input: (N, C) or (N, C, L)
/// - Output: Same as input
pub struct BatchNorm1d {
    /// Learnable scale parameter (gamma).
    pub weight: Parameter,
    /// Learnable shift parameter (beta).
    pub bias: Parameter,
    /// Running mean for inference (updated during training).
    running_mean: RwLock<Tensor<f32>>,
    /// Running variance for inference (updated during training).
    running_var: RwLock<Tensor<f32>>,
    /// Number of features.
    num_features: usize,
    /// Epsilon for numerical stability.
    eps: f32,
    /// Momentum for running stats update: running = (1 - momentum) * running + momentum * batch.
    momentum: f32,
    /// Whether to track running stats.
    track_running_stats: bool,
    /// Whether in training mode.
    training: AtomicBool,
}

impl BatchNorm1d {
    /// Creates a new BatchNorm1d layer.
    pub fn new(num_features: usize) -> Self {
        Self::with_options(num_features, 1e-5, 0.1, true)
    }

    /// Creates a BatchNorm1d with custom options.
    pub fn with_options(
        num_features: usize,
        eps: f32,
        momentum: f32,
        track_running_stats: bool,
    ) -> Self {
        Self {
            weight: Parameter::named("weight", ones(&[num_features]), true),
            bias: Parameter::named("bias", zeros(&[num_features]), true),
            running_mean: RwLock::new(zeros(&[num_features])),
            running_var: RwLock::new(ones(&[num_features])),
            num_features,
            eps,
            momentum,
            track_running_stats,
            training: AtomicBool::new(true),
        }
    }

    /// Returns the number of features.
    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let num_features = shape[1];

        // Validate input matches expected features
        assert_eq!(
            num_features, self.num_features,
            "BatchNorm1d: expected {} features, got {}",
            self.num_features, num_features
        );

        let input_vec = input_data.to_vec();
        let weight_vec = self.weight.data().to_vec();
        let bias_vec = self.bias.data().to_vec();

        let is_training = self.training.load(Ordering::Relaxed);
        let spatial_size: usize = if shape.len() > 2 {
            shape[2..].iter().product()
        } else {
            1
        };

        let mut means = vec![0.0f32; num_features];
        let mut vars = vec![0.0f32; num_features];

        if is_training {
            // Calculate batch statistics
            for c in 0..num_features {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * num_features * spatial_size + c * spatial_size + s;
                        sum += input_vec[idx];
                    }
                }
                means[c] = sum / (batch_size * spatial_size) as f32;

                let mut var_sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * num_features * spatial_size + c * spatial_size + s;
                        let diff = input_vec[idx] - means[c];
                        var_sum += diff * diff;
                    }
                }
                vars[c] = var_sum / (batch_size * spatial_size) as f32;
            }

            // Update running statistics if tracking is enabled
            if self.track_running_stats {
                let mut running_mean = self.running_mean.write();
                let mut running_var = self.running_var.write();
                let running_mean_vec = running_mean.to_vec();
                let running_var_vec = running_var.to_vec();

                let new_mean: Vec<f32> = running_mean_vec
                    .iter()
                    .zip(means.iter())
                    .map(|(&rm, &m)| (1.0 - self.momentum) * rm + self.momentum * m)
                    .collect();
                let new_var: Vec<f32> = running_var_vec
                    .iter()
                    .zip(vars.iter())
                    .map(|(&rv, &v)| (1.0 - self.momentum) * rv + self.momentum * v)
                    .collect();

                *running_mean = Tensor::from_vec(new_mean, &[num_features]).unwrap();
                *running_var = Tensor::from_vec(new_var, &[num_features]).unwrap();
            }
        } else {
            // Use running statistics for inference
            means = self.running_mean.read().to_vec();
            vars = self.running_var.read().to_vec();
        }

        // Normalize: y = (x - mean) / sqrt(var + eps) * weight + bias
        let mut output_vec = vec![0.0f32; input_vec.len()];
        for b in 0..batch_size {
            for c in 0..num_features {
                for s in 0..spatial_size {
                    let idx = b * num_features * spatial_size + c * spatial_size + s;
                    let normalized = (input_vec[idx] - means[c]) / (vars[c] + self.eps).sqrt();
                    output_vec[idx] = normalized * weight_vec[c] + bias_vec[c];
                }
            }
        }

        let output = Tensor::from_vec(output_vec, &shape).unwrap();
        Variable::new(output, input.requires_grad())
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params.insert("bias".to_string(), self.bias.clone());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    fn name(&self) -> &'static str {
        "BatchNorm1d"
    }
}

// =============================================================================
// BatchNorm2d
// =============================================================================

/// Applies Batch Normalization over a 4D input (images).
///
/// # Shape
/// - Input: (N, C, H, W)
/// - Output: Same as input
pub struct BatchNorm2d {
    /// Learnable scale parameter (gamma).
    pub weight: Parameter,
    /// Learnable shift parameter (beta).
    pub bias: Parameter,
    /// Running mean for inference (updated during training).
    running_mean: RwLock<Tensor<f32>>,
    /// Running variance for inference (updated during training).
    running_var: RwLock<Tensor<f32>>,
    /// Number of features (channels).
    num_features: usize,
    /// Epsilon for numerical stability.
    eps: f32,
    /// Momentum for running stats update.
    momentum: f32,
    /// Whether in training mode.
    training: AtomicBool,
}

impl BatchNorm2d {
    /// Creates a new BatchNorm2d layer.
    pub fn new(num_features: usize) -> Self {
        Self::with_options(num_features, 1e-5, 0.1)
    }

    /// Creates a BatchNorm2d with custom options.
    pub fn with_options(num_features: usize, eps: f32, momentum: f32) -> Self {
        Self {
            weight: Parameter::named("weight", ones(&[num_features]), true),
            bias: Parameter::named("bias", zeros(&[num_features]), true),
            running_mean: RwLock::new(zeros(&[num_features])),
            running_var: RwLock::new(ones(&[num_features])),
            num_features,
            eps,
            momentum,
            training: AtomicBool::new(true),
        }
    }

    /// Returns the number of features (channels).
    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial_size = height * width;

        // Validate input matches expected channels
        assert_eq!(
            channels, self.num_features,
            "BatchNorm2d: expected {} channels, got {}",
            self.num_features, channels
        );

        let input_vec = input_data.to_vec();
        let weight_vec = self.weight.data().to_vec();
        let bias_vec = self.bias.data().to_vec();

        let is_training = self.training.load(Ordering::Relaxed);

        let mut means = vec![0.0f32; channels];
        let mut vars = vec![0.0f32; channels];

        if is_training {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx =
                                b * channels * spatial_size + c * spatial_size + h * width + w;
                            sum += input_vec[idx];
                        }
                    }
                }
                means[c] = sum / (batch_size * spatial_size) as f32;

                let mut var_sum = 0.0f32;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx =
                                b * channels * spatial_size + c * spatial_size + h * width + w;
                            let diff = input_vec[idx] - means[c];
                            var_sum += diff * diff;
                        }
                    }
                }
                vars[c] = var_sum / (batch_size * spatial_size) as f32;
            }

            // Update running statistics
            let mut running_mean = self.running_mean.write();
            let mut running_var = self.running_var.write();
            let running_mean_vec = running_mean.to_vec();
            let running_var_vec = running_var.to_vec();

            let new_mean: Vec<f32> = running_mean_vec
                .iter()
                .zip(means.iter())
                .map(|(&rm, &m)| (1.0 - self.momentum) * rm + self.momentum * m)
                .collect();
            let new_var: Vec<f32> = running_var_vec
                .iter()
                .zip(vars.iter())
                .map(|(&rv, &v)| (1.0 - self.momentum) * rv + self.momentum * v)
                .collect();

            *running_mean = Tensor::from_vec(new_mean, &[channels]).unwrap();
            *running_var = Tensor::from_vec(new_var, &[channels]).unwrap();
        } else {
            means = self.running_mean.read().to_vec();
            vars = self.running_var.read().to_vec();
        }

        let mut output_vec = vec![0.0f32; input_vec.len()];
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let idx = b * channels * spatial_size + c * spatial_size + h * width + w;
                        let normalized = (input_vec[idx] - means[c]) / (vars[c] + self.eps).sqrt();
                        output_vec[idx] = normalized * weight_vec[c] + bias_vec[c];
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_vec, &shape).unwrap();
        Variable::new(output, input.requires_grad())
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params.insert("bias".to_string(), self.bias.clone());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    fn name(&self) -> &'static str {
        "BatchNorm2d"
    }
}

// =============================================================================
// LayerNorm
// =============================================================================

/// Applies Layer Normalization over the last D dimensions.
///
/// y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
///
/// Unlike BatchNorm, LayerNorm normalizes over features, not batch.
pub struct LayerNorm {
    /// Learnable scale parameter (gamma).
    pub weight: Parameter,
    /// Learnable shift parameter (beta).
    pub bias: Parameter,
    /// Normalized shape.
    normalized_shape: Vec<usize>,
    /// Epsilon for numerical stability.
    eps: f32,
}

impl LayerNorm {
    /// Creates a new LayerNorm layer.
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self::with_eps(normalized_shape, 1e-5)
    }

    /// Creates a LayerNorm for a single dimension.
    pub fn single(size: usize) -> Self {
        Self::new(vec![size])
    }

    /// Creates a LayerNorm with custom epsilon.
    pub fn with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let numel: usize = normalized_shape.iter().product();
        Self {
            weight: Parameter::named("weight", ones(&[numel]), true),
            bias: Parameter::named("bias", zeros(&[numel]), true),
            normalized_shape,
            eps,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        let input_vec = input_data.to_vec();

        let weight_vec = self.weight.data().to_vec();
        let bias_vec = self.bias.data().to_vec();

        // Calculate the size of the normalized dimensions
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = input_vec.len() / norm_size;

        let mut output_vec = vec![0.0f32; input_vec.len()];

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &input_vec[start..end];

            // Calculate mean
            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;

            // Calculate variance
            let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;

            // Normalize and apply affine transform
            for i in 0..norm_size {
                let normalized = (slice[i] - mean) / (var + self.eps).sqrt();
                output_vec[start + i] = normalized * weight_vec[i] + bias_vec[i];
            }
        }

        let output = Tensor::from_vec(output_vec, &shape).unwrap();
        Variable::new(output, input.requires_grad())
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params.insert("bias".to_string(), self.bias.clone());
        params
    }

    fn name(&self) -> &'static str {
        "LayerNorm"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batchnorm1d() {
        let bn = BatchNorm1d::new(3);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
            false,
        );
        let output = bn.forward(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_batchnorm2d() {
        let bn = BatchNorm2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 32], &[2, 2, 2, 4]).unwrap(),
            false,
        );
        let output = bn.forward(&input);
        assert_eq!(output.shape(), vec![2, 2, 2, 4]);
    }

    #[test]
    fn test_layernorm() {
        let ln = LayerNorm::single(4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap(),
            false,
        );
        let output = ln.forward(&input);
        assert_eq!(output.shape(), vec![2, 4]);
    }

    #[test]
    fn test_batchnorm_parameters() {
        let bn = BatchNorm1d::new(10);
        assert_eq!(bn.parameters().len(), 2);
        assert_eq!(bn.num_parameters(), 20); // weight + bias
    }
}
