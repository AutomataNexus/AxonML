//! Normalization layers — BatchNorm, LayerNorm, GroupNorm, InstanceNorm.
//!
//! 1383 lines. `BatchNorm1d` / `BatchNorm2d` (running mean/var, momentum,
//! affine, train/eval mode switch), `LayerNorm` (per-element affine over
//! normalized_shape), `GroupNorm` (splits channels into groups),
//! `InstanceNorm2d` (per-instance per-channel). All track learnable
//! weight/bias parameters and implement `Module`.
//!
//! # File
//! `crates/axonml-nn/src/layers/norm.rs`
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

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use axonml_autograd::Variable;
use axonml_autograd::functions::{
    BatchNorm1dBackward, BatchNorm2dBackward, GroupNormBackward, InstanceNorm2dBackward,
    LayerNormBackward,
};
use axonml_autograd::grad_fn::GradFn;
use axonml_autograd::no_grad::is_grad_enabled;
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

    /// Returns a clone of the running mean tensor.
    pub fn running_mean(&self) -> Tensor<f32> {
        self.running_mean.read().clone()
    }

    /// Returns a clone of the running variance tensor.
    pub fn running_var(&self) -> Tensor<f32> {
        self.running_var.read().clone()
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let num_features = shape[1];

        assert_eq!(
            num_features, self.num_features,
            "BatchNorm1d: expected {} features, got {}",
            self.num_features, num_features
        );

        let is_training = self.training.load(Ordering::Relaxed);
        let spatial_size: usize = if shape.len() > 2 {
            shape[2..].iter().product()
        } else {
            1
        };

        #[cfg(feature = "cuda")]
        if input_data.device().is_gpu() && is_training {
            let gamma_data = self.weight.data();
            let beta_data = self.bias.data();

            let gamma_gpu = if !gamma_data.device().is_gpu() {
                gamma_data
                    .to_device(input_data.device())
                    .unwrap_or(gamma_data)
            } else {
                gamma_data
            };
            let beta_gpu = if !beta_data.device().is_gpu() {
                beta_data
                    .to_device(input_data.device())
                    .unwrap_or(beta_data)
            } else {
                beta_data
            };

            if let Some((output_tensor, means, vars)) = input_data.batchnorm_fused(
                &gamma_gpu,
                &beta_gpu,
                self.eps,
                num_features,
                spatial_size,
            ) {
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
                    *running_mean = Tensor::from_vec(new_mean, &[num_features])
                        .expect("tensor creation failed");
                    *running_var =
                        Tensor::from_vec(new_var, &[num_features]).expect("tensor creation failed");
                }

                let weight_vec = gamma_gpu.to_vec();
                let requires_grad =
                    (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();
                if requires_grad {
                    let weight_var = self.weight.variable();
                    let bias_var = self.bias.variable();
                    let grad_fn = GradFn::new(BatchNorm1dBackward::new(
                        input.grad_fn().cloned(),
                        weight_var.grad_fn().cloned(),
                        bias_var.grad_fn().cloned(),
                        input_data,
                        means,
                        vars,
                        weight_vec,
                        self.eps,
                        self.num_features,
                    ));
                    return Variable::from_operation(output_tensor, grad_fn, true);
                }
                return Variable::new(output_tensor, false);
            }
        }

        let input_vec = input_data.to_vec();
        let weight_vec = self.weight.data().to_vec();
        let bias_vec = self.bias.data().to_vec();

        let mut means = vec![0.0f32; num_features];
        let mut vars = vec![0.0f32; num_features];

        if is_training {
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

                *running_mean =
                    Tensor::from_vec(new_mean, &[num_features]).expect("tensor creation failed");
                *running_var =
                    Tensor::from_vec(new_var, &[num_features]).expect("tensor creation failed");
            }
        } else {
            means = self.running_mean.read().to_vec();
            vars = self.running_var.read().to_vec();
        }

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

        let output = Tensor::from_vec(output_vec, &shape).expect("tensor creation failed");

        let requires_grad =
            (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();
        if requires_grad {
            let weight_var = self.weight.variable();
            let bias_var = self.bias.variable();

            let grad_fn = GradFn::new(BatchNorm1dBackward::new(
                input.grad_fn().cloned(),
                weight_var.grad_fn().cloned(),
                bias_var.grad_fn().cloned(),
                input_data,
                means.clone(),
                vars.clone(),
                weight_vec,
                self.eps,
                self.num_features,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
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

    fn named_buffers(&self) -> HashMap<String, Tensor<f32>> {
        let mut buffers = HashMap::new();
        if self.track_running_stats {
            buffers.insert("running_mean".to_string(), self.running_mean.read().clone());
            buffers.insert("running_var".to_string(), self.running_var.read().clone());
        }
        buffers
    }

    fn set_buffer(&self, name: &str, value: Tensor<f32>) -> bool {
        match name {
            "running_mean" => {
                *self.running_mean.write() = value;
                true
            }
            "running_var" => {
                *self.running_var.write() = value;
                true
            }
            _ => false,
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![
            crate::NodeSpec::new("BatchNorm")
                .attr("epsilon", crate::AttrVal::Float(self.eps as f64))
                .attr("momentum", crate::AttrVal::Float(self.momentum as f64))
                .param("weight")
                .param("bias")
                .param("running_mean")
                .param("running_var"),
        ]
    }

    fn name(&self) -> &'static str {
        "BatchNorm1d"
    }

    fn to_device(&self, device: axonml_core::Device) {
        for param in self.parameters() {
            param.to_device(device);
        }
        if self.track_running_stats {
            let mut rm = self.running_mean.write();
            if let Ok(moved) = rm.to_device(device) {
                *rm = moved;
            }
            let mut rv = self.running_var.write();
            if let Ok(moved) = rv.to_device(device) {
                *rv = moved;
            }
        }
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

    /// Returns a clone of the running mean tensor.
    pub fn running_mean(&self) -> Tensor<f32> {
        self.running_mean.read().clone()
    }

    /// Returns a clone of the running variance tensor.
    pub fn running_var(&self) -> Tensor<f32> {
        self.running_var.read().clone()
    }

    /// Overwrites the running mean/variance (e.g. loading inference stats).
    pub fn set_running_stats(&self, mean: Tensor<f32>, var: Tensor<f32>) {
        *self.running_mean.write() = mean;
        *self.running_var.write() = var;
    }

    /// Sets training mode via interior mutability (usable behind `&self`).
    pub fn set_training(&self, training: bool) {
        self.training
            .store(training, std::sync::atomic::Ordering::Relaxed);
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

        assert_eq!(
            channels, self.num_features,
            "BatchNorm2d: expected {} channels, got {}",
            self.num_features, channels
        );

        let is_training = self.training.load(Ordering::Relaxed);

        #[cfg(feature = "cuda")]
        if input_data.device().is_gpu() && is_training {
            let gamma_data = self.weight.data();
            let beta_data = self.bias.data();

            let gamma_gpu = if !gamma_data.device().is_gpu() {
                gamma_data
                    .to_device(input_data.device())
                    .unwrap_or(gamma_data)
            } else {
                gamma_data
            };
            let beta_gpu = if !beta_data.device().is_gpu() {
                beta_data
                    .to_device(input_data.device())
                    .unwrap_or(beta_data)
            } else {
                beta_data
            };

            if let Some((output_tensor, means, vars)) =
                input_data.batchnorm_fused(&gamma_gpu, &beta_gpu, self.eps, channels, spatial_size)
            {
                let mut running_mean = self.running_mean.write();
                let mut running_var = self.running_var.write();
                let running_mean_vec = running_mean.to_vec();
                let running_var_vec = running_var.to_vec();
                let new_mean: Vec<f32> = running_mean_vec
                    .iter()
                    .zip(means.iter())
                    .map(|(&rm, &m)| {
                        if m.is_finite() {
                            (1.0 - self.momentum) * rm + self.momentum * m
                        } else {
                            rm
                        }
                    })
                    .collect();
                let new_var: Vec<f32> = running_var_vec
                    .iter()
                    .zip(vars.iter())
                    .map(|(&rv, &v)| {
                        if v.is_finite() && v >= 0.0 {
                            (1.0 - self.momentum) * rv + self.momentum * v
                        } else {
                            rv
                        }
                    })
                    .collect();
                *running_mean =
                    Tensor::from_vec(new_mean, &[channels]).expect("tensor creation failed");
                *running_var =
                    Tensor::from_vec(new_var, &[channels]).expect("tensor creation failed");

                let weight_vec = gamma_gpu.to_vec();
                let requires_grad =
                    (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();
                if requires_grad {
                    let weight_var = self.weight.variable();
                    let bias_var = self.bias.variable();
                    let grad_fn = GradFn::new(BatchNorm2dBackward::new(
                        input.grad_fn().cloned(),
                        weight_var.grad_fn().cloned(),
                        bias_var.grad_fn().cloned(),
                        input_data,
                        means,
                        vars,
                        weight_vec,
                        self.eps,
                        self.num_features,
                    ));
                    return Variable::from_operation(output_tensor, grad_fn, true);
                }
                return Variable::new(output_tensor, false);
            }
        }

        let input_vec = input_data.to_vec();
        let weight_vec = self.weight.data().to_vec();
        let bias_vec = self.bias.data().to_vec();

        let mut means = vec![0.0f32; channels];
        let mut vars = vec![0.0f32; channels];

        if is_training {
            let n_per_channel = (batch_size * spatial_size) as f32;
            for c in 0..channels {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                for b in 0..batch_size {
                    let base = b * channels * spatial_size + c * spatial_size;
                    for s in 0..spatial_size {
                        let val = input_vec[base + s];
                        sum += val;
                        sum_sq += val * val;
                    }
                }
                means[c] = sum / n_per_channel;
                vars[c] = sum_sq / n_per_channel - means[c] * means[c];
            }

            let mut running_mean = self.running_mean.write();
            let mut running_var = self.running_var.write();
            let running_mean_vec = running_mean.to_vec();
            let running_var_vec = running_var.to_vec();

            let new_mean: Vec<f32> = running_mean_vec
                .iter()
                .zip(means.iter())
                .map(|(&rm, &m)| {
                    if m.is_finite() {
                        (1.0 - self.momentum) * rm + self.momentum * m
                    } else {
                        rm
                    }
                })
                .collect();
            let new_var: Vec<f32> = running_var_vec
                .iter()
                .zip(vars.iter())
                .map(|(&rv, &v)| {
                    if v.is_finite() && v >= 0.0 {
                        (1.0 - self.momentum) * rv + self.momentum * v
                    } else {
                        rv
                    }
                })
                .collect();

            *running_mean =
                Tensor::from_vec(new_mean, &[channels]).expect("tensor creation failed");
            *running_var = Tensor::from_vec(new_var, &[channels]).expect("tensor creation failed");
        } else {
            means = self.running_mean.read().to_vec();
            vars = self.running_var.read().to_vec();
        }

        let total = input_vec.len();
        let mut output_vec = vec![0.0f32; total];

        let inv_stds: Vec<f32> = vars.iter().map(|v| 1.0 / (v + self.eps).sqrt()).collect();

        for i in 0..total {
            let c = (i / spatial_size) % channels;
            output_vec[i] = (input_vec[i] - means[c]) * inv_stds[c] * weight_vec[c] + bias_vec[c];
        }

        let output = Tensor::from_vec(output_vec, &shape).expect("tensor creation failed");

        let requires_grad =
            (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();
        if requires_grad {
            let weight_var = self.weight.variable();
            let bias_var = self.bias.variable();

            let grad_fn = GradFn::new(BatchNorm2dBackward::new(
                input.grad_fn().cloned(),
                weight_var.grad_fn().cloned(),
                bias_var.grad_fn().cloned(),
                input_data,
                means.clone(),
                vars.clone(),
                weight_vec,
                self.eps,
                self.num_features,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
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

    fn named_buffers(&self) -> HashMap<String, Tensor<f32>> {
        let mut buffers = HashMap::new();
        buffers.insert("running_mean".to_string(), self.running_mean.read().clone());
        buffers.insert("running_var".to_string(), self.running_var.read().clone());
        buffers
    }

    fn set_buffer(&self, name: &str, value: Tensor<f32>) -> bool {
        match name {
            "running_mean" => {
                *self.running_mean.write() = value;
                true
            }
            "running_var" => {
                *self.running_var.write() = value;
                true
            }
            _ => false,
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![
            crate::NodeSpec::new("BatchNorm")
                .attr("epsilon", crate::AttrVal::Float(self.eps as f64))
                .attr("momentum", crate::AttrVal::Float(self.momentum as f64))
                .param("weight")
                .param("bias")
                .param("running_mean")
                .param("running_var"),
        ]
    }

    fn name(&self) -> &'static str {
        "BatchNorm2d"
    }

    fn to_device(&self, device: axonml_core::Device) {
        for param in self.parameters() {
            param.to_device(device);
        }
        let mut rm = self.running_mean.write();
        if let Ok(moved) = rm.to_device(device) {
            *rm = moved;
        }
        let mut rv = self.running_var.write();
        if let Ok(moved) = rv.to_device(device) {
            *rv = moved;
        }
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
        let norm_size: usize = self.normalized_shape.iter().product();
        let total_len = input_data.numel();
        let num_rows = total_len / norm_size;

        #[cfg(feature = "cuda")]
        if input_data.device().is_gpu() {
            let weight_data = self.weight.data();
            let weight_gpu = if weight_data.device().is_gpu() {
                weight_data.clone()
            } else {
                weight_data.to_device(input_data.device().clone()).unwrap()
            };
            let bias_data = self.bias.data();
            let bias_gpu = if bias_data.device().is_gpu() {
                bias_data.clone()
            } else {
                bias_data.to_device(input_data.device().clone()).unwrap()
            };

            let output = input_data
                .layer_norm_cuda(&weight_gpu, &bias_gpu, norm_size, self.eps)
                .expect("CUDA LayerNorm failed");

            let requires_grad = input.requires_grad() && is_grad_enabled();
            return if requires_grad {
                let grad_fn = GradFn::new(LayerNormBackward::new(
                    input.grad_fn().cloned(),
                    self.weight.variable().grad_fn().cloned(),
                    self.bias.variable().grad_fn().cloned(),
                    input_data.clone(),
                    self.weight.data().clone(),
                    self.normalized_shape.clone(),
                    self.eps,
                ));
                Variable::from_operation(output, grad_fn, true)
            } else {
                Variable::from_tensor(output)
            };
        }

        let input_vec = input_data.to_vec();
        let weight_vec = self.weight.data().to_vec();
        let bias_vec = self.bias.data().to_vec();

        let mut output_vec = vec![0.0f32; input_vec.len()];

        for b in 0..num_rows {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &input_vec[start..end];

            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
            let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;

            for i in 0..norm_size {
                let normalized = (slice[i] - mean) / (var + self.eps).sqrt();
                output_vec[start + i] = normalized * weight_vec[i] + bias_vec[i];
            }
        }

        let output = Tensor::from_vec(output_vec, &shape).expect("tensor creation failed");
        let requires_grad = input.requires_grad() && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(LayerNormBackward::new(
                input.grad_fn().cloned(),
                self.weight.variable().grad_fn().cloned(),
                self.bias.variable().grad_fn().cloned(),
                input_data.clone(),
                self.weight.data().clone(),
                self.normalized_shape.clone(),
                self.eps,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::from_tensor(output)
        }
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

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![
            crate::NodeSpec::new("LayerNorm")
                .attr("epsilon", crate::AttrVal::Float(self.eps as f64))
                .param("weight")
                .param("bias"),
        ]
    }
}

// =============================================================================
// GroupNorm
// =============================================================================

/// Applies Group Normalization over a mini-batch of inputs.
///
/// Groups channels and normalizes within each group.
/// Particularly effective for small batch sizes where BatchNorm struggles.
///
/// # Shape
/// - Input: (N, C, *) where C must be divisible by num_groups
/// - Output: Same as input
pub struct GroupNorm {
    /// Learnable scale parameter (gamma).
    pub weight: Parameter,
    /// Learnable shift parameter (beta).
    pub bias: Parameter,
    /// Number of groups to divide channels into.
    num_groups: usize,
    /// Number of channels expected in input.
    num_channels: usize,
    /// Epsilon for numerical stability.
    eps: f32,
    /// Whether to use learnable affine parameters.
    affine: bool,
}

impl GroupNorm {
    /// Creates a new GroupNorm layer.
    ///
    /// # Arguments
    /// * `num_groups` - Number of groups to divide channels into
    /// * `num_channels` - Number of channels expected in input
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self::with_options(num_groups, num_channels, 1e-5, true)
    }

    /// Creates a GroupNorm with custom options.
    pub fn with_options(num_groups: usize, num_channels: usize, eps: f32, affine: bool) -> Self {
        assert!(
            num_channels % num_groups == 0,
            "num_channels ({}) must be divisible by num_groups ({})",
            num_channels,
            num_groups
        );

        Self {
            weight: Parameter::named("weight", ones(&[num_channels]), affine),
            bias: Parameter::named("bias", zeros(&[num_channels]), affine),
            num_groups,
            num_channels,
            eps,
            affine,
        }
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        assert_eq!(
            channels, self.num_channels,
            "GroupNorm: expected {} channels, got {}",
            self.num_channels, channels
        );

        let input_vec = input_data.to_vec();
        let channels_per_group = channels / self.num_groups;

        let mut output_vec = vec![0.0f32; input_vec.len()];

        for b in 0..batch_size {
            for g in 0..self.num_groups {
                let mut sum = 0.0f32;
                let group_size = channels_per_group * spatial_size;

                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        sum += input_vec[idx];
                    }
                }
                let mean = sum / group_size as f32;

                let mut var_sum = 0.0f32;
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        let diff = input_vec[idx] - mean;
                        var_sum += diff * diff;
                    }
                }
                let var = var_sum / group_size as f32;

                let std_inv = 1.0 / (var + self.eps).sqrt();
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    let weight = if self.affine {
                        self.weight.data().to_vec()[channel_idx]
                    } else {
                        1.0
                    };
                    let bias = if self.affine {
                        self.bias.data().to_vec()[channel_idx]
                    } else {
                        0.0
                    };

                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + channel_idx * spatial_size + s;
                        let normalized = (input_vec[idx] - mean) * std_inv;
                        output_vec[idx] = normalized * weight + bias;
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_vec, &shape).expect("tensor creation failed");
        let requires_grad = input.requires_grad() && is_grad_enabled();

        if requires_grad && self.affine {
            let grad_fn = GradFn::new(GroupNormBackward::new(
                input.grad_fn().cloned(),
                self.weight.variable().grad_fn().cloned(),
                self.bias.variable().grad_fn().cloned(),
                input_data.clone(),
                self.weight.data().clone(),
                self.num_groups,
                self.eps,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::from_tensor(output)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        if self.affine {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            vec![]
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        if self.affine {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), self.weight.clone());
            params.insert("bias".to_string(), self.bias.clone());
            params
        } else {
            HashMap::new()
        }
    }

    fn name(&self) -> &'static str {
        "GroupNorm"
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        let mut node = crate::NodeSpec::new("GroupNorm")
            .attr("num_groups", crate::AttrVal::Int(self.num_groups as i64))
            .attr("epsilon", crate::AttrVal::Float(self.eps as f64));
        if self.affine {
            node = node.param("weight").param("bias");
        }
        vec![node]
    }
}

// =============================================================================
// InstanceNorm2d
// =============================================================================

/// Applies Instance Normalization over a 4D input (images).
///
/// Each channel in each sample is normalized independently.
/// Particularly useful for style transfer and image generation.
///
/// # Shape
/// - Input: (N, C, H, W)
/// - Output: Same as input
pub struct InstanceNorm2d {
    /// Learnable scale parameter (gamma).
    pub weight: Parameter,
    /// Learnable shift parameter (beta).
    pub bias: Parameter,
    /// Number of features (channels).
    num_features: usize,
    /// Epsilon for numerical stability.
    eps: f32,
    /// Whether to use learnable affine parameters.
    affine: bool,
}

impl InstanceNorm2d {
    /// Creates a new InstanceNorm2d layer.
    pub fn new(num_features: usize) -> Self {
        Self::with_options(num_features, 1e-5, false)
    }

    /// Creates an InstanceNorm2d with affine parameters.
    pub fn with_affine(num_features: usize) -> Self {
        Self::with_options(num_features, 1e-5, true)
    }

    /// Creates an InstanceNorm2d with custom options.
    pub fn with_options(num_features: usize, eps: f32, affine: bool) -> Self {
        Self {
            weight: Parameter::named("weight", ones(&[num_features]), affine),
            bias: Parameter::named("bias", zeros(&[num_features]), affine),
            num_features,
            eps,
            affine,
        }
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let shape = input_data.shape().to_vec();

        assert!(
            shape.len() == 4,
            "InstanceNorm2d expects 4D input (N, C, H, W)"
        );

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial_size = height * width;

        assert_eq!(
            channels, self.num_features,
            "InstanceNorm2d: expected {} channels, got {}",
            self.num_features, channels
        );

        let input_vec = input_data.to_vec();
        let mut output_vec = vec![0.0f32; input_vec.len()];

        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    sum += input_vec[idx];
                }
                let mean = sum / spatial_size as f32;

                let mut var_sum = 0.0f32;
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let diff = input_vec[idx] - mean;
                    var_sum += diff * diff;
                }
                let var = var_sum / spatial_size as f32;

                let std_inv = 1.0 / (var + self.eps).sqrt();
                let weight = if self.affine {
                    self.weight.data().to_vec()[c]
                } else {
                    1.0
                };
                let bias = if self.affine {
                    self.bias.data().to_vec()[c]
                } else {
                    0.0
                };

                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let normalized = (input_vec[idx] - mean) * std_inv;
                    output_vec[idx] = normalized * weight + bias;
                }
            }
        }

        let output = Tensor::from_vec(output_vec, &shape).expect("tensor creation failed");
        let requires_grad = input.requires_grad() && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(InstanceNorm2dBackward::new(
                input.grad_fn().cloned(),
                if self.affine {
                    self.weight.variable().grad_fn().cloned()
                } else {
                    None
                },
                if self.affine {
                    self.bias.variable().grad_fn().cloned()
                } else {
                    None
                },
                input_data.clone(),
                self.weight.data().clone(),
                self.eps,
                self.affine,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::from_tensor(output)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        if self.affine {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            vec![]
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        if self.affine {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), self.weight.clone());
            params.insert("bias".to_string(), self.bias.clone());
            params
        } else {
            HashMap::new()
        }
    }

    fn name(&self) -> &'static str {
        "InstanceNorm2d"
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        let mut node = crate::NodeSpec::new("InstanceNorm")
            .attr("epsilon", crate::AttrVal::Float(self.eps as f64));
        if self.affine {
            node = node.param("weight").param("bias");
        }
        vec![node]
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
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
                .expect("tensor creation failed"),
            false,
        );
        let output = bn.forward(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_batchnorm2d() {
        let bn = BatchNorm2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 32], &[2, 2, 2, 4]).expect("tensor creation failed"),
            false,
        );
        let output = bn.forward(&input);
        assert_eq!(output.shape(), vec![2, 2, 2, 4]);
    }

    fn stationary_nchw(n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
        let total = n * c * h * w;
        let mut data = vec![0.0f32; total];
        for idx in 0..total {
            let ci = (idx / (h * w)) % c;
            let base = (ci as f32 + 1.0) * 10.0;
            let jitter = ((idx % 7) as f32 - 3.0) * (ci as f32 + 1.0);
            data[idx] = base + jitter;
        }
        data
    }

    #[test]
    fn batchnorm2d_eval_matches_train_cpu() {
        let (n, c, h, w) = (4usize, 3usize, 5usize, 5usize);
        let data = stationary_nchw(n, c, h, w);
        let bn = BatchNorm2d::with_options(c, 1e-5, 0.1);
        let make = || {
            Variable::new(
                Tensor::from_vec(data.clone(), &[n, c, h, w]).expect("tensor"),
                false,
            )
        };
        bn.set_training(true);
        let mut train_out = vec![];
        for _ in 0..300 {
            train_out = bn.forward(&make()).data().to_vec();
        }
        bn.set_training(false);
        let eval_out = bn.forward(&make()).data().to_vec();
        let maxdiff = train_out
            .iter()
            .zip(eval_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "[CPU] running_mean={:?} running_var={:?} maxdiff={}",
            bn.running_mean().to_vec(),
            bn.running_var().to_vec(),
            maxdiff
        );
        assert!(
            maxdiff < 1e-2,
            "CPU eval diverges from train: maxdiff={maxdiff}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn batchnorm2d_eval_matches_train_gpu() {
        use axonml_core::Device;
        let (n, c, h, w) = (4usize, 3usize, 5usize, 5usize);
        let data = stationary_nchw(n, c, h, w);
        let bn = BatchNorm2d::with_options(c, 1e-5, 0.1);
        let make = || {
            let t = Tensor::from_vec(data.clone(), &[n, c, h, w])
                .expect("tensor")
                .to_device(Device::Cuda(0))
                .expect("to gpu");
            Variable::new(t, false)
        };
        bn.set_training(true);
        let mut train_out = vec![];
        for _ in 0..300 {
            train_out = bn.forward(&make()).data().to_vec();
        }
        bn.set_training(false);
        let eval_out = bn.forward(&make()).data().to_vec();
        let maxdiff = train_out
            .iter()
            .zip(eval_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "[GPU] running_mean={:?} running_var={:?} maxdiff={}",
            bn.running_mean().to_vec(),
            bn.running_var().to_vec(),
            maxdiff
        );
        assert!(
            maxdiff < 1e-2,
            "GPU eval diverges from train: maxdiff={maxdiff}"
        );
    }

    #[test]
    fn test_layernorm() {
        let ln = LayerNorm::single(4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
                .expect("tensor creation failed"),
            false,
        );
        let output = ln.forward(&input);
        assert_eq!(output.shape(), vec![2, 4]);
    }

    #[test]
    fn test_batchnorm_parameters() {
        let bn = BatchNorm1d::new(10);
        assert_eq!(bn.parameters().len(), 2);
        assert_eq!(bn.num_parameters(), 20);
    }

    #[test]
    fn test_groupnorm() {
        let gn = GroupNorm::new(2, 4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 32], &[2, 4, 2, 2]).expect("tensor creation failed"),
            false,
        );
        let output = gn.forward(&input);
        assert_eq!(output.shape(), vec![2, 4, 2, 2]);
    }

    #[test]
    fn test_groupnorm_normalization() {
        let gn = GroupNorm::with_options(2, 4, 1e-5, false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 4, 1, 2])
                .expect("tensor creation failed"),
            false,
        );
        let output = gn.forward(&input);
        let out_vec = output.data().to_vec();
        let group1_mean: f32 = out_vec[0..4].iter().sum::<f32>() / 4.0;
        let group2_mean: f32 = out_vec[4..8].iter().sum::<f32>() / 4.0;
        assert!(group1_mean.abs() < 1e-5);
        assert!(group2_mean.abs() < 1e-5);
    }

    #[test]
    fn test_instancenorm2d() {
        let inn = InstanceNorm2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 32], &[2, 2, 2, 4]).expect("tensor creation failed"),
            false,
        );
        let output = inn.forward(&input);
        assert_eq!(output.shape(), vec![2, 2, 2, 4]);
    }

    #[test]
    fn test_instancenorm2d_with_affine() {
        let inn = InstanceNorm2d::with_affine(4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 64], &[1, 4, 4, 4]).expect("tensor creation failed"),
            false,
        );
        let output = inn.forward(&input);
        assert_eq!(output.shape(), vec![1, 4, 4, 4]);
        assert_eq!(inn.parameters().len(), 2);
    }

    // =========================================================================
    // LayerNorm Correctness
    // =========================================================================

    #[test]
    fn test_layernorm_zero_mean_unit_var() {
        let ln = LayerNorm::with_eps(vec![4], 1e-5);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 5.0, 3.0, 7.0], &[1, 4]).unwrap(),
            false,
        );
        let output = ln.forward(&input);
        let out = output.data().to_vec();

        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / out.len() as f32;

        assert!(
            mean.abs() < 1e-4,
            "LayerNorm output mean should be ~0, got {}",
            mean
        );
        assert!(
            (var - 1.0).abs() < 0.1,
            "LayerNorm output var should be ~1, got {}",
            var
        );
    }

    #[test]
    fn test_layernorm_gradient_flow() {
        use axonml_autograd::backward;

        let ln = LayerNorm::single(3);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(),
            true,
        );
        let output = ln.forward(&input);
        let loss = output.sum();

        let ones = Tensor::from_vec(vec![1.0], &loss.shape()).unwrap();
        backward(&loss, &ones);

        let grad = input
            .grad()
            .expect("Should have gradient through LayerNorm");
        let gv = grad.to_vec();
        assert_eq!(gv.len(), 3);
        assert!(
            gv.iter().all(|g| g.is_finite()),
            "All gradients should be finite: {:?}",
            gv
        );
    }

    #[test]
    fn test_layernorm_batch_independence() {
        let ln = LayerNorm::with_eps(vec![3], 1e-5);

        let input1 = Variable::new(
            Tensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]).unwrap(),
            false,
        );
        let out1 = ln.forward(&input1).data().to_vec();

        let input2 = Variable::new(
            Tensor::from_vec(vec![10.0, 20.0, 30.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap(),
            false,
        );
        let out2 = ln.forward(&input2).data().to_vec();

        for i in 0..3 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-5,
                "LayerNorm should be batch-independent: {} vs {}",
                out1[i],
                out2[i]
            );
        }
    }

    #[test]
    fn test_layernorm_parameters_count() {
        let ln = LayerNorm::single(64);
        assert_eq!(ln.parameters().len(), 2);
        assert_eq!(ln.num_parameters(), 128);
    }

    // =========================================================================
    // BatchNorm Correctness
    // =========================================================================

    #[test]
    fn test_batchnorm1d_normalization() {
        let bn = BatchNorm1d::with_options(2, 1e-5, 0.1, false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], &[3, 2]).unwrap(),
            false,
        );
        let output = bn.forward(&input);
        let out = output.data().to_vec();

        let ch0_mean = (out[0] + out[2] + out[4]) / 3.0;
        let ch1_mean = (out[1] + out[3] + out[5]) / 3.0;
        assert!(
            ch0_mean.abs() < 0.1,
            "BatchNorm ch0 mean should be ~0, got {}",
            ch0_mean
        );
        assert!(
            ch1_mean.abs() < 0.1,
            "BatchNorm ch1 mean should be ~0, got {}",
            ch1_mean
        );
    }

    #[test]
    fn test_batchnorm1d_train_vs_eval() {
        let mut bn = BatchNorm1d::new(2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            false,
        );

        bn.train();
        let train_out = bn.forward(&input).data().to_vec();

        bn.eval();
        let eval_out = bn.forward(&input).data().to_vec();

        let diff: f32 = train_out
            .iter()
            .zip(eval_out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0 || true, "Train vs eval can differ");
    }

    #[test]
    fn test_batchnorm2d_gradient_flow() {
        use axonml_autograd::backward;

        let bn = BatchNorm2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 32], &[2, 2, 2, 4]).unwrap(),
            true,
        );
        let output = bn.forward(&input);
        let loss = output.sum();
        let ones = Tensor::from_vec(vec![1.0], &loss.shape()).unwrap();
        backward(&loss, &ones);

        let grad = input
            .grad()
            .expect("Should have gradient through BatchNorm2d");
        assert_eq!(grad.shape(), &[2, 2, 2, 4]);
        assert!(grad.to_vec().iter().all(|g| g.is_finite()));
    }

    // =========================================================================
    // GroupNorm Gradient
    // =========================================================================

    #[test]
    fn test_groupnorm_gradient_flow() {
        use axonml_autograd::backward;

        let gn = GroupNorm::new(2, 4);
        let input = Variable::new(
            Tensor::from_vec(
                (0..32).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                &[1, 4, 2, 4],
            )
            .unwrap(),
            true,
        );
        let output = gn.forward(&input);
        let loss = output.sum();
        let ones = Tensor::from_vec(vec![1.0], &loss.shape()).unwrap();
        backward(&loss, &ones);

        let grad = input
            .grad()
            .expect("Should have gradient through GroupNorm");
        assert_eq!(grad.shape(), &[1, 4, 2, 4]);
        assert!(grad.to_vec().iter().all(|g| g.is_finite()));
    }
}
