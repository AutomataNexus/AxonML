//! Activation Gradient Functions
//!
//! # File
//! `crates/axonml-autograd/src/functions/activation.rs`
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

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// ReLU Backward
// =============================================================================

/// Gradient function for `ReLU`.
///
/// d/dx(relu(x)) = 1 if x > 0, else 0
#[derive(Debug)]
pub struct ReluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
}

impl ReluBackward {
    /// Creates a new `ReluBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
        }
    }
}

impl GradientFunction for ReluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: use CUDA relu_backward kernel
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_input.device()).unwrap()
            };
            return vec![Some(grad_gpu.relu_backward_cuda(&self.saved_input))];
        }

        // CPU path: grad_input = grad_output * (input > 0)
        vec![Some(
            self.saved_input
                .zip_map(grad_output, |x, g| if x > 0.0 { g } else { 0.0 }),
        )]
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Sigmoid Backward
// =============================================================================

/// Gradient function for Sigmoid.
///
/// d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
#[derive(Debug)]
pub struct SigmoidBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
}

impl SigmoidBackward {
    /// Creates a new `SigmoidBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
        }
    }
}

impl GradientFunction for SigmoidBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path
        #[cfg(feature = "cuda")]
        if self.saved_output.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_output.device()).unwrap()
            };
            return vec![Some(grad_gpu.sigmoid_backward_cuda(&self.saved_output))];
        }

        // CPU path: grad = grad_output * output * (1 - output)
        vec![Some(
            self.saved_output
                .zip_map(grad_output, |o, g| g * o * (1.0 - o)),
        )]
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tanh Backward
// =============================================================================

/// Gradient function for Tanh.
///
/// d/dx(tanh(x)) = 1 - tanh(x)^2
#[derive(Debug)]
pub struct TanhBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
}

impl TanhBackward {
    /// Creates a new `TanhBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
        }
    }
}

impl GradientFunction for TanhBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path
        #[cfg(feature = "cuda")]
        if self.saved_output.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_output.device()).unwrap()
            };
            return vec![Some(grad_gpu.tanh_backward_cuda(&self.saved_output))];
        }

        // CPU path: grad = grad_output * (1 - output^2)
        vec![Some(
            self.saved_output
                .zip_map(grad_output, |o, g| g * (1.0 - o * o)),
        )]
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Softmax Backward
// =============================================================================

/// Gradient function for Softmax.
///
/// The Jacobian of softmax is: diag(s) - s * s^T
/// For element i: `ds_i/dx_j` = `s_i` * (`delta_ij` - `s_j`)
#[derive(Debug)]
pub struct SoftmaxBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
    dim: i64,
}

impl SoftmaxBackward {
    /// Creates a new `SoftmaxBackward`.
    ///
    /// # Arguments
    /// * `input_grad_fn` - The gradient function from the input
    /// * `output` - The softmax output (saved for backward computation)
    /// * `dim` - The dimension along which softmax was applied
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>, dim: i64) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
            dim,
        }
    }
}

impl GradientFunction for SoftmaxBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape = self.saved_output.shape();
        let ndim = shape.len();

        // Normalize dim to positive index
        let dim = if self.dim < 0 {
            (ndim as i64 + self.dim) as usize
        } else {
            self.dim as usize
        };

        // GPU fast path: use CUDA softmax_backward kernel (last dim only)
        #[cfg(feature = "cuda")]
        if self.saved_output.device().is_gpu() && dim == ndim - 1 {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_output.device()).unwrap()
            };
            return vec![Some(grad_gpu.softmax_backward_cuda(&self.saved_output))];
        }

        let s = self.saved_output.to_vec();
        let g = grad_output.to_vec();
        let mut result = vec![0.0f32; s.len()];

        if ndim == 1 {
            // 1D case: simple dot product
            let dot: f32 = s.iter().zip(g.iter()).map(|(&si, &gi)| si * gi).sum();
            for i in 0..s.len() {
                result[i] = s[i] * (g[i] - dot);
            }
        } else if ndim == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            if dim == 0 {
                // Softmax along rows (each column is independent)
                for col in 0..cols {
                    let mut dot = 0.0f32;
                    for row in 0..rows {
                        let idx = row * cols + col;
                        dot += s[idx] * g[idx];
                    }
                    for row in 0..rows {
                        let idx = row * cols + col;
                        result[idx] = s[idx] * (g[idx] - dot);
                    }
                }
            } else {
                // Softmax along columns (each row is independent) - most common case
                for row in 0..rows {
                    let start = row * cols;
                    let mut dot = 0.0f32;
                    for col in 0..cols {
                        let idx = start + col;
                        dot += s[idx] * g[idx];
                    }
                    for col in 0..cols {
                        let idx = start + col;
                        result[idx] = s[idx] * (g[idx] - dot);
                    }
                }
            }
        } else {
            // General N-D case: iterate over all "outer" positions (all dims except `dim`)
            // and compute softmax backward along each slice of the softmax dimension.
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            let dim_size = shape[dim];
            let dim_stride = strides[dim];
            let total = s.len();
            let outer_size = total / dim_size;

            // Build strides for the "outer" coordinate system (all dims except `dim`)
            let mut outer_dims: Vec<usize> = Vec::with_capacity(ndim - 1);
            let mut outer_strides: Vec<usize> = Vec::with_capacity(ndim - 1);
            for d in 0..ndim {
                if d != dim {
                    outer_dims.push(shape[d]);
                    outer_strides.push(strides[d]);
                }
            }

            // Precompute strides for outer_dims coordinate decomposition
            let mut outer_dim_strides = vec![1usize; outer_dims.len()];
            for i in (0..outer_dims.len().saturating_sub(1)).rev() {
                outer_dim_strides[i] = outer_dim_strides[i + 1] * outer_dims[i + 1];
            }

            for outer in 0..outer_size {
                // Decompose `outer` into coordinates for the non-dim dimensions
                let mut base_idx = 0;
                let mut temp = outer;
                for i in 0..outer_dims.len() {
                    let coord = temp / outer_dim_strides[i];
                    temp %= outer_dim_strides[i];
                    base_idx += coord * outer_strides[i];
                }

                // Compute dot product along this slice
                let mut dot = 0.0f32;
                for i in 0..dim_size {
                    let idx = base_idx + i * dim_stride;
                    if idx < total {
                        dot += s[idx] * g[idx];
                    }
                }

                // Compute gradient for this slice
                for i in 0..dim_size {
                    let idx = base_idx + i * dim_stride;
                    if idx < total {
                        result[idx] = s[idx] * (g[idx] - dot);
                    }
                }
            }
        }

        vec![Some(
            Tensor::from_vec(result, shape).expect("backward: tensor creation failed"),
        )]
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// LeakyReLU Backward
// =============================================================================

/// Gradient function for `LeakyReLU`.
///
/// `d/dx(leaky_relu(x))` = 1 if x > 0, else `negative_slope`
#[derive(Debug)]
pub struct LeakyReluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    negative_slope: f32,
}

impl LeakyReluBackward {
    /// Creates a new `LeakyReluBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>, negative_slope: f32) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
            negative_slope,
        }
    }
}

impl GradientFunction for LeakyReluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: use relu_backward as mask, blend with negative_slope
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_input.device()).unwrap()
            };
            // relu_backward gives: grad where input > 0, else 0
            let pos_grad = grad_gpu.relu_backward_cuda(&self.saved_input);
            // For negative part: (grad * negative_slope) where input <= 0
            // = grad * negative_slope - pos_grad * negative_slope + pos_grad
            // Simpler: pos_grad + (grad - pos_grad) * negative_slope
            //        = pos_grad * (1 - negative_slope) + grad * negative_slope
            let neg_part = grad_gpu.mul_scalar(self.negative_slope);
            let pos_part = pos_grad.mul_scalar(1.0 - self.negative_slope);
            let result = neg_part
                .add(&pos_part)
                .expect("backward: tensor add failed");
            return vec![Some(result)];
        }

        let ns = self.negative_slope;
        vec![Some(self.saved_input.zip_map(grad_output, move |x, g| {
            if x > 0.0 { g } else { g * ns }
        }))]
    }

    fn name(&self) -> &'static str {
        "LeakyReluBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// GELU Backward
// =============================================================================

/// Gradient function for GELU (Gaussian Error Linear Unit).
///
/// GELU(x) = x * Phi(x), where Phi is the CDF of standard normal.
/// Approximate: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[derive(Debug)]
pub struct GeluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
}

impl GeluBackward {
    /// Creates a new `GeluBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
        }
    }
}

impl GradientFunction for GeluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: compute gelu backward using tensor ops (all GPU-native)
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_input.device()).unwrap()
            };
            let x = &self.saved_input;
            // GELU(x) = 0.5 * x * (1 + tanh(inner))
            // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
            // d/dx = 0.5*(1+tanh(inner)) + 0.5*x*sech^2(inner)*d_inner
            // d_inner = sqrt(2/pi) * (1 + 3*0.044715*x^2)
            let sqrt_2_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
            let x2 = x.mul(x).expect("backward: tensor mul failed");
            let x3 = x2.mul(x).expect("backward: tensor mul failed");
            let inner = x
                .add(&x3.mul_scalar(0.044715))
                .unwrap()
                .mul_scalar(sqrt_2_pi);
            // tanh via tensor ops
            let tanh_inner = inner.tanh();
            // sech^2 = 1 - tanh^2
            let tanh2 = tanh_inner
                .mul(&tanh_inner)
                .expect("backward: tensor mul failed");
            let ones = Tensor::ones(x.shape());
            let ones_gpu = ones.to_device(x.device()).unwrap();
            let sech2 = ones_gpu.sub(&tanh2).expect("backward: tensor sub failed");
            // d_inner = sqrt(2/pi) * (1 + 3*0.044715*x^2)
            let d_inner = ones_gpu
                .add(&x2.mul_scalar(3.0 * 0.044715))
                .unwrap()
                .mul_scalar(sqrt_2_pi);
            // 0.5*(1+tanh) + 0.5*x*sech2*d_inner
            let term1 = ones_gpu
                .add(&tanh_inner)
                .expect("backward: tensor add failed")
                .mul_scalar(0.5);
            let term2 = x
                .mul(&sech2)
                .unwrap()
                .mul(&d_inner)
                .unwrap()
                .mul_scalar(0.5);
            let deriv = term1.add(&term2).expect("backward: tensor add failed");
            return vec![Some(
                grad_gpu.mul(&deriv).expect("backward: tensor mul failed"),
            )];
        }

        let sqrt_2_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let c = 0.044715_f32;

        vec![Some(self.saved_input.zip_map(grad_output, move |x, g| {
            let x3 = x * x * x;
            let inner = sqrt_2_pi * (x + c * x3);
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let d_inner = sqrt_2_pi * (1.0 + 3.0 * c * x * x);
            g * (0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner)
        }))]
    }

    fn name(&self) -> &'static str {
        "GeluBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Exp Backward
// =============================================================================

/// Gradient function for element-wise exponential.
///
/// d/dx(exp(x)) = exp(x) = output
#[derive(Debug)]
pub struct ExpBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
}

impl ExpBackward {
    /// Creates a new `ExpBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
        }
    }
}

impl GradientFunction for ExpBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // d/dx(exp(x)) = exp(x) = output → element-wise multiply (GPU-native)
        vec![Some(
            grad_output
                .mul(&self.saved_output)
                .expect("backward: tensor mul failed"),
        )]
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Log Backward
// =============================================================================

/// Gradient function for element-wise natural logarithm.
///
/// d/dx(log(x)) = 1/x
#[derive(Debug)]
pub struct LogBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
}

impl LogBackward {
    /// Creates a new `LogBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
        }
    }
}

impl GradientFunction for LogBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // d/dx(log(x)) = 1/x → element-wise divide (GPU-native)
        vec![Some(grad_output.div(&self.saved_input).unwrap())]
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Clamp Backward
// =============================================================================

/// Gradient function for element-wise clamp.
///
/// Gradient passes through where input is not clamped, zero elsewhere.
#[derive(Debug)]
pub struct ClampBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    min_val: f32,
    max_val: f32,
}

impl ClampBackward {
    /// Creates a new `ClampBackward`.
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input: Tensor<f32>,
        min_val: f32,
        max_val: f32,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
            min_val,
            max_val,
        }
    }
}

impl GradientFunction for ClampBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: gradient passes through where input is not clamped
        // clamp(x, min, max) has gradient 1 where min < x < max, 0 otherwise
        // = relu_backward(x - min) * relu_backward(max - x) * grad  (approximately)
        // Simpler: use (x - min).relu_backward * grad, then (max - x).relu_backward * that
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_input.device()).unwrap()
            };
            // Mask: input > min_val → relu_backward(input - min_val) gives 1 where true
            let shifted_low = self.saved_input.add_scalar(-self.min_val);
            let mask_low = grad_gpu.relu_backward_cuda(&shifted_low);
            // Mask: input < max_val → relu_backward(max_val - input) gives 1 where true
            let shifted_high = self.saved_input.mul_scalar(-1.0).add_scalar(self.max_val);
            let result = mask_low.relu_backward_cuda(&shifted_high);
            return vec![Some(result)];
        }

        // CPU path
        let min_v = self.min_val;
        let max_v = self.max_val;
        vec![Some(self.saved_input.zip_map(grad_output, move |x, g| {
            if x > min_v && x < max_v { g } else { 0.0 }
        }))]
    }

    fn name(&self) -> &'static str {
        "ClampBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// LogSoftmax Backward
// =============================================================================

/// Gradient function for LogSoftmax.
///
/// d/dx(log_softmax(x)) = grad_output - softmax(x) * sum(grad_output, dim)
#[derive(Debug)]
pub struct LogSoftmaxBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
    dim: i64,
}

impl LogSoftmaxBackward {
    /// Creates a new `LogSoftmaxBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>, dim: i64) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
            dim,
        }
    }
}

impl GradientFunction for LogSoftmaxBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape = self.saved_output.shape();
        let ndim = shape.len();

        let dim = if self.dim < 0 {
            (ndim as i64 + self.dim) as usize
        } else {
            self.dim as usize
        };

        // GPU fast path: grad_input = grad_output - softmax * sum(grad_output, dim)
        // softmax = exp(log_softmax_output)
        // Uses tensor ops which all dispatch to GPU natively
        #[cfg(feature = "cuda")]
        if self.saved_output.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_output.device()).unwrap()
            };
            // softmax = exp(output) — GPU-native
            let softmax = self.saved_output.exp();
            // sum_g = grad_output.sum(dim, keepdim=true) — stays on GPU via tensor ops
            let sum_g = grad_gpu.sum_dim(dim as i32, true);
            // grad_input = grad_output - softmax * sum_g (broadcast mul + sub)
            let scaled = softmax.mul(&sum_g).expect("backward: tensor mul failed");
            let result = grad_gpu.sub(&scaled).expect("backward: tensor sub failed");
            return vec![Some(result)];
        }

        let output_vec = self.saved_output.to_vec();
        let g = grad_output.to_vec();
        let mut result = vec![0.0f32; output_vec.len()];

        // softmax = exp(log_softmax) = exp(output)
        // grad_input = grad_output - softmax * sum(grad_output, dim)
        if ndim == 1 {
            let sum_g: f32 = g.iter().sum();
            for i in 0..output_vec.len() {
                let softmax_i = output_vec[i].exp();
                result[i] = g[i] - softmax_i * sum_g;
            }
        } else if ndim == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            if dim == 1 {
                for row in 0..rows {
                    let start = row * cols;
                    let mut sum_g = 0.0f32;
                    for col in 0..cols {
                        sum_g += g[start + col];
                    }
                    for col in 0..cols {
                        let idx = start + col;
                        let softmax_i = output_vec[idx].exp();
                        result[idx] = g[idx] - softmax_i * sum_g;
                    }
                }
            } else {
                for col in 0..cols {
                    let mut sum_g = 0.0f32;
                    for row in 0..rows {
                        sum_g += g[row * cols + col];
                    }
                    for row in 0..rows {
                        let idx = row * cols + col;
                        let softmax_i = output_vec[idx].exp();
                        result[idx] = g[idx] - softmax_i * sum_g;
                    }
                }
            }
        } else {
            // General N-D case
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            let dim_size = shape[dim];
            let dim_stride = strides[dim];
            let total = output_vec.len();
            let outer_size = total / dim_size;

            for outer in 0..outer_size {
                let mut base_idx = 0;
                let mut temp = outer;
                for d in (0..ndim).rev() {
                    if d != dim {
                        let _s = if d > dim {
                            strides[d]
                        } else {
                            strides[d] / dim_size
                        };
                        let coord = temp % shape[d];
                        temp /= shape[d];
                        base_idx += coord * strides[d];
                    }
                }

                let mut sum_g = 0.0f32;
                for i in 0..dim_size {
                    let idx = base_idx + i * dim_stride;
                    if idx < total {
                        sum_g += g[idx];
                    }
                }

                for i in 0..dim_size {
                    let idx = base_idx + i * dim_stride;
                    if idx < total {
                        let softmax_i = output_vec[idx].exp();
                        result[idx] = g[idx] - softmax_i * sum_g;
                    }
                }
            }
        }

        vec![Some(
            Tensor::from_vec(result, shape).expect("backward: tensor creation failed"),
        )]
    }

    fn name(&self) -> &'static str {
        "LogSoftmaxBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// SiLU Backward
// =============================================================================

/// Gradient function for SiLU/Swish activation (x * sigmoid(x)).
///
/// d/dx(x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
///                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#[derive(Debug)]
pub struct SiluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
}

impl SiluBackward {
    /// Creates a new `SiluBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
        }
    }
}

impl GradientFunction for SiluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: compute silu backward using tensor ops (all GPU-native)
        // SiLU(x) = x * sigmoid(x)
        // d/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_input.device()).unwrap()
            };
            let x = &self.saved_input;
            let sig = x.sigmoid();
            let ones = Tensor::ones(x.shape()).to_device(x.device()).unwrap();
            let one_minus_sig = ones.sub(&sig).expect("backward: tensor sub failed");
            let x_term = x.mul(&one_minus_sig).expect("backward: tensor mul failed");
            let bracket = ones.add(&x_term).expect("backward: tensor add failed");
            let deriv = sig.mul(&bracket).expect("backward: tensor mul failed");
            return vec![Some(
                grad_gpu.mul(&deriv).expect("backward: tensor mul failed"),
            )];
        }

        vec![Some(self.saved_input.zip_map(grad_output, |x, g| {
            let sig = 1.0 / (1.0 + (-x).exp());
            g * (sig + x * sig * (1.0 - sig))
        }))]
    }

    fn name(&self) -> &'static str {
        "SiluBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Sqrt Backward
// =============================================================================

/// Gradient function for element-wise square root.
///
/// d/dx(sqrt(x)) = 0.5 / sqrt(x)
#[derive(Debug)]
pub struct SqrtBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_output: Tensor<f32>,
}

impl SqrtBackward {
    /// Creates a new `SqrtBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, output: Tensor<f32>) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_output: output,
        }
    }
}

impl GradientFunction for SqrtBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // d/dx(sqrt(x)) = 0.5 / sqrt(x) = 0.5 / output → GPU-native via tensor ops
        // grad_output / (2 * output)
        let two_output = self.saved_output.mul_scalar(2.0);
        vec![Some(grad_output.div(&two_output).unwrap())]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// ELU Backward
// =============================================================================

/// Gradient function for ELU (Exponential Linear Unit).
///
/// ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
/// d/dx ELU(x) = 1 if x > 0, else alpha * exp(x)
#[derive(Debug)]
pub struct EluBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    alpha: f32,
}

impl EluBackward {
    /// Creates a new `EluBackward`.
    #[must_use]
    pub fn new(input_grad_fn: Option<GradFn>, input: Tensor<f32>, alpha: f32) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            saved_input: input,
            alpha,
        }
    }
}

impl GradientFunction for EluBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // GPU fast path: elu'(x) = 1 if x > 0, alpha*exp(x) if x <= 0
        // = relu_mask * 1 + (1 - relu_mask) * alpha*exp(x)
        // Use relu_backward to get the mask effect
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_input.device()).unwrap()
            };
            // pos_grad = grad where input > 0, else 0
            let pos_grad = grad_gpu.relu_backward_cuda(&self.saved_input);
            // neg_part = alpha * exp(x) * grad where input <= 0
            // = (grad - pos_grad) * alpha * exp(x)
            let neg_grad = grad_gpu
                .sub(&pos_grad)
                .expect("backward: tensor sub failed");
            let exp_x = self.saved_input.exp();
            let neg_result = neg_grad
                .mul(&exp_x)
                .expect("backward: tensor mul failed")
                .mul_scalar(self.alpha);
            let result = pos_grad
                .add(&neg_result)
                .expect("backward: tensor add failed");
            return vec![Some(result)];
        }

        let alpha = self.alpha;
        vec![Some(self.saved_input.zip_map(grad_output, move |x, g| {
            if x > 0.0 { g } else { g * alpha * x.exp() }
        }))]
    }

    fn name(&self) -> &'static str {
        "EluBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_backward() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4])
            .expect("backward: tensor creation failed");
        let grad_fn = ReluBackward::new(None, input);

        let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])
            .expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        // Gradient is 0 where input <= 0, 1 where input > 0
        assert_eq!(
            grads[0].as_ref().unwrap().to_vec(),
            vec![0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn test_sigmoid_backward() {
        // sigmoid(0) = 0.5, derivative at 0 is 0.5 * 0.5 = 0.25
        let output = Tensor::from_vec(vec![0.5], &[1]).expect("backward: tensor creation failed");
        let grad_fn = SigmoidBackward::new(None, output);

        let grad_output =
            Tensor::from_vec(vec![1.0], &[1]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        assert!((grads[0].as_ref().unwrap().to_vec()[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_backward() {
        // tanh(0) = 0, derivative at 0 is 1 - 0^2 = 1
        let output = Tensor::from_vec(vec![0.0], &[1]).expect("backward: tensor creation failed");
        let grad_fn = TanhBackward::new(None, output);

        let grad_output =
            Tensor::from_vec(vec![1.0], &[1]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        assert!((grads[0].as_ref().unwrap().to_vec()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_backward() {
        let input =
            Tensor::from_vec(vec![-1.0, 1.0], &[2]).expect("backward: tensor creation failed");
        let grad_fn = LeakyReluBackward::new(None, input, 0.01);

        let grad_output =
            Tensor::from_vec(vec![1.0, 1.0], &[2]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        let result = grads[0].as_ref().unwrap().to_vec();
        assert!((result[0] - 0.01).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exp_backward() {
        // exp([0, 1, 2]) = [1, e, e^2]
        let output = Tensor::from_vec(
            vec![
                1.0,
                std::f32::consts::E,
                std::f32::consts::E * std::f32::consts::E,
            ],
            &[3],
        )
        .unwrap();
        let grad_fn = ExpBackward::new(None, output);

        let grad_output =
            Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        let result = grads[0].as_ref().unwrap().to_vec();
        // d/dx(exp(x)) = exp(x)
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn test_log_backward() {
        let input =
            Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3]).expect("backward: tensor creation failed");
        let grad_fn = LogBackward::new(None, input);

        let grad_output =
            Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        let result = grads[0].as_ref().unwrap().to_vec();
        // d/dx(log(x)) = 1/x
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_backward() {
        let input =
            Tensor::from_vec(vec![-1.0, 0.5, 2.0], &[3]).expect("backward: tensor creation failed");
        let grad_fn = ClampBackward::new(None, input, 0.0, 1.0);

        let grad_output =
            Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).expect("backward: tensor creation failed");
        let grads = grad_fn.apply(&grad_output);

        let result = grads[0].as_ref().unwrap().to_vec();
        // Gradient is 0 where clamped, 1 where not
        assert_eq!(result[0], 0.0); // clamped at min
        assert_eq!(result[1], 1.0); // not clamped
        assert_eq!(result[2], 0.0); // clamped at max
    }
}
