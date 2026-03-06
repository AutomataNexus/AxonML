//! Convolution Gradient Functions
//!
//! Gradient functions for Conv1d, Conv2d, ConvTranspose2d,
//! and all pooling operations.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::any::Any;

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// Conv2d Backward
// =============================================================================

/// Gradient function for 2D convolution.
///
/// For output = conv2d(input, weight, bias):
/// - d_input = full convolution of grad_output with weight (flipped)
/// - d_weight = correlation of input with grad_output
/// - d_bias = sum of grad_output over (N, H_out, W_out)
#[derive(Debug)]
pub struct Conv2dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_weight: Tensor<f32>,
    input_shape: Vec<usize>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    has_bias: bool,
}

impl Conv2dBackward {
    /// Creates a new Conv2dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<Option<GradFn>>,
        saved_input: Tensor<f32>,
        saved_weight: Tensor<f32>,
        input_shape: Vec<usize>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        has_bias: bool,
    ) -> Self {
        let mut next_fns = vec![input_grad_fn, weight_grad_fn];
        if let Some(bias_fn) = bias_grad_fn {
            next_fns.push(bias_fn);
        }
        Self {
            next_fns,
            saved_input,
            saved_weight,
            input_shape,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            has_bias,
        }
    }
}

impl GradientFunction for Conv2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_out_vec = grad_output.to_vec();
        let grad_out_shape = grad_output.shape();
        let batch_size = grad_out_shape[0];
        let out_h = grad_out_shape[2];
        let out_w = grad_out_shape[3];

        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();

        // === d_input: "full" convolution with flipped weights ===
        // d_input[n,ic,ih,iw] += grad_out[n,oc,oh,ow] * weight[oc,ic,kh_i,kw_j]
        // where ih = oh*sh + ki - ph, iw = ow*sw + kj - pw
        let mut grad_input = vec![0.0f32; batch_size * self.in_channels * in_h * in_w];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let go_idx = b * self.out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow;
                        let go_val = grad_out_vec[go_idx];

                        for ic in 0..self.in_channels {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih_signed =
                                        (oh * sh + ki) as isize - ph as isize;
                                    let iw_signed =
                                        (ow * sw + kj) as isize - pw as isize;

                                    if ih_signed >= 0
                                        && (ih_signed as usize) < in_h
                                        && iw_signed >= 0
                                        && (iw_signed as usize) < in_w
                                    {
                                        let ih = ih_signed as usize;
                                        let iw = iw_signed as usize;
                                        let in_idx = b * self.in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + ih * in_w
                                            + iw;
                                        let w_idx = oc * self.in_channels * kh * kw
                                            + ic * kh * kw
                                            + ki * kw
                                            + kj;
                                        grad_input[in_idx] += go_val * weight_vec[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // === d_weight: correlation of input with grad_output ===
        // d_weight[oc,ic,ki,kj] += input[n,ic,oh*sh+ki-ph,ow*sw+kj-pw] * grad_out[n,oc,oh,ow]
        let mut grad_weight =
            vec![0.0f32; self.out_channels * self.in_channels * kh * kw];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let go_idx = b * self.out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow;
                        let go_val = grad_out_vec[go_idx];

                        for ic in 0..self.in_channels {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih_signed =
                                        (oh * sh + ki) as isize - ph as isize;
                                    let iw_signed =
                                        (ow * sw + kj) as isize - pw as isize;

                                    if ih_signed >= 0
                                        && (ih_signed as usize) < in_h
                                        && iw_signed >= 0
                                        && (iw_signed as usize) < in_w
                                    {
                                        let ih = ih_signed as usize;
                                        let iw = iw_signed as usize;
                                        let in_idx = b * self.in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + ih * in_w
                                            + iw;
                                        let w_idx = oc * self.in_channels * kh * kw
                                            + ic * kh * kw
                                            + ki * kw
                                            + kj;
                                        grad_weight[w_idx] += input_vec[in_idx] * go_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_input_tensor =
            Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        let grad_weight_tensor = Tensor::from_vec(
            grad_weight,
            &[self.out_channels, self.in_channels, kh, kw],
        )
        .unwrap();

        let mut result = vec![Some(grad_input_tensor), Some(grad_weight_tensor)];

        // === d_bias: sum over (N, H_out, W_out) ===
        if self.has_bias {
            let mut grad_bias = vec![0.0f32; self.out_channels];
            for b in 0..batch_size {
                for oc in 0..self.out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let go_idx = b * self.out_channels * out_h * out_w
                                + oc * out_h * out_w
                                + oh * out_w
                                + ow;
                            grad_bias[oc] += grad_out_vec[go_idx];
                        }
                    }
                }
            }
            result.push(Some(
                Tensor::from_vec(grad_bias, &[self.out_channels]).unwrap(),
            ));
        }

        result
    }

    fn name(&self) -> &'static str {
        "Conv2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Grouped Conv2d Backward (depthwise separable, etc.)
// =============================================================================

/// Gradient function for grouped 2D convolution.
///
/// Handles groups > 1 (depthwise separable convolutions, etc.).
/// For groups=G: in_channels and out_channels are split into G groups,
/// each group operates independently.
///
/// Weight shape: (out_channels, in_channels/groups, kh, kw)
#[derive(Debug)]
pub struct GroupedConv2dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_weight: Tensor<f32>,
    input_shape: Vec<usize>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
    has_bias: bool,
}

impl GroupedConv2dBackward {
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<Option<GradFn>>,
        saved_input: Tensor<f32>,
        saved_weight: Tensor<f32>,
        input_shape: Vec<usize>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
        has_bias: bool,
    ) -> Self {
        let mut next_fns = vec![input_grad_fn, weight_grad_fn];
        if let Some(bias_fn) = bias_grad_fn {
            next_fns.push(bias_fn);
        }
        Self {
            next_fns, saved_input, saved_weight, input_shape,
            in_channels, out_channels, kernel_size, stride, padding,
            groups, has_bias,
        }
    }
}

impl GradientFunction for GroupedConv2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_out_vec = grad_output.to_vec();
        let grad_out_shape = grad_output.shape();
        let batch_size = grad_out_shape[0];
        let out_h = grad_out_shape[2];
        let out_w = grad_out_shape[3];

        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();

        let ic_per_group = self.in_channels / self.groups;
        let oc_per_group = self.out_channels / self.groups;

        // === d_input ===
        let mut grad_input = vec![0.0f32; batch_size * self.in_channels * in_h * in_w];

        for b in 0..batch_size {
            for g in 0..self.groups {
                let ic_start = g * ic_per_group;
                let oc_start = g * oc_per_group;

                for oc_local in 0..oc_per_group {
                    let oc = oc_start + oc_local;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let go_idx = b * self.out_channels * out_h * out_w
                                + oc * out_h * out_w + oh * out_w + ow;
                            let go_val = grad_out_vec[go_idx];

                            for ic_local in 0..ic_per_group {
                                let ic = ic_start + ic_local;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih_s = (oh * sh + ki) as isize - ph as isize;
                                        let iw_s = (ow * sw + kj) as isize - pw as isize;
                                        if ih_s >= 0 && (ih_s as usize) < in_h
                                            && iw_s >= 0 && (iw_s as usize) < in_w
                                        {
                                            let ih = ih_s as usize;
                                            let iw = iw_s as usize;
                                            let in_idx = b * self.in_channels * in_h * in_w
                                                + ic * in_h * in_w + ih * in_w + iw;
                                            // weight: (out_channels, ic_per_group, kh, kw)
                                            let w_idx = oc * ic_per_group * kh * kw
                                                + ic_local * kh * kw + ki * kw + kj;
                                            grad_input[in_idx] += go_val * weight_vec[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // === d_weight ===
        let mut grad_weight = vec![0.0f32; self.out_channels * ic_per_group * kh * kw];

        for b in 0..batch_size {
            for g in 0..self.groups {
                let ic_start = g * ic_per_group;
                let oc_start = g * oc_per_group;

                for oc_local in 0..oc_per_group {
                    let oc = oc_start + oc_local;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let go_idx = b * self.out_channels * out_h * out_w
                                + oc * out_h * out_w + oh * out_w + ow;
                            let go_val = grad_out_vec[go_idx];

                            for ic_local in 0..ic_per_group {
                                let ic = ic_start + ic_local;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih_s = (oh * sh + ki) as isize - ph as isize;
                                        let iw_s = (ow * sw + kj) as isize - pw as isize;
                                        if ih_s >= 0 && (ih_s as usize) < in_h
                                            && iw_s >= 0 && (iw_s as usize) < in_w
                                        {
                                            let ih = ih_s as usize;
                                            let iw = iw_s as usize;
                                            let in_idx = b * self.in_channels * in_h * in_w
                                                + ic * in_h * in_w + ih * in_w + iw;
                                            let w_idx = oc * ic_per_group * kh * kw
                                                + ic_local * kh * kw + ki * kw + kj;
                                            grad_weight[w_idx] += input_vec[in_idx] * go_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_input_tensor = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        let grad_weight_tensor = Tensor::from_vec(
            grad_weight, &[self.out_channels, ic_per_group, kh, kw],
        ).unwrap();

        let mut result = vec![Some(grad_input_tensor), Some(grad_weight_tensor)];

        // === d_bias ===
        if self.has_bias {
            let mut grad_bias = vec![0.0f32; self.out_channels];
            for b in 0..batch_size {
                for oc in 0..self.out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let go_idx = b * self.out_channels * out_h * out_w
                                + oc * out_h * out_w + oh * out_w + ow;
                            grad_bias[oc] += grad_out_vec[go_idx];
                        }
                    }
                }
            }
            result.push(Some(Tensor::from_vec(grad_bias, &[self.out_channels]).unwrap()));
        }

        result
    }

    fn name(&self) -> &'static str {
        "GroupedConv2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// BatchNorm2d Backward
// =============================================================================

/// Gradient function for 2D batch normalization.
///
/// For y = (x - mean) / sqrt(var + eps) * weight + bias:
/// - d_input = weight / sqrt(var+eps) * (grad - mean(grad) - (x-mean)*mean(grad*(x-mean)) / (var+eps))
/// - d_weight = sum(grad * (x - mean) / sqrt(var + eps))
/// - d_bias = sum(grad)
#[derive(Debug)]
pub struct BatchNorm2dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_mean: Vec<f32>,
    saved_var: Vec<f32>,
    saved_weight: Vec<f32>,
    eps: f32,
    num_features: usize,
}

impl BatchNorm2dBackward {
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<GradFn>,
        saved_input: Tensor<f32>,
        saved_mean: Vec<f32>,
        saved_var: Vec<f32>,
        saved_weight: Vec<f32>,
        eps: f32,
        num_features: usize,
    ) -> Self {
        let next_fns = vec![input_grad_fn, weight_grad_fn, bias_grad_fn];
        Self {
            next_fns, saved_input, saved_mean, saved_var,
            saved_weight, eps, num_features,
        }
    }
}

impl GradientFunction for BatchNorm2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_vec = grad_output.to_vec();
        let shape = grad_output.shape();
        let batch = shape[0];
        let channels = shape[1];
        let h = shape[2];
        let w = shape[3];
        let spatial = h * w;
        let n = (batch * spatial) as f32; // number of elements per channel

        let input_vec = self.saved_input.to_vec();

        let mut grad_input = vec![0.0f32; grad_vec.len()];
        let mut grad_weight = vec![0.0f32; channels];
        let mut grad_bias = vec![0.0f32; channels];

        for c in 0..channels {
            let mean_c = self.saved_mean[c];
            let var_c = self.saved_var[c];
            let std_inv = 1.0 / (var_c + self.eps).sqrt();
            let weight_c = self.saved_weight[c];

            // Accumulate grad_bias = sum(grad), grad_weight = sum(grad * x_hat)
            // Also compute sum_grad and sum_grad_xhat for d_input
            let mut sum_grad = 0.0f32;
            let mut sum_grad_xhat = 0.0f32;

            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = (input_vec[idx] - mean_c) * std_inv;
                    let g = grad_vec[idx];

                    grad_bias[c] += g;
                    grad_weight[c] += g * x_hat;

                    sum_grad += g;
                    sum_grad_xhat += g * x_hat;
                }
            }

            // d_input = weight * std_inv / N * (N * grad - sum_grad - x_hat * sum_grad_xhat)
            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = (input_vec[idx] - mean_c) * std_inv;
                    let g = grad_vec[idx];

                    grad_input[idx] = weight_c * std_inv / n
                        * (n * g - sum_grad - x_hat * sum_grad_xhat);
                }
            }
        }

        let grad_input_tensor = Tensor::from_vec(grad_input, &shape.to_vec()).unwrap();
        let grad_weight_tensor = Tensor::from_vec(grad_weight, &[channels]).unwrap();
        let grad_bias_tensor = Tensor::from_vec(grad_bias, &[channels]).unwrap();

        vec![
            Some(grad_input_tensor),
            Some(grad_weight_tensor),
            Some(grad_bias_tensor),
        ]
    }

    fn name(&self) -> &'static str {
        "BatchNorm2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// BatchNorm1d Backward
// =============================================================================

/// Gradient function for 1D batch normalization.
///
/// Same math as `BatchNorm2dBackward` but handles (N, C) or (N, C, L) inputs.
#[derive(Debug)]
pub struct BatchNorm1dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_mean: Vec<f32>,
    saved_var: Vec<f32>,
    saved_weight: Vec<f32>,
    eps: f32,
    num_features: usize,
}

impl BatchNorm1dBackward {
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<GradFn>,
        saved_input: Tensor<f32>,
        saved_mean: Vec<f32>,
        saved_var: Vec<f32>,
        saved_weight: Vec<f32>,
        eps: f32,
        num_features: usize,
    ) -> Self {
        let next_fns = vec![input_grad_fn, weight_grad_fn, bias_grad_fn];
        Self {
            next_fns, saved_input, saved_mean, saved_var,
            saved_weight, eps, num_features,
        }
    }
}

impl GradientFunction for BatchNorm1dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_vec = grad_output.to_vec();
        let shape = grad_output.shape();
        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = if shape.len() > 2 {
            shape[2..].iter().product()
        } else {
            1
        };
        let n = (batch * spatial) as f32;

        let input_vec = self.saved_input.to_vec();

        let mut grad_input = vec![0.0f32; grad_vec.len()];
        let mut grad_weight = vec![0.0f32; channels];
        let mut grad_bias = vec![0.0f32; channels];

        for c in 0..channels {
            let mean_c = self.saved_mean[c];
            let var_c = self.saved_var[c];
            let std_inv = 1.0 / (var_c + self.eps).sqrt();
            let weight_c = self.saved_weight[c];

            let mut sum_grad = 0.0f32;
            let mut sum_grad_xhat = 0.0f32;

            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = (input_vec[idx] - mean_c) * std_inv;
                    let g = grad_vec[idx];

                    grad_bias[c] += g;
                    grad_weight[c] += g * x_hat;

                    sum_grad += g;
                    sum_grad_xhat += g * x_hat;
                }
            }

            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = (input_vec[idx] - mean_c) * std_inv;
                    let g = grad_vec[idx];

                    grad_input[idx] = weight_c * std_inv / n
                        * (n * g - sum_grad - x_hat * sum_grad_xhat);
                }
            }
        }

        let grad_input_tensor = Tensor::from_vec(grad_input, &shape.to_vec()).unwrap();
        let grad_weight_tensor = Tensor::from_vec(grad_weight, &[channels]).unwrap();
        let grad_bias_tensor = Tensor::from_vec(grad_bias, &[channels]).unwrap();

        vec![
            Some(grad_input_tensor),
            Some(grad_weight_tensor),
            Some(grad_bias_tensor),
        ]
    }

    fn name(&self) -> &'static str {
        "BatchNorm1dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Conv1d Backward
// =============================================================================

/// Gradient function for 1D convolution.
#[derive(Debug)]
pub struct Conv1dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_weight: Tensor<f32>,
    input_shape: Vec<usize>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    has_bias: bool,
}

impl Conv1dBackward {
    /// Creates a new Conv1dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<Option<GradFn>>,
        saved_input: Tensor<f32>,
        saved_weight: Tensor<f32>,
        input_shape: Vec<usize>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        has_bias: bool,
    ) -> Self {
        let mut next_fns = vec![input_grad_fn, weight_grad_fn];
        if let Some(bias_fn) = bias_grad_fn {
            next_fns.push(bias_fn);
        }
        Self {
            next_fns,
            saved_input,
            saved_weight,
            input_shape,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            has_bias,
        }
    }
}

impl GradientFunction for Conv1dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_out_vec = grad_output.to_vec();
        let grad_out_shape = grad_output.shape();
        let batch_size = grad_out_shape[0];
        let out_length = grad_out_shape[2];

        let in_length = self.input_shape[2];
        let ks = self.kernel_size;

        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();

        // d_input
        let mut grad_input = vec![0.0f32; batch_size * self.in_channels * in_length];
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let go_idx =
                        b * self.out_channels * out_length + oc * out_length + ol;
                    let go_val = grad_out_vec[go_idx];

                    for ic in 0..self.in_channels {
                        for k in 0..ks {
                            let il_signed =
                                (ol * self.stride + k) as isize - self.padding as isize;
                            if il_signed >= 0 && (il_signed as usize) < in_length {
                                let il = il_signed as usize;
                                let in_idx =
                                    b * self.in_channels * in_length + ic * in_length + il;
                                let w_idx =
                                    oc * self.in_channels * ks + ic * ks + k;
                                grad_input[in_idx] += go_val * weight_vec[w_idx];
                            }
                        }
                    }
                }
            }
        }

        // d_weight
        let mut grad_weight = vec![0.0f32; self.out_channels * self.in_channels * ks];
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let go_idx =
                        b * self.out_channels * out_length + oc * out_length + ol;
                    let go_val = grad_out_vec[go_idx];

                    for ic in 0..self.in_channels {
                        for k in 0..ks {
                            let il_signed =
                                (ol * self.stride + k) as isize - self.padding as isize;
                            if il_signed >= 0 && (il_signed as usize) < in_length {
                                let il = il_signed as usize;
                                let in_idx =
                                    b * self.in_channels * in_length + ic * in_length + il;
                                let w_idx =
                                    oc * self.in_channels * ks + ic * ks + k;
                                grad_weight[w_idx] += input_vec[in_idx] * go_val;
                            }
                        }
                    }
                }
            }
        }

        let grad_input_tensor =
            Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        let grad_weight_tensor = Tensor::from_vec(
            grad_weight,
            &[self.out_channels, self.in_channels, ks],
        )
        .unwrap();

        let mut result = vec![Some(grad_input_tensor), Some(grad_weight_tensor)];

        if self.has_bias {
            let mut grad_bias = vec![0.0f32; self.out_channels];
            for b in 0..batch_size {
                for oc in 0..self.out_channels {
                    for ol in 0..out_length {
                        let go_idx =
                            b * self.out_channels * out_length + oc * out_length + ol;
                        grad_bias[oc] += grad_out_vec[go_idx];
                    }
                }
            }
            result.push(Some(
                Tensor::from_vec(grad_bias, &[self.out_channels]).unwrap(),
            ));
        }

        result
    }

    fn name(&self) -> &'static str {
        "Conv1dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// MaxPool2d Backward
// =============================================================================

/// Gradient function for 2D max pooling.
///
/// Gradient flows only to the position of the max value in each pooling window.
#[derive(Debug)]
pub struct MaxPool2dBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    /// Flat index in the input for each output position.
    max_indices: Vec<usize>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2dBackward {
    /// Creates a new MaxPool2dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        max_indices: Vec<usize>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            max_indices,
            kernel_size,
            stride,
            padding,
        }
    }
}

impl GradientFunction for MaxPool2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let in_numel: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; in_numel];
        let grad_out_vec = grad_output.to_vec();

        // Scatter gradient to max positions
        for (out_idx, &in_idx) in self.max_indices.iter().enumerate() {
            if in_idx < in_numel {
                grad_input[in_idx] += grad_out_vec[out_idx];
            }
        }

        let grad = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "MaxPool2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// MaxPool1d Backward
// =============================================================================

/// Gradient function for 1D max pooling.
#[derive(Debug)]
pub struct MaxPool1dBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    max_indices: Vec<usize>,
}

impl MaxPool1dBackward {
    /// Creates a new MaxPool1dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        max_indices: Vec<usize>,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            max_indices,
        }
    }
}

impl GradientFunction for MaxPool1dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let in_numel: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; in_numel];
        let grad_out_vec = grad_output.to_vec();

        for (out_idx, &in_idx) in self.max_indices.iter().enumerate() {
            if in_idx < in_numel {
                grad_input[in_idx] += grad_out_vec[out_idx];
            }
        }

        let grad = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "MaxPool1dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// AvgPool2d Backward
// =============================================================================

/// Gradient function for 2D average pooling.
///
/// Each input element that contributed to the average gets an equal share of the gradient.
#[derive(Debug)]
pub struct AvgPool2dBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2dBackward {
    /// Creates a new AvgPool2dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            kernel_size,
            stride,
            padding,
        }
    }
}

impl GradientFunction for AvgPool2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_out_vec = grad_output.to_vec();
        let grad_out_shape = grad_output.shape();
        let batch = grad_out_shape[0];
        let channels = grad_out_shape[1];
        let out_h = grad_out_shape[2];
        let out_w = grad_out_shape[3];

        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let mut grad_input = vec![0.0f32; batch * channels * in_h * in_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Count valid elements in this pooling window
                        let mut count = 0;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                if ih >= ph && ih < in_h + ph && iw >= pw && iw < in_w + pw {
                                    count += 1;
                                }
                            }
                        }

                        let go_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        let go_val = grad_out_vec[go_idx];
                        let grad_per_elem = if count > 0 { go_val / count as f32 } else { 0.0 };

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                if ih >= ph && ih < in_h + ph && iw >= pw && iw < in_w + pw {
                                    let actual_ih = ih - ph;
                                    let actual_iw = iw - pw;
                                    let in_idx = b * channels * in_h * in_w
                                        + c * in_h * in_w
                                        + actual_ih * in_w
                                        + actual_iw;
                                    grad_input[in_idx] += grad_per_elem;
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "AvgPool2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// AvgPool1d Backward
// =============================================================================

/// Gradient function for 1D average pooling.
#[derive(Debug)]
pub struct AvgPool1dBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool1dBackward {
    /// Creates a new AvgPool1dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            kernel_size,
            stride,
            padding,
        }
    }
}

impl GradientFunction for AvgPool1dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_out_vec = grad_output.to_vec();
        let grad_out_shape = grad_output.shape();
        let batch = grad_out_shape[0];
        let channels = grad_out_shape[1];
        let out_length = grad_out_shape[2];

        let in_length = self.input_shape[2];

        let mut grad_input = vec![0.0f32; batch * channels * in_length];

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_length {
                    let in_start = ol * self.stride;
                    let mut count = 0;
                    for k in 0..self.kernel_size {
                        let il = in_start + k;
                        if il >= self.padding && il < in_length + self.padding {
                            count += 1;
                        }
                    }

                    let go_idx =
                        b * channels * out_length + c * out_length + ol;
                    let go_val = grad_out_vec[go_idx];
                    let grad_per_elem = if count > 0 { go_val / count as f32 } else { 0.0 };

                    for k in 0..self.kernel_size {
                        let il = in_start + k;
                        if il >= self.padding && il < in_length + self.padding {
                            let actual_il = il - self.padding;
                            let in_idx =
                                b * channels * in_length + c * in_length + actual_il;
                            grad_input[in_idx] += grad_per_elem;
                        }
                    }
                }
            }
        }

        let grad = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "AvgPool1dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// AdaptiveAvgPool2d Backward
// =============================================================================

/// Gradient function for adaptive 2D average pooling.
#[derive(Debug)]
pub struct AdaptiveAvgPool2dBackward {
    next_fns: Vec<Option<GradFn>>,
    input_shape: Vec<usize>,
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2dBackward {
    /// Creates a new AdaptiveAvgPool2dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        input_shape: Vec<usize>,
        output_size: (usize, usize),
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn],
            input_shape,
            output_size,
        }
    }
}

impl GradientFunction for AdaptiveAvgPool2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let grad_out_vec = grad_output.to_vec();
        let batch = self.input_shape[0];
        let channels = self.input_shape[1];
        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];
        let (out_h, out_w) = self.output_size;

        let mut grad_input = vec![0.0f32; batch * channels * in_h * in_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih_start = (oh * in_h) / out_h;
                        let ih_end = ((oh + 1) * in_h) / out_h;
                        let iw_start = (ow * in_w) / out_w;
                        let iw_end = ((ow + 1) * in_w) / out_w;

                        let count = (ih_end - ih_start) * (iw_end - iw_start);
                        let go_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        let go_val = grad_out_vec[go_idx];
                        let grad_per_elem = if count > 0 { go_val / count as f32 } else { 0.0 };

                        for ih in ih_start..ih_end {
                            for iw in iw_start..iw_end {
                                let in_idx = b * channels * in_h * in_w
                                    + c * in_h * in_w
                                    + ih * in_w
                                    + iw;
                                grad_input[in_idx] += grad_per_elem;
                            }
                        }
                    }
                }
            }
        }

        let grad = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &'static str {
        "AdaptiveAvgPool2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// ConvTranspose2d Backward
// =============================================================================

/// Gradient function for 2D transposed convolution.
///
/// ConvTranspose2d backward is essentially a forward Conv2d.
#[derive(Debug)]
pub struct ConvTranspose2dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_weight: Tensor<f32>,
    input_shape: Vec<usize>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    has_bias: bool,
}

impl ConvTranspose2dBackward {
    /// Creates a new ConvTranspose2dBackward.
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<Option<GradFn>>,
        saved_input: Tensor<f32>,
        saved_weight: Tensor<f32>,
        input_shape: Vec<usize>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        has_bias: bool,
    ) -> Self {
        let mut next_fns = vec![input_grad_fn, weight_grad_fn];
        if let Some(bias_fn) = bias_grad_fn {
            next_fns.push(bias_fn);
        }
        Self {
            next_fns,
            saved_input,
            saved_weight,
            input_shape,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            has_bias,
        }
    }
}

impl GradientFunction for ConvTranspose2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        // ConvTranspose2d backward w.r.t. input is a standard Conv2d
        // ConvTranspose2d backward w.r.t. weight uses the input and grad_output
        let grad_out_vec = grad_output.to_vec();
        let grad_out_shape = grad_output.shape();
        let batch_size = grad_out_shape[0];
        let out_h = grad_out_shape[2];
        let out_w = grad_out_shape[3];

        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();

        // d_input: standard conv2d of grad_output with weight
        // weight shape: (in_channels, out_channels, kh, kw)
        let mut grad_input = vec![0.0f32; batch_size * self.in_channels * in_h * in_w];

        for b in 0..batch_size {
            for ic in 0..self.in_channels {
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let mut sum = 0.0f32;
                        for oc in 0..self.out_channels {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let oh_signed = (ih as isize) * (sh as isize) + (ki as isize) - (ph as isize);
                                    let ow_signed = (iw as isize) * (sw as isize) + (kj as isize) - (pw as isize);
                                    if oh_signed >= 0 && (oh_signed as usize) < out_h
                                        && ow_signed >= 0 && (ow_signed as usize) < out_w
                                    {
                                        let oh = oh_signed as usize;
                                        let ow = ow_signed as usize;
                                        let go_idx = b * self.out_channels * out_h * out_w
                                            + oc * out_h * out_w
                                            + oh * out_w
                                            + ow;
                                        let w_idx = ic * self.out_channels * kh * kw
                                            + oc * kh * kw
                                            + ki * kw
                                            + kj;
                                        sum += grad_out_vec[go_idx] * weight_vec[w_idx];
                                    }
                                }
                            }
                        }
                        let in_idx = b * self.in_channels * in_h * in_w
                            + ic * in_h * in_w
                            + ih * in_w
                            + iw;
                        grad_input[in_idx] = sum;
                    }
                }
            }
        }

        // d_weight
        let mut grad_weight = vec![0.0f32; self.in_channels * self.out_channels * kh * kw];
        for b in 0..batch_size {
            for ic in 0..self.in_channels {
                for oc in 0..self.out_channels {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let mut sum = 0.0f32;
                            for ih in 0..in_h {
                                for iw in 0..in_w {
                                    let oh_signed = (ih as isize) * (sh as isize) + (ki as isize) - (ph as isize);
                                    let ow_signed = (iw as isize) * (sw as isize) + (kj as isize) - (pw as isize);
                                    if oh_signed >= 0 && (oh_signed as usize) < out_h
                                        && ow_signed >= 0 && (ow_signed as usize) < out_w
                                    {
                                        let oh = oh_signed as usize;
                                        let ow = ow_signed as usize;
                                        let in_idx = b * self.in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + ih * in_w
                                            + iw;
                                        let go_idx = b * self.out_channels * out_h * out_w
                                            + oc * out_h * out_w
                                            + oh * out_w
                                            + ow;
                                        sum += input_vec[in_idx] * grad_out_vec[go_idx];
                                    }
                                }
                            }
                            let w_idx = ic * self.out_channels * kh * kw
                                + oc * kh * kw
                                + ki * kw
                                + kj;
                            grad_weight[w_idx] += sum;
                        }
                    }
                }
            }
        }

        let grad_input_tensor = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        let grad_weight_tensor = Tensor::from_vec(
            grad_weight,
            &[self.in_channels, self.out_channels, kh, kw],
        ).unwrap();

        let mut result = vec![Some(grad_input_tensor), Some(grad_weight_tensor)];

        if self.has_bias {
            let mut grad_bias = vec![0.0f32; self.out_channels];
            for b in 0..batch_size {
                for oc in 0..self.out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let go_idx = b * self.out_channels * out_h * out_w
                                + oc * out_h * out_w
                                + oh * out_w
                                + ow;
                            grad_bias[oc] += grad_out_vec[go_idx];
                        }
                    }
                }
            }
            result.push(Some(
                Tensor::from_vec(grad_bias, &[self.out_channels]).unwrap(),
            ));
        }

        result
    }

    fn name(&self) -> &'static str {
        "ConvTranspose2dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// LayerNorm Backward
// =============================================================================

/// Gradient function for LayerNorm.
///
/// LayerNorm normalizes over the last `normalized_shape` dimensions.
/// Gradients flow back through: d_input, d_weight, d_bias.
#[derive(Debug)]
pub struct LayerNormBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_weight: Tensor<f32>,
    normalized_shape: Vec<usize>,
    eps: f32,
}

impl LayerNormBackward {
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<GradFn>,
        input: Tensor<f32>,
        weight: Tensor<f32>,
        normalized_shape: Vec<usize>,
        eps: f32,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn, weight_grad_fn, bias_grad_fn],
            saved_input: input,
            saved_weight: weight,
            normalized_shape,
            eps,
        }
    }
}

impl GradientFunction for LayerNormBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let norm_size: usize = self.normalized_shape.iter().product();

        // GPU fast path: use CUDA LayerNorm backward kernels
        #[cfg(feature = "cuda")]
        if self.saved_input.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_input.device()).unwrap()
            };
            let weight_gpu = if self.saved_weight.device().is_gpu() {
                self.saved_weight.clone()
            } else {
                self.saved_weight.to_device(self.saved_input.device()).unwrap()
            };

            // d_input via CUDA kernel
            let d_input = grad_gpu.layer_norm_backward_dinput_cuda(
                &self.saved_input,
                &weight_gpu,
                norm_size,
                self.eps,
            );

            // d_weight, d_bias via CUDA kernel
            let (d_weight, d_bias) = grad_gpu.layer_norm_backward_dweight_dbias_cuda(
                &self.saved_input,
                norm_size,
                self.eps,
            );

            return vec![Some(d_input), Some(d_weight), Some(d_bias)];
        }

        // CPU path
        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();
        let grad_vec = grad_output.to_vec();
        let shape = self.saved_input.shape().to_vec();

        let batch_size = input_vec.len() / norm_size;
        let n = norm_size as f32;

        let mut d_bias = vec![0.0f32; norm_size];
        let mut d_weight = vec![0.0f32; norm_size];
        let mut d_input = vec![0.0f32; input_vec.len()];

        for b in 0..batch_size {
            let start = b * norm_size;
            let slice = &input_vec[start..start + norm_size];

            let mean: f32 = slice.iter().sum::<f32>() / n;
            let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let std_inv = 1.0 / (var + self.eps).sqrt();

            let mut sum_dy = 0.0f32;
            let mut sum_dy_xhat = 0.0f32;

            for i in 0..norm_size {
                let x_hat = (slice[i] - mean) * std_inv;
                let dy = grad_vec[start + i] * weight_vec[i];
                sum_dy += dy;
                sum_dy_xhat += dy * x_hat;

                d_bias[i] += grad_vec[start + i];
                d_weight[i] += grad_vec[start + i] * x_hat;
            }

            for i in 0..norm_size {
                let x_hat = (slice[i] - mean) * std_inv;
                let dy = grad_vec[start + i] * weight_vec[i];
                d_input[start + i] = std_inv * (dy - sum_dy / n - x_hat * sum_dy_xhat / n);
            }
        }

        let d_input_tensor = Tensor::from_vec(d_input, &shape).unwrap();
        let d_weight_tensor = Tensor::from_vec(d_weight, self.saved_weight.shape()).unwrap();
        let d_bias_tensor = Tensor::from_vec(d_bias, self.saved_weight.shape()).unwrap();

        vec![Some(d_input_tensor), Some(d_weight_tensor), Some(d_bias_tensor)]
    }

    fn name(&self) -> &'static str {
        "LayerNormBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// GroupNorm Backward
// =============================================================================

/// Gradient function for GroupNorm.
///
/// Normalizes within each group of channels. Similar to BatchNorm but
/// groups channels instead of normalizing per-channel across the batch.
#[derive(Debug)]
pub struct GroupNormBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_weight: Tensor<f32>,
    num_groups: usize,
    eps: f32,
}

impl GroupNormBackward {
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<GradFn>,
        input: Tensor<f32>,
        weight: Tensor<f32>,
        num_groups: usize,
        eps: f32,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn, weight_grad_fn, bias_grad_fn],
            saved_input: input,
            saved_weight: weight,
            num_groups,
            eps,
        }
    }
}

impl GradientFunction for GroupNormBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();
        let grad_vec = grad_output.to_vec();
        let shape = self.saved_input.shape().to_vec();

        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();
        let channels_per_group = channels / self.num_groups;
        let group_size = channels_per_group * spatial_size;
        let n = group_size as f32;

        let num_channels = channels;
        let mut d_input = vec![0.0f32; input_vec.len()];
        let mut d_weight = vec![0.0f32; num_channels];
        let mut d_bias = vec![0.0f32; num_channels];

        for b in 0..batch_size {
            for g in 0..self.num_groups {
                // Compute group mean and variance
                let mut sum = 0.0f32;
                for c in 0..channels_per_group {
                    let ch = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + ch * spatial_size + s;
                        sum += input_vec[idx];
                    }
                }
                let mean = sum / n;

                let mut var_sum = 0.0f32;
                for c in 0..channels_per_group {
                    let ch = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + ch * spatial_size + s;
                        let diff = input_vec[idx] - mean;
                        var_sum += diff * diff;
                    }
                }
                let var = var_sum / n;
                let std_inv = 1.0 / (var + self.eps).sqrt();

                // Accumulate d_weight, d_bias and compute intermediates
                let mut sum_dy = 0.0f32;
                let mut sum_dy_xhat = 0.0f32;

                for c in 0..channels_per_group {
                    let ch = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + ch * spatial_size + s;
                        let x_hat = (input_vec[idx] - mean) * std_inv;
                        let dy = grad_vec[idx] * weight_vec[ch];
                        sum_dy += dy;
                        sum_dy_xhat += dy * x_hat;

                        d_weight[ch] += grad_vec[idx] * x_hat;
                        d_bias[ch] += grad_vec[idx];
                    }
                }

                // Compute d_input
                for c in 0..channels_per_group {
                    let ch = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + ch * spatial_size + s;
                        let x_hat = (input_vec[idx] - mean) * std_inv;
                        let dy = grad_vec[idx] * weight_vec[ch];
                        d_input[idx] = std_inv * (dy - sum_dy / n - x_hat * sum_dy_xhat / n);
                    }
                }
            }
        }

        let d_input_tensor = Tensor::from_vec(d_input, &shape).unwrap();
        let d_weight_tensor = Tensor::from_vec(d_weight, &[num_channels]).unwrap();
        let d_bias_tensor = Tensor::from_vec(d_bias, &[num_channels]).unwrap();

        vec![Some(d_input_tensor), Some(d_weight_tensor), Some(d_bias_tensor)]
    }

    fn name(&self) -> &'static str {
        "GroupNormBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// InstanceNorm2d Backward
// =============================================================================

/// Gradient function for InstanceNorm2d.
///
/// Instance normalization normalizes each (batch, channel) pair independently
/// over the spatial dimensions (H, W).
#[derive(Debug)]
pub struct InstanceNorm2dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_weight: Tensor<f32>,
    eps: f32,
    affine: bool,
}

impl InstanceNorm2dBackward {
    #[must_use]
    pub fn new(
        input_grad_fn: Option<GradFn>,
        weight_grad_fn: Option<GradFn>,
        bias_grad_fn: Option<GradFn>,
        input: Tensor<f32>,
        weight: Tensor<f32>,
        eps: f32,
        affine: bool,
    ) -> Self {
        Self {
            next_fns: vec![input_grad_fn, weight_grad_fn, bias_grad_fn],
            saved_input: input,
            saved_weight: weight,
            eps,
            affine,
        }
    }
}

impl GradientFunction for InstanceNorm2dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();
        let grad_vec = grad_output.to_vec();
        let shape = self.saved_input.shape().to_vec();

        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();
        let n = spatial_size as f32;

        let mut d_input = vec![0.0f32; input_vec.len()];
        let mut d_weight = vec![0.0f32; channels];
        let mut d_bias = vec![0.0f32; channels];

        for b in 0..batch_size {
            for c in 0..channels {
                let base = b * channels * spatial_size + c * spatial_size;

                // Compute mean and variance for this (b, c) pair
                let mut sum = 0.0f32;
                for s in 0..spatial_size {
                    sum += input_vec[base + s];
                }
                let mean = sum / n;

                let mut var_sum = 0.0f32;
                for s in 0..spatial_size {
                    let diff = input_vec[base + s] - mean;
                    var_sum += diff * diff;
                }
                let var = var_sum / n;
                let std_inv = 1.0 / (var + self.eps).sqrt();

                let w = if self.affine { weight_vec[c] } else { 1.0 };

                let mut sum_dy = 0.0f32;
                let mut sum_dy_xhat = 0.0f32;

                for s in 0..spatial_size {
                    let x_hat = (input_vec[base + s] - mean) * std_inv;
                    let dy = grad_vec[base + s] * w;
                    sum_dy += dy;
                    sum_dy_xhat += dy * x_hat;

                    if self.affine {
                        d_weight[c] += grad_vec[base + s] * x_hat;
                        d_bias[c] += grad_vec[base + s];
                    }
                }

                for s in 0..spatial_size {
                    let x_hat = (input_vec[base + s] - mean) * std_inv;
                    let dy = grad_vec[base + s] * w;
                    d_input[base + s] = std_inv * (dy - sum_dy / n - x_hat * sum_dy_xhat / n);
                }
            }
        }

        let d_input_tensor = Tensor::from_vec(d_input, &shape).unwrap();
        let d_weight_tensor = Tensor::from_vec(d_weight, &[channels]).unwrap();
        let d_bias_tensor = Tensor::from_vec(d_bias, &[channels]).unwrap();

        vec![Some(d_input_tensor), Some(d_weight_tensor), Some(d_bias_tensor)]
    }

    fn name(&self) -> &'static str {
        "InstanceNorm2dBackward"
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
    fn test_conv2d_backward_shapes() {
        // Input: (1, 1, 4, 4), Weight: (1, 1, 3, 3), no padding, stride 1
        // Output: (1, 1, 2, 2)
        let input = Tensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4]).unwrap();
        let weight = Tensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3]).unwrap();

        let backward = Conv2dBackward::new(
            None, None, None,
            input, weight,
            vec![1, 1, 4, 4],
            1, 1,
            (3, 3), (1, 1), (0, 0),
            false,
        );

        let grad_output = Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).unwrap();
        let grads = backward.apply(&grad_output);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 4, 4]);
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_backward_with_bias() {
        let input = Tensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4]).unwrap();
        let weight = Tensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3]).unwrap();

        let backward = Conv2dBackward::new(
            None, None, Some(None),
            input, weight,
            vec![1, 1, 4, 4],
            1, 1,
            (3, 3), (1, 1), (0, 0),
            true,
        );

        let grad_output = Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).unwrap();
        let grads = backward.apply(&grad_output);

        assert_eq!(grads.len(), 3);
        // bias grad shape: [out_channels]
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[1]);
        // bias grad = sum of grad_output = 4.0
        assert!((grads[2].as_ref().unwrap().to_vec()[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_maxpool2d_backward() {
        // Input: (1,1,4,4), pool 2x2, output (1,1,2,2)
        // Max indices should point to max positions
        let max_indices = vec![5, 7, 13, 15]; // positions of max in each 2x2 block
        let backward = MaxPool2dBackward::new(
            None,
            vec![1, 1, 4, 4],
            max_indices,
            (2, 2), (2, 2), (0, 0),
        );

        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let grads = backward.apply(&grad_output);

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[1, 1, 4, 4]);
        let grad_vec = grad.to_vec();
        // Only max positions should have gradient
        assert_eq!(grad_vec[5], 1.0);
        assert_eq!(grad_vec[7], 2.0);
        assert_eq!(grad_vec[13], 3.0);
        assert_eq!(grad_vec[15], 4.0);
        // Other positions should be zero
        assert_eq!(grad_vec[0], 0.0);
    }

    #[test]
    fn test_avgpool2d_backward() {
        let backward = AvgPool2dBackward::new(
            None,
            vec![1, 1, 4, 4],
            (2, 2), (2, 2), (0, 0),
        );

        let grad_output = Tensor::from_vec(vec![4.0, 4.0, 4.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let grads = backward.apply(&grad_output);

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[1, 1, 4, 4]);
        // Each element in 2x2 window gets 4.0 / 4 = 1.0
        for &v in &grad.to_vec() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_adaptive_avgpool2d_backward() {
        let backward = AdaptiveAvgPool2dBackward::new(
            None,
            vec![1, 1, 4, 4],
            (1, 1),
        );

        let grad_output = Tensor::from_vec(vec![16.0], &[1, 1, 1, 1]).unwrap();
        let grads = backward.apply(&grad_output);

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[1, 1, 4, 4]);
        // 16 elements averaged, grad = 16.0 / 16 = 1.0
        for &v in &grad.to_vec() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }
}
