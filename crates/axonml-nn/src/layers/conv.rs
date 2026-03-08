//! Convolutional Layers - 1D and 2D Convolutions
//!
//! # File
//! `crates/axonml-nn/src/layers/conv.rs`
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

use std::collections::HashMap;

use axonml_autograd::functions::{Conv1dBackward, Conv2dBackward, ConvTranspose2dBackward, GroupedConv2dBackward};
use axonml_autograd::grad_fn::GradFn;
use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;
use rayon::prelude::*;

use crate::init::{kaiming_uniform, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// Conv1d
// =============================================================================

/// Applies a 1D convolution over an input signal.
///
/// # Shape
/// - Input: (N, C_in, L)
/// - Output: (N, C_out, L_out)
///
/// where L_out = (L + 2*padding - kernel_size) / stride + 1
pub struct Conv1d {
    /// Weight tensor of shape (out_channels, in_channels, kernel_size).
    pub weight: Parameter,
    /// Bias tensor of shape (out_channels).
    pub bias: Option<Parameter>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Size of the convolving kernel.
    kernel_size: usize,
    /// Stride of the convolution.
    stride: usize,
    /// Zero-padding added to both sides.
    padding: usize,
}

impl Conv1d {
    /// Creates a new Conv1d layer.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(in_channels, out_channels, kernel_size, 1, 0, true)
    }

    /// Creates a Conv1d layer with all options.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        // Initialize weights
        let fan_in = in_channels * kernel_size;
        let weight_data = kaiming_uniform(out_channels, fan_in);
        let weight_reshaped = weight_data
            .reshape(&[
                out_channels as isize,
                in_channels as isize,
                kernel_size as isize,
            ])
            .unwrap();
        let weight = Parameter::named("weight", weight_reshaped, true);

        let bias_param = if bias {
            Some(Parameter::named("bias", zeros(&[out_channels]), true))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Variable) -> Variable {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let _in_channels = input_shape[1];
        let in_length = input_shape[2];

        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let input_data = input.data();
        let weight_data = self.weight.data();
        let input_vec = input_data.to_vec();
        let weight_vec = weight_data.to_vec();

        let mut output_data = vec![0.0f32; batch_size * self.out_channels * out_length];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let mut sum = 0.0f32;
                    let in_start = ol * self.stride;

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let in_idx = in_start + k;
                            if in_idx < self.padding || in_idx >= in_length + self.padding {
                                continue;
                            }
                            let actual_idx = in_idx - self.padding;

                            let input_idx =
                                b * self.in_channels * in_length + ic * in_length + actual_idx;
                            let weight_idx = oc * self.in_channels * self.kernel_size
                                + ic * self.kernel_size
                                + k;

                            sum += input_vec[input_idx] * weight_vec[weight_idx];
                        }
                    }

                    if let Some(ref bias) = self.bias {
                        sum += bias.data().to_vec()[oc];
                    }

                    let output_idx = b * self.out_channels * out_length + oc * out_length + ol;
                    output_data[output_idx] = sum;
                }
            }
        }

        let output_tensor =
            Tensor::from_vec(output_data, &[batch_size, self.out_channels, out_length]).unwrap();

        let requires_grad = (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();

        if requires_grad {
            let weight_var = self.weight.variable();
            let bias_grad_fn = self.bias.as_ref().map(|b| b.variable().grad_fn().cloned());

            let grad_fn = GradFn::new(Conv1dBackward::new(
                input.grad_fn().cloned(),
                weight_var.grad_fn().cloned(),
                bias_grad_fn,
                input_data,
                weight_data,
                input_shape,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.bias.is_some(),
            ));
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "Conv1d"
    }
}

// =============================================================================
// Conv2d
// =============================================================================

/// Applies a 2D convolution over an input image.
///
/// # Shape
/// - Input: (N, C_in, H, W)
/// - Output: (N, C_out, H_out, W_out)
///
/// where H_out = (H + 2*padding - kernel_size) / stride + 1
pub struct Conv2d {
    /// Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w).
    pub weight: Parameter,
    /// Bias tensor of shape (out_channels).
    pub bias: Option<Parameter>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Size of the convolving kernel (height, width).
    kernel_size: (usize, usize),
    /// Stride of the convolution (height, width).
    stride: (usize, usize),
    /// Zero-padding added to both sides (height, width).
    padding: (usize, usize),
    /// Number of groups for grouped convolution.
    groups: usize,
}

impl Conv2d {
    /// Creates a new Conv2d layer with square kernel.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (0, 0),
            true,
        )
    }

    /// Creates a Conv2d layer with all options.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        Self::with_groups(in_channels, out_channels, kernel_size, stride, padding, bias, 1)
    }

    /// Creates a Conv2d layer with grouped convolution support.
    ///
    /// When `groups == in_channels` and `out_channels == in_channels`, this is
    /// a depthwise convolution.
    pub fn with_groups(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
        groups: usize,
    ) -> Self {
        assert!(in_channels % groups == 0, "in_channels must be divisible by groups");
        assert!(out_channels % groups == 0, "out_channels must be divisible by groups");

        let (kh, kw) = kernel_size;
        let in_channels_per_group = in_channels / groups;
        let fan_in = in_channels_per_group * kh * kw;

        let weight_data = kaiming_uniform(out_channels, fan_in);
        let weight_reshaped = weight_data
            .reshape(&[
                out_channels as isize,
                in_channels_per_group as isize,
                kh as isize,
                kw as isize,
            ])
            .unwrap();
        let weight = Parameter::named("weight", weight_reshaped, true);

        let bias_param = if bias {
            Some(Parameter::named("bias", zeros(&[out_channels]), true))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
        }
    }

    /// Creates a depthwise convolution (groups = in_channels).
    pub fn depthwise(channels: usize, kernel_size: usize) -> Self {
        Self::with_groups(
            channels,
            channels,
            (kernel_size, kernel_size),
            (1, 1),
            (kernel_size / 2, kernel_size / 2),
            true,
            channels,
        )
    }
}

// =============================================================================
// im2col + GEMM Conv2d Implementation
// =============================================================================

/// Unfold input patches into a column matrix (im2col).
///
/// Input: `[C_in, H, W]` (one batch element, one group's channels)
/// Output: `[C_in * kH * kW, out_H * out_W]`
fn im2col(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let col_h = channels * kernel_h * kernel_w;
    let col_w = out_h * out_w;
    let mut col = vec![0.0f32; col_h * col_w];
    let hw = height * width;
    let kk = kernel_h * kernel_w;
    let h_signed = height as isize;
    let w_signed = width as isize;
    let pad_h_s = pad_h as isize;
    let pad_w_s = pad_w as isize;

    // Fused single-pass: iterate linearly over output col matrix
    // col_row = c * kH * kW + kh_off * kW + kw_off
    // col_col = oh * out_w + ow
    for col_row in 0..col_h {
        let c = col_row / kk;
        let k_idx = col_row % kk;
        let kh_off = k_idx / kernel_w;
        let kw_off = k_idx % kernel_w;
        let input_c = c * hw;
        let col_base = col_row * col_w;

        for oh in 0..out_h {
            let h_in = (oh * stride_h + kh_off) as isize - pad_h_s;
            if h_in < 0 || h_in >= h_signed { continue; }
            let input_row = input_c + h_in as usize * width;
            let col_row_base = col_base + oh * out_w;

            for ow in 0..out_w {
                let w_in = (ow * stride_w + kw_off) as isize - pad_w_s;
                if w_in >= 0 && w_in < w_signed {
                    unsafe {
                        *col.get_unchecked_mut(col_row_base + ow) =
                            *input.get_unchecked(input_row + w_in as usize);
                    }
                }
            }
        }
    }

    col
}

/// Conv2d forward using im2col + matmul. Supports groups.
fn conv2d_im2col(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch_size: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    groups: usize,
) -> Vec<f32> {
    let out_h = (in_height + 2 * ph - kh) / sh + 1;
    let out_w = (in_width + 2 * pw - kw) / sw + 1;
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;
    let col_h = in_channels_per_group * kh * kw;
    let col_w = out_h * out_w;
    let spatial = out_h * out_w;
    let in_spatial = in_height * in_width;

    // Parallel: each batch element produces its own output slice
    let out_per_batch = out_channels * spatial;
    let per_batch: Vec<Vec<f32>> = (0..batch_size)
        .into_par_iter()
        .map(|b| {
            let mut batch_out = vec![0.0f32; out_per_batch];

            for g in 0..groups {
                let ic_start = g * in_channels_per_group;
                let oc_start = g * out_channels_per_group;

                // Extract input for this batch+group
                let in_offset = b * in_channels * in_spatial + ic_start * in_spatial;
                let input_slice = &input[in_offset..in_offset + in_channels_per_group * in_spatial];

                // im2col
                let col = im2col(
                    input_slice,
                    in_channels_per_group,
                    in_height, in_width,
                    kh, kw, ph, pw, sh, sw,
                    out_h, out_w,
                );

                // Weight for this group
                let w_offset = oc_start * in_channels_per_group * kh * kw;
                let w_size = out_channels_per_group * col_h;
                let weight_slice = &weight[w_offset..w_offset + w_size];

                // GEMM via Tensor::matmul
                let w_tensor = Tensor::from_vec(weight_slice.to_vec(), &[out_channels_per_group, col_h]).unwrap();
                let col_tensor = Tensor::from_vec(col, &[col_h, col_w]).unwrap();
                let result = w_tensor.matmul(&col_tensor).unwrap();
                let result_vec = result.to_vec();

                // Copy to output with bias
                let out_offset = oc_start * spatial;
                for oc_local in 0..out_channels_per_group {
                    let oc = oc_start + oc_local;
                    let bias_val = bias.map_or(0.0, |bv| bv[oc]);
                    let src_start = oc_local * col_w;
                    let dst_start = out_offset + oc_local * spatial;
                    if bias_val != 0.0 {
                        for i in 0..spatial {
                            batch_out[dst_start + i] = result_vec[src_start + i] + bias_val;
                        }
                    } else {
                        batch_out[dst_start..dst_start + spatial]
                            .copy_from_slice(&result_vec[src_start..src_start + spatial]);
                    }
                }
            }

            batch_out
        })
        .collect();

    // Flatten per-batch results into single output
    let mut output = Vec::with_capacity(batch_size * out_per_batch);
    for batch_out in per_batch {
        output.extend_from_slice(&batch_out);
    }
    output
}

impl Module for Conv2d {
    fn forward(&self, input: &Variable) -> Variable {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_height = (in_height + 2 * ph - kh) / sh + 1;
        let out_width = (in_width + 2 * pw - kw) / sw + 1;

        let input_data = input.data();
        let weight_data = self.weight.data();

        // GPU-resident fast path: when input is already on GPU, do everything on GPU
        // without any CPU↔GPU copies.
        #[cfg(feature = "cuda")]
        if input_data.device().is_gpu() {
            // Auto-migrate weights to GPU if needed (one-time cost, cached via Arc)
            let input_dev = input_data.device();
            if !weight_data.device().is_gpu() {
                self.weight.to_device(input_dev);
                if let Some(ref b) = self.bias {
                    b.to_device(input_dev);
                }
            }
            let weight_data = self.weight.data();

            let gpu_output = if self.groups == 1 {
                // Standard convolution: single im2col + GEMM
                let bias_tensor = self.bias.as_ref().map(|b| b.data());
                input_data.conv2d_cuda(
                    &weight_data,
                    bias_tensor.as_ref(),
                    self.stride,
                    self.padding,
                )
            } else {
                // Grouped convolution: run per-group im2col + GEMM on GPU
                input_data.conv2d_grouped_cuda(
                    &weight_data,
                    self.bias.as_ref().map(|b| b.data()).as_ref(),
                    self.stride,
                    self.padding,
                    self.groups,
                )
            };

            if let Some(output_tensor) = gpu_output {
                let requires_grad = (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();
                if requires_grad {
                    let weight_var = self.weight.variable();
                    let bias_grad_fn = self.bias.as_ref().map(|b| b.variable().grad_fn().cloned());
                    if self.groups == 1 {
                        let grad_fn = GradFn::new(Conv2dBackward::new(
                            input.grad_fn().cloned(),
                            weight_var.grad_fn().cloned(),
                            bias_grad_fn,
                            input_data,
                            weight_data,
                            input_shape,
                            self.in_channels,
                            self.out_channels,
                            self.kernel_size,
                            self.stride,
                            self.padding,
                            self.bias.is_some(),
                        ));
                        return Variable::from_operation(output_tensor, grad_fn, true);
                    } else {
                        let grad_fn = GradFn::new(GroupedConv2dBackward::new(
                            input.grad_fn().cloned(),
                            weight_var.grad_fn().cloned(),
                            bias_grad_fn,
                            input_data,
                            weight_data,
                            input_shape,
                            self.in_channels,
                            self.out_channels,
                            self.kernel_size,
                            self.stride,
                            self.padding,
                            self.groups,
                            self.bias.is_some(),
                        ));
                        return Variable::from_operation(output_tensor, grad_fn, true);
                    }
                } else {
                    return Variable::new(output_tensor, false);
                }
            }
            // Fall through to CPU path if GPU conv failed
        }

        let input_vec = input_data.to_vec();
        let weight_vec = weight_data.to_vec();

        // Try GPU im2col+GEMM for groups=1 when data is on CPU but GPU is available
        let conv_flops = self.out_channels * self.in_channels * kh * kw * out_height * out_width;
        let output_data = if self.groups == 1 && conv_flops >= 500_000 {
            let bias_vec = self.bias.as_ref().map(|b| b.data().to_vec());
            let gpu_result = axonml_core::backends::cuda::cuda_conv2d_forward(
                &input_vec,
                &weight_vec,
                bias_vec.as_deref(),
                batch_size,
                self.in_channels,
                in_height,
                in_width,
                self.out_channels,
                kh, kw,
                sh, sw,
                ph, pw,
            );

            if let Some(result) = gpu_result {
                result
            } else {
                conv2d_im2col(
                    &input_vec,
                    &weight_vec,
                    self.bias.as_ref().map(|b| b.data().to_vec()).as_deref(),
                    batch_size,
                    self.in_channels,
                    in_height,
                    in_width,
                    self.out_channels,
                    kh, kw,
                    sh, sw,
                    ph, pw,
                    self.groups,
                )
            }
        } else {
            conv2d_im2col(
                &input_vec,
                &weight_vec,
                self.bias.as_ref().map(|b| b.data().to_vec()).as_deref(),
                batch_size,
                self.in_channels,
                in_height,
                in_width,
                self.out_channels,
                kh, kw,
                sh, sw,
                ph, pw,
                self.groups,
            )
        };

        let output_tensor = Tensor::from_vec(
            output_data,
            &[batch_size, self.out_channels, out_height, out_width],
        )
        .unwrap();

        let requires_grad = (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();

        if requires_grad && self.groups == 1 {
            // Full backward pass for standard convolution
            let weight_var = self.weight.variable();
            let bias_grad_fn = self.bias.as_ref().map(|b| b.variable().grad_fn().cloned());

            let grad_fn = GradFn::new(Conv2dBackward::new(
                input.grad_fn().cloned(),
                weight_var.grad_fn().cloned(),
                bias_grad_fn,
                input_data,
                weight_data,
                input_shape,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.bias.is_some(),
            ));
            Variable::from_operation(output_tensor, grad_fn, true)
        } else if requires_grad {
            // Grouped convolution backward (depthwise separable, etc.)
            let weight_var = self.weight.variable();
            let bias_grad_fn = self.bias.as_ref().map(|b| b.variable().grad_fn().cloned());

            let grad_fn = GradFn::new(GroupedConv2dBackward::new(
                input.grad_fn().cloned(),
                weight_var.grad_fn().cloned(),
                bias_grad_fn,
                input_data,
                weight_data,
                input_shape,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.groups,
                self.bias.is_some(),
            ));
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "Conv2d"
    }
}

// =============================================================================
// ConvTranspose2d
// =============================================================================

/// Applies a 2D transposed convolution (deconvolution) for upsampling.
///
/// # Shape
/// - Input: (N, C_in, H, W)
/// - Output: (N, C_out, H_out, W_out)
///
/// where H_out = (H - 1) * stride - 2*padding + kernel_size + output_padding
pub struct ConvTranspose2d {
    /// Weight tensor of shape (in_channels, out_channels, kernel_h, kernel_w).
    pub weight: Parameter,
    /// Bias tensor of shape (out_channels).
    pub bias: Option<Parameter>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
}

impl ConvTranspose2d {
    /// Creates a new ConvTranspose2d layer with square kernel.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (0, 0),
            (0, 0),
            true,
        )
    }

    /// Creates a ConvTranspose2d layer with all options.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let (kh, kw) = kernel_size;
        let fan_in = in_channels * kh * kw;

        let weight_data = kaiming_uniform(out_channels, fan_in);
        let weight_reshaped = weight_data
            .reshape(&[
                in_channels as isize,
                out_channels as isize,
                kh as isize,
                kw as isize,
            ])
            .unwrap();
        let weight = Parameter::named("weight", weight_reshaped, true);

        let bias_param = if bias {
            Some(Parameter::named("bias", zeros(&[out_channels]), true))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        }
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Variable) -> Variable {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (oph, opw) = self.output_padding;

        let out_h = (in_h - 1) * sh - 2 * ph + kh + oph;
        let out_w = (in_w - 1) * sw - 2 * pw + kw + opw;

        let input_data = input.data();
        let weight_data = self.weight.data();
        let input_vec = input_data.to_vec();
        let weight_vec = weight_data.to_vec();

        let mut output_data = vec![0.0f32; batch_size * self.out_channels * out_h * out_w];

        // Transposed convolution: scatter input values through the kernel
        for b in 0..batch_size {
            for ic in 0..self.in_channels {
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let in_idx = b * self.in_channels * in_h * in_w
                            + ic * in_h * in_w
                            + ih * in_w
                            + iw;
                        let in_val = input_vec[in_idx];

                        for oc in 0..self.out_channels {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let oh_signed = (ih * sh + ki) as isize - ph as isize;
                                    let ow_signed = (iw * sw + kj) as isize - pw as isize;

                                    if oh_signed >= 0
                                        && (oh_signed as usize) < out_h
                                        && ow_signed >= 0
                                        && (ow_signed as usize) < out_w
                                    {
                                        let oh = oh_signed as usize;
                                        let ow = ow_signed as usize;
                                        let out_idx = b * self.out_channels * out_h * out_w
                                            + oc * out_h * out_w
                                            + oh * out_w
                                            + ow;
                                        // weight: (in_channels, out_channels, kh, kw)
                                        let w_idx = ic * self.out_channels * kh * kw
                                            + oc * kh * kw
                                            + ki * kw
                                            + kj;
                                        output_data[out_idx] += in_val * weight_vec[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_vec = bias.data().to_vec();
            for b in 0..batch_size {
                for oc in 0..self.out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let out_idx = b * self.out_channels * out_h * out_w
                                + oc * out_h * out_w
                                + oh * out_w
                                + ow;
                            output_data[out_idx] += bias_vec[oc];
                        }
                    }
                }
            }
        }

        let output_tensor = Tensor::from_vec(
            output_data,
            &[batch_size, self.out_channels, out_h, out_w],
        )
        .unwrap();

        let requires_grad = (input.requires_grad() || self.weight.requires_grad()) && is_grad_enabled();

        if requires_grad {
            let weight_var = self.weight.variable();
            let bias_grad_fn = self.bias.as_ref().map(|b| b.variable().grad_fn().cloned());

            let grad_fn = GradFn::new(ConvTranspose2dBackward::new(
                input.grad_fn().cloned(),
                weight_var.grad_fn().cloned(),
                bias_grad_fn,
                input_data,
                weight_data,
                input_shape,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.bias.is_some(),
            ));
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "ConvTranspose2d"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_creation() {
        let conv = Conv1d::new(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, 3);
    }

    #[test]
    fn test_conv1d_forward() {
        let conv = Conv1d::with_options(1, 1, 3, 1, 1, false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]).unwrap(),
            false,
        );
        let output = conv.forward(&input);
        assert_eq!(output.shape(), vec![1, 1, 5]);
    }

    #[test]
    fn test_conv1d_backward() {
        let conv = Conv1d::with_options(1, 1, 3, 1, 1, false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]).unwrap(),
            true,
        );
        let output = conv.forward(&input);
        let loss = output.sum();
        loss.backward();

        // Input should have gradient (not None)
        assert!(input.grad().is_some(), "Conv1d: input gradient should flow through backward pass");
        let grad = input.grad().unwrap();
        assert_eq!(grad.shape(), &[1, 1, 5]);
    }

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2d::new(3, 64, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 64);
        assert_eq!(conv.kernel_size, (3, 3));
    }

    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2d::with_options(1, 1, (3, 3), (1, 1), (1, 1), false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 25], &[1, 1, 5, 5]).unwrap(),
            false,
        );
        let output = conv.forward(&input);
        assert_eq!(output.shape(), vec![1, 1, 5, 5]);
    }

    #[test]
    fn test_conv2d_backward() {
        let conv = Conv2d::with_options(1, 1, (3, 3), (1, 1), (1, 1), false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 25], &[1, 1, 5, 5]).unwrap(),
            true,
        );
        let output = conv.forward(&input);
        let loss = output.sum();
        loss.backward();

        assert!(input.grad().is_some(), "Conv2d: input gradient should flow through backward pass");
        let grad = input.grad().unwrap();
        assert_eq!(grad.shape(), &[1, 1, 5, 5]);

        // Weight should also have gradient
        let w_grad = conv.weight.grad();
        assert!(w_grad.is_some(), "Conv2d: weight gradient should be computed");
    }

    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::new(3, 64, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_conv2d_grouped() {
        // Depthwise: groups = in_channels = out_channels
        let conv = Conv2d::depthwise(4, 3);
        assert_eq!(conv.groups, 4);
        assert_eq!(conv.in_channels, 4);
        assert_eq!(conv.out_channels, 4);

        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4 * 5 * 5], &[1, 4, 5, 5]).unwrap(),
            false,
        );
        let output = conv.forward(&input);
        assert_eq!(output.shape(), vec![1, 4, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d_forward() {
        let conv_t = ConvTranspose2d::with_options(1, 1, (3, 3), (2, 2), (1, 1), (1, 1), false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).unwrap(),
            false,
        );
        let output = conv_t.forward(&input);
        // H_out = (2-1)*2 - 2*1 + 3 + 1 = 4
        assert_eq!(output.shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_conv_transpose2d_backward() {
        let conv_t = ConvTranspose2d::new(1, 1, 3);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3]).unwrap(),
            true,
        );
        let output = conv_t.forward(&input);
        let loss = output.sum();
        loss.backward();

        assert!(input.grad().is_some(), "ConvTranspose2d: input gradient should flow through backward");
    }
}
