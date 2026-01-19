//! Convolutional Layers - 1D and 2D Convolutions
//!
//! Applies convolution operations over input signals.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

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
        // Basic implementation using im2col approach
        // For a full implementation, we'd use optimized convolution kernels
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let _in_channels = input_shape[1];
        let in_length = input_shape[2];

        // Calculate output length
        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // For now, implement a simple direct convolution
        // A full implementation would use im2col or FFT
        let input_data = input.data();
        let weight_data = self.weight.data();
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

                            sum +=
                                input_data.to_vec()[input_idx] * weight_data.to_vec()[weight_idx];
                        }
                    }

                    // Add bias
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

        Variable::new(output_tensor, input.requires_grad())
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
        let (kh, kw) = kernel_size;
        let fan_in = in_channels * kh * kw;

        // Initialize weights
        let weight_data = kaiming_uniform(out_channels, fan_in);
        let weight_reshaped = weight_data
            .reshape(&[
                out_channels as isize,
                in_channels as isize,
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
        }
    }
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
        let input_vec = input_data.to_vec();
        let weight_vec = weight_data.to_vec();

        let mut output_data = vec![0.0f32; batch_size * self.out_channels * out_height * out_width];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;

                        for ic in 0..self.in_channels {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih = oh * sh + ki;
                                    let iw = ow * sw + kj;

                                    // Handle padding
                                    if ih < ph
                                        || ih >= in_height + ph
                                        || iw < pw
                                        || iw >= in_width + pw
                                    {
                                        continue;
                                    }

                                    let actual_ih = ih - ph;
                                    let actual_iw = iw - pw;

                                    let input_idx = b * self.in_channels * in_height * in_width
                                        + ic * in_height * in_width
                                        + actual_ih * in_width
                                        + actual_iw;

                                    let weight_idx = oc * self.in_channels * kh * kw
                                        + ic * kh * kw
                                        + ki * kw
                                        + kj;

                                    sum += input_vec[input_idx] * weight_vec[weight_idx];
                                }
                            }
                        }

                        // Add bias
                        if let Some(ref bias) = self.bias {
                            sum += bias.data().to_vec()[oc];
                        }

                        let output_idx = b * self.out_channels * out_height * out_width
                            + oc * out_height * out_width
                            + oh * out_width
                            + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        let output_tensor = Tensor::from_vec(
            output_data,
            &[batch_size, self.out_channels, out_height, out_width],
        )
        .unwrap();

        Variable::new(output_tensor, input.requires_grad())
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
    fn test_conv2d_parameters() {
        let conv = Conv2d::new(3, 64, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }
}
