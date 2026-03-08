//! Pooling Layers - Max and Average Pooling
//!
//! # File
//! `crates/axonml-nn/src/layers/pooling.rs`
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

use axonml_autograd::functions::{
    AdaptiveAvgPool2dBackward, AvgPool1dBackward, AvgPool2dBackward,
    MaxPool1dBackward, MaxPool2dBackward,
};
use axonml_autograd::grad_fn::GradFn;
use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::module::Module;

// =============================================================================
// MaxPool1d
// =============================================================================

/// Applies max pooling over a 1D signal.
///
/// # Shape
/// - Input: (N, C, L)
/// - Output: (N, C, L_out)
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool1d {
    /// Creates a new MaxPool1d layer.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
        }
    }

    /// Creates a MaxPool1d with custom stride and padding.
    pub fn with_options(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module for MaxPool1d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let length = shape[2];

        let out_length = (length + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let input_vec = input.data().to_vec();
        let mut output_data = vec![f32::NEG_INFINITY; batch * channels * out_length];
        let mut max_indices = vec![0usize; batch * channels * out_length];

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_length {
                    let in_start = ol * self.stride;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;

                    for k in 0..self.kernel_size {
                        let il = in_start + k;
                        if il >= self.padding && il < length + self.padding {
                            let actual_il = il - self.padding;
                            let idx = b * channels * length + c * length + actual_il;
                            if input_vec[idx] > max_val {
                                max_val = input_vec[idx];
                                max_idx = idx;
                            }
                        }
                    }

                    let out_idx = b * channels * out_length + c * out_length + ol;
                    output_data[out_idx] = max_val;
                    max_indices[out_idx] = max_idx;
                }
            }
        }

        let output = Tensor::from_vec(output_data, &[batch, channels, out_length]).unwrap();

        let requires_grad = input.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(MaxPool1dBackward::new(
                input.grad_fn().cloned(),
                shape,
                max_indices,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
    }

    fn name(&self) -> &'static str {
        "MaxPool1d"
    }
}

// =============================================================================
// MaxPool2d
// =============================================================================

/// Applies max pooling over a 2D signal (image).
///
/// # Shape
/// - Input: (N, C, H, W)
/// - Output: (N, C, H_out, W_out)
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    /// Creates a new MaxPool2d layer with square kernel.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
            padding: (0, 0),
        }
    }

    /// Creates a MaxPool2d with all options.
    pub fn with_options(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_h = (height + 2 * ph - kh) / sh + 1;
        let out_w = (width + 2 * pw - kw) / sw + 1;

        let input_vec = input.data().to_vec();
        let mut output_data = vec![f32::NEG_INFINITY; batch * channels * out_h * out_w];
        let mut max_indices = vec![0usize; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;

                                if ih >= ph && ih < height + ph && iw >= pw && iw < width + pw {
                                    let actual_ih = ih - ph;
                                    let actual_iw = iw - pw;
                                    let idx = b * channels * height * width
                                        + c * height * width
                                        + actual_ih * width
                                        + actual_iw;
                                    if input_vec[idx] > max_val {
                                        max_val = input_vec[idx];
                                        max_idx = idx;
                                    }
                                }
                            }
                        }

                        let out_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output_data[out_idx] = max_val;
                        max_indices[out_idx] = max_idx;
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, &[batch, channels, out_h, out_w]).unwrap();

        let requires_grad = input.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(MaxPool2dBackward::new(
                input.grad_fn().cloned(),
                shape,
                max_indices,
                self.kernel_size,
                self.stride,
                self.padding,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
    }

    fn name(&self) -> &'static str {
        "MaxPool2d"
    }
}

// =============================================================================
// AvgPool1d
// =============================================================================

/// Applies average pooling over a 1D signal.
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool1d {
    /// Creates a new AvgPool1d layer.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
        }
    }

    /// Creates an AvgPool1d with custom stride and padding.
    pub fn with_options(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module for AvgPool1d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let length = shape[2];

        let out_length = (length + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let input_vec = input.data().to_vec();
        let mut output_data = vec![0.0f32; batch * channels * out_length];

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_length {
                    let in_start = ol * self.stride;
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for k in 0..self.kernel_size {
                        let il = in_start + k;
                        if il >= self.padding && il < length + self.padding {
                            let actual_il = il - self.padding;
                            let idx = b * channels * length + c * length + actual_il;
                            sum += input_vec[idx];
                            count += 1;
                        }
                    }

                    let out_idx = b * channels * out_length + c * out_length + ol;
                    output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }

        let output = Tensor::from_vec(output_data, &[batch, channels, out_length]).unwrap();

        let requires_grad = input.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(AvgPool1dBackward::new(
                input.grad_fn().cloned(),
                shape,
                self.kernel_size,
                self.stride,
                self.padding,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
    }

    fn name(&self) -> &'static str {
        "AvgPool1d"
    }
}

// =============================================================================
// AvgPool2d
// =============================================================================

/// Applies average pooling over a 2D signal (image).
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    /// Creates a new AvgPool2d layer with square kernel.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
            padding: (0, 0),
        }
    }

    /// Creates an AvgPool2d with all options.
    pub fn with_options(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_h = (height + 2 * ph - kh) / sh + 1;
        let out_w = (width + 2 * pw - kw) / sw + 1;

        let input_vec = input.data().to_vec();
        let mut output_data = vec![0.0f32; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;

                                if ih >= ph && ih < height + ph && iw >= pw && iw < width + pw {
                                    let actual_ih = ih - ph;
                                    let actual_iw = iw - pw;
                                    let idx = b * channels * height * width
                                        + c * height * width
                                        + actual_ih * width
                                        + actual_iw;
                                    sum += input_vec[idx];
                                    count += 1;
                                }
                            }
                        }

                        let out_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, &[batch, channels, out_h, out_w]).unwrap();

        let requires_grad = input.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(AvgPool2dBackward::new(
                input.grad_fn().cloned(),
                shape,
                self.kernel_size,
                self.stride,
                self.padding,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
    }

    fn name(&self) -> &'static str {
        "AvgPool2d"
    }
}

// =============================================================================
// AdaptiveAvgPool2d
// =============================================================================

/// Applies adaptive average pooling to produce specified output size.
pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// Creates a new AdaptiveAvgPool2d.
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Creates an AdaptiveAvgPool2d with square output.
    pub fn square(size: usize) -> Self {
        Self {
            output_size: (size, size),
        }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let in_h = shape[2];
        let in_w = shape[3];

        let (out_h, out_w) = self.output_size;
        let input_vec = input.data().to_vec();
        let mut output_data = vec![0.0f32; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih_start = (oh * in_h) / out_h;
                        let ih_end = ((oh + 1) * in_h) / out_h;
                        let iw_start = (ow * in_w) / out_w;
                        let iw_end = ((ow + 1) * in_w) / out_w;

                        let mut sum = 0.0f32;
                        let mut count = 0;

                        for ih in ih_start..ih_end {
                            for iw in iw_start..iw_end {
                                let idx =
                                    b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                                sum += input_vec[idx];
                                count += 1;
                            }
                        }

                        let out_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, &[batch, channels, out_h, out_w]).unwrap();

        let requires_grad = input.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(AdaptiveAvgPool2dBackward::new(
                input.grad_fn().cloned(),
                shape,
                self.output_size,
            ));
            Variable::from_operation(output, grad_fn, true)
        } else {
            Variable::new(output, false)
        }
    }

    fn name(&self) -> &'static str {
        "AdaptiveAvgPool2d"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d() {
        let pool = MaxPool2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
                &[1, 1, 4, 4],
            )
            .unwrap(),
            false,
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape(), vec![1, 1, 2, 2]);
        assert_eq!(output.data().to_vec(), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_maxpool2d_backward() {
        let pool = MaxPool2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
                &[1, 1, 4, 4],
            )
            .unwrap(),
            true,
        );
        let output = pool.forward(&input);
        let loss = output.sum();
        loss.backward();

        assert!(input.grad().is_some(), "MaxPool2d: gradient should flow");
        let grad = input.grad().unwrap();
        assert_eq!(grad.shape(), &[1, 1, 4, 4]);
        let grad_vec = grad.to_vec();
        // Only max positions (6,8,14,16) at indices [5,7,13,15] should have gradient
        assert_eq!(grad_vec[5], 1.0);
        assert_eq!(grad_vec[7], 1.0);
        assert_eq!(grad_vec[13], 1.0);
        assert_eq!(grad_vec[15], 1.0);
        assert_eq!(grad_vec[0], 0.0);
    }

    #[test]
    fn test_avgpool2d() {
        let pool = AvgPool2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
                &[1, 1, 4, 4],
            )
            .unwrap(),
            false,
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape(), vec![1, 1, 2, 2]);
        assert_eq!(output.data().to_vec(), vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_avgpool2d_backward() {
        let pool = AvgPool2d::new(2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4]).unwrap(),
            true,
        );
        let output = pool.forward(&input);
        let loss = output.sum();
        loss.backward();

        assert!(input.grad().is_some(), "AvgPool2d: gradient should flow");
        let grad = input.grad().unwrap();
        // Each input element contributes to exactly one pool window, gets 1/4 of the gradient
        for &v in &grad.to_vec() {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_adaptive_avgpool2d() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap(),
            false,
        );
        let output = pool.forward(&input);
        assert_eq!(output.shape(), vec![1, 1, 1, 1]);
        assert_eq!(output.data().to_vec(), vec![2.5]);
    }

    #[test]
    fn test_adaptive_avgpool2d_backward() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap(),
            true,
        );
        let output = pool.forward(&input);
        let loss = output.sum();
        loss.backward();

        assert!(input.grad().is_some(), "AdaptiveAvgPool2d: gradient should flow");
        let grad = input.grad().unwrap();
        for &v in &grad.to_vec() {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }
}
