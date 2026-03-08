//! Convolution Gradient Functions
//!
//! Gradient functions for Conv1d, Conv2d, ConvTranspose2d,
//! and all pooling operations.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::any::Any;

use axonml_tensor::Tensor;
use rayon::prelude::*;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// GEMM helper for Conv2d backward (im2col approach, BLAS-accelerated)
// =============================================================================

/// C += A × B  using matrixmultiply (BLAS-level GEMM).
/// A: [m, k] (or [k, m] if trans_a), B: [k, n] (or [n, k] if trans_b), C: [m, n]
fn gemm_acc(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize,
    trans_a: bool, trans_b: bool)
{
    let (rsa, csa) = if trans_a { (1isize, m as isize) } else { (k as isize, 1isize) };
    let (rsb, csb) = if trans_b { (1isize, k as isize) } else { (n as isize, 1isize) };
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,               // alpha
            a.as_ptr(), rsa, csa,
            b.as_ptr(), rsb, csb,
            1.0,               // beta (accumulate into C)
            c.as_mut_ptr(), n as isize, 1,
        );
    }
}

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
        // GPU-resident fast path: do entire backward on GPU without CPU copies
        #[cfg(feature = "cuda")]
        if grad_output.device().is_gpu() && self.saved_input.device().is_gpu() {
            if let Some((grad_input, grad_weight, grad_bias)) = grad_output.conv2d_backward_cuda(
                &self.saved_input,
                &self.saved_weight,
                &self.input_shape,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.has_bias,
            ) {
                let mut result = vec![Some(grad_input), Some(grad_weight)];
                if self.has_bias {
                    result.push(grad_bias);
                }
                return result;
            }
            // Fall through to CPU path if GPU backward failed
        }

        let grad_out_shape = grad_output.shape();
        let batch_size = grad_out_shape[0];
        let out_h = grad_out_shape[2];
        let out_w = grad_out_shape[3];

        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let out_hw = out_h * out_w;
        let col_rows = self.in_channels * kh * kw;

        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();
        let grad_out_vec = grad_output.to_vec();

        // Use im2col + GEMM for efficient Conv2d backward (Rayon-parallelized across batch)
        // grad_weight = sum_over_batch( grad_out_reshaped × im2col(input)^T )
        // grad_input = col2im( weight^T × grad_out_reshaped )

        let in_per_batch = self.in_channels * in_h * in_w;
        let out_channels = self.out_channels;

        // Parallel: each batch element computes its own grad_input slice + partial grad_weight
        let per_batch_results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                // Fused im2col for this batch element
                let input_offset = b * in_per_batch;
                let mut col = vec![0.0f32; col_rows * out_hw];
                let kk = kh * kw;
                let ph_s = ph as isize;
                let pw_s = pw as isize;
                let in_h_s = in_h as isize;
                let in_w_s = in_w as isize;
                for cr in 0..col_rows {
                    let c = cr / kk;
                    let k_idx = cr % kk;
                    let ki = k_idx / kw;
                    let kj = k_idx % kw;
                    let input_c = input_offset + c * in_h * in_w;
                    let col_base = cr * out_hw;
                    for oh in 0..out_h {
                        let ih = (oh * sh + ki) as isize - ph_s;
                        if ih < 0 || ih >= in_h_s { continue; }
                        let input_row = input_c + ih as usize * in_w;
                        let col_row_base = col_base + oh * out_w;
                        for ow in 0..out_w {
                            let iw = (ow * sw + kj) as isize - pw_s;
                            if iw >= 0 && iw < in_w_s {
                                unsafe {
                                    *col.get_unchecked_mut(col_row_base + ow) =
                                        *input_vec.get_unchecked(input_row + iw as usize);
                                }
                            }
                        }
                    }
                }

                let go_offset = b * out_channels * out_hw;
                let go_slice = &grad_out_vec[go_offset..go_offset + out_channels * out_hw];

                // Thread-local grad_weight
                let mut local_grad_weight = vec![0.0f32; out_channels * col_rows];
                gemm_acc(go_slice, &col, &mut local_grad_weight,
                    out_channels, out_hw, col_rows, false, true);

                // grad_col = weight^T × grad_out
                let mut grad_col = vec![0.0f32; col_rows * out_hw];
                gemm_acc(&weight_vec, go_slice, &mut grad_col,
                    col_rows, out_channels, out_hw, true, false);

                // Fused col2im → local grad_input for this batch element
                let mut gi_batch = vec![0.0f32; in_per_batch];
                for cr in 0..col_rows {
                    let c = cr / kk;
                    let k_idx = cr % kk;
                    let ki = k_idx / kw;
                    let kj = k_idx % kw;
                    let gi_c = c * in_h * in_w;
                    let col_base = cr * out_hw;
                    for oh in 0..out_h {
                        let ih = (oh * sh + ki) as isize - ph_s;
                        if ih < 0 || ih >= in_h_s { continue; }
                        let gi_row = gi_c + ih as usize * in_w;
                        let col_row_base = col_base + oh * out_w;
                        for ow in 0..out_w {
                            let iw = (ow * sw + kj) as isize - pw_s;
                            if iw >= 0 && iw < in_w_s {
                                unsafe {
                                    *gi_batch.get_unchecked_mut(gi_row + iw as usize)
                                        += *grad_col.get_unchecked(col_row_base + ow);
                                }
                            }
                        }
                    }
                }

                (gi_batch, local_grad_weight)
            })
            .collect();

        // Assemble grad_input (concatenate) and reduce grad_weight (sum)
        let mut grad_input = Vec::with_capacity(batch_size * in_per_batch);
        let mut grad_weight = vec![0.0f32; out_channels * col_rows];
        for (gi_batch, local_gw) in &per_batch_results {
            grad_input.extend_from_slice(gi_batch);
            for (w, lw) in grad_weight.iter_mut().zip(local_gw.iter()) {
                *w += *lw;
            }
        }

        let grad_input_tensor = Tensor::from_vec(grad_input, &self.input_shape).unwrap();
        let grad_weight_tensor = Tensor::from_vec(
            grad_weight, &[self.out_channels, self.in_channels, kh, kw],
        ).unwrap();

        let mut result = vec![Some(grad_input_tensor), Some(grad_weight_tensor)];

        // === d_bias: sum over (N, H_out, W_out) ===
        if self.has_bias {
            let mut grad_bias = vec![0.0f32; self.out_channels];
            for b in 0..batch_size {
                let go_offset = b * self.out_channels * out_hw;
                for oc in 0..self.out_channels {
                    let start = go_offset + oc * out_hw;
                    grad_bias[oc] += grad_out_vec[start..start + out_hw].iter().sum::<f32>();
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
    /// Creates a new GroupedConv2dBackward gradient function.
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
        let out_hw = out_h * out_w;
        let col_rows_g = ic_per_group * kh * kw; // columns per group

        // Use im2col + GEMM per group for efficient backward (Rayon-parallelized across batch)
        let in_per_batch = self.in_channels * in_h * in_w;
        let out_channels = self.out_channels;
        let groups = self.groups;
        let weight_total = out_channels * ic_per_group * kh * kw;

        let per_batch_results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut gi_batch = vec![0.0f32; in_per_batch];
                let mut local_grad_weight = vec![0.0f32; weight_total];

                for g in 0..groups {
                    let ic_start = g * ic_per_group;
                    let oc_start = g * oc_per_group;

                    // im2col for this group's input channels
                    let mut col = vec![0.0f32; col_rows_g * out_hw];
                    for c_local in 0..ic_per_group {
                        let c = ic_start + c_local;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let col_row = c_local * kh * kw + ki * kw + kj;
                                for oh in 0..out_h {
                                    for ow in 0..out_w {
                                        let ih = (oh * sh + ki) as isize - ph as isize;
                                        let iw = (ow * sw + kj) as isize - pw as isize;
                                        let val = if ih >= 0 && (ih as usize) < in_h
                                            && iw >= 0 && (iw as usize) < in_w
                                        {
                                            input_vec[b * in_per_batch
                                                + c * in_h * in_w + ih as usize * in_w + iw as usize]
                                        } else { 0.0 };
                                        col[col_row * out_hw + oh * out_w + ow] = val;
                                    }
                                }
                            }
                        }
                    }

                    // grad_out slice for this group
                    let mut go_group = vec![0.0f32; oc_per_group * out_hw];
                    for oc_local in 0..oc_per_group {
                        let oc = oc_start + oc_local;
                        let src_off = b * out_channels * out_hw + oc * out_hw;
                        go_group[oc_local * out_hw..(oc_local + 1) * out_hw]
                            .copy_from_slice(&grad_out_vec[src_off..src_off + out_hw]);
                    }

                    // grad_weight[group] += go_group × col^T
                    let w_offset = oc_start * ic_per_group * kh * kw;
                    gemm_acc(&go_group, &col,
                        &mut local_grad_weight[w_offset..w_offset + oc_per_group * col_rows_g],
                        oc_per_group, out_hw, col_rows_g, false, true);

                    // grad_col = weight[group]^T × go_group
                    let w_group = &weight_vec[w_offset..w_offset + oc_per_group * col_rows_g];
                    let mut grad_col = vec![0.0f32; col_rows_g * out_hw];
                    gemm_acc(w_group, &go_group, &mut grad_col,
                        col_rows_g, oc_per_group, out_hw, true, false);

                    // col2im: scatter grad_col back
                    for c_local in 0..ic_per_group {
                        let c = ic_start + c_local;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let col_row = c_local * kh * kw + ki * kw + kj;
                                for oh in 0..out_h {
                                    for ow in 0..out_w {
                                        let ih = (oh * sh + ki) as isize - ph as isize;
                                        let iw = (ow * sw + kj) as isize - pw as isize;
                                        if ih >= 0 && (ih as usize) < in_h
                                            && iw >= 0 && (iw as usize) < in_w
                                        {
                                            gi_batch[c * in_h * in_w
                                                + ih as usize * in_w + iw as usize]
                                                += grad_col[col_row * out_hw + oh * out_w + ow];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                (gi_batch, local_grad_weight)
            })
            .collect();

        // Assemble grad_input (concatenate) and reduce grad_weight (sum)
        let mut grad_input = Vec::with_capacity(batch_size * in_per_batch);
        let mut grad_weight = vec![0.0f32; weight_total];
        for (gi_batch, local_gw) in &per_batch_results {
            grad_input.extend_from_slice(gi_batch);
            for (w, lw) in grad_weight.iter_mut().zip(local_gw.iter()) {
                *w += *lw;
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
                let go_offset = b * self.out_channels * out_hw;
                for oc in 0..self.out_channels {
                    let start = go_offset + oc * out_hw;
                    grad_bias[oc] += grad_out_vec[start..start + out_hw].iter().sum::<f32>();
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
    _num_features: usize,
}

impl BatchNorm2dBackward {
    /// Creates a new BatchNorm2dBackward gradient function.
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
            saved_weight, eps, _num_features: num_features,
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
            // Cache x_hat to avoid recomputing in second pass
            let mut sum_grad = 0.0f32;
            let mut sum_grad_xhat = 0.0f32;
            let mut x_hat_cache = vec![0.0f32; batch * spatial];

            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = (input_vec[idx] - mean_c) * std_inv;
                    x_hat_cache[b_idx * spatial + s] = x_hat;
                    let g = grad_vec[idx];

                    grad_bias[c] += g;
                    grad_weight[c] += g * x_hat;

                    sum_grad += g;
                    sum_grad_xhat += g * x_hat;
                }
            }

            // d_input = weight * std_inv / N * (N * grad - sum_grad - x_hat * sum_grad_xhat)
            let scale = weight_c * std_inv / n;
            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = x_hat_cache[b_idx * spatial + s];
                    let g = grad_vec[idx];

                    grad_input[idx] = scale
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
    _num_features: usize,
}

impl BatchNorm1dBackward {
    /// Creates a new BatchNorm1dBackward gradient function.
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
            saved_weight, eps, _num_features: num_features,
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
            let mut x_hat_cache = vec![0.0f32; batch * spatial];

            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = (input_vec[idx] - mean_c) * std_inv;
                    x_hat_cache[b_idx * spatial + s] = x_hat;
                    let g = grad_vec[idx];

                    grad_bias[c] += g;
                    grad_weight[c] += g * x_hat;

                    sum_grad += g;
                    sum_grad_xhat += g * x_hat;
                }
            }

            let scale = weight_c * std_inv / n;
            for b_idx in 0..batch {
                for s in 0..spatial {
                    let idx = b_idx * channels * spatial + c * spatial + s;
                    let x_hat = x_hat_cache[b_idx * spatial + s];
                    let g = grad_vec[idx];

                    grad_input[idx] = scale
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
        let col_rows = self.in_channels * ks; // im2col column height

        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();

        // Use im2col + GEMM approach (same pattern as Conv2d backward)
        // im2col unfolds 1D patches: col[c*ks+k, ol] = input[c, ol*stride+k-padding]
        // grad_weight = grad_out × col^T  (GEMM)
        // grad_col = weight^T × grad_out  (GEMM)
        // grad_input = col2im(grad_col)

        let in_per_batch = self.in_channels * in_length;
        let out_channels = self.out_channels;
        let in_channels = self.in_channels;
        let stride = self.stride;
        let padding = self.padding;

        // Parallel: each batch element computes its own grad_input + partial grad_weight
        let per_batch_results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let input_offset = b * in_per_batch;
                let mut col = vec![0.0f32; col_rows * out_length];
                for c in 0..in_channels {
                    for k in 0..ks {
                        let col_row = c * ks + k;
                        for ol in 0..out_length {
                            let il_signed = (ol * stride + k) as isize - padding as isize;
                            let val = if il_signed >= 0 && (il_signed as usize) < in_length {
                                input_vec[input_offset + c * in_length + il_signed as usize]
                            } else {
                                0.0
                            };
                            col[col_row * out_length + ol] = val;
                        }
                    }
                }

                let go_offset = b * out_channels * out_length;
                let go_slice = &grad_out_vec[go_offset..go_offset + out_channels * out_length];

                // Thread-local grad_weight
                let mut local_grad_weight = vec![0.0f32; out_channels * col_rows];
                gemm_acc(go_slice, &col, &mut local_grad_weight,
                    out_channels, out_length, col_rows, false, true);

                // grad_col = weight^T × grad_out
                let mut grad_col = vec![0.0f32; col_rows * out_length];
                gemm_acc(&weight_vec, go_slice, &mut grad_col,
                    col_rows, out_channels, out_length, true, false);

                // col2im → local grad_input
                let mut gi_batch = vec![0.0f32; in_per_batch];
                for c in 0..in_channels {
                    for k in 0..ks {
                        let col_row = c * ks + k;
                        for ol in 0..out_length {
                            let il_signed = (ol * stride + k) as isize - padding as isize;
                            if il_signed >= 0 && (il_signed as usize) < in_length {
                                gi_batch[c * in_length + il_signed as usize]
                                    += grad_col[col_row * out_length + ol];
                            }
                        }
                    }
                }

                (gi_batch, local_grad_weight)
            })
            .collect();

        // Assemble grad_input (concatenate) and reduce grad_weight (sum)
        let mut grad_input = Vec::with_capacity(batch_size * in_per_batch);
        let mut grad_weight = vec![0.0f32; out_channels * col_rows];
        for (gi_batch, local_gw) in &per_batch_results {
            grad_input.extend_from_slice(gi_batch);
            for (w, lw) in grad_weight.iter_mut().zip(local_gw.iter()) {
                *w += *lw;
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
                let go_offset = b * self.out_channels * out_length;
                for oc in 0..self.out_channels {
                    let start = go_offset + oc * out_length;
                    grad_bias[oc] += grad_out_vec[start..start + out_length].iter().sum::<f32>();
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
    _kernel_size: (usize, usize),
    _stride: (usize, usize),
    _padding: (usize, usize),
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
            _kernel_size: kernel_size,
            _stride: stride,
            _padding: padding,
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
                        // Compute count analytically instead of iterating kernel twice
                        let ih_start = (oh * sh).max(ph) - ph;
                        let ih_end = ((oh * sh + kh).min(in_h + ph)) - ph;
                        let iw_start = (ow * sw).max(pw) - pw;
                        let iw_end = ((ow * sw + kw).min(in_w + pw)) - pw;
                        let count = if ih_end > ih_start && iw_end > iw_start {
                            (ih_end - ih_start) * (iw_end - iw_start)
                        } else {
                            0
                        };

                        let go_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        let go_val = grad_out_vec[go_idx];
                        let grad_per_elem = if count > 0 { go_val / count as f32 } else { 0.0 };

                        // Single pass to scatter gradients
                        for ki in 0..kh {
                            let ih = oh * sh + ki;
                            if ih >= ph && ih < in_h + ph {
                                let actual_ih = ih - ph;
                                for kj in 0..kw {
                                    let iw = ow * sw + kj;
                                    if iw >= pw && iw < in_w + pw {
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
                    // Compute count analytically instead of iterating kernel twice
                    let il_begin = in_start.max(self.padding) - self.padding;
                    let il_end = ((in_start + self.kernel_size).min(in_length + self.padding)) - self.padding;
                    let count = if il_end > il_begin { il_end - il_begin } else { 0 };

                    let go_idx =
                        b * channels * out_length + c * out_length + ol;
                    let go_val = grad_out_vec[go_idx];
                    let grad_per_elem = if count > 0 { go_val / count as f32 } else { 0.0 };

                    // Single pass to scatter gradients
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
    _output_padding: (usize, usize),
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
            _output_padding: output_padding,
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
        let out_hw = out_h * out_w;

        let in_h = self.input_shape[2];
        let in_w = self.input_shape[3];
        let in_hw = in_h * in_w;
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let input_vec = self.saved_input.to_vec();
        let weight_vec = self.saved_weight.to_vec();

        // d_input: standard conv2d of grad_output with weight
        // weight shape: (in_channels, out_channels, kh, kw)
        // grad_input[b, ic, ih, iw] = sum_{oc,ki,kj} weight[ic,oc,ki,kj] * grad_out[b,oc,oh,ow]
        // where oh = ih*sh + ki - ph,  ow = iw*sw + kj - pw
        //
        // Cache-friendly: pre-compute base offsets, skip out-of-bounds early
        let mut grad_input = vec![0.0f32; batch_size * self.in_channels * in_hw];

        for b in 0..batch_size {
            let go_b = b * self.out_channels * out_hw;
            let gi_b = b * self.in_channels * in_hw;

            for ic in 0..self.in_channels {
                let w_ic = ic * self.out_channels * kh * kw;
                let gi_ic = gi_b + ic * in_hw;

                for ih in 0..in_h {
                    let oh_base = (ih * sh) as isize - ph as isize;

                    for iw in 0..in_w {
                        let ow_base = (iw * sw) as isize - pw as isize;
                        let mut sum = 0.0f32;

                        for ki in 0..kh {
                            let oh_signed = oh_base + ki as isize;
                            if oh_signed < 0 || oh_signed as usize >= out_h {
                                continue;
                            }
                            let oh = oh_signed as usize;

                            for kj in 0..kw {
                                let ow_signed = ow_base + kj as isize;
                                if ow_signed < 0 || ow_signed as usize >= out_w {
                                    continue;
                                }
                                let ow = ow_signed as usize;
                                let go_spatial = oh * out_w + ow;
                                let w_kij = w_ic + ki * kw + kj;

                                for oc in 0..self.out_channels {
                                    sum += grad_out_vec[go_b + oc * out_hw + go_spatial]
                                        * weight_vec[w_kij + oc * kh * kw];
                                }
                            }
                        }
                        grad_input[gi_ic + ih * in_w + iw] = sum;
                    }
                }
            }
        }

        // d_weight: accumulate over batch and spatial positions
        // grad_weight[ic, oc, ki, kj] += input[b, ic, ih, iw] * grad_out[b, oc, oh, ow]
        // where oh = ih*sh + ki - ph,  ow = iw*sw + kj - pw
        //
        // Loop order: batch -> ic -> spatial(ih,iw) -> kernel(ki,kj) -> oc
        // This keeps input access sequential and skips zero inputs
        let mut grad_weight = vec![0.0f32; self.in_channels * self.out_channels * kh * kw];

        for b in 0..batch_size {
            let in_b = b * self.in_channels * in_hw;
            let go_b = b * self.out_channels * out_hw;

            for ic in 0..self.in_channels {
                let in_ic = in_b + ic * in_hw;
                let gw_ic = ic * self.out_channels * kh * kw;

                for ih in 0..in_h {
                    let oh_base = (ih * sh) as isize - ph as isize;

                    for iw in 0..in_w {
                        let in_val = input_vec[in_ic + ih * in_w + iw];
                        if in_val == 0.0 {
                            continue; // skip zero inputs (common with ReLU)
                        }

                        let ow_base = (iw * sw) as isize - pw as isize;

                        for ki in 0..kh {
                            let oh_signed = oh_base + ki as isize;
                            if oh_signed < 0 || oh_signed as usize >= out_h {
                                continue;
                            }
                            let oh = oh_signed as usize;

                            for kj in 0..kw {
                                let ow_signed = ow_base + kj as isize;
                                if ow_signed < 0 || ow_signed as usize >= out_w {
                                    continue;
                                }
                                let ow = ow_signed as usize;
                                let go_spatial = oh * out_w + ow;
                                let gw_kij = gw_ic + ki * kw + kj;

                                for oc in 0..self.out_channels {
                                    grad_weight[gw_kij + oc * kh * kw]
                                        += in_val * grad_out_vec[go_b + oc * out_hw + go_spatial];
                                }
                            }
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
                let go_offset = b * self.out_channels * out_hw;
                for oc in 0..self.out_channels {
                    let start = go_offset + oc * out_hw;
                    grad_bias[oc] += grad_out_vec[start..start + out_hw].iter().sum::<f32>();
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
    /// Creates a new LayerNormBackward gradient function.
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
    /// Creates a new GroupNormBackward gradient function.
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
    /// Creates a new InstanceNorm2dBackward gradient function.
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
