//! cuDNN-accelerated conv2d operations — forward, backward-data, backward-filter.
//!
//! Three public functions wrapping cuDNN 8 via cudarc's Cudnn bindings:
//! `cudnn_conv2d_forward` (auto-selects best algorithm, optional bias add via
//! CUDA kernel, Tensor Core math when available), `cudnn_conv2d_backward_data`
//! (gradient w.r.t. input), `cudnn_conv2d_backward_filter` (gradient w.r.t.
//! weight). All three handle grouped convolution, return `Option<CudaSlice<f32>>`
//! (caller falls back to im2col+GEMM or CPU if cuDNN fails), and allocate
//! workspace on the fly. Feature-gated behind `cudnn`.
//!
//! # File
//! `crates/axonml-core/src/backends/cudnn_ops.rs`
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

#[cfg(feature = "cudnn")]
use cudarc::cudnn::sys::{cudnnConvolutionMode_t, cudnnTensorFormat_t};
#[cfg(feature = "cudnn")]
use cudarc::cudnn::{ConvBackwardData, ConvBackwardFilter, ConvForward, Cudnn};
#[cfg(feature = "cudnn")]
use cudarc::driver::{CudaSlice, CudaStream};
#[cfg(feature = "cudnn")]
use std::sync::Arc;

#[cfg(feature = "cudnn")]
use super::cuda::CudaBackend;

// =============================================================================
// CUDNN Conv2d Forward
// =============================================================================

/// Performs a conv2d forward pass using cuDNN.
///
/// All data must already be on GPU as `CudaSlice<f32>`.
/// Returns the output `CudaSlice<f32>` of shape `[N, C_out, H_out, W_out]`.
///
/// # Arguments
/// * `cudnn` - The cuDNN handle
/// * `stream` - The CUDA stream
/// * `input` - Input data `[N, C_in, H, W]` on GPU
/// * `weight` - Filter data `[C_out, C_in/groups, kH, kW]` on GPU
/// * `bias` - Optional bias `[C_out]` on GPU
/// * `batch_size` - N
/// * `in_channels` - C_in
/// * `in_height` - H
/// * `in_width` - W
/// * `out_channels` - C_out
/// * `kernel_h` - kH
/// * `kernel_w` - kW
/// * `stride` - (stride_h, stride_w)
/// * `padding` - (pad_h, pad_w)
/// * `groups` - Number of convolution groups
///
/// Returns `None` if any cuDNN operation fails (caller should fall back to
/// im2col+GEMM or CPU path).
#[cfg(feature = "cudnn")]
#[allow(clippy::too_many_arguments)]
pub fn cudnn_conv2d_forward(
    cudnn: &Arc<Cudnn>,
    stream: &Arc<CudaStream>,
    cuda_backend: &CudaBackend,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    batch_size: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Option<CudaSlice<f32>> {
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let out_h = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_w = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    let n = batch_size as i32;
    let c_in = in_channels as i32;
    let h = in_height as i32;
    let w = in_width as i32;
    let c_out = out_channels as i32;
    let kh = kernel_h as i32;
    let kw = kernel_w as i32;
    let oh = out_h as i32;
    let ow = out_w as i32;

    // Create descriptors
    let x_desc = cudnn
        .create_4d_tensor::<f32>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, [n, c_in, h, w])
        .ok()?;

    let c_in_per_group = c_in / groups as i32;
    let filter_desc = cudnn
        .create_4d_filter::<f32>(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [c_out, c_in_per_group, kh, kw],
        )
        .ok()?;

    let y_desc = cudnn
        .create_4d_tensor::<f32>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, [n, c_out, oh, ow])
        .ok()?;

    let mut conv_desc = cudnn
        .create_conv2d::<f32>(
            [pad_h as i32, pad_w as i32],
            [stride_h as i32, stride_w as i32],
            [1, 1], // dilation
            cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .ok()?;

    if groups > 1 {
        conv_desc.set_group_count(groups as i32).ok()?;
    }

    // Use Tensor Core math if available
    conv_desc
        .set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_DEFAULT_MATH)
        .ok()?;

    let op = ConvForward {
        conv: &conv_desc,
        x: &x_desc,
        w: &filter_desc,
        y: &y_desc,
    };

    // Pick best algorithm
    let algo = op.pick_algorithm().ok()?;

    // Get workspace size and allocate
    let workspace_size = op.get_workspace_size(algo).ok()?;
    let mut workspace = stream.alloc_zeros::<u8>(workspace_size.max(1)).ok()?;

    // Allocate output
    let total_out = batch_size * out_channels * out_h * out_w;
    let mut output = stream.alloc_zeros::<f32>(total_out).ok()?;

    // Launch conv forward
    unsafe {
        op.launch(
            algo,
            Some(&mut workspace),
            (1.0f32, 0.0f32),
            input,
            weight,
            &mut output,
        )
        .ok()?;
    }

    // Add bias if present using the existing CUDA kernel
    if let Some(bias_data) = bias {
        let spatial = out_h * out_w;
        cuda_backend
            .bias_add_channels_f32(&mut output, bias_data, spatial, total_out)
            .ok()?;
    }

    Some(output)
}

// =============================================================================
// CUDNN Conv2d Backward Data (gradient w.r.t. input)
// =============================================================================

/// Computes the gradient of the input tensor for a conv2d backward pass using cuDNN.
///
/// Returns `grad_input` as a `CudaSlice<f32>` of shape `[N, C_in, H, W]`.
#[cfg(feature = "cudnn")]
#[allow(clippy::too_many_arguments)]
pub fn cudnn_conv2d_backward_data(
    cudnn: &Arc<Cudnn>,
    stream: &Arc<CudaStream>,
    grad_output: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    batch_size: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Option<CudaSlice<f32>> {
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;

    let n = batch_size as i32;
    let c_in = in_channels as i32;
    let h = in_height as i32;
    let w = in_width as i32;
    let c_out = out_channels as i32;
    let kh = kernel_h as i32;
    let kw = kernel_w as i32;
    let oh = out_h as i32;
    let ow = out_w as i32;

    let c_in_per_group = c_in / groups as i32;

    let dx_desc = cudnn
        .create_4d_tensor::<f32>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, [n, c_in, h, w])
        .ok()?;

    let filter_desc = cudnn
        .create_4d_filter::<f32>(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [c_out, c_in_per_group, kh, kw],
        )
        .ok()?;

    let dy_desc = cudnn
        .create_4d_tensor::<f32>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, [n, c_out, oh, ow])
        .ok()?;

    let mut conv_desc = cudnn
        .create_conv2d::<f32>(
            [pad_h as i32, pad_w as i32],
            [stride_h as i32, stride_w as i32],
            [1, 1],
            cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .ok()?;

    if groups > 1 {
        conv_desc.set_group_count(groups as i32).ok()?;
    }

    let op = ConvBackwardData {
        conv: &conv_desc,
        dx: &dx_desc,
        w: &filter_desc,
        dy: &dy_desc,
    };

    let algo = op.pick_algorithm().ok()?;
    let workspace_size = op.get_workspace_size(algo).ok()?;
    let mut workspace = stream.alloc_zeros::<u8>(workspace_size.max(1)).ok()?;

    let total_in = batch_size * in_channels * in_height * in_width;
    let mut grad_input = stream.alloc_zeros::<f32>(total_in).ok()?;

    unsafe {
        op.launch(
            algo,
            Some(&mut workspace),
            (1.0f32, 0.0f32),
            &mut grad_input,
            weight,
            grad_output,
        )
        .ok()?;
    }

    Some(grad_input)
}

// =============================================================================
// CUDNN Conv2d Backward Filter (gradient w.r.t. weight)
// =============================================================================

/// Computes the gradient of the filters for a conv2d backward pass using cuDNN.
///
/// Returns `grad_weight` as a `CudaSlice<f32>` of shape `[C_out, C_in/groups, kH, kW]`.
#[cfg(feature = "cudnn")]
#[allow(clippy::too_many_arguments)]
pub fn cudnn_conv2d_backward_filter(
    cudnn: &Arc<Cudnn>,
    stream: &Arc<CudaStream>,
    grad_output: &CudaSlice<f32>,
    input: &CudaSlice<f32>,
    batch_size: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Option<CudaSlice<f32>> {
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;

    let n = batch_size as i32;
    let c_in = in_channels as i32;
    let h = in_height as i32;
    let w = in_width as i32;
    let c_out = out_channels as i32;
    let kh = kernel_h as i32;
    let kw = kernel_w as i32;
    let oh = out_h as i32;
    let ow = out_w as i32;

    let c_in_per_group = c_in / groups as i32;

    let x_desc = cudnn
        .create_4d_tensor::<f32>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, [n, c_in, h, w])
        .ok()?;

    let dw_desc = cudnn
        .create_4d_filter::<f32>(
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [c_out, c_in_per_group, kh, kw],
        )
        .ok()?;

    let dy_desc = cudnn
        .create_4d_tensor::<f32>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, [n, c_out, oh, ow])
        .ok()?;

    let mut conv_desc = cudnn
        .create_conv2d::<f32>(
            [pad_h as i32, pad_w as i32],
            [stride_h as i32, stride_w as i32],
            [1, 1],
            cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .ok()?;

    if groups > 1 {
        conv_desc.set_group_count(groups as i32).ok()?;
    }

    let op = ConvBackwardFilter {
        conv: &conv_desc,
        x: &x_desc,
        dw: &dw_desc,
        dy: &dy_desc,
    };

    let algo = op.pick_algorithm().ok()?;
    let workspace_size = op.get_workspace_size(algo).ok()?;
    let mut workspace = stream.alloc_zeros::<u8>(workspace_size.max(1)).ok()?;

    let weight_size = out_channels * (in_channels / groups) * kernel_h * kernel_w;
    let mut grad_weight = stream.alloc_zeros::<f32>(weight_size).ok()?;

    unsafe {
        op.launch(
            algo,
            Some(&mut workspace),
            (1.0f32, 0.0f32),
            input,
            &mut grad_weight,
            grad_output,
        )
        .ok()?;
    }

    Some(grad_weight)
}

// =============================================================================
// Stubs when cudnn feature is disabled
// =============================================================================

/// Stub - returns None when cudnn feature is disabled.
#[cfg(not(feature = "cudnn"))]
#[allow(clippy::too_many_arguments)]
pub fn cudnn_conv2d_forward(
    _input: &[f32],
    _weight: &[f32],
    _bias: Option<&[f32]>,
    _batch_size: usize,
    _in_channels: usize,
    _in_height: usize,
    _in_width: usize,
    _out_channels: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _stride: (usize, usize),
    _padding: (usize, usize),
    _groups: usize,
) -> Option<Vec<f32>> {
    None
}
