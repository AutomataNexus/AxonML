//! Ternary Linear Layer - 1.58-bit Weight Quantization (BitNet b1.58)
//!
//! Implements TernaryLinear: a linear layer with ternary weights {-1, 0, +1}.
//! Weights are stored as packed 2-bit integers (4 weights per byte) for inference,
//! with full-precision shadow weights maintained during training.
//!
//! Forward pass uses absmean quantization:
//!   w_ternary = sign(w) * round(|w| / mean(|w|))
//!
//! The ternary matmul reduces to addition/subtraction — no multiply needed:
//!   y[i] = scale * (sum_{w=+1} x[j] - sum_{w=-1} x[j])
//!
//! # File
//! `crates/axonml-nn/src/layers/ternary.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 19, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::any::Any;
use std::collections::HashMap;

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_tensor::Tensor;

use crate::init::{kaiming_uniform, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// Packed Ternary Weights
// =============================================================================

/// Packed ternary weight storage: 4 weights per byte using 2-bit encoding.
///
/// Encoding: 0b00 = 0, 0b01 = +1, 0b10 = -1 (0b11 unused)
#[derive(Debug, Clone)]
pub struct PackedTernaryWeights {
    /// Packed bytes (4 weights per byte)
    data: Vec<u8>,
    /// Number of actual weight values
    num_weights: usize,
    /// Absmean scale factor
    scale: f32,
}

impl PackedTernaryWeights {
    /// Pack ternary values {-1, 0, +1} into 2-bit representation.
    pub fn pack(ternary_values: &[i8], scale: f32) -> Self {
        let num_weights = ternary_values.len();
        let num_bytes = (num_weights + 3) / 4;
        let mut data = vec![0u8; num_bytes];

        for (i, &val) in ternary_values.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let encoded = match val {
                0 => 0b00u8,
                1 => 0b01u8,
                -1 => 0b10u8,
                _ => 0b00u8, // Clamp invalid values to zero
            };
            data[byte_idx] |= encoded << bit_offset;
        }

        Self {
            data,
            num_weights,
            scale,
        }
    }

    /// Unpack to dense ternary values {-1, 0, +1}.
    pub fn unpack(&self) -> Vec<i8> {
        let mut values = Vec::with_capacity(self.num_weights);
        for i in 0..self.num_weights {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let encoded = (self.data[byte_idx] >> bit_offset) & 0b11;
            let val = match encoded {
                0b00 => 0i8,
                0b01 => 1i8,
                0b10 => -1i8,
                _ => 0i8,
            };
            values.push(val);
        }
        values
    }

    /// Returns the scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Returns the packed storage size in bytes.
    pub fn storage_bytes(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of weights.
    pub fn num_weights(&self) -> usize {
        self.num_weights
    }

    /// Count zeros (sparsity).
    pub fn count_zeros(&self) -> usize {
        let values = self.unpack();
        values.iter().filter(|&&v| v == 0).count()
    }
}

// =============================================================================
// TernaryLinear
// =============================================================================

/// A linear layer with 1.58-bit ternary weights (BitNet b1.58).
///
/// During training, full-precision shadow weights are maintained and quantized
/// to ternary {-1, 0, +1} on each forward pass using absmean quantization.
/// Gradients flow through the quantization via the Straight-Through Estimator (STE).
///
/// During inference, pre-quantized packed weights are used for efficient
/// addition/subtraction-only matmul.
///
/// # Architecture
/// - Shadow weights: fp32 (out_features x in_features), used during training
/// - Ternary weights: packed 2-bit (4 per byte), used during inference
/// - Scale factor: mean(|w|), applied after ternary matmul
/// - Bias: optional fp32
///
/// # Shape
/// - Input: (*, in_features)
/// - Output: (*, out_features)
///
/// # Example
/// ```ignore
/// let layer = TernaryLinear::new(512, 512);
/// let input = Variable::new(Tensor::randn(&[2, 512]), true);
/// let output = layer.forward(&input);  // Shape: [2, 512]
/// ```
pub struct TernaryLinear {
    /// Shadow weight (fp32) for training — holds the latent continuous weights.
    pub shadow_weight: Parameter,
    /// Optional bias (fp32).
    pub bias: Option<Parameter>,
    /// Pre-quantized packed weights for inference.
    packed_weights: Option<PackedTernaryWeights>,
    /// Input features.
    in_features: usize,
    /// Output features.
    out_features: usize,
    /// Whether to use packed inference mode.
    inference_mode: bool,
}

impl TernaryLinear {
    /// Creates a new TernaryLinear layer with bias.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, true)
    }

    /// Creates a new TernaryLinear layer with optional bias.
    pub fn with_bias(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight_data = kaiming_uniform(out_features, in_features);
        let shadow_weight = Parameter::named("shadow_weight", weight_data, true);

        let bias_param = if bias {
            let bias_data = zeros(&[out_features]);
            Some(Parameter::named("bias", bias_data, true))
        } else {
            None
        };

        Self {
            shadow_weight,
            bias: bias_param,
            packed_weights: None,
            in_features,
            out_features,
            inference_mode: false,
        }
    }

    /// Returns the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Quantize shadow weights to ternary using absmean quantization.
    ///
    /// w_ternary = sign(w) * round(|w| / mean(|w|))
    ///
    /// Returns (ternary values as i8, scale factor).
    pub fn quantize_weights(&self) -> (Vec<i8>, f32) {
        let w = self.shadow_weight.data();
        let w_vec = w.to_vec();
        let n = w_vec.len();

        // Compute absmean scale
        let abs_mean: f32 = w_vec.iter().map(|v| v.abs()).sum::<f32>() / n as f32;
        let scale = abs_mean.max(1e-8); // Avoid division by zero

        // Quantize: sign(w) * round(|w| / scale)
        let ternary: Vec<i8> = w_vec
            .iter()
            .map(|&w| {
                let normalized = (w.abs() / scale).round().min(1.0);
                let sign = if w > 0.0 {
                    1i8
                } else if w < 0.0 {
                    -1i8
                } else {
                    0i8
                };
                sign * (normalized as i8)
            })
            .collect();

        (ternary, scale)
    }

    /// Pre-quantize weights for inference (pack to 2-bit representation).
    pub fn quantize_for_inference(&mut self) {
        let (ternary, scale) = self.quantize_weights();
        self.packed_weights = Some(PackedTernaryWeights::pack(&ternary, scale));
        self.inference_mode = true;
    }

    /// Switch back to training mode (use shadow weights).
    pub fn use_shadow_weights(&mut self) {
        self.inference_mode = false;
    }

    /// Get weight sparsity (fraction of zeros in ternary representation).
    pub fn weight_sparsity(&self) -> f32 {
        let (ternary, _) = self.quantize_weights();
        let zeros = ternary.iter().filter(|&&v| v == 0).count();
        zeros as f32 / ternary.len() as f32
    }

    /// Get compression ratio vs fp32.
    pub fn compression_ratio(&self) -> f32 {
        let fp32_bytes = self.in_features * self.out_features * 4;
        let ternary_bytes = (self.in_features * self.out_features + 3) / 4 + 4; // +4 for scale
        fp32_bytes as f32 / ternary_bytes as f32
    }

    /// Get the packed weight storage if quantized.
    pub fn packed_weights(&self) -> Option<&PackedTernaryWeights> {
        self.packed_weights.as_ref()
    }

    /// Perform ternary matmul: y = scale * (sum_positive - sum_negative).
    ///
    /// For each output element, we sum input values where the ternary weight is +1,
    /// subtract input values where the ternary weight is -1, and multiply by scale.
    /// This is pure addition/subtraction — no floating-point multiply for the matmul itself.
    fn ternary_matmul(
        input: &[f32],
        ternary: &[i8],
        scale: f32,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * out_features];

        for b in 0..batch_size {
            let x_off = b * in_features;
            let y_off = b * out_features;

            for o in 0..out_features {
                let w_off = o * in_features;
                let mut sum_pos = 0.0f32;
                let mut sum_neg = 0.0f32;

                for j in 0..in_features {
                    let w = ternary[w_off + j];
                    let x = input[x_off + j];
                    if w == 1 {
                        sum_pos += x;
                    } else if w == -1 {
                        sum_neg += x;
                    }
                    // w == 0: skip (zero contribution)
                }

                output[y_off + o] = scale * (sum_pos - sum_neg);
            }
        }

        output
    }

    /// Forward pass during training: quantize-on-the-fly with STE backward.
    fn forward_training(&self, input: &Variable) -> Variable {
        let input_data = input.data();
        let input_shape = input_data.shape();
        let batch_dims: Vec<usize> = input_shape[..input_shape.len() - 1].to_vec();
        let total_batch: usize = batch_dims.iter().product();

        // Quantize shadow weights to ternary
        let (ternary, scale) = self.quantize_weights();

        // Flatten input to 2D
        let input_vec = input_data.to_vec();

        // Ternary matmul
        let output_vec =
            Self::ternary_matmul(&input_vec, &ternary, scale, total_batch, self.in_features, self.out_features);

        // Build output tensor
        let mut out_shape = batch_dims.clone();
        out_shape.push(self.out_features);
        let output_tensor = Tensor::from_vec(output_vec, &out_shape).unwrap();

        // Add bias
        let output_tensor = if let Some(ref bias) = self.bias {
            let bias_vec = bias.data().to_vec();
            let mut out = output_tensor.to_vec();
            for b in 0..total_batch {
                for o in 0..self.out_features {
                    out[b * self.out_features + o] += bias_vec[o];
                }
            }
            Tensor::from_vec(out, &out_shape).unwrap()
        } else {
            output_tensor
        };

        let requires_grad = input.requires_grad() && is_grad_enabled();
        if requires_grad {
            // STE backward: gradients pass through quantization as if it were identity.
            // The gradient w.r.t. shadow_weight is computed as if the ternary quantization
            // were not there (straight-through estimator).
            let saved_input = input_data.clone();
            let saved_ternary = ternary;
            let saved_scale = scale;
            let in_f = self.in_features;
            let out_f = self.out_features;
            let shadow_grad_fn = self.shadow_weight.variable().grad_fn().cloned();
            let bias_grad_fn = self.bias.as_ref().and_then(|b| b.variable().grad_fn().cloned());

            let mut next_fns = vec![input.grad_fn().cloned(), shadow_grad_fn];
            if bias_grad_fn.is_some() {
                next_fns.push(bias_grad_fn);
            }

            let grad_fn = GradFn::new(TernaryLinearBackward {
                next_fns,
                saved_input,
                saved_ternary,
                saved_scale,
                in_features: in_f,
                out_features: out_f,
                has_bias: self.bias.is_some(),
                total_batch,
            });
            Variable::from_operation(output_tensor, grad_fn, true)
        } else {
            Variable::new(output_tensor, false)
        }
    }

    /// Forward pass during inference: use pre-quantized packed weights.
    fn forward_inference(&self, input: &Variable) -> Variable {
        let packed = self
            .packed_weights
            .as_ref()
            .expect("Must call quantize_for_inference() before inference forward");

        let input_data = input.data();
        let input_shape = input_data.shape();
        let batch_dims: Vec<usize> = input_shape[..input_shape.len() - 1].to_vec();
        let total_batch: usize = batch_dims.iter().product();

        // Unpack ternary weights
        let ternary = packed.unpack();
        let scale = packed.scale();

        let input_vec = input_data.to_vec();
        let output_vec =
            Self::ternary_matmul(&input_vec, &ternary, scale, total_batch, self.in_features, self.out_features);

        let mut out_shape = batch_dims;
        out_shape.push(self.out_features);
        let mut output_tensor = Tensor::from_vec(output_vec, &out_shape).unwrap();

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_vec = bias.data().to_vec();
            let mut out = output_tensor.to_vec();
            for b in 0..total_batch {
                for o in 0..self.out_features {
                    out[b * self.out_features + o] += bias_vec[o];
                }
            }
            output_tensor = Tensor::from_vec(out, &out_shape).unwrap();
        }

        Variable::new(output_tensor, false)
    }
}

impl Module for TernaryLinear {
    fn forward(&self, input: &Variable) -> Variable {
        if self.inference_mode {
            self.forward_inference(input)
        } else {
            self.forward_training(input)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.shadow_weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("shadow_weight".to_string(), self.shadow_weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "TernaryLinear"
    }
}

impl std::fmt::Debug for TernaryLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TernaryLinear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bias", &self.bias.is_some())
            .field("inference_mode", &self.inference_mode)
            .finish()
    }
}

// =============================================================================
// TernaryLinearBackward (Straight-Through Estimator)
// =============================================================================

/// Gradient function for TernaryLinear using the Straight-Through Estimator.
///
/// The STE passes gradients through the ternary quantization as if it were
/// an identity function. This allows training the shadow weights with standard
/// gradient-based optimizers.
///
/// For the forward y = scale * T(W) @ x where T is ternary quantization:
/// - grad_input = scale * T(W)^T @ grad_output  (ternary transpose matmul)
/// - grad_weight = grad_output^T @ x             (STE: treat T as identity)
/// - grad_bias = sum(grad_output, dim=0)
#[derive(Debug)]
struct TernaryLinearBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    saved_ternary: Vec<i8>,
    saved_scale: f32,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    total_batch: usize,
}

impl GradientFunction for TernaryLinearBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let g_vec = grad_output.to_vec();
        let x_vec = self.saved_input.to_vec();

        // 1. grad_input = scale * ternary_W^T @ grad_output
        //    For each batch element and input dimension:
        //    grad_input[b,j] = scale * sum_o(ternary[o,j] * grad_output[b,o])
        let mut grad_input = vec![0.0f32; self.total_batch * self.in_features];
        for b in 0..self.total_batch {
            let g_off = b * self.out_features;
            let gi_off = b * self.in_features;

            for j in 0..self.in_features {
                let mut sum = 0.0f32;
                for o in 0..self.out_features {
                    let w = self.saved_ternary[o * self.in_features + j];
                    if w == 1 {
                        sum += g_vec[g_off + o];
                    } else if w == -1 {
                        sum -= g_vec[g_off + o];
                    }
                }
                grad_input[gi_off + j] = self.saved_scale * sum;
            }
        }

        let gi_tensor =
            Tensor::from_vec(grad_input, self.saved_input.shape()).unwrap();

        // 2. grad_weight (STE): grad_output^T @ input
        //    grad_weight[o,j] = sum_b(grad_output[b,o] * input[b,j])
        let mut grad_weight = vec![0.0f32; self.out_features * self.in_features];
        for b in 0..self.total_batch {
            let g_off = b * self.out_features;
            let x_off = b * self.in_features;

            for o in 0..self.out_features {
                let go = g_vec[g_off + o];
                let w_off = o * self.in_features;
                for j in 0..self.in_features {
                    grad_weight[w_off + j] += go * x_vec[x_off + j];
                }
            }
        }
        let gw_tensor =
            Tensor::from_vec(grad_weight, &[self.out_features, self.in_features]).unwrap();

        let mut results: Vec<Option<Tensor<f32>>> = vec![Some(gi_tensor), Some(gw_tensor)];

        // 3. grad_bias = sum(grad_output, dim=0)
        if self.has_bias {
            let mut grad_bias = vec![0.0f32; self.out_features];
            for b in 0..self.total_batch {
                for o in 0..self.out_features {
                    grad_bias[o] += g_vec[b * self.out_features + o];
                }
            }
            let gb_tensor = Tensor::from_vec(grad_bias, &[self.out_features]).unwrap();
            results.push(Some(gb_tensor));
        }

        results
    }

    fn name(&self) -> &'static str {
        "TernaryLinearBackward"
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
    fn test_ternary_linear_creation() {
        let layer = TernaryLinear::new(64, 32);
        assert_eq!(layer.in_features(), 64);
        assert_eq!(layer.out_features(), 32);
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_ternary_linear_no_bias() {
        let layer = TernaryLinear::with_bias(64, 32, false);
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_ternary_linear_forward() {
        let layer = TernaryLinear::new(8, 4);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 16], &[2, 8]).unwrap(),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![2, 4]);
    }

    #[test]
    fn test_ternary_quantization() {
        let layer = TernaryLinear::new(16, 8);
        let (ternary, scale) = layer.quantize_weights();

        // All values should be in {-1, 0, +1}
        for &v in &ternary {
            assert!(v == -1 || v == 0 || v == 1, "got {}", v);
        }

        // Scale should be positive
        assert!(scale > 0.0);

        // Should have the right number of values
        assert_eq!(ternary.len(), 16 * 8);
    }

    #[test]
    fn test_packed_ternary_roundtrip() {
        let values: Vec<i8> = vec![1, 0, -1, 1, 0, 0, -1, -1, 1, 0];
        let packed = PackedTernaryWeights::pack(&values, 0.5);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
        assert_eq!(packed.scale(), 0.5);
    }

    #[test]
    fn test_packed_storage_compression() {
        let n = 1024;
        let values: Vec<i8> = (0..n).map(|i| ((i % 3) as i8) - 1).collect();
        let packed = PackedTernaryWeights::pack(&values, 1.0);
        // 1024 weights / 4 per byte = 256 bytes (vs 1024 * 4 = 4096 bytes fp32)
        assert_eq!(packed.storage_bytes(), 256);
    }

    #[test]
    fn test_ternary_matmul_simple() {
        // 2x3 ternary weight: [[1, -1, 0], [0, 1, 1]]
        let ternary = vec![1i8, -1, 0, 0, 1, 1];
        let scale = 1.0;
        let input = vec![2.0f32, 3.0, 5.0]; // 1x3

        let output = TernaryLinear::ternary_matmul(&input, &ternary, scale, 1, 3, 2);

        // y[0] = 1.0 * (2.0 - 3.0) = -1.0
        // y[1] = 1.0 * (3.0 + 5.0) = 8.0
        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_linear_inference_mode() {
        let mut layer = TernaryLinear::new(8, 4);

        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 8], &[1, 8]).unwrap(),
            false,
        );

        // Training forward
        let train_out = layer.forward(&input);

        // Quantize and switch to inference
        layer.quantize_for_inference();
        let infer_out = layer.forward(&input);

        // Should produce the same result
        let train_vec = train_out.data().to_vec();
        let infer_vec = infer_out.data().to_vec();
        for (a, b) in train_vec.iter().zip(infer_vec.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Training {} vs inference {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_ternary_linear_sparsity() {
        let layer = TernaryLinear::new(64, 32);
        let sparsity = layer.weight_sparsity();
        // Sparsity should be between 0 and 1
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
    }

    #[test]
    fn test_ternary_linear_compression_ratio() {
        let layer = TernaryLinear::new(512, 512);
        let ratio = layer.compression_ratio();
        // Should be close to 16x (32 bits / 2 bits)
        assert!(ratio > 14.0 && ratio < 17.0, "ratio = {}", ratio);
    }

    #[test]
    fn test_ternary_linear_parameters() {
        let layer = TernaryLinear::new(16, 8);
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // shadow_weight + bias

        let layer_no_bias = TernaryLinear::with_bias(16, 8, false);
        assert_eq!(layer_no_bias.parameters().len(), 1);
    }

    #[test]
    fn test_ternary_linear_backward() {
        let layer = TernaryLinear::new(4, 2);

        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap(),
            true,
        );
        let output = layer.forward(&input);
        let loss = output.sum();
        loss.backward();

        // Gradients should exist
        assert!(input.grad().is_some());
    }
}
