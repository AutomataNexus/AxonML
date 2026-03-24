//! State Space Model (SSM) - Mamba-style Selective Scan
//!
//! # File
//! `crates/axonml-llm/src/ssm.rs`
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

use axonml_autograd::no_grad::is_grad_enabled;
use axonml_autograd::{GradFn, GradientFunction, Variable};
use axonml_nn::{Linear, Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// SSM Configuration
// =============================================================================

/// Configuration for a selective SSM block.
#[derive(Debug, Clone)]
pub struct SSMConfig {
    /// Model dimension
    pub d_model: usize,
    /// SSM state expansion factor (state dimension)
    pub d_state: usize,
    /// Inner dimension (expansion ratio * d_model)
    pub d_inner: usize,
    /// 1D convolution kernel size
    pub d_conv: usize,
    /// Rank of dt projection
    pub dt_rank: usize,
}

impl SSMConfig {
    /// Create SSM config from model dimension with standard expansion.
    pub fn from_d_model(d_model: usize) -> Self {
        let d_inner = d_model * 2;
        let d_state = 16;
        Self {
            d_model,
            d_state,
            d_inner,
            d_conv: 4,
            dt_rank: d_model.div_ceil(16), // ceil(d_model / 16)
        }
    }
}

// =============================================================================
// Conv1D (depthwise)
// =============================================================================

/// Simple 1D depthwise convolution for SSM.
///
/// Operates on [batch, seq_len, channels] with kernel applied per-channel.
#[derive(Debug)]
pub struct DepthwiseConv1d {
    /// Weight: [channels, kernel_size]
    weight: Tensor<f32>,
    /// Bias: [channels]
    bias: Tensor<f32>,
    /// Kernel size
    kernel_size: usize,
    /// Number of channels
    channels: usize,
}

impl DepthwiseConv1d {
    /// Create new depthwise 1D convolution.
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        // Xavier uniform init
        let bound = (6.0 / (channels + kernel_size) as f32).sqrt();
        let n = channels * kernel_size;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42 + channels as u64);
        let weight_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-bound..bound)).collect();
        let bias_data = vec![0.0f32; channels];

        Self {
            weight: Tensor::from_vec(weight_data, &[channels, kernel_size]).unwrap(),
            bias: Tensor::from_vec(bias_data, &[channels]).unwrap(),
            kernel_size,
            channels,
        }
    }

    /// Forward pass: [batch, seq_len, channels] -> [batch, seq_len, channels]
    ///
    /// Causal convolution with left padding.
    pub fn forward(&self, x: &Variable) -> Variable {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        assert_eq!(channels, self.channels);

        let x_vec = x_data.to_vec();
        let w_vec = self.weight.to_vec();
        let b_vec = self.bias.to_vec();
        let pad = self.kernel_size - 1; // causal: left padding

        let mut output = vec![0.0f32; batch_size * seq_len * channels];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for c in 0..channels {
                    let mut val = b_vec[c];
                    for k in 0..self.kernel_size {
                        let input_pos = s as isize + k as isize - pad as isize;
                        if input_pos >= 0 && (input_pos as usize) < seq_len {
                            let x_idx = (b * seq_len + input_pos as usize) * channels + c;
                            let w_idx = c * self.kernel_size + k;
                            val += x_vec[x_idx] * w_vec[w_idx];
                        }
                    }
                    output[(b * seq_len + s) * channels + c] = val;
                }
            }
        }

        let out_tensor = Tensor::from_vec(output, &[batch_size, seq_len, channels]).unwrap();

        let requires_grad = x.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(DepthwiseConv1dBackward {
                next_fns: vec![x.grad_fn().cloned()],
                saved_input: x_data.clone(),
                weight: self.weight.clone(),
                kernel_size: self.kernel_size,
            });
            Variable::from_operation(out_tensor, grad_fn, true)
        } else {
            Variable::new(out_tensor, false)
        }
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::named("weight", self.weight.clone(), true),
            Parameter::named("bias", self.bias.clone(), true),
        ]
    }
}

// =============================================================================
// DepthwiseConv1dBackward
// =============================================================================

#[derive(Debug)]
struct DepthwiseConv1dBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    weight: Tensor<f32>,
    kernel_size: usize,
}

impl GradientFunction for DepthwiseConv1dBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape = self.saved_input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        let pad = self.kernel_size - 1;

        let g_vec = grad_output.to_vec();
        let w_vec = self.weight.to_vec();
        let mut grad_input = vec![0.0f32; g_vec.len()];

        // grad_input[b, t, c] = sum_k w[c,k] * grad_out[b, t-k+pad, c]
        for b in 0..batch_size {
            for s in 0..seq_len {
                for c in 0..channels {
                    let mut val = 0.0f32;
                    for k in 0..self.kernel_size {
                        let out_pos = s as isize - k as isize + pad as isize;
                        if out_pos >= 0 && (out_pos as usize) < seq_len {
                            let g_idx = (b * seq_len + out_pos as usize) * channels + c;
                            let w_idx = c * self.kernel_size + k;
                            val += g_vec[g_idx] * w_vec[w_idx];
                        }
                    }
                    grad_input[(b * seq_len + s) * channels + c] = val;
                }
            }
        }

        let gi = Tensor::from_vec(grad_input, shape).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "DepthwiseConv1dBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Selective Scan
// =============================================================================

/// Selective scan (S6) — the core recurrence of Mamba-style SSMs.
///
/// For each channel independently:
///   h[t] = A_bar * h[t-1] + B_bar * x[t]
///   y[t] = C * h[t] + D * x[t]
/// where A_bar = exp(delta * A), B_bar = delta * B
///
/// Parameters A, B, C are input-dependent (selective), projected from x.
#[derive(Debug)]
pub struct SelectiveScan {
    /// Log of A parameter: [d_inner, d_state] (stored as log for stability)
    a_log: Tensor<f32>,
    /// D parameter (skip connection): [d_inner]
    d_param: Tensor<f32>,
    /// Projection from d_inner to dt, B, C
    x_proj: Linear,
    /// Projection for dt (from dt_rank to d_inner)
    dt_proj: Linear,
    /// State dimension
    d_state: usize,
    /// Inner dimension
    d_inner: usize,
    /// dt rank
    dt_rank: usize,
}

impl SelectiveScan {
    /// Create new selective scan.
    pub fn new(d_inner: usize, d_state: usize, dt_rank: usize) -> Self {
        // Initialize A as a range matrix (S4D real init)
        let mut a_data = vec![0.0f32; d_inner * d_state];
        for i in 0..d_inner {
            for j in 0..d_state {
                // A = -log(range(1, d_state+1)) repeated across d_inner
                a_data[i * d_state + j] = -((j + 1) as f32).ln();
            }
        }

        let d_data = vec![1.0f32; d_inner];

        Self {
            a_log: Tensor::from_vec(a_data, &[d_inner, d_state]).unwrap(),
            d_param: Tensor::from_vec(d_data, &[d_inner]).unwrap(),
            // x_proj: project x to (dt, B, C) = (dt_rank + 2*d_state)
            x_proj: Linear::new(d_inner, dt_rank + 2 * d_state),
            // dt_proj: project dt from dt_rank to d_inner
            dt_proj: Linear::new(dt_rank, d_inner),
            d_state,
            d_inner,
            dt_rank,
        }
    }

    /// Forward pass: [batch, seq_len, d_inner] -> [batch, seq_len, d_inner]
    pub fn forward(&self, x: &Variable) -> Variable {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_inner = shape[2];
        assert_eq!(d_inner, self.d_inner);

        // Project x to get dt, B, C
        let x_proj = self.x_proj.forward(x);
        // x_proj: [batch, seq_len, dt_rank + 2*d_state]

        // Split into dt, B, C using narrow
        let dt_raw = x_proj.narrow(2, 0, self.dt_rank);
        let b_var = x_proj.narrow(2, self.dt_rank, self.d_state);
        let c_var = x_proj.narrow(2, self.dt_rank + self.d_state, self.d_state);

        // Project dt to d_inner and apply softplus
        let dt_proj = self.dt_proj.forward(&dt_raw);
        // Softplus: log(1 + exp(x))
        let dt_data = dt_proj.data();
        let dt_vec = dt_data.to_vec();
        let dt_softplus: Vec<f32> = dt_vec
            .iter()
            .map(|&v| {
                if v > 20.0 {
                    v // numerical stability
                } else {
                    (1.0 + v.exp()).ln()
                }
            })
            .collect();
        let dt_tensor = Tensor::from_vec(dt_softplus, &[batch_size, seq_len, d_inner]).unwrap();

        // Get A (negative exponent)
        let a_vec = self.a_log.to_vec(); // [d_inner, d_state] - these are already -log values
        let a_exp: Vec<f32> = a_vec.iter().map(|&v| v.exp()).collect(); // A = exp(a_log), so A is negative

        let d_vec = self.d_param.to_vec();
        let b_data = b_var.data();
        let c_data = c_var.data();
        let x_vec = x_data.to_vec();
        let dt_vals = dt_tensor.to_vec();
        let b_vec = b_data.to_vec();
        let c_vec = c_data.to_vec();

        // Run selective scan recurrence
        let mut output = vec![0.0f32; batch_size * seq_len * d_inner];
        let d_state = self.d_state;

        for batch in 0..batch_size {
            // State: [d_inner, d_state]
            let mut h = vec![0.0f32; d_inner * d_state];

            for t in 0..seq_len {
                let bt_offset = (batch * seq_len + t) * d_inner;
                let bc_offset = (batch * seq_len + t) * d_state;

                for d in 0..d_inner {
                    let x_val = x_vec[bt_offset + d];
                    let dt_val = dt_vals[bt_offset + d];

                    let mut y_val = 0.0f32;

                    for s in 0..d_state {
                        let a_val = a_exp[d * d_state + s]; // exp(a_log) which is negative
                                                            // Clamp dt*A to prevent extreme values
                        let dt_a = (dt_val * a_val).clamp(-20.0, 0.0);
                        let a_bar = dt_a.exp(); // discretized A: exp(dt * A)
                        let b_val = b_vec[bc_offset + s];
                        let b_bar = dt_val * b_val;

                        // h[d,s] = A_bar * h[d,s] + B_bar * x
                        let h_idx = d * d_state + s;
                        h[h_idx] = a_bar * h[h_idx] + b_bar * x_val;
                        // Clamp state to prevent NaN
                        h[h_idx] = h[h_idx].clamp(-1e6, 1e6);

                        // y += C * h
                        let c_val = c_vec[bc_offset + s];
                        y_val += c_val * h[h_idx];
                    }

                    // Add skip connection: y += D * x
                    y_val += d_vec[d] * x_val;

                    // Clamp to prevent NaN/Inf
                    output[bt_offset + d] = y_val.clamp(-1e6, 1e6);
                }
            }
        }

        let out_tensor = Tensor::from_vec(output, &[batch_size, seq_len, d_inner]).unwrap();

        // For gradient flow, wrap with backward fn
        let requires_grad = x.requires_grad() && is_grad_enabled();
        if requires_grad {
            let grad_fn = GradFn::new(SelectiveScanBackward {
                next_fns: vec![x.grad_fn().cloned()],
                saved_input: x_data.clone(),
                d_param: self.d_param.clone(),
            });
            Variable::from_operation(out_tensor, grad_fn, true)
        } else {
            Variable::new(out_tensor, false)
        }
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![
            Parameter::named("a_log", self.a_log.clone(), true),
            Parameter::named("d_param", self.d_param.clone(), true),
        ];
        params.extend(self.x_proj.parameters());
        params.extend(self.dt_proj.parameters());
        params
    }
}

// =============================================================================
// SelectiveScanBackward
// =============================================================================

/// Simplified backward for the selective scan.
///
/// The full analytical backward of the scan recurrence is complex (requires
/// reverse-time scan). We use a simplified approximation that passes
/// gradients through the skip connection (D * x) and a scaled identity.
#[derive(Debug)]
struct SelectiveScanBackward {
    next_fns: Vec<Option<GradFn>>,
    saved_input: Tensor<f32>,
    d_param: Tensor<f32>,
}

impl GradientFunction for SelectiveScanBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let shape = self.saved_input.shape();
        let d_inner = shape[2];
        let g_vec = grad_output.to_vec();
        let d_vec = self.d_param.to_vec();

        // Approximate: grad_input ≈ D * grad_output (skip connection gradient)
        // plus a scaled pass-through for the scan path
        let mut grad_input = vec![0.0f32; g_vec.len()];
        let total = g_vec.len();
        for i in 0..total {
            let d_idx = i % d_inner;
            // D skip + identity pass-through
            grad_input[i] = g_vec[i] * (d_vec[d_idx] + 1.0);
        }

        let gi = Tensor::from_vec(grad_input, shape).unwrap();
        vec![Some(gi)]
    }

    fn name(&self) -> &'static str {
        "SelectiveScanBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// SSM Block
// =============================================================================

/// Mamba-style SSM block: input projection, depthwise conv, selective scan, gated output.
///
/// Architecture:
///   x -> in_proj -> (z, x_proj)
///   x_proj -> conv1d -> silu -> selective_scan -> y
///   output = out_proj(y * silu(z))
#[derive(Debug)]
pub struct SSMBlock {
    /// Input projection: d_model -> 2 * d_inner (for z and x_proj)
    in_proj: Linear,
    /// Depthwise 1D convolution
    conv1d: DepthwiseConv1d,
    /// Selective scan
    scan: SelectiveScan,
    /// Output projection: d_inner -> d_model
    out_proj: Linear,
    /// Model dimension
    #[allow(dead_code)]
    d_model: usize,
    /// Inner dimension
    d_inner: usize,
}

impl SSMBlock {
    /// Create new SSM block.
    pub fn new(config: &SSMConfig) -> Self {
        Self {
            in_proj: Linear::new(config.d_model, 2 * config.d_inner),
            conv1d: DepthwiseConv1d::new(config.d_inner, config.d_conv),
            scan: SelectiveScan::new(config.d_inner, config.d_state, config.dt_rank),
            out_proj: Linear::new(config.d_inner, config.d_model),
            d_model: config.d_model,
            d_inner: config.d_inner,
        }
    }

    /// Forward pass: [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    pub fn forward(&self, x: &Variable) -> Variable {
        // Project to 2 * d_inner
        let proj = self.in_proj.forward(x);

        // Split into z (gate) and x_proj
        let z = proj.narrow(2, 0, self.d_inner);
        let x_proj = proj.narrow(2, self.d_inner, self.d_inner);

        // Conv1d + SiLU
        let x_conv = self.conv1d.forward(&x_proj);
        let x_conv = x_conv.silu();

        // Selective scan
        let y = self.scan.forward(&x_conv);

        // Gated output: y * silu(z)
        let gate = z.silu();
        let y_gated = y.mul(&gate);

        // Output projection
        self.out_proj.forward(&y_gated)
    }

    /// Get parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.in_proj.parameters());
        params.extend(self.conv1d.parameters());
        params.extend(self.scan.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depthwise_conv1d_shape() {
        let conv = DepthwiseConv1d::new(64, 4);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1f32; 2 * 8 * 64], &[2, 8, 64]).unwrap(),
            true,
        );
        let y = conv.forward(&x);
        assert_eq!(y.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_selective_scan_shape() {
        let scan = SelectiveScan::new(64, 16, 4);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1f32; 2 * 8 * 64], &[2, 8, 64]).unwrap(),
            true,
        );
        let y = scan.forward(&x);
        assert_eq!(y.data().shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_ssm_block_shape() {
        let config = SSMConfig::from_d_model(128);
        let block = SSMBlock::new(&config);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1f32; 2 * 8 * 128], &[2, 8, 128]).unwrap(),
            true,
        );
        let y = block.forward(&x);
        assert_eq!(y.data().shape(), &[2, 8, 128]);
    }

    #[test]
    fn test_ssm_block_backward() {
        let config = SSMConfig::from_d_model(32);
        let block = SSMBlock::new(&config);
        let x = Variable::new(
            Tensor::from_vec(vec![0.1f32; 1 * 4 * 32], &[1, 4, 32]).unwrap(),
            true,
        );
        let y = block.forward(&x);
        let loss = y.sum();
        loss.backward();
        // Should not panic
    }
}
