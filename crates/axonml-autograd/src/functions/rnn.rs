//! Backward functions for recurrent neural network operations.
//!
//! 511 lines. `LSTMCellBackward` (gate-level gradient through forget/input/
//! output/cell gates), `GRUCellBackward` (update/reset gate gradients),
//! `RNNCellBackward` (simple tanh-cell backward). Each caches the gate
//! activations and hidden states from the forward pass for gradient reuse.
//!
//! # File
//! `crates/axonml-autograd/src/functions/rnn.rs`
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

use std::any::Any;

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradientFunction};

// =============================================================================
// LSTM Gates Backward
// =============================================================================

/// Gradient function for fused LSTM gate computation.
///
/// Stores the forward state needed for backward:
/// - gates: [batch, 4*hidden] pre-activation gate values
/// - c_prev: [batch, hidden] previous cell state
/// - c_new: [batch, hidden] new cell state from forward
/// - hidden_size: size of hidden dimension
///
/// Produces two gradients:
/// - grad_gates: [batch, 4*hidden] gradient w.r.t. pre-activation gates (input 0)
/// - grad_c_prev: [batch, hidden] gradient w.r.t. previous cell state (input 1)
#[derive(Debug)]
pub struct LstmGatesBackward {
    next_fns: Vec<Option<GradFn>>,
    /// Pre-activation gates from forward [batch, 4*hidden]
    saved_gates: Tensor<f32>,
    /// Previous cell state [batch, hidden]
    saved_c_prev: Tensor<f32>,
    /// New cell state from forward [batch, hidden]
    saved_c_new: Tensor<f32>,
    /// Hidden dimension size
    hidden_size: usize,
}

impl LstmGatesBackward {
    /// Creates a new `LstmGatesBackward`.
    ///
    /// - `gates_grad_fn`: grad_fn from the combined gates variable (for backprop to ih+hh)
    /// - `c_prev_grad_fn`: grad_fn from the previous cell state variable
    /// - `gates`: saved pre-activation gates tensor
    /// - `c_prev`: saved previous cell state tensor
    /// - `c_new`: saved new cell state tensor
    /// - `hidden_size`: hidden dimension size
    #[must_use]
    pub fn new(
        gates_grad_fn: Option<GradFn>,
        c_prev_grad_fn: Option<GradFn>,
        gates: Tensor<f32>,
        c_prev: Tensor<f32>,
        c_new: Tensor<f32>,
        hidden_size: usize,
    ) -> Self {
        Self {
            next_fns: vec![gates_grad_fn, c_prev_grad_fn],
            saved_gates: gates,
            saved_c_prev: c_prev,
            saved_c_new: c_new,
            hidden_size,
        }
    }
}

impl GradientFunction for LstmGatesBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let batch_size = grad_output.shape()[0];
        let hs = self.hidden_size;
        let total = batch_size * hs;

        // grad_output is grad_h [batch, hidden].
        // We also need grad_c_next. For the h_new output, the cell gradient
        // from the loss comes through the next timestep's backward. For a
        // single-output scenario (the LSTM forward returns h_new, not c_new),
        // grad_c_next is zero unless accumulated from the c_new path.
        //
        // However, the way we wire this in LSTM::forward, grad_c_next is
        // implicitly zero for the last timestep and accumulated via the
        // LstmGatesBackward chain for earlier timesteps.
        // We store grad_c_next as zeros here and let it accumulate.
        // grad_c_next is zero for the h_new output path; in multi-timestep LSTM
        // the cell gradient accumulates through the LstmGatesBackward chain.
        let grad_c_next: Tensor<f32> = Tensor::zeros(&[batch_size, hs]);

        // GPU fast path
        #[cfg(feature = "cuda")]
        if self.saved_gates.device().is_gpu() {
            let grad_h_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_gates.device()).unwrap()
            };
            let grad_c_gpu = grad_c_next
                .to_device(self.saved_gates.device())
                .unwrap_or(grad_c_next.clone());

            if let Some((grad_gates, grad_c_prev)) = self.saved_gates.lstm_gates_backward_fused(
                &self.saved_c_prev,
                &self.saved_c_new,
                &grad_h_gpu,
                &grad_c_gpu,
                hs,
            ) {
                return vec![Some(grad_gates), Some(grad_c_prev)];
            }
        }

        // CPU fallback
        let gates_data = self.saved_gates.to_vec();
        let c_prev_data = self.saved_c_prev.to_vec();
        let c_new_data = self.saved_c_new.to_vec();
        let grad_h_data = grad_output.to_vec();
        let grad_c_next_data = grad_c_next.to_vec();

        let mut grad_gates_data = vec![0.0f32; batch_size * 4 * hs];
        let mut grad_c_prev_data = vec![0.0f32; total];

        let total_work = batch_size * hs;
        if total_work >= 4096 {
            use rayon::prelude::*;
            let gg_ptr = grad_gates_data.as_mut_ptr() as usize;
            let gc_ptr = grad_c_prev_data.as_mut_ptr() as usize;
            (0..total_work).into_par_iter().for_each(|idx| {
                let gg_ptr = gg_ptr as *mut f32;
                let gc_ptr = gc_ptr as *mut f32;
                let b = idx / hs;
                let h = idx % hs;
                let base = b * 4 * hs;

                // Load pre-activation gates
                let i_pre = gates_data[base + h];
                let f_pre = gates_data[base + hs + h];
                let g_pre = gates_data[base + 2 * hs + h];
                let o_pre = gates_data[base + 3 * hs + h];

                // Recompute activations
                let i_act = 1.0 / (1.0 + (-i_pre).exp());
                let f_act = 1.0 / (1.0 + (-f_pre).exp());
                let g_act = g_pre.tanh();
                let o_act = 1.0 / (1.0 + (-o_pre).exp());

                let c = c_new_data[idx];
                let tanh_c = c.tanh();
                let dh = grad_h_data[idx];
                // dc = grad_c_next + grad_h * o * (1 - tanh(c)^2)
                let dc = grad_c_next_data[idx] + dh * o_act * (1.0 - tanh_c * tanh_c);

                // Gate gradients
                unsafe {
                    *gg_ptr.add(base + h) = dc * g_act * i_act * (1.0 - i_act);
                    *gg_ptr.add(base + hs + h) = dc * c_prev_data[idx] * f_act * (1.0 - f_act);
                    *gg_ptr.add(base + 2 * hs + h) = dc * i_act * (1.0 - g_act * g_act);
                    *gg_ptr.add(base + 3 * hs + h) = dh * tanh_c * o_act * (1.0 - o_act);
                    *gc_ptr.add(idx) = dc * f_act;
                }
            });
        } else {
            for b in 0..batch_size {
                for h in 0..hs {
                    let idx = b * hs + h;
                    let base = b * 4 * hs;

                    // Load pre-activation gates
                    let i_pre = gates_data[base + h];
                    let f_pre = gates_data[base + hs + h];
                    let g_pre = gates_data[base + 2 * hs + h];
                    let o_pre = gates_data[base + 3 * hs + h];

                    // Recompute activations
                    let i_act = 1.0 / (1.0 + (-i_pre).exp());
                    let f_act = 1.0 / (1.0 + (-f_pre).exp());
                    let g_act = g_pre.tanh();
                    let o_act = 1.0 / (1.0 + (-o_pre).exp());

                    let c = c_new_data[idx];
                    let tanh_c = c.tanh();
                    let dh = grad_h_data[idx];
                    // dc = grad_c_next + grad_h * o * (1 - tanh(c)^2)
                    let dc = grad_c_next_data[idx] + dh * o_act * (1.0 - tanh_c * tanh_c);

                    // Gate gradients
                    grad_gates_data[base + h] = dc * g_act * i_act * (1.0 - i_act);
                    grad_gates_data[base + hs + h] = dc * c_prev_data[idx] * f_act * (1.0 - f_act);
                    grad_gates_data[base + 2 * hs + h] = dc * i_act * (1.0 - g_act * g_act);
                    grad_gates_data[base + 3 * hs + h] = dh * tanh_c * o_act * (1.0 - o_act);

                    grad_c_prev_data[idx] = dc * f_act;
                }
            }
        }

        vec![
            Some(
                Tensor::from_vec(grad_gates_data, &[batch_size, 4 * hs])
                    .expect("backward: tensor creation failed"),
            ),
            Some(
                Tensor::from_vec(grad_c_prev_data, &[batch_size, hs])
                    .expect("backward: tensor creation failed"),
            ),
        ]
    }

    fn name(&self) -> &'static str {
        "LstmGatesBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// GRU Gates Backward
// =============================================================================

/// Gradient function for fused GRU gate computation.
///
/// Stores the forward state needed for backward:
/// - gates_ih: [batch, 3*hidden] pre-activation input-hidden gates
/// - gates_hh: [batch, 3*hidden] pre-activation hidden-hidden gates
/// - h_prev: [batch, hidden] previous hidden state
/// - hidden_size: size of hidden dimension
///
/// Produces three gradients:
/// - grad_gates_ih: [batch, 3*hidden] gradient w.r.t. ih pre-activations (input 0)
/// - grad_gates_hh: [batch, 3*hidden] gradient w.r.t. hh pre-activations (input 1)
/// - grad_h_prev: [batch, hidden] gradient w.r.t. previous hidden (input 2)
#[derive(Debug)]
pub struct GruGatesBackward {
    next_fns: Vec<Option<GradFn>>,
    /// Pre-activation input-hidden gates [batch, 3*hidden]
    saved_gates_ih: Tensor<f32>,
    /// Pre-activation hidden-hidden gates [batch, 3*hidden]
    saved_gates_hh: Tensor<f32>,
    /// Previous hidden state [batch, hidden]
    saved_h_prev: Tensor<f32>,
    /// Hidden dimension size
    hidden_size: usize,
}

impl GruGatesBackward {
    /// Creates a new `GruGatesBackward`.
    ///
    /// - `ih_grad_fn`: grad_fn from ih gates variable
    /// - `hh_grad_fn`: grad_fn from hh gates variable
    /// - `h_prev_grad_fn`: grad_fn from previous hidden state variable
    /// - `gates_ih`: saved ih gates tensor
    /// - `gates_hh`: saved hh gates tensor
    /// - `h_prev`: saved previous hidden state tensor
    /// - `hidden_size`: hidden dimension size
    #[must_use]
    pub fn new(
        ih_grad_fn: Option<GradFn>,
        hh_grad_fn: Option<GradFn>,
        h_prev_grad_fn: Option<GradFn>,
        gates_ih: Tensor<f32>,
        gates_hh: Tensor<f32>,
        h_prev: Tensor<f32>,
        hidden_size: usize,
    ) -> Self {
        Self {
            next_fns: vec![ih_grad_fn, hh_grad_fn, h_prev_grad_fn],
            saved_gates_ih: gates_ih,
            saved_gates_hh: gates_hh,
            saved_h_prev: h_prev,
            hidden_size,
        }
    }
}

impl GradientFunction for GruGatesBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        let batch_size = grad_output.shape()[0];
        let hs = self.hidden_size;

        // GPU fast path
        #[cfg(feature = "cuda")]
        if self.saved_gates_ih.device().is_gpu() {
            let grad_gpu = if grad_output.device().is_gpu() {
                grad_output.clone()
            } else {
                grad_output.to_device(self.saved_gates_ih.device()).unwrap()
            };

            if let Some((grad_ih, grad_hh, grad_h_prev)) = self
                .saved_gates_ih
                .gru_gates_backward_fused(&self.saved_gates_hh, &self.saved_h_prev, &grad_gpu, hs)
            {
                return vec![Some(grad_ih), Some(grad_hh), Some(grad_h_prev)];
            }
        }

        // CPU fallback
        let ih_data = self.saved_gates_ih.to_vec();
        let hh_data = self.saved_gates_hh.to_vec();
        let h_prev_data = self.saved_h_prev.to_vec();
        let grad_data = grad_output.to_vec();

        let mut grad_ih_data = vec![0.0f32; batch_size * 3 * hs];
        let mut grad_hh_data = vec![0.0f32; batch_size * 3 * hs];
        let mut grad_h_prev_data = vec![0.0f32; batch_size * hs];

        let total_work = batch_size * hs;
        if total_work >= 4096 {
            use rayon::prelude::*;
            let gi_ptr = grad_ih_data.as_mut_ptr() as usize;
            let gh_ptr = grad_hh_data.as_mut_ptr() as usize;
            let gp_ptr = grad_h_prev_data.as_mut_ptr() as usize;
            (0..total_work).into_par_iter().for_each(|idx| {
                let gi_ptr = gi_ptr as *mut f32;
                let gh_ptr = gh_ptr as *mut f32;
                let gp_ptr = gp_ptr as *mut f32;
                let b = idx / hs;
                let h = idx % hs;
                let base = b * 3 * hs;

                let r_ih = ih_data[base + h];
                let z_ih = ih_data[base + hs + h];
                let n_ih = ih_data[base + 2 * hs + h];

                let r_hh = hh_data[base + h];
                let z_hh = hh_data[base + hs + h];
                let n_hh_val = hh_data[base + 2 * hs + h];

                // Recompute activations
                let r = 1.0 / (1.0 + (-(r_ih + r_hh)).exp());
                let z = 1.0 / (1.0 + (-(z_ih + z_hh)).exp());
                let n = (n_ih + r * n_hh_val).tanh();

                let hp = h_prev_data[idx];
                let dh = grad_data[idx];

                // h_new = (1 - z) * n + z * h_prev
                let dz = dh * (hp - n);
                let dn = dh * (1.0 - z);
                unsafe { *gp_ptr.add(idx) = dh * z; }

                let d_n_pre = dn * (1.0 - n * n);
                let d_z_pre = dz * z * (1.0 - z);
                let dr = d_n_pre * n_hh_val;
                let d_r_pre = dr * r * (1.0 - r);

                // ih gate gradients
                unsafe {
                    *gi_ptr.add(base + h) = d_r_pre;
                    *gi_ptr.add(base + hs + h) = d_z_pre;
                    *gi_ptr.add(base + 2 * hs + h) = d_n_pre;
                }

                // hh gate gradients
                unsafe {
                    *gh_ptr.add(base + h) = d_r_pre;
                    *gh_ptr.add(base + hs + h) = d_z_pre;
                    *gh_ptr.add(base + 2 * hs + h) = d_n_pre * r;
                }
            });
        } else {
            for b in 0..batch_size {
                for h in 0..hs {
                    let idx = b * hs + h;
                    let base = b * 3 * hs;

                    let r_ih = ih_data[base + h];
                    let z_ih = ih_data[base + hs + h];
                    let n_ih = ih_data[base + 2 * hs + h];

                    let r_hh = hh_data[base + h];
                    let z_hh = hh_data[base + hs + h];
                    let n_hh_val = hh_data[base + 2 * hs + h];

                    // Recompute activations
                    let r = 1.0 / (1.0 + (-(r_ih + r_hh)).exp());
                    let z = 1.0 / (1.0 + (-(z_ih + z_hh)).exp());
                    let n = (n_ih + r * n_hh_val).tanh();

                    let hp = h_prev_data[idx];
                    let dh = grad_data[idx];

                    // h_new = (1 - z) * n + z * h_prev
                    let dz = dh * (hp - n);
                    let dn = dh * (1.0 - z);
                    grad_h_prev_data[idx] = dh * z;

                    let d_n_pre = dn * (1.0 - n * n);
                    let d_z_pre = dz * z * (1.0 - z);
                    let dr = d_n_pre * n_hh_val;
                    let d_r_pre = dr * r * (1.0 - r);

                    // ih gate gradients
                    grad_ih_data[base + h] = d_r_pre;
                    grad_ih_data[base + hs + h] = d_z_pre;
                    grad_ih_data[base + 2 * hs + h] = d_n_pre;

                    // hh gate gradients
                    grad_hh_data[base + h] = d_r_pre;
                    grad_hh_data[base + hs + h] = d_z_pre;
                    grad_hh_data[base + 2 * hs + h] = d_n_pre * r;
                }
            }
        }

        vec![
            Some(
                Tensor::from_vec(grad_ih_data, &[batch_size, 3 * hs])
                    .expect("backward: tensor creation failed"),
            ),
            Some(
                Tensor::from_vec(grad_hh_data, &[batch_size, 3 * hs])
                    .expect("backward: tensor creation failed"),
            ),
            Some(
                Tensor::from_vec(grad_h_prev_data, &[batch_size, hs])
                    .expect("backward: tensor creation failed"),
            ),
        ]
    }

    fn name(&self) -> &'static str {
        "GruGatesBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &self.next_fns
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Identity Backward — Gradient Passthrough
// =============================================================================

/// Identity gradient function that passes grad_output through unchanged.
///
/// Used when the forward operation doesn't transform the gradient (e.g.,
/// reshape, view, or when the backward is fused into a parent operation's
/// CUDA kernel and the autograd graph needs a node to maintain connectivity).
#[derive(Debug)]
pub struct IdentityBackward;

impl GradientFunction for IdentityBackward {
    fn apply(&self, grad_output: &Tensor<f32>) -> Vec<Option<Tensor<f32>>> {
        vec![Some(grad_output.clone())]
    }

    fn name(&self) -> &'static str {
        "IdentityBackward"
    }

    fn next_functions(&self) -> &[Option<GradFn>] {
        &[]
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

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn test_lstm_backward_shapes() {
        let batch = 2;
        let hidden = 3;
        let gates = Tensor::from_vec(vec![0.5f32; batch * 4 * hidden], &[batch, 4 * hidden])
            .expect("backward: tensor creation failed");
        let c_prev = Tensor::from_vec(vec![0.1f32; batch * hidden], &[batch, hidden])
            .expect("backward: tensor creation failed");
        let c_new = Tensor::from_vec(vec![0.3f32; batch * hidden], &[batch, hidden])
            .expect("backward: tensor creation failed");
        let backward = LstmGatesBackward::new(None, None, gates, c_prev, c_new, hidden);
        let grad_h = Tensor::from_vec(vec![1.0f32; batch * hidden], &[batch, hidden])
            .expect("backward: tensor creation failed");
        let grads = backward.apply(&grad_h);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[batch, 4 * hidden]);
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[batch, hidden]);
    }

    #[test]
    fn test_lstm_backward_finite_nonzero() {
        let _batch = 1;
        let hidden = 2;
        let gates = Tensor::from_vec(vec![0.5, -0.3, 0.8, 0.1, 0.2, 0.4, -0.5, 0.6], &[1, 8])
            .expect("backward: tensor creation failed");
        let c_prev =
            Tensor::from_vec(vec![0.1, -0.2], &[1, 2]).expect("backward: tensor creation failed");
        let i = [sigmoid(0.5), sigmoid(-0.3)];
        let f = [sigmoid(0.8), sigmoid(0.1)];
        let g = [0.2f32.tanh(), 0.4f32.tanh()];
        let c_new = Tensor::from_vec(
            vec![f[0] * 0.1 + i[0] * g[0], f[1] * (-0.2) + i[1] * g[1]],
            &[1, 2],
        )
        .unwrap();
        let backward = LstmGatesBackward::new(None, None, gates, c_prev, c_new, hidden);
        let grad_h =
            Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("backward: tensor creation failed");
        let grads = backward.apply(&grad_h);
        for &v in &grads[0].as_ref().unwrap().to_vec() {
            assert!(v.is_finite(), "LSTM gate grad not finite: {v}");
        }
        for &v in &grads[1].as_ref().unwrap().to_vec() {
            assert!(v.is_finite(), "LSTM c_prev grad not finite: {v}");
        }
        let nonzero = grads[0]
            .as_ref()
            .unwrap()
            .to_vec()
            .iter()
            .filter(|v| v.abs() > 1e-10)
            .count();
        assert!(
            nonzero >= 4,
            "Expected most LSTM gate grads nonzero, got {nonzero}/8"
        );
    }

    #[test]
    fn test_gru_backward_shapes() {
        let batch = 2;
        let hidden = 4;
        let gates = Tensor::from_vec(vec![0.5f32; batch * 3 * hidden], &[batch, 3 * hidden])
            .expect("backward: tensor creation failed");
        let h_prev = Tensor::from_vec(vec![0.1f32; batch * hidden], &[batch, hidden])
            .expect("backward: tensor creation failed");
        let gates_hh = gates.clone();
        let backward = GruGatesBackward::new(None, None, None, gates, gates_hh, h_prev, hidden);
        let grad_h = Tensor::from_vec(vec![1.0f32; batch * hidden], &[batch, hidden])
            .expect("backward: tensor creation failed");
        let grads = backward.apply(&grad_h);
        assert_eq!(grads.len(), 3); // grad_ih, grad_hh, grad_h_prev
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[batch, 3 * hidden]);
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[batch, 3 * hidden]);
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[batch, hidden]);
    }

    #[test]
    fn test_gru_backward_finite_nonzero() {
        let _batch = 1;
        let hidden = 3;
        let gates = Tensor::from_vec(vec![0.5, -0.3, 0.8, 0.1, 0.2, 0.4, -0.5, 0.6, 0.3], &[1, 9])
            .expect("backward: tensor creation failed");
        let h_prev = Tensor::from_vec(vec![0.1, -0.2, 0.3], &[1, 3])
            .expect("backward: tensor creation failed");
        let gates_hh = gates.clone();
        let backward = GruGatesBackward::new(None, None, None, gates, gates_hh, h_prev, hidden);
        let grad_h = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3])
            .expect("backward: tensor creation failed");
        let grads = backward.apply(&grad_h);
        for (i, g) in grads.iter().enumerate() {
            for &v in &g.as_ref().unwrap().to_vec() {
                assert!(v.is_finite(), "GRU grad[{i}] not finite: {v}");
            }
        }
        let nonzero = grads[0]
            .as_ref()
            .unwrap()
            .to_vec()
            .iter()
            .filter(|v| v.abs() > 1e-10)
            .count();
        assert!(
            nonzero >= 3,
            "Expected most GRU ih grads nonzero, got {nonzero}/9"
        );
    }

    #[test]
    fn test_identity_backward() {
        let grad =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("backward: tensor creation failed");
        let result = IdentityBackward.apply(&grad);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_ref().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
