//! Recurrent Neural Network Layers - RNN, LSTM, GRU
//!
//! Processes sequential data with recurrent connections.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::init::{xavier_uniform, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// RNNCell
// =============================================================================

/// A single RNN cell.
///
/// h' = tanh(W_ih * x + b_ih + W_hh * h + b_hh)
pub struct RNNCell {
    /// Input-hidden weights.
    pub weight_ih: Parameter,
    /// Hidden-hidden weights.
    pub weight_hh: Parameter,
    /// Input-hidden bias.
    pub bias_ih: Parameter,
    /// Hidden-hidden bias.
    pub bias_hh: Parameter,
    /// Input size.
    input_size: usize,
    /// Hidden size.
    hidden_size: usize,
}

impl RNNCell {
    /// Creates a new RNNCell.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            weight_ih: Parameter::named("weight_ih", xavier_uniform(input_size, hidden_size), true),
            weight_hh: Parameter::named(
                "weight_hh",
                xavier_uniform(hidden_size, hidden_size),
                true,
            ),
            bias_ih: Parameter::named("bias_ih", zeros(&[hidden_size]), true),
            bias_hh: Parameter::named("bias_hh", zeros(&[hidden_size]), true),
            input_size,
            hidden_size,
        }
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden state size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Forward pass for a single time step.
    pub fn forward_step(&self, input: &Variable, hidden: &Variable) -> Variable {
        let input_features = input.data().shape().last().copied().unwrap_or(0);
        assert_eq!(
            input_features, self.input_size,
            "RNNCell: expected input size {}, got {}",
            self.input_size, input_features
        );
        // x @ W_ih^T + b_ih
        let weight_ih = self.weight_ih.variable();
        let weight_ih_t = Variable::new(weight_ih.data().t().unwrap(), weight_ih.requires_grad());
        let ih = input.matmul(&weight_ih_t);
        let bias_ih = self.bias_ih.variable();
        let ih = ih.add_var(&bias_ih);

        // h @ W_hh^T + b_hh
        let weight_hh = self.weight_hh.variable();
        let weight_hh_t = Variable::new(weight_hh.data().t().unwrap(), weight_hh.requires_grad());
        let hh = hidden.matmul(&weight_hh_t);
        let bias_hh = self.bias_hh.variable();
        let hh = hh.add_var(&bias_hh);

        // tanh(ih + hh)
        ih.add_var(&hh).tanh()
    }
}

impl Module for RNNCell {
    fn forward(&self, input: &Variable) -> Variable {
        // Initialize hidden state to zeros
        let batch_size = input.shape()[0];
        let hidden = Variable::new(
            zeros(&[batch_size, self.hidden_size]),
            input.requires_grad(),
        );
        self.forward_step(input, &hidden)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.bias_ih.clone(),
            self.bias_hh.clone(),
        ]
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight_ih".to_string(), self.weight_ih.clone());
        params.insert("weight_hh".to_string(), self.weight_hh.clone());
        params.insert("bias_ih".to_string(), self.bias_ih.clone());
        params.insert("bias_hh".to_string(), self.bias_hh.clone());
        params
    }

    fn name(&self) -> &'static str {
        "RNNCell"
    }
}

// =============================================================================
// RNN
// =============================================================================

/// Multi-layer RNN.
///
/// Processes sequences through stacked RNN layers.
pub struct RNN {
    /// RNN cells for each layer.
    cells: Vec<RNNCell>,
    /// Input size.
    input_size: usize,
    /// Hidden size.
    hidden_size: usize,
    /// Number of layers.
    num_layers: usize,
    /// Batch first flag.
    batch_first: bool,
}

impl RNN {
    /// Creates a new multi-layer RNN.
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::with_options(input_size, hidden_size, num_layers, true)
    }

    /// Creates an RNN with all options.
    pub fn with_options(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_first: bool,
    ) -> Self {
        let mut cells = Vec::with_capacity(num_layers);

        // First layer takes input_size
        cells.push(RNNCell::new(input_size, hidden_size));

        // Subsequent layers take hidden_size
        for _ in 1..num_layers {
            cells.push(RNNCell::new(hidden_size, hidden_size));
        }

        Self {
            cells,
            input_size,
            hidden_size,
            num_layers,
            batch_first,
        }
    }
}

impl Module for RNN {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let (batch_size, seq_len, _) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        // Initialize hidden states
        let mut hiddens: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                Variable::new(
                    zeros(&[batch_size, self.hidden_size]),
                    input.requires_grad(),
                )
            })
            .collect();

        // Process each time step
        let input_data = input.data();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract input at time t
            let t_input = if self.batch_first {
                // [batch, seq, features] -> extract [batch, features] at t
                let mut slice_data = vec![0.0f32; batch_size * self.input_size];
                let input_vec = input_data.to_vec();
                for b in 0..batch_size {
                    for f in 0..self.input_size {
                        let src_idx = b * seq_len * self.input_size + t * self.input_size + f;
                        let dst_idx = b * self.input_size + f;
                        slice_data[dst_idx] = input_vec[src_idx];
                    }
                }
                Variable::new(
                    Tensor::from_vec(slice_data, &[batch_size, self.input_size]).unwrap(),
                    input.requires_grad(),
                )
            } else {
                // [seq, batch, features] -> extract [batch, features] at t
                let mut slice_data = vec![0.0f32; batch_size * self.input_size];
                let input_vec = input_data.to_vec();
                for b in 0..batch_size {
                    for f in 0..self.input_size {
                        let src_idx = t * batch_size * self.input_size + b * self.input_size + f;
                        let dst_idx = b * self.input_size + f;
                        slice_data[dst_idx] = input_vec[src_idx];
                    }
                }
                Variable::new(
                    Tensor::from_vec(slice_data, &[batch_size, self.input_size]).unwrap(),
                    input.requires_grad(),
                )
            };

            // Process through layers
            let mut layer_input = t_input;
            for (l, cell) in self.cells.iter().enumerate() {
                hiddens[l] = cell.forward_step(&layer_input, &hiddens[l]);
                layer_input = hiddens[l].clone();
            }

            outputs.push(hiddens[self.num_layers - 1].clone());
        }

        // Stack outputs
        let output_size = batch_size * seq_len * self.hidden_size;
        let mut output_data = vec![0.0f32; output_size];

        for (t, out) in outputs.iter().enumerate() {
            let out_vec = out.data().to_vec();
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    let src_idx = b * self.hidden_size + h;
                    let dst_idx = if self.batch_first {
                        b * seq_len * self.hidden_size + t * self.hidden_size + h
                    } else {
                        t * batch_size * self.hidden_size + b * self.hidden_size + h
                    };
                    output_data[dst_idx] = out_vec[src_idx];
                }
            }
        }

        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size]
        } else {
            vec![seq_len, batch_size, self.hidden_size]
        };

        Variable::new(
            Tensor::from_vec(output_data, &output_shape).unwrap(),
            input.requires_grad(),
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }

    fn name(&self) -> &'static str {
        "RNN"
    }
}

// =============================================================================
// LSTMCell
// =============================================================================

/// A single LSTM cell.
pub struct LSTMCell {
    /// Input-hidden weights for all gates.
    pub weight_ih: Parameter,
    /// Hidden-hidden weights for all gates.
    pub weight_hh: Parameter,
    /// Input-hidden bias for all gates.
    pub bias_ih: Parameter,
    /// Hidden-hidden bias for all gates.
    pub bias_hh: Parameter,
    /// Input size.
    input_size: usize,
    /// Hidden size.
    hidden_size: usize,
}

impl LSTMCell {
    /// Creates a new LSTMCell.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // LSTM has 4 gates, so weight size is 4*hidden_size
        Self {
            weight_ih: Parameter::named(
                "weight_ih",
                xavier_uniform(input_size, 4 * hidden_size),
                true,
            ),
            weight_hh: Parameter::named(
                "weight_hh",
                xavier_uniform(hidden_size, 4 * hidden_size),
                true,
            ),
            bias_ih: Parameter::named("bias_ih", zeros(&[4 * hidden_size]), true),
            bias_hh: Parameter::named("bias_hh", zeros(&[4 * hidden_size]), true),
            input_size,
            hidden_size,
        }
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden state size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Forward pass returning (h', c').
    pub fn forward_step(
        &self,
        input: &Variable,
        hx: &(Variable, Variable),
    ) -> (Variable, Variable) {
        let input_features = input.data().shape().last().copied().unwrap_or(0);
        assert_eq!(
            input_features, self.input_size,
            "LSTMCell: expected input size {}, got {}",
            self.input_size, input_features
        );

        let (h, c) = hx;

        // Compute all gates at once (x @ W^T + b)
        let weight_ih = self.weight_ih.variable();
        let weight_ih_t = Variable::new(weight_ih.data().t().unwrap(), weight_ih.requires_grad());
        let ih = input.matmul(&weight_ih_t);
        let bias_ih = self.bias_ih.variable();
        let ih = ih.add_var(&bias_ih);

        let weight_hh = self.weight_hh.variable();
        let weight_hh_t = Variable::new(weight_hh.data().t().unwrap(), weight_hh.requires_grad());
        let hh = h.matmul(&weight_hh_t);
        let bias_hh = self.bias_hh.variable();
        let hh = hh.add_var(&bias_hh);

        let gates = ih.add_var(&hh);
        let gates_vec = gates.data().to_vec();
        let batch_size = input.shape()[0];

        // Split into 4 gates: i, f, g, o
        let mut i_data = vec![0.0f32; batch_size * self.hidden_size];
        let mut f_data = vec![0.0f32; batch_size * self.hidden_size];
        let mut g_data = vec![0.0f32; batch_size * self.hidden_size];
        let mut o_data = vec![0.0f32; batch_size * self.hidden_size];

        for b in 0..batch_size {
            for j in 0..self.hidden_size {
                let base = b * 4 * self.hidden_size;
                i_data[b * self.hidden_size + j] = gates_vec[base + j];
                f_data[b * self.hidden_size + j] = gates_vec[base + self.hidden_size + j];
                g_data[b * self.hidden_size + j] = gates_vec[base + 2 * self.hidden_size + j];
                o_data[b * self.hidden_size + j] = gates_vec[base + 3 * self.hidden_size + j];
            }
        }

        let i = Variable::new(
            Tensor::from_vec(i_data, &[batch_size, self.hidden_size]).unwrap(),
            input.requires_grad(),
        )
        .sigmoid();
        let f = Variable::new(
            Tensor::from_vec(f_data, &[batch_size, self.hidden_size]).unwrap(),
            input.requires_grad(),
        )
        .sigmoid();
        let g = Variable::new(
            Tensor::from_vec(g_data, &[batch_size, self.hidden_size]).unwrap(),
            input.requires_grad(),
        )
        .tanh();
        let o = Variable::new(
            Tensor::from_vec(o_data, &[batch_size, self.hidden_size]).unwrap(),
            input.requires_grad(),
        )
        .sigmoid();

        // c' = f * c + i * g
        let c_new = f.mul_var(c).add_var(&i.mul_var(&g));

        // h' = o * tanh(c')
        let h_new = o.mul_var(&c_new.tanh());

        (h_new, c_new)
    }
}

impl Module for LSTMCell {
    fn forward(&self, input: &Variable) -> Variable {
        let batch_size = input.shape()[0];
        let h = Variable::new(
            zeros(&[batch_size, self.hidden_size]),
            input.requires_grad(),
        );
        let c = Variable::new(
            zeros(&[batch_size, self.hidden_size]),
            input.requires_grad(),
        );
        let (h_new, _) = self.forward_step(input, &(h, c));
        h_new
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.bias_ih.clone(),
            self.bias_hh.clone(),
        ]
    }

    fn name(&self) -> &'static str {
        "LSTMCell"
    }
}

// =============================================================================
// LSTM
// =============================================================================

/// Multi-layer LSTM.
pub struct LSTM {
    /// LSTM cells for each layer.
    cells: Vec<LSTMCell>,
    /// Input size.
    input_size: usize,
    /// Hidden size.
    hidden_size: usize,
    /// Number of layers.
    num_layers: usize,
    /// Batch first flag.
    batch_first: bool,
}

impl LSTM {
    /// Creates a new multi-layer LSTM.
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::with_options(input_size, hidden_size, num_layers, true)
    }

    /// Creates an LSTM with all options.
    pub fn with_options(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_first: bool,
    ) -> Self {
        let mut cells = Vec::with_capacity(num_layers);
        cells.push(LSTMCell::new(input_size, hidden_size));
        for _ in 1..num_layers {
            cells.push(LSTMCell::new(hidden_size, hidden_size));
        }

        Self {
            cells,
            input_size,
            hidden_size,
            num_layers,
            batch_first,
        }
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden state size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Variable) -> Variable {
        // Similar to RNN forward but using LSTM cells
        // For brevity, implementing a simplified version
        let shape = input.shape();
        let (batch_size, seq_len, input_features) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        let mut states: Vec<(Variable, Variable)> = (0..self.num_layers)
            .map(|_| {
                (
                    Variable::new(
                        zeros(&[batch_size, self.hidden_size]),
                        input.requires_grad(),
                    ),
                    Variable::new(
                        zeros(&[batch_size, self.hidden_size]),
                        input.requires_grad(),
                    ),
                )
            })
            .collect();

        let input_data = input.data();
        let input_vec = input_data.to_vec();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let mut slice_data = vec![0.0f32; batch_size * input_features];
            for b in 0..batch_size {
                for f in 0..input_features {
                    let src_idx = if self.batch_first {
                        b * seq_len * input_features + t * input_features + f
                    } else {
                        t * batch_size * input_features + b * input_features + f
                    };
                    slice_data[b * input_features + f] = input_vec[src_idx];
                }
            }

            // Input slice always has input_features dimensions
            let mut layer_input = Variable::new(
                Tensor::from_vec(slice_data.clone(), &[batch_size, input_features]).unwrap(),
                input.requires_grad(),
            );

            for (l, cell) in self.cells.iter().enumerate() {
                // Resize input if needed for subsequent layers
                if l > 0 {
                    layer_input = states[l - 1].0.clone();
                }
                states[l] = cell.forward_step(&layer_input, &states[l]);
            }

            outputs.push(states[self.num_layers - 1].0.clone());
        }

        // Stack outputs
        let mut output_data = vec![0.0f32; batch_size * seq_len * self.hidden_size];
        for (t, out) in outputs.iter().enumerate() {
            let out_vec = out.data().to_vec();
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    let dst_idx = if self.batch_first {
                        b * seq_len * self.hidden_size + t * self.hidden_size + h
                    } else {
                        t * batch_size * self.hidden_size + b * self.hidden_size + h
                    };
                    output_data[dst_idx] = out_vec[b * self.hidden_size + h];
                }
            }
        }

        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size]
        } else {
            vec![seq_len, batch_size, self.hidden_size]
        };

        Variable::new(
            Tensor::from_vec(output_data, &output_shape).unwrap(),
            input.requires_grad(),
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }

    fn name(&self) -> &'static str {
        "LSTM"
    }
}

// =============================================================================
// GRUCell and GRU
// =============================================================================

/// A single GRU cell.
///
/// h' = (1 - z) * n + z * h
/// where:
///   r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)  (reset gate)
///   z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)  (update gate)
///   n = tanh(W_in * x + b_in + r * (W_hn * h + b_hn))  (new gate)
pub struct GRUCell {
    /// Input-hidden weights for all gates (reset, update, new).
    pub weight_ih: Parameter,
    /// Hidden-hidden weights for all gates (reset, update, new).
    pub weight_hh: Parameter,
    /// Input-hidden bias for all gates.
    pub bias_ih: Parameter,
    /// Hidden-hidden bias for all gates.
    pub bias_hh: Parameter,
    /// Input size.
    input_size: usize,
    /// Hidden size.
    hidden_size: usize,
}

impl GRUCell {
    /// Creates a new GRU cell.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            weight_ih: Parameter::named(
                "weight_ih",
                xavier_uniform(input_size, 3 * hidden_size),
                true,
            ),
            weight_hh: Parameter::named(
                "weight_hh",
                xavier_uniform(hidden_size, 3 * hidden_size),
                true,
            ),
            bias_ih: Parameter::named("bias_ih", zeros(&[3 * hidden_size]), true),
            bias_hh: Parameter::named("bias_hh", zeros(&[3 * hidden_size]), true),
            input_size,
            hidden_size,
        }
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden state size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl GRUCell {
    /// Forward pass for a single time step with explicit hidden state.
    ///
    /// GRU equations:
    /// r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
    /// z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
    /// n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
    /// h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    ///
    /// All computations use Variable operations for proper gradient flow.
    pub fn forward_step(&self, input: &Variable, hidden: &Variable) -> Variable {
        let batch_size = input.shape()[0];
        let hidden_size = self.hidden_size;

        // Get weight matrices
        let weight_ih = self.weight_ih.variable();
        let weight_hh = self.weight_hh.variable();
        let bias_ih = self.bias_ih.variable();
        let bias_hh = self.bias_hh.variable();

        // Compute input transformation: x @ W_ih^T + b_ih
        // Shape: [batch, 3*hidden_size]
        let weight_ih_t = Variable::new(weight_ih.data().t().unwrap(), weight_ih.requires_grad());
        let ih = input.matmul(&weight_ih_t).add_var(&bias_ih);

        // Compute hidden transformation: h @ W_hh^T + b_hh
        // Shape: [batch, 3*hidden_size]
        let weight_hh_t = Variable::new(weight_hh.data().t().unwrap(), weight_hh.requires_grad());
        let hh = hidden.matmul(&weight_hh_t).add_var(&bias_hh);

        // Use narrow to split into gates (preserves gradient flow)
        // Each gate slice: [batch, hidden_size]
        let ih_r = ih.narrow(1, 0, hidden_size);
        let ih_z = ih.narrow(1, hidden_size, hidden_size);
        let ih_n = ih.narrow(1, 2 * hidden_size, hidden_size);

        let hh_r = hh.narrow(1, 0, hidden_size);
        let hh_z = hh.narrow(1, hidden_size, hidden_size);
        let hh_n = hh.narrow(1, 2 * hidden_size, hidden_size);

        // Compute gates using Variable operations for gradient flow
        // r = sigmoid(ih_r + hh_r)
        let r = ih_r.add_var(&hh_r).sigmoid();

        // z = sigmoid(ih_z + hh_z)
        let z = ih_z.add_var(&hh_z).sigmoid();

        // n = tanh(ih_n + r * hh_n)
        let n = ih_n.add_var(&r.mul_var(&hh_n)).tanh();

        // h_new = (1 - z) * n + z * h_prev
        // Create ones for (1 - z)
        let shape = [batch_size, hidden_size];
        let ones = Variable::new(
            Tensor::from_vec(vec![1.0f32; batch_size * hidden_size], &shape).unwrap(),
            false,
        );
        let one_minus_z = ones.sub_var(&z);

        // h_new = one_minus_z * n + z * h_prev
        one_minus_z.mul_var(&n).add_var(&z.mul_var(hidden))
    }
}

impl Module for GRUCell {
    fn forward(&self, input: &Variable) -> Variable {
        let batch_size = input.shape()[0];

        // Initialize hidden state to zeros
        let hidden = Variable::new(
            zeros(&[batch_size, self.hidden_size]),
            input.requires_grad(),
        );

        self.forward_step(input, &hidden)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.bias_ih.clone(),
            self.bias_hh.clone(),
        ]
    }

    fn name(&self) -> &'static str {
        "GRUCell"
    }
}

/// Multi-layer GRU.
pub struct GRU {
    /// GRU cells for each layer.
    cells: Vec<GRUCell>,
    /// Hidden state size.
    hidden_size: usize,
    /// Number of layers.
    num_layers: usize,
    /// If true, input is (batch, seq, features), else (seq, batch, features).
    batch_first: bool,
}

impl GRU {
    /// Creates a new multi-layer GRU.
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::with_capacity(num_layers);
        cells.push(GRUCell::new(input_size, hidden_size));
        for _ in 1..num_layers {
            cells.push(GRUCell::new(hidden_size, hidden_size));
        }
        Self {
            cells,
            hidden_size,
            num_layers,
            batch_first: true,
        }
    }

    /// Returns the hidden state size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl Module for GRU {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let (batch_size, seq_len, _input_size) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        // Initialize hidden states for all layers as Variables (with gradients)
        let mut hidden_states: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                Variable::new(
                    zeros(&[batch_size, self.hidden_size]),
                    input.requires_grad(),
                )
            })
            .collect();

        // Collect output Variables for each time step
        let mut output_vars: Vec<Variable> = Vec::with_capacity(seq_len);

        // Process each time step
        for t in 0..seq_len {
            // Extract input for this time step using narrow (preserves gradients)
            // input shape: [batch, seq, features]
            // narrow to [batch, 1, features], then reshape to [batch, features]
            // narrow gives [batch, 1, features], reshape to [batch, features]
            let narrowed = input.narrow(1, t, 1);
            let step_input = narrowed.reshape(&[batch_size, narrowed.data().numel() / batch_size]);

            // Process through each layer
            let mut layer_input = step_input;

            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let new_hidden = cell.forward_step(&layer_input, &hidden_states[layer_idx]);

                // Update hidden state for this layer (keeps gradient chain)
                hidden_states[layer_idx] = new_hidden.clone();

                // Output of this layer becomes input to next layer
                layer_input = new_hidden;
            }

            // Store output from last layer for this time step
            output_vars.push(layer_input);
        }

        // Stack outputs along the time dimension
        // Each output_var has shape [batch, hidden_size]
        // We need to combine them into [batch, seq, hidden_size]
        self.stack_outputs(&output_vars, batch_size, seq_len)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }

    fn name(&self) -> &'static str {
        "GRU"
    }
}

impl GRU {
    /// Forward pass that returns the mean of all hidden states.
    /// This is equivalent to processing then mean pooling, but with proper gradient flow.
    pub fn forward_mean(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let (batch_size, seq_len, _input_size) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        // Initialize hidden states for all layers as Variables (with gradients)
        let mut hidden_states: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                Variable::new(
                    zeros(&[batch_size, self.hidden_size]),
                    input.requires_grad(),
                )
            })
            .collect();

        // Accumulator for mean of outputs
        let mut output_sum: Option<Variable> = None;

        // Process each time step
        for t in 0..seq_len {
            // Extract input for this time step using narrow (preserves gradients)
            // narrow gives [batch, 1, features], reshape to [batch, features]
            let narrowed = input.narrow(1, t, 1);
            let step_input = narrowed.reshape(&[batch_size, narrowed.data().numel() / batch_size]);

            // Process through each layer
            let mut layer_input = step_input;

            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let new_hidden = cell.forward_step(&layer_input, &hidden_states[layer_idx]);
                hidden_states[layer_idx] = new_hidden.clone();
                layer_input = new_hidden;
            }

            // Accumulate output (last layer's hidden state)
            output_sum = Some(match output_sum {
                None => layer_input,
                Some(acc) => acc.add_var(&layer_input),
            });
        }

        // Return mean of all hidden states
        match output_sum {
            Some(sum) => sum.mul_scalar(1.0 / seq_len as f32),
            None => Variable::new(zeros(&[batch_size, self.hidden_size]), false),
        }
    }

    /// Forward pass that returns the last hidden state.
    /// Good for sequence classification with proper gradient flow.
    pub fn forward_last(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let (batch_size, seq_len, _input_size) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        // Initialize hidden states for all layers
        let mut hidden_states: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                Variable::new(
                    zeros(&[batch_size, self.hidden_size]),
                    input.requires_grad(),
                )
            })
            .collect();

        // Process each time step
        for t in 0..seq_len {
            // narrow gives [batch, 1, features], reshape to [batch, features]
            let narrowed = input.narrow(1, t, 1);
            let step_input = narrowed.reshape(&[batch_size, narrowed.data().numel() / batch_size]);

            let mut layer_input = step_input;

            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let new_hidden = cell.forward_step(&layer_input, &hidden_states[layer_idx]);
                hidden_states[layer_idx] = new_hidden.clone();
                layer_input = new_hidden;
            }
        }

        // Return last hidden state from last layer
        hidden_states
            .pop()
            .unwrap_or_else(|| Variable::new(zeros(&[batch_size, self.hidden_size]), false))
    }

    /// Stack output Variables into a single [batch, seq, hidden] tensor.
    /// Note: This creates a new tensor without gradient connections to individual timesteps.
    /// For gradient flow, use forward_mean() or forward_last() instead.
    fn stack_outputs(&self, outputs: &[Variable], batch_size: usize, seq_len: usize) -> Variable {
        if outputs.is_empty() {
            return Variable::new(zeros(&[batch_size, 0, self.hidden_size]), false);
        }

        let output_shape = [batch_size, seq_len, self.hidden_size];
        let requires_grad = outputs.iter().any(|o| o.requires_grad());

        let mut stacked_data = vec![0.0f32; batch_size * seq_len * self.hidden_size];
        for (t, out) in outputs.iter().enumerate() {
            let out_data = out.data().to_vec();
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    let idx = b * seq_len * self.hidden_size + t * self.hidden_size + h;
                    stacked_data[idx] = out_data[b * self.hidden_size + h];
                }
            }
        }

        Variable::new(
            Tensor::from_vec(stacked_data, &output_shape).unwrap(),
            requires_grad,
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_cell() {
        let cell = RNNCell::new(10, 20);
        let input = Variable::new(Tensor::from_vec(vec![1.0; 20], &[2, 10]).unwrap(), false);
        let hidden = Variable::new(Tensor::from_vec(vec![0.0; 40], &[2, 20]).unwrap(), false);
        let output = cell.forward_step(&input, &hidden);
        assert_eq!(output.shape(), vec![2, 20]);
    }

    #[test]
    fn test_rnn() {
        let rnn = RNN::new(10, 20, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 100], &[2, 5, 10]).unwrap(),
            false,
        );
        let output = rnn.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 20]);
    }

    #[test]
    fn test_lstm() {
        let lstm = LSTM::new(10, 20, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 100], &[2, 5, 10]).unwrap(),
            false,
        );
        let output = lstm.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 20]);
    }
}
