//! Recurrent Neural Network Layers - RNN, LSTM, GRU
//!
//! Processes sequential data with recurrent connections.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;

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
        let weight_ih_t = weight_ih.transpose(0, 1);
        let ih = input.matmul(&weight_ih_t);
        let bias_ih = self.bias_ih.variable();
        let ih = ih.add_var(&bias_ih);

        // h @ W_hh^T + b_hh
        let weight_hh = self.weight_hh.variable();
        let weight_hh_t = weight_hh.transpose(0, 1);
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
    _input_size: usize,
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
            _input_size: input_size,
            hidden_size,
            num_layers,
            batch_first,
        }
    }
}

impl Module for RNN {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let (batch_size, seq_len, input_features) = if self.batch_first {
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

        // Pre-compute input-to-hidden projection for layer 0 across ALL timesteps
        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, self.hidden_size]);

        // Hoist weight transposes out of the per-timestep loop
        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Layer 0: use pre-computed ih projection + hoisted weight transpose
            let ih_t = ih_all_3d.select(1, t);
            let hh = hiddens[0].matmul(&w_hh_t_0).add_var(&bias_hh_0);
            hiddens[0] = ih_t.add_var(&hh).tanh();

            // Subsequent layers
            for l in 1..self.num_layers {
                let layer_input = hiddens[l - 1].clone();
                hiddens[l] = self.cells[l].forward_step(&layer_input, &hiddens[l]);
            }

            outputs.push(hiddens[self.num_layers - 1].clone());
        }

        // Stack outputs using graph-tracked cat (unsqueeze + cat along time dim)
        let time_dim = if self.batch_first { 1 } else { 0 };
        let unsqueezed: Vec<Variable> = outputs.iter()
            .map(|o| o.unsqueeze(time_dim))
            .collect();
        let refs: Vec<&Variable> = unsqueezed.iter().collect();
        Variable::cat(&refs, time_dim)
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
        let weight_ih_t = weight_ih.transpose(0, 1);
        let ih = input.matmul(&weight_ih_t);
        let bias_ih = self.bias_ih.variable();
        let ih = ih.add_var(&bias_ih);

        let weight_hh = self.weight_hh.variable();
        let weight_hh_t = weight_hh.transpose(0, 1);
        let hh = h.matmul(&weight_hh_t);
        let bias_hh = self.bias_hh.variable();
        let hh = hh.add_var(&bias_hh);

        let gates = ih.add_var(&hh);
        let hs = self.hidden_size;

        // Split into 4 gates using narrow (preserves gradient flow)
        let i = gates.narrow(1, 0, hs).sigmoid();
        let f = gates.narrow(1, hs, hs).sigmoid();
        let g = gates.narrow(1, 2 * hs, hs).tanh();
        let o = gates.narrow(1, 3 * hs, hs).sigmoid();

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

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight_ih".to_string(), self.weight_ih.clone());
        params.insert("weight_hh".to_string(), self.weight_hh.clone());
        params.insert("bias_ih".to_string(), self.bias_ih.clone());
        params.insert("bias_hh".to_string(), self.bias_hh.clone());
        params
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
        let shape = input.shape();
        let (batch_size, seq_len, _input_features) = if self.batch_first {
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

        // Pre-compute input-to-hidden projection for layer 0 across ALL timesteps
        // input: [batch, seq, features] -> reshaped to [batch*seq, features]
        // ih_all: [batch*seq, 4*hidden] = input_2d @ W_ih^T + bias_ih
        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, _input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        // ih_all_3d: [batch, seq, 4*hidden]
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 4 * self.hidden_size]);

        // Hoist weight transpose + bias out of the per-timestep loop
        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Layer 0: use pre-computed ih projection + hoisted weight transpose
            let ih_t = ih_all_3d.select(1, t);
            let (h, c) = &states[0];

            let hh = h.matmul(&w_hh_t_0).add_var(&bias_hh_0);

            let gates = ih_t.add_var(&hh);
            let hs = self.hidden_size;
            let i_gate = gates.narrow(1, 0, hs).sigmoid();
            let f_gate = gates.narrow(1, hs, hs).sigmoid();
            let g_gate = gates.narrow(1, 2 * hs, hs).tanh();
            let o_gate = gates.narrow(1, 3 * hs, hs).sigmoid();
            let c_new = f_gate.mul_var(c).add_var(&i_gate.mul_var(&g_gate));
            let h_new = o_gate.mul_var(&c_new.tanh());
            states[0] = (h_new, c_new);

            // Subsequent layers use the regular cell forward_step
            for l in 1..self.num_layers {
                let layer_input = states[l - 1].0.clone();
                states[l] = self.cells[l].forward_step(&layer_input, &states[l]);
            }

            outputs.push(states[self.num_layers - 1].0.clone());
        }

        // Stack outputs along the time dimension
        let time_dim = if self.batch_first { 1 } else { 0 };
        let unsqueezed: Vec<Variable> = outputs.iter()
            .map(|o| o.unsqueeze(time_dim))
            .collect();
        let refs: Vec<&Variable> = unsqueezed.iter().collect();
        Variable::cat(&refs, time_dim)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        if self.cells.len() == 1 {
            // Single layer: expose directly without cell index prefix
            for (n, p) in self.cells[0].named_parameters() {
                params.insert(n, p);
            }
        } else {
            for (i, cell) in self.cells.iter().enumerate() {
                for (n, p) in cell.named_parameters() {
                    params.insert(format!("cells.{i}.{n}"), p);
                }
            }
        }
        params
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
        let _batch_size = input.shape()[0];
        let hidden_size = self.hidden_size;

        // Get weight matrices
        let weight_ih = self.weight_ih.variable();
        let weight_hh = self.weight_hh.variable();
        let bias_ih = self.bias_ih.variable();
        let bias_hh = self.bias_hh.variable();

        // Compute input transformation: x @ W_ih^T + b_ih
        // Shape: [batch, 3*hidden_size]
        let weight_ih_t = weight_ih.transpose(0, 1);
        let ih = input.matmul(&weight_ih_t).add_var(&bias_ih);

        // Compute hidden transformation: h @ W_hh^T + b_hh
        // Shape: [batch, 3*hidden_size]
        let weight_hh_t = weight_hh.transpose(0, 1);
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
        // Rewritten as: n + z * (h_prev - n)  to avoid allocating a ones tensor
        let h_minus_n = hidden.sub_var(&n);
        n.add_var(&z.mul_var(&h_minus_n))
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

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight_ih".to_string(), self.weight_ih.clone());
        params.insert("weight_hh".to_string(), self.weight_hh.clone());
        params.insert("bias_ih".to_string(), self.bias_ih.clone());
        params.insert("bias_hh".to_string(), self.bias_hh.clone());
        params
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
        let (batch_size, seq_len, input_features) = if self.batch_first {
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

        // Pre-compute input-to-hidden projection for layer 0 across ALL timesteps
        // One big matmul instead of seq_len small ones
        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 3 * self.hidden_size]);

        // Hoist weight transpose + bias out of the per-timestep loop
        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut output_vars: Vec<Variable> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Layer 0: use pre-computed ih projection + hoisted weight transpose
            let ih_t = ih_all_3d.select(1, t);
            let hidden = &hidden_states[0];
            let hs = self.hidden_size;

            let hh = hidden.matmul(&w_hh_t_0).add_var(&bias_hh_0);

            let ih_r = ih_t.narrow(1, 0, hs);
            let ih_z = ih_t.narrow(1, hs, hs);
            let ih_n = ih_t.narrow(1, 2 * hs, hs);
            let hh_r = hh.narrow(1, 0, hs);
            let hh_z = hh.narrow(1, hs, hs);
            let hh_n = hh.narrow(1, 2 * hs, hs);

            let r = ih_r.add_var(&hh_r).sigmoid();
            let z = ih_z.add_var(&hh_z).sigmoid();
            let n = ih_n.add_var(&r.mul_var(&hh_n)).tanh();
            let h_minus_n = hidden.sub_var(&n);
            let h_new = n.add_var(&z.mul_var(&h_minus_n));
            hidden_states[0] = h_new.clone();

            // Subsequent layers use the regular cell forward_step
            let mut layer_output = h_new;
            for l in 1..self.num_layers {
                let new_hidden = self.cells[l].forward_step(&layer_output, &hidden_states[l]);
                hidden_states[l] = new_hidden.clone();
                layer_output = new_hidden;
            }

            output_vars.push(layer_output);
        }

        // Stack outputs along the time dimension
        self.stack_outputs(&output_vars, batch_size, seq_len)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        if self.cells.len() == 1 {
            for (n, p) in self.cells[0].named_parameters() {
                params.insert(n, p);
            }
        } else {
            for (i, cell) in self.cells.iter().enumerate() {
                for (n, p) in cell.named_parameters() {
                    params.insert(format!("cells.{i}.{n}"), p);
                }
            }
        }
        params
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
        let (batch_size, seq_len, input_features) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        let mut hidden_states: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                Variable::new(
                    zeros(&[batch_size, self.hidden_size]),
                    input.requires_grad(),
                )
            })
            .collect();

        // Pre-compute input-to-hidden projection for layer 0 across ALL timesteps
        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 3 * self.hidden_size]);

        // Hoist weight transpose + bias out of per-timestep loop
        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut output_sum: Option<Variable> = None;
        let hs = self.hidden_size;

        for t in 0..seq_len {
            // Layer 0: use pre-computed ih projection + hoisted weight transpose
            let ih_t = ih_all_3d.select(1, t);
            let hidden = &hidden_states[0];
            let hh = hidden.matmul(&w_hh_t_0).add_var(&bias_hh_0);

            let ih_r = ih_t.narrow(1, 0, hs);
            let ih_z = ih_t.narrow(1, hs, hs);
            let ih_n = ih_t.narrow(1, 2 * hs, hs);
            let hh_r = hh.narrow(1, 0, hs);
            let hh_z = hh.narrow(1, hs, hs);
            let hh_n = hh.narrow(1, 2 * hs, hs);

            let r = ih_r.add_var(&hh_r).sigmoid();
            let z = ih_z.add_var(&hh_z).sigmoid();
            let n = ih_n.add_var(&r.mul_var(&hh_n)).tanh();
            let h_minus_n = hidden.sub_var(&n);
            let h_new = n.add_var(&z.mul_var(&h_minus_n));
            hidden_states[0] = h_new.clone();

            // Subsequent layers
            let mut layer_output = h_new;
            for l in 1..self.num_layers {
                let new_hidden = self.cells[l].forward_step(&layer_output, &hidden_states[l]);
                hidden_states[l] = new_hidden.clone();
                layer_output = new_hidden;
            }

            output_sum = Some(match output_sum {
                None => layer_output,
                Some(acc) => acc.add_var(&layer_output),
            });
        }

        match output_sum {
            Some(sum) => sum.mul_scalar(1.0 / seq_len as f32),
            None => Variable::new(zeros(&[batch_size, self.hidden_size]), false),
        }
    }

    /// Forward pass that returns the last hidden state.
    /// Good for sequence classification with proper gradient flow.
    pub fn forward_last(&self, input: &Variable) -> Variable {
        let shape = input.shape();
        let (batch_size, seq_len, input_features) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        let mut hidden_states: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                Variable::new(
                    zeros(&[batch_size, self.hidden_size]),
                    input.requires_grad(),
                )
            })
            .collect();

        // Pre-compute input-to-hidden projection for layer 0 across ALL timesteps
        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 3 * self.hidden_size]);

        // Hoist weight transpose + bias out of per-timestep loop
        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();
        let hs = self.hidden_size;

        for t in 0..seq_len {
            // Layer 0: use pre-computed ih projection + hoisted weight transpose
            let ih_t = ih_all_3d.select(1, t);
            let hidden = &hidden_states[0];
            let hh = hidden.matmul(&w_hh_t_0).add_var(&bias_hh_0);

            let ih_r = ih_t.narrow(1, 0, hs);
            let ih_z = ih_t.narrow(1, hs, hs);
            let ih_n = ih_t.narrow(1, 2 * hs, hs);
            let hh_r = hh.narrow(1, 0, hs);
            let hh_z = hh.narrow(1, hs, hs);
            let hh_n = hh.narrow(1, 2 * hs, hs);

            let r = ih_r.add_var(&hh_r).sigmoid();
            let z = ih_z.add_var(&hh_z).sigmoid();
            let n = ih_n.add_var(&r.mul_var(&hh_n)).tanh();
            let h_minus_n = hidden.sub_var(&n);
            let h_new = n.add_var(&z.mul_var(&h_minus_n));
            hidden_states[0] = h_new.clone();

            // Subsequent layers
            let mut layer_input = h_new;

            for (layer_idx, cell) in self.cells.iter().enumerate().skip(1) {
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
    fn stack_outputs(&self, outputs: &[Variable], batch_size: usize, _seq_len: usize) -> Variable {
        if outputs.is_empty() {
            return Variable::new(zeros(&[batch_size, 0, self.hidden_size]), false);
        }

        // Unsqueeze each (batch, hidden) → (batch, 1, hidden), then cat along dim=1
        let unsqueezed: Vec<Variable> = outputs.iter()
            .map(|o| o.unsqueeze(1))
            .collect();
        let refs: Vec<&Variable> = unsqueezed.iter().collect();
        Variable::cat(&refs, 1)
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

    #[test]
    fn test_gru_gradients_reach_parameters() {
        let gru = GRU::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 2 * 3 * 4], &[2, 3, 4]).unwrap(),
            true,
        );
        let output = gru.forward(&input);
        println!("Output shape: {:?}, requires_grad: {}", output.shape(), output.requires_grad());
        let loss = output.sum();
        println!("Loss: {:?}, requires_grad: {}", loss.data().to_vec(), loss.requires_grad());
        loss.backward();

        // Check input gradient
        println!("Input grad: {:?}", input.grad().map(|g| g.to_vec().iter().map(|x| x.abs()).sum::<f32>()));

        let params = gru.parameters();
        println!("Number of parameters: {}", params.len());
        let mut has_grad = false;
        for (i, p) in params.iter().enumerate() {
            let grad = p.grad();
            match grad {
                Some(g) => {
                    let gv = g.to_vec();
                    let sum_abs: f32 = gv.iter().map(|x| x.abs()).sum();
                    println!("Param {} shape {:?} requires_grad={}: grad sum_abs={:.6}",
                        i, p.shape(), p.requires_grad(), sum_abs);
                    if sum_abs > 0.0 {
                        has_grad = true;
                    }
                }
                None => {
                    println!("Param {} shape {:?} requires_grad={}: NO GRADIENT",
                        i, p.shape(), p.requires_grad());
                }
            }
        }
        assert!(has_grad, "At least one GRU parameter should have non-zero gradients");
    }
}
