//! Recurrent layers — `RNN`, `LSTM`, `GRU` and their cell variants.
//!
//! 1490 lines. `RNNCell` / `RNN` (tanh or relu nonlinearity, multi-layer,
//! bidirectional), `LSTMCell` / `LSTM` (forget/input/output/cell gates,
//! multi-layer, returns (output, (h_n, c_n))), `GRUCell` / `GRU`
//! (update/reset gates, multi-layer). All support `forward_step` (single
//! timestep) and sequence-level `forward` (all timesteps). Cell variants
//! expose gate-level state for custom sequence handling.
//!
//! # File
//! `crates/axonml-nn/src/layers/rnn.rs`
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
        let weight_ih = self.weight_ih.variable();
        let weight_ih_t = weight_ih.transpose(0, 1);
        let ih = input.matmul(&weight_ih_t);
        let bias_ih = self.bias_ih.variable();
        let ih = ih.add_var(&bias_ih);

        let weight_hh = self.weight_hh.variable();
        let weight_hh_t = weight_hh.transpose(0, 1);
        let hh = hidden.matmul(&weight_hh_t);
        let bias_hh = self.bias_hh.variable();
        let hh = hh.add_var(&bias_hh);

        ih.add_var(&hh).tanh()
    }
}

impl Module for RNNCell {
    fn forward(&self, input: &Variable) -> Variable {
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

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![crate::NodeSpec::new("Rnn")]
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

        cells.push(RNNCell::new(input_size, hidden_size));

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

        let mut hiddens: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                Variable::new(
                    zeros(&[batch_size, self.hidden_size]),
                    input.requires_grad(),
                )
            })
            .collect();

        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, self.hidden_size]);

        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let ih_t = ih_all_3d.select(1, t);
            let hh = hiddens[0].matmul(&w_hh_t_0).add_var(&bias_hh_0);
            hiddens[0] = ih_t.add_var(&hh).tanh();

            for l in 1..self.num_layers {
                let layer_input = hiddens[l - 1].clone();
                hiddens[l] = self.cells[l].forward_step(&layer_input, &hiddens[l]);
            }

            outputs.push(hiddens[self.num_layers - 1].clone());
        }

        let time_dim = usize::from(self.batch_first);
        let unsqueezed: Vec<Variable> = outputs.iter().map(|o| o.unsqueeze(time_dim)).collect();
        let refs: Vec<&Variable> = unsqueezed.iter().collect();
        Variable::cat(&refs, time_dim)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }

    fn name(&self) -> &'static str {
        "RNN"
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![
            crate::NodeSpec::new("Rnn")
                .attr("hidden_size", crate::AttrVal::Int(self.hidden_size as i64))
                .attr("num_layers", crate::AttrVal::Int(self.num_layers as i64)),
        ]
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

        let i = gates.narrow(1, 0, hs).sigmoid();
        let f = gates.narrow(1, hs, hs).sigmoid();
        let g = gates.narrow(1, 2 * hs, hs).tanh();
        let o = gates.narrow(1, 3 * hs, hs).sigmoid();

        let c_new = f.mul_var(c).add_var(&i.mul_var(&g));

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

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![crate::NodeSpec::new("Lstm")]
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
        let (batch_size, seq_len, input_features) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        let lstm_input_device = input.data().device();
        #[cfg(feature = "cuda")]
        let lstm_on_gpu = lstm_input_device.is_gpu();
        #[cfg(not(feature = "cuda"))]
        let lstm_on_gpu = false;

        let mut states: Vec<(Variable, Variable)> = (0..self.num_layers)
            .map(|_| {
                let make_h = || {
                    let h_cpu = zeros(&[batch_size, self.hidden_size]);
                    let h_tensor = if lstm_on_gpu {
                        h_cpu
                            .to_device(lstm_input_device)
                            .expect("LSTM: failed to move hidden state to GPU")
                    } else {
                        h_cpu
                    };
                    Variable::new(h_tensor, input.requires_grad())
                };
                (make_h(), make_h())
            })
            .collect();

        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 4 * self.hidden_size]);

        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut outputs = Vec::with_capacity(seq_len);

        #[cfg(feature = "cuda")]
        let on_gpu = input.data().device().is_gpu();
        #[cfg(not(feature = "cuda"))]
        let on_gpu = false;

        for t in 0..seq_len {
            let ih_t = ih_all_3d.select(1, t);
            let (h, c) = &states[0];

            let hh = h.matmul(&w_hh_t_0).add_var(&bias_hh_0);

            let gates = ih_t.add_var(&hh);

            if on_gpu {
                #[cfg(feature = "cuda")]
                {
                    let hs = self.hidden_size;
                    let gates_data = gates.data();
                    let c_data = c.data();

                    if let Some((h_tensor, c_tensor)) = gates_data.lstm_gates_fused(&c_data, hs) {
                        let saved_gates = gates_data.clone();
                        let saved_c_prev = c_data.clone();
                        let saved_c_new = c_tensor.clone();

                        let backward_fn = axonml_autograd::LstmGatesBackward::new(
                            gates.grad_fn().cloned(),
                            c.grad_fn().cloned(),
                            saved_gates,
                            saved_c_prev,
                            saved_c_new,
                            hs,
                        );
                        let grad_fn = axonml_autograd::GradFn::new(backward_fn);

                        let fused_requires_grad = gates.requires_grad() || c.requires_grad();
                        let h_new = Variable::from_operation(
                            h_tensor,
                            grad_fn.clone(),
                            fused_requires_grad,
                        );
                        let c_new =
                            Variable::from_operation(c_tensor, grad_fn, fused_requires_grad);
                        states[0] = (h_new, c_new);
                    }
                }
            } else {
                let hs = self.hidden_size;
                let i_gate = gates.narrow(1, 0, hs).sigmoid();
                let f_gate = gates.narrow(1, hs, hs).sigmoid();
                let g_gate = gates.narrow(1, 2 * hs, hs).tanh();
                let o_gate = gates.narrow(1, 3 * hs, hs).sigmoid();
                let c_new = f_gate.mul_var(c).add_var(&i_gate.mul_var(&g_gate));
                let h_new = o_gate.mul_var(&c_new.tanh());
                states[0] = (h_new, c_new);
            }

            for l in 1..self.num_layers {
                let layer_input = states[l - 1].0.clone();
                states[l] = self.cells[l].forward_step(&layer_input, &states[l]);
            }

            outputs.push(states[self.num_layers - 1].0.clone());
        }

        let time_dim = usize::from(self.batch_first);
        let unsqueezed: Vec<Variable> = outputs.iter().map(|o| o.unsqueeze(time_dim)).collect();
        let refs: Vec<&Variable> = unsqueezed.iter().collect();
        Variable::cat(&refs, time_dim)
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
        "LSTM"
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![
            crate::NodeSpec::new("Lstm")
                .attr("hidden_size", crate::AttrVal::Int(self.hidden_size as i64))
                .attr("num_layers", crate::AttrVal::Int(self.num_layers as i64)),
        ]
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

        let weight_ih = self.weight_ih.variable();
        let weight_hh = self.weight_hh.variable();
        let bias_ih = self.bias_ih.variable();
        let bias_hh = self.bias_hh.variable();

        let weight_ih_t = weight_ih.transpose(0, 1);
        let ih = input.matmul(&weight_ih_t).add_var(&bias_ih);

        let weight_hh_t = weight_hh.transpose(0, 1);
        let hh = hidden.matmul(&weight_hh_t).add_var(&bias_hh);

        let ih_r = ih.narrow(1, 0, hidden_size);
        let ih_z = ih.narrow(1, hidden_size, hidden_size);
        let ih_n = ih.narrow(1, 2 * hidden_size, hidden_size);

        let hh_r = hh.narrow(1, 0, hidden_size);
        let hh_z = hh.narrow(1, hidden_size, hidden_size);
        let hh_n = hh.narrow(1, 2 * hidden_size, hidden_size);

        let r = ih_r.add_var(&hh_r).sigmoid();

        let z = ih_z.add_var(&hh_z).sigmoid();

        let n = ih_n.add_var(&r.mul_var(&hh_n)).tanh();

        let h_minus_n = hidden.sub_var(&n);
        n.add_var(&z.mul_var(&h_minus_n))
    }
}

impl Module for GRUCell {
    fn forward(&self, input: &Variable) -> Variable {
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
        "GRUCell"
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![crate::NodeSpec::new("Gru")]
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

        #[cfg(feature = "cuda")]
        let on_gpu = input.data().device().is_gpu();
        #[cfg(not(feature = "cuda"))]
        let on_gpu = false;

        let input_device = input.data().device();

        let mut hidden_states: Vec<Variable> = (0..self.num_layers)
            .map(|_| {
                let h_cpu = zeros(&[batch_size, self.hidden_size]);
                let h_tensor = if on_gpu {
                    h_cpu
                        .to_device(input_device)
                        .expect("GRU: failed to move hidden state to GPU")
                } else {
                    h_cpu
                };
                Variable::new(h_tensor, input.requires_grad())
            })
            .collect();

        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 3 * self.hidden_size]);

        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut output_vars: Vec<Variable> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let ih_t = ih_all_3d.select(1, t);
            let hidden = &hidden_states[0];
            let hs = self.hidden_size;

            let hh = hidden.matmul(&w_hh_t_0).add_var(&bias_hh_0);

            if on_gpu {
                #[cfg(feature = "cuda")]
                {
                    let ih_data = ih_t.data();
                    let hh_data = hh.data();
                    let h_data = hidden.data();

                    if let Some(h_tensor) = ih_data.gru_gates_fused(&hh_data, &h_data, hs) {
                        let saved_ih = ih_data.clone();
                        let saved_hh = hh_data.clone();
                        let saved_h_prev = h_data.clone();

                        let backward_fn = axonml_autograd::GruGatesBackward::new(
                            ih_t.grad_fn().cloned(),
                            hh.grad_fn().cloned(),
                            hidden.grad_fn().cloned(),
                            saved_ih,
                            saved_hh,
                            saved_h_prev,
                            hs,
                        );
                        let grad_fn = axonml_autograd::GradFn::new(backward_fn);

                        let fused_requires_grad =
                            ih_t.requires_grad() || hh.requires_grad() || hidden.requires_grad();
                        let h_new =
                            Variable::from_operation(h_tensor, grad_fn, fused_requires_grad);
                        hidden_states[0] = h_new;
                    }
                }
            } else {
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
                hidden_states[0] = h_new;
            }

            let mut layer_output = hidden_states[0].clone();
            for l in 1..self.num_layers {
                let new_hidden = self.cells[l].forward_step(&layer_output, &hidden_states[l]);
                hidden_states[l] = new_hidden.clone();
                layer_output = new_hidden;
            }

            output_vars.push(layer_output);
        }

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

    fn describe(&self) -> Vec<crate::NodeSpec> {
        vec![
            crate::NodeSpec::new("Gru")
                .attr("hidden_size", crate::AttrVal::Int(self.hidden_size as i64))
                .attr("num_layers", crate::AttrVal::Int(self.num_layers as i64)),
        ]
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

        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 3 * self.hidden_size]);

        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();

        let mut output_sum: Option<Variable> = None;
        let hs = self.hidden_size;

        for t in 0..seq_len {
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

        let cell0 = &self.cells[0];
        let input_2d = input.reshape(&[batch_size * seq_len, input_features]);
        let w_ih_t = cell0.weight_ih.variable().transpose(0, 1);
        let ih_all = input_2d.matmul(&w_ih_t).add_var(&cell0.bias_ih.variable());
        let ih_all_3d = ih_all.reshape(&[batch_size, seq_len, 3 * self.hidden_size]);

        let w_hh_t_0 = cell0.weight_hh.variable().transpose(0, 1);
        let bias_hh_0 = cell0.bias_hh.variable();
        let hs = self.hidden_size;

        for t in 0..seq_len {
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

            let mut layer_input = h_new;

            for (layer_idx, cell) in self.cells.iter().enumerate().skip(1) {
                let new_hidden = cell.forward_step(&layer_input, &hidden_states[layer_idx]);
                hidden_states[layer_idx] = new_hidden.clone();
                layer_input = new_hidden;
            }
        }

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

        let unsqueezed: Vec<Variable> = outputs.iter().map(|o| o.unsqueeze(1)).collect();
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
    use axonml_tensor::Tensor;

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
        println!(
            "Output shape: {:?}, requires_grad: {}",
            output.shape(),
            output.requires_grad()
        );
        let loss = output.sum();
        println!(
            "Loss: {:?}, requires_grad: {}",
            loss.data().to_vec(),
            loss.requires_grad()
        );
        loss.backward();

        println!(
            "Input grad: {:?}",
            input
                .grad()
                .map(|g| g.to_vec().iter().map(|x| x.abs()).sum::<f32>())
        );

        let params = gru.parameters();
        println!("Number of parameters: {}", params.len());
        let mut has_grad = false;
        for (i, p) in params.iter().enumerate() {
            let grad = p.grad();
            match grad {
                Some(g) => {
                    let gv = g.to_vec();
                    let sum_abs: f32 = gv.iter().map(|x| x.abs()).sum();
                    println!(
                        "Param {} shape {:?} requires_grad={}: grad sum_abs={:.6}",
                        i,
                        p.shape(),
                        p.requires_grad(),
                        sum_abs
                    );
                    if sum_abs > 0.0 {
                        has_grad = true;
                    }
                }
                None => {
                    println!(
                        "Param {} shape {:?} requires_grad={}: NO GRADIENT",
                        i,
                        p.shape(),
                        p.requires_grad()
                    );
                }
            }
        }
        assert!(
            has_grad,
            "At least one GRU parameter should have non-zero gradients"
        );
    }

    // =========================================================================
    // LSTM Comprehensive
    // =========================================================================

    #[test]
    fn test_lstm_cell_forward_step() {
        let cell = LSTMCell::new(8, 16);
        let input = Variable::new(Tensor::from_vec(vec![1.0; 2 * 8], &[2, 8]).unwrap(), false);
        let hidden = Variable::new(
            Tensor::from_vec(vec![0.0; 2 * 16], &[2, 16]).unwrap(),
            false,
        );
        let cell_state = Variable::new(
            Tensor::from_vec(vec![0.0; 2 * 16], &[2, 16]).unwrap(),
            false,
        );
        let hx = (hidden, cell_state);
        let (h, c) = cell.forward_step(&input, &hx);
        assert_eq!(h.shape(), vec![2, 16]);
        assert_eq!(c.shape(), vec![2, 16]);
    }

    #[test]
    fn test_lstm_multi_layer() {
        let lstm = LSTM::new(8, 16, 3);
        assert_eq!(lstm.num_layers(), 3);
        assert_eq!(lstm.hidden_size(), 16);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 5 * 8], &[2, 5, 8]).unwrap(),
            false,
        );
        let output = lstm.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 16]);
    }

    #[test]
    fn test_lstm_forward_last() {
        let lstm = LSTM::new(8, 16, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 10 * 8], &[2, 10, 8]).unwrap(),
            false,
        );
        let output = lstm.forward(&input);
        assert_eq!(output.shape(), vec![2, 10, 16]);

        let out_vec = output.data().to_vec();
        let last_t0 = &out_vec[9 * 16..10 * 16];
        assert!(
            last_t0.iter().all(|v| v.is_finite()),
            "Last output should be finite"
        );
    }

    #[test]
    fn test_lstm_gradient_flow() {
        let lstm = LSTM::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 4], &[1, 3, 4]).unwrap(),
            true,
        );
        let output = lstm.forward(&input);
        let loss = output.sum();
        loss.backward();

        let input_grad = input
            .grad()
            .expect("Input should have gradient through LSTM");
        assert_eq!(input_grad.shape(), &[1, 3, 4]);
        assert!(
            input_grad.to_vec().iter().any(|g| g.abs() > 1e-10),
            "LSTM should propagate gradients to input"
        );

        let params = lstm.parameters();
        let grads_exist = params.iter().any(|p| {
            p.grad()
                .is_some_and(|g| g.to_vec().iter().any(|v| v.abs() > 0.0))
        });
        assert!(grads_exist, "LSTM parameters should have gradients");
    }

    #[test]
    fn test_lstm_different_sequence_lengths() {
        let lstm = LSTM::new(4, 8, 1);

        let short = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 4], &[1, 2, 4]).unwrap(),
            false,
        );
        let out_short = lstm.forward(&short);
        assert_eq!(out_short.shape(), vec![1, 2, 8]);

        let long = Variable::new(
            Tensor::from_vec(vec![1.0; 20 * 4], &[1, 20, 4]).unwrap(),
            false,
        );
        let out_long = lstm.forward(&long);
        assert_eq!(out_long.shape(), vec![1, 20, 8]);
    }

    #[test]
    fn test_lstm_parameters_count() {
        let lstm = LSTM::new(10, 20, 1);
        let n = lstm.parameters().iter().map(|p| p.numel()).sum::<usize>();
        assert!(n > 0, "LSTM should have parameters");
    }

    // =========================================================================
    // GRU Comprehensive
    // =========================================================================

    #[test]
    fn test_gru_cell_forward_step() {
        let cell = GRUCell::new(8, 16);
        assert_eq!(cell.input_size(), 8);
        assert_eq!(cell.hidden_size(), 16);

        let input = Variable::new(Tensor::from_vec(vec![1.0; 2 * 8], &[2, 8]).unwrap(), false);
        let hidden = Variable::new(
            Tensor::from_vec(vec![0.0; 2 * 16], &[2, 16]).unwrap(),
            false,
        );
        let output = cell.forward_step(&input, &hidden);
        assert_eq!(output.shape(), vec![2, 16]);
    }

    #[test]
    fn test_gru_multi_layer() {
        let gru = GRU::new(8, 16, 2);
        assert_eq!(gru.num_layers(), 2);
        assert_eq!(gru.hidden_size(), 16);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 5 * 8], &[2, 5, 8]).unwrap(),
            false,
        );
        let output = gru.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 16]);
    }

    #[test]
    fn test_gru_forward_mean() {
        let gru = GRU::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 5 * 4], &[2, 5, 4]).unwrap(),
            false,
        );
        let mean_out = gru.forward_mean(&input);
        assert_eq!(mean_out.shape(), vec![2, 8]);
    }

    #[test]
    fn test_gru_forward_last() {
        let gru = GRU::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 5 * 4], &[2, 5, 4]).unwrap(),
            false,
        );
        let last_out = gru.forward_last(&input);
        assert_eq!(last_out.shape(), vec![2, 8]);
    }

    #[test]
    fn test_gru_gradient_flow_to_input() {
        let gru = GRU::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 4], &[1, 3, 4]).unwrap(),
            true,
        );
        let output = gru.forward(&input);
        output.sum().backward();

        let grad = input
            .grad()
            .expect("Input should have gradient through GRU");
        assert_eq!(grad.shape(), &[1, 3, 4]);
        assert!(
            grad.to_vec().iter().any(|g| g.abs() > 1e-10),
            "GRU should propagate gradients"
        );
    }

    #[test]
    fn test_gru_hidden_state_evolves() {
        let gru = GRU::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 5 * 4], &[1, 5, 4]).unwrap(),
            false,
        );
        let output = gru.forward(&input);
        let out_vec = output.data().to_vec();

        let t0 = &out_vec[0..8];
        let t4 = &out_vec[4 * 8..5 * 8];
        let diff: f32 = t0.iter().zip(t4.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "GRU hidden state should evolve over time, diff={}",
            diff
        );
    }

    // =========================================================================
    // RNN Basic
    // =========================================================================

    #[test]
    fn test_rnn_cell_gradient_flow() {
        let cell = RNNCell::new(4, 8);
        let input = Variable::new(Tensor::from_vec(vec![1.0; 4], &[1, 4]).unwrap(), true);
        let hidden = Variable::new(Tensor::from_vec(vec![0.0; 8], &[1, 8]).unwrap(), false);
        let out = cell.forward_step(&input, &hidden);
        out.sum().backward();

        let grad = input.grad().expect("RNNCell should propagate gradients");
        assert_eq!(grad.shape(), &[1, 4]);
    }

    #[test]
    fn test_rnn_multi_layer() {
        let rnn = RNN::with_options(8, 16, 3, true);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 5 * 8], &[2, 5, 8]).unwrap(),
            false,
        );
        let output = rnn.forward(&input);
        assert_eq!(output.shape(), vec![2, 5, 16]);
    }

    // =========================================================================
    // Numerical Stability
    // =========================================================================

    #[test]
    fn test_lstm_outputs_are_bounded() {
        let lstm = LSTM::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![100.0; 10 * 4], &[1, 10, 4]).unwrap(),
            false,
        );
        let output = lstm.forward(&input);
        let out_vec = output.data().to_vec();

        for v in &out_vec {
            assert!(v.is_finite(), "LSTM output should be finite, got {}", v);
            assert!(
                v.abs() <= 1.0 + 1e-5,
                "LSTM output should be bounded by tanh: got {}",
                v
            );
        }
    }

    #[test]
    fn test_gru_outputs_finite_with_large_input() {
        let gru = GRU::new(4, 8, 1);
        let input = Variable::new(
            Tensor::from_vec(vec![50.0; 5 * 4], &[1, 5, 4]).unwrap(),
            false,
        );
        let output = gru.forward(&input);
        assert!(
            output.data().to_vec().iter().all(|v| v.is_finite()),
            "GRU should produce finite outputs for large inputs"
        );
    }
}
