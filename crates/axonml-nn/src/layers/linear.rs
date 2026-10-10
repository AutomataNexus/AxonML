//! `Linear` — fully connected layer (y = xW^T + b).
//!
//! 245 lines. `Linear::new(in_features, out_features)` with Xavier-uniform
//! weight init and optional bias. Implements `Module` (forward, parameters,
//! train/eval, zero_grad, to_device). Also `Linear::no_bias(in, out)` for
//! bias-free variants used in some LLM architectures.
//!
//! # File
//! `crates/axonml-nn/src/layers/linear.rs`
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
use axonml_tensor::Tensor;

use crate::init::{kaiming_uniform, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// Linear
// =============================================================================

/// Applies a linear transformation to the input.
///
/// y = xW^T + b
///
/// # Arguments
/// * `in_features` - Size of each input sample
/// * `out_features` - Size of each output sample
/// * `bias` - If true, adds a learnable bias (default: true)
///
/// # Shape
/// - Input: (*, in_features) where * means any number of dimensions
/// - Output: (*, out_features)
///
/// # Example
/// ```ignore
/// let linear = Linear::new(20, 30);
/// let input = Variable::new(randn(&[128, 20]), true);
/// let output = linear.forward(&input);  // Shape: [128, 30]
/// ```
/// Low-rank adapter on a frozen `Linear`: `y += (x @ Aᵀ) @ Bᵀ · (alpha / rank)`.
/// `b` starts at zero so an attached adapter is an exact identity until it is trained.
#[derive(Clone)]
pub struct LoraAdapter {
    /// `[rank, in_features]`.
    pub a: Parameter,
    /// `[out_features, rank]`.
    pub b: Parameter,
    /// `alpha / rank`.
    pub scale: f32,
}

/// Fully connected layer `y = x W^T + b` (optionally with a LoRA adapter).
#[derive(Clone)]
pub struct Linear {
    /// Weight matrix of shape (out_features, in_features).
    pub weight: Parameter,
    /// Bias vector of shape (out_features).
    pub bias: Option<Parameter>,
    /// Opt-in low-rank adapter. `None` (the default) keeps this layer byte-identical to before.
    pub lora: Option<LoraAdapter>,
    /// Input features.
    in_features: usize,
    /// Output features.
    out_features: usize,
}

impl Linear {
    /// Creates a new Linear layer with bias.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, true)
    }

    /// Creates a new Linear layer with optional bias.
    pub fn with_bias(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight_data = kaiming_uniform(out_features, in_features);
        let weight = Parameter::named("weight", weight_data, true);

        let bias_param = if bias {
            let bias_data = zeros(&[out_features]);
            Some(Parameter::named("bias", bias_data, true))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            lora: None,
            in_features,
            out_features,
        }
    }

    /// Creates a Linear layer from existing weight and bias tensors.
    pub fn from_weights(weight: Tensor<f32>, bias: Option<Tensor<f32>>) -> Self {
        let out_features = weight.shape()[0];
        let in_features = weight.shape()[1];

        Self {
            weight: Parameter::named("weight", weight, true),
            bias: bias.map(|b| Parameter::named("bias", b, true)),
            in_features,
            out_features,
            lora: None,
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

    // ── low-rank adapters: one small module per domain, trained alone against a frozen base ──

    /// Attach a rank-`r` adapter and return its two parameters. `b` is zero, so the layer's output is
    /// unchanged until the adapter trains. The base weight is left exactly as it is.
    pub fn attach_lora(&mut self, rank: usize, alpha: f32) -> Vec<Parameter> {
        // ── the adapter lands on the base weight's device: attach is normally called after `to_device` ──
        let dev = self.weight.data().device();
        let a_data = kaiming_uniform(rank, self.in_features)
            .to_device(dev)
            .expect("lora_a to device");
        let b_data = zeros(&[self.out_features, rank])
            .to_device(dev)
            .expect("lora_b to device");
        let a = Parameter::named("lora_a", a_data, true);
        let b = Parameter::named("lora_b", b_data, true);
        self.lora = Some(LoraAdapter {
            a: a.clone(),
            b: b.clone(),
            scale: alpha / rank as f32,
        });
        vec![a, b]
    }

    /// Drop the adapter, restoring the bare layer.
    pub fn detach_lora(&mut self) {
        self.lora = None;
    }

    /// The adapter's parameters, if one is attached.
    pub fn lora_parameters(&self) -> Vec<Parameter> {
        self.lora
            .as_ref()
            .map_or_else(Vec::new, |l| vec![l.a.clone(), l.b.clone()])
    }

    /// Fold the adapter into the base weight (`w += scale * b @ a`) and drop it, so the layer is a plain
    /// `Linear` again. An exporter that walks `parameters()` positionally sees no extra adapter tensors.
    pub fn merge_lora(&mut self) {
        let Some(l) = self.lora.take() else { return };
        let delta =
            l.b.data()
                .matmul(&l.a.data())
                .expect("lora merge matmul")
                .mul_scalar(l.scale);
        let merged = self.weight.data().add(&delta).expect("lora merge add");
        self.weight.variable().set_data(merged);
    }

    /// Freeze the base weight and bias so only an attached adapter trains. The backward pass then
    /// skips the base gradient entirely rather than computing one nothing consumes.
    pub fn freeze_base(&mut self) {
        self.weight = Parameter::from_variable(Variable::new(self.weight.data(), false));
        if let Some(b) = &self.bias {
            self.bias = Some(Parameter::from_variable(Variable::new(b.data(), false)));
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Variable {
        let input_shape = input.shape();
        let batch_dims: Vec<usize> = input_shape[..input_shape.len() - 1].to_vec();

        let total_batch: usize = batch_dims.iter().product();
        let input_2d = if input_shape.len() > 2 {
            input.reshape(&[total_batch, self.in_features])
        } else {
            input.clone()
        };

        let weight_var = self.weight.variable();
        let weight_t = weight_var.transpose(0, 1);
        let mut output = input_2d.matmul(&weight_t);

        // ── adapter: the low-rank path runs beside the frozen base and is added back ──
        if let Some(ref l) = self.lora {
            let down = input_2d.matmul(&l.a.variable().transpose(0, 1));
            let up = down.matmul(&l.b.variable().transpose(0, 1));
            output = output.add_var(&up.mul_scalar(l.scale));
        }

        if let Some(ref bias) = self.bias {
            let bias_var = bias.variable();
            output = output.add_var(&bias_var);
        }

        if batch_dims.len() > 1 || (batch_dims.len() == 1 && input_shape.len() > 2) {
            let mut output_shape: Vec<usize> = batch_dims.clone();
            output_shape.push(self.out_features);
            output.reshape(&output_shape)
        } else {
            output
        }
    }

    fn describe(&self) -> Vec<crate::NodeSpec> {
        let mut node = crate::NodeSpec::new("Gemm")
            .attr("trans_b", crate::AttrVal::Bool(true))
            .param("weight");
        if self.bias.is_some() {
            node = node.param("bias");
        }
        vec![node]
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        if let Some(ref l) = self.lora {
            params.push(l.a.clone());
            params.push(l.b.clone());
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
        "Linear"
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(10, 5);
        assert_eq!(linear.in_features(), 10);
        assert_eq!(linear.out_features(), 5);
        assert!(linear.bias.is_some());
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::with_bias(10, 5, false);
        assert!(linear.bias.is_none());
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(3, 2);

        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(),
            false,
        );
        let output = linear.forward(&input);

        assert_eq!(output.shape(), vec![1, 2]);
    }

    #[test]
    fn test_linear_batch_forward() {
        let linear = Linear::new(4, 2);

        let input = Variable::new(Tensor::from_vec(vec![1.0; 12], &[3, 4]).unwrap(), false);
        let output = linear.forward(&input);

        assert_eq!(output.shape(), vec![3, 2]);
    }

    #[test]
    fn test_linear_parameters() {
        let linear = Linear::new(10, 5);
        let params = linear.parameters();
        assert_eq!(params.len(), 2);

        let linear_no_bias = Linear::with_bias(10, 5, false);
        let params_no_bias = linear_no_bias.parameters();
        assert_eq!(params_no_bias.len(), 1);
    }

    #[test]
    fn test_linear_num_parameters() {
        let linear = Linear::new(10, 5);
        assert_eq!(linear.num_parameters(), 55);
    }
}
