//! `Module` trait — the core interface for all neural network layers.
//!
//! 260 lines. `Module` requires: `forward(&self, &Variable) -> Variable`,
//! `parameters(&self) -> Vec<Parameter>`, `train(&mut self)`, `eval(&mut self)`,
//! `is_training(&self) -> bool`, `zero_grad(&mut self)`, `name() -> &str`,
//! `to_device(&mut self, Device)`, `named_parameters() -> HashMap`. Also
//! `ModuleList` (heterogeneous `Vec<Box<dyn Module>>` with forward-sequential,
//! parameter aggregation, and train/eval propagation).
//!
//! # File
//! `crates/axonml-nn/src/module.rs`
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
use axonml_core::Device;

use crate::parameter::Parameter;

// =============================================================================
// Graph description primitives
// =============================================================================

/// A typed attribute value attached to a graph node.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrVal {
    /// Signed integer attribute.
    Int(i64),
    /// Floating-point attribute.
    Float(f64),
    /// List of signed integers.
    Ints(Vec<i64>),
    /// String attribute.
    Str(String),
    /// Boolean attribute.
    Bool(bool),
}

/// A single node in a module's graph description: op name, attributes, and local parameter names.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeSpec {
    /// Operation name (e.g. "Gemm", "Conv", "BatchNorm").
    pub op: String,
    /// Named attributes for this op.
    pub attrs: Vec<(String, AttrVal)>,
    /// Local parameter names this op consumes.
    pub params: Vec<String>,
}

impl NodeSpec {
    /// Creates a new node spec for the given op name.
    pub fn new(op: &str) -> Self {
        Self {
            op: op.to_string(),
            attrs: Vec::new(),
            params: Vec::new(),
        }
    }

    /// Adds an attribute and returns self for chaining.
    pub fn attr(mut self, key: &str, val: AttrVal) -> Self {
        self.attrs.push((key.to_string(), val));
        self
    }

    /// Adds a parameter name and returns self for chaining.
    pub fn param(mut self, name: &str) -> Self {
        self.params.push(name.to_string());
        self
    }
}

// =============================================================================
// Module Trait
// =============================================================================

/// Core trait for all neural network modules.
///
/// Every layer in Axonml implements this trait, which provides:
/// - Forward pass computation
/// - Parameter management
/// - Training/evaluation mode switching
/// - Module naming
pub trait Module: Send + Sync {
    /// Performs the forward pass.
    ///
    /// # Arguments
    /// * `input` - Input variable
    ///
    /// # Returns
    /// Output variable after applying this module's transformation.
    fn forward(&self, input: &Variable) -> Variable;

    /// Returns all parameters of this module.
    ///
    /// This includes parameters from all child modules.
    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    /// Returns named parameters of this module.
    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    /// Returns persistent non-parameter buffers (e.g. BatchNorm running mean/var).
    fn named_buffers(&self) -> HashMap<String, axonml_tensor::Tensor<f32>> {
        HashMap::new()
    }

    /// Sets a named buffer; returns true if the buffer exists on this module.
    fn set_buffer(&self, _name: &str, _value: axonml_tensor::Tensor<f32>) -> bool {
        false
    }

    /// Describes this module's forward op(s) for graph tracing.
    ///
    /// Leaves emit their op(s) + attrs + local param names; containers expose
    /// ordered children (via `named_children`) so a tracer can compose the full graph.
    fn describe(&self) -> Vec<NodeSpec> {
        Vec::new()
    }

    /// Returns ordered (name, child) pairs for container modules.
    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        Vec::new()
    }

    /// Returns the number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .filter(|p| p.requires_grad())
            .map(|p| p.numel())
            .sum()
    }

    /// Sets the module to training mode.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Sets the module to evaluation mode.
    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Sets the training mode.
    /// Sets the training mode.
    ///
    /// Modules with training-dependent behavior (Dropout, BatchNorm) MUST
    /// override this AND `is_training()` to track the mode in an internal field.
    fn set_training(&mut self, _training: bool) {}

    /// Returns whether the module is in training mode.
    ///
    /// Default returns `true`. Modules that override `set_training()` should
    /// also override this to return their tracked state.
    fn is_training(&self) -> bool {
        true
    }

    /// Zeros all gradients of parameters.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Moves all parameters to the specified device.
    ///
    /// **Note:** This only moves `Parameter` tensors. Modules with non-parameter
    /// state (e.g., BatchNorm running_mean/running_var) should override this
    /// method to also move their buffers.
    fn to_device(&self, device: Device) {
        for param in self.parameters() {
            param.to_device(device);
        }
    }

    /// Returns the module name for debugging.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

// =============================================================================
// ModuleList
// =============================================================================

/// A container for holding a list of modules.
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl ModuleList {
    /// Creates a new empty ModuleList.
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Creates a ModuleList from a vector of modules.
    pub fn from_vec(modules: Vec<Box<dyn Module>>) -> Self {
        Self {
            modules,
            training: true,
        }
    }

    /// Adds a module to the list.
    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    /// Returns the number of modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Returns true if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Returns an iterator over the modules.
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Module>> {
        self.modules.iter()
    }

    /// Returns a mutable iterator over the modules.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Module>> {
        self.modules.iter_mut()
    }

    /// Gets a module by index.
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|m| m.as_ref())
    }
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleList {
    fn forward(&self, input: &Variable) -> Variable {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.named_parameters() {
                params.insert(format!("{i}.{name}"), param);
            }
        }
        params
    }

    fn named_buffers(&self) -> HashMap<String, axonml_tensor::Tensor<f32>> {
        let mut buffers = HashMap::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, buf) in module.named_buffers() {
                buffers.insert(format!("{i}.{name}"), buf);
            }
        }
        buffers
    }

    fn set_buffer(&self, name: &str, value: axonml_tensor::Tensor<f32>) -> bool {
        match name.split_once('.') {
            Some((idx, rest)) => match idx.parse::<usize>() {
                Ok(i) => self
                    .modules
                    .get(i)
                    .is_some_and(|m| m.set_buffer(rest, value)),
                Err(_) => false,
            },
            None => false,
        }
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.modules
            .iter()
            .enumerate()
            .map(|(i, m)| (i.to_string(), m.as_ref()))
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in &mut self.modules {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "ModuleList"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    struct Identity;

    impl Module for Identity {
        fn forward(&self, input: &Variable) -> Variable {
            input.clone()
        }

        fn name(&self) -> &'static str {
            "Identity"
        }
    }

    #[test]
    fn test_module_list() {
        let mut list = ModuleList::new();
        list.push(Identity);
        list.push(Identity);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_module_list_forward() {
        let mut list = ModuleList::new();
        list.push(Identity);

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let output = list.forward(&input);
        assert_eq!(output.data().to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
