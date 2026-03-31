//! Module Trait - Neural Network Module Interface
//!
//! # File
//! `crates/axonml-nn/src/module.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
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
    fn set_training(&mut self, _training: bool) {
        // Default: no-op. Stateless modules (Linear, Conv, activations)
        // don't need training mode tracking.
    }

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

    // Simple test module
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
