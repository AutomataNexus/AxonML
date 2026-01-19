//! Sequential - Sequential Container for Modules
//!
//! A container that runs modules in sequence, passing the output
//! of each module as input to the next.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;

use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// Sequential
// =============================================================================

/// A sequential container that chains modules together.
///
/// Modules are added in the order they should be executed.
/// The forward pass executes each module in order, passing
/// the output of one as the input to the next.
///
/// # Example
/// ```ignore
/// let model = Sequential::new()
///     .add(Linear::new(784, 256))
///     .add(ReLU)
///     .add(Linear::new(256, 10));
///
/// let output = model.forward(&input);
/// ```
pub struct Sequential {
    modules: Vec<(String, Box<dyn Module>)>,
    training: bool,
}

impl Sequential {
    /// Creates a new empty Sequential container.
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Adds a module with an auto-generated name.
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        let name = format!("{}", self.modules.len());
        self.modules.push((name, Box::new(module)));
        self
    }

    /// Adds a module with a specific name.
    pub fn add_named<M: Module + 'static>(mut self, name: impl Into<String>, module: M) -> Self {
        self.modules.push((name.into(), Box::new(module)));
        self
    }

    /// Pushes a module (non-builder pattern).
    pub fn push<M: Module + 'static>(&mut self, module: M) {
        let name = format!("{}", self.modules.len());
        self.modules.push((name, Box::new(module)));
    }

    /// Pushes a named module (non-builder pattern).
    pub fn push_named<M: Module + 'static>(&mut self, name: impl Into<String>, module: M) {
        self.modules.push((name.into(), Box::new(module)));
    }

    /// Returns the number of modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Returns an iterator over named modules.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn Module)> {
        self.modules.iter().map(|(n, m)| (n.as_str(), m.as_ref()))
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Variable) -> Variable {
        let mut x = input.clone();
        for (_, module) in &self.modules {
            x = module.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.modules
            .iter()
            .flat_map(|(_, m)| m.parameters())
            .collect()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (module_name, module) in &self.modules {
            for (param_name, param) in module.named_parameters() {
                params.insert(format!("{module_name}.{param_name}"), param);
            }
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for (_, module) in &mut self.modules {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "Sequential"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    // Test identity module
    struct TestIdentity;

    impl Module for TestIdentity {
        fn forward(&self, input: &Variable) -> Variable {
            input.clone()
        }
    }

    // Test doubling module
    struct TestDouble;

    impl Module for TestDouble {
        fn forward(&self, input: &Variable) -> Variable {
            input.add_var(input)
        }
    }

    #[test]
    fn test_sequential_creation() {
        let seq = Sequential::new().add(TestIdentity).add(TestIdentity);
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_sequential_forward() {
        let seq = Sequential::new().add(TestDouble).add(TestDouble);

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), false);
        let output = seq.forward(&input);

        // Double twice: 1*2*2=4, 2*2*2=8
        assert_eq!(output.data().to_vec(), vec![4.0, 8.0]);
    }

    #[test]
    fn test_sequential_named() {
        let seq = Sequential::new()
            .add_named("layer1", TestIdentity)
            .add_named("layer2", TestDouble);

        let names: Vec<&str> = seq.iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["layer1", "layer2"]);
    }
}
