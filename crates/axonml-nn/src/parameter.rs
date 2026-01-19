//! Parameter - Learnable Parameter Wrapper
//!
//! Wraps Variables that are learnable parameters of a module.
//! Parameters are special Variables that are registered with modules
//! and can be optimized during training.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::sync::Arc;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;
use parking_lot::RwLock;

// =============================================================================
// Parameter
// =============================================================================

/// A learnable parameter of a neural network module.
///
/// Parameters wrap Variables and provide additional functionality:
/// - Automatic requires_grad=true by default
/// - Registration with parent modules
/// - Easy access to data and gradients
#[derive(Clone)]
pub struct Parameter {
    /// The underlying variable.
    data: Arc<RwLock<Variable>>,
    /// Parameter name (for debugging and serialization).
    name: String,
}

impl Parameter {
    /// Creates a new parameter from a tensor.
    ///
    /// # Arguments
    /// * `data` - The tensor data
    /// * `requires_grad` - Whether to track gradients (default true)
    pub fn new(data: Tensor<f32>, requires_grad: bool) -> Self {
        Self {
            data: Arc::new(RwLock::new(Variable::new(data, requires_grad))),
            name: String::new(),
        }
    }

    /// Creates a new parameter with a name.
    pub fn named(name: impl Into<String>, data: Tensor<f32>, requires_grad: bool) -> Self {
        Self {
            data: Arc::new(RwLock::new(Variable::new(data, requires_grad))),
            name: name.into(),
        }
    }

    /// Creates a parameter from an existing Variable.
    pub fn from_variable(var: Variable) -> Self {
        Self {
            data: Arc::new(RwLock::new(var)),
            name: String::new(),
        }
    }

    /// Returns the parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets the parameter name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Returns a clone of the underlying Variable.
    pub fn variable(&self) -> Variable {
        self.data.read().clone()
    }

    /// Returns a clone of the tensor data.
    pub fn data(&self) -> Tensor<f32> {
        self.data.read().data()
    }

    /// Returns the shape of the parameter.
    pub fn shape(&self) -> Vec<usize> {
        self.data.read().shape()
    }

    /// Returns the number of elements.
    pub fn numel(&self) -> usize {
        self.data.read().numel()
    }

    /// Returns whether this parameter requires gradients.
    pub fn requires_grad(&self) -> bool {
        self.data.read().requires_grad()
    }

    /// Returns the gradient if available.
    pub fn grad(&self) -> Option<Tensor<f32>> {
        self.data.read().grad()
    }

    /// Zeros the gradient.
    pub fn zero_grad(&self) {
        self.data.read().zero_grad();
    }

    /// Updates the parameter data in-place.
    ///
    /// Used by optimizers to update weights.
    pub fn update_data(&self, new_data: Tensor<f32>) {
        let mut guard = self.data.write();
        let requires_grad = guard.requires_grad();
        *guard = Variable::new(new_data, requires_grad);
    }

    /// Applies a function to the parameter data.
    pub fn apply_update<F>(&self, f: F)
    where
        F: FnOnce(&Tensor<f32>) -> Tensor<f32>,
    {
        let current = self.data();
        let updated = f(&current);
        self.update_data(updated);
    }
}

impl std::fmt::Debug for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parameter")
            .field("name", &self.name)
            .field("shape", &self.shape())
            .field("requires_grad", &self.requires_grad())
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
    fn test_parameter_creation() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let param = Parameter::new(data, true);
        assert!(param.requires_grad());
        assert_eq!(param.shape(), vec![3]);
        assert_eq!(param.numel(), 3);
    }

    #[test]
    fn test_parameter_named() {
        let data = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let param = Parameter::named("weight", data, true);
        assert_eq!(param.name(), "weight");
    }

    #[test]
    fn test_parameter_update() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let param = Parameter::new(data, true);

        let new_data = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        param.update_data(new_data);

        assert_eq!(param.data().to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_parameter_apply_update() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let param = Parameter::new(data, true);

        param.apply_update(|d| d.mul_scalar(2.0));

        assert_eq!(param.data().to_vec(), vec![2.0, 4.0, 6.0]);
    }
}
