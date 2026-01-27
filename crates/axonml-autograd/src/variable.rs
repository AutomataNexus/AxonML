//! Variable - Tensor with Gradient Tracking
//!
//! The Variable struct wraps a Tensor and adds automatic differentiation
//! capabilities. Variables track their computational history to enable
//! gradient computation via backpropagation.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use parking_lot::RwLock;

use axonml_tensor::Tensor;

use crate::functions::{
    AddBackward, DivBackward, MatMulBackward, MeanBackward, MulBackward, NarrowBackward,
    NegBackward, PowBackward, ReluBackward, ReshapeBackward, SigmoidBackward, SubBackward,
    SumBackward, TanhBackward, TransposeBackward,
};
use crate::grad_fn::{AccumulateGrad, GradAccumulator, GradFn};
use crate::graph::{with_graph, GraphNode};
use crate::no_grad::is_grad_enabled;

// =============================================================================
// Variable Struct
// =============================================================================

/// A tensor with automatic differentiation support.
///
/// Variable wraps a Tensor and tracks operations performed on it to enable
/// automatic gradient computation. When `requires_grad` is true, all operations
/// are recorded in a computational graph.
#[derive(Clone)]
pub struct Variable {
    /// The underlying tensor data.
    data: Arc<RwLock<Tensor<f32>>>,
    /// Shared gradient accumulator (for leaf variables, shared with `AccumulateGrad`).
    grad: GradAccumulator,
    /// Whether this variable requires gradient computation.
    requires_grad: bool,
    /// Whether this is a leaf variable (created by user, not an operation).
    is_leaf: bool,
    /// The gradient function for backpropagation.
    grad_fn: Option<GradFn>,
    /// Graph node for this variable.
    node: Option<Arc<GraphNode>>,
}

impl Variable {
    /// Creates a new variable from a tensor.
    ///
    /// # Arguments
    /// * `data` - The tensor data
    /// * `requires_grad` - Whether to track gradients for this variable
    #[must_use]
    pub fn new(data: Tensor<f32>, requires_grad: bool) -> Self {
        // Create shared gradient accumulator
        let grad: GradAccumulator = Arc::new(RwLock::new(None));

        let node = if requires_grad {
            Some(with_graph(|g| g.register_leaf(true)))
        } else {
            None
        };

        // Create AccumulateGrad with shared gradient storage
        let grad_fn = if requires_grad {
            Some(GradFn::new(AccumulateGrad::new(Arc::clone(&grad))))
        } else {
            None
        };

        Self {
            data: Arc::new(RwLock::new(data)),
            grad,
            requires_grad,
            is_leaf: true,
            grad_fn,
            node,
        }
    }

    /// Creates a variable that doesn't require gradients.
    #[must_use]
    pub fn from_tensor(data: Tensor<f32>) -> Self {
        Self::new(data, false)
    }

    /// Creates a new variable from an operation result.
    fn from_operation(data: Tensor<f32>, grad_fn: GradFn, requires_grad: bool) -> Self {
        let node = if requires_grad {
            Some(with_graph(|g| g.register_operation(grad_fn.clone(), true)))
        } else {
            None
        };

        Self {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            is_leaf: false,
            grad_fn: if requires_grad { Some(grad_fn) } else { None },
            node,
        }
    }

    /// Returns a reference to the underlying tensor data.
    #[must_use]
    pub fn data(&self) -> Tensor<f32> {
        self.data.read().clone()
    }

    /// Returns the shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.data.read().shape().to_vec()
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.data.read().ndim()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.read().numel()
    }

    /// Returns whether this variable requires gradients.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns whether this is a leaf variable.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// Returns the gradient of this variable.
    ///
    /// Only available for leaf variables after `backward()` has been called.
    #[must_use]
    pub fn grad(&self) -> Option<Tensor<f32>> {
        self.grad.read().clone()
    }

    /// Returns the gradient function.
    #[must_use]
    pub fn grad_fn(&self) -> Option<&GradFn> {
        self.grad_fn.as_ref()
    }

    /// Sets the gradient (used during backward pass).
    pub fn set_grad(&self, grad: Tensor<f32>) {
        *self.grad.write() = Some(grad);
    }

    /// Accumulates gradient (adds to existing gradient).
    pub fn accumulate_grad(&self, grad: &Tensor<f32>) {
        let mut grad_lock = self.grad.write();
        if let Some(ref existing) = *grad_lock {
            *grad_lock = Some(existing.add(grad).unwrap());
        } else {
            *grad_lock = Some(grad.clone());
        }
    }

    /// Clears the gradient.
    pub fn zero_grad(&self) {
        *self.grad.write() = None;
    }

    /// Detaches this variable from the computation graph.
    ///
    /// Returns a new variable with the same data but no gradient history.
    #[must_use]
    pub fn detach(&self) -> Self {
        Self {
            data: Arc::new(RwLock::new(self.data.read().clone())),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            is_leaf: true,
            grad_fn: None,
            node: None,
        }
    }

    /// Returns a new variable with `requires_grad` set.
    #[must_use]
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad && self.is_leaf {
            // AccumulateGrad shares the gradient accumulator with this variable
            self.grad_fn = Some(GradFn::new(AccumulateGrad::new(Arc::clone(&self.grad))));
            self.node = Some(with_graph(|g| g.register_leaf(true)));
        }
        self
    }

    /// Computes gradients via backpropagation.
    ///
    /// This should only be called on scalar (single-element) tensors,
    /// typically the loss value.
    pub fn backward(&self) {
        assert!(
            self.requires_grad,
            "Cannot call backward on a variable that doesn't require gradients"
        );

        assert!(
            (self.numel() == 1),
            "backward() can only be called on scalar tensors"
        );

        // Start with gradient of 1.0 for the output
        let grad_output = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        crate::backward::backward(self, &grad_output);
    }

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    /// Element-wise addition.
    #[must_use]
    pub fn add_var(&self, other: &Variable) -> Variable {
        let result = self.data.read().add(&other.data.read()).unwrap();
        let requires_grad = (self.requires_grad || other.requires_grad) && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(AddBackward::new(
                self.grad_fn.clone(),
                other.grad_fn.clone(),
                self.shape(),
                other.shape(),
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Element-wise subtraction.
    #[must_use]
    pub fn sub_var(&self, other: &Variable) -> Variable {
        let result = self.data.read().sub(&other.data.read()).unwrap();
        let requires_grad = (self.requires_grad || other.requires_grad) && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SubBackward::new(
                self.grad_fn.clone(),
                other.grad_fn.clone(),
                self.shape(),
                other.shape(),
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Element-wise multiplication.
    #[must_use]
    pub fn mul_var(&self, other: &Variable) -> Variable {
        let self_data = self.data.read().clone();
        let other_data = other.data.read().clone();
        let result = self_data.mul(&other_data).unwrap();
        let requires_grad = (self.requires_grad || other.requires_grad) && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(MulBackward::new(
                self.grad_fn.clone(),
                other.grad_fn.clone(),
                self_data,
                other_data,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Element-wise division.
    #[must_use]
    pub fn div_var(&self, other: &Variable) -> Variable {
        let self_data = self.data.read().clone();
        let other_data = other.data.read().clone();
        let result = self_data.div(&other_data).unwrap();
        let requires_grad = (self.requires_grad || other.requires_grad) && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(DivBackward::new(
                self.grad_fn.clone(),
                other.grad_fn.clone(),
                self_data,
                other_data,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Negation.
    #[must_use]
    pub fn neg_var(&self) -> Variable {
        let result = self.data.read().neg();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(NegBackward::new(self.grad_fn.clone()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Matrix multiplication.
    #[must_use]
    pub fn matmul(&self, other: &Variable) -> Variable {
        let self_data = self.data.read().clone();
        let other_data = other.data.read().clone();
        let result = self_data.matmul(&other_data).unwrap();
        let requires_grad = (self.requires_grad || other.requires_grad) && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(MatMulBackward::new(
                self.grad_fn.clone(),
                other.grad_fn.clone(),
                self_data,
                other_data,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Power operation.
    #[must_use]
    pub fn pow(&self, exponent: f32) -> Variable {
        let self_data = self.data.read().clone();
        let result = self_data.pow(exponent);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(PowBackward::new(self.grad_fn.clone(), self_data, exponent));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    // =========================================================================
    // Activation Functions
    // =========================================================================

    /// `ReLU` activation.
    #[must_use]
    pub fn relu(&self) -> Variable {
        let self_data = self.data.read().clone();
        let result = self_data.relu();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(ReluBackward::new(self.grad_fn.clone(), self_data));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Sigmoid activation.
    #[must_use]
    pub fn sigmoid(&self) -> Variable {
        let result = self.data.read().sigmoid();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SigmoidBackward::new(self.grad_fn.clone(), result.clone()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Tanh activation.
    #[must_use]
    pub fn tanh(&self) -> Variable {
        let result = self.data.read().tanh();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(TanhBackward::new(self.grad_fn.clone(), result.clone()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    // =========================================================================
    // Reduction Operations
    // =========================================================================

    /// Sum all elements.
    #[must_use]
    pub fn sum(&self) -> Variable {
        let self_data = self.data.read().clone();
        let result = self_data.sum(); // Returns a scalar Tensor
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SumBackward::new(self.grad_fn.clone(), self.shape()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Mean of all elements.
    #[must_use]
    pub fn mean(&self) -> Variable {
        let self_data = self.data.read().clone();
        let result = self_data.mean().unwrap(); // Returns a scalar Tensor
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(MeanBackward::new(self.grad_fn.clone(), self.shape()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    // =========================================================================
    // Loss Functions
    // =========================================================================

    /// Mean Squared Error loss.
    #[must_use]
    pub fn mse_loss(&self, target: &Variable) -> Variable {
        let diff = self.sub_var(target);
        let squared = diff.pow(2.0);
        squared.mean()
    }

    /// Binary Cross Entropy loss (expects sigmoid output).
    #[must_use]
    pub fn binary_cross_entropy(&self, target: &Variable) -> Variable {
        let eps = Variable::from_tensor(Tensor::scalar(1e-7));
        let one = Variable::from_tensor(Tensor::scalar(1.0));

        // -[y * log(p + eps) + (1 - y) * log(1 - p + eps)]
        let log_p = self.add_var(&eps);
        let log_1_p = one.sub_var(self).add_var(&eps);

        let term1 = target.mul_var(&Variable::from_tensor(log_p.data().ln()));
        let term2 = one
            .sub_var(target)
            .mul_var(&Variable::from_tensor(log_1_p.data().ln()));

        term1.add_var(&term2).neg_var().mean()
    }

    // =========================================================================
    // Shape Operations
    // =========================================================================

    /// Reshapes the variable to a new shape.
    #[must_use]
    pub fn reshape(&self, shape: &[usize]) -> Variable {
        let isize_shape: Vec<isize> = shape.iter().map(|&x| x as isize).collect();
        let original_shape = self.shape();
        let new_data = self
            .data()
            .reshape(&isize_shape)
            .unwrap_or_else(|_| self.data().clone());
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(ReshapeBackward::new(self.grad_fn.clone(), original_shape));
            Variable::from_operation(new_data, grad_fn, true)
        } else {
            Variable::from_tensor(new_data)
        }
    }

    /// Transposes two dimensions.
    #[must_use]
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Variable {
        let new_data = self
            .data()
            .transpose(dim0 as i64, dim1 as i64)
            .unwrap_or_else(|_| self.data().clone());
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(TransposeBackward::new(self.grad_fn.clone(), dim0, dim1));
            Variable::from_operation(new_data, grad_fn, true)
        } else {
            Variable::from_tensor(new_data)
        }
    }

    /// Slices the variable along specified ranges.
    #[must_use]
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Variable {
        let new_data = self.data().slice(ranges);
        Variable::new(new_data, self.requires_grad())
    }

    /// Narrows the variable along a dimension.
    ///
    /// Returns a view of the tensor containing elements from `start` to `start + length`
    /// along the specified dimension. This operation preserves gradients for backpropagation.
    #[must_use]
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Variable {
        let input_shape = self.shape();
        let new_data = self
            .data()
            .narrow(dim, start, length)
            .unwrap_or_else(|_| self.data().clone());
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(NarrowBackward::new(
                self.grad_fn.clone(),
                input_shape,
                dim,
                start,
            ));
            Variable::from_operation(new_data, grad_fn, true)
        } else {
            Variable::from_tensor(new_data)
        }
    }

    /// Expands the variable to a new shape (broadcast).
    #[must_use]
    pub fn expand(&self, shape: &[usize]) -> Variable {
        let new_data = self.data().broadcast_to(shape);
        Variable::new(new_data, self.requires_grad())
    }

    // =========================================================================
    // Scalar Operations
    // =========================================================================

    /// Multiplies by a scalar.
    #[must_use]
    pub fn mul_scalar(&self, scalar: f32) -> Variable {
        let data = self.data();
        let shape = data.shape();
        let numel: usize = shape.iter().product();
        let scalar_tensor = Tensor::from_vec(vec![scalar; numel], shape).unwrap();
        let scalar_var = Variable::new(scalar_tensor, false);
        self.mul_var(&scalar_var)
    }

    /// Adds a scalar.
    #[must_use]
    pub fn add_scalar(&self, scalar: f32) -> Variable {
        let data = self.data();
        let shape = data.shape();
        let numel: usize = shape.iter().product();
        let scalar_tensor = Tensor::from_vec(vec![scalar; numel], shape).unwrap();
        let scalar_var = Variable::new(scalar_tensor, false);
        self.add_var(&scalar_var)
    }

    /// Subtracts a scalar.
    #[must_use]
    pub fn sub_scalar(&self, scalar: f32) -> Variable {
        self.add_scalar(-scalar)
    }

    /// Divides by a scalar.
    #[must_use]
    pub fn div_scalar(&self, scalar: f32) -> Variable {
        self.mul_scalar(1.0 / scalar)
    }

    // =========================================================================
    // Additional Activations
    // =========================================================================

    /// GELU activation function (Gaussian Error Linear Unit).
    #[must_use]
    pub fn gelu(&self) -> Variable {
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let data = self.data();
        let result = data.gelu();
        Variable::new(result, self.requires_grad())
    }

    /// SiLU/Swish activation function (x * sigmoid(x)).
    #[must_use]
    pub fn silu(&self) -> Variable {
        let data = self.data();
        let result = data.silu();
        Variable::new(result, self.requires_grad())
    }

    /// Square root.
    #[must_use]
    pub fn sqrt(&self) -> Variable {
        let data = self.data();
        let result = data.sqrt();
        Variable::new(result, self.requires_grad())
    }

    // =========================================================================
    // Softmax Operations
    // =========================================================================

    /// Softmax along specified dimension.
    #[must_use]
    pub fn softmax(&self, dim: i32) -> Variable {
        let data = self.data();
        let result = data.softmax(dim);
        Variable::new(result, self.requires_grad())
    }

    /// Log softmax along specified dimension.
    #[must_use]
    pub fn log_softmax(&self, dim: i32) -> Variable {
        let data = self.data();
        let result = data.log_softmax(dim);
        Variable::new(result, self.requires_grad())
    }

    // =========================================================================
    // Reduction Operations with Dimensions
    // =========================================================================

    /// Mean along a dimension, optionally keeping the dimension.
    #[must_use]
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Variable {
        let data = self.data();
        let result = data.mean_dim(dim, keepdim);
        Variable::new(result, self.requires_grad())
    }

    /// Variance along a dimension, optionally keeping the dimension.
    #[must_use]
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Variable {
        let data = self.data();
        let result = data.var_dim(dim, keepdim);
        Variable::new(result, self.requires_grad())
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /// Creates a Variable from a tensor and requires_grad flag (for weight access).
    /// This is typically used internally by Parameter types.
    #[must_use]
    pub fn from_tensor_with_grad(data: Tensor<f32>, requires_grad: bool) -> Variable {
        Variable::new(data, requires_grad)
    }

    /// Clones the variable (alias for Clone trait).
    #[must_use]
    pub fn clone_var(&self) -> Variable {
        self.clone()
    }

    /// Adds another variable (alias for add_var for method chaining).
    #[must_use]
    pub fn add(&self, other: &Variable) -> Variable {
        self.add_var(other)
    }

    /// Subtracts another variable (alias for sub_var for method chaining).
    #[must_use]
    pub fn sub(&self, other: &Variable) -> Variable {
        self.sub_var(other)
    }

    /// Multiplies by another variable (alias for mul_var for method chaining).
    #[must_use]
    pub fn mul(&self, other: &Variable) -> Variable {
        self.mul_var(other)
    }

    /// Divides by another variable (alias for div_var for method chaining).
    #[must_use]
    pub fn div(&self, other: &Variable) -> Variable {
        self.div_var(other)
    }
}

// =============================================================================
// Operator Overloads
// =============================================================================

impl Add for &Variable {
    type Output = Variable;

    fn add(self, other: &Variable) -> Variable {
        self.add_var(other)
    }
}

impl Sub for &Variable {
    type Output = Variable;

    fn sub(self, other: &Variable) -> Variable {
        self.sub_var(other)
    }
}

impl Mul for &Variable {
    type Output = Variable;

    fn mul(self, other: &Variable) -> Variable {
        self.mul_var(other)
    }
}

impl Div for &Variable {
    type Output = Variable;

    fn div(self, other: &Variable) -> Variable {
        self.div_var(other)
    }
}

impl Neg for &Variable {
    type Output = Variable;

    fn neg(self) -> Variable {
        self.neg_var()
    }
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("shape", &self.shape())
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field(
                "grad_fn",
                &self.grad_fn.as_ref().map(super::grad_fn::GradFn::name),
            )
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::zeros;

    #[test]
    fn test_variable_creation() {
        let t = zeros::<f32>(&[2, 3]);
        let v = Variable::new(t, true);
        assert!(v.requires_grad());
        assert!(v.is_leaf());
        assert_eq!(v.shape(), vec![2, 3]);
    }

    #[test]
    fn test_variable_no_grad() {
        let t = zeros::<f32>(&[2, 3]);
        let v = Variable::from_tensor(t);
        assert!(!v.requires_grad());
    }

    #[test]
    fn test_variable_add() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap(), true);
        let c = &a + &b;
        assert_eq!(c.data().to_vec(), vec![5.0, 7.0, 9.0]);
        assert!(c.requires_grad());
        assert!(!c.is_leaf());
    }

    #[test]
    fn test_variable_detach() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let b = a.detach();
        assert!(!b.requires_grad());
        assert!(b.is_leaf());
    }

    #[test]
    fn test_mse_loss() {
        let pred = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let target = Variable::from_tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());
        let loss = pred.mse_loss(&target);
        assert_eq!(loss.numel(), 1);
        assert!((loss.data().to_vec()[0] - 0.0).abs() < 1e-6);
    }
}
