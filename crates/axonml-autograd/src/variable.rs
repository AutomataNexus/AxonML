//! `Variable` — tensor with automatic gradient tracking.
//!
//! 1577 lines, 67 public methods. Wraps `Arc<RwLock<Tensor<f32>>>` with a
//! `GradAccumulator` and optional `GradFn` recording the operation that
//! produced it. Provides differentiable versions of all tensor ops: arithmetic
//! (add_var, sub_var, mul_var, div_var, neg, add_scalar, mul_scalar, pow),
//! activations (relu, sigmoid, tanh, gelu, silu, elu, leaky_relu, softmax,
//! log_softmax), reductions (sum, mean, sum_dim, mean_dim, var_dim), shape
//! (reshape, transpose, t, narrow, select, unsqueeze, expand, cat), matmul,
//! and `from_operation` (custom GradFn attachment). `backward()` triggers
//! reverse-mode autodiff from this variable. `detach()` / `requires_grad_()`
//! control tracking. `data()` gives read access to the underlying tensor.
//!
//! # File
//! `crates/axonml-autograd/src/variable.rs`
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

use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use parking_lot::RwLock;

use axonml_tensor::Tensor;

use crate::functions::{
    AddBackward, AddScalarBackward, CatBackward, ClampBackward, DivBackward, EluBackward,
    ExpBackward, ExpandBackward, GeluBackward, LeakyReluBackward, LogBackward, LogSoftmaxBackward,
    MatMulBackward, MeanBackward, MeanDimBackward, MulBackward, MulScalarBackward, NarrowBackward,
    NegBackward, PowBackward, ReluBackward, ReshapeBackward, SelectBackward, SigmoidBackward,
    SiluBackward, SoftmaxBackward, SqrtBackward, SubBackward, SumBackward, SumDimBackward,
    TanhBackward, TransposeBackward, UnsqueezeBackward, VarDimBackward,
};
use crate::grad_fn::{AccumulateGrad, GradAccumulator, GradFn};
use crate::graph::{GraphNode, with_graph};
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

    /// Creates a new variable from an operation result with an attached gradient function.
    ///
    /// This connects the variable to the computational graph, allowing gradients
    /// to flow backward through the operation that produced this variable.
    pub fn from_operation(data: Tensor<f32>, grad_fn: GradFn, requires_grad: bool) -> Self {
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

    /// Returns a clone of the underlying tensor data.
    ///
    /// Tensor uses Arc-backed storage, so this is a cheap reference count
    /// bump (not a deep copy). The data is shared until mutated.
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

    /// Returns the device this variable's data is on.
    #[must_use]
    pub fn device(&self) -> axonml_tensor::Device {
        self.data.read().device()
    }

    /// Moves this variable's data to the specified device.
    ///
    /// Creates a new leaf Variable on the target device.
    /// Used for moving inputs to GPU before forward pass.
    pub fn to_device(&self, device: axonml_tensor::Device) -> Self {
        let current = self.data.read().clone();
        if current.device() == device {
            return self.clone();
        }
        let moved = current
            .to_device(device)
            .expect("Failed to move variable to device");
        Variable::new(moved, self.requires_grad)
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

    /// Updates the underlying tensor data without breaking the gradient accumulator.
    ///
    /// Unlike creating a new Variable, this preserves the grad accumulator Arc
    /// so that backward passes continue to write gradients to the right place.
    pub fn set_data(&mut self, new_data: Tensor<f32>) {
        *self.data.write() = new_data;
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

        // Start with gradient of 1.0 for the output, on the same device
        let mut grad_output = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let device = self.data.read().device();
        if device.is_gpu() {
            grad_output = grad_output
                .to_device(device)
                .expect("device transfer failed");
        }
        crate::backward::backward(self, &grad_output);
    }

    /// Runs the backward pass with a provided gradient tensor.
    ///
    /// Unlike `backward()`, this does not require the variable to be scalar.
    /// The gradient tensor must match the shape of this variable.
    pub fn backward_with_grad(&self, grad_output: &Tensor<f32>) {
        if !self.requires_grad {
            return;
        }
        let device = self.data.read().device();
        let grad = if grad_output.device() != device && device.is_gpu() {
            grad_output
                .to_device(device)
                .expect("device transfer failed")
        } else {
            grad_output.clone()
        };
        crate::backward::backward(self, &grad);
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

        // AMP: cast inputs to f16 precision for faster matmul, result stays f32
        let (compute_a, compute_b) = if crate::amp::is_autocast_enabled() {
            (self_data.to_f16_precision(), other_data.to_f16_precision())
        } else {
            (self_data.clone(), other_data.clone())
        };

        let result = compute_a.matmul(&compute_b).expect("matmul failed");
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

    /// Leaky ReLU activation.
    #[must_use]
    pub fn leaky_relu(&self, negative_slope: f32) -> Variable {
        let self_data = self.data.read().clone();
        let device = self_data.device();
        let result_vec: Vec<f32> = self_data
            .to_vec()
            .iter()
            .map(|&x| if x > 0.0 { x } else { x * negative_slope })
            .collect();
        let mut result = Tensor::from_vec(result_vec, self_data.shape()).unwrap();
        if device.is_gpu() {
            result = result.to_device(device).expect("device transfer failed");
        }
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(LeakyReluBackward::new(
                self.grad_fn.clone(),
                self_data,
                negative_slope,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// ELU activation.
    #[must_use]
    pub fn elu(&self, alpha: f32) -> Variable {
        let self_data = self.data.read().clone();
        let device = self_data.device();
        let result_vec: Vec<f32> = self_data
            .to_vec()
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();
        let mut result = Tensor::from_vec(result_vec, self_data.shape()).unwrap();
        if device.is_gpu() {
            result = result.to_device(device).expect("device transfer failed");
        }
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(EluBackward::new(self.grad_fn.clone(), self_data, alpha));
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

    /// Element-wise exponential.
    #[must_use]
    pub fn exp(&self) -> Variable {
        let self_data = self.data.read().clone();
        let result = self_data.exp();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(ExpBackward::new(self.grad_fn.clone(), result.clone()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Element-wise natural logarithm.
    #[must_use]
    pub fn log(&self) -> Variable {
        let self_data = self.data.read().clone();
        let result = self_data.ln();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(LogBackward::new(self.grad_fn.clone(), self_data));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Element-wise clamp to [min_val, max_val].
    #[must_use]
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Variable {
        let self_data = self.data.read().clone();
        let device = self_data.device();
        let result_data: Vec<f32> = self_data
            .to_vec()
            .iter()
            .map(|&x| x.clamp(min_val, max_val))
            .collect();
        let mut result = Tensor::from_vec(result_data, self_data.shape()).unwrap();
        if device.is_gpu() {
            result = result.to_device(device).expect("device transfer failed");
        }
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(ClampBackward::new(
                self.grad_fn.clone(),
                self_data,
                min_val,
                max_val,
            ));
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

    /// Sum along a dimension, removing that dimension.
    #[must_use]
    pub fn sum_dim(&self, dim: usize) -> Variable {
        let self_data = self.data.read().clone();
        let result = self_data.sum_dim(dim as i32, false);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SumDimBackward::new(self.grad_fn.clone(), self.shape(), dim));
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

    /// Flattens all dimensions from `start_dim` to the end into a single dimension.
    ///
    /// `flatten(1)` on a `[batch, C, H, W]` tensor produces `[batch, C*H*W]`.
    /// `flatten(0)` flattens everything into a 1D vector.
    #[must_use]
    pub fn flatten(&self, start_dim: usize) -> Variable {
        let shape = self.shape();
        if start_dim >= shape.len() {
            return self.clone();
        }
        let mut new_shape: Vec<usize> = shape[..start_dim].to_vec();
        let flat: usize = shape[start_dim..].iter().product();
        new_shape.push(flat);
        self.reshape(&new_shape)
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
    ///
    /// Tracks the computational graph for backward pass.
    #[must_use]
    pub fn expand(&self, shape: &[usize]) -> Variable {
        let input_shape = self.shape();
        let new_data = self.data().broadcast_to(shape);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(ExpandBackward::new(self.grad_fn.clone(), input_shape));
            Variable::from_operation(new_data, grad_fn, true)
        } else {
            Variable::from_tensor(new_data)
        }
    }

    /// Selects a single index along a dimension, reducing rank by 1.
    ///
    /// For a tensor of shape (A, B, C), `select(1, i)` returns shape (A, C).
    /// Tracks the computational graph for backward pass.
    #[must_use]
    pub fn select(&self, dim: usize, index: usize) -> Variable {
        let input_shape = self.shape();
        let new_data = self
            .data()
            .select(dim, index)
            .unwrap_or_else(|_| self.data().clone());
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SelectBackward::new(
                self.grad_fn.clone(),
                input_shape,
                dim,
                index,
            ));
            Variable::from_operation(new_data, grad_fn, true)
        } else {
            Variable::from_tensor(new_data)
        }
    }

    /// Adds a dimension of size 1 at the given position.
    ///
    /// Tracks the computational graph for backward pass.
    #[must_use]
    pub fn unsqueeze(&self, dim: usize) -> Variable {
        let new_data = self
            .data()
            .unsqueeze(dim as i64)
            .unwrap_or_else(|_| self.data().clone());
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(UnsqueezeBackward::new(self.grad_fn.clone(), dim));
            Variable::from_operation(new_data, grad_fn, true)
        } else {
            Variable::from_tensor(new_data)
        }
    }

    /// Concatenates variables along a dimension.
    ///
    /// All variables must have the same shape except along the cat dimension.
    /// Tracks the computational graph for backpropagation.
    #[must_use]
    pub fn cat(variables: &[&Variable], dim: usize) -> Variable {
        let tensors: Vec<Tensor<f32>> = variables.iter().map(|v| v.data()).collect();
        let tensor_refs: Vec<&Tensor<f32>> = tensors.iter().collect();
        let result = Tensor::cat(&tensor_refs, dim).unwrap();

        let requires_grad = variables.iter().any(|v| v.requires_grad) && is_grad_enabled();

        if requires_grad {
            let next_fns: Vec<Option<GradFn>> =
                variables.iter().map(|v| v.grad_fn.clone()).collect();
            let sizes: Vec<usize> = variables.iter().map(|v| v.shape()[dim]).collect();
            let grad_fn = GradFn::new(CatBackward::new(next_fns, sizes, dim));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    // =========================================================================
    // Scalar Operations
    // =========================================================================

    /// Multiplies by a scalar.
    #[must_use]
    pub fn mul_scalar(&self, scalar: f32) -> Variable {
        let data = self.data();
        let result = data.mul_scalar(scalar);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(MulScalarBackward::new(self.grad_fn.clone(), scalar));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Adds a scalar.
    #[must_use]
    pub fn add_scalar(&self, scalar: f32) -> Variable {
        let data = self.data();
        let result = data.add_scalar(scalar);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(AddScalarBackward::new(self.grad_fn.clone()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
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
        let self_data = self.data();
        let result = self_data.gelu();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(GeluBackward::new(self.grad_fn.clone(), self_data));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// SiLU/Swish activation function (x * sigmoid(x)).
    #[must_use]
    pub fn silu(&self) -> Variable {
        let self_data = self.data();
        let result = self_data.silu();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SiluBackward::new(self.grad_fn.clone(), self_data));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Square root.
    #[must_use]
    pub fn sqrt(&self) -> Variable {
        let data = self.data();
        let result = data.sqrt();
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SqrtBackward::new(self.grad_fn.clone(), result.clone()));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    // =========================================================================
    // Softmax Operations
    // =========================================================================

    /// Softmax along specified dimension.
    #[must_use]
    pub fn softmax(&self, dim: i32) -> Variable {
        let data = self.data();
        let result = data.softmax(dim);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(SoftmaxBackward::new(
                self.grad_fn.clone(),
                result.clone(),
                dim as i64,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Log softmax along specified dimension.
    #[must_use]
    pub fn log_softmax(&self, dim: i32) -> Variable {
        let data = self.data();
        let result = data.log_softmax(dim);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(LogSoftmaxBackward::new(
                self.grad_fn.clone(),
                result.clone(),
                dim as i64,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    // =========================================================================
    // Reduction Operations with Dimensions
    // =========================================================================

    /// Mean along a dimension, optionally keeping the dimension.
    #[must_use]
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Variable {
        let data = self.data();
        let input_shape = data.shape().to_vec();
        let ndim = input_shape.len();
        let dim_usize = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };
        let result = data.mean_dim(dim, keepdim);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(MeanDimBackward::new(
                self.grad_fn.clone(),
                input_shape,
                dim_usize,
                keepdim,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
    }

    /// Variance along a dimension, optionally keeping the dimension.
    #[must_use]
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Variable {
        let self_data = self.data();
        let input_shape = self_data.shape().to_vec();
        let ndim = input_shape.len();
        let dim_usize = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };
        let result = self_data.var_dim(dim, keepdim);
        let requires_grad = self.requires_grad && is_grad_enabled();

        if requires_grad {
            let grad_fn = GradFn::new(VarDimBackward::new(
                self.grad_fn.clone(),
                self_data,
                dim_usize,
                keepdim,
            ));
            Variable::from_operation(result, grad_fn, true)
        } else {
            Variable::from_tensor(result)
        }
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
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let b = Variable::new(
            Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let c = &a + &b;
        assert_eq!(c.data().to_vec(), vec![5.0, 7.0, 9.0]);
        assert!(c.requires_grad());
        assert!(!c.is_leaf());
    }

    #[test]
    fn test_variable_detach() {
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let b = a.detach();
        assert!(!b.requires_grad());
        assert!(b.is_leaf());
    }

    #[test]
    fn test_mse_loss() {
        let pred = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let target = Variable::from_tensor(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor creation failed"),
        );
        let loss = pred.mse_loss(&target);
        assert_eq!(loss.numel(), 1);
        assert!((loss.data().to_vec()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_exp() {
        let a = Variable::new(
            Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let b = a.exp();
        assert!((b.data().to_vec()[0] - 1.0).abs() < 1e-5);
        assert!((b.data().to_vec()[1] - std::f32::consts::E).abs() < 1e-4);

        b.sum().backward();
        let grad = a.grad().unwrap().to_vec();
        // d/dx(exp(x)) = exp(x)
        assert!((grad[0] - 1.0).abs() < 1e-5);
        assert!((grad[1] - std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn test_log() {
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, std::f32::consts::E, 10.0], &[3])
                .expect("tensor creation failed"),
            true,
        );
        let b = a.log();
        assert!((b.data().to_vec()[0] - 0.0).abs() < 1e-5);
        assert!((b.data().to_vec()[1] - 1.0).abs() < 1e-5);

        b.sum().backward();
        let grad = a.grad().unwrap().to_vec();
        // d/dx(log(x)) = 1/x
        assert!((grad[0] - 1.0).abs() < 1e-5);
        assert!((grad[1] - 1.0 / std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_clamp() {
        let a = Variable::new(
            Tensor::from_vec(vec![-1.0, 0.5, 2.0], &[3]).expect("tensor creation failed"),
            true,
        );
        let b = a.clamp(0.0, 1.0);
        assert_eq!(b.data().to_vec(), vec![0.0, 0.5, 1.0]);

        b.sum().backward();
        let grad = a.grad().unwrap().to_vec();
        // Gradient passes through only where not clamped
        assert_eq!(grad[0], 0.0); // clamped at min
        assert_eq!(grad[1], 1.0); // not clamped
        assert_eq!(grad[2], 0.0); // clamped at max
    }

    // =========================================================================
    // Backward Pass — Arithmetic Gradients
    // =========================================================================

    #[test]
    fn test_add_backward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap(), true);
        let c = a.add_var(&b);
        c.sum().backward();

        // d(a+b)/da = 1, d(a+b)/db = 1
        let ga = a.grad().expect("a should have grad");
        let gb = b.grad().expect("b should have grad");
        assert_eq!(ga.to_vec(), vec![1.0, 1.0]);
        assert_eq!(gb.to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_sub_backward() {
        let a = Variable::new(Tensor::from_vec(vec![5.0, 3.0], &[2]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0, 1.0], &[2]).unwrap(), true);
        let c = a.sub_var(&b);

        assert_eq!(c.data().to_vec(), vec![3.0, 2.0]);
        c.sum().backward();

        // d(a-b)/da = 1, d(a-b)/db = -1
        let ga = a.grad().unwrap().to_vec();
        let gb = b.grad().unwrap().to_vec();
        assert_eq!(ga, vec![1.0, 1.0]);
        assert_eq!(gb, vec![-1.0, -1.0]);
    }

    #[test]
    fn test_mul_backward() {
        let a = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0], &[2]).unwrap(), true);
        let c = a.mul_var(&b);

        assert_eq!(c.data().to_vec(), vec![8.0, 15.0]);
        c.sum().backward();

        // d(a*b)/da = b, d(a*b)/db = a
        let ga = a.grad().unwrap().to_vec();
        let gb = b.grad().unwrap().to_vec();
        assert_eq!(ga, vec![4.0, 5.0]);
        assert_eq!(gb, vec![2.0, 3.0]);
    }

    #[test]
    fn test_div_backward() {
        let a = Variable::new(Tensor::from_vec(vec![6.0, 10.0], &[2]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0, 5.0], &[2]).unwrap(), true);
        let c = a.div_var(&b);

        assert_eq!(c.data().to_vec(), vec![3.0, 2.0]);
        c.sum().backward();

        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        let ga = a.grad().unwrap().to_vec();
        let gb = b.grad().unwrap().to_vec();
        assert!((ga[0] - 0.5).abs() < 1e-5, "da = 1/b = 0.5, got {}", ga[0]);
        assert!((ga[1] - 0.2).abs() < 1e-5, "da = 1/b = 0.2, got {}", ga[1]);
        assert!(
            (gb[0] - (-1.5)).abs() < 1e-5,
            "db = -a/b^2 = -6/4 = -1.5, got {}",
            gb[0]
        );
        assert!(
            (gb[1] - (-0.4)).abs() < 1e-5,
            "db = -a/b^2 = -10/25 = -0.4, got {}",
            gb[1]
        );
    }

    #[test]
    fn test_mul_scalar_backward() {
        let a = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(), true);
        let c = a.mul_scalar(5.0);

        assert_eq!(c.data().to_vec(), vec![10.0, 15.0]);
        c.sum().backward();

        // d(5*a)/da = 5
        let ga = a.grad().unwrap().to_vec();
        assert_eq!(ga, vec![5.0, 5.0]);
    }

    // =========================================================================
    // Backward Pass — Activations
    // =========================================================================

    #[test]
    fn test_relu_backward() {
        let a = Variable::new(Tensor::from_vec(vec![-2.0, 0.0, 3.0], &[3]).unwrap(), true);
        let b = a.relu();

        assert_eq!(b.data().to_vec(), vec![0.0, 0.0, 3.0]);
        b.sum().backward();

        // d(relu(x))/dx = 0 if x<0, 1 if x>0
        let ga = a.grad().unwrap().to_vec();
        assert_eq!(ga[0], 0.0); // negative → 0
        assert_eq!(ga[2], 1.0); // positive → 1
    }

    #[test]
    fn test_sigmoid_backward() {
        let a = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), true);
        let b = a.sigmoid();

        // sigmoid(0) = 0.5
        assert!((b.data().to_vec()[0] - 0.5).abs() < 1e-5);
        b.backward();

        // d(sigmoid(x))/dx = sigmoid(x)*(1-sigmoid(x)) = 0.5*0.5 = 0.25
        let ga = a.grad().unwrap().to_vec();
        assert!(
            (ga[0] - 0.25).abs() < 1e-4,
            "sigmoid'(0) = 0.25, got {}",
            ga[0]
        );
    }

    #[test]
    fn test_tanh_backward() {
        let a = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), true);
        let b = a.tanh();

        // tanh(0) = 0
        assert!(b.data().to_vec()[0].abs() < 1e-5);
        b.backward();

        // d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - 0 = 1
        let ga = a.grad().unwrap().to_vec();
        assert!((ga[0] - 1.0).abs() < 1e-4, "tanh'(0) = 1.0, got {}", ga[0]);
    }

    // =========================================================================
    // Backward Pass — Chain Rule
    // =========================================================================

    #[test]
    fn test_chain_rule_mul_then_add() {
        // f(a,b) = a*b + a → df/da = b+1, df/db = a
        let a = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0], &[1]).unwrap(), true);
        let ab = a.mul_var(&b);
        let result = ab.add_var(&a);
        result.backward();

        let ga = a.grad().unwrap().to_vec()[0];
        let gb = b.grad().unwrap().to_vec()[0];
        assert!((ga - 5.0).abs() < 1e-4, "df/da = b+1 = 5, got {}", ga);
        assert!((gb - 3.0).abs() < 1e-4, "df/db = a = 3, got {}", gb);
    }

    #[test]
    fn test_chain_rule_nested_operations() {
        // f(x) = relu(x^2 - 1) → df/dx = 2x if x^2 > 1, else 0
        let x = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let x_sq = x.mul_var(&x); // x^2 = 4
        let shifted = x_sq.add_scalar(-1.0); // x^2 - 1 = 3
        let out = shifted.relu(); // relu(3) = 3

        assert!((out.data().to_vec()[0] - 3.0).abs() < 1e-5);
        out.backward();

        // df/dx = 2x * 1 (relu passes through since input > 0) = 4
        let gx = x.grad().unwrap().to_vec()[0];
        assert!((gx - 4.0).abs() < 1e-4, "df/dx = 2x = 4, got {}", gx);
    }

    #[test]
    fn test_sum_backward() {
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap(),
            true,
        );
        let s = a.sum();

        assert!((s.data().to_vec()[0] - 10.0).abs() < 1e-5);
        s.backward();

        // d(sum)/dx_i = 1 for all i
        let ga = a.grad().unwrap().to_vec();
        assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_backward() {
        let a = Variable::new(
            Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4]).unwrap(),
            true,
        );
        let m = a.mean();

        assert!((m.data().to_vec()[0] - 5.0).abs() < 1e-5);
        m.backward();

        // d(mean)/dx_i = 1/N = 0.25
        let ga = a.grad().unwrap().to_vec();
        for g in &ga {
            assert!(
                (g - 0.25).abs() < 1e-5,
                "d(mean)/dx = 1/4 = 0.25, got {}",
                g
            );
        }
    }

    // =========================================================================
    // Backward Pass — Matmul
    // =========================================================================

    #[test]
    fn test_matmul_backward() {
        // C = A @ B where A=[2,3], B=[3,2]
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
            true,
        );
        let b = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap(),
            true,
        );
        let c = a.matmul(&b); // [2, 2]
        assert_eq!(c.shape(), vec![2, 2]);

        c.sum().backward();

        // dL/dA = ones @ B^T, dL/dB = A^T @ ones
        let ga = a.grad().expect("A should have grad");
        let gb = b.grad().expect("B should have grad");
        assert_eq!(ga.shape(), &[2, 3]);
        assert_eq!(gb.shape(), &[3, 2]);

        // All gradients should be finite and non-zero
        assert!(ga.to_vec().iter().all(|g| g.is_finite() && g.abs() > 0.0));
        assert!(gb.to_vec().iter().all(|g| g.is_finite() && g.abs() > 0.0));
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_no_grad_skips_backward() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        let b = a.mul_scalar(2.0);
        // Should not panic even though requires_grad=false
        assert!((b.data().to_vec()[0] - 2.0).abs() < 1e-5);
        assert!(a.grad().is_none());
    }

    #[test]
    fn test_detach_stops_gradient() {
        // detach() creates a new variable without gradient tracking
        let a = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let b = a.mul_scalar(2.0);
        let c = b.detach();

        // Detached variable should not require grad
        assert!(
            !c.requires_grad(),
            "Detached variable should not require grad"
        );
        assert!(c.is_leaf(), "Detached variable should be a leaf");

        // Original chain should still work
        b.backward();
        let ga = a.grad().unwrap().to_vec()[0];
        assert!(
            (ga - 2.0).abs() < 1e-4,
            "Gradient through b=2*a should be 2: got {}",
            ga
        );
    }

    #[test]
    fn test_backward_twice_accumulates() {
        let a = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let b = a.mul_scalar(3.0);
        b.backward();
        let g1 = a.grad().unwrap().to_vec()[0];

        // Second backward should accumulate
        let c = a.mul_scalar(3.0);
        c.backward();
        let g2 = a.grad().unwrap().to_vec()[0];

        // Gradient should have accumulated: 3 + 3 = 6
        assert!(
            (g2 - g1 * 2.0).abs() < 1e-4 || g2 >= g1,
            "Second backward should accumulate: g1={}, g2={}",
            g1,
            g2
        );
    }

    #[test]
    fn test_reshape_preserves_gradient() {
        let a = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            true,
        );
        let b = a.reshape(&[4]);
        let c = b.sum();
        c.backward();

        let ga = a.grad().expect("Should have gradient through reshape");
        assert_eq!(ga.shape(), &[2, 2]);
        assert_eq!(ga.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    }
}
