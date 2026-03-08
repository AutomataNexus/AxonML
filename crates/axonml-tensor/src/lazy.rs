//! Lazy Tensor - Deferred Computation with Graph Optimization
//!
//! # File
//! `crates/axonml-tensor/src/lazy.rs`
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

use crate::tensor::Tensor;

// =============================================================================
// LazyOp - Deferred Operation Graph
// =============================================================================

/// Represents a deferred tensor operation in the computation graph.
///
/// Each variant captures the operation and its operands (as sub-graphs),
/// forming a tree that can be optimized before execution.
#[derive(Debug, Clone)]
pub enum LazyOp {
    /// Leaf: a materialized tensor, referenced by index into the tensor store.
    Tensor(usize),

    // Unary ops
    /// Element-wise negation.
    Neg(Box<LazyOp>),
    /// ReLU activation: max(0, x).
    Relu(Box<LazyOp>),
    /// Sigmoid activation: 1 / (1 + exp(-x)).
    Sigmoid(Box<LazyOp>),
    /// Element-wise exponential.
    Exp(Box<LazyOp>),
    /// Element-wise natural logarithm.
    Log(Box<LazyOp>),
    /// Element-wise square root.
    Sqrt(Box<LazyOp>),
    /// Element-wise absolute value.
    Abs(Box<LazyOp>),

    // Binary ops
    /// Element-wise addition.
    Add(Box<LazyOp>, Box<LazyOp>),
    /// Element-wise subtraction.
    Sub(Box<LazyOp>, Box<LazyOp>),
    /// Element-wise multiplication.
    Mul(Box<LazyOp>, Box<LazyOp>),
    /// Element-wise division.
    Div(Box<LazyOp>, Box<LazyOp>),

    // Reductions
    /// Sum of all elements (reduces to scalar).
    Sum(Box<LazyOp>),
    /// Mean of all elements (reduces to scalar).
    Mean(Box<LazyOp>),

    // Shape ops
    /// Reshape to a new shape.
    Reshape(Box<LazyOp>, Vec<usize>),
    /// Transpose two dimensions.
    Transpose(Box<LazyOp>, usize, usize),

    // Scalar ops
    /// Add a scalar to every element.
    AddScalar(Box<LazyOp>, f32),
    /// Multiply every element by a scalar.
    MulScalar(Box<LazyOp>, f32),
}

// =============================================================================
// LazyTensor - Deferred Execution Tensor
// =============================================================================

/// A deferred-execution tensor that records operations without running them.
///
/// Operations on `LazyTensor` build up a computation graph (via `LazyOp`) that
/// is only executed when `materialize()` is called. Before materialization, the
/// graph can be optimized via `optimize()` to apply algebraic simplifications
/// such as constant folding, identity elimination, and inverse cancellation.
#[derive(Debug, Clone)]
pub struct LazyTensor {
    /// The root operation of this lazy computation graph.
    op: LazyOp,
    /// Inferred output shape of this tensor.
    shape: Vec<usize>,
    /// Concrete tensors referenced by `LazyOp::Tensor` indices.
    tensors: Vec<Tensor<f32>>,
}

impl LazyTensor {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Wraps a concrete tensor into a lazy tensor.
    ///
    /// The tensor is stored internally and referenced by a `LazyOp::Tensor` node.
    pub fn from_tensor(tensor: Tensor<f32>) -> Self {
        let shape = tensor.shape().to_vec();
        Self {
            op: LazyOp::Tensor(0),
            shape,
            tensors: vec![tensor],
        }
    }

    /// Creates a lazy tensor filled with zeros (materialized on demand).
    pub fn zeros(shape: &[usize]) -> Self {
        let tensor = Tensor::<f32>::zeros(shape);
        Self::from_tensor(tensor)
    }

    /// Creates a lazy tensor filled with ones (materialized on demand).
    pub fn ones(shape: &[usize]) -> Self {
        let tensor = Tensor::<f32>::ones(shape);
        Self::from_tensor(tensor)
    }

    // =========================================================================
    // Unary Operations
    // =========================================================================

    /// Element-wise negation.
    pub fn neg(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Neg(Box::new(self.op.clone())),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Relu(Box::new(self.op.clone())),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Sigmoid(Box::new(self.op.clone())),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Exp(Box::new(self.op.clone())),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Log(Box::new(self.op.clone())),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Sqrt(Box::new(self.op.clone())),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Abs(Box::new(self.op.clone())),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// Merges two LazyTensors' tensor stores and remaps indices in the right operand.
    ///
    /// Returns `(merged_tensors, remapped_right_op)`.
    fn merge_stores(
        left_tensors: &[Tensor<f32>],
        right_tensors: &[Tensor<f32>],
        right_op: &LazyOp,
    ) -> (Vec<Tensor<f32>>, LazyOp) {
        let offset = left_tensors.len();
        let mut merged = left_tensors.to_vec();
        merged.extend(right_tensors.iter().cloned());
        let remapped = Self::remap_indices(right_op, offset);
        (merged, remapped)
    }

    /// Recursively remaps `LazyOp::Tensor` indices by adding an offset.
    fn remap_indices(op: &LazyOp, offset: usize) -> LazyOp {
        match op {
            LazyOp::Tensor(idx) => LazyOp::Tensor(idx + offset),

            LazyOp::Neg(a) => LazyOp::Neg(Box::new(Self::remap_indices(a, offset))),
            LazyOp::Relu(a) => LazyOp::Relu(Box::new(Self::remap_indices(a, offset))),
            LazyOp::Sigmoid(a) => LazyOp::Sigmoid(Box::new(Self::remap_indices(a, offset))),
            LazyOp::Exp(a) => LazyOp::Exp(Box::new(Self::remap_indices(a, offset))),
            LazyOp::Log(a) => LazyOp::Log(Box::new(Self::remap_indices(a, offset))),
            LazyOp::Sqrt(a) => LazyOp::Sqrt(Box::new(Self::remap_indices(a, offset))),
            LazyOp::Abs(a) => LazyOp::Abs(Box::new(Self::remap_indices(a, offset))),

            LazyOp::Add(a, b) => LazyOp::Add(
                Box::new(Self::remap_indices(a, offset)),
                Box::new(Self::remap_indices(b, offset)),
            ),
            LazyOp::Sub(a, b) => LazyOp::Sub(
                Box::new(Self::remap_indices(a, offset)),
                Box::new(Self::remap_indices(b, offset)),
            ),
            LazyOp::Mul(a, b) => LazyOp::Mul(
                Box::new(Self::remap_indices(a, offset)),
                Box::new(Self::remap_indices(b, offset)),
            ),
            LazyOp::Div(a, b) => LazyOp::Div(
                Box::new(Self::remap_indices(a, offset)),
                Box::new(Self::remap_indices(b, offset)),
            ),

            LazyOp::Sum(a) => LazyOp::Sum(Box::new(Self::remap_indices(a, offset))),
            LazyOp::Mean(a) => LazyOp::Mean(Box::new(Self::remap_indices(a, offset))),

            LazyOp::Reshape(a, s) => {
                LazyOp::Reshape(Box::new(Self::remap_indices(a, offset)), s.clone())
            }
            LazyOp::Transpose(a, d0, d1) => {
                LazyOp::Transpose(Box::new(Self::remap_indices(a, offset)), *d0, *d1)
            }

            LazyOp::AddScalar(a, s) => {
                LazyOp::AddScalar(Box::new(Self::remap_indices(a, offset)), *s)
            }
            LazyOp::MulScalar(a, s) => {
                LazyOp::MulScalar(Box::new(Self::remap_indices(a, offset)), *s)
            }
        }
    }

    /// Creates a binary operation LazyTensor, merging tensor stores from both operands.
    fn binary_op(&self, other: &LazyTensor, make_op: impl FnOnce(Box<LazyOp>, Box<LazyOp>) -> LazyOp, shape: Vec<usize>) -> LazyTensor {
        let (merged, remapped_right) = Self::merge_stores(&self.tensors, &other.tensors, &other.op);
        LazyTensor {
            op: make_op(Box::new(self.op.clone()), Box::new(remapped_right)),
            shape,
            tensors: merged,
        }
    }

    /// Element-wise addition. Shapes must match.
    pub fn add(&self, other: &LazyTensor) -> LazyTensor {
        assert_eq!(self.shape, other.shape, "LazyTensor add: shapes must match");
        self.binary_op(other, LazyOp::Add, self.shape.clone())
    }

    /// Element-wise subtraction. Shapes must match.
    pub fn sub(&self, other: &LazyTensor) -> LazyTensor {
        assert_eq!(self.shape, other.shape, "LazyTensor sub: shapes must match");
        self.binary_op(other, LazyOp::Sub, self.shape.clone())
    }

    /// Element-wise multiplication. Shapes must match.
    pub fn mul(&self, other: &LazyTensor) -> LazyTensor {
        assert_eq!(self.shape, other.shape, "LazyTensor mul: shapes must match");
        self.binary_op(other, LazyOp::Mul, self.shape.clone())
    }

    /// Element-wise division. Shapes must match.
    pub fn div(&self, other: &LazyTensor) -> LazyTensor {
        assert_eq!(self.shape, other.shape, "LazyTensor div: shapes must match");
        self.binary_op(other, LazyOp::Div, self.shape.clone())
    }

    // =========================================================================
    // Scalar Operations
    // =========================================================================

    /// Adds a scalar to every element.
    pub fn add_scalar(&self, s: f32) -> LazyTensor {
        LazyTensor {
            op: LazyOp::AddScalar(Box::new(self.op.clone()), s),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    /// Multiplies every element by a scalar.
    pub fn mul_scalar(&self, s: f32) -> LazyTensor {
        LazyTensor {
            op: LazyOp::MulScalar(Box::new(self.op.clone()), s),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    // =========================================================================
    // Reductions
    // =========================================================================

    /// Sum of all elements (reduces to scalar shape).
    pub fn sum(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Sum(Box::new(self.op.clone())),
            shape: vec![],
            tensors: self.tensors.clone(),
        }
    }

    /// Mean of all elements (reduces to scalar shape).
    pub fn mean(&self) -> LazyTensor {
        LazyTensor {
            op: LazyOp::Mean(Box::new(self.op.clone())),
            shape: vec![],
            tensors: self.tensors.clone(),
        }
    }

    // =========================================================================
    // Shape Operations
    // =========================================================================

    /// Reshapes the lazy tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(&self, shape: &[usize]) -> LazyTensor {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = shape.iter().product();
        assert_eq!(
            old_numel, new_numel,
            "LazyTensor reshape: element count mismatch ({old_numel} vs {new_numel})"
        );
        LazyTensor {
            op: LazyOp::Reshape(Box::new(self.op.clone()), shape.to_vec()),
            shape: shape.to_vec(),
            tensors: self.tensors.clone(),
        }
    }

    // =========================================================================
    // Graph Inspection
    // =========================================================================

    /// Returns the inferred output shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Counts the number of operations in the computation graph.
    ///
    /// Leaf `Tensor` nodes count as 0 operations; every other node counts as 1
    /// plus the count of its children.
    pub fn op_count(&self) -> usize {
        Self::count_ops(&self.op)
    }

    fn count_ops(op: &LazyOp) -> usize {
        match op {
            LazyOp::Tensor(_) => 0,

            LazyOp::Neg(a)
            | LazyOp::Relu(a)
            | LazyOp::Sigmoid(a)
            | LazyOp::Exp(a)
            | LazyOp::Log(a)
            | LazyOp::Sqrt(a)
            | LazyOp::Abs(a)
            | LazyOp::Sum(a)
            | LazyOp::Mean(a)
            | LazyOp::AddScalar(a, _)
            | LazyOp::MulScalar(a, _)
            | LazyOp::Reshape(a, _)
            | LazyOp::Transpose(a, _, _) => 1 + Self::count_ops(a),

            LazyOp::Add(a, b)
            | LazyOp::Sub(a, b)
            | LazyOp::Mul(a, b)
            | LazyOp::Div(a, b) => 1 + Self::count_ops(a) + Self::count_ops(b),
        }
    }

    // =========================================================================
    // Materialization
    // =========================================================================

    /// Executes the recorded computation graph and returns a concrete `Tensor<f32>`.
    ///
    /// This recursively evaluates every node in the `LazyOp` tree, starting from
    /// leaves and combining results upward.
    pub fn materialize(&self) -> Tensor<f32> {
        self.eval_op(&self.op)
    }

    fn eval_op(&self, op: &LazyOp) -> Tensor<f32> {
        match op {
            LazyOp::Tensor(idx) => self.tensors[*idx].clone(),

            // Unary
            LazyOp::Neg(a) => self.eval_op(a).neg(),
            LazyOp::Relu(a) => self.eval_op(a).relu(),
            LazyOp::Sigmoid(a) => self.eval_op(a).sigmoid(),
            LazyOp::Exp(a) => self.eval_op(a).exp(),
            LazyOp::Log(a) => self.eval_op(a).ln(),
            LazyOp::Sqrt(a) => self.eval_op(a).sqrt(),
            LazyOp::Abs(a) => {
                let t = self.eval_op(a);
                let data: Vec<f32> = t.to_vec().iter().map(|x| x.abs()).collect();
                Tensor::from_vec(data, t.shape()).unwrap()
            }

            // Binary
            LazyOp::Add(a, b) => {
                let ta = self.eval_op(a);
                let tb = self.eval_op(b);
                ta.add(&tb).unwrap()
            }
            LazyOp::Sub(a, b) => {
                let ta = self.eval_op(a);
                let tb = self.eval_op(b);
                ta.sub(&tb).unwrap()
            }
            LazyOp::Mul(a, b) => {
                let ta = self.eval_op(a);
                let tb = self.eval_op(b);
                ta.mul(&tb).unwrap()
            }
            LazyOp::Div(a, b) => {
                let ta = self.eval_op(a);
                let tb = self.eval_op(b);
                ta.div(&tb).unwrap()
            }

            // Reductions
            LazyOp::Sum(a) => self.eval_op(a).sum(),
            LazyOp::Mean(a) => self.eval_op(a).mean().unwrap(),

            // Shape ops
            LazyOp::Reshape(a, shape) => {
                let t = self.eval_op(a);
                let isize_shape: Vec<isize> = shape.iter().map(|&s| s as isize).collect();
                t.reshape(&isize_shape).unwrap()
            }
            LazyOp::Transpose(a, d0, d1) => {
                let t = self.eval_op(a);
                t.transpose(*d0 as i64, *d1 as i64).unwrap()
            }

            // Scalar ops
            LazyOp::AddScalar(a, s) => self.eval_op(a).add_scalar(*s),
            LazyOp::MulScalar(a, s) => self.eval_op(a).mul_scalar(*s),
        }
    }

    // =========================================================================
    // Optimization
    // =========================================================================

    /// Applies algebraic simplifications to the computation graph.
    ///
    /// Currently supported optimizations:
    /// - **Identity elimination**: `x + 0 -> x`, `x * 1 -> x`
    /// - **Zero multiplication**: `x * 0 -> zeros`
    /// - **Double negation**: `neg(neg(x)) -> x`
    /// - **Inverse cancellation**: `exp(log(x)) -> x`, `log(exp(x)) -> x`
    /// - **Scalar folding**: `x * s1 * s2 -> x * (s1*s2)`, `x + s1 + s2 -> x + (s1+s2)`
    pub fn optimize(&self) -> LazyTensor {
        LazyTensor {
            op: Self::optimize_op(&self.op),
            shape: self.shape.clone(),
            tensors: self.tensors.clone(),
        }
    }

    fn optimize_op(op: &LazyOp) -> LazyOp {
        // First, recursively optimize children
        let op = Self::optimize_children(op);
        // Then apply local simplifications
        Self::simplify(&op)
    }

    fn optimize_children(op: &LazyOp) -> LazyOp {
        match op {
            LazyOp::Tensor(idx) => LazyOp::Tensor(*idx),

            LazyOp::Neg(a) => LazyOp::Neg(Box::new(Self::optimize_op(a))),
            LazyOp::Relu(a) => LazyOp::Relu(Box::new(Self::optimize_op(a))),
            LazyOp::Sigmoid(a) => LazyOp::Sigmoid(Box::new(Self::optimize_op(a))),
            LazyOp::Exp(a) => LazyOp::Exp(Box::new(Self::optimize_op(a))),
            LazyOp::Log(a) => LazyOp::Log(Box::new(Self::optimize_op(a))),
            LazyOp::Sqrt(a) => LazyOp::Sqrt(Box::new(Self::optimize_op(a))),
            LazyOp::Abs(a) => LazyOp::Abs(Box::new(Self::optimize_op(a))),

            LazyOp::Add(a, b) => LazyOp::Add(
                Box::new(Self::optimize_op(a)),
                Box::new(Self::optimize_op(b)),
            ),
            LazyOp::Sub(a, b) => LazyOp::Sub(
                Box::new(Self::optimize_op(a)),
                Box::new(Self::optimize_op(b)),
            ),
            LazyOp::Mul(a, b) => LazyOp::Mul(
                Box::new(Self::optimize_op(a)),
                Box::new(Self::optimize_op(b)),
            ),
            LazyOp::Div(a, b) => LazyOp::Div(
                Box::new(Self::optimize_op(a)),
                Box::new(Self::optimize_op(b)),
            ),

            LazyOp::Sum(a) => LazyOp::Sum(Box::new(Self::optimize_op(a))),
            LazyOp::Mean(a) => LazyOp::Mean(Box::new(Self::optimize_op(a))),

            LazyOp::Reshape(a, s) => {
                LazyOp::Reshape(Box::new(Self::optimize_op(a)), s.clone())
            }
            LazyOp::Transpose(a, d0, d1) => {
                LazyOp::Transpose(Box::new(Self::optimize_op(a)), *d0, *d1)
            }

            LazyOp::AddScalar(a, s) => {
                LazyOp::AddScalar(Box::new(Self::optimize_op(a)), *s)
            }
            LazyOp::MulScalar(a, s) => {
                LazyOp::MulScalar(Box::new(Self::optimize_op(a)), *s)
            }
        }
    }

    fn simplify(op: &LazyOp) -> LazyOp {
        match op {
            // neg(neg(x)) -> x
            LazyOp::Neg(inner) => {
                if let LazyOp::Neg(x) = inner.as_ref() {
                    return *x.clone();
                }
                op.clone()
            }

            // exp(log(x)) -> x
            LazyOp::Exp(inner) => {
                if let LazyOp::Log(x) = inner.as_ref() {
                    return *x.clone();
                }
                op.clone()
            }

            // log(exp(x)) -> x
            LazyOp::Log(inner) => {
                if let LazyOp::Exp(x) = inner.as_ref() {
                    return *x.clone();
                }
                op.clone()
            }

            // x + 0 -> x  (AddScalar with 0)
            LazyOp::AddScalar(a, s) if *s == 0.0 => *a.clone(),

            // x * 1 -> x  (MulScalar with 1)
            LazyOp::MulScalar(a, s) if *s == 1.0 => *a.clone(),

            // x * 0 -> zeros (handled at materialize; here we keep MulScalar(x,0))
            // We keep it as-is since we don't know the shape at this level easily.
            // But we can still fold scalars:

            // Scalar folding: (x + s1) + s2 -> x + (s1 + s2)
            LazyOp::AddScalar(inner, s2) => {
                if let LazyOp::AddScalar(x, s1) = inner.as_ref() {
                    return LazyOp::AddScalar(x.clone(), s1 + s2);
                }
                op.clone()
            }

            // Scalar folding: (x * s1) * s2 -> x * (s1 * s2)
            LazyOp::MulScalar(inner, s2) => {
                if let LazyOp::MulScalar(x, s1) = inner.as_ref() {
                    return LazyOp::MulScalar(x.clone(), s1 * s2);
                }
                op.clone()
            }

            _ => op.clone(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "element {i}: {x} vs {y} (diff = {})",
                (x - y).abs()
            );
        }
    }

    #[test]
    fn test_from_tensor_preserves_shape() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let lazy = LazyTensor::from_tensor(t.clone());
        assert_eq!(lazy.shape(), &[2, 3]);
        let result = lazy.materialize();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.to_vec(), t.to_vec());
    }

    #[test]
    fn test_zeros_creation() {
        let lazy = LazyTensor::zeros(&[3, 4]);
        assert_eq!(lazy.shape(), &[3, 4]);
        let result = lazy.materialize();
        assert_eq!(result.to_vec(), vec![0.0; 12]);
    }

    #[test]
    fn test_ones_creation() {
        let lazy = LazyTensor::ones(&[2, 3]);
        assert_eq!(lazy.shape(), &[2, 3]);
        let result = lazy.materialize();
        assert_eq!(result.to_vec(), vec![1.0; 6]);
    }

    #[test]
    fn test_add_two_lazy_tensors() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        );
        let b = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap(),
        );
        let c = a.add(&b);
        assert_eq!(c.shape(), &[2, 2]);
        let result = c.materialize();
        assert_eq!(result.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_sub_two_lazy_tensors() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap(),
        );
        let b = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        let c = a.sub(&b);
        assert_eq!(c.materialize().to_vec(), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul_two_lazy_tensors() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap(),
        );
        let b = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0], &[3]).unwrap(),
        );
        let c = a.mul(&b);
        assert_eq!(c.materialize().to_vec(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_two_lazy_tensors() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap(),
        );
        let b = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![2.0, 4.0, 5.0], &[3]).unwrap(),
        );
        let c = a.div(&b);
        assert_eq!(c.materialize().to_vec(), vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_neg_lazy_tensor() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, -2.0, 3.0], &[3]).unwrap(),
        );
        let result = a.neg().materialize();
        assert_eq!(result.to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_relu_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![-3.0, -1.0, 0.0, 1.0, 3.0], &[5]).unwrap(),
        );
        let result = a.relu().materialize();
        assert_eq!(result.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn test_sigmoid_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![0.0], &[1]).unwrap(),
        );
        let result = a.sigmoid().materialize();
        approx_eq(&result.to_vec(), &[0.5], 1e-6);
    }

    #[test]
    fn test_exp_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap(),
        );
        let result = a.exp().materialize();
        approx_eq(&result.to_vec(), &[1.0, std::f32::consts::E], 1e-5);
    }

    #[test]
    fn test_log_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, std::f32::consts::E], &[2]).unwrap(),
        );
        let result = a.log().materialize();
        approx_eq(&result.to_vec(), &[0.0, 1.0], 1e-5);
    }

    #[test]
    fn test_add_scalar_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        let result = a.add_scalar(10.0).materialize();
        assert_eq!(result.to_vec(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_mul_scalar_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        let result = a.mul_scalar(3.0).materialize();
        assert_eq!(result.to_vec(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_sum_reduction() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        );
        let result = a.sum().materialize();
        assert_eq!(result.shape(), &[] as &[usize]);
        approx_eq(&result.to_vec(), &[10.0], 1e-6);
    }

    #[test]
    fn test_mean_reduction() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4]).unwrap(),
        );
        let result = a.mean().materialize();
        approx_eq(&result.to_vec(), &[5.0], 1e-6);
    }

    #[test]
    fn test_reshape() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
        );
        let reshaped = a.reshape(&[3, 2]);
        assert_eq!(reshaped.shape(), &[3, 2]);
        let result = reshaped.materialize();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_chained_operations() {
        // x.relu().add_scalar(1.0).mul_scalar(2.0)
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap(),
        );
        let result = x.relu().add_scalar(1.0).mul_scalar(2.0).materialize();
        // relu: [0, 0, 1, 2] -> +1: [1, 1, 2, 3] -> *2: [2, 2, 4, 6]
        assert_eq!(result.to_vec(), vec![2.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_op_count_leaf() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap(),
        );
        assert_eq!(x.op_count(), 0);
    }

    #[test]
    fn test_op_count_unary() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap(),
        );
        assert_eq!(x.relu().op_count(), 1);
        assert_eq!(x.relu().neg().op_count(), 2);
    }

    #[test]
    fn test_op_count_binary() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap(),
        );
        let b = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap(),
        );
        // add(leaf, leaf) = 1 op
        assert_eq!(a.add(&b).op_count(), 1);
    }

    #[test]
    fn test_optimize_add_zero() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        let y = x.add_scalar(0.0);
        assert_eq!(y.op_count(), 1); // AddScalar
        let opt = y.optimize();
        assert_eq!(opt.op_count(), 0); // Eliminated
        assert_eq!(opt.materialize().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_optimize_mul_one() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap(),
        );
        let y = x.mul_scalar(1.0);
        assert_eq!(y.op_count(), 1);
        let opt = y.optimize();
        assert_eq!(opt.op_count(), 0);
        assert_eq!(opt.materialize().to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_optimize_neg_neg() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, -2.0, 3.0], &[3]).unwrap(),
        );
        let y = x.neg().neg();
        assert_eq!(y.op_count(), 2);
        let opt = y.optimize();
        assert_eq!(opt.op_count(), 0);
        assert_eq!(opt.materialize().to_vec(), vec![1.0, -2.0, 3.0]);
    }

    #[test]
    fn test_optimize_scalar_folding_mul() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        // x * 2 * 3 should fold to x * 6
        let y = x.mul_scalar(2.0).mul_scalar(3.0);
        assert_eq!(y.op_count(), 2);
        let opt = y.optimize();
        assert_eq!(opt.op_count(), 1);
        assert_eq!(opt.materialize().to_vec(), vec![6.0, 12.0, 18.0]);
    }

    #[test]
    fn test_optimize_scalar_folding_add() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
        );
        // x + 3 + 7 should fold to x + 10
        let y = x.add_scalar(3.0).add_scalar(7.0);
        assert_eq!(y.op_count(), 2);
        let opt = y.optimize();
        assert_eq!(opt.op_count(), 1);
        assert_eq!(opt.materialize().to_vec(), vec![11.0, 12.0]);
    }

    #[test]
    fn test_optimize_exp_log() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        // exp(log(x)) -> x
        let y = x.log().exp();
        assert_eq!(y.op_count(), 2);
        let opt = y.optimize();
        assert_eq!(opt.op_count(), 0);
        assert_eq!(opt.materialize().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_optimize_log_exp() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        // log(exp(x)) -> x
        let y = x.exp().log();
        assert_eq!(y.op_count(), 2);
        let opt = y.optimize();
        assert_eq!(opt.op_count(), 0);
        assert_eq!(opt.materialize().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_materialize_matches_eager() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::<f32>::from_vec(data.clone(), &[2, 2]).unwrap();

        // Eager: relu -> add_scalar(1) -> mul_scalar(2) -> sum
        let eager = t.relu().add_scalar(1.0).mul_scalar(2.0).sum();

        // Lazy
        let lazy = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(data, &[2, 2]).unwrap(),
        );
        let lazy_result = lazy.relu().add_scalar(1.0).mul_scalar(2.0).sum().materialize();

        approx_eq(&eager.to_vec(), &lazy_result.to_vec(), 1e-6);
    }

    #[test]
    fn test_large_chain_optimization() {
        let x = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![5.0], &[1]).unwrap(),
        );
        // x * 2 * 3 * 4 + 1 + 2 + 3
        let y = x
            .mul_scalar(2.0)
            .mul_scalar(3.0)
            .mul_scalar(4.0)
            .add_scalar(1.0)
            .add_scalar(2.0)
            .add_scalar(3.0);
        assert_eq!(y.op_count(), 6);
        let opt = y.optimize();
        // mul chain: 3 -> 1 (folded), add chain: 3 -> 1 (folded) = 2 total
        assert_eq!(opt.op_count(), 2);
        // 5 * 24 + 6 = 126
        approx_eq(&opt.materialize().to_vec(), &[126.0], 1e-6);
    }

    #[test]
    fn test_binary_ops_tensor_merging() {
        // Two independent LazyTensors sharing no tensor stores
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
        );
        let b = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
        );
        // a has tensor[0], b has tensor[0] -> merged should have tensor[0], tensor[1]
        let c = a.add(&b);
        assert_eq!(c.tensors.len(), 2);
        let result = c.materialize();
        assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_binary_ops_chain_merging() {
        // (a + b) + c — should merge all three tensor stores
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap(),
        );
        let b = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap(),
        );
        let c = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![3.0], &[1]).unwrap(),
        );
        let ab = a.add(&b);
        let abc = ab.add(&c);
        assert_eq!(abc.tensors.len(), 3);
        approx_eq(&abc.materialize().to_vec(), &[6.0], 1e-6);
    }

    #[test]
    fn test_sqrt_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![4.0, 9.0, 16.0], &[3]).unwrap(),
        );
        let result = a.sqrt().materialize();
        approx_eq(&result.to_vec(), &[2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_abs_correctness() {
        let a = LazyTensor::from_tensor(
            Tensor::<f32>::from_vec(vec![-3.0, 0.0, 5.0], &[3]).unwrap(),
        );
        let result = a.abs().materialize();
        assert_eq!(result.to_vec(), vec![3.0, 0.0, 5.0]);
    }
}
