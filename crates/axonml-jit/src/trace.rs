//! Operation Tracing
//!
//! Provides tracing functionality to record tensor operations and build
//! computation graphs for JIT compilation.

use crate::ir::{DataType, Graph, NodeId, Op, Shape};
use std::cell::RefCell;

/// A traced value representing a node in the computation graph.
#[derive(Debug, Clone, Copy)]
pub struct TracedValue {
    /// Node ID in the graph.
    pub(crate) id: NodeId,
    /// Reference to the tracer (for chaining operations).
    #[allow(dead_code)]
    tracer_id: usize,
}

impl TracedValue {
    /// Creates a new traced value.
    fn new(id: NodeId, tracer_id: usize) -> Self {
        Self { id, tracer_id }
    }

    /// Returns the node ID.
    pub fn node_id(&self) -> NodeId {
        self.id
    }

    // Binary operations

    /// Element-wise addition.
    pub fn add(&self, other: &TracedValue) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.binary_op(
                Op::Add {
                    lhs: self.id,
                    rhs: other.id,
                },
                self.id,
                other.id,
            )
        })
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &TracedValue) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.binary_op(
                Op::Sub {
                    lhs: self.id,
                    rhs: other.id,
                },
                self.id,
                other.id,
            )
        })
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &TracedValue) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.binary_op(
                Op::Mul {
                    lhs: self.id,
                    rhs: other.id,
                },
                self.id,
                other.id,
            )
        })
    }

    /// Element-wise division.
    pub fn div(&self, other: &TracedValue) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.binary_op(
                Op::Div {
                    lhs: self.id,
                    rhs: other.id,
                },
                self.id,
                other.id,
            )
        })
    }

    /// Element-wise power.
    pub fn pow(&self, exp: &TracedValue) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.binary_op(
                Op::Pow {
                    base: self.id,
                    exp: exp.id,
                },
                self.id,
                exp.id,
            )
        })
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &TracedValue) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.matmul_op(self.id, other.id)
        })
    }

    // Scalar operations

    /// Add scalar.
    pub fn add_scalar(&self, scalar: f64) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(
                Op::AddScalar {
                    input: self.id,
                    scalar,
                },
                self.id,
            )
        })
    }

    /// Multiply by scalar.
    pub fn mul_scalar(&self, scalar: f64) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(
                Op::MulScalar {
                    input: self.id,
                    scalar,
                },
                self.id,
            )
        })
    }

    // Unary operations

    /// Negation.
    pub fn neg(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Neg { input: self.id }, self.id)
        })
    }

    /// Absolute value.
    pub fn abs(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Abs { input: self.id }, self.id)
        })
    }

    /// Square root.
    pub fn sqrt(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Sqrt { input: self.id }, self.id)
        })
    }

    /// Exponential.
    pub fn exp(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Exp { input: self.id }, self.id)
        })
    }

    /// Natural logarithm.
    pub fn log(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Log { input: self.id }, self.id)
        })
    }

    /// Sine.
    pub fn sin(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Sin { input: self.id }, self.id)
        })
    }

    /// Cosine.
    pub fn cos(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Cos { input: self.id }, self.id)
        })
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Tanh { input: self.id }, self.id)
        })
    }

    // Activation functions

    /// ReLU activation.
    pub fn relu(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Relu { input: self.id }, self.id)
        })
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Sigmoid { input: self.id }, self.id)
        })
    }

    /// GELU activation.
    pub fn gelu(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Gelu { input: self.id }, self.id)
        })
    }

    /// SiLU/Swish activation.
    pub fn silu(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unary_op(Op::Silu { input: self.id }, self.id)
        })
    }

    // Reduction operations

    /// Sum over all elements.
    pub fn sum(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.reduction_op(Op::Sum { input: self.id }, self.id, None, false)
        })
    }

    /// Sum over axis.
    pub fn sum_axis(&self, axis: i32, keepdim: bool) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.reduction_op(
                Op::SumAxis {
                    input: self.id,
                    axis,
                    keepdim,
                },
                self.id,
                Some(axis),
                keepdim,
            )
        })
    }

    /// Mean over all elements.
    pub fn mean(&self) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.reduction_op(Op::Mean { input: self.id }, self.id, None, false)
        })
    }

    /// Mean over axis.
    pub fn mean_axis(&self, axis: i32, keepdim: bool) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.reduction_op(
                Op::MeanAxis {
                    input: self.id,
                    axis,
                    keepdim,
                },
                self.id,
                Some(axis),
                keepdim,
            )
        })
    }

    // Shape operations

    /// Reshape tensor.
    pub fn reshape(&self, shape: &[isize]) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.reshape_op(self.id, shape)
        })
    }

    /// Transpose dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.transpose_op(self.id, dim0, dim1)
        })
    }

    /// Squeeze dimension.
    pub fn squeeze(&self, dim: i32) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.squeeze_op(self.id, dim)
        })
    }

    /// Unsqueeze (add dimension).
    pub fn unsqueeze(&self, dim: i32) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            tracer.unsqueeze_op(self.id, dim)
        })
    }
}

// Thread-local tracer for operation recording
thread_local! {
    static TRACER: RefCell<TracerState> = RefCell::new(TracerState::new());
}

/// Internal tracer state.
struct TracerState {
    graph: Graph,
    active: bool,
    tracer_id: usize,
}

impl TracerState {
    fn new() -> Self {
        Self {
            graph: Graph::new(),
            active: false,
            tracer_id: 0,
        }
    }

    fn unary_op(&mut self, op: Op, input: NodeId) -> TracedValue {
        let node = self.graph.node(input);
        let dtype = node.dtype;
        let shape = node.shape.clone();
        let id = self.graph.add_node(op, dtype, shape);
        TracedValue::new(id, self.tracer_id)
    }

    fn binary_op(&mut self, op: Op, lhs: NodeId, rhs: NodeId) -> TracedValue {
        let lhs_node = self.graph.node(lhs);
        let rhs_node = self.graph.node(rhs);

        // Use broadcast shape
        let shape = lhs_node
            .shape
            .broadcast_shape(&rhs_node.shape)
            .unwrap_or_else(|| lhs_node.shape.clone());
        let dtype = lhs_node.dtype; // Assume same dtype

        let id = self.graph.add_node(op, dtype, shape);
        TracedValue::new(id, self.tracer_id)
    }

    fn matmul_op(&mut self, lhs: NodeId, rhs: NodeId) -> TracedValue {
        let lhs_node = self.graph.node(lhs);
        let rhs_node = self.graph.node(rhs);

        let lhs_shape = lhs_node.shape.dims();
        let rhs_shape = rhs_node.shape.dims();

        // Compute output shape for matmul
        let mut output_shape = lhs_shape[..lhs_shape.len() - 1].to_vec();
        if rhs_shape.len() > 1 {
            output_shape.push(rhs_shape[rhs_shape.len() - 1]);
        }

        let id = self.graph.add_node(
            Op::MatMul { lhs, rhs },
            lhs_node.dtype,
            Shape::from(output_shape),
        );
        TracedValue::new(id, self.tracer_id)
    }

    fn reduction_op(
        &mut self,
        op: Op,
        input: NodeId,
        axis: Option<i32>,
        keepdim: bool,
    ) -> TracedValue {
        let node = self.graph.node(input);
        let dtype = node.dtype;

        let shape = if let Some(ax) = axis {
            let mut dims = node.shape.dims().to_vec();
            let ax = if ax < 0 {
                (dims.len() as i32 + ax) as usize
            } else {
                ax as usize
            };
            if keepdim {
                dims[ax] = 1;
            } else {
                dims.remove(ax);
            }
            Shape::from(dims)
        } else {
            // Full reduction
            if keepdim {
                Shape::from(vec![1; node.shape.ndim()])
            } else {
                Shape::from(vec![])
            }
        };

        let id = self.graph.add_node(op, dtype, shape);
        TracedValue::new(id, self.tracer_id)
    }

    fn reshape_op(&mut self, input: NodeId, new_shape: &[isize]) -> TracedValue {
        let node = self.graph.node(input);
        let dtype = node.dtype;
        let old_numel = node.shape.numel();

        // Resolve -1 in shape
        let mut shape: Vec<usize> = Vec::with_capacity(new_shape.len());
        let mut neg_idx = None;
        let mut known_numel = 1usize;

        for (i, &dim) in new_shape.iter().enumerate() {
            if dim == -1 {
                neg_idx = Some(i);
                shape.push(0); // Placeholder
            } else {
                let d = dim as usize;
                known_numel *= d;
                shape.push(d);
            }
        }

        if let Some(idx) = neg_idx {
            shape[idx] = old_numel / known_numel;
        }

        let id = self.graph.add_node(
            Op::Reshape {
                input,
                shape: new_shape.to_vec(),
            },
            dtype,
            Shape::from(shape),
        );
        TracedValue::new(id, self.tracer_id)
    }

    fn transpose_op(&mut self, input: NodeId, dim0: usize, dim1: usize) -> TracedValue {
        let node = self.graph.node(input);
        let dtype = node.dtype;

        let mut shape = node.shape.dims().to_vec();
        shape.swap(dim0, dim1);

        let id = self.graph.add_node(
            Op::Transpose { input, dim0, dim1 },
            dtype,
            Shape::from(shape),
        );
        TracedValue::new(id, self.tracer_id)
    }

    fn squeeze_op(&mut self, input: NodeId, dim: i32) -> TracedValue {
        let node = self.graph.node(input);
        let dtype = node.dtype;

        let mut shape = node.shape.dims().to_vec();
        let d = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };
        if shape[d] == 1 {
            shape.remove(d);
        }

        let id = self
            .graph
            .add_node(Op::Squeeze { input, dim }, dtype, Shape::from(shape));
        TracedValue::new(id, self.tracer_id)
    }

    fn unsqueeze_op(&mut self, input: NodeId, dim: i32) -> TracedValue {
        let node = self.graph.node(input);
        let dtype = node.dtype;

        let mut shape = node.shape.dims().to_vec();
        let d = if dim < 0 {
            (shape.len() as i32 + 1 + dim) as usize
        } else {
            dim as usize
        };
        shape.insert(d, 1);

        let id = self
            .graph
            .add_node(Op::Unsqueeze { input, dim }, dtype, Shape::from(shape));
        TracedValue::new(id, self.tracer_id)
    }
}

/// Tracer handle for recording operations.
pub struct Tracer {
    tracer_id: usize,
}

impl Tracer {
    /// Creates an input placeholder.
    pub fn input(&self, name: &str, shape: &[usize]) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            let id = tracer.graph.add_node(
                Op::Input {
                    name: name.to_string(),
                },
                DataType::F32,
                Shape::new(shape),
            );
            tracer.graph.register_input(name, id);
            TracedValue::new(id, self.tracer_id)
        })
    }

    /// Creates a constant tensor.
    pub fn constant(&self, value: f64, shape: &[usize]) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            let id =
                tracer
                    .graph
                    .add_node(Op::Constant { value }, DataType::F32, Shape::new(shape));
            TracedValue::new(id, self.tracer_id)
        })
    }

    /// Marks a value as output.
    pub fn output(&self, name: &str, value: TracedValue) -> TracedValue {
        TRACER.with(|t| {
            let mut tracer = t.borrow_mut();
            let node = tracer.graph.node(value.id);
            let dtype = node.dtype;
            let shape = node.shape.clone();

            let id = tracer.graph.add_node(
                Op::Output {
                    name: name.to_string(),
                    input: value.id,
                },
                dtype,
                shape,
            );
            tracer.graph.register_output(name, id);
            TracedValue::new(id, self.tracer_id)
        })
    }
}

/// Traces operations and builds a computation graph.
///
/// # Example
///
/// ```
/// use axonml_jit::trace;
///
/// let graph = trace(|tracer| {
///     let a = tracer.input("a", &[2, 3]);
///     let b = tracer.input("b", &[2, 3]);
///     let c = a.add(&b).relu();
///     tracer.output("result", c)
/// });
///
/// assert_eq!(graph.inputs().len(), 2);
/// ```
pub fn trace<F>(f: F) -> Graph
where
    F: FnOnce(&Tracer) -> TracedValue,
{
    TRACER.with(|t| {
        // Initialize fresh graph
        let mut tracer = t.borrow_mut();
        tracer.graph = Graph::new();
        tracer.active = true;
        tracer.tracer_id += 1;
        let tracer_id = tracer.tracer_id;
        drop(tracer);

        // Run the tracing function
        let tracer_handle = Tracer { tracer_id };
        let _ = f(&tracer_handle);

        // Extract the graph
        let mut tracer = t.borrow_mut();
        tracer.active = false;
        std::mem::take(&mut tracer.graph)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_simple() {
        let graph = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[2, 3]);
            let c = a.add(&b);
            tracer.output("result", c)
        });

        assert_eq!(graph.inputs().len(), 2);
        assert_eq!(graph.outputs().len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_trace_chain() {
        let graph = trace(|tracer| {
            let x = tracer.input("x", &[4, 4]);
            let y = x.relu().mul_scalar(2.0).add_scalar(1.0);
            tracer.output("y", y)
        });

        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.len(), 5); // input, relu, mul_scalar, add_scalar, output
    }

    #[test]
    fn test_trace_matmul() {
        let graph = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[3, 4]);
            let c = a.matmul(&b);
            tracer.output("c", c)
        });

        let output_id = graph.output("c").unwrap();
        let output_node = graph.node(output_id);

        // Output should be the Output node which wraps matmul
        assert!(matches!(output_node.op, Op::Output { .. }));
    }

    #[test]
    fn test_trace_reduction() {
        let graph = trace(|tracer| {
            let x = tracer.input("x", &[2, 3, 4]);
            let y = x.sum_axis(1, true);
            tracer.output("y", y)
        });

        let output_id = graph.output("y").unwrap();
        let output_node = graph.node(output_id);
        // Shape should be [2, 1, 4]
        if let Op::Output { input, .. } = &output_node.op {
            let sum_node = graph.node(*input);
            assert_eq!(sum_node.shape.dims(), &[2, 1, 4]);
        }
    }
}
