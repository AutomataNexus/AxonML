//! Intermediate Representation
//!
//! Defines the graph-based IR for JIT compilation.

use rustc_hash::FxHashMap;

/// Unique identifier for a node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// Returns the raw index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Boolean.
    Bool,
}

impl DataType {
    /// Size in bytes.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::Bool => 1,
        }
    }
}

impl Default for DataType {
    fn default() -> Self {
        Self::F32
    }
}

/// Shape of a tensor (dimensions).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Creates a new shape.
    pub fn new(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    /// Returns the dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Checks if shapes are broadcast compatible.
    pub fn broadcast_compatible(&self, other: &Self) -> bool {
        let max_ndim = self.ndim().max(other.ndim());
        for i in 0..max_ndim {
            let d1 = if i < self.ndim() {
                self.0[self.ndim() - 1 - i]
            } else {
                1
            };
            let d2 = if i < other.ndim() {
                other.0[other.ndim() - 1 - i]
            } else {
                1
            };
            if d1 != d2 && d1 != 1 && d2 != 1 {
                return false;
            }
        }
        true
    }

    /// Computes broadcast shape.
    pub fn broadcast_shape(&self, other: &Self) -> Option<Self> {
        if !self.broadcast_compatible(other) {
            return None;
        }

        let max_ndim = self.ndim().max(other.ndim());
        let mut result = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let d1 = if i < self.ndim() {
                self.0[self.ndim() - 1 - i]
            } else {
                1
            };
            let d2 = if i < other.ndim() {
                other.0[other.ndim() - 1 - i]
            } else {
                1
            };
            result.push(d1.max(d2));
        }

        result.reverse();
        Some(Self(result))
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

/// Operations supported by the JIT compiler.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum Op {
    // Inputs/Outputs
    /// Input placeholder.
    Input { name: String },
    /// Output marker.
    Output { name: String, input: NodeId },
    /// Constant value.
    Constant { value: f64 },

    // Binary operations
    /// Element-wise addition.
    Add { lhs: NodeId, rhs: NodeId },
    /// Element-wise subtraction.
    Sub { lhs: NodeId, rhs: NodeId },
    /// Element-wise multiplication.
    Mul { lhs: NodeId, rhs: NodeId },
    /// Element-wise division.
    Div { lhs: NodeId, rhs: NodeId },
    /// Element-wise power.
    Pow { base: NodeId, exp: NodeId },
    /// Element-wise maximum.
    Max { lhs: NodeId, rhs: NodeId },
    /// Element-wise minimum.
    Min { lhs: NodeId, rhs: NodeId },

    // Unary operations
    /// Negation.
    Neg { input: NodeId },
    /// Absolute value.
    Abs { input: NodeId },
    /// Square root.
    Sqrt { input: NodeId },
    /// Exponential.
    Exp { input: NodeId },
    /// Natural logarithm.
    Log { input: NodeId },
    /// Sine.
    Sin { input: NodeId },
    /// Cosine.
    Cos { input: NodeId },
    /// Hyperbolic tangent.
    Tanh { input: NodeId },

    // Activation functions
    /// ReLU activation.
    Relu { input: NodeId },
    /// Sigmoid activation.
    Sigmoid { input: NodeId },
    /// GELU activation.
    Gelu { input: NodeId },
    /// SiLU/Swish activation.
    Silu { input: NodeId },

    // Scalar operations
    /// Add scalar.
    AddScalar { input: NodeId, scalar: f64 },
    /// Multiply by scalar.
    MulScalar { input: NodeId, scalar: f64 },

    // Reduction operations
    /// Sum over all elements.
    Sum { input: NodeId },
    /// Sum over axis.
    SumAxis { input: NodeId, axis: i32, keepdim: bool },
    /// Mean over all elements.
    Mean { input: NodeId },
    /// Mean over axis.
    MeanAxis { input: NodeId, axis: i32, keepdim: bool },
    /// Maximum over axis.
    MaxAxis { input: NodeId, axis: i32, keepdim: bool },

    // Shape operations
    /// Reshape tensor.
    Reshape { input: NodeId, shape: Vec<isize> },
    /// Transpose dimensions.
    Transpose { input: NodeId, dim0: usize, dim1: usize },
    /// Squeeze dimension.
    Squeeze { input: NodeId, dim: i32 },
    /// Unsqueeze (add dimension).
    Unsqueeze { input: NodeId, dim: i32 },
    /// Broadcast to shape.
    Broadcast { input: NodeId, shape: Vec<usize> },

    // Matrix operations
    /// Matrix multiplication.
    MatMul { lhs: NodeId, rhs: NodeId },

    // Comparison operations
    /// Element-wise greater than.
    Gt { lhs: NodeId, rhs: NodeId },
    /// Element-wise less than.
    Lt { lhs: NodeId, rhs: NodeId },
    /// Element-wise equality.
    Eq { lhs: NodeId, rhs: NodeId },

    // Conditional
    /// Where/select operation.
    Where { condition: NodeId, x: NodeId, y: NodeId },

    // Special
    /// Cast to different dtype.
    Cast { input: NodeId, dtype: DataType },
    /// Contiguous (copy to contiguous memory).
    Contiguous { input: NodeId },
}

impl Op {
    /// Returns the input node IDs for this operation.
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            Self::Input { .. } | Self::Constant { .. } => vec![],
            Self::Output { input, .. }
            | Self::Neg { input }
            | Self::Abs { input }
            | Self::Sqrt { input }
            | Self::Exp { input }
            | Self::Log { input }
            | Self::Sin { input }
            | Self::Cos { input }
            | Self::Tanh { input }
            | Self::Relu { input }
            | Self::Sigmoid { input }
            | Self::Gelu { input }
            | Self::Silu { input }
            | Self::AddScalar { input, .. }
            | Self::MulScalar { input, .. }
            | Self::Sum { input }
            | Self::SumAxis { input, .. }
            | Self::Mean { input }
            | Self::MeanAxis { input, .. }
            | Self::MaxAxis { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Squeeze { input, .. }
            | Self::Unsqueeze { input, .. }
            | Self::Broadcast { input, .. }
            | Self::Cast { input, .. }
            | Self::Contiguous { input } => vec![*input],
            Self::Add { lhs, rhs }
            | Self::Sub { lhs, rhs }
            | Self::Mul { lhs, rhs }
            | Self::Div { lhs, rhs }
            | Self::Pow { base: lhs, exp: rhs }
            | Self::Max { lhs, rhs }
            | Self::Min { lhs, rhs }
            | Self::MatMul { lhs, rhs }
            | Self::Gt { lhs, rhs }
            | Self::Lt { lhs, rhs }
            | Self::Eq { lhs, rhs } => vec![*lhs, *rhs],
            Self::Where { condition, x, y } => vec![*condition, *x, *y],
        }
    }

    /// Returns whether this is an elementwise operation.
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Self::Add { .. }
                | Self::Sub { .. }
                | Self::Mul { .. }
                | Self::Div { .. }
                | Self::Pow { .. }
                | Self::Max { .. }
                | Self::Min { .. }
                | Self::Neg { .. }
                | Self::Abs { .. }
                | Self::Sqrt { .. }
                | Self::Exp { .. }
                | Self::Log { .. }
                | Self::Sin { .. }
                | Self::Cos { .. }
                | Self::Tanh { .. }
                | Self::Relu { .. }
                | Self::Sigmoid { .. }
                | Self::Gelu { .. }
                | Self::Silu { .. }
                | Self::AddScalar { .. }
                | Self::MulScalar { .. }
                | Self::Gt { .. }
                | Self::Lt { .. }
                | Self::Eq { .. }
                | Self::Where { .. }
        )
    }

    /// Returns whether this is a reduction operation.
    pub fn is_reduction(&self) -> bool {
        matches!(
            self,
            Self::Sum { .. }
                | Self::SumAxis { .. }
                | Self::Mean { .. }
                | Self::MeanAxis { .. }
                | Self::MaxAxis { .. }
        )
    }
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// Operation performed by this node.
    pub op: Op,
    /// Output data type.
    pub dtype: DataType,
    /// Output shape.
    pub shape: Shape,
}

/// Computation graph for JIT compilation.
#[derive(Debug, Clone)]
pub struct Graph {
    /// All nodes in the graph.
    nodes: Vec<Node>,
    /// Input nodes (name -> NodeId).
    inputs: FxHashMap<String, NodeId>,
    /// Output nodes (name -> NodeId).
    outputs: FxHashMap<String, NodeId>,
}

impl Graph {
    /// Creates a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: FxHashMap::default(),
            outputs: FxHashMap::default(),
        }
    }

    /// Adds a node to the graph.
    pub fn add_node(&mut self, op: Op, dtype: DataType, shape: Shape) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { id, op, dtype, shape });
        id
    }

    /// Registers an input node.
    pub fn register_input(&mut self, name: &str, id: NodeId) {
        self.inputs.insert(name.to_string(), id);
    }

    /// Registers an output node.
    pub fn register_output(&mut self, name: &str, id: NodeId) {
        self.outputs.insert(name.to_string(), id);
    }

    /// Returns the node for an ID.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Returns mutable node for an ID.
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0]
    }

    /// Returns all nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns input names and node IDs.
    pub fn inputs(&self) -> &FxHashMap<String, NodeId> {
        &self.inputs
    }

    /// Returns output names and node IDs.
    pub fn outputs(&self) -> &FxHashMap<String, NodeId> {
        &self.outputs
    }

    /// Returns the input node ID for a name.
    pub fn input(&self, name: &str) -> Option<NodeId> {
        self.inputs.get(name).copied()
    }

    /// Returns the output node ID for a name.
    pub fn output(&self, name: &str) -> Option<NodeId> {
        self.outputs.get(name).copied()
    }

    /// Returns nodes in topological order.
    pub fn topological_order(&self) -> Vec<NodeId> {
        // Simple topological sort since nodes are already added in order
        (0..self.nodes.len()).map(NodeId).collect()
    }

    /// Validates the graph structure.
    pub fn validate(&self) -> Result<(), String> {
        // Check all input references are valid
        for node in &self.nodes {
            for input_id in node.op.inputs() {
                if input_id.0 >= self.nodes.len() {
                    return Err(format!(
                        "Node {:?} references invalid input {:?}",
                        node.id, input_id
                    ));
                }
                if input_id.0 >= node.id.0 {
                    return Err(format!(
                        "Node {:?} references future node {:?} (not DAG)",
                        node.id, input_id
                    ));
                }
            }
        }

        // Check inputs are actually Input ops
        for (name, id) in &self.inputs {
            let node = &self.nodes[id.0];
            if !matches!(node.op, Op::Input { .. }) {
                return Err(format!("Input '{}' points to non-Input node", name));
            }
        }

        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_numel() {
        let shape = Shape::new(&[2, 3, 4]);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    fn test_shape_broadcast() {
        let s1 = Shape::new(&[2, 1, 4]);
        let s2 = Shape::new(&[3, 4]);
        assert!(s1.broadcast_compatible(&s2));

        let result = s1.broadcast_shape(&s2).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new();

        let input = graph.add_node(
            Op::Input { name: "x".to_string() },
            DataType::F32,
            Shape::new(&[2, 3]),
        );
        graph.register_input("x", input);

        let relu = graph.add_node(
            Op::Relu { input },
            DataType::F32,
            Shape::new(&[2, 3]),
        );

        let output = graph.add_node(
            Op::Output { name: "y".to_string(), input: relu },
            DataType::F32,
            Shape::new(&[2, 3]),
        );
        graph.register_output("y", output);

        assert_eq!(graph.len(), 3);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_op_inputs() {
        let add = Op::Add { lhs: NodeId(0), rhs: NodeId(1) };
        assert_eq!(add.inputs(), vec![NodeId(0), NodeId(1)]);

        let relu = Op::Relu { input: NodeId(2) };
        assert_eq!(relu.inputs(), vec![NodeId(2)]);

        let input = Op::Input { name: "x".to_string() };
        assert!(input.inputs().is_empty());
    }
}
