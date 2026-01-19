//! Computational Graph - Dynamic Graph Construction
//!
//! Manages the computational graph that tracks operations for automatic
//! differentiation. The graph is built dynamically during the forward pass
//! and traversed in reverse during backward pass.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::grad_fn::GradFn;

// =============================================================================
// Graph Node
// =============================================================================

/// Unique identifier for graph nodes.
pub type NodeId = u64;

/// Counter for generating unique node IDs.
static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generates a new unique node ID.
fn new_node_id() -> NodeId {
    NODE_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// A node in the computational graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// The gradient function for this node.
    pub grad_fn: Option<GradFn>,
    /// Whether this node requires gradient computation.
    pub requires_grad: bool,
    /// Whether this is a leaf node (user-created variable).
    pub is_leaf: bool,
    /// Topological order for efficient backward traversal.
    pub topo_order: u64,
}

impl GraphNode {
    /// Creates a new leaf node (for user-created variables).
    #[must_use] pub fn leaf(requires_grad: bool) -> Self {
        Self {
            id: new_node_id(),
            grad_fn: None,
            requires_grad,
            is_leaf: true,
            topo_order: 0,
        }
    }

    /// Creates a new intermediate node (result of an operation).
    #[must_use] pub fn intermediate(grad_fn: GradFn, requires_grad: bool, topo_order: u64) -> Self {
        Self {
            id: new_node_id(),
            grad_fn: Some(grad_fn),
            requires_grad,
            is_leaf: false,
            topo_order,
        }
    }
}

// =============================================================================
// Computation Graph
// =============================================================================

/// The computational graph for automatic differentiation.
///
/// Tracks all operations performed on tensors that require gradients,
/// enabling automatic computation of derivatives via backpropagation.
#[derive(Debug)]
pub struct ComputationGraph {
    /// All nodes in the graph, indexed by ID.
    nodes: RwLock<HashMap<NodeId, Arc<GraphNode>>>,
    /// Current maximum topological order.
    max_topo_order: AtomicU64,
}

impl ComputationGraph {
    /// Creates a new empty computation graph.
    #[must_use] pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            max_topo_order: AtomicU64::new(0),
        }
    }

    /// Registers a new leaf node in the graph.
    pub fn register_leaf(&self, requires_grad: bool) -> Arc<GraphNode> {
        let node = Arc::new(GraphNode::leaf(requires_grad));
        self.nodes.write().insert(node.id, Arc::clone(&node));
        node
    }

    /// Registers a new intermediate node resulting from an operation.
    pub fn register_operation(&self, grad_fn: GradFn, requires_grad: bool) -> Arc<GraphNode> {
        let topo_order = self.max_topo_order.fetch_add(1, Ordering::SeqCst) + 1;
        let node = Arc::new(GraphNode::intermediate(grad_fn, requires_grad, topo_order));
        self.nodes.write().insert(node.id, Arc::clone(&node));
        node
    }

    /// Gets a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<Arc<GraphNode>> {
        self.nodes.read().get(&id).cloned()
    }

    /// Returns the number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Returns true if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Clears all nodes from the graph.
    pub fn clear(&self) {
        self.nodes.write().clear();
        self.max_topo_order.store(0, Ordering::SeqCst);
    }

    /// Resets the node ID counter (use with caution, mainly for testing).
    pub fn reset_id_counter() {
        NODE_ID_COUNTER.store(0, Ordering::SeqCst);
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Thread-Local Graph
// =============================================================================

thread_local! {
    /// Thread-local computation graph.
    static GRAPH: ComputationGraph = ComputationGraph::new();
}

/// Gets a reference to the thread-local computation graph.
pub fn with_graph<F, R>(f: F) -> R
where
    F: FnOnce(&ComputationGraph) -> R,
{
    GRAPH.with(|g| f(g))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad_fn::{AccumulateGrad, GradAccumulator};
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[test]
    fn test_graph_node_leaf() {
        let node = GraphNode::leaf(true);
        assert!(node.is_leaf);
        assert!(node.requires_grad);
        assert!(node.grad_fn.is_none());
    }

    #[test]
    fn test_graph_node_intermediate() {
        let grad_acc: GradAccumulator = Arc::new(RwLock::new(None));
        let grad_fn = GradFn::new(AccumulateGrad::new(grad_acc));
        let node = GraphNode::intermediate(grad_fn, true, 1);
        assert!(!node.is_leaf);
        assert!(node.requires_grad);
        assert!(node.grad_fn.is_some());
    }

    #[test]
    fn test_computation_graph() {
        let graph = ComputationGraph::new();
        assert!(graph.is_empty());

        let leaf = graph.register_leaf(true);
        assert_eq!(graph.len(), 1);
        assert!(leaf.is_leaf);

        let grad_acc: GradAccumulator = Arc::new(RwLock::new(None));
        let grad_fn = GradFn::new(AccumulateGrad::new(grad_acc));
        let intermediate = graph.register_operation(grad_fn, true);
        assert_eq!(graph.len(), 2);
        assert!(!intermediate.is_leaf);
    }

    #[test]
    fn test_graph_clear() {
        let graph = ComputationGraph::new();
        graph.register_leaf(true);
        graph.register_leaf(false);
        assert_eq!(graph.len(), 2);

        graph.clear();
        assert!(graph.is_empty());
    }
}
