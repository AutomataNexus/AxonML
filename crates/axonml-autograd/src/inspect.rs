//! Graph Inspection and Visualization
//!
//! # File
//! `crates/axonml-autograd/src/inspect.rs`
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

use std::collections::{HashMap, HashSet};

use crate::grad_fn::{GradFn, GradFnId};
use crate::variable::Variable;

// =============================================================================
// Types
// =============================================================================

/// A frozen snapshot of the computation graph for analysis.
///
/// Since the live computation graph is thread-local and mutable, this struct
/// captures a point-in-time view that can be safely analyzed, serialized,
/// or visualized.
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// All nodes in the snapshot, in the order they were discovered.
    pub nodes: Vec<SnapshotNode>,
    /// Directed edges representing gradient flow: `(from_idx, to_idx)` where
    /// indices refer to positions in `nodes`.
    pub edges: Vec<(usize, usize)>,
}

/// A single node in a [`GraphSnapshot`].
#[derive(Debug, Clone)]
pub struct SnapshotNode {
    /// Internal ID (derived from the `GradFnId`).
    pub id: usize,
    /// Human-readable name (e.g. "AddBackward", "AccumulateGrad", "Leaf").
    pub name: String,
    /// Whether this node is a leaf (parameter / user-created variable).
    pub is_leaf: bool,
    /// Whether this node requires gradient computation.
    pub requires_grad: bool,
    /// Tensor shape, if available.
    pub shape: Option<Vec<usize>>,
}

// =============================================================================
// Core Inspection Functions
// =============================================================================

/// Traces the backward graph reachable from `variable` and returns a frozen
/// [`GraphSnapshot`].
///
/// Walks backward through the `grad_fn` chain using DFS, deduplicating by
/// `GradFnId`. Each `GradFn` becomes a node; `AccumulateGrad` nodes are
/// marked as leaves.
///
/// # Arguments
/// * `variable` - The variable from which to trace backward.
///
/// # Returns
/// A `GraphSnapshot` containing all reachable nodes and edges.
pub fn trace_backward(variable: &Variable) -> GraphSnapshot {
    let mut nodes: Vec<SnapshotNode> = Vec::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut visited: HashMap<GradFnId, usize> = HashMap::new(); // id -> index in nodes

    match variable.grad_fn() {
        Some(gf) => {
            trace_dfs(gf, &mut nodes, &mut edges, &mut visited);
        }
        None => {
            // Leaf variable with no grad_fn chain — emit a single "Leaf" node
            nodes.push(SnapshotNode {
                id: 0,
                name: "Leaf".to_string(),
                is_leaf: true,
                requires_grad: variable.requires_grad(),
                shape: Some(variable.shape()),
            });
        }
    }

    GraphSnapshot { nodes, edges }
}

/// Recursive DFS helper for [`trace_backward`].
fn trace_dfs(
    grad_fn: &GradFn,
    nodes: &mut Vec<SnapshotNode>,
    edges: &mut Vec<(usize, usize)>,
    visited: &mut HashMap<GradFnId, usize>,
) -> usize {
    let fn_id = grad_fn.id();

    // Already visited — return existing index
    if let Some(&idx) = visited.get(&fn_id) {
        return idx;
    }

    let name = grad_fn.name().to_string();
    let is_leaf = name == "AccumulateGrad";

    let idx = nodes.len();
    nodes.push(SnapshotNode {
        id: fn_id,
        name,
        is_leaf,
        requires_grad: true, // anything in the grad_fn chain requires grad
        shape: None,         // shape not directly available from GradFn
    });
    visited.insert(fn_id, idx);

    // Recurse into parents
    for maybe_next in grad_fn.next_functions() {
        if let Some(next) = maybe_next {
            let child_idx = trace_dfs(next, nodes, edges, visited);
            edges.push((idx, child_idx));
        }
    }

    idx
}

/// Converts a [`GraphSnapshot`] to Graphviz DOT format.
///
/// Leaf nodes (parameters) are rendered as boxes, operation nodes as ellipses.
/// Edges represent the direction of gradient flow (backward).
///
/// # Arguments
/// * `snapshot` - The graph snapshot to convert.
///
/// # Returns
/// A `String` containing valid DOT source.
pub fn to_dot(snapshot: &GraphSnapshot) -> String {
    let mut dot = String::from("digraph computation_graph {\n");
    dot.push_str("    rankdir=TB;\n");
    dot.push_str("    node [fontname=\"Helvetica\"];\n\n");

    for (i, node) in snapshot.nodes.iter().enumerate() {
        let shape_label = if let Some(ref s) = node.shape {
            format!("\\n{:?}", s)
        } else {
            String::new()
        };

        if node.is_leaf {
            dot.push_str(&format!(
                "    n{} [label=\"{}{}\" shape=box style=filled fillcolor=\"#E8F5E9\"];\n",
                i, node.name, shape_label
            ));
        } else {
            dot.push_str(&format!(
                "    n{} [label=\"{}{}\" shape=ellipse style=filled fillcolor=\"#E3F2FD\"];\n",
                i, node.name, shape_label
            ));
        }
    }

    dot.push('\n');

    for &(from, to) in &snapshot.edges {
        dot.push_str(&format!("    n{} -> n{};\n", from, to));
    }

    dot.push_str("}\n");
    dot
}

/// Returns the total number of nodes reachable from `variable` in the
/// backward graph.
pub fn node_count(variable: &Variable) -> usize {
    match variable.grad_fn() {
        Some(gf) => {
            let mut visited = HashSet::new();
            count_dfs(gf, &mut visited);
            visited.len()
        }
        None => 1, // leaf node counts as 1
    }
}

/// DFS helper that populates a visited set.
fn count_dfs(grad_fn: &GradFn, visited: &mut HashSet<GradFnId>) {
    let fn_id = grad_fn.id();
    if !visited.insert(fn_id) {
        return;
    }
    for maybe_next in grad_fn.next_functions() {
        if let Some(next) = maybe_next {
            count_dfs(next, visited);
        }
    }
}

/// Returns the maximum depth of the backward graph from `variable`.
///
/// A leaf variable has depth 0. A variable produced by one operation on
/// leaf inputs has depth 1, and so on.
pub fn depth(variable: &Variable) -> usize {
    match variable.grad_fn() {
        Some(gf) => {
            let mut visited = HashSet::new();
            depth_dfs(gf, &mut visited)
        }
        None => 0,
    }
}

/// DFS helper that computes maximum depth.
fn depth_dfs(grad_fn: &GradFn, visited: &mut HashSet<GradFnId>) -> usize {
    let fn_id = grad_fn.id();
    if !visited.insert(fn_id) {
        return 0;
    }

    let mut max_child_depth: usize = 0;
    for maybe_next in grad_fn.next_functions() {
        if let Some(next) = maybe_next {
            let d = depth_dfs(next, visited);
            max_child_depth = max_child_depth.max(d);
        }
    }

    // Remove from visited so other paths can explore this node at different depths
    visited.remove(&fn_id);

    max_child_depth + 1
}

/// Returns the number of leaf (AccumulateGrad) nodes reachable from `variable`.
pub fn leaf_count(variable: &Variable) -> usize {
    match variable.grad_fn() {
        Some(gf) => {
            let mut visited = HashSet::new();
            leaf_count_dfs(gf, &mut visited)
        }
        None => {
            if variable.requires_grad() {
                1
            } else {
                0
            }
        }
    }
}

/// DFS helper for counting leaf nodes.
fn leaf_count_dfs(grad_fn: &GradFn, visited: &mut HashSet<GradFnId>) -> usize {
    let fn_id = grad_fn.id();
    if !visited.insert(fn_id) {
        return 0;
    }

    if grad_fn.name() == "AccumulateGrad" {
        return 1;
    }

    let mut count = 0;
    for maybe_next in grad_fn.next_functions() {
        if let Some(next) = maybe_next {
            count += leaf_count_dfs(next, visited);
        }
    }
    count
}

/// Returns a list of unique operation names in the backward graph.
///
/// Excludes "AccumulateGrad" (leaf sentinel). The returned list is sorted
/// alphabetically for deterministic output.
pub fn operation_names(variable: &Variable) -> Vec<String> {
    match variable.grad_fn() {
        Some(gf) => {
            let mut visited = HashSet::new();
            let mut names = HashSet::new();
            op_names_dfs(gf, &mut visited, &mut names);
            let mut result: Vec<String> = names.into_iter().collect();
            result.sort();
            result
        }
        None => Vec::new(),
    }
}

/// DFS helper for collecting operation names.
fn op_names_dfs(grad_fn: &GradFn, visited: &mut HashSet<GradFnId>, names: &mut HashSet<String>) {
    let fn_id = grad_fn.id();
    if !visited.insert(fn_id) {
        return;
    }

    let name = grad_fn.name();
    if name != "AccumulateGrad" {
        names.insert(name.to_string());
    }

    for maybe_next in grad_fn.next_functions() {
        if let Some(next) = maybe_next {
            op_names_dfs(next, visited, names);
        }
    }
}

/// Returns a summary of operation types and their counts in the backward graph.
///
/// Each entry is `(operation_name, count)`. Excludes "AccumulateGrad".
/// Results are sorted by count descending, then alphabetically for ties.
pub fn gradient_flow_summary(variable: &Variable) -> Vec<(String, usize)> {
    match variable.grad_fn() {
        Some(gf) => {
            let mut visited = HashSet::new();
            let mut counts: HashMap<String, usize> = HashMap::new();
            summary_dfs(gf, &mut visited, &mut counts);

            let mut result: Vec<(String, usize)> = counts.into_iter().collect();
            result.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            result
        }
        None => Vec::new(),
    }
}

/// DFS helper for building the operation summary.
fn summary_dfs(
    grad_fn: &GradFn,
    visited: &mut HashSet<GradFnId>,
    counts: &mut HashMap<String, usize>,
) {
    let fn_id = grad_fn.id();
    if !visited.insert(fn_id) {
        return;
    }

    let name = grad_fn.name();
    if name != "AccumulateGrad" {
        *counts.entry(name.to_string()).or_insert(0) += 1;
    }

    for maybe_next in grad_fn.next_functions() {
        if let Some(next) = maybe_next {
            summary_dfs(next, visited, counts);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    // -------------------------------------------------------------------------
    // trace_backward tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trace_leaf_no_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        let snap = trace_backward(&x);
        assert_eq!(snap.nodes.len(), 1);
        assert_eq!(snap.edges.len(), 0);
        assert_eq!(snap.nodes[0].name, "Leaf");
        assert!(snap.nodes[0].is_leaf);
        assert!(!snap.nodes[0].requires_grad);
    }

    #[test]
    fn test_trace_leaf_with_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), true);
        let snap = trace_backward(&x);
        // A leaf with requires_grad has an AccumulateGrad grad_fn
        assert_eq!(snap.nodes.len(), 1);
        assert!(snap.nodes[0].is_leaf);
        assert_eq!(snap.nodes[0].name, "AccumulateGrad");
    }

    #[test]
    fn test_trace_simple_add() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = a.add_var(&b);
        let snap = trace_backward(&c);

        // Should have 3 nodes: AddBackward + 2 AccumulateGrad
        assert_eq!(snap.nodes.len(), 3);
        // Should have 2 edges: AddBackward -> AccumulateGrad(a), AddBackward -> AccumulateGrad(b)
        assert_eq!(snap.edges.len(), 2);

        // Root should be AddBackward
        assert_eq!(snap.nodes[0].name, "AddBackward");
        assert!(!snap.nodes[0].is_leaf);
    }

    #[test]
    fn test_trace_chain_relu_sigmoid() {
        let a = Variable::new(Tensor::from_vec(vec![0.5], &[1]).unwrap(), true);
        let b = a.relu();
        let c = b.sigmoid();
        let snap = trace_backward(&c);

        // SigmoidBackward -> ReluBackward -> AccumulateGrad
        assert_eq!(snap.nodes.len(), 3);
        assert_eq!(snap.edges.len(), 2);
        assert_eq!(snap.nodes[0].name, "SigmoidBackward");
        assert_eq!(snap.nodes[1].name, "ReluBackward");
        assert_eq!(snap.nodes[2].name, "AccumulateGrad");
    }

    #[test]
    fn test_trace_diamond_topology() {
        // Diamond: a -> b, a -> c, b + c -> d
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.relu();
        let c = a.sigmoid();
        let d = b.add_var(&c);
        let snap = trace_backward(&d);

        // AddBackward, ReluBackward, SigmoidBackward, AccumulateGrad(a) — shared leaf
        assert_eq!(snap.nodes.len(), 4);
        // AddBackward->Relu, AddBackward->Sigmoid, Relu->Accum, Sigmoid->Accum
        assert_eq!(snap.edges.len(), 4);
    }

    #[test]
    fn test_trace_snapshot_shape_on_leaf() {
        let x = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            false,
        );
        let snap = trace_backward(&x);
        assert_eq!(snap.nodes[0].shape, Some(vec![2, 2]));
    }

    // -------------------------------------------------------------------------
    // node_count tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_node_count_leaf() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        assert_eq!(node_count(&x), 1);
    }

    #[test]
    fn test_node_count_leaf_with_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        // AccumulateGrad node
        assert_eq!(node_count(&x), 1);
    }

    #[test]
    fn test_node_count_simple() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = a.add_var(&b);
        // AddBackward + 2 AccumulateGrad
        assert_eq!(node_count(&c), 3);
    }

    #[test]
    fn test_node_count_chain() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.relu().sigmoid().tanh();
        // TanhBackward -> SigmoidBackward -> ReluBackward -> AccumulateGrad
        assert_eq!(node_count(&b), 4);
    }

    #[test]
    fn test_node_count_diamond() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.relu();
        let c = a.sigmoid();
        let d = b.add_var(&c);
        // AddBackward, ReluBackward, SigmoidBackward, AccumulateGrad (shared)
        assert_eq!(node_count(&d), 4);
    }

    // -------------------------------------------------------------------------
    // depth tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_leaf() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        assert_eq!(depth(&x), 0);
    }

    #[test]
    fn test_depth_single_op() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.relu();
        // ReluBackward -> AccumulateGrad  =>  depth 2
        assert_eq!(depth(&b), 2);
    }

    #[test]
    fn test_depth_chain() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.relu().sigmoid().tanh();
        // Tanh -> Sigmoid -> Relu -> Accum  =>  depth 4
        assert_eq!(depth(&b), 4);
    }

    #[test]
    fn test_depth_branching() {
        // Two branches of different length merging
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let left = a.relu().sigmoid(); // depth 3 from leaf
        let right = b.relu(); // depth 2 from leaf
        let merged = left.add_var(&right);
        // AddBackward at top, max path is left side: Add->Sig->Relu->Accum = 4
        assert_eq!(depth(&merged), 4);
    }

    // -------------------------------------------------------------------------
    // leaf_count tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_leaf_count_leaf_no_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        assert_eq!(leaf_count(&x), 0);
    }

    #[test]
    fn test_leaf_count_leaf_with_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        assert_eq!(leaf_count(&x), 1);
    }

    #[test]
    fn test_leaf_count_two_inputs() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = a.add_var(&b);
        assert_eq!(leaf_count(&c), 2);
    }

    #[test]
    fn test_leaf_count_shared_input() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.relu();
        let c = a.sigmoid();
        let d = b.add_var(&c);
        // Only one leaf (a), even though it's used twice
        assert_eq!(leaf_count(&d), 1);
    }

    #[test]
    fn test_leaf_count_three_inputs() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let ab = a.add_var(&b);
        let abc = ab.add_var(&c);
        assert_eq!(leaf_count(&abc), 3);
    }

    // -------------------------------------------------------------------------
    // operation_names tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operation_names_leaf() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let names = operation_names(&x);
        // AccumulateGrad is excluded
        assert!(names.is_empty());
    }

    #[test]
    fn test_operation_names_chain() {
        let a = Variable::new(Tensor::from_vec(vec![0.5], &[1]).unwrap(), true);
        let b = a.relu().sigmoid().tanh();
        let names = operation_names(&b);
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"ReluBackward".to_string()));
        assert!(names.contains(&"SigmoidBackward".to_string()));
        assert!(names.contains(&"TanhBackward".to_string()));
    }

    #[test]
    fn test_operation_names_sorted() {
        let a = Variable::new(Tensor::from_vec(vec![0.5], &[1]).unwrap(), true);
        let b = a.tanh().relu().sigmoid();
        let names = operation_names(&b);
        // Should be alphabetically sorted
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_operation_names_no_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        let names = operation_names(&x);
        assert!(names.is_empty());
    }

    // -------------------------------------------------------------------------
    // gradient_flow_summary tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_summary_leaf() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let summary = gradient_flow_summary(&x);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_summary_simple() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = a.add_var(&b);
        let summary = gradient_flow_summary(&c);
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0], ("AddBackward".to_string(), 1));
    }

    #[test]
    fn test_summary_multiple_adds() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let ab = a.add_var(&b);
        let abc = ab.add_var(&c);
        let summary = gradient_flow_summary(&abc);
        // Two AddBackward nodes
        assert!(summary
            .iter()
            .any(|(name, count)| name == "AddBackward" && *count == 2));
    }

    #[test]
    fn test_summary_sorted_by_count() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        // a.relu() + b => AddBackward(1), ReluBackward(1)
        let c = a.relu().add_var(&b);
        let summary = gradient_flow_summary(&c);
        assert_eq!(summary.len(), 2);
        // Both have count 1, so sorted alphabetically
        assert_eq!(summary[0].0, "AddBackward");
        assert_eq!(summary[1].0, "ReluBackward");
    }

    #[test]
    fn test_summary_no_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        let summary = gradient_flow_summary(&x);
        assert!(summary.is_empty());
    }

    // -------------------------------------------------------------------------
    // to_dot tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_to_dot_contains_digraph() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        let snap = trace_backward(&x);
        let dot = to_dot(&snap);
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_to_dot_contains_edges() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = a.add_var(&b);
        let snap = trace_backward(&c);
        let dot = to_dot(&snap);
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_to_dot_leaf_box_shape() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), false);
        let snap = trace_backward(&x);
        let dot = to_dot(&snap);
        assert!(dot.contains("shape=box"));
    }

    #[test]
    fn test_to_dot_operation_ellipse() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.relu();
        let snap = trace_backward(&b);
        let dot = to_dot(&snap);
        assert!(dot.contains("shape=ellipse"));
        assert!(dot.contains("ReluBackward"));
    }

    #[test]
    fn test_to_dot_node_labels() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = a.sigmoid();
        let snap = trace_backward(&b);
        let dot = to_dot(&snap);
        assert!(dot.contains("SigmoidBackward"));
        assert!(dot.contains("AccumulateGrad"));
    }

    // -------------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_snapshot_dot() {
        let snap = GraphSnapshot {
            nodes: Vec::new(),
            edges: Vec::new(),
        };
        let dot = to_dot(&snap);
        assert!(dot.contains("digraph"));
        assert!(!dot.contains("->"));
    }

    #[test]
    fn test_trace_mul_has_correct_structure() {
        let a = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let c = a.mul_var(&b);
        let snap = trace_backward(&c);

        assert_eq!(snap.nodes.len(), 3);
        assert_eq!(snap.nodes[0].name, "MulBackward");
        assert_eq!(snap.edges.len(), 2);
    }

    #[test]
    fn test_trace_sum_mean_chain() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let b = a.sum();
        let snap = trace_backward(&b);
        assert_eq!(snap.nodes.len(), 2); // SumBackward + AccumulateGrad
        assert!(snap.nodes.iter().any(|n| n.name == "SumBackward"));
    }

    #[test]
    fn test_node_count_matches_snapshot() {
        let a = Variable::new(Tensor::from_vec(vec![1.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let c = a.relu().add_var(&b.sigmoid());
        assert_eq!(node_count(&c), trace_backward(&c).nodes.len());
    }
}
