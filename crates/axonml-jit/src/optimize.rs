//! Graph Optimization
//!
//! Provides optimization passes for computation graphs.

use crate::ir::{Graph, NodeId, Op};
use rustc_hash::{FxHashMap, FxHashSet};

/// Optimization passes available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Fold constant expressions.
    ConstantFolding,
    /// Remove dead (unused) code.
    DeadCodeElimination,
    /// Fuse consecutive elementwise operations.
    ElementwiseFusion,
    /// Common subexpression elimination.
    CommonSubexpressionElimination,
    /// Algebraic simplifications (x * 1 = x, x + 0 = x, etc).
    AlgebraicSimplification,
    /// Strength reduction (expensive ops -> cheaper ops).
    StrengthReduction,
}

/// Graph optimizer.
pub struct Optimizer {
    passes: Vec<OptimizationPass>,
}

impl Optimizer {
    /// Creates a new optimizer with no passes.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Creates an optimizer with default passes.
    pub fn default_passes() -> Self {
        Self {
            passes: vec![
                OptimizationPass::ConstantFolding,
                OptimizationPass::AlgebraicSimplification,
                OptimizationPass::DeadCodeElimination,
                OptimizationPass::CommonSubexpressionElimination,
            ],
        }
    }

    /// Adds an optimization pass.
    pub fn add_pass(&mut self, pass: OptimizationPass) {
        self.passes.push(pass);
    }

    /// Runs all optimization passes on the graph.
    pub fn optimize(&self, mut graph: Graph) -> Graph {
        for pass in &self.passes {
            graph = self.run_pass(graph, *pass);
        }
        graph
    }

    fn run_pass(&self, graph: Graph, pass: OptimizationPass) -> Graph {
        match pass {
            OptimizationPass::ConstantFolding => constant_folding(graph),
            OptimizationPass::DeadCodeElimination => dead_code_elimination(graph),
            OptimizationPass::ElementwiseFusion => elementwise_fusion(graph),
            OptimizationPass::CommonSubexpressionElimination => cse(graph),
            OptimizationPass::AlgebraicSimplification => algebraic_simplification(graph),
            OptimizationPass::StrengthReduction => strength_reduction(graph),
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::default_passes()
    }
}

/// Constant folding: evaluate constant expressions at compile time.
fn constant_folding(graph: Graph) -> Graph {
    // For now, just identify constant nodes
    // Full implementation would evaluate constant subgraphs
    let mut new_graph = Graph::new();
    let mut node_map: FxHashMap<NodeId, NodeId> = FxHashMap::default();
    let mut constants: FxHashMap<NodeId, f64> = FxHashMap::default();

    for node in graph.nodes() {
        // Track constant values
        if let Op::Constant { value } = &node.op {
            constants.insert(node.id, *value);
        }

        // Try to fold binary ops with constants
        let new_op = match &node.op {
            Op::MulScalar { input, scalar } if *scalar == 1.0 => {
                // x * 1 = x
                let new_input = node_map.get(input).copied().unwrap_or(*input);
                node_map.insert(node.id, new_input);
                continue;
            }
            Op::MulScalar { input: _, scalar } if *scalar == 0.0 => {
                // x * 0 = 0
                Op::Constant { value: 0.0 }
            }
            Op::AddScalar { input, scalar } if *scalar == 0.0 => {
                // x + 0 = x
                let new_input = node_map.get(input).copied().unwrap_or(*input);
                node_map.insert(node.id, new_input);
                continue;
            }
            other => remap_op(other, &node_map),
        };

        let new_id = new_graph.add_node(new_op, node.dtype, node.shape.clone());
        node_map.insert(node.id, new_id);
    }

    // Remap inputs and outputs
    for (name, id) in graph.inputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_input(name, new_id);
        }
    }
    for (name, id) in graph.outputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_output(name, new_id);
        }
    }

    new_graph
}

/// Dead code elimination: remove nodes that don't contribute to outputs.
fn dead_code_elimination(graph: Graph) -> Graph {
    // Find all nodes reachable from outputs
    let mut live_nodes: FxHashSet<NodeId> = FxHashSet::default();
    let mut worklist: Vec<NodeId> = graph.outputs().values().copied().collect();

    while let Some(id) = worklist.pop() {
        if live_nodes.insert(id) {
            let node = graph.node(id);
            for input_id in node.op.inputs() {
                worklist.push(input_id);
            }
        }
    }

    // Rebuild graph with only live nodes
    let mut new_graph = Graph::new();
    let mut node_map: FxHashMap<NodeId, NodeId> = FxHashMap::default();

    for node in graph.nodes() {
        if !live_nodes.contains(&node.id) {
            continue;
        }

        let new_op = remap_op(&node.op, &node_map);
        let new_id = new_graph.add_node(new_op, node.dtype, node.shape.clone());
        node_map.insert(node.id, new_id);
    }

    // Remap inputs and outputs
    for (name, id) in graph.inputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_input(name, new_id);
        }
    }
    for (name, id) in graph.outputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_output(name, new_id);
        }
    }

    new_graph
}

/// Elementwise fusion: combine consecutive elementwise ops into kernels.
fn elementwise_fusion(graph: Graph) -> Graph {
    // For now, just return the graph unchanged
    // Full implementation would identify fusible sequences
    // and create FusedElementwise nodes
    graph
}

/// Common subexpression elimination.
fn cse(graph: Graph) -> Graph {
    // Hash-based CSE
    let mut new_graph = Graph::new();
    let mut node_map: FxHashMap<NodeId, NodeId> = FxHashMap::default();
    let mut expr_map: FxHashMap<String, NodeId> = FxHashMap::default();

    for node in graph.nodes() {
        let remapped_op = remap_op(&node.op, &node_map);
        let expr_key = format!("{:?}", remapped_op);

        if let Some(&existing_id) = expr_map.get(&expr_key) {
            // Reuse existing node
            node_map.insert(node.id, existing_id);
        } else {
            let new_id = new_graph.add_node(remapped_op, node.dtype, node.shape.clone());
            node_map.insert(node.id, new_id);
            expr_map.insert(expr_key, new_id);
        }
    }

    // Remap inputs and outputs
    for (name, id) in graph.inputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_input(name, new_id);
        }
    }
    for (name, id) in graph.outputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_output(name, new_id);
        }
    }

    new_graph
}

/// Algebraic simplifications.
fn algebraic_simplification(graph: Graph) -> Graph {
    let mut new_graph = Graph::new();
    let mut node_map: FxHashMap<NodeId, NodeId> = FxHashMap::default();

    for node in graph.nodes() {
        let simplified_op = match &node.op {
            // x * 1 = x
            Op::MulScalar { input, scalar } if *scalar == 1.0 => {
                let new_input = node_map.get(input).copied().unwrap_or(*input);
                node_map.insert(node.id, new_input);
                continue;
            }
            // x + 0 = x
            Op::AddScalar { input, scalar } if *scalar == 0.0 => {
                let new_input = node_map.get(input).copied().unwrap_or(*input);
                node_map.insert(node.id, new_input);
                continue;
            }
            // x - 0 = x (via AddScalar with -0)
            // x / 1 = x (via MulScalar with 1)
            // --x = x
            Op::Neg { input } => {
                let actual_input = node_map.get(input).copied().unwrap_or(*input);
                if let Some(input_node) = new_graph.nodes().iter().find(|n| n.id == actual_input) {
                    if let Op::Neg { input: inner } = &input_node.op {
                        node_map.insert(node.id, *inner);
                        continue;
                    }
                }
                Op::Neg { input: actual_input }
            }
            other => remap_op(other, &node_map),
        };

        let new_id = new_graph.add_node(simplified_op, node.dtype, node.shape.clone());
        node_map.insert(node.id, new_id);
    }

    // Remap inputs and outputs
    for (name, id) in graph.inputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_input(name, new_id);
        }
    }
    for (name, id) in graph.outputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_output(name, new_id);
        }
    }

    new_graph
}

/// Strength reduction: replace expensive ops with cheaper equivalents.
fn strength_reduction(graph: Graph) -> Graph {
    let mut new_graph = Graph::new();
    let mut node_map: FxHashMap<NodeId, NodeId> = FxHashMap::default();

    for node in graph.nodes() {
        let reduced_op = match &node.op {
            // x^2 -> x * x
            Op::Pow { .. } => {
                // Check if exp is constant 2
                // For now, just pass through
                remap_op(&node.op, &node_map)
            }
            // x / c -> x * (1/c) for constant c
            Op::Div { .. } => {
                // Would need to check if rhs is constant
                remap_op(&node.op, &node_map)
            }
            other => remap_op(other, &node_map),
        };

        let new_id = new_graph.add_node(reduced_op, node.dtype, node.shape.clone());
        node_map.insert(node.id, new_id);
    }

    // Remap inputs and outputs
    for (name, id) in graph.inputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_input(name, new_id);
        }
    }
    for (name, id) in graph.outputs() {
        if let Some(&new_id) = node_map.get(id) {
            new_graph.register_output(name, new_id);
        }
    }

    new_graph
}

/// Remaps node IDs in an operation using the provided mapping.
fn remap_op(op: &Op, node_map: &FxHashMap<NodeId, NodeId>) -> Op {
    let remap = |id: &NodeId| node_map.get(id).copied().unwrap_or(*id);

    match op {
        Op::Input { name } => Op::Input { name: name.clone() },
        Op::Output { name, input } => Op::Output { name: name.clone(), input: remap(input) },
        Op::Constant { value } => Op::Constant { value: *value },

        Op::Add { lhs, rhs } => Op::Add { lhs: remap(lhs), rhs: remap(rhs) },
        Op::Sub { lhs, rhs } => Op::Sub { lhs: remap(lhs), rhs: remap(rhs) },
        Op::Mul { lhs, rhs } => Op::Mul { lhs: remap(lhs), rhs: remap(rhs) },
        Op::Div { lhs, rhs } => Op::Div { lhs: remap(lhs), rhs: remap(rhs) },
        Op::Pow { base, exp } => Op::Pow { base: remap(base), exp: remap(exp) },
        Op::Max { lhs, rhs } => Op::Max { lhs: remap(lhs), rhs: remap(rhs) },
        Op::Min { lhs, rhs } => Op::Min { lhs: remap(lhs), rhs: remap(rhs) },

        Op::Neg { input } => Op::Neg { input: remap(input) },
        Op::Abs { input } => Op::Abs { input: remap(input) },
        Op::Sqrt { input } => Op::Sqrt { input: remap(input) },
        Op::Exp { input } => Op::Exp { input: remap(input) },
        Op::Log { input } => Op::Log { input: remap(input) },
        Op::Sin { input } => Op::Sin { input: remap(input) },
        Op::Cos { input } => Op::Cos { input: remap(input) },
        Op::Tanh { input } => Op::Tanh { input: remap(input) },

        Op::Relu { input } => Op::Relu { input: remap(input) },
        Op::Sigmoid { input } => Op::Sigmoid { input: remap(input) },
        Op::Gelu { input } => Op::Gelu { input: remap(input) },
        Op::Silu { input } => Op::Silu { input: remap(input) },

        Op::AddScalar { input, scalar } => Op::AddScalar { input: remap(input), scalar: *scalar },
        Op::MulScalar { input, scalar } => Op::MulScalar { input: remap(input), scalar: *scalar },

        Op::Sum { input } => Op::Sum { input: remap(input) },
        Op::SumAxis { input, axis, keepdim } => Op::SumAxis { input: remap(input), axis: *axis, keepdim: *keepdim },
        Op::Mean { input } => Op::Mean { input: remap(input) },
        Op::MeanAxis { input, axis, keepdim } => Op::MeanAxis { input: remap(input), axis: *axis, keepdim: *keepdim },
        Op::MaxAxis { input, axis, keepdim } => Op::MaxAxis { input: remap(input), axis: *axis, keepdim: *keepdim },

        Op::Reshape { input, shape } => Op::Reshape { input: remap(input), shape: shape.clone() },
        Op::Transpose { input, dim0, dim1 } => Op::Transpose { input: remap(input), dim0: *dim0, dim1: *dim1 },
        Op::Squeeze { input, dim } => Op::Squeeze { input: remap(input), dim: *dim },
        Op::Unsqueeze { input, dim } => Op::Unsqueeze { input: remap(input), dim: *dim },
        Op::Broadcast { input, shape } => Op::Broadcast { input: remap(input), shape: shape.clone() },

        Op::MatMul { lhs, rhs } => Op::MatMul { lhs: remap(lhs), rhs: remap(rhs) },

        Op::Gt { lhs, rhs } => Op::Gt { lhs: remap(lhs), rhs: remap(rhs) },
        Op::Lt { lhs, rhs } => Op::Lt { lhs: remap(lhs), rhs: remap(rhs) },
        Op::Eq { lhs, rhs } => Op::Eq { lhs: remap(lhs), rhs: remap(rhs) },

        Op::Where { condition, x, y } => Op::Where { condition: remap(condition), x: remap(x), y: remap(y) },

        Op::Cast { input, dtype } => Op::Cast { input: remap(input), dtype: *dtype },
        Op::Contiguous { input } => Op::Contiguous { input: remap(input) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::trace;

    #[test]
    fn test_dead_code_elimination() {
        let graph = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[2, 3]);
            let _unused = a.mul(&b); // This should be eliminated
            let c = a.add(&b);
            tracer.output("result", c)
        });

        let optimizer = Optimizer::new();
        let mut opt = optimizer;
        opt.add_pass(OptimizationPass::DeadCodeElimination);
        let optimized = opt.optimize(graph);

        // Mul node should be eliminated
        let has_mul = optimized.nodes().iter().any(|n| matches!(n.op, Op::Mul { .. }));
        assert!(!has_mul);
    }

    #[test]
    fn test_algebraic_simplification() {
        let graph = trace(|tracer| {
            let x = tracer.input("x", &[2, 3]);
            let y = x.mul_scalar(1.0); // Should be simplified to x
            tracer.output("y", y)
        });

        let mut optimizer = Optimizer::new();
        optimizer.add_pass(OptimizationPass::AlgebraicSimplification);
        let optimized = optimizer.optimize(graph);

        // MulScalar(1.0) should be eliminated
        let has_mul_scalar = optimized.nodes().iter().any(|n| matches!(n.op, Op::MulScalar { .. }));
        assert!(!has_mul_scalar);
    }

    #[test]
    fn test_constant_folding() {
        let graph = trace(|tracer| {
            let x = tracer.input("x", &[2, 3]);
            let y = x.mul_scalar(0.0); // Should become constant 0
            tracer.output("y", y)
        });

        let mut optimizer = Optimizer::new();
        optimizer.add_pass(OptimizationPass::ConstantFolding);
        let optimized = optimizer.optimize(graph);

        // Should have a Constant node
        let has_constant = optimized.nodes().iter().any(|n| matches!(n.op, Op::Constant { .. }));
        assert!(has_constant);
    }
}
