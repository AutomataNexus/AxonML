//! Graph Optimization — Six-Pass IR Transformation Pipeline
//!
//! Implements `Optimizer`, the driver that runs an ordered list of
//! `OptimizationPass` transformations over a `Graph`. `OptimizationPass`
//! enumerates ConstantFolding, DeadCodeElimination, ElementwiseFusion,
//! CommonSubexpressionElimination, AlgebraicSimplification, and
//! StrengthReduction. `default_passes` seeds the pipeline with folding, algebra,
//! DCE, and CSE. Each pass is a standalone `fn(Graph) -> Graph`:
//! `constant_folding` rewrites `x * 1 -> x`, `x * 0 -> Constant(0)`, and
//! `x + 0 -> x` while tracking known-constant node values;
//! `dead_code_elimination` walks backward from the registered outputs to build
//! a `FxHashSet` of live nodes and rebuilds the graph keeping only those;
//! `elementwise_fusion` is a no-op stub; `cse` uses a debug-formatted op string
//! as the hash key in an `FxHashMap<String, NodeId>` to deduplicate structurally
//! identical subexpressions; `algebraic_simplification` collapses `x * 1`,
//! `x + 0`, and double negation (`--x -> x`); `strength_reduction` is a
//! pass-through placeholder for Pow/Div cheapening. All passes share `remap_op`,
//! a dense helper that rewrites every `Op` variant's `NodeId` references through
//! a `FxHashMap` produced during graph rebuilding. Tests verify DCE removes an
//! unused Mul, algebraic simplification eliminates `MulScalar(1.0)`, and
//! constant folding materializes a Constant for `MulScalar(0.0)`.
//!
//! # File
//! `crates/axonml-jit/src/optimize.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use crate::ir::{Graph, NodeId, Op};
use rustc_hash::{FxHashMap, FxHashSet};

// =============================================================================
// OptimizationPass
// =============================================================================

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

// =============================================================================
// Optimizer Driver
// =============================================================================

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

// =============================================================================
// Optimizer Passes
// =============================================================================

// -----------------------------------------------------------------------------
// Constant Folding
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Dead Code Elimination
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Elementwise Fusion
// -----------------------------------------------------------------------------

/// Elementwise fusion: collapse maximal chains of SINGLE-USE unary elementwise ops into one
/// `FusedChain` node (one memory pass instead of one per op). A node is a chain link iff it is a
/// unary elementwise op whose input is produced by another node used ONLY by this node (so fusing
/// cannot change any other consumer's result). Binary/reduction/matmul ops break a chain.
fn elementwise_fusion(graph: Graph) -> Graph {
    use crate::ir::FusedStep;

    // Classify a node's op as a fusible unary step (with its scalar), or None.
    fn as_step(op: &Op) -> Option<(FusedStep, f64)> {
        Some(match op {
            Op::Neg { .. } => (FusedStep::Neg, 0.0),
            Op::Abs { .. } => (FusedStep::Abs, 0.0),
            Op::Sqrt { .. } => (FusedStep::Sqrt, 0.0),
            Op::Exp { .. } => (FusedStep::Exp, 0.0),
            Op::Log { .. } => (FusedStep::Log, 0.0),
            Op::Sin { .. } => (FusedStep::Sin, 0.0),
            Op::Cos { .. } => (FusedStep::Cos, 0.0),
            Op::Tanh { .. } => (FusedStep::Tanh, 0.0),
            Op::Relu { .. } => (FusedStep::Relu, 0.0),
            Op::Sigmoid { .. } => (FusedStep::Sigmoid, 0.0),
            Op::Gelu { .. } => (FusedStep::Gelu, 0.0),
            Op::Silu { .. } => (FusedStep::Silu, 0.0),
            Op::AddScalar { scalar, .. } => (FusedStep::AddScalar, *scalar),
            Op::MulScalar { scalar, .. } => (FusedStep::MulScalar, *scalar),
            _ => return None,
        })
    }
    fn step_input(op: &Op) -> Option<NodeId> {
        match op {
            Op::Neg { input }
            | Op::Abs { input }
            | Op::Sqrt { input }
            | Op::Exp { input }
            | Op::Log { input }
            | Op::Sin { input }
            | Op::Cos { input }
            | Op::Tanh { input }
            | Op::Relu { input }
            | Op::Sigmoid { input }
            | Op::Gelu { input }
            | Op::Silu { input }
            | Op::AddScalar { input, .. }
            | Op::MulScalar { input, .. } => Some(*input),
            _ => None,
        }
    }

    // use-count over the whole graph (a node feeding an Output counts as a use too).
    let mut uses: FxHashMap<NodeId, usize> = FxHashMap::default();
    for node in graph.nodes() {
        for inp in node.op.inputs() {
            *uses.entry(inp).or_insert(0) += 1;
        }
    }

    // Rebuild the graph; when a fusible node's chain of single-use unary parents can be extended,
    // emit one FusedChain instead. Process in original order (topological by construction).
    let mut ng = Graph::new();
    let mut map: FxHashMap<NodeId, NodeId> = FxHashMap::default();

    for node in graph.nodes() {
        // Walk back through single-use unary parents to find the chain root + steps.
        if let Some((_, _)) = as_step(&node.op) {
            let mut steps_rev: Vec<(FusedStep, f64)> = Vec::new();
            let mut cur = node.id;
            let mut root_input: Option<NodeId> = None;
            loop {
                let op = &graph.node(cur).op;
                match as_step(op) {
                    Some(st) => {
                        steps_rev.push(st);
                        let inp = step_input(op).unwrap();
                        // extend only if the parent is used solely by `cur`
                        if uses.get(&inp).copied().unwrap_or(0) == 1
                            && as_step(&graph.node(inp).op).is_some()
                        {
                            cur = inp;
                        } else {
                            root_input = Some(inp);
                            break;
                        }
                    }
                    None => {
                        break;
                    }
                }
            }
            if let Some(rin) = root_input {
                if steps_rev.len() >= 2 {
                    steps_rev.reverse();
                    let rin_new = map.get(&rin).copied().unwrap_or(rin);
                    let nid = ng.add_node(
                        Op::FusedChain {
                            input: rin_new,
                            steps: steps_rev,
                        },
                        node.dtype,
                        node.shape.clone(),
                    );
                    map.insert(node.id, nid);
                    continue;
                }
            }
        }
        // Not a chain tail (or chain too short): copy, remapping inputs. Skip nodes already
        // subsumed into a FusedChain (they have no map entry and are not referenced downstream).
        let remapped = remap_op(&node.op, &map);
        let nid = ng.add_node(remapped, node.dtype, node.shape.clone());
        map.insert(node.id, nid);
    }

    for (name, id) in graph.inputs() {
        if let Some(&n) = map.get(id) {
            ng.register_input(name, n);
        }
    }
    for (name, id) in graph.outputs() {
        if let Some(&n) = map.get(id) {
            ng.register_output(name, n);
        }
    }
    dead_code_elimination(ng)
}

// -----------------------------------------------------------------------------
// Common Subexpression Elimination
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Algebraic Simplification
// -----------------------------------------------------------------------------

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
                Op::Neg {
                    input: actual_input,
                }
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

// -----------------------------------------------------------------------------
// Strength Reduction
// -----------------------------------------------------------------------------

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

// =============================================================================
// Helpers
// =============================================================================

/// Remaps node IDs in an operation using the provided mapping.
fn remap_op(op: &Op, node_map: &FxHashMap<NodeId, NodeId>) -> Op {
    let remap = |id: &NodeId| node_map.get(id).copied().unwrap_or(*id);

    match op {
        Op::FusedChain { input, steps } => Op::FusedChain {
            input: remap(input),
            steps: steps.clone(),
        },
        Op::Input { name } => Op::Input { name: name.clone() },
        Op::Output { name, input } => Op::Output {
            name: name.clone(),
            input: remap(input),
        },
        Op::Constant { value } => Op::Constant { value: *value },

        Op::Add { lhs, rhs } => Op::Add {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        Op::Sub { lhs, rhs } => Op::Sub {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        Op::Mul { lhs, rhs } => Op::Mul {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        Op::Div { lhs, rhs } => Op::Div {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        Op::Pow { base, exp } => Op::Pow {
            base: remap(base),
            exp: remap(exp),
        },
        Op::Max { lhs, rhs } => Op::Max {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        Op::Min { lhs, rhs } => Op::Min {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },

        Op::Neg { input } => Op::Neg {
            input: remap(input),
        },
        Op::Abs { input } => Op::Abs {
            input: remap(input),
        },
        Op::Sqrt { input } => Op::Sqrt {
            input: remap(input),
        },
        Op::Exp { input } => Op::Exp {
            input: remap(input),
        },
        Op::Log { input } => Op::Log {
            input: remap(input),
        },
        Op::Sin { input } => Op::Sin {
            input: remap(input),
        },
        Op::Cos { input } => Op::Cos {
            input: remap(input),
        },
        Op::Tanh { input } => Op::Tanh {
            input: remap(input),
        },

        Op::Relu { input } => Op::Relu {
            input: remap(input),
        },
        Op::Sigmoid { input } => Op::Sigmoid {
            input: remap(input),
        },
        Op::Gelu { input } => Op::Gelu {
            input: remap(input),
        },
        Op::Silu { input } => Op::Silu {
            input: remap(input),
        },

        Op::AddScalar { input, scalar } => Op::AddScalar {
            input: remap(input),
            scalar: *scalar,
        },
        Op::MulScalar { input, scalar } => Op::MulScalar {
            input: remap(input),
            scalar: *scalar,
        },

        Op::Sum { input } => Op::Sum {
            input: remap(input),
        },
        Op::SumAxis {
            input,
            axis,
            keepdim,
        } => Op::SumAxis {
            input: remap(input),
            axis: *axis,
            keepdim: *keepdim,
        },
        Op::Mean { input } => Op::Mean {
            input: remap(input),
        },
        Op::MeanAxis {
            input,
            axis,
            keepdim,
        } => Op::MeanAxis {
            input: remap(input),
            axis: *axis,
            keepdim: *keepdim,
        },
        Op::MaxAxis {
            input,
            axis,
            keepdim,
        } => Op::MaxAxis {
            input: remap(input),
            axis: *axis,
            keepdim: *keepdim,
        },

        Op::Reshape { input, shape } => Op::Reshape {
            input: remap(input),
            shape: shape.clone(),
        },
        Op::Transpose { input, dim0, dim1 } => Op::Transpose {
            input: remap(input),
            dim0: *dim0,
            dim1: *dim1,
        },
        Op::Squeeze { input, dim } => Op::Squeeze {
            input: remap(input),
            dim: *dim,
        },
        Op::Unsqueeze { input, dim } => Op::Unsqueeze {
            input: remap(input),
            dim: *dim,
        },
        Op::Broadcast { input, shape } => Op::Broadcast {
            input: remap(input),
            shape: shape.clone(),
        },

        Op::MatMul { lhs, rhs } => Op::MatMul {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },

        Op::Gt { lhs, rhs } => Op::Gt {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        Op::Lt { lhs, rhs } => Op::Lt {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },
        Op::Eq { lhs, rhs } => Op::Eq {
            lhs: remap(lhs),
            rhs: remap(rhs),
        },

        Op::Where { condition, x, y } => Op::Where {
            condition: remap(condition),
            x: remap(x),
            y: remap(y),
        },

        Op::Cast { input, dtype } => Op::Cast {
            input: remap(input),
            dtype: *dtype,
        },
        Op::Contiguous { input } => Op::Contiguous {
            input: remap(input),
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

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
        let has_mul = optimized
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Mul { .. }));
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
        let has_mul_scalar = optimized
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::MulScalar { .. }));
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
        let has_constant = optimized
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Constant { .. }));
        assert!(has_constant);
    }

    #[test]
    fn fuses_unary_chain_and_computes_same() {
        use crate::ir::{DataType, Shape};
        // out = sqrt(relu(x*2 + 1)) -> 4 single-use unary steps over one input.
        let mut g = Graph::new();
        let x = g.add_node(
            Op::Input { name: "x".into() },
            DataType::F32,
            Shape(vec![8]),
        );
        g.register_input("x", x);
        let a = g.add_node(
            Op::MulScalar {
                input: x,
                scalar: 2.0,
            },
            DataType::F32,
            Shape(vec![8]),
        );
        let b = g.add_node(
            Op::AddScalar {
                input: a,
                scalar: 1.0,
            },
            DataType::F32,
            Shape(vec![8]),
        );
        let c = g.add_node(Op::Relu { input: b }, DataType::F32, Shape(vec![8]));
        let d = g.add_node(Op::Sqrt { input: c }, DataType::F32, Shape(vec![8]));
        let o = g.add_node(
            Op::Output {
                name: "out".into(),
                input: d,
            },
            DataType::F32,
            Shape(vec![8]),
        );
        g.register_output("out", o);

        let fused = elementwise_fusion(g.clone());
        let n_fused = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedChain { .. }))
            .count();
        assert!(
            n_fused >= 1,
            "expected a FusedChain node, got graph: {:?}",
            fused.nodes().iter().map(|n| &n.op).collect::<Vec<_>>()
        );

        let run = |graph: &Graph| -> Vec<f32> {
            crate::codegen::CompiledFunction::from_graph_for_test(graph.clone())
                .run(&[("x", &[-1.0, 0.5, 2.0, -3.0, 4.0, 0.1, -0.2, 9.0])])
                .unwrap()
        };
        // fusion + DCE must REDUCE node count (4 unary ops + I/O -> 1 FusedChain + I/O).
        assert!(
            fused.nodes().len() < g.nodes().len(),
            "fusion should shrink the graph: {} -> {}",
            g.nodes().len(),
            fused.nodes().len()
        );
        let (ra, rb) = (run(&g), run(&fused));
        let md = ra
            .iter()
            .zip(&rb)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-6, "fused graph diverges: {md} ({ra:?} vs {rb:?})");
    }
}
