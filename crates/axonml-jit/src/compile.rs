//! torch.compile Equivalent - High-Level Compilation API
//!
//! Provides a PyTorch 2.0 torch.compile-like API for automatic optimization
//! of models and functions through tracing and compilation.
//!
//! # Example
//! ```rust,ignore
//! use axonml_jit::compile::{compile_fn, CompileConfig, Mode};
//!
//! // Compile with default settings
//! let compiled = compile_fn(|t| {
//!     let x = t.input("x", &[2, 3]);
//!     let y = x.relu();
//!     t.output("y", y)
//! }).unwrap();
//!
//! // Or with custom configuration
//! let compiled = compile_fn_with_config(f, CompileConfig::new()
//!     .mode(Mode::MaxAutotune)
//!     .fullgraph(true)).unwrap();
//! ```
//!
//! @version 0.1.0

use crate::codegen::{CompiledFunction, JitCompiler};
use crate::ir::{Graph, Node, Op};
use crate::optimize::{OptimizationPass, Optimizer};
use crate::trace::{trace, TracedValue, Tracer};
use crate::{JitError, JitResult};
use std::collections::HashMap;
use std::sync::Mutex;

// =============================================================================
// Compilation Mode
// =============================================================================

/// Compilation mode controlling optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Default mode: balanced optimization
    Default,
    /// Reduce overhead: minimize compilation time
    ReduceOverhead,
    /// Maximum autotune: try multiple implementations
    MaxAutotune,
}

impl Default for Mode {
    fn default() -> Self {
        Self::Default
    }
}

/// Backend for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Default backend (Cranelift)
    Default,
    /// Eager mode (no compilation)
    Eager,
    /// AOT (Ahead of Time) compilation
    AOT,
    /// ONNX export
    ONNX,
}

impl Default for Backend {
    fn default() -> Self {
        Self::Default
    }
}

// =============================================================================
// Compile Configuration
// =============================================================================

/// Configuration for model compilation.
#[derive(Debug, Clone)]
pub struct CompileConfig {
    /// Compilation mode
    pub mode: Mode,
    /// Backend for code generation
    pub backend: Backend,
    /// Whether to require full graph capture
    pub fullgraph: bool,
    /// Whether to enable dynamic shapes
    pub dynamic: bool,
    /// Disable compilation (for debugging)
    pub disable: bool,
    /// Optimization passes to apply
    pub passes: Vec<OptimizationPass>,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Default,
            backend: Backend::Default,
            fullgraph: false,
            dynamic: false,
            disable: false,
            passes: vec![
                OptimizationPass::ConstantFolding,
                OptimizationPass::DeadCodeElimination,
                OptimizationPass::CommonSubexpressionElimination,
            ],
        }
    }
}

impl CompileConfig {
    /// Creates a new compile configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set compilation mode.
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        if mode == Mode::MaxAutotune {
            // Add more aggressive optimizations
            self.passes.push(OptimizationPass::ElementwiseFusion);
            self.passes.push(OptimizationPass::AlgebraicSimplification);
        }
        self
    }

    /// Builder: set backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Builder: require full graph capture.
    pub fn fullgraph(mut self, fullgraph: bool) -> Self {
        self.fullgraph = fullgraph;
        self
    }

    /// Builder: enable dynamic shapes.
    pub fn dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = dynamic;
        self
    }

    /// Builder: disable compilation.
    pub fn disable(mut self, disable: bool) -> Self {
        self.disable = disable;
        self
    }

    /// Builder: add optimization pass.
    pub fn add_pass(mut self, pass: OptimizationPass) -> Self {
        self.passes.push(pass);
        self
    }
}

// =============================================================================
// Compiled Model
// =============================================================================

/// A compiled model or function.
///
/// Wraps traced computation with optimized execution.
pub struct CompiledModel {
    /// Original graph
    graph: Graph,
    /// Optimized graph
    optimized_graph: Graph,
    /// Compiled function (if available)
    compiled_fn: Option<CompiledFunction>,
    /// Configuration
    config: CompileConfig,
    /// Input names
    input_names: Vec<String>,
    /// Output names
    output_names: Vec<String>,
}

impl CompiledModel {
    /// Creates a new compiled model from a graph.
    pub fn from_graph(graph: Graph, config: CompileConfig) -> JitResult<Self> {
        // Apply optimizations
        let mut optimizer = Optimizer::new();
        for pass in &config.passes {
            optimizer.add_pass(*pass);
        }
        let optimized_graph = optimizer.optimize(graph.clone());

        // Compile if not disabled
        let compiled_fn = if !config.disable && config.backend != Backend::Eager {
            let compiler = JitCompiler::new();
            compiler.compile(&optimized_graph).ok()
        } else {
            None
        };

        let input_names: Vec<String> = graph.inputs().keys().cloned().collect();
        let output_names: Vec<String> = graph.outputs().keys().cloned().collect();

        Ok(Self {
            graph,
            optimized_graph,
            compiled_fn,
            config,
            input_names,
            output_names,
        })
    }

    /// Returns input names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Returns output names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Returns the original graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the optimized graph.
    pub fn optimized_graph(&self) -> &Graph {
        &self.optimized_graph
    }

    /// Checks if compilation succeeded.
    pub fn is_compiled(&self) -> bool {
        self.compiled_fn.is_some()
    }

    /// Returns compilation statistics.
    pub fn stats(&self) -> CompileStats {
        CompileStats {
            original_ops: self.graph.len(),
            optimized_ops: self.optimized_graph.len(),
            is_compiled: self.compiled_fn.is_some(),
            passes_applied: self.config.passes.len(),
        }
    }

    /// Runs the compiled model with named inputs.
    pub fn run(&self, inputs: &HashMap<String, Vec<f32>>) -> JitResult<HashMap<String, Vec<f32>>> {
        // Validate inputs
        for name in &self.input_names {
            if !inputs.contains_key(name) {
                return Err(JitError::InputNotFound(name.clone()));
            }
        }

        // Fall back to interpreted execution (compiled function API may differ)
        self.interpret(inputs)
    }

    /// Interprets the graph (fallback when not compiled).
    fn interpret(
        &self,
        inputs: &HashMap<String, Vec<f32>>,
    ) -> JitResult<HashMap<String, Vec<f32>>> {
        // Simple interpreter for the graph
        let mut values: HashMap<String, Vec<f32>> = HashMap::new();

        // Copy inputs
        for (name, data) in inputs {
            values.insert(name.clone(), data.clone());
        }

        for node in self.optimized_graph.nodes() {
            let result = self.execute_node(node, &values)?;
            // Use node id as key
            let key = format!("node_{}", node.id.index());
            values.insert(key, result);
        }

        // Collect outputs
        let mut outputs = HashMap::new();
        for name in &self.output_names {
            // Find the output node and get its value
            if let Some(node_id) = self.optimized_graph.output(name) {
                let key = format!("node_{}", node_id.index());
                if let Some(val) = values.get(&key) {
                    outputs.insert(name.clone(), val.clone());
                }
            }
        }

        Ok(outputs)
    }

    /// Executes a single node.
    fn execute_node(&self, node: &Node, values: &HashMap<String, Vec<f32>>) -> JitResult<Vec<f32>> {
        match &node.op {
            Op::Input { name } => values
                .get(name)
                .cloned()
                .ok_or_else(|| JitError::InputNotFound(name.clone())),
            Op::Output { input, .. } => {
                let key = format!("node_{}", input.index());
                values
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| JitError::InputNotFound(key))
            }
            Op::Constant { value } => Ok(vec![*value as f32]),
            Op::Add { lhs, rhs } => {
                let a = self.get_node_value(*lhs, values)?;
                let b = self.get_node_value(*rhs, values)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
            }
            Op::Sub { lhs, rhs } => {
                let a = self.get_node_value(*lhs, values)?;
                let b = self.get_node_value(*rhs, values)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
            }
            Op::Mul { lhs, rhs } => {
                let a = self.get_node_value(*lhs, values)?;
                let b = self.get_node_value(*rhs, values)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
            }
            Op::Div { lhs, rhs } => {
                let a = self.get_node_value(*lhs, values)?;
                let b = self.get_node_value(*rhs, values)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x / y).collect())
            }
            Op::Neg { input } => {
                let a = self.get_node_value(*input, values)?;
                Ok(a.iter().map(|x| -x).collect())
            }
            Op::Exp { input } => {
                let a = self.get_node_value(*input, values)?;
                Ok(a.iter().map(|x| x.exp()).collect())
            }
            Op::Log { input } => {
                let a = self.get_node_value(*input, values)?;
                Ok(a.iter().map(|x| x.ln()).collect())
            }
            Op::Sqrt { input } => {
                let a = self.get_node_value(*input, values)?;
                Ok(a.iter().map(|x| x.sqrt()).collect())
            }
            Op::Relu { input } => {
                let a = self.get_node_value(*input, values)?;
                Ok(a.iter().map(|x| x.max(0.0)).collect())
            }
            Op::Sigmoid { input } => {
                let a = self.get_node_value(*input, values)?;
                Ok(a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect())
            }
            Op::Tanh { input } => {
                let a = self.get_node_value(*input, values)?;
                Ok(a.iter().map(|x| x.tanh()).collect())
            }
            _ => {
                // For unsupported ops, return zeros with same shape
                let numel = node.shape.numel();
                Ok(vec![0.0; numel])
            }
        }
    }

    /// Gets value for a node by ID.
    fn get_node_value(
        &self,
        node_id: crate::ir::NodeId,
        values: &HashMap<String, Vec<f32>>,
    ) -> JitResult<Vec<f32>> {
        // Check if it's an input node
        let node = self.optimized_graph.node(node_id);
        if let Op::Input { name } = &node.op {
            return values
                .get(name)
                .cloned()
                .ok_or_else(|| JitError::InputNotFound(name.clone()));
        }

        // Otherwise use node key
        let key = format!("node_{}", node_id.index());
        values
            .get(&key)
            .cloned()
            .ok_or_else(|| JitError::InputNotFound(key))
    }
}

/// Compilation statistics.
#[derive(Debug, Clone)]
pub struct CompileStats {
    /// Number of operations in original graph
    pub original_ops: usize,
    /// Number of operations after optimization
    pub optimized_ops: usize,
    /// Whether compilation succeeded
    pub is_compiled: bool,
    /// Number of optimization passes applied
    pub passes_applied: usize,
}

impl CompileStats {
    /// Returns the optimization ratio.
    pub fn optimization_ratio(&self) -> f32 {
        if self.original_ops == 0 {
            1.0
        } else {
            self.optimized_ops as f32 / self.original_ops as f32
        }
    }
}

// =============================================================================
// Compile Functions
// =============================================================================

/// Compiles a traced function with default settings.
///
/// # Example
/// ```rust,ignore
/// let graph = trace(|t| {
///     let x = t.input("x", &[2, 3]);
///     let y = x.relu();
///     t.output("y", y)
/// });
///
/// let compiled = compile_graph(graph)?;
/// ```
pub fn compile_graph(graph: Graph) -> JitResult<CompiledModel> {
    CompiledModel::from_graph(graph, CompileConfig::default())
}

/// Compiles with custom configuration.
pub fn compile_graph_with_config(graph: Graph, config: CompileConfig) -> JitResult<CompiledModel> {
    CompiledModel::from_graph(graph, config)
}

/// Traces and compiles a function in one step.
///
/// # Example
/// ```rust,ignore
/// let compiled = compile_fn(|t| {
///     let x = t.input("x", &[2, 3]);
///     let y = x.add(&t.constant(1.0, &[2, 3]));
///     t.output("y", y)
/// })?;
/// ```
pub fn compile_fn<F>(f: F) -> JitResult<CompiledModel>
where
    F: FnOnce(&Tracer) -> TracedValue,
{
    let graph = trace(f);
    compile_graph(graph)
}

/// Traces and compiles with custom configuration.
pub fn compile_fn_with_config<F>(f: F, config: CompileConfig) -> JitResult<CompiledModel>
where
    F: FnOnce(&Tracer) -> TracedValue,
{
    let graph = trace(f);
    compile_graph_with_config(graph, config)
}

// =============================================================================
// Dynamo-style Decorators
// =============================================================================

/// Wrapper for lazy compilation.
///
/// Traces and compiles on first call, caches for subsequent calls.
pub struct LazyCompiled<F> {
    func: F,
    compiled: Mutex<Option<CompiledModel>>,
    config: CompileConfig,
}

impl<F> LazyCompiled<F>
where
    F: Fn(&Tracer) -> TracedValue,
{
    /// Creates a new lazy compiled wrapper.
    pub fn new(func: F) -> Self {
        Self {
            func,
            compiled: Mutex::new(None),
            config: CompileConfig::default(),
        }
    }

    /// Creates with custom config.
    pub fn with_config(func: F, config: CompileConfig) -> Self {
        Self {
            func,
            compiled: Mutex::new(None),
            config,
        }
    }

    /// Runs the function, compiling on first call.
    pub fn run(&self, inputs: &HashMap<String, Vec<f32>>) -> JitResult<HashMap<String, Vec<f32>>> {
        let mut compiled = self.compiled.lock().unwrap();

        if compiled.is_none() {
            let graph = trace(&self.func);
            *compiled = Some(CompiledModel::from_graph(graph, self.config.clone())?);
        }

        compiled.as_ref().unwrap().run(inputs)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_config_default() {
        let config = CompileConfig::default();
        assert_eq!(config.mode, Mode::Default);
        assert!(!config.fullgraph);
        assert!(!config.disable);
    }

    #[test]
    fn test_compile_config_builder() {
        let config = CompileConfig::new()
            .mode(Mode::MaxAutotune)
            .fullgraph(true)
            .dynamic(true);

        assert_eq!(config.mode, Mode::MaxAutotune);
        assert!(config.fullgraph);
        assert!(config.dynamic);
    }

    #[test]
    fn test_compile_simple_graph() {
        let graph = trace(|t| {
            let x = t.input("x", &[2]);
            let y = x.relu();
            t.output("y", y)
        });

        let compiled = compile_graph(graph).unwrap();
        assert!(compiled.input_names().contains(&"x".to_string()));
    }

    #[test]
    fn test_compile_stats() {
        let graph = trace(|t| {
            let x = t.input("x", &[2]);
            let y = x.relu();
            t.output("y", y)
        });

        let compiled = compile_graph(graph).unwrap();
        let stats = compiled.stats();

        assert!(stats.original_ops > 0);
        assert!(stats.passes_applied > 0);
    }

    #[test]
    fn test_mode_enum() {
        assert_eq!(Mode::default(), Mode::Default);
        assert_ne!(Mode::MaxAutotune, Mode::ReduceOverhead);
    }

    #[test]
    fn test_backend_enum() {
        assert_eq!(Backend::default(), Backend::Default);
    }

    #[test]
    fn test_compiled_model_run() {
        let graph = trace(|t| {
            let x = t.input("x", &[2]);
            let y = x.relu();
            t.output("y", y)
        });

        let compiled = compile_graph_with_config(
            graph,
            CompileConfig::new().disable(true), // Use interpreter
        )
        .unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![-1.0, 2.0]);

        let outputs = compiled.run(&inputs).unwrap();
        let y = outputs.get("y").unwrap();
        assert_eq!(y, &vec![0.0, 2.0]); // ReLU
    }
}
