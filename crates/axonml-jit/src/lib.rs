//! JIT Compilation for Axonml
//!
//! This crate provides Just-In-Time compilation for tensor operations,
//! enabling significant performance improvements through:
//!
//! - Operation tracing and graph construction
//! - Graph optimization (fusion, constant folding, dead code elimination)
//! - Native code generation via Cranelift
//! - Compiled function caching
//!
//! # Example
//!
//! ```ignore
//! use axonml_jit::{JitCompiler, trace};
//!
//! // Trace operations to build a computation graph
//! let graph = trace(|tracer| {
//!     let a = tracer.input("a", &[2, 3]);
//!     let b = tracer.input("b", &[2, 3]);
//!     let c = a.add(&b);
//!     let d = c.mul_scalar(2.0);
//!     tracer.output("result", d)
//! });
//!
//! // Compile the graph
//! let compiler = JitCompiler::new();
//! let compiled = compiler.compile(&graph)?;
//!
//! // Execute with real tensors
//! let a = Tensor::randn(&[2, 3]);
//! let b = Tensor::randn(&[2, 3]);
//! let result = compiled.run(&[("a", &a), ("b", &b)])?;
//! ```
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod cache;
pub mod codegen;
pub mod compile;
pub mod error;
pub mod ir;
pub mod optimize;
pub mod trace;

pub use cache::FunctionCache;
pub use codegen::{CompiledFunction, JitCompiler};
pub use compile::{
    compile_fn, compile_fn_with_config, compile_graph, compile_graph_with_config, Backend,
    CompileConfig, CompileStats, CompiledModel, LazyCompiled, Mode,
};
pub use error::{JitError, JitResult};
pub use ir::{DataType, Graph, Node, NodeId, Op, Shape};
pub use optimize::{OptimizationPass, Optimizer};
pub use trace::{trace, TracedValue, Tracer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_trace() {
        let graph = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[2, 3]);
            let c = a.add(&b);
            tracer.output("result", c)
        });

        assert_eq!(graph.inputs().len(), 2);
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_optimization() {
        let graph = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.constant(2.0, &[2, 3]);
            let c = a.mul(&b);
            tracer.output("result", c)
        });

        let mut optimizer = Optimizer::new();
        optimizer.add_pass(OptimizationPass::ConstantFolding);
        let optimized = optimizer.optimize(graph);

        // Graph should still be valid
        assert_eq!(optimized.inputs().len(), 1);
    }
}
