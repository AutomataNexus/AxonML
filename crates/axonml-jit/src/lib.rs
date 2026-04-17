//! AxonML JIT — Tracing, IR, Optimization, and Cranelift Codegen
//!
//! Top-level crate module for the AxonML just-in-time compiler. Re-exports the
//! tracing surface (`Tracer`, `TracedValue`, `trace`), typed IR (`Graph`,
//! `Node`, `NodeId`, `Op`, `Shape`, `DataType`), six-pass optimizer
//! (`Optimizer`, `OptimizationPass` covering constant folding, DCE, CSE,
//! algebraic simplification, elementwise fusion, and strength reduction), the
//! Cranelift-backed code generator (`JitCompiler`, `CompiledFunction`), the
//! higher-level `compile_fn` / `compile_graph` / `CompiledModel` /
//! `LazyCompiled` facade with `CompileConfig`, `CompileStats`, `Backend`, and
//! `Mode`, the `FunctionCache` LRU for compiled-function reuse, and the error
//! types `JitError` / `JitResult`. Unit tests exercise a basic trace round-trip
//! and a constant-folding optimization pass.
//!
//! # File
//! `crates/axonml-jit/src/lib.rs`
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

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod cache;
pub mod codegen;
pub mod compile;
pub mod error;
pub mod ir;
pub mod optimize;
pub mod trace;

// =============================================================================
// Public Re-exports
// =============================================================================

pub use cache::FunctionCache;
pub use codegen::{CompiledFunction, JitCompiler};
pub use compile::{
    Backend, CompileConfig, CompileStats, CompiledModel, LazyCompiled, Mode, compile_fn,
    compile_fn_with_config, compile_graph, compile_graph_with_config,
};
pub use error::{JitError, JitResult};
pub use ir::{DataType, Graph, Node, NodeId, Op, Shape};
pub use optimize::{OptimizationPass, Optimizer};
pub use trace::{TracedValue, Tracer, trace};

// =============================================================================
// Tests
// =============================================================================

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
