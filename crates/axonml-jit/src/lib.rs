//! JIT Compilation for Axonml
//!
//! # File
//! `crates/axonml-jit/src/lib.rs`
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
    Backend, CompileConfig, CompileStats, CompiledModel, LazyCompiled, Mode, compile_fn,
    compile_fn_with_config, compile_graph, compile_graph_with_config,
};
pub use error::{JitError, JitResult};
pub use ir::{DataType, Graph, Node, NodeId, Op, Shape};
pub use optimize::{OptimizationPass, Optimizer};
pub use trace::{TracedValue, Tracer, trace};

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
