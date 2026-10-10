//! Capture bridge: run the model's real forward once, fuse it, and get a callable that runs the
//! fused graph on GPU tensors. This is the seam that makes the whole JIT pipeline usable end to end
//! — `trace_forward -> ElementwiseFusion -> run_gpu` behind a single `JitFn`, with the fused graph
//! built once and reused on every call. There is no tracer DSL: you pass the actual `forward`.
//!
//! ```ignore
//! let f = JitFn::trace_forward(&[("x", &x_var)], |v| model.forward(&v[0]));
//! let y = f.run_gpu(&[("x", &x_tensor)])?;   // one fused GPU kernel for the chain
//! ```

use axonml_autograd::{Variable, capture as autograd_capture};

use crate::ir::Graph;
use crate::lower::lower_recording;
use crate::optimize::{OptimizationPass, Optimizer};

/// A captured, fused forward. Holds the optimized graph; dispatches via the CPU interpreter or the
/// GPU executor.
pub struct JitFn {
    graph: Graph,
    input_names: Vec<String>,
}

impl JitFn {
    /// Names of the inputs the captured forward was recorded with, in order.
    #[must_use]
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Capture the model's REAL forward and fuse it — no second declaration, no DSL.
    ///
    /// You pass the actual `forward` closure and the input variables it runs on. The recorder
    /// observes exactly the elementwise ops that execute (with their scalar operands intact) and
    /// lowers them to a fused graph. This is the automatic path: the same code that trains is the
    /// code that gets captured.
    ///
    /// ```ignore
    /// // `model.forward` is unchanged — we just run it under capture.
    /// let f = JitFn::trace_forward(&[("x", &x_var)], |v| model.forward(&v[0]));
    /// let y = f.run_gpu(&[("x", &x_tensor)])?;
    /// ```
    #[must_use]
    pub fn trace_forward<F>(inputs: &[(&str, &Variable)], forward: F) -> Self
    where
        F: FnOnce(&[Variable]) -> Variable,
    {
        let vars: Vec<Variable> = inputs.iter().map(|(_, v)| (*v).clone()).collect();
        let (rec, out_id) = autograd_capture(|| forward(&vars));

        let named: Vec<(String, u64)> = inputs
            .iter()
            .zip(vars.iter())
            .filter_map(|((name, _), v)| v.node_id().map(|id| ((*name).to_string(), id)))
            .collect();
        let shape = vars.first().map_or_else(Vec::new, Variable::shape);
        let out_id = out_id.expect("captured forward produced an untracked output");

        let graph = lower_recording(&rec, &named, out_id, &shape);
        let mut opt = Optimizer::new();
        opt.add_pass(OptimizationPass::ElementwiseFusion);
        let graph = opt.optimize(graph);

        Self {
            input_names: inputs.iter().map(|(n, _)| (*n).to_string()).collect(),
            graph,
        }
    }

    /// The fused graph (for inspection / testing).
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Number of `FusedChain` nodes in the fused graph.
    #[must_use]
    pub fn fused_chain_count(&self) -> usize {
        self.graph
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, crate::ir::Op::FusedChain { .. }))
            .count()
    }

    /// Run on the CPU interpreter (reference / no-GPU path). `inputs` matched by name.
    pub fn run_cpu(&self, inputs: &[(&str, &[f32])]) -> crate::error::JitResult<Vec<f32>> {
        crate::codegen::CompiledFunction::from_graph_for_test(self.graph.clone()).run(inputs)
    }

    /// Run the fused graph on GPU tensors, dispatching each FusedChain to its JIT kernel.
    #[cfg(feature = "cuda")]
    pub fn run_gpu(
        &self,
        inputs: &[(&str, &axonml_tensor::Tensor<f32>)],
    ) -> crate::error::JitResult<axonml_tensor::Tensor<f32>> {
        for name in &self.input_names {
            if !inputs.iter().any(|(n, _)| n == name) {
                return Err(crate::error::JitError::RuntimeError(format!(
                    "JitFn::run_gpu missing input '{name}'"
                )));
            }
        }
        crate::gpu_exec::run_gpu(&self.graph, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_forward_captures_real_ops_and_fuses() {
        // No re-declaration: `body` is written in ordinary Variable ops, exactly as a model's
        // forward would be. trace_forward runs it under the recorder and fuses what actually ran.
        use axonml_autograd::Variable;
        use axonml_tensor::Tensor;
        let host = vec![-1.0f32, 0.5, 2.0, -3.0, 4.0, 0.1, -0.2, 9.0];
        let x = Variable::new(Tensor::from_vec(host.clone(), &[8]).unwrap(), true);
        let f = JitFn::trace_forward(&[("x", &x)], |v| {
            v[0].mul_scalar(2.0).add_scalar(1.0).sigmoid().relu().sqrt()
        });
        assert!(
            f.fused_chain_count() >= 1,
            "real forward must fuse into a chain"
        );
        let got = f.run_cpu(&[("x", &host)]).unwrap();
        let want: Vec<f32> = host
            .iter()
            .map(|v| {
                let s = 1.0 / (1.0 + (-(v * 2.0 + 1.0)).exp());
                s.max(0.0).sqrt()
            })
            .collect();
        let md = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-6, "captured real forward diverges: {md}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn run_gpu_matches_run_cpu() {
        use axonml_core::device::Device;
        use axonml_tensor::Tensor;
        if Tensor::from_vec(vec![0.0f32; 1], &[1])
            .unwrap()
            .to_device(Device::Cuda(0))
            .is_err()
        {
            return;
        }
        let n = 1024usize;
        let host: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 40.0).collect();
        let x = Variable::new(Tensor::from_vec(host.clone(), &[n]).unwrap(), true);
        let f = JitFn::trace_forward(&[("x", &x)], |v| {
            v[0].mul_scalar(2.0).add_scalar(1.0).sigmoid().relu().sqrt()
        });
        let cpu = f.run_cpu(&[("x", &host)]).unwrap();
        let xt = Tensor::from_vec(host, &[n])
            .unwrap()
            .to_device(Device::Cuda(0))
            .unwrap();
        let gpu = f
            .run_gpu(&[("x", &xt)])
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec();
        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "gpu vs cpu diverges: {md}");
    }
}
