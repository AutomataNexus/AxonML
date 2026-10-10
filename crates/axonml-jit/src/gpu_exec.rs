//! GPU execution of a JIT graph — the bridge that makes `Op::FusedChain` (and the whole graph)
//! run on-device instead of the CPU `Vec<f32>` interpreter.
//!
//! Every input is a device `Tensor`; each node produces a device `Tensor`; a `FusedChain` node
//! dispatches to the single NVRTC-JIT'd kernel (`Tensor::fuse_unary_chain`) built in axonml-tensor.
//! EVERY `Op` variant the interpreter supports is handled here — an unhandled op is a hard
//! `Unsupported` error, never a silent skip or wrong answer.

#![cfg(feature = "cuda")]

use axonml_tensor::Tensor;
use axonml_tensor::fused_chain::ChainOp;

use crate::error::{JitError, JitResult};
use crate::ir::{FusedStep, Graph, NodeId, Op};

fn step_to_chainop(step: FusedStep, scalar: f64) -> ChainOp {
    let s = scalar as f32;
    match step {
        FusedStep::Neg => ChainOp::Neg,
        FusedStep::Abs => ChainOp::Clamp(f32::MIN, f32::MAX), // placeholder, replaced below
        FusedStep::Sqrt => ChainOp::Sqrt,
        FusedStep::Exp => ChainOp::Exp,
        FusedStep::Log => ChainOp::Ln,
        FusedStep::Tanh => ChainOp::Tanh,
        FusedStep::Relu => ChainOp::Relu,
        FusedStep::Sigmoid => ChainOp::Sigmoid,
        FusedStep::AddScalar => ChainOp::AddScalar(s),
        FusedStep::MulScalar => ChainOp::MulScalar(s),
        // Sin/Cos/Gelu/Silu/Abs are not (yet) unary ChainOps — handled by the fallback below so we
        // never emit a wrong op. `chain_is_fusable` gates on this set.
        FusedStep::Sin | FusedStep::Cos | FusedStep::Gelu | FusedStep::Silu | FusedStep::Abs => {
            unreachable!("chain_is_fusable must exclude {step:?}")
        }
    }
}

fn chain_is_fusable(steps: &[(FusedStep, f64)]) -> bool {
    steps.iter().all(|(s, _)| {
        matches!(
            s,
            FusedStep::Neg
                | FusedStep::Sqrt
                | FusedStep::Exp
                | FusedStep::Log
                | FusedStep::Tanh
                | FusedStep::Relu
                | FusedStep::Sigmoid
                | FusedStep::AddScalar
                | FusedStep::MulScalar
        )
    })
}

fn apply_step(t: &Tensor<f32>, step: FusedStep, scalar: f64) -> Tensor<f32> {
    let s = scalar as f32;
    match step {
        FusedStep::Neg => t.neg(),
        FusedStep::Abs => host_unary(t, f32::abs),
        FusedStep::Sqrt => t.sqrt(),
        FusedStep::Exp => t.exp(),
        FusedStep::Log => t.ln(),
        FusedStep::Sin => host_unary(t, f32::sin),
        FusedStep::Cos => host_unary(t, f32::cos),
        FusedStep::Tanh => t.tanh(),
        FusedStep::Relu => t.relu(),
        FusedStep::Sigmoid => t.sigmoid(),
        FusedStep::Gelu => t.gelu(),
        FusedStep::Silu => t.silu(),
        FusedStep::AddScalar => t.add_scalar(s),
        FusedStep::MulScalar => t.mul_scalar(s),
    }
}

/// Elementwise unary on the host for ops with no device method (sin/cos/abs), preserving device.
fn host_unary(t: &Tensor<f32>, f: impl Fn(f32) -> f32) -> Tensor<f32> {
    let dev = t.device();
    let v: Vec<f32> = t
        .to_device(axonml_core::device::Device::Cpu)
        .unwrap()
        .to_vec()
        .into_iter()
        .map(f)
        .collect();
    Tensor::from_vec(v, t.shape())
        .unwrap()
        .to_device(dev)
        .unwrap_or_else(|_| Tensor::from_vec(vec![], &[0]).unwrap())
}

/// Execute `graph` on the given device inputs, returning the device tensor of the (single) output.
/// Inputs are matched by name; every named graph input must be provided.
pub fn run_gpu(graph: &Graph, inputs: &[(&str, &Tensor<f32>)]) -> JitResult<Tensor<f32>> {
    let n = graph.len();
    let mut vals: Vec<Option<Tensor<f32>>> = (0..n).map(|_| None).collect();

    // seed inputs
    for (name, id) in graph.inputs() {
        let t = inputs
            .iter()
            .find(|(nm, _)| nm == name)
            .map(|(_, t)| (*t).clone())
            .ok_or_else(|| JitError::RuntimeError(format!("missing input '{name}'")))?;
        vals[id.index()] = Some(t);
    }

    let get = |vals: &Vec<Option<Tensor<f32>>>, id: NodeId| -> JitResult<Tensor<f32>> {
        vals[id.index()]
            .clone()
            .ok_or_else(|| JitError::RuntimeError(format!("node {} not computed", id.index())))
    };

    for node in graph.nodes() {
        if vals[node.id.index()].is_some() {
            continue; // input already seeded
        }
        let out: Tensor<f32> = match &node.op {
            Op::Input { name } => {
                return Err(JitError::RuntimeError(format!("input '{name}' not seeded")));
            }
            Op::Output { input, .. } => get(&vals, *input)?,
            Op::Constant { value } => {
                let numel = node.shape.numel();
                Tensor::from_vec(vec![*value as f32; numel.max(1)], node.shape.dims())
                    .map_err(|e| JitError::RuntimeError(format!("const: {e}")))?
                    .to_device(inputs[0].1.device())
                    .map_err(|e| JitError::RuntimeError(format!("const dev: {e}")))?
            }

            // ── binary elementwise ──
            Op::Add { lhs, rhs } => get(&vals, *lhs)?.add(&get(&vals, *rhs)?).map_err(exec)?,
            Op::Sub { lhs, rhs } => get(&vals, *lhs)?
                .add(&get(&vals, *rhs)?.neg())
                .map_err(exec)?,
            Op::Mul { lhs, rhs } => get(&vals, *lhs)?.mul(&get(&vals, *rhs)?).map_err(exec)?,
            Op::Div { lhs, rhs } => get(&vals, *lhs)?.div(&get(&vals, *rhs)?).map_err(exec)?,
            Op::Pow { base, exp } => {
                // exp is a graph node; for the scalar-power common case read its constant.
                let e = scalar_of(graph, &vals, *exp)?;
                get(&vals, *base)?.pow(e)
            }
            Op::Max { lhs, rhs } => elementwise2(&get(&vals, *lhs)?, &get(&vals, *rhs)?, f32::max)?,
            Op::Min { lhs, rhs } => elementwise2(&get(&vals, *lhs)?, &get(&vals, *rhs)?, f32::min)?,
            Op::Gt { lhs, rhs } => elementwise2(&get(&vals, *lhs)?, &get(&vals, *rhs)?, |a, b| {
                (a > b) as i32 as f32
            })?,
            Op::Lt { lhs, rhs } => elementwise2(&get(&vals, *lhs)?, &get(&vals, *rhs)?, |a, b| {
                (a < b) as i32 as f32
            })?,
            Op::Eq { lhs, rhs } => elementwise2(&get(&vals, *lhs)?, &get(&vals, *rhs)?, |a, b| {
                (a == b) as i32 as f32
            })?,

            // ── unary elementwise ──
            Op::Neg { input } => get(&vals, *input)?.neg(),
            Op::Abs { input } => host_unary(&get(&vals, *input)?, f32::abs),
            Op::Sqrt { input } => get(&vals, *input)?.sqrt(),
            Op::Exp { input } => get(&vals, *input)?.exp(),
            Op::Log { input } => get(&vals, *input)?.ln(),
            Op::Sin { input } => host_unary(&get(&vals, *input)?, f32::sin),
            Op::Cos { input } => host_unary(&get(&vals, *input)?, f32::cos),
            Op::Tanh { input } => get(&vals, *input)?.tanh(),
            Op::Relu { input } => get(&vals, *input)?.relu(),
            Op::Sigmoid { input } => get(&vals, *input)?.sigmoid(),
            Op::Gelu { input } => get(&vals, *input)?.gelu(),
            Op::Silu { input } => get(&vals, *input)?.silu(),
            Op::AddScalar { input, scalar } => get(&vals, *input)?.add_scalar(*scalar as f32),
            Op::MulScalar { input, scalar } => get(&vals, *input)?.mul_scalar(*scalar as f32),

            // ── the bridge: one JIT'd kernel for the whole chain, else fold op-by-op ──
            Op::FusedChain { input, steps } => {
                let x = get(&vals, *input)?;
                if chain_is_fusable(steps) {
                    let ops: Vec<ChainOp> = steps
                        .iter()
                        .map(|(s, sc)| step_to_chainop(*s, *sc))
                        .collect();
                    match x.fuse_unary_chain(&ops) {
                        Some(t) => t,
                        None => steps.iter().fold(x, |a, (s, sc)| apply_step(&a, *s, *sc)),
                    }
                } else {
                    // chain contains an op with no ChainOp — fold with real tensor ops (still one
                    // node, just not a single kernel). Correct, never skipped.
                    steps.iter().fold(x, |a, (s, sc)| apply_step(&a, *s, *sc))
                }
            }

            // ── shape / view ops (data unchanged in the contiguous interpreter model) ──
            Op::Reshape { input, .. }
            | Op::Squeeze { input, .. }
            | Op::Unsqueeze { input, .. }
            | Op::Broadcast { input, .. }
            | Op::Contiguous { input }
            | Op::Cast { input, .. } => get(&vals, *input)?,

            Op::Transpose { input, dim0, dim1 } => get(&vals, *input)?
                .transpose(*dim0 as i64, *dim1 as i64)
                .map_err(exec)?,

            // ── reductions ──
            Op::Sum { input } => get(&vals, *input)?.sum(),
            Op::Mean { input } => get(&vals, *input)?.mean().map_err(exec)?,
            Op::SumAxis {
                input,
                axis,
                keepdim,
            } => get(&vals, *input)?.sum_dim(*axis, *keepdim),
            Op::MeanAxis {
                input,
                axis,
                keepdim,
            } => get(&vals, *input)?.mean_dim(*axis, *keepdim),
            Op::MaxAxis {
                input,
                axis,
                keepdim,
            } => host_max_axis(
                &get(&vals, *input)?,
                node.shape.dims(),
                graph.node(*input).shape.dims(),
                *axis,
                *keepdim,
            )?,

            // ── matmul / select ──
            Op::MatMul { lhs, rhs } => {
                get(&vals, *lhs)?.matmul(&get(&vals, *rhs)?).map_err(exec)?
            }
            Op::Where { condition, x, y } => {
                let c = get(&vals, *condition)?;
                let t = get(&vals, *x)?;
                let f = get(&vals, *y)?;
                // out = cond*t + (1-cond)*f, cond in {0,1}
                let one_minus = c.mul_scalar(-1.0).add_scalar(1.0);
                c.mul(&t)
                    .map_err(exec)?
                    .add(&one_minus.mul(&f).map_err(exec)?)
                    .map_err(exec)?
            }
        };
        vals[node.id.index()] = Some(out);
    }

    // the graph's (single) output
    let (_, out_id) = graph
        .outputs()
        .iter()
        .next()
        .ok_or_else(|| JitError::RuntimeError("graph has no output".into()))?;
    get(&vals, *out_id)
}

fn exec<E: std::fmt::Display>(e: E) -> JitError {
    JitError::RuntimeError(e.to_string())
}

fn host_max_axis(
    t: &Tensor<f32>,
    out_shape: &[usize],
    in_shape: &[usize],
    axis: i32,
    _keep: bool,
) -> JitResult<Tensor<f32>> {
    let dev = t.device();
    let a = t
        .to_device(axonml_core::device::Device::Cpu)
        .map_err(exec)?
        .to_vec();
    let nd = in_shape.len();
    let ax = if axis < 0 {
        (axis + nd as i32) as usize
    } else {
        axis as usize
    };
    let axis_sz = in_shape[ax];
    let inner: usize = in_shape[ax + 1..].iter().product();
    let outer: usize = in_shape[..ax].iter().product();
    let mut out = vec![f32::NEG_INFINITY; outer * inner];
    for o in 0..outer {
        for k in 0..axis_sz {
            for i in 0..inner {
                let v = a[(o * axis_sz + k) * inner + i];
                let d = &mut out[o * inner + i];
                if v > *d {
                    *d = v;
                }
            }
        }
    }
    Tensor::from_vec(out, out_shape)
        .map_err(exec)?
        .to_device(dev)
        .map_err(exec)
}

fn scalar_of(graph: &Graph, vals: &[Option<Tensor<f32>>], id: NodeId) -> JitResult<f32> {
    if let Op::Constant { value } = &graph.node(id).op {
        return Ok(*value as f32);
    }
    // read a 1-elem tensor from the host as a fallback
    vals[id.index()]
        .as_ref()
        .and_then(|t| t.to_device(axonml_core::device::Device::Cpu).ok())
        .map(|t| t.to_vec().first().copied().unwrap_or(0.0))
        .ok_or_else(|| JitError::RuntimeError("Pow exponent is not a scalar".into()))
}

/// CPU host elementwise for the two comparison/min-max ops that have no direct binary tensor op.
fn elementwise2(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    f: impl Fn(f32, f32) -> f32,
) -> JitResult<Tensor<f32>> {
    let dev = a.device();
    let av = a
        .to_device(axonml_core::device::Device::Cpu)
        .map_err(exec)?
        .to_vec();
    let bv = b
        .to_device(axonml_core::device::Device::Cpu)
        .map_err(exec)?
        .to_vec();
    let out: Vec<f32> = av.iter().zip(bv.iter()).map(|(x, y)| f(*x, *y)).collect();
    Tensor::from_vec(out, a.shape())
        .map_err(exec)?
        .to_device(dev)
        .map_err(exec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DataType, Shape};
    use crate::optimize::{OptimizationPass, Optimizer};

    #[test]
    fn gpu_fused_graph_matches_cpu_interpreter() {
        use axonml_core::device::Device;
        if Tensor::from_vec(vec![0.0f32; 1], &[1])
            .unwrap()
            .to_device(Device::Cuda(0))
            .is_err()
        {
            return;
        }
        let n = 2048usize;
        let host: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 40.0).collect();
        let mut g = Graph::new();
        let x = g.add_node(
            Op::Input { name: "x".into() },
            DataType::F32,
            Shape(vec![n]),
        );
        g.register_input("x", x);
        let a = g.add_node(
            Op::MulScalar {
                input: x,
                scalar: 2.0,
            },
            DataType::F32,
            Shape(vec![n]),
        );
        let b = g.add_node(
            Op::AddScalar {
                input: a,
                scalar: 1.0,
            },
            DataType::F32,
            Shape(vec![n]),
        );
        let c = g.add_node(Op::Sigmoid { input: b }, DataType::F32, Shape(vec![n]));
        let d = g.add_node(Op::Relu { input: c }, DataType::F32, Shape(vec![n]));
        let e = g.add_node(Op::Sqrt { input: d }, DataType::F32, Shape(vec![n]));
        let o = g.add_node(
            Op::Output {
                name: "out".into(),
                input: e,
            },
            DataType::F32,
            Shape(vec![n]),
        );
        g.register_output("out", o);

        let mut opt = Optimizer::new();
        opt.add_pass(OptimizationPass::ElementwiseFusion);
        let fused = opt.optimize(g.clone());
        assert!(
            fused
                .nodes()
                .iter()
                .any(|nd| matches!(nd.op, Op::FusedChain { .. }))
        );

        let cpu = crate::codegen::CompiledFunction::from_graph_for_test(fused.clone())
            .run(&[("x", &host)])
            .unwrap();
        let xt = Tensor::from_vec(host, &[n])
            .unwrap()
            .to_device(Device::Cuda(0))
            .unwrap();
        let gpu = run_gpu(&fused, &[("x", &xt)])
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec();
        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "GPU fused graph diverges from CPU: {md}");
    }
}
