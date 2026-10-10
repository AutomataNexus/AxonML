//! Lower a captured forward ([`axonml_autograd::Recording`]) into the JIT `Graph` IR.
//!
//! The recorder in autograd observes the model's real `forward` as it runs and yields an ordered
//! list of fusable elementwise ops keyed by autograd node id. Here we translate that, one-to-one, to
//! the JIT graph so the existing `ElementwiseFusion` pass can collapse the chain into a single kernel.
//! Named inputs map to `Op::Input`; any input id with no producing op is a fusion boundary (a matmul
//! output, a parameter, …) and becomes a synthesized leaf input. Because the recorded region is
//! elementwise, every node shares the output shape.

use std::collections::HashMap;

use axonml_autograd::{Recording, TraceKind};

use crate::ir::{DataType, Graph, NodeId, Op, Shape};

fn ensure_leaf(
    g: &mut Graph,
    map: &mut HashMap<u64, NodeId>,
    ag: u64,
    dtype: DataType,
    sh: &Shape,
) -> NodeId {
    if let Some(&n) = map.get(&ag) {
        return n;
    }
    let name = format!("__leaf{ag}");
    let id = g.add_node(Op::Input { name: name.clone() }, dtype, sh.clone());
    g.register_input(&name, id);
    map.insert(ag, id);
    id
}

/// Build a JIT graph from a recording. `inputs` maps each named input to its autograd leaf node id;
/// `output_id` is the captured output's node id; `shape` is the (shared, elementwise) tensor shape.
#[must_use]
pub fn lower_recording(
    rec: &Recording,
    inputs: &[(String, u64)],
    output_id: u64,
    shape: &[usize],
) -> Graph {
    let mut g = Graph::new();
    let dtype = DataType::F32;
    let sh = Shape::new(shape);
    let mut map: HashMap<u64, NodeId> = HashMap::new();

    for (name, ag_id) in inputs {
        let id = g.add_node(Op::Input { name: name.clone() }, dtype, sh.clone());
        g.register_input(name, id);
        map.insert(*ag_id, id);
    }

    for rop in &rec.ops {
        let a = ensure_leaf(&mut g, &mut map, rop.inputs[0], dtype, &sh);
        let b = if rop.kind.arity() == 2 {
            ensure_leaf(&mut g, &mut map, rop.inputs[1], dtype, &sh)
        } else {
            a
        };
        let op = match rop.kind {
            TraceKind::AddScalar(s) => Op::AddScalar {
                input: a,
                scalar: f64::from(s),
            },
            TraceKind::MulScalar(s) => Op::MulScalar {
                input: a,
                scalar: f64::from(s),
            },
            TraceKind::SubScalar(s) => Op::AddScalar {
                input: a,
                scalar: f64::from(-s),
            },
            TraceKind::Neg => Op::Neg { input: a },
            TraceKind::Square => Op::Mul { lhs: a, rhs: a },
            TraceKind::Sqrt => Op::Sqrt { input: a },
            TraceKind::Exp => Op::Exp { input: a },
            TraceKind::Ln => Op::Log { input: a },
            TraceKind::Relu => Op::Relu { input: a },
            TraceKind::Sigmoid => Op::Sigmoid { input: a },
            TraceKind::Tanh => Op::Tanh { input: a },
            TraceKind::AddTensor => Op::Add { lhs: a, rhs: b },
            TraceKind::MulTensor => Op::Mul { lhs: a, rhs: b },
            TraceKind::SubTensor => Op::Sub { lhs: a, rhs: b },
            TraceKind::Pow(e) => {
                let c = g.add_node(
                    Op::Constant {
                        value: f64::from(e),
                    },
                    dtype,
                    Shape::new(&[1]),
                );
                Op::Pow { base: a, exp: c }
            }
            TraceKind::Clamp(lo, hi) => {
                let clo = g.add_node(
                    Op::Constant {
                        value: f64::from(lo),
                    },
                    dtype,
                    sh.clone(),
                );
                let mx = g.add_node(Op::Max { lhs: a, rhs: clo }, dtype, sh.clone());
                let chi = g.add_node(
                    Op::Constant {
                        value: f64::from(hi),
                    },
                    dtype,
                    sh.clone(),
                );
                let id = g.add_node(Op::Min { lhs: mx, rhs: chi }, dtype, sh.clone());
                map.insert(rop.out, id);
                continue;
            }
        };
        let id = g.add_node(op, dtype, sh.clone());
        map.insert(rop.out, id);
    }

    let out = map
        .get(&output_id)
        .copied()
        .unwrap_or_else(|| ensure_leaf(&mut g, &mut map, output_id, dtype, &sh));
    let oid = g.add_node(
        Op::Output {
            name: "out".into(),
            input: out,
        },
        dtype,
        sh.clone(),
    );
    g.register_output("out", oid);
    g
}
