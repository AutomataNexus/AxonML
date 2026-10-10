//! Graph tracer — composes a `BundleGraph` from a live model so every `.axonml`
//! can carry its exact compute graph instead of a weights-only, arch-tag-only blob.
//!
//! The module tree (`Module::named_children` + `Module::describe`) supplies forward
//! op types, attributes, and initializer binding; a forward pass supplies the true
//! input/output tensor shapes. Feedforward stacks chain linearly; skip topology is a
//! follow-on refinement over the autograd DAG.
//!
//! Copyright (c) 2026 Andrew Jewell Sr. — AutomataNexus LLC.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use axonml_autograd::{Variable, inspect};
use axonml_nn::{AttrVal, Module, NodeSpec};
use axonml_tensor::Tensor;

use crate::bundle::{BundleGraph, BundleResult, GraphIo, GraphNode, NamedTensor};

// ── module-tree flattening ──
fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

fn flatten(module: &dyn Module, prefix: &str, out: &mut Vec<(String, NodeSpec)>) {
    let children = module.named_children();
    if children.is_empty() {
        for (i, mut spec) in module.describe().into_iter().enumerate() {
            spec.params = spec.params.iter().map(|p| qualify(prefix, p)).collect();
            let name = if prefix.is_empty() {
                format!("node{i}")
            } else {
                format!("{prefix}.{i}")
            };
            out.push((name, spec));
        }
    } else {
        for (cname, child) in children {
            flatten(child, &qualify(prefix, &cname), out);
        }
    }
}

// ── attribute + tensor lowering ──
fn attrs_to_json(attrs: &[(String, AttrVal)]) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    for (k, v) in attrs {
        let jv = match v {
            AttrVal::Int(i) => serde_json::json!(i),
            AttrVal::Float(f) => serde_json::json!(f),
            AttrVal::Ints(vs) => serde_json::json!(vs),
            AttrVal::Str(s) => serde_json::json!(s),
            AttrVal::Bool(b) => serde_json::json!(b),
        };
        m.insert(k.clone(), jv);
    }
    serde_json::Value::Object(m)
}

fn tensor_to_named(t: &Tensor<f32>) -> NamedTensor {
    NamedTensor {
        shape: t.shape().iter().map(|&d| d as i64).collect(),
        dtype: "f32".to_string(),
        data: t.to_vec(),
    }
}

fn dyn_batch(shape: &[usize]) -> Vec<i64> {
    let mut s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    if !s.is_empty() {
        s[0] = -1;
    }
    s
}

// ── backward grad-fn name -> forward op name ──
fn fwd_name(backward: &str) -> String {
    let base = backward.strip_suffix("Backward").unwrap_or(backward);
    let mapped = match base {
        "Silu" => "SiLU",
        "Cat" => "Concat",
        "View" => "Reshape",
        "Narrow" | "Select" => "Slice",
        "Conv2d" => "Conv2d",
        "BatchNorm2d" | "BatchNorm1d" => "BatchNorm",
        "MaxPool2d" => "MaxPool",
        "AvgPool2d" => "AvgPool",
        "AdaptiveAvgPool2d" => "GlobalAvgPool",
        other => other,
    };
    mapped.to_string()
}

fn is_fused_attr_op(op: &str) -> bool {
    matches!(
        op,
        "Conv2d" | "DepthwiseConv2d" | "BatchNorm" | "MaxPool" | "AvgPool"
    )
}

fn compatible(dag_fwd: &str, module_op: &str) -> bool {
    if dag_fwd == module_op {
        return true;
    }
    let conv_dag = matches!(
        dag_fwd,
        "Conv2d" | "GroupedConv2d" | "DepthwiseConv2d" | "Conv1d"
    );
    let conv_mod = matches!(module_op, "Conv2d" | "DepthwiseConv2d" | "Conv1d");
    conv_dag && conv_mod
}

// ── public tracer: DAG wiring (skips included) + module attrs for fused ops ──
/// Traces `model`'s forward on `example` into a bundle graph: the module DAG with fused-op attrs.
pub fn trace_graph<M: Module>(model: &M, example: &Tensor<f32>) -> BundleGraph {
    let mut flat: Vec<(String, NodeSpec)> = Vec::new();
    flatten(model, "", &mut flat);

    let mut param_to_node: HashMap<String, usize> = HashMap::new();
    let mut op_queues: HashMap<String, VecDeque<usize>> = HashMap::new();
    for (idx, (_, spec)) in flat.iter().enumerate() {
        for p in &spec.params {
            param_to_node.insert(p.clone(), idx);
        }
        op_queues.entry(spec.op.clone()).or_default().push_back(idx);
    }

    let params = model.named_parameters();
    let mut id_to_param: HashMap<usize, String> = HashMap::new();
    for (name, p) in &params {
        if let Some(gf) = p.variable().grad_fn() {
            id_to_param.insert(gf.id(), name.clone());
        }
    }

    let input_shape = dyn_batch(example.shape());
    let output = model.forward(&Variable::new(example.clone(), true));
    let output_shape = dyn_batch(&output.shape());
    let snap = inspect::trace_backward(&output);

    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for (from, to) in &snap.edges {
        children.entry(*from).or_default().push(*to);
    }
    let is_leaf: Vec<bool> = snap.nodes.iter().map(|n| n.is_leaf).collect();
    let id_of: Vec<usize> = snap.nodes.iter().map(|n| n.id).collect();

    let tensor_name = |idx: usize| -> String {
        let node_id = id_of[idx];
        if is_leaf[idx] {
            id_to_param
                .get(&node_id)
                .cloned()
                .unwrap_or_else(|| "input".to_string())
        } else {
            format!("t{node_id}")
        }
    };

    let order = topo_order(snap.nodes.len(), &children);
    let mut nodes = Vec::new();
    for idx in order {
        if is_leaf[idx] {
            continue;
        }
        let kids = children.get(&idx).cloned().unwrap_or_default();
        let mut act_inputs = Vec::new();
        let mut param_names = Vec::new();
        for c in &kids {
            if is_leaf[*c] {
                match id_to_param.get(&id_of[*c]) {
                    Some(p) => param_names.push(p.clone()),
                    None => act_inputs.push(tensor_name(*c)),
                }
            } else {
                act_inputs.push(tensor_name(*c));
            }
        }

        let fwd = fwd_name(&snap.nodes[idx].name);
        let bound = param_names.iter().find_map(|p| {
            param_to_node
                .get(p)
                .copied()
                .filter(|&fi| compatible(&fwd, &flat[fi].1.op))
        });
        let (op, attrs, inputs) = if let Some(fi) = bound {
            let spec = &flat[fi].1;
            let mut inp = act_inputs.clone();
            inp.extend(spec.params.iter().cloned());
            (spec.op.clone(), attrs_to_json(&spec.attrs), inp)
        } else if is_fused_attr_op(&fwd) {
            let attrs = op_queues
                .get_mut(&fwd)
                .and_then(|q| q.pop_front())
                .map_or(serde_json::Value::Null, |fi| {
                    attrs_to_json(&flat[fi].1.attrs)
                });
            (fwd, attrs, act_inputs)
        } else {
            let mut inp = act_inputs.clone();
            inp.extend(param_names.clone());
            (fwd, serde_json::Value::Null, inp)
        };

        nodes.push(GraphNode {
            name: format!("n{}", id_of[idx]),
            op,
            attrs,
            inputs,
            outputs: vec![format!("t{}", id_of[idx])],
        });
    }

    let out_name = if snap.nodes.is_empty() {
        "input".to_string()
    } else {
        tensor_name(0)
    };

    let mut initializers = HashMap::new();
    for (n, p) in &params {
        initializers.insert(n.clone(), tensor_to_named(&p.data()));
    }
    for (n, t) in model.named_buffers() {
        initializers.insert(n, tensor_to_named(&t));
    }

    BundleGraph {
        inputs: vec![GraphIo {
            name: "input".to_string(),
            shape: input_shape,
            dtype: "f32".to_string(),
        }],
        outputs: vec![GraphIo {
            name: out_name,
            shape: output_shape,
            dtype: "f32".to_string(),
        }],
        nodes,
        initializers,
    }
}

// ── topological order (inputs before consumers) over the reversed backward edges ──
fn topo_order(n: usize, children: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
    let mut state = vec![0u8; n];
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for start in 0..n {
        if state[start] != 0 {
            continue;
        }
        stack.push((start, 0));
        while let Some((node, ci)) = stack.pop() {
            if ci == 0 {
                if state[node] == 2 {
                    continue;
                }
                state[node] = 1;
            }
            let kids = children.get(&node).map_or(&[][..], |v| v.as_slice());
            if ci < kids.len() {
                stack.push((node, ci + 1));
                let c = kids[ci];
                if state[c] == 0 {
                    stack.push((c, 0));
                }
            } else if state[node] != 2 {
                state[node] = 2;
                order.push(node);
            }
        }
    }
    order
}

// ── save with graph attached ──
/// Saves `model` as a bundle with its traced graph attached (see [`trace_graph`]).
pub fn save_model_with_graph<M: Module, P: AsRef<Path>>(
    model: &M,
    example: &Tensor<f32>,
    architecture: &str,
    path: P,
) -> BundleResult<std::path::PathBuf> {
    let graph = trace_graph(model, example);
    let input_features: usize = example.shape().iter().skip(1).product();
    let mut weights: Vec<f32> = Vec::new();
    for p in model.parameters() {
        weights.extend(p.data().to_vec());
    }
    let bundle =
        crate::bundle::ModelBundle::new(architecture, input_features, weights).with_graph(graph);
    crate::bundle::save_bundle(&bundle, path)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_nn::{BatchNorm2d, Conv2d, Flatten, Linear, ReLU, Sequential};

    fn cnn() -> Sequential {
        Sequential::new()
            .add(Conv2d::new(3, 8, 3))
            .add(BatchNorm2d::new(8))
            .add(ReLU::new())
            .add(Flatten::new())
            .add(Linear::new(8 * 6 * 6, 4))
    }

    #[test]
    fn traces_cnn_ops_io_and_initializers() {
        let model = cnn();
        let example = Tensor::from_vec(vec![0.1f32; 3 * 8 * 8], &[1, 3, 8, 8]).expect("tensor");
        let g = trace_graph(&model, &example);

        let ops: Vec<&str> = g.nodes.iter().map(|n| n.op.as_str()).collect();
        assert!(ops.contains(&"Conv2d"));
        assert!(ops.contains(&"BatchNorm"));
        assert!(ops.contains(&"Relu"));
        assert!(!ops.iter().any(|o| *o == "Gemm"));

        assert_eq!(g.inputs[0].shape, vec![-1, 3, 8, 8]);
        assert_eq!(g.outputs[0].shape, vec![-1, 4]);

        assert!(g.initializers.contains_key("0.weight"));
        assert!(g.initializers.keys().any(|k| k.contains("running_mean")));

        let conv = g
            .nodes
            .iter()
            .find(|n| n.op == "Conv2d")
            .expect("conv node");
        assert_eq!(conv.inputs[0], "input");
        assert!(conv.inputs.iter().any(|i| i == "0.weight"));
        assert!(conv.inputs.iter().any(|i| i == "0.bias"));
        let bn = g
            .nodes
            .iter()
            .find(|n| n.op == "BatchNorm")
            .expect("bn node");
        assert!(bn.inputs.iter().any(|i| i == "1.running_mean"));
    }

    #[test]
    fn traces_residual_skip_edge() {
        use axonml_nn::{Conv2d, ResidualBlock};
        let main = Sequential::new().add(Conv2d::new(4, 4, 1)).add(ReLU::new());
        let block = ResidualBlock::new(main);
        let example = Tensor::from_vec(vec![0.1f32; 4 * 6 * 6], &[1, 4, 6, 6]).expect("tensor");
        let g = trace_graph(&block, &example);

        let ops: Vec<&str> = g.nodes.iter().map(|n| n.op.as_str()).collect();
        assert!(ops.contains(&"Conv2d"));
        let add = g
            .nodes
            .iter()
            .find(|n| n.op == "Add")
            .expect("residual Add node");
        let act_inputs: Vec<&String> = add
            .inputs
            .iter()
            .filter(|i| i.starts_with('t') || *i == "input")
            .collect();
        assert!(
            act_inputs.len() >= 2,
            "residual Add must fuse two activation paths, got {:?}",
            add.inputs
        );
    }

    #[test]
    fn save_load_roundtrips_graph() {
        let model = cnn();
        let example = Tensor::from_vec(vec![0.05f32; 3 * 8 * 8], &[1, 3, 8, 8]).expect("tensor");
        let path = std::env::temp_dir().join("axonml_graph_trace_rt.axonml");
        save_model_with_graph(&model, &example, "cnn", &path).expect("save");
        let (_h, bundle) = crate::bundle::load_bundle(&path).expect("load");
        let g = bundle.graph.expect("bundle must carry the graph");
        assert!(g.nodes.iter().any(|n| n.op == "Conv2d"));
        assert!(g.nodes.iter().any(|n| n.op == "MatMul"));
        assert!(!g.initializers.is_empty());
        std::fs::remove_file(&path).ok();
    }
}
