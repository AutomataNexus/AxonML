//! Convert an AxonML bundle (.axonml with embedded graph) to ONNX.
//!
//! Usage:
//!   cargo run --release --example bundle_to_onnx -p axonml-onnx -- \
//!       <bundle.axonml> <out.onnx>
//!
//! Walks the bundle's `BundleGraph` (per-node IrOp-named topology +
//! initializer tensors with explicit shapes) and emits an ONNX model via
//! axonml-onnx's `OnnxExporter`. The resulting ONNX is ingestible by ONNX Runtime and
//! downstream NPU compilers.
//!
//! Op-name mapping (BundleGraph node `op` field → ONNX op_type):
//!   Conv2d        → Conv
//!   BatchNorm     → BatchNormalization
//!   Relu / Sigmoid / Tanh / Add / Sub / Mul / Div / MatMul / Identity → same
//!   MaxPool       → MaxPool
//!   AvgPool       → AveragePool
//!   GlobalAvgPool → GlobalAveragePool
//!   Gemm          → Gemm
//!   Softmax       → Softmax
//!   Concat        → Concat
//!   Reshape       → Reshape

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use axonml_onnx::export::{AttributeValue, OnnxExporter, export_onnx};
use axonml_onnx::proto::TensorDataType;
use axonml_serialize::{BundleGraph, GraphNode, load_bundle};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bundle_to_onnx <bundle.axonml> <out.onnx>");
        std::process::exit(2);
    }
    let bundle_path = PathBuf::from(&args[1]);
    let onnx_path = PathBuf::from(&args[2]);

    let (header, bundle) = load_bundle(&bundle_path).expect("load_bundle failed");
    let graph: &BundleGraph = bundle
        .graph
        .as_ref()
        .expect("bundle has no `graph` field — cannot emit ONNX");

    eprintln!(
        "loaded bundle: arch={} nodes={} initializers={} inputs={} outputs={}",
        header.architecture,
        graph.nodes.len(),
        graph.initializers.len(),
        graph.inputs.len(),
        graph.outputs.len(),
    );

    let has_rnn = graph
        .nodes
        .iter()
        .any(|n| matches!(n.op.as_str(), "GRU" | "LSTM"));
    let mut exporter = OnnxExporter::new(&header.architecture)
        .with_producer("axonml-bundle-to-onnx", env!("CARGO_PKG_VERSION"))
        .with_opset(if has_rnn { 11 } else { 17 });

    // 1. Inputs
    for io in &graph.inputs {
        exporter.add_input(&io.name, &io.shape, TensorDataType::Float);
    }

    // 2. Outputs
    for io in &graph.outputs {
        exporter.add_output(&io.name, &io.shape, TensorDataType::Float);
    }

    // 3. Initializers — detect Reshape shape tensors and emit as int64
    let reshape_shape_names: HashSet<&str> = graph
        .nodes
        .iter()
        .filter(|n| n.op == "Reshape")
        .filter_map(|n| n.inputs.get(1).map(|s| s.as_str()))
        .collect();

    for (name, t) in &graph.initializers {
        if reshape_shape_names.contains(name.as_str()) {
            let int64_data: Vec<i64> = t.data.iter().map(|&v| v as i64).collect();
            exporter.add_initializer_int64(name, &t.shape, &int64_data);
        } else {
            exporter.add_initializer_data(name, &t.shape, &t.data);
        }
    }

    // 4. Compute nodes
    for (i, n) in graph.nodes.iter().enumerate() {
        // Slice: opset-13 takes starts/ends/axes/steps as int64 INPUT tensors, not attrs.
        if n.op == "Slice" {
            let get = |k: &str| -> Vec<i64> {
                n.attrs
                    .get(k)
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                    .unwrap_or_default()
            };
            let starts = get("starts");
            let ends = get("ends");
            let mut axes = get("axes");
            let mut steps = get("steps");
            if axes.is_empty() {
                axes = (0..starts.len() as i64).collect();
            }
            if steps.is_empty() {
                steps = vec![1i64; starts.len()];
            }
            let (sn, en, an, stn) = (
                format!("{}_starts", n.name),
                format!("{}_ends", n.name),
                format!("{}_axes", n.name),
                format!("{}_steps", n.name),
            );
            exporter.add_initializer_int64(&sn, &[starts.len() as i64], &starts);
            exporter.add_initializer_int64(&en, &[ends.len() as i64], &ends);
            exporter.add_initializer_int64(&an, &[axes.len() as i64], &axes);
            exporter.add_initializer_int64(&stn, &[steps.len() as i64], &steps);
            let in_refs: Vec<&str> = vec![n.inputs[0].as_str(), &sn, &en, &an, &stn];
            let out_refs: Vec<&str> = n.outputs.iter().map(|s| s.as_str()).collect();
            exporter.add_node("Slice", &in_refs, &out_refs, HashMap::new());
            continue;
        }
        let (op_type, attrs) = map_node_to_onnx(n).unwrap_or_else(|e| {
            panic!("node[{i}] `{}`: {}", n.name, e);
        });
        let in_refs: Vec<&str> = n.inputs.iter().map(|s| s.as_str()).collect();
        let out_refs: Vec<&str> = n.outputs.iter().map(|s| s.as_str()).collect();
        exporter.add_node(&op_type, &in_refs, &out_refs, attrs);
    }

    export_onnx(&exporter, &onnx_path).expect("export_onnx failed");

    let size = std::fs::metadata(&onnx_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("wrote ONNX: {} ({} bytes)", onnx_path.display(), size);
}

fn map_node_to_onnx(n: &GraphNode) -> Result<(String, HashMap<String, AttributeValue>), String> {
    use serde_json::Value;
    let mut attrs: HashMap<String, AttributeValue> = HashMap::new();

    let as_i64_vec = |v: &Value| -> Vec<i64> {
        v.as_array()
            .map(|a| a.iter().filter_map(|x| x.as_i64()).collect::<Vec<i64>>())
            .unwrap_or_default()
    };
    let as_i64 = |v: &Value| v.as_i64();
    let as_f32 = |v: &Value| v.as_f64().map(|x| x as f32);

    let onnx_op = match n.op.as_str() {
        "Conv" | "Conv2d" => {
            attrs.insert(
                "kernel_shape".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["kernel_shape"])),
            );
            attrs.insert(
                "strides".into(),
                AttributeValue::Ints(if n.attrs.get("strides").is_some() {
                    as_i64_vec(&n.attrs["strides"])
                } else {
                    vec![1]
                }),
            );
            let pads = if n.attrs.get("pads").map(|v| v.is_array()).unwrap_or(false) {
                as_i64_vec(&n.attrs["pads"])
            } else if n.attrs.get("padding").is_some() {
                as_i64_vec(&n.attrs["padding"])
            } else {
                vec![0, 0]
            };
            attrs.insert("pads".into(), AttributeValue::Ints(pads));
            if n.attrs.get("dilations").is_some() {
                attrs.insert(
                    "dilations".into(),
                    AttributeValue::Ints(as_i64_vec(&n.attrs["dilations"])),
                );
            }
            attrs.insert(
                "group".into(),
                AttributeValue::Int(
                    as_i64(&n.attrs.get("group").unwrap_or(&Value::Null)).unwrap_or(1),
                ),
            );
            "Conv"
        }
        "TransposedConv2d" => {
            attrs.insert(
                "kernel_shape".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["kernel_shape"])),
            );
            attrs.insert(
                "strides".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["strides"])),
            );
            let pads = if n.attrs["pads"].is_array() {
                as_i64_vec(&n.attrs["pads"])
            } else {
                as_i64_vec(&n.attrs["padding"])
            };
            attrs.insert("pads".into(), AttributeValue::Ints(pads));
            attrs.insert(
                "dilations".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["dilations"])),
            );
            attrs.insert(
                "group".into(),
                AttributeValue::Int(as_i64(&n.attrs["group"]).unwrap_or(1)),
            );
            "ConvTranspose"
        }
        "BatchNorm" | "BatchNormalization" => {
            attrs.insert(
                "epsilon".into(),
                AttributeValue::Float(as_f32(&n.attrs["epsilon"]).unwrap_or(1e-5)),
            );
            attrs.insert(
                "momentum".into(),
                AttributeValue::Float(as_f32(&n.attrs["momentum"]).unwrap_or(0.9)),
            );
            "BatchNormalization"
        }
        "Relu" => "Relu",
        "Sigmoid" => "Sigmoid",
        "Tanh" => "Tanh",
        "Unsqueeze" => "Unsqueeze",
        "Add" => "Add",
        "Sub" => "Sub",
        "Mul" => "Mul",
        "Div" => "Div",
        "MatMul" => "MatMul",
        "Identity" => "Identity",
        "MaxPool" => {
            attrs.insert(
                "kernel_shape".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["kernel_shape"])),
            );
            attrs.insert(
                "strides".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["strides"])),
            );
            attrs.insert(
                "pads".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["pads"])),
            );
            "MaxPool"
        }
        "AvgPool" => {
            attrs.insert(
                "kernel_shape".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["kernel_shape"])),
            );
            attrs.insert(
                "strides".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["strides"])),
            );
            attrs.insert(
                "pads".into(),
                AttributeValue::Ints(as_i64_vec(&n.attrs["pads"])),
            );
            "AveragePool"
        }
        "GlobalAvgPool" => "GlobalAveragePool",
        "Gemm" => {
            attrs.insert(
                "alpha".into(),
                AttributeValue::Float(as_f32(&n.attrs["alpha"]).unwrap_or(1.0)),
            );
            attrs.insert(
                "beta".into(),
                AttributeValue::Float(as_f32(&n.attrs["beta"]).unwrap_or(1.0)),
            );
            let ta = n.attrs.get("transA").or_else(|| n.attrs.get("trans_a"));
            let tb = n.attrs.get("transB").or_else(|| n.attrs.get("trans_b"));
            let flag = |v: Option<&Value>| -> i64 {
                v.map(|x| x.as_bool().unwrap_or_else(|| x.as_i64().unwrap_or(0) != 0))
                    .unwrap_or(false) as i64
            };
            attrs.insert("transA".into(), AttributeValue::Int(flag(ta)));
            attrs.insert("transB".into(), AttributeValue::Int(flag(tb)));
            "Gemm"
        }
        "Softmax" => {
            attrs.insert(
                "axis".into(),
                AttributeValue::Int(as_i64(&n.attrs["axis"]).unwrap_or(-1)),
            );
            "Softmax"
        }
        "Concat" => {
            attrs.insert(
                "axis".into(),
                AttributeValue::Int(as_i64(&n.attrs["axis"]).unwrap_or(0)),
            );
            "Concat"
        }
        "Reshape" => "Reshape",
        "Transpose" => {
            if let Some(perm) = n.attrs.get("perm") {
                attrs.insert("perm".into(), AttributeValue::Ints(as_i64_vec(perm)));
            }
            "Transpose"
        }
        "Squeeze" => {
            // Axes travel as an attribute; the exporter handles opset compatibility.
            if let Some(axes) = n.attrs.get("axes") {
                attrs.insert("axes".into(), AttributeValue::Ints(as_i64_vec(axes)));
            }
            "Squeeze"
        }
        "Slice" => {
            if let Some(v) = n.attrs.get("starts") {
                attrs.insert("starts".into(), AttributeValue::Ints(as_i64_vec(v)));
            }
            if let Some(v) = n.attrs.get("ends") {
                attrs.insert("ends".into(), AttributeValue::Ints(as_i64_vec(v)));
            }
            if let Some(v) = n.attrs.get("axes") {
                attrs.insert("axes".into(), AttributeValue::Ints(as_i64_vec(v)));
            }
            "Slice"
        }
        "GRU" => {
            attrs.insert(
                "hidden_size".into(),
                AttributeValue::Int(as_i64(&n.attrs["hidden_size"]).unwrap_or(64)),
            );
            if let Some(dir) = n.attrs.get("direction") {
                attrs.insert(
                    "direction".into(),
                    AttributeValue::String(dir.as_str().unwrap_or("forward").to_string()),
                );
            }
            attrs.insert(
                "linear_before_reset".into(),
                AttributeValue::Int(as_i64(&n.attrs["linear_before_reset"]).unwrap_or(0)),
            );
            "GRU"
        }
        "LSTM" => {
            attrs.insert(
                "hidden_size".into(),
                AttributeValue::Int(as_i64(&n.attrs["hidden_size"]).unwrap_or(64)),
            );
            if let Some(dir) = n.attrs.get("direction") {
                attrs.insert(
                    "direction".into(),
                    AttributeValue::String(dir.as_str().unwrap_or("forward").to_string()),
                );
            }
            "LSTM"
        }
        "Flatten" => {
            attrs.insert(
                "axis".into(),
                AttributeValue::Int(as_i64(&n.attrs["axis"]).unwrap_or(1)),
            );
            "Flatten"
        }
        "Gather" => {
            attrs.insert(
                "axis".into(),
                AttributeValue::Int(as_i64(&n.attrs["axis"]).unwrap_or(0)),
            );
            "Gather"
        }
        "Resize" => {
            attrs.insert(
                "mode".into(),
                AttributeValue::String(n.attrs["mode"].as_str().unwrap_or("nearest").to_string()),
            );
            attrs.insert(
                "coordinate_transformation_mode".into(),
                AttributeValue::String(
                    n.attrs["coordinate_transformation_mode"]
                        .as_str()
                        .unwrap_or("asymmetric")
                        .to_string(),
                ),
            );
            attrs.insert(
                "nearest_mode".into(),
                AttributeValue::String(
                    n.attrs["nearest_mode"]
                        .as_str()
                        .unwrap_or("floor")
                        .to_string(),
                ),
            );
            "Resize"
        }
        other => return Err(format!("unsupported op `{other}`")),
    };

    Ok((onnx_op.to_string(), attrs))
}
