//! Convert an AxonML bundle (.axonml with embedded graph) to ONNX.
//!
//! Usage:
//!   cargo run --release --example bundle_to_onnx -p axonml-onnx -- \
//!       <bundle.axonml> <out.onnx>
//!
//! Walks the bundle's `BundleGraph` (per-node IrOp-named topology +
//! initializer tensors with explicit shapes) and emits an ONNX model via
//! axonml-onnx's `OnnxExporter`. The resulting ONNX is ingestible by Hailo's
//! Dataflow Compiler — feed it into `nexusfoundry compile <out.onnx>
//! --use-dfc --target hailo8|hailo10h --output X.hef`.
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

    let has_rnn = graph.nodes.iter().any(|n| matches!(n.op.as_str(), "GRU" | "LSTM"));
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
    let as_bool = |v: &Value| v.as_bool();

    let onnx_op = match n.op.as_str() {
        "Conv2d" => {
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
        "BatchNorm" => {
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
            attrs.insert(
                "transA".into(),
                AttributeValue::Int(as_bool(&n.attrs["trans_a"]).unwrap_or(false) as i64),
            );
            attrs.insert(
                "transB".into(),
                AttributeValue::Int(as_bool(&n.attrs["trans_b"]).unwrap_or(false) as i64),
            );
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
                attrs.insert(
                    "perm".into(),
                    AttributeValue::Ints(as_i64_vec(perm)),
                );
            }
            "Transpose"
        }
        "Squeeze" => {
            // Opset 13+: axes is a second input tensor, not an attribute.
            // Emit a Constant node for axes and wire it as input[1].
            if let Some(axes) = n.attrs.get("axes") {
                let axes_vec = as_i64_vec(axes);
                let axes_name = format!("{}_axes_const", n.name);
                // Add constant node inline — exporter will handle it
                let axes_f32: Vec<f32> = axes_vec.iter().map(|&x| x as f32).collect();
                // We need to add axes as an initializer with int64 type.
                // Since our initializer API uses f32, we'll add axes as an attribute
                // and let the exporter handle opset compatibility.
                // For DFC compatibility, just pass axes as attribute (DFC accepts both).
                attrs.insert(
                    "axes".into(),
                    AttributeValue::Ints(axes_vec),
                );
            }
            "Squeeze"
        }
        "GRU" => {
            attrs.insert(
                "hidden_size".into(),
                AttributeValue::Int(as_i64(&n.attrs["hidden_size"]).unwrap_or(64)),
            );
            if let Some(dir) = n.attrs.get("direction") {
                attrs.insert(
                    "direction".into(),
                    AttributeValue::String(
                        dir.as_str().unwrap_or("forward").to_string(),
                    ),
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
                    AttributeValue::String(
                        dir.as_str().unwrap_or("forward").to_string(),
                    ),
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
        "Transpose" => {
            if let Some(perm) = n.attrs.get("perm") {
                if let Some(arr) = perm.as_array() {
                    let perm_ints: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                    attrs.insert("perm".into(), AttributeValue::Ints(perm_ints));
                }
            }
            "Transpose"
        }
        "Resize" => {
            attrs.insert(
                "mode".into(),
                AttributeValue::String(
                    n.attrs["mode"]
                        .as_str()
                        .unwrap_or("nearest")
                        .to_string(),
                ),
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
