//! Inspect - Model Inspection Command
//!
//! Displays detailed information about a model's architecture and parameters.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;

use super::utils::{detect_model_format, path_exists, print_header, print_info, print_kv};
use crate::cli::InspectArgs;
use crate::error::{CliError, CliResult};
use axonml_serialize::load_state_dict;

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `inspect` command
pub fn execute(args: InspectArgs) -> CliResult<()> {
    // Verify model exists
    let model_path = PathBuf::from(&args.model);
    if !path_exists(&model_path) {
        return Err(CliError::Model(format!(
            "Model file not found: {}",
            args.model
        )));
    }

    // Detect format
    let format = detect_model_format(&model_path).unwrap_or_else(|| "unknown".to_string());

    // Load and inspect model
    let info = inspect_model(&model_path, &format)?;

    // Output based on format
    match args.format.to_lowercase().as_str() {
        "json" => output_json(&info, args.detailed)?,
        _ => output_text(&info, &args, &format)?,
    }

    Ok(())
}

// =============================================================================
// Model Information
// =============================================================================

#[derive(Debug)]
struct ModelInfo {
    name: String,
    format: String,
    file_size: u64,
    num_parameters: u64,
    num_layers: usize,
    layers: Vec<LayerInfo>,
    metadata: Vec<(String, String)>,
}

#[derive(Debug)]
struct LayerInfo {
    index: usize,
    name: String,
    layer_type: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    num_params: u64,
    trainable: bool,
}

fn inspect_model(path: &PathBuf, format: &str) -> CliResult<ModelInfo> {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let name = path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();

    // Load the actual state dict from the model file
    let state_dict = load_state_dict(path)
        .map_err(|e| CliError::Model(format!("Failed to load model: {}", e)))?;

    // Extract layer information from state dict entries
    let mut layers = Vec::new();
    let mut total_params: u64 = 0;

    // Group parameters by layer prefix
    let mut layer_params: std::collections::BTreeMap<String, Vec<(String, Vec<usize>, u64, bool)>> =
        std::collections::BTreeMap::new();

    for (param_name, entry) in state_dict.entries() {
        let shape = entry.data.shape().to_vec();
        let num_params: u64 = shape.iter().map(|&s| s as u64).product();
        total_params += num_params;

        // Extract layer name from parameter name (e.g., "layer1.conv.weight" -> "layer1.conv")
        let layer_name = if let Some(idx) = param_name.rfind('.') {
            param_name[..idx].to_string()
        } else {
            param_name.clone()
        };

        layer_params
            .entry(layer_name)
            .or_default()
            .push((param_name.clone(), shape, num_params, entry.requires_grad));
    }

    // Create LayerInfo for each unique layer
    for (index, (layer_name, params)) in layer_params.iter().enumerate() {
        let layer_num_params: u64 = params.iter().map(|(_, _, p, _)| *p).sum();
        let trainable = params.iter().any(|(_, _, _, t)| *t);

        // Infer layer type from parameter names
        let layer_type = infer_layer_type(&params);

        // Get shape from weight parameter if available
        let (input_shape, output_shape) = infer_shapes(&params);

        layers.push(LayerInfo {
            index,
            name: layer_name.clone(),
            layer_type,
            input_shape,
            output_shape,
            num_params: layer_num_params,
            trainable,
        });
    }

    // Extract metadata
    let mut metadata = Vec::new();
    metadata.push(("framework".to_string(), "axonml".to_string()));
    metadata.push(("format".to_string(), format.to_string()));
    metadata.push(("total_parameters".to_string(), total_params.to_string()));

    // Add common metadata keys if present
    for key in &["version", "model_type", "dtype", "created"] {
        if let Some(value) = state_dict.get_metadata(key) {
            metadata.push((key.to_string(), value.clone()));
        }
    }

    Ok(ModelInfo {
        name,
        format: format.to_string(),
        file_size,
        num_parameters: total_params,
        num_layers: layers.len(),
        layers,
        metadata,
    })
}

/// Infer layer type from parameter names
fn infer_layer_type(params: &[(String, Vec<usize>, u64, bool)]) -> String {
    for (name, shape, _, _) in params {
        if name.ends_with(".weight") {
            let dims = shape.len();
            if dims == 4 {
                return "Conv2d".to_string();
            } else if dims == 2 {
                return "Linear".to_string();
            } else if dims == 1 {
                return "BatchNorm".to_string();
            }
        }
        if name.ends_with(".gamma") || name.ends_with(".beta") {
            return "LayerNorm".to_string();
        }
        if name.ends_with(".embedding") {
            return "Embedding".to_string();
        }
    }
    "Unknown".to_string()
}

/// Infer input and output shapes from parameters
fn infer_shapes(params: &[(String, Vec<usize>, u64, bool)]) -> (Vec<usize>, Vec<usize>) {
    for (name, shape, _, _) in params {
        if name.ends_with(".weight") && shape.len() >= 2 {
            // For Linear: [out_features, in_features]
            // For Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
            if shape.len() == 2 {
                return (vec![1, shape[1]], vec![1, shape[0]]);
            } else if shape.len() == 4 {
                return (
                    vec![1, shape[1], 0, 0], // [batch, in_channels, H, W]
                    vec![1, shape[0], 0, 0], // [batch, out_channels, H, W]
                );
            }
        }
    }
    (vec![], vec![])
}

// =============================================================================
// Text Output
// =============================================================================

fn output_text(info: &ModelInfo, args: &InspectArgs, format: &str) -> CliResult<()> {
    print_header(&format!("Model: {}", info.name));

    print_kv("Format", format);
    print_kv("File size", &format_size(info.file_size));
    print_kv("Parameters", &format_params(info.num_parameters));
    print_kv("Layers", &info.num_layers.to_string());

    // Metadata
    if !info.metadata.is_empty() {
        println!();
        print_info("Metadata:");
        for (key, value) in &info.metadata {
            print_kv(&format!("  {key}"), value);
        }
    }

    // Layer summary
    println!();
    print_header("Architecture");

    if args.detailed {
        // Detailed layer view
        println!(
            "{:<6} {:<25} {:<15} {:<20} {:<20} {:<12}",
            "Index", "Name", "Type", "Input Shape", "Output Shape", "Params"
        );
        println!("{}", "-".repeat(100));

        for layer in &info.layers {
            println!(
                "{:<6} {:<25} {:<15} {:<20} {:<20} {:<12}",
                layer.index,
                truncate(&layer.name, 24),
                &layer.layer_type,
                format_shape(&layer.input_shape),
                format_shape(&layer.output_shape),
                format_params(layer.num_params),
            );
        }
    } else {
        // Compact view
        println!("{:<25} {:<15} {:<12}", "Layer", "Type", "Params");
        println!("{}", "-".repeat(55));

        for layer in &info.layers {
            println!(
                "{:<25} {:<15} {:<12}",
                truncate(&layer.name, 24),
                &layer.layer_type,
                format_params(layer.num_params),
            );
        }
    }

    // Parameter breakdown
    println!();
    print_header("Parameter Summary");

    let trainable: u64 = info
        .layers
        .iter()
        .filter(|l| l.trainable)
        .map(|l| l.num_params)
        .sum();
    let non_trainable = info.num_parameters - trainable;

    print_kv("Total parameters", &format_params(info.num_parameters));
    print_kv("Trainable", &format_params(trainable));
    print_kv("Non-trainable", &format_params(non_trainable));

    // Memory estimate
    let memory_bytes = info.num_parameters * 4; // Assuming float32
    print_kv("Memory (fp32)", &format_size(memory_bytes));
    print_kv("Memory (fp16)", &format_size(memory_bytes / 2));

    // Show sample parameters if requested
    if let Some(n) = args.show_params {
        println!();
        print_header(&format!("Sample Parameters (first {n} per layer)"));
        print_info("Parameter values not shown in simulation mode");
    }

    Ok(())
}

// =============================================================================
// JSON Output
// =============================================================================

fn output_json(info: &ModelInfo, detailed: bool) -> CliResult<()> {
    use serde_json::json;

    let layers_json: Vec<serde_json::Value> = if detailed {
        info.layers
            .iter()
            .map(|l| {
                json!({
                    "index": l.index,
                    "name": l.name,
                    "type": l.layer_type,
                    "input_shape": l.input_shape,
                    "output_shape": l.output_shape,
                    "parameters": l.num_params,
                    "trainable": l.trainable,
                })
            })
            .collect()
    } else {
        info.layers
            .iter()
            .map(|l| {
                json!({
                    "name": l.name,
                    "type": l.layer_type,
                    "parameters": l.num_params,
                })
            })
            .collect()
    };

    let output = json!({
        "name": info.name,
        "format": info.format,
        "file_size": info.file_size,
        "num_parameters": info.num_parameters,
        "num_layers": info.num_layers,
        "layers": layers_json,
        "metadata": info.metadata.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<std::collections::HashMap<_, _>>(),
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

// =============================================================================
// Formatting Helpers
// =============================================================================

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_params(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1000 {
        format!("{:.2}K", n as f64 / 1000.0)
    } else {
        format!("{n}")
    }
}

fn format_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
    format!("[{}]", parts.join(", "))
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1500), "1.50K");
        assert_eq!(format_params(1_500_000), "1.50M");
    }

    #[test]
    fn test_format_shape() {
        assert_eq!(format_shape(&[1, 3, 224, 224]), "[1, 3, 224, 224]");
        assert_eq!(format_shape(&[512]), "[512]");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a very long string", 10), "this is...");
    }
}
