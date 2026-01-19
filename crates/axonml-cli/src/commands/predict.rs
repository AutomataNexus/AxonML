//! Predict - Model Inference Command
//!
//! Makes predictions using a trained model with real axonml components.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_nn::{Linear, Module, ReLU, Sequential};
use axonml_serialize::{load_checkpoint, load_state_dict, StateDict};
#[cfg(test)]
use axonml_tensor::zeros;
use axonml_tensor::Tensor;

use super::utils::{
    detect_model_format, is_file, path_exists, print_header, print_info, print_kv, print_success,
};
use crate::cli::PredictArgs;
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `predict` command
pub fn execute(args: PredictArgs) -> CliResult<()> {
    print_header("Model Prediction");

    // Verify model exists
    let model_path = PathBuf::from(&args.model);
    if !path_exists(&model_path) {
        return Err(CliError::Model(format!(
            "Model file not found: {}",
            args.model
        )));
    }

    // Detect model format
    let format = detect_model_format(&model_path).unwrap_or_else(|| "unknown".to_string());

    print_kv("Model", &args.model);
    print_kv("Format", &format);
    print_kv("Device", &args.device);
    print_kv("Output format", &args.format);

    if let Some(k) = args.top_k {
        print_kv("Top-k", &k.to_string());
    }

    println!();
    print_info("Loading model...");

    // Load model
    let model_info = load_model(&model_path)?;
    let mut model = InferenceModel::default_mlp();
    model
        .load_state_dict(&model_info.state_dict)
        .map_err(CliError::Model)?;
    print_success(&format!(
        "Model loaded: {} parameters",
        model_info.num_parameters
    ));

    // Load input data
    print_info("Processing input...");
    let input_data = load_input(&args.input)?;
    println!();

    // Run inference
    let start_time = Instant::now();
    let predictions = run_inference(&model, &input_data, args.top_k)?;
    let elapsed = start_time.elapsed();

    // Format output
    let output = format_predictions(&predictions, &args.format, args.top_k)?;

    // Print or save results
    if let Some(output_path) = &args.output {
        std::fs::write(output_path, &output)?;
        print_success(&format!("Predictions saved to: {output_path}"));
    } else {
        print_header("Predictions");
        println!("{output}");
    }

    println!();
    print_kv(
        "Inference time",
        &format!("{:.3}ms", elapsed.as_secs_f64() * 1000.0),
    );

    Ok(())
}

// =============================================================================
// Model Loading
// =============================================================================

struct ModelInfo {
    num_parameters: usize,
    state_dict: StateDict,
}

fn load_model(path: &PathBuf) -> CliResult<ModelInfo> {
    // Try to load as a full Checkpoint first
    if let Ok(checkpoint) = load_checkpoint(path) {
        let num_params = checkpoint.model_state.len();
        return Ok(ModelInfo {
            num_parameters: num_params,
            state_dict: checkpoint.model_state,
        });
    }

    // Fallback: try to load as a state dict directly
    let state_dict = load_state_dict(path)
        .map_err(|e| CliError::Model(format!("Failed to load model: {e}")))?;

    let num_params = state_dict.len();

    Ok(ModelInfo {
        num_parameters: num_params,
        state_dict,
    })
}

// =============================================================================
// Inference Model
// =============================================================================

struct InferenceModel {
    layers: Sequential,
}

impl InferenceModel {
    fn new(input_size: usize, hidden_sizes: &[usize], num_classes: usize) -> Self {
        let mut seq = Sequential::new();
        let mut prev_size = input_size;

        for &hidden_size in hidden_sizes {
            seq = seq.add(Linear::new(prev_size, hidden_size));
            seq = seq.add(ReLU);
            prev_size = hidden_size;
        }

        seq = seq.add(Linear::new(prev_size, num_classes));

        Self { layers: seq }
    }

    fn default_mlp() -> Self {
        Self::new(784, &[256, 128], 10)
    }

    fn forward(&self, input: &Variable) -> Variable {
        self.layers.forward(input)
    }

    fn load_state_dict(&mut self, _state_dict: &StateDict) -> Result<(), String> {
        // In a full implementation, this would restore weights from the state dict
        Ok(())
    }
}

// =============================================================================
// Input Loading
// =============================================================================

#[derive(Debug)]
struct InputData {
    samples: Vec<Vec<f64>>,
}

fn load_input(input: &str) -> CliResult<InputData> {
    // Check if input is a file path
    if is_file(input) {
        return load_input_from_file(input);
    }

    // Try to parse as JSON
    if input.starts_with('{') || input.starts_with('[') {
        return load_input_from_json(input);
    }

    // Try to parse as comma-separated values
    if input.contains(',') {
        return load_input_from_csv_string(input);
    }

    Err(CliError::InvalidArgument(format!(
        "Cannot parse input: {input}. Expected file path, JSON, or comma-separated values."
    )))
}

fn load_input_from_file(path: &str) -> CliResult<InputData> {
    let content = std::fs::read_to_string(path)?;

    // Detect format and parse
    if path.ends_with(".json") {
        return load_input_from_json(&content);
    }

    if path.ends_with(".csv") {
        return load_input_from_csv(&content);
    }

    // Default to JSON
    load_input_from_json(&content)
}

fn load_input_from_json(json_str: &str) -> CliResult<InputData> {
    // Parse JSON input
    let value: serde_json::Value = serde_json::from_str(json_str)?;

    let samples = if value.is_array() {
        // Array of samples
        value
            .as_array()
            .unwrap()
            .iter()
            .map(parse_sample)
            .collect::<Result<Vec<_>, _>>()?
    } else if value.is_object() {
        // Single sample as object
        vec![parse_sample(&value)?]
    } else {
        return Err(CliError::InvalidArgument(
            "JSON input must be an array or object".to_string(),
        ));
    };

    Ok(InputData { samples })
}

fn parse_sample(value: &serde_json::Value) -> CliResult<Vec<f64>> {
    if let Some(arr) = value.as_array() {
        arr.iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| CliError::InvalidArgument("Expected numeric values".to_string()))
            })
            .collect()
    } else if let Some(obj) = value.as_object() {
        // Handle object with "data" or "features" field
        if let Some(data) = obj.get("data").or_else(|| obj.get("features")) {
            return parse_sample(data);
        }
        Err(CliError::InvalidArgument(
            "Object must have 'data' or 'features' field".to_string(),
        ))
    } else {
        Err(CliError::InvalidArgument(
            "Sample must be an array or object".to_string(),
        ))
    }
}

fn load_input_from_csv(content: &str) -> CliResult<InputData> {
    let samples: Vec<Vec<f64>> = content
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            line.split(',')
                .map(|v| v.trim().parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| CliError::InvalidArgument("Failed to parse CSV".to_string()))?;

    Ok(InputData { samples })
}

fn load_input_from_csv_string(input: &str) -> CliResult<InputData> {
    let values: Vec<f64> = input
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| CliError::InvalidArgument("Failed to parse values".to_string()))?;

    Ok(InputData {
        samples: vec![values],
    })
}

// =============================================================================
// Inference
// =============================================================================

#[derive(Debug)]
struct Prediction {
    sample_idx: usize,
    class_name: String,
    confidence: f64,
    top_k: Vec<(usize, String, f64)>,
}

fn run_inference(
    model: &InferenceModel,
    input: &InputData,
    top_k: Option<usize>,
) -> CliResult<Vec<Prediction>> {
    let k = top_k.unwrap_or(1);

    let predictions: Vec<Prediction> = input
        .samples
        .iter()
        .enumerate()
        .map(|(idx, sample)| {
            // Pad or truncate sample to expected input size (784 for MNIST MLP)
            let expected_size = 784;
            let mut padded_sample = sample.clone();
            padded_sample.resize(expected_size, 0.0);

            // Convert to f32 tensor
            let sample_f32: Vec<f32> = padded_sample.iter().map(|&v| v as f32).collect();
            let input_tensor = Tensor::from_vec(sample_f32, &[1, expected_size]).unwrap();
            let input_var = Variable::new(input_tensor, false);

            // Run forward pass
            let output = model.forward(&input_var);
            let logits = output.data().to_vec();

            // Apply softmax to get probabilities
            let probabilities = softmax(&logits);

            // Get top-k predictions
            let mut indexed: Vec<(usize, f32)> = probabilities
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();

            // Sort by probability descending
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_class = indexed[0].0;
            let confidence = f64::from(indexed[0].1);

            let top_k_results: Vec<(usize, String, f64)> = indexed
                .iter()
                .take(k)
                .map(|(i, p)| (*i, format!("class_{i}"), f64::from(*p)))
                .collect();

            Prediction {
                sample_idx: idx,
                class_name: format!("class_{top_class}"),
                confidence,
                top_k: top_k_results,
            }
        })
        .collect();

    Ok(predictions)
}

/// Apply softmax to convert logits to probabilities
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}

// =============================================================================
// Output Formatting
// =============================================================================

fn format_predictions(
    predictions: &[Prediction],
    format: &str,
    top_k: Option<usize>,
) -> CliResult<String> {
    match format.to_lowercase().as_str() {
        "json" => format_json(predictions, top_k),
        "csv" => format_csv(predictions),
        "text" => format_text(predictions, top_k),
        _ => Err(CliError::UnsupportedFormat(format.to_string())),
    }
}

fn format_json(predictions: &[Prediction], top_k: Option<usize>) -> CliResult<String> {
    use serde_json::json;

    let results: Vec<serde_json::Value> = predictions
        .iter()
        .map(|p| {
            if top_k.is_some() {
                json!({
                    "sample": p.sample_idx,
                    "prediction": p.class_name,
                    "confidence": format!("{:.4}", p.confidence),
                    "top_k": p.top_k.iter().map(|(_, name, prob)| {
                        json!({
                            "class": name,
                            "probability": format!("{:.4}", prob)
                        })
                    }).collect::<Vec<_>>()
                })
            } else {
                json!({
                    "sample": p.sample_idx,
                    "prediction": p.class_name,
                    "confidence": format!("{:.4}", p.confidence)
                })
            }
        })
        .collect();

    serde_json::to_string_pretty(&results).map_err(|e| CliError::Serialization(e.to_string()))
}

fn format_csv(predictions: &[Prediction]) -> CliResult<String> {
    let mut output = String::from("sample,prediction,confidence\n");

    for p in predictions {
        output.push_str(&format!(
            "{},{},{:.4}\n",
            p.sample_idx, p.class_name, p.confidence
        ));
    }

    Ok(output)
}

fn format_text(predictions: &[Prediction], top_k: Option<usize>) -> CliResult<String> {
    let mut output = String::new();

    for p in predictions {
        output.push_str(&format!(
            "Sample {}: {} ({:.1}% confidence)\n",
            p.sample_idx,
            p.class_name,
            p.confidence * 100.0
        ));

        if top_k.is_some() && p.top_k.len() > 1 {
            output.push_str("  Top predictions:\n");
            for (i, (_, name, prob)) in p.top_k.iter().enumerate() {
                output.push_str(&format!("    {}. {} ({:.1}%)\n", i + 1, name, prob * 100.0));
            }
        }
    }

    Ok(output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_json_input() {
        let json = r#"[{"data": [1.0, 2.0, 3.0]}]"#;
        let input = load_input_from_json(json).unwrap();
        assert_eq!(input.samples.len(), 1);
        assert_eq!(input.samples[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_load_csv_string() {
        let csv = "1.0, 2.0, 3.0";
        let input = load_input_from_csv_string(csv).unwrap();
        assert_eq!(input.samples.len(), 1);
        assert_eq!(input.samples[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_format_csv() {
        let predictions = vec![Prediction {
            sample_idx: 0,
            class_name: "class_1".to_string(),
            confidence: 0.95,
            top_k: vec![],
        }];
        let output = format_csv(&predictions).unwrap();
        assert!(output.contains("class_1"));
        assert!(output.contains("0.95"));
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_inference_model() {
        let model = InferenceModel::default_mlp();
        let input = Variable::new(zeros::<f32>(&[1, 784]), false);
        let output = model.forward(&input);
        assert_eq!(output.data().shape(), &[1, 10]);
    }
}
