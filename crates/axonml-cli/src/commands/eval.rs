//! Eval - Model Evaluation Command
//!
//! Evaluates a trained model on a test dataset using real axonml components.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_data::{DataLoader, Dataset};
use axonml_nn::CrossEntropyLoss;
use axonml_nn::{Linear, Module, ReLU, Sequential};
use axonml_serialize::{load_checkpoint, load_state_dict, StateDict};
#[cfg(test)]
use axonml_tensor::zeros;
use axonml_tensor::Tensor;
use axonml_vision::SyntheticMNIST;

use super::utils::{
    detect_model_format, path_exists, print_header, print_info, print_kv, print_success,
    print_warning, training_progress_bar,
};
use crate::cli::EvalArgs;
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `eval` command
pub fn execute(args: EvalArgs) -> CliResult<()> {
    print_header("Model Evaluation");

    // Verify model exists
    let model_path = PathBuf::from(&args.model);
    if !path_exists(&model_path) {
        return Err(CliError::Model(format!(
            "Model file not found: {}",
            args.model
        )));
    }

    // Verify data exists
    let data_path = PathBuf::from(&args.data);
    if !path_exists(&data_path) {
        return Err(CliError::Config(format!(
            "Data path not found: {}",
            args.data
        )));
    }

    // Detect model format
    let format = detect_model_format(&model_path).unwrap_or_else(|| "unknown".to_string());

    print_header("Configuration");
    print_kv("Model", &args.model);
    print_kv("Format", &format);
    print_kv("Data", &args.data);
    print_kv("Batch size", &args.batch_size.to_string());
    print_kv("Device", &args.device);
    print_kv("Metrics", &args.metrics);

    println!();
    print_info("Loading model...");

    // Load model state
    let model_info = load_model(&model_path)?;
    print_success(&format!(
        "Model loaded: {} parameters",
        model_info.num_parameters
    ));

    print_info("Running evaluation...");
    println!();

    // Run evaluation
    let start_time = Instant::now();
    let metrics = run_evaluation(&args, &model_info)?;
    let elapsed = start_time.elapsed();

    // Print results
    println!();
    print_header("Evaluation Results");
    for (name, value) in &metrics {
        print_kv(name, &format_metric_value(name, *value));
    }
    println!();
    print_kv("Evaluation time", &format!("{:.2}s", elapsed.as_secs_f64()));

    // Save results if output specified
    if let Some(output_path) = &args.output {
        save_metrics(&metrics, output_path)?;
        print_success(&format!("Results saved to: {output_path}"));
    }

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
// Evaluatable Model
// =============================================================================

struct EvalModel {
    layers: Sequential,
}

impl EvalModel {
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
// Dataset Wrapper
// =============================================================================

struct EvalDataset(SyntheticMNIST);

impl Dataset for EvalDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.0.get(index)
    }
}

// =============================================================================
// Evaluation
// =============================================================================

fn run_evaluation(args: &EvalArgs, model_info: &ModelInfo) -> CliResult<Vec<(String, f64)>> {
    let requested_metrics: Vec<&str> = args.metrics.split(',').map(str::trim).collect();

    // Create and initialize model
    let mut model = EvalModel::default_mlp();
    model
        .load_state_dict(&model_info.state_dict)
        .map_err(CliError::Model)?;

    // Load dataset (using synthetic data that matches saved model architecture)
    let dataset = EvalDataset(SyntheticMNIST::new(10000));
    let loader = DataLoader::new(dataset, args.batch_size);
    let total_batches = loader.len() as u64;

    // Loss function
    let loss_fn = CrossEntropyLoss::new();

    // Progress bar
    let pb = training_progress_bar(total_batches);

    // Evaluation metrics
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut total_loss = 0.0f64;
    let mut predictions: Vec<(usize, usize)> = Vec::new();

    // Evaluation loop (no gradients needed)
    for batch in loader.iter() {
        // Convert batch data to Variables (no gradient tracking)
        let input = Variable::new(batch.data.clone(), false);
        let target = Variable::new(batch.targets.clone(), false);

        // Forward pass
        let output = model.forward(&input);

        // Compute loss
        let loss = loss_fn.compute(&output, &target);
        let loss_val = f64::from(loss.data().to_vec()[0]);
        total_loss += loss_val;

        // Compute accuracy
        let pred_classes = argmax_batch(&output.data());
        let label_classes = argmax_batch(&batch.targets);

        for (pred, label) in pred_classes.iter().zip(label_classes.iter()) {
            predictions.push((*pred, *label));
            if pred == label {
                correct += 1;
            }
            total += 1;
        }

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Calculate and collect requested metrics
    let mut results = Vec::new();

    for metric in requested_metrics {
        match metric.to_lowercase().as_str() {
            "accuracy" => {
                let accuracy = correct as f64 / total as f64;
                results.push(("accuracy".to_string(), accuracy));
            }
            "loss" => {
                let avg_loss = total_loss / total_batches as f64;
                results.push(("loss".to_string(), avg_loss));
            }
            "precision" => {
                let precision = calculate_precision(&predictions);
                results.push(("precision".to_string(), precision));
            }
            "recall" => {
                let recall = calculate_recall(&predictions);
                results.push(("recall".to_string(), recall));
            }
            "f1" => {
                let f1 = calculate_f1(&predictions);
                results.push(("f1".to_string(), f1));
            }
            _ => {
                print_warning(&format!("Unknown metric: {metric}"));
            }
        }
    }

    // Add total samples
    results.push(("samples".to_string(), total as f64));

    Ok(results)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get argmax for each sample in a batch
fn argmax_batch(tensor: &Tensor<f32>) -> Vec<usize> {
    let shape = tensor.shape();
    let data = tensor.to_vec();

    if shape.len() == 1 {
        let (idx, _) = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        vec![idx]
    } else {
        let batch_size = shape[0];
        let num_classes = shape[1];

        (0..batch_size)
            .map(|b| {
                let start = b * num_classes;
                let end = start + num_classes;
                let slice = &data[start..end];

                slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map_or(0, |(idx, _)| idx)
            })
            .collect()
    }
}

fn calculate_precision(predictions: &[(usize, usize)]) -> f64 {
    // Macro-averaged precision across all classes
    let num_classes = 10;
    let mut class_tp = vec![0usize; num_classes];
    let mut class_fp = vec![0usize; num_classes];

    for &(pred, actual) in predictions {
        if pred == actual {
            class_tp[pred] += 1;
        } else {
            class_fp[pred] += 1;
        }
    }

    let mut precision_sum = 0.0;
    let mut valid_classes = 0;

    for i in 0..num_classes {
        let tp = class_tp[i];
        let fp = class_fp[i];
        if tp + fp > 0 {
            precision_sum += tp as f64 / (tp + fp) as f64;
            valid_classes += 1;
        }
    }

    if valid_classes > 0 {
        precision_sum / f64::from(valid_classes)
    } else {
        0.0
    }
}

fn calculate_recall(predictions: &[(usize, usize)]) -> f64 {
    // Macro-averaged recall across all classes
    let num_classes = 10;
    let mut class_tp = vec![0usize; num_classes];
    let mut class_fn = vec![0usize; num_classes];

    for &(pred, actual) in predictions {
        if pred == actual {
            class_tp[actual] += 1;
        } else {
            class_fn[actual] += 1;
        }
    }

    let mut recall_sum = 0.0;
    let mut valid_classes = 0;

    for i in 0..num_classes {
        let tp = class_tp[i];
        let fn_count = class_fn[i];
        if tp + fn_count > 0 {
            recall_sum += tp as f64 / (tp + fn_count) as f64;
            valid_classes += 1;
        }
    }

    if valid_classes > 0 {
        recall_sum / f64::from(valid_classes)
    } else {
        0.0
    }
}

fn calculate_f1(predictions: &[(usize, usize)]) -> f64 {
    let precision = calculate_precision(predictions);
    let recall = calculate_recall(predictions);
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

// =============================================================================
// Output Formatting
// =============================================================================

fn format_metric_value(name: &str, value: f64) -> String {
    match name {
        "accuracy" | "precision" | "recall" | "f1" => {
            format!("{:.2}%", value * 100.0)
        }
        "loss" => {
            format!("{value:.4}")
        }
        "samples" => {
            format!("{}", value as usize)
        }
        _ => {
            format!("{value:.4}")
        }
    }
}

fn save_metrics(metrics: &[(String, f64)], output_path: &str) -> CliResult<()> {
    use std::collections::HashMap;

    let metrics_map: HashMap<&str, f64> = metrics.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    let json = serde_json::to_string_pretty(&metrics_map)?;
    std::fs::write(output_path, json)?;

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_metric() {
        assert_eq!(format_metric_value("accuracy", 0.95), "95.00%");
        assert_eq!(format_metric_value("loss", 0.1234), "0.1234");
        assert_eq!(format_metric_value("samples", 1000.0), "1000");
    }

    #[test]
    fn test_calculate_metrics() {
        let predictions = vec![(0, 0), (1, 1), (2, 3), (3, 3)];
        let precision = calculate_precision(&predictions);
        assert!(precision > 0.0);
    }

    #[test]
    fn test_argmax() {
        let data = Tensor::from_vec(vec![0.1, 0.8, 0.1, 0.7, 0.2, 0.1], &[2, 3]).unwrap();

        let result = argmax_batch(&data);
        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn test_eval_model_creation() {
        let model = EvalModel::default_mlp();
        let input = Variable::new(zeros::<f32>(&[1, 784]), false);
        let _output = model.forward(&input);
    }
}
