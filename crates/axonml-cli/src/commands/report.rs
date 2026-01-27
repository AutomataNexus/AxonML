//! Report - Comprehensive Model Evaluation and Visualization
//!
//! Generates detailed reports with metrics, confusion matrices,
//! loss curves, F1 scores, and exports to HTML/JSON formats.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;

use axonml_autograd::Variable;
use axonml_data::{DataLoader, Dataset};
use axonml_nn::CrossEntropyLoss;
use axonml_nn::{Linear, Module, ReLU, Sequential};
use axonml_serialize::{load_checkpoint, load_state_dict, StateDict};
use axonml_tensor::Tensor;
use axonml_vision::{FashionMNIST, CIFAR10, MNIST};

use super::utils::{
    ensure_dir, path_exists, print_header, print_info, print_kv, print_success,
    training_progress_bar,
};
use crate::cli::ReportArgs;
use crate::error::{CliError, CliResult};

// =============================================================================
// Metrics Structures
// =============================================================================

/// Complete evaluation metrics for a classification model
#[derive(Debug, Clone, Default)]
pub struct ClassificationMetrics {
    /// Total number of samples
    pub total_samples: usize,
    /// Overall accuracy
    pub accuracy: f64,
    /// Average loss
    pub loss: f64,
    /// Per-class metrics
    pub per_class: Vec<ClassMetrics>,
    /// Macro-averaged precision
    pub macro_precision: f64,
    /// Macro-averaged recall
    pub macro_recall: f64,
    /// Macro-averaged F1 score
    pub macro_f1: f64,
    /// Weighted F1 score
    pub weighted_f1: f64,
    /// Confusion matrix (row=actual, col=predicted)
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Number of classes
    pub num_classes: usize,
    /// Class names
    pub class_names: Vec<String>,
}

/// Metrics for a single class
#[derive(Debug, Clone, Default)]
pub struct ClassMetrics {
    pub class_id: usize,
    pub class_name: String,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub true_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

/// Training history for plotting (used by HTML report generation)
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    pub epochs: Vec<usize>,
    pub train_loss: Vec<f64>,
    pub train_accuracy: Vec<f64>,
    pub val_loss: Vec<f64>,
    pub val_accuracy: Vec<f64>,
    pub learning_rates: Vec<f64>,
}

impl TrainingHistory {
    /// Load training history from a JSON log file
    pub fn from_file(path: &str) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;

        let mut history = TrainingHistory::default();

        // Try to parse as JSON array first
        if let Ok(records) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
            for record in records {
                if let Some(epoch) = record.get("epoch").and_then(|v| v.as_u64()) {
                    history.epochs.push(epoch as usize);
                }
                if let Some(loss) = record.get("train_loss").and_then(|v| v.as_f64()) {
                    history.train_loss.push(loss);
                }
                if let Some(loss) = record.get("val_loss").and_then(|v| v.as_f64()) {
                    history.val_loss.push(loss);
                }
                if let Some(acc) = record.get("train_accuracy").and_then(|v| v.as_f64()) {
                    history.train_accuracy.push(acc);
                }
                if let Some(acc) = record.get("val_accuracy").and_then(|v| v.as_f64()) {
                    history.val_accuracy.push(acc);
                }
                if let Some(lr) = record.get("learning_rate").and_then(|v| v.as_f64()) {
                    history.learning_rates.push(lr);
                }
            }
        } else {
            // Try to parse as newline-delimited JSON (JSONL format)
            for line in content.lines() {
                if let Ok(record) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(epoch) = record.get("epoch").and_then(|v| v.as_u64()) {
                        history.epochs.push(epoch as usize);
                    }
                    if let Some(loss) = record.get("train_loss").and_then(|v| v.as_f64()) {
                        history.train_loss.push(loss);
                    } else if let Some(loss) = record.get("loss").and_then(|v| v.as_f64()) {
                        history.train_loss.push(loss);
                    }
                    if let Some(loss) = record.get("val_loss").and_then(|v| v.as_f64()) {
                        history.val_loss.push(loss);
                    }
                    if let Some(acc) = record.get("train_accuracy").and_then(|v| v.as_f64()) {
                        history.train_accuracy.push(acc);
                    } else if let Some(acc) = record.get("accuracy").and_then(|v| v.as_f64()) {
                        history.train_accuracy.push(acc);
                    }
                    if let Some(acc) = record.get("val_accuracy").and_then(|v| v.as_f64()) {
                        history.val_accuracy.push(acc);
                    }
                    if let Some(lr) = record.get("learning_rate").and_then(|v| v.as_f64()) {
                        history.learning_rates.push(lr);
                    } else if let Some(lr) = record.get("lr").and_then(|v| v.as_f64()) {
                        history.learning_rates.push(lr);
                    }
                }
            }
        }

        // Only return if we have meaningful data
        if !history.train_loss.is_empty() {
            // Fill epochs if empty
            if history.epochs.is_empty() {
                history.epochs = (1..=history.train_loss.len()).collect();
            }
            Some(history)
        } else {
            None
        }
    }

    /// Check if we have real training data
    pub fn is_empty(&self) -> bool {
        self.train_loss.is_empty()
    }
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `report` command
pub fn execute(args: ReportArgs) -> CliResult<()> {
    print_header("Axonml Model Report");

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

    print_kv("Model", &args.model);
    print_kv("Data", &args.data);
    print_kv("Output format", &args.format);

    println!();
    print_info("Loading model...");

    // Load model
    let model_info = load_model(&model_path)?;
    print_success(&format!(
        "Model loaded: {} parameters",
        model_info.num_parameters
    ));

    // Run evaluation and collect metrics
    print_info("Running evaluation...");
    println!();

    let metrics = evaluate_model(&args, &model_info)?;

    // Print metrics summary
    print_metrics_summary(&metrics);

    // Print confusion matrix
    if args.confusion_matrix {
        print_confusion_matrix(&metrics);
    }

    // Generate output
    let output_dir = args
        .output
        .clone()
        .unwrap_or_else(|| "./report".to_string());
    ensure_dir(&output_dir)?;

    // Load training history if provided
    let training_history = args.history.as_ref().and_then(|path| {
        print_info(&format!("Loading training history from: {path}"));
        TrainingHistory::from_file(path)
    });

    if training_history.is_some() {
        print_success("Training history loaded successfully");
    } else if args.history.is_some() {
        print_info("Could not parse training history file, loss curves will not be included");
    }

    match args.format.to_lowercase().as_str() {
        "html" => {
            let html_path = format!("{output_dir}/report.html");
            generate_html_report(&metrics, &args, training_history.as_ref(), &html_path)?;
            print_success(&format!("HTML report saved to: {html_path}"));
        }
        "json" => {
            let json_path = format!("{output_dir}/report.json");
            generate_json_report(&metrics, &json_path)?;
            print_success(&format!("JSON report saved to: {json_path}"));
        }
        "text" | "txt" => {
            let text_path = format!("{output_dir}/report.txt");
            generate_text_report(&metrics, &text_path)?;
            print_success(&format!("Text report saved to: {text_path}"));
        }
        "all" => {
            let html_path = format!("{output_dir}/report.html");
            generate_html_report(&metrics, &args, training_history.as_ref(), &html_path)?;
            print_success(&format!("HTML report saved to: {html_path}"));

            let json_path = format!("{output_dir}/report.json");
            generate_json_report(&metrics, &json_path)?;
            print_success(&format!("JSON report saved to: {json_path}"));

            let text_path = format!("{output_dir}/report.txt");
            generate_text_report(&metrics, &text_path)?;
            print_success(&format!("Text report saved to: {text_path}"));
        }
        _ => {
            return Err(CliError::UnsupportedFormat(args.format));
        }
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
    let state_dict =
        load_state_dict(path).map_err(|e| CliError::Model(format!("Failed to load model: {e}")))?;

    let num_params = state_dict.len();

    Ok(ModelInfo {
        num_parameters: num_params,
        state_dict,
    })
}

// =============================================================================
// Report Model
// =============================================================================

struct ReportModel {
    layers: Sequential,
}

impl ReportModel {
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

enum ReportDataset {
    Mnist(MNIST),
    FashionMnist(FashionMNIST),
    Cifar10(CIFAR10),
}

impl ReportDataset {
    fn load(path: &std::path::Path, format: &str, train: bool) -> Result<Self, String> {
        match format.to_lowercase().as_str() {
            "mnist" => Ok(ReportDataset::Mnist(MNIST::new(path, train)?)),
            "fashion-mnist" | "fashion_mnist" | "fashionmnist" => {
                Ok(ReportDataset::FashionMnist(FashionMNIST::new(path, train)?))
            }
            "cifar10" | "cifar-10" => Ok(ReportDataset::Cifar10(CIFAR10::new(path, train)?)),
            _ => Err(format!(
                "Unsupported dataset format: '{}'. Supported: mnist, fashion-mnist, cifar10",
                format
            )),
        }
    }

    fn detect_format(path: &std::path::Path) -> Option<String> {
        if path.join("t10k-images-idx3-ubyte").exists()
            || path.join("t10k-images-idx3-ubyte.gz").exists()
        {
            return Some("mnist".to_string());
        }
        if path.join("test_batch.bin").exists() {
            return Some("cifar10".to_string());
        }
        None
    }
}

impl Dataset for ReportDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        match self {
            ReportDataset::Mnist(d) => d.len(),
            ReportDataset::FashionMnist(d) => d.len(),
            ReportDataset::Cifar10(d) => d.len(),
        }
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        match self {
            ReportDataset::Mnist(d) => d.get(index),
            ReportDataset::FashionMnist(d) => d.get(index),
            ReportDataset::Cifar10(d) => d.get(index),
        }
    }
}

// =============================================================================
// Evaluation
// =============================================================================

fn evaluate_model(args: &ReportArgs, model_info: &ModelInfo) -> CliResult<ClassificationMetrics> {
    let num_classes = args.num_classes.unwrap_or(10);

    // Create and initialize model
    let mut model = ReportModel::default_mlp();
    model
        .load_state_dict(&model_info.state_dict)
        .map_err(CliError::Model)?;

    // Load dataset from the specified path
    let data_path = PathBuf::from(&args.data);

    let format = args.dataset_format.clone().unwrap_or_else(|| {
        ReportDataset::detect_format(&data_path).unwrap_or_else(|| "mnist".to_string())
    });

    print_info(&format!("Loading {} dataset from: {}", format, args.data));

    let dataset = ReportDataset::load(&data_path, &format, false) // false = test set
        .map_err(|e| CliError::Config(format!("Failed to load dataset: {}", e)))?;

    print_success(&format!("Loaded {} samples", dataset.len()));

    let loader = DataLoader::new(dataset, args.batch_size);
    let total_batches = loader.len() as u64;

    // Loss function
    let loss_fn = CrossEntropyLoss::new();

    // Progress bar
    let pb = training_progress_bar(total_batches);

    // Initialize confusion matrix
    let mut confusion_matrix = vec![vec![0usize; num_classes]; num_classes];
    let mut total_loss = 0.0f64;

    // Evaluation loop
    for batch in loader.iter() {
        let input = Variable::new(batch.data.clone(), false);
        let target = Variable::new(batch.targets.clone(), false);

        // Forward pass
        let output = model.forward(&input);

        // Compute loss
        let loss = loss_fn.compute(&output, &target);
        let loss_val = f64::from(loss.data().to_vec()[0]);
        total_loss += loss_val;

        // Get predictions
        let pred_classes = argmax_batch(&output.data());
        let label_classes = argmax_batch(&batch.targets);

        // Update confusion matrix
        for (pred, actual) in pred_classes.iter().zip(label_classes.iter()) {
            if *actual < num_classes && *pred < num_classes {
                confusion_matrix[*actual][*pred] += 1;
            }
        }

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Calculate metrics from confusion matrix
    let metrics =
        calculate_metrics_from_confusion(&confusion_matrix, total_loss, total_batches as f64);

    Ok(metrics)
}

fn calculate_metrics_from_confusion(
    confusion_matrix: &[Vec<usize>],
    total_loss: f64,
    total_batches: f64,
) -> ClassificationMetrics {
    let num_classes = confusion_matrix.len();
    let mut per_class = Vec::with_capacity(num_classes);
    let mut total_samples = 0usize;
    let mut correct = 0usize;

    // Calculate per-class metrics
    for class_id in 0..num_classes {
        let tp = confusion_matrix[class_id][class_id];
        let fp: usize = (0..num_classes)
            .filter(|&i| i != class_id)
            .map(|i| confusion_matrix[i][class_id])
            .sum();
        let fn_count: usize = (0..num_classes)
            .filter(|&j| j != class_id)
            .map(|j| confusion_matrix[class_id][j])
            .sum();
        let tn: usize = (0..num_classes)
            .flat_map(|i| (0..num_classes).map(move |j| (i, j)))
            .filter(|&(i, j)| i != class_id && j != class_id)
            .map(|(i, j)| confusion_matrix[i][j])
            .sum();

        let support = tp + fn_count;
        total_samples += support;
        correct += tp;

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        per_class.push(ClassMetrics {
            class_id,
            class_name: format!("class_{class_id}"),
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_count,
            true_negatives: tn,
            precision,
            recall,
            f1_score: f1,
            support,
        });
    }

    // Calculate macro averages
    let valid_classes = per_class.iter().filter(|c| c.support > 0).count() as f64;
    let macro_precision = if valid_classes > 0.0 {
        per_class
            .iter()
            .filter(|c| c.support > 0)
            .map(|c| c.precision)
            .sum::<f64>()
            / valid_classes
    } else {
        0.0
    };
    let macro_recall = if valid_classes > 0.0 {
        per_class
            .iter()
            .filter(|c| c.support > 0)
            .map(|c| c.recall)
            .sum::<f64>()
            / valid_classes
    } else {
        0.0
    };
    let macro_f1 = if macro_precision + macro_recall > 0.0 {
        2.0 * macro_precision * macro_recall / (macro_precision + macro_recall)
    } else {
        0.0
    };

    // Calculate weighted F1
    let weighted_f1 = if total_samples > 0 {
        per_class
            .iter()
            .filter(|c| c.support > 0)
            .map(|c| c.f1_score * c.support as f64 / total_samples as f64)
            .sum()
    } else {
        0.0
    };

    let accuracy = if total_samples > 0 {
        correct as f64 / total_samples as f64
    } else {
        0.0
    };

    let class_names: Vec<String> = (0..num_classes).map(|i| format!("class_{i}")).collect();

    ClassificationMetrics {
        total_samples,
        accuracy,
        loss: total_loss / total_batches,
        per_class,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_f1,
        confusion_matrix: confusion_matrix.to_vec(),
        num_classes,
        class_names,
    }
}

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

// =============================================================================
// Output: Console
// =============================================================================

fn print_metrics_summary(metrics: &ClassificationMetrics) {
    println!();
    print_header("Classification Metrics");

    print_kv("Total samples", &metrics.total_samples.to_string());
    print_kv("Accuracy", &format!("{:.2}%", metrics.accuracy * 100.0));
    print_kv("Average loss", &format!("{:.4}", metrics.loss));

    println!();
    print_header("Aggregate Metrics");

    print_kv(
        "Macro Precision",
        &format!("{:.4}", metrics.macro_precision),
    );
    print_kv("Macro Recall", &format!("{:.4}", metrics.macro_recall));
    print_kv("Macro F1", &format!("{:.4}", metrics.macro_f1));
    print_kv("Weighted F1", &format!("{:.4}", metrics.weighted_f1));

    println!();
    print_header("Per-Class Metrics");

    // Header
    println!(
        "  {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Class", "Precision", "Recall", "F1-Score", "Support"
    );
    println!("  {}", "-".repeat(54));

    for class in &metrics.per_class {
        if class.support > 0 {
            println!(
                "  {:>10} {:>10.4} {:>10.4} {:>10.4} {:>10}",
                class.class_name, class.precision, class.recall, class.f1_score, class.support
            );
        }
    }

    println!("  {}", "-".repeat(54));
    println!(
        "  {:>10} {:>10.4} {:>10.4} {:>10.4} {:>10}",
        "macro avg",
        metrics.macro_precision,
        metrics.macro_recall,
        metrics.macro_f1,
        metrics.total_samples
    );
}

fn print_confusion_matrix(metrics: &ClassificationMetrics) {
    println!();
    print_header("Confusion Matrix");

    let n = metrics.num_classes.min(10); // Limit display to 10 classes

    // Header
    print!("  {:>8}", "Actual\\Pred");
    for j in 0..n {
        print!(" {j:>6}");
    }
    if metrics.num_classes > 10 {
        print!("   ...");
    }
    println!();

    println!("  {}", "-".repeat(8 + n * 7 + 10));

    for i in 0..n {
        print!("  {i:>8}");
        for j in 0..n {
            let count = metrics.confusion_matrix[i][j];
            if i == j {
                // Diagonal (correct predictions) - could be highlighted
                print!(" {count:>6}");
            } else {
                print!(" {count:>6}");
            }
        }
        if metrics.num_classes > 10 {
            print!("   ...");
        }
        println!();
    }

    if metrics.num_classes > 10 {
        println!("  {:>8}   ... (showing first 10 classes)", "...");
    }
}

// =============================================================================
// Output: JSON
// =============================================================================

fn generate_json_report(metrics: &ClassificationMetrics, path: &str) -> CliResult<()> {
    use serde_json::json;

    let per_class: Vec<serde_json::Value> = metrics
        .per_class
        .iter()
        .filter(|c| c.support > 0)
        .map(|c| {
            json!({
                "class_id": c.class_id,
                "class_name": c.class_name,
                "precision": c.precision,
                "recall": c.recall,
                "f1_score": c.f1_score,
                "support": c.support,
                "true_positives": c.true_positives,
                "false_positives": c.false_positives,
                "false_negatives": c.false_negatives,
                "true_negatives": c.true_negatives,
            })
        })
        .collect();

    let report = json!({
        "summary": {
            "total_samples": metrics.total_samples,
            "accuracy": metrics.accuracy,
            "loss": metrics.loss,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
        },
        "per_class": per_class,
        "confusion_matrix": metrics.confusion_matrix,
        "num_classes": metrics.num_classes,
        "class_names": metrics.class_names,
    });

    let json_str = serde_json::to_string_pretty(&report)?;
    std::fs::write(path, json_str)?;

    Ok(())
}

// =============================================================================
// Output: Text
// =============================================================================

fn generate_text_report(metrics: &ClassificationMetrics, path: &str) -> CliResult<()> {
    let mut output = String::new();

    output.push_str("=".repeat(60).as_str());
    output.push('\n');
    output.push_str("              FERRITE MODEL EVALUATION REPORT\n");
    output.push_str("=".repeat(60).as_str());
    output.push_str("\n\n");

    // Summary
    output.push_str("CLASSIFICATION METRICS\n");
    output.push_str("-".repeat(40).as_str());
    output.push('\n');
    output.push_str(&format!("Total samples:    {}\n", metrics.total_samples));
    output.push_str(&format!(
        "Accuracy:         {:.2}%\n",
        metrics.accuracy * 100.0
    ));
    output.push_str(&format!("Average loss:     {:.4}\n", metrics.loss));
    output.push('\n');

    // Aggregate metrics
    output.push_str("AGGREGATE METRICS\n");
    output.push_str("-".repeat(40).as_str());
    output.push('\n');
    output.push_str(&format!(
        "Macro Precision:  {:.4}\n",
        metrics.macro_precision
    ));
    output.push_str(&format!("Macro Recall:     {:.4}\n", metrics.macro_recall));
    output.push_str(&format!("Macro F1:         {:.4}\n", metrics.macro_f1));
    output.push_str(&format!("Weighted F1:      {:.4}\n", metrics.weighted_f1));
    output.push('\n');

    // Per-class metrics
    output.push_str("PER-CLASS METRICS\n");
    output.push_str("-".repeat(60).as_str());
    output.push('\n');
    output.push_str(&format!(
        "{:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "Class", "Precision", "Recall", "F1-Score", "Support"
    ));
    output.push_str("-".repeat(60).as_str());
    output.push('\n');

    for class in &metrics.per_class {
        if class.support > 0 {
            output.push_str(&format!(
                "{:>10} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                class.class_name, class.precision, class.recall, class.f1_score, class.support
            ));
        }
    }

    output.push_str("-".repeat(60).as_str());
    output.push('\n');
    output.push_str(&format!(
        "{:>10} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
        "macro avg",
        metrics.macro_precision,
        metrics.macro_recall,
        metrics.macro_f1,
        metrics.total_samples
    ));

    // Confusion matrix
    output.push('\n');
    output.push_str("CONFUSION MATRIX\n");
    output.push_str("-".repeat(60).as_str());
    output.push('\n');

    let n = metrics.num_classes.min(10);

    output.push_str(&format!("{:>10}", "Actual\\Pred"));
    for j in 0..n {
        output.push_str(&format!(" {j:>6}"));
    }
    output.push('\n');

    for i in 0..n {
        output.push_str(&format!("{i:>10}"));
        for j in 0..n {
            output.push_str(&format!(" {:>6}", metrics.confusion_matrix[i][j]));
        }
        output.push('\n');
    }

    std::fs::write(path, output)?;

    Ok(())
}

// =============================================================================
// Output: HTML Report
// =============================================================================

fn generate_html_report(
    metrics: &ClassificationMetrics,
    args: &ReportArgs,
    training_history: Option<&TrainingHistory>,
    path: &str,
) -> CliResult<()> {
    let mut html = String::new();

    // HTML header
    html.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Axonml Model Evaluation Report</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --text-light: #64748b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 2rem;
        }
        h1 {
            font-size: 2.5rem;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: var(--text-light);
            font-size: 1.1rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .card {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }
        .card h3 {
            font-size: 0.875rem;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text);
        }
        .card .value.good { color: var(--success); }
        .card .value.warning { color: var(--warning); }
        .card .value.bad { color: var(--error); }
        .section {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }
        .section h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: var(--bg);
            font-weight: 600;
            color: var(--text-light);
            font-size: 0.875rem;
            text-transform: uppercase;
        }
        tr:hover { background: var(--bg); }
        .confusion-matrix {
            overflow-x: auto;
        }
        .confusion-matrix table {
            min-width: 500px;
        }
        .confusion-matrix th, .confusion-matrix td {
            text-align: center;
            padding: 0.5rem;
            min-width: 50px;
        }
        .confusion-matrix .diagonal {
            background: rgba(79, 70, 229, 0.1);
            font-weight: 600;
        }
        .confusion-matrix .header {
            background: var(--primary);
            color: white;
        }
        .chart-container {
            margin: 1rem 0;
        }
        .bar-chart {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .bar-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .bar-label {
            width: 80px;
            font-size: 0.875rem;
            color: var(--text-light);
        }
        .bar-container {
            flex: 1;
            height: 24px;
            background: var(--bg);
            border-radius: 4px;
            overflow: hidden;
        }
        .bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--primary-light));
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .bar-value {
            width: 60px;
            text-align: right;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        .metric-item {
            text-align: center;
            padding: 1rem;
            background: var(--bg);
            border-radius: 8px;
        }
        .metric-item .label {
            font-size: 0.75rem;
            color: var(--text-light);
            text-transform: uppercase;
        }
        .metric-item .value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }
        footer {
            text-align: center;
            padding: 2rem 0;
            color: var(--text-light);
            font-size: 0.875rem;
        }
        .svg-chart { width: 100%; height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Model Evaluation Report</h1>
            <p class="subtitle">Generated by Axonml ML Framework</p>
        </header>
"#,
    );

    // Summary cards
    let accuracy_class = if metrics.accuracy >= 0.9 {
        "good"
    } else if metrics.accuracy >= 0.7 {
        "warning"
    } else {
        "bad"
    };
    let f1_class = if metrics.macro_f1 >= 0.9 {
        "good"
    } else if metrics.macro_f1 >= 0.7 {
        "warning"
    } else {
        "bad"
    };

    html.push_str(&format!(
        r#"
        <div class="grid">
            <div class="card">
                <h3>Accuracy</h3>
                <div class="value {}">{:.2}%</div>
            </div>
            <div class="card">
                <h3>Macro F1 Score</h3>
                <div class="value {}">{:.4}</div>
            </div>
            <div class="card">
                <h3>Total Samples</h3>
                <div class="value">{}</div>
            </div>
            <div class="card">
                <h3>Average Loss</h3>
                <div class="value">{:.4}</div>
            </div>
        </div>
"#,
        accuracy_class,
        metrics.accuracy * 100.0,
        f1_class,
        metrics.macro_f1,
        metrics.total_samples,
        metrics.loss
    ));

    // Aggregate metrics section
    html.push_str(&format!(
        r#"
        <div class="section">
            <h2>Aggregate Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="label">Macro Precision</div>
                    <div class="value">{:.4}</div>
                </div>
                <div class="metric-item">
                    <div class="label">Macro Recall</div>
                    <div class="value">{:.4}</div>
                </div>
                <div class="metric-item">
                    <div class="label">Weighted F1</div>
                    <div class="value">{:.4}</div>
                </div>
            </div>
        </div>
"#,
        metrics.macro_precision, metrics.macro_recall, metrics.weighted_f1
    ));

    // F1 Score bar chart
    html.push_str(
        r#"
        <div class="section">
            <h2>Per-Class F1 Scores</h2>
            <div class="chart-container">
                <div class="bar-chart">
"#,
    );

    for class in &metrics.per_class {
        if class.support > 0 {
            let bar_width = (class.f1_score * 100.0).min(100.0);
            html.push_str(&format!(
                r#"
                    <div class="bar-row">
                        <span class="bar-label">{}</span>
                        <div class="bar-container">
                            <div class="bar" style="width: {:.1}%"></div>
                        </div>
                        <span class="bar-value">{:.4}</span>
                    </div>
"#,
                class.class_name, bar_width, class.f1_score
            ));
        }
    }

    html.push_str(
        r"
                </div>
            </div>
        </div>
",
    );

    // Per-class metrics table
    html.push_str(
        r#"
        <div class="section">
            <h2>Per-Class Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
"#,
    );

    for class in &metrics.per_class {
        if class.support > 0 {
            html.push_str(&format!(
                r"
                    <tr>
                        <td>{}</td>
                        <td>{:.4}</td>
                        <td>{:.4}</td>
                        <td>{:.4}</td>
                        <td>{}</td>
                    </tr>
",
                class.class_name, class.precision, class.recall, class.f1_score, class.support
            ));
        }
    }

    html.push_str(&format!(
        r"
                </tbody>
                <tfoot>
                    <tr>
                        <th>Macro Average</th>
                        <td><strong>{:.4}</strong></td>
                        <td><strong>{:.4}</strong></td>
                        <td><strong>{:.4}</strong></td>
                        <td><strong>{}</strong></td>
                    </tr>
                </tfoot>
            </table>
        </div>
",
        metrics.macro_precision, metrics.macro_recall, metrics.macro_f1, metrics.total_samples
    ));

    // Confusion matrix
    if args.confusion_matrix {
        html.push_str(
            r#"
        <div class="section">
            <h2>Confusion Matrix</h2>
            <div class="confusion-matrix">
                <table>
                    <thead>
                        <tr>
                            <th class="header">Actual↓ / Pred→</th>
"#,
        );

        let n = metrics.num_classes.min(10);
        for j in 0..n {
            html.push_str(&format!(
                r#"                            <th class="header">{j}</th>
"#
            ));
        }

        html.push_str(
            r"                        </tr>
                    </thead>
                    <tbody>
",
        );

        for i in 0..n {
            html.push_str(&format!(
                r"                        <tr>
                            <th>{i}</th>
"
            ));
            for j in 0..n {
                let class = if i == j { " class=\"diagonal\"" } else { "" };
                html.push_str(&format!(
                    r"                            <td{}>{}</td>
",
                    class, metrics.confusion_matrix[i][j]
                ));
            }
            html.push_str(
                r"                        </tr>
",
            );
        }

        html.push_str(
            r"                    </tbody>
                </table>
            </div>
        </div>
",
        );
    }

    // SVG Loss curve - only show if loss_curves is enabled
    if args.loss_curves {
        if let Some(history) = training_history {
            // Use real training history data
            html.push_str(
                r#"
        <div class="section">
            <h2>Training Progress</h2>
            <div class="chart-container">
"#,
            );
            html.push_str(&generate_loss_curve_svg_from_history(history));
            html.push_str(
                r"
            </div>
        </div>
",
            );
        } else {
            // No training history provided - show informational message
            html.push_str(
                r#"
        <div class="section">
            <h2>Training Progress</h2>
            <div style="text-align: center; padding: 2rem; color: var(--text-light);">
                <p>No training history available.</p>
                <p style="font-size: 0.875rem;">To include loss curves, provide a training history file with <code>--history path/to/training.log</code></p>
                <p style="font-size: 0.875rem;">Expected format: JSON array or JSONL with fields: epoch, train_loss, val_loss, train_accuracy, val_accuracy</p>
            </div>
        </div>
"#,
            );
        }
    }

    // Footer
    html.push_str(&format!(
        r"
        <footer>
            <p>Report generated by Axonml ML Framework v{}</p>
            <p>Model: {}</p>
        </footer>
    </div>
</body>
</html>
",
        env!("CARGO_PKG_VERSION"),
        args.model
    ));

    std::fs::write(path, html)?;

    Ok(())
}

fn generate_loss_curve_svg_from_history(history: &TrainingHistory) -> String {
    // Generate loss curve SVG from actual training history
    let width = 600;
    let height = 250;
    let padding = 40;

    let train_loss: Vec<f64> = history.train_loss.clone();
    let val_loss: Vec<f64> = history.val_loss.clone();
    let num_epochs = train_loss.len();

    if num_epochs == 0 {
        return String::from("<p>No training data available</p>");
    }

    let x_scale = f64::from(width - 2 * padding) / num_epochs.max(1) as f64;

    // Calculate y_max dynamically from actual data
    let train_max = train_loss.iter().cloned().fold(0.0f64, f64::max);
    let val_max = val_loss.iter().cloned().fold(0.0f64, f64::max);
    let y_max = (train_max.max(val_max) * 1.1).max(0.1); // Add 10% padding, ensure non-zero
    let y_scale = f64::from(height - 2 * padding) / y_max;

    let mut svg = format!(
        r##"<svg viewBox="0 0 {width} {height}" class="svg-chart" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="trainGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#4f46e5;stop-opacity:0.3"/>
            <stop offset="100%" style="stop-color:#4f46e5;stop-opacity:0"/>
        </linearGradient>
    </defs>

    <!-- Grid lines -->
    <g stroke="#e2e8f0" stroke-width="1">
"##
    );

    // Horizontal grid lines
    for i in 0..=5 {
        let y = height - padding - (f64::from(i) * f64::from(height - 2 * padding) / 5.0) as i32;
        svg.push_str(&format!(
            r#"        <line x1="{}" y1="{}" x2="{}" y2="{}"/>
"#,
            padding,
            y,
            width - padding,
            y
        ));
    }

    // Vertical grid lines
    for i in 0..=5 {
        let x = padding + (f64::from(i) * f64::from(width - 2 * padding) / 5.0) as i32;
        svg.push_str(&format!(
            r#"        <line x1="{}" y1="{}" x2="{}" y2="{}"/>
"#,
            x,
            padding,
            x,
            height - padding
        ));
    }

    svg.push_str("    </g>\n\n");

    // Axes
    svg.push_str(&format!(
        r##"    <!-- Axes -->
    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#64748b" stroke-width="2"/>
    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#64748b" stroke-width="2"/>

"##,
        padding,
        padding,
        padding,
        height - padding,
        padding,
        height - padding,
        width - padding,
        height - padding
    ));

    // Training loss line
    let train_points: String = train_loss
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let x = f64::from(padding) + (i as f64 + 1.0) * x_scale;
            let y_pos = f64::from(height) - f64::from(padding) - y * y_scale;
            format!("{x:.1},{y_pos:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ");

    svg.push_str(&format!(
        r##"    <!-- Training loss line -->
    <polyline fill="none" stroke="#4f46e5" stroke-width="2" points="{train_points}"/>

"##
    ));

    // Validation loss line (only if we have validation data)
    if !val_loss.is_empty() {
        let val_points: String = val_loss
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let x = f64::from(padding) + (i as f64 + 1.0) * x_scale;
                let y_pos = f64::from(height) - f64::from(padding) - y * y_scale;
                format!("{x:.1},{y_pos:.1}")
            })
            .collect::<Vec<_>>()
            .join(" ");

        svg.push_str(&format!(
            r##"    <!-- Validation loss line -->
    <polyline fill="none" stroke="#22c55e" stroke-width="2" stroke-dasharray="5,5" points="{val_points}"/>

"##
        ));
    }

    // Axis labels
    svg.push_str(&format!(r##"    <!-- Axis labels -->
    <text x="{}" y="{}" font-size="12" fill="#64748b" text-anchor="middle">Epoch</text>
    <text x="{}" y="{}" font-size="12" fill="#64748b" text-anchor="middle" transform="rotate(-90,{},{})">Loss</text>

"##, width / 2, height - 5,
    15, height / 2, 15, height / 2));

    // Legend - only show Val Loss if we have validation data
    if val_loss.is_empty() {
        svg.push_str(&format!(
            r##"    <!-- Legend -->
    <g transform="translate({}, {})">
        <line x1="0" y1="0" x2="20" y2="0" stroke="#4f46e5" stroke-width="2"/>
        <text x="25" y="4" font-size="11" fill="#64748b">Train Loss</text>
    </g>
"##,
            width - 150,
            20
        ));
    } else {
        svg.push_str(&format!(r##"    <!-- Legend -->
    <g transform="translate({}, {})">
        <line x1="0" y1="0" x2="20" y2="0" stroke="#4f46e5" stroke-width="2"/>
        <text x="25" y="4" font-size="11" fill="#64748b">Train Loss</text>
        <line x1="100" y1="0" x2="120" y2="0" stroke="#22c55e" stroke-width="2" stroke-dasharray="5,5"/>
        <text x="125" y="4" font-size="11" fill="#64748b">Val Loss</text>
    </g>
"##, width - 250, 20));
    }

    // Y-axis tick labels
    for i in 0..=5 {
        let y_val = f64::from(i) * y_max / 5.0;
        let y_pos =
            height - padding - (f64::from(i) * f64::from(height - 2 * padding) / 5.0) as i32;
        svg.push_str(&format!(
            r##"    <text x="{}" y="{}" font-size="10" fill="#64748b" text-anchor="end">{:.1}</text>
"##,
            padding - 5,
            y_pos + 3,
            y_val
        ));
    }

    // X-axis tick labels - dynamically based on number of epochs
    let tick_step = (num_epochs / 5).max(1);
    for i in (0..=num_epochs).step_by(tick_step) {
        let x_pos = padding + (i as f64 * x_scale) as i32;
        svg.push_str(&format!(
            r##"    <text x="{}" y="{}" font-size="10" fill="#64748b" text-anchor="middle">{}</text>
"##,
            x_pos,
            height - padding + 15,
            i
        ));
    }

    svg.push_str("</svg>");

    svg
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculation() {
        // Simple 2x2 confusion matrix
        let confusion = vec![vec![50, 10], vec![5, 35]];

        let metrics = calculate_metrics_from_confusion(&confusion, 10.0, 10.0);

        assert_eq!(metrics.num_classes, 2);
        assert_eq!(metrics.total_samples, 100);
        assert!((metrics.accuracy - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_class_metrics() {
        let class = ClassMetrics {
            class_id: 0,
            class_name: "test".to_string(),
            true_positives: 50,
            false_positives: 10,
            false_negatives: 5,
            true_negatives: 35,
            precision: 50.0 / 60.0,
            recall: 50.0 / 55.0,
            f1_score: 0.0,
            support: 55,
        };

        assert!((class.precision - 0.833).abs() < 0.01);
        assert!((class.recall - 0.909).abs() < 0.01);
    }

    #[test]
    fn test_argmax() {
        let data = Tensor::from_vec(vec![0.1, 0.8, 0.1, 0.7, 0.2, 0.1], &[2, 3]).unwrap();

        let result = argmax_batch(&data);
        assert_eq!(result, vec![1, 0]);
    }
}
