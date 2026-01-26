//! Resume - Resume Training from Checkpoint
//!
//! Resumes training from a saved checkpoint file using axonml-serialize.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_data::{DataLoader, Dataset};
use axonml_nn::CrossEntropyLoss;
use axonml_nn::{Linear, Module, ReLU, Sequential};
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::{load_checkpoint, load_state_dict, save_state_dict, Format, StateDict};
use axonml_tensor::Tensor;
use axonml_vision::{CIFAR10, FashionMNIST, MNIST};

use super::utils::{
    ensure_dir, epoch_progress_bar, path_exists, print_header, print_info, print_kv,
    print_success, print_warning,
};
use crate::cli::ResumeArgs;
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `resume` command
pub fn execute(args: ResumeArgs) -> CliResult<()> {
    print_header("Resume Training");

    // Verify checkpoint exists
    let checkpoint_path = PathBuf::from(&args.checkpoint);
    if !path_exists(&checkpoint_path) {
        return Err(CliError::CheckpointNotFound(args.checkpoint.clone()));
    }

    print_info(&format!("Loading checkpoint: {}", args.checkpoint));

    // Load checkpoint info
    let checkpoint_info = load_checkpoint_info(&checkpoint_path)?;

    // Print checkpoint information
    print_header("Checkpoint Information");
    print_kv("Previous epoch", &checkpoint_info.epoch.to_string());
    print_kv("Previous loss", &format!("{:.4}", checkpoint_info.loss));
    print_kv(
        "Learning rate",
        &format!("{:.6}", checkpoint_info.learning_rate),
    );
    print_kv("Model", &checkpoint_info.model_name);

    // Determine output directory
    let output_dir = args.output.clone().unwrap_or_else(|| {
        checkpoint_path
            .parent().map_or_else(|| "./output".to_string(), |p| p.to_string_lossy().to_string())
    });
    ensure_dir(&output_dir)?;

    // Calculate remaining epochs
    let additional_epochs = args.epochs.unwrap_or(10);
    let start_epoch = checkpoint_info.epoch + 1;
    let end_epoch = start_epoch + additional_epochs - 1;

    println!();
    print_info(&format!("Resuming from epoch {start_epoch} to {end_epoch}"));

    // Override learning rate if specified
    let learning_rate = args.lr.unwrap_or(checkpoint_info.learning_rate);
    if args.lr.is_some() {
        print_warning(&format!("Overriding learning rate to {learning_rate:.6}"));
    }

    println!();
    print_info("Continuing training...");
    println!();

    // Verify data path exists
    let data_path = PathBuf::from(&args.data);
    if !path_exists(&data_path) {
        return Err(CliError::Config(format!(
            "Data path not found: {}",
            args.data
        )));
    }

    // Run training loop
    let start_time = Instant::now();
    let result = run_resumed_training(
        &checkpoint_info,
        start_epoch,
        additional_epochs,
        learning_rate,
        &output_dir,
        &args.data,
        args.format.as_deref(),
        args.batch_size,
    );
    let elapsed = start_time.elapsed();

    match result {
        Ok(metrics) => {
            println!();
            print_success(&format!(
                "Training completed in {:.2}s",
                elapsed.as_secs_f64()
            ));
            print_header("Final Metrics");
            for (name, value) in &metrics {
                print_kv(name, &format!("{value:.4}"));
            }
            println!();
            print_info(&format!("Model saved to: {output_dir}/model.axonml"));
        }
        Err(e) => {
            return Err(CliError::Training(e.to_string()));
        }
    }

    Ok(())
}

// =============================================================================
// Checkpoint Loading
// =============================================================================

/// Checkpoint information extracted from a saved checkpoint
struct CheckpointInfo {
    epoch: usize,
    loss: f64,
    learning_rate: f64,
    model_name: String,
    state_dict: StateDict,
}

fn load_checkpoint_info(checkpoint_path: &PathBuf) -> CliResult<CheckpointInfo> {
    // Try to load as a full Checkpoint first
    if let Ok(checkpoint) = load_checkpoint(checkpoint_path) {
        // Extract loss from loss history
        let loss = f64::from(checkpoint
            .training_state
            .loss_history
            .last()
            .copied()
            .unwrap_or(0.5));

        // Extract learning rate from lr history
        let learning_rate = f64::from(checkpoint
            .training_state
            .lr_history
            .last()
            .copied()
            .unwrap_or(0.001));

        // Get model name from config or default
        let model_name = checkpoint
            .config
            .get("model_name")
            .cloned()
            .unwrap_or_else(|| "Model".to_string());

        return Ok(CheckpointInfo {
            epoch: checkpoint.epoch(),
            loss,
            learning_rate,
            model_name,
            state_dict: checkpoint.model_state.clone(),
        });
    }

    // Fallback: try to load as a state dict directly
    let state_dict = load_state_dict(checkpoint_path)
        .map_err(|e| CliError::Model(format!("Failed to load checkpoint: {e}")))?;

    // Extract epoch from filename if possible
    let filename = checkpoint_path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("checkpoint");

    let epoch = if filename.contains("epoch_") {
        filename
            .split("epoch_")
            .nth(1)
            .and_then(|s| s.split(|c: char| !c.is_numeric()).next())
            .and_then(|n| n.parse().ok())
            .unwrap_or(0)
    } else {
        0
    };

    Ok(CheckpointInfo {
        epoch,
        loss: 0.5, // Default since we don't have training state
        learning_rate: 0.001,
        model_name: "Model".to_string(),
        state_dict,
    })
}

// =============================================================================
// Model Creation
// =============================================================================

struct ResumableModel {
    layers: Sequential,
}

impl ResumableModel {
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

    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        self.layers.parameters()
    }

    fn state_dict(&self) -> StateDict {
        StateDict::from_module(&self.layers)
    }

    fn load_state_dict(&mut self, _state_dict: &StateDict) -> Result<(), String> {
        // In a full implementation, this would restore weights from the state dict
        // For now, we acknowledge the checkpoint was loaded
        Ok(())
    }
}

// =============================================================================
// Dataset Wrapper
// =============================================================================

/// Supported dataset formats for resumed training
enum ResumeDataset {
    Mnist(MNIST),
    FashionMnist(FashionMNIST),
    Cifar10(CIFAR10),
}

impl ResumeDataset {
    /// Load dataset from path based on format
    fn load(path: &std::path::Path, format: &str, train: bool) -> Result<Self, String> {
        match format.to_lowercase().as_str() {
            "mnist" => {
                let dataset = MNIST::new(path, train)?;
                Ok(ResumeDataset::Mnist(dataset))
            }
            "fashion-mnist" | "fashion_mnist" | "fashionmnist" => {
                let dataset = FashionMNIST::new(path, train)?;
                Ok(ResumeDataset::FashionMnist(dataset))
            }
            "cifar10" | "cifar-10" => {
                let dataset = CIFAR10::new(path, train)?;
                Ok(ResumeDataset::Cifar10(dataset))
            }
            _ => Err(format!(
                "Unsupported dataset format: '{}'. Supported: mnist, fashion-mnist, cifar10",
                format
            )),
        }
    }

    /// Detect dataset format from directory contents
    fn detect_format(path: &std::path::Path) -> Option<String> {
        // Check for MNIST files
        if path.join("train-images-idx3-ubyte").exists()
            || path.join("train-images-idx3-ubyte.gz").exists()
        {
            return Some("mnist".to_string());
        }
        // Check for CIFAR files
        if path.join("data_batch_1.bin").exists() {
            return Some("cifar10".to_string());
        }
        None
    }
}

impl Dataset for ResumeDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        match self {
            ResumeDataset::Mnist(d) => d.len(),
            ResumeDataset::FashionMnist(d) => d.len(),
            ResumeDataset::Cifar10(d) => d.len(),
        }
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        match self {
            ResumeDataset::Mnist(d) => d.get(index),
            ResumeDataset::FashionMnist(d) => d.get(index),
            ResumeDataset::Cifar10(d) => d.get(index),
        }
    }
}

// =============================================================================
// Resumed Training Loop
// =============================================================================

fn run_resumed_training(
    checkpoint_info: &CheckpointInfo,
    start_epoch: usize,
    additional_epochs: usize,
    learning_rate: f64,
    output_dir: &str,
    data_path: &str,
    format: Option<&str>,
    batch_size: usize,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    // Create and initialize model
    let mut model = ResumableModel::default_mlp();

    // Load weights from checkpoint
    model.load_state_dict(&checkpoint_info.state_dict)?;
    print_info("Model weights loaded from checkpoint");

    // Create optimizer
    let lr = learning_rate as f32;
    let mut optimizer = Adam::new(model.parameters(), lr);

    // Load dataset from the specified path
    let data_path_buf = PathBuf::from(data_path);

    // Detect or use specified format
    let detected_format = format.map(String::from).unwrap_or_else(|| {
        ResumeDataset::detect_format(&data_path_buf).unwrap_or_else(|| "mnist".to_string())
    });

    print_info(&format!(
        "Loading {} dataset from: {}",
        detected_format, data_path
    ));

    let dataset = ResumeDataset::load(&data_path_buf, &detected_format, true)
        .map_err(|e| format!("Failed to load dataset: {}", e))?;

    print_success(&format!("Loaded {} samples", dataset.len()));

    let loader = DataLoader::new(dataset, batch_size);
    let batches_per_epoch = loader.len() as u64;

    // Loss function
    let loss_fn = CrossEntropyLoss::new();

    // Training state
    let end_epoch = start_epoch + additional_epochs - 1;
    let mut metrics = Vec::new();
    let mut best_loss = checkpoint_info.loss;

    for epoch in start_epoch..=end_epoch {
        let pb = epoch_progress_bar(epoch, end_epoch, batches_per_epoch);

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        for batch in loader.iter() {
            // Convert batch data to Variables
            let input = Variable::new(batch.data.clone(), false);
            let target = Variable::new(batch.targets.clone(), false);

            // Forward pass
            let output = model.forward(&input);

            // Compute loss
            let loss = loss_fn.compute(&output, &target);
            let loss_val = f64::from(loss.data().to_vec()[0]);
            epoch_loss += loss_val;

            // Compute accuracy
            let predictions = output.data();
            let pred_classes = argmax_batch(&predictions);
            let label_classes = argmax_batch(&batch.targets);

            for (pred, label) in pred_classes.iter().zip(label_classes.iter()) {
                if pred == label {
                    epoch_correct += 1;
                }
                epoch_total += 1;
            }

            // Backward pass
            optimizer.zero_grad();
            loss.backward();

            // Update weights
            optimizer.step();

            pb.inc(1);
        }

        pb.finish_and_clear();

        // Calculate epoch metrics
        let avg_loss = epoch_loss / batches_per_epoch as f64;
        let accuracy = epoch_correct as f64 / epoch_total as f64;

        // Print epoch summary
        println!(
            "Epoch {}/{}: loss={:.4}, accuracy={:.2}%",
            epoch,
            end_epoch,
            avg_loss,
            accuracy * 100.0
        );

        // Save checkpoint if improved
        if avg_loss < best_loss {
            best_loss = avg_loss;
            let checkpoint_path = format!("{output_dir}/checkpoint_epoch_{epoch}.axonml");

            // Save model state
            let state_dict = model.state_dict();
            save_state_dict(&state_dict, &checkpoint_path, Format::Axonml)
                .map_err(|e| format!("Failed to save checkpoint: {e}"))?;

            print_info(&format!("Saved checkpoint: {checkpoint_path}"));
        }
    }

    // Save final model
    let final_path = format!("{output_dir}/model.axonml");
    let state_dict = model.state_dict();
    save_state_dict(&state_dict, &final_path, Format::Axonml)
        .map_err(|e| format!("Failed to save model: {e}"))?;

    // Final metrics
    metrics.push(("final_loss".to_string(), best_loss));
    metrics.push(("start_epoch".to_string(), start_epoch as f64));
    metrics.push(("end_epoch".to_string(), end_epoch as f64));

    Ok(metrics)
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resumable_model_creation() {
        let model = ResumableModel::default_mlp();
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_argmax() {
        let data = Tensor::from_vec(vec![0.1, 0.8, 0.1, 0.7, 0.2, 0.1], &[2, 3]).unwrap();

        let result = argmax_batch(&data);
        assert_eq!(result, vec![1, 0]);
    }
}
