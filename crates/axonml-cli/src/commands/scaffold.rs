//! Scaffold - Rust Training Project Generator
//!
//! Generates complete Rust training projects based on user's model
//! and dataset configuration. Creates ready-to-run Cargo projects.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::path::PathBuf;

use super::data::DatasetAnalysis;
use super::utils::{ensure_dir, path_exists, print_header, print_info, print_kv, print_success};
use crate::cli::{ScaffoldArgs, ScaffoldGenerateArgs, ScaffoldSubcommand, ScaffoldTemplatesArgs};
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `scaffold` command
pub fn execute(args: ScaffoldArgs) -> CliResult<()> {
    match args.action {
        ScaffoldSubcommand::Generate(gen_args) => execute_generate(gen_args),
        ScaffoldSubcommand::Templates(tmpl_args) => execute_templates(tmpl_args),
    }
}

// =============================================================================
// Generate Subcommand
// =============================================================================

fn execute_generate(args: ScaffoldGenerateArgs) -> CliResult<()> {
    print_header("Scaffold Rust Training Project");

    let project_name = args.name.clone();
    let project_path = PathBuf::from(&args.output).join(&project_name);

    // Check if output exists
    if path_exists(&project_path) && !args.overwrite {
        return Err(CliError::Other(format!(
            "Directory already exists: {}. Use --overwrite to replace.",
            project_path.display()
        )));
    }

    print_kv("Project", &project_name);
    print_kv("Template", &args.template);
    print_kv("Output", &project_path.display().to_string());

    // Load data analysis if provided
    let data_analysis = if let Some(data_path) = &args.data {
        load_data_analysis(data_path)?
    } else {
        None
    };

    // Determine task and architecture
    let task = args
        .task
        .clone()
        .or_else(|| data_analysis.as_ref().map(|a| a.task_type.clone()))
        .unwrap_or_else(|| "classification".to_string());

    let architecture = args
        .architecture
        .clone()
        .or_else(|| {
            data_analysis
                .as_ref()
                .map(|a| a.recommendations.architecture.clone())
        })
        .unwrap_or_else(|| "mlp".to_string());

    print_kv("Task", &task);
    print_kv("Architecture", &architecture);

    println!();
    print_info("Generating project...");

    // Create project structure
    if args.overwrite && project_path.exists() {
        fs::remove_dir_all(&project_path)?;
    }
    ensure_dir(project_path.display().to_string())?;

    // Generate files
    generate_cargo_toml(&project_path, &project_name, args.wandb)?;
    generate_main_rs(
        &project_path,
        &task,
        &architecture,
        data_analysis.as_ref(),
        args.model.as_deref(),
    )?;
    generate_lib_rs(&project_path, &architecture)?;
    generate_config_toml(&project_path, data_analysis.as_ref())?;
    generate_readme(&project_path, &project_name, &task, &architecture)?;

    // Create directories
    fs::create_dir_all(project_path.join("src"))?;
    fs::create_dir_all(project_path.join("data"))?;
    fs::create_dir_all(project_path.join("models"))?;
    fs::create_dir_all(project_path.join("output"))?;

    println!();
    print_success(&format!("Project created: {}", project_path.display()));
    print_header("Next Steps");
    println!("  1. cd {}", project_path.display());
    println!("  2. cargo build");
    println!("  3. cargo run -- train");
    println!();
    print_info("Edit axonml.toml to configure training parameters");

    Ok(())
}

fn load_data_analysis(path: &str) -> CliResult<Option<DatasetAnalysis>> {
    let path = PathBuf::from(path);

    if path.extension().is_some_and(|e| e == "json") {
        let content = fs::read_to_string(&path)?;
        let analysis: DatasetAnalysis = serde_json::from_str(&content)?;
        return Ok(Some(analysis));
    }

    let analysis_path = path.join("dataset_analysis.json");
    if analysis_path.exists() {
        let content = fs::read_to_string(&analysis_path)?;
        let analysis: DatasetAnalysis = serde_json::from_str(&content)?;
        return Ok(Some(analysis));
    }

    Ok(None)
}

// =============================================================================
// File Generators
// =============================================================================

fn generate_cargo_toml(path: &PathBuf, name: &str, include_wandb: bool) -> CliResult<()> {
    let wandb_dep = if include_wandb {
        "\n# Experiment tracking\naxonml-wandb = { path = \"../axonml-wandb\" }"
    } else {
        ""
    };

    let content = format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "Axonml ML training project"

[dependencies]
# Axonml ML Framework
axonml = "0.1"
axonml-core = "0.1"
axonml-tensor = "0.1"
axonml-autograd = "0.1"
axonml-nn = "0.1"
axonml-optim = "0.1"
axonml-data = "0.1"
axonml-vision = "0.1"
axonml-serialize = "0.1"
{wandb_dep}

# CLI and config
clap = {{ version = "4.5", features = ["derive"] }}
toml = "0.8"
serde = {{ version = "1.0", features = ["derive"] }}

# Progress display
indicatif = "0.17"

# Random
rand = "0.8"

[[bin]]
name = "{name}"
path = "src/main.rs"
"#
    );

    fs::write(path.join("Cargo.toml"), content)?;
    Ok(())
}

fn generate_main_rs(
    path: &PathBuf,
    task: &str,
    architecture: &str,
    data_analysis: Option<&DatasetAnalysis>,
    model_path: Option<&str>,
) -> CliResult<()> {
    let (batch_size, epochs, lr) = if let Some(a) = data_analysis {
        (
            a.recommendations.batch_size,
            a.recommendations.epochs,
            a.recommendations.learning_rate,
        )
    } else {
        (32, 10, 0.001)
    };

    let model_load = if let Some(mp) = model_path {
        format!(
            r#"
    // Load pretrained model
    let state_dict = axonml_serialize::load_state_dict("{mp}")?;
    model.load_state_dict(&state_dict)?;
    println!("Loaded pretrained weights");
"#
        )
    } else {
        String::new()
    };

    let content = format!(
        r#"//! {task} Training with Axonml
//!
//! Generated by: axonml scaffold generate
//! Architecture: {architecture}

use std::error::Error;

use clap::{{Parser, Subcommand}};
use axonml::prelude::*;
use axonml_autograd::Variable;
use axonml_data::{{Dataset, DataLoader}};
use axonml_nn::{{Module, Sequential, Linear, ReLU, Dropout, CrossEntropyLoss}};
use axonml_optim::{{Adam, Optimizer}};
use axonml_serialize::{{save_state_dict, Format}};
use axonml_vision::SyntheticMNIST;
use indicatif::{{ProgressBar, ProgressStyle}};

mod model;

#[derive(Parser)]
#[command(name = "training", about = "Axonml ML Training")]
struct Cli {{
    #[command(subcommand)]
    command: Commands,
}}

#[derive(Subcommand)]
enum Commands {{
    /// Train the model
    Train {{
        /// Number of epochs
        #[arg(short, long, default_value = "{epochs}")]
        epochs: usize,

        /// Batch size
        #[arg(short, long, default_value = "{batch_size}")]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value = "{lr}")]
        lr: f64,
    }},
    /// Evaluate the model
    Eval {{
        /// Path to model checkpoint
        #[arg(short, long, default_value = "output/model.axonml")]
        model: String,
    }},
}}

fn main() -> Result<(), Box<dyn Error>> {{
    let cli = Cli::parse();

    match cli.command {{
        Commands::Train {{ epochs, batch_size, lr }} => {{
            train(epochs, batch_size, lr)?;
        }}
        Commands::Eval {{ model }} => {{
            evaluate(&model)?;
        }}
    }}

    Ok(())
}}

fn train(epochs: usize, batch_size: usize, lr: f64) -> Result<(), Box<dyn Error>> {{
    println!("=== Axonml Training ===");
    println!("Epochs: {{}}", epochs);
    println!("Batch size: {{}}", batch_size);
    println!("Learning rate: {{}}", lr);
    println!();

    // Create model
    let mut model = model::create_model();
    model.train();
    {model_load}
    // Create optimizer
    let mut optimizer = Adam::new(model.parameters(), lr as f32);
    let loss_fn = CrossEntropyLoss::new();

    // Load dataset
    println!("Loading dataset...");
    let train_dataset = SyntheticMNIST::new(10000);
    let train_loader = DataLoader::new(train_dataset, batch_size);
    println!("Batches per epoch: {{}}", train_loader.len());
    println!();

    // Training loop
    let mut best_loss = f64::INFINITY;

    for epoch in 1..=epochs {{
        let pb = ProgressBar::new(train_loader.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("Epoch {{msg}} [{{bar:40}}] {{pos}}/{{len}}")
            .unwrap());
        pb.set_message(format!("{{}}/{{}}", epoch, epochs));

        let mut epoch_loss = 0.0;
        let mut correct = 0usize;
        let mut total = 0usize;

        for batch in train_loader.iter() {{
            let input = Variable::new(batch.data.clone(), false);
            let target = Variable::new(batch.targets.clone(), false);

            // Forward pass
            let output = model.forward(&input);
            let loss = loss_fn.compute(&output, &target);
            let loss_val = loss.data().to_vec()[0] as f64;
            epoch_loss += loss_val;

            // Compute accuracy
            let preds = argmax_batch(&output.data());
            let labels = argmax_batch(&batch.targets);
            for (p, l) in preds.iter().zip(labels.iter()) {{
                if p == l {{ correct += 1; }}
                total += 1;
            }}

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            pb.inc(1);
        }}

        pb.finish_and_clear();

        let avg_loss = epoch_loss / train_loader.len() as f64;
        let accuracy = correct as f64 / total as f64 * 100.0;

        println!("Epoch {{}}/{{}}: loss={{:.4}}, accuracy={{:.2}}%", epoch, epochs, avg_loss, accuracy);

        // Save best model
        if avg_loss < best_loss {{
            best_loss = avg_loss;
            let state_dict = model.state_dict();
            save_state_dict(&state_dict, "output/model.axonml", Format::Axonml)?;
        }}
    }}

    println!();
    println!("Training complete! Best loss: {{:.4}}", best_loss);
    println!("Model saved to: output/model.axonml");

    Ok(())
}}

fn evaluate(model_path: &str) -> Result<(), Box<dyn Error>> {{
    println!("=== Model Evaluation ===");
    println!("Model: {{}}", model_path);
    println!();

    // Load model
    let model = model::create_model();
    let state_dict = axonml_serialize::load_state_dict(model_path)?;
    // model.load_state_dict(&state_dict)?;

    // Load test data
    let test_dataset = SyntheticMNIST::new(1000);
    let test_loader = DataLoader::new(test_dataset, 32);

    // Evaluate
    let mut correct = 0usize;
    let mut total = 0usize;

    for batch in test_loader.iter() {{
        let input = Variable::new(batch.data.clone(), false);
        let output = model.forward(&input);

        let preds = argmax_batch(&output.data());
        let labels = argmax_batch(&batch.targets);

        for (p, l) in preds.iter().zip(labels.iter()) {{
            if p == l {{ correct += 1; }}
            total += 1;
        }}
    }}

    let accuracy = correct as f64 / total as f64 * 100.0;
    println!("Test Accuracy: {{:.2}}%", accuracy);

    Ok(())
}}

fn argmax_batch(tensor: &axonml_tensor::Tensor<f32>) -> Vec<usize> {{
    let shape = tensor.shape();
    let data = tensor.to_vec();
    let batch_size = shape[0];
    let num_classes = shape[1];

    (0..batch_size)
        .map(|b| {{
            let start = b * num_classes;
            let end = start + num_classes;
            let slice = &data[start..end];
            slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }})
        .collect()
}}
"#
    );

    fs::write(path.join("src").join("main.rs"), content)?;
    Ok(())
}

fn generate_lib_rs(path: &PathBuf, architecture: &str) -> CliResult<()> {
    let model_code = match architecture.to_lowercase().as_str() {
        "cnn" | "conv" => {
            r"//! Model definition

use axonml_nn::{Module, Sequential, Linear, Conv2d, MaxPool2d, ReLU, Dropout};
use axonml_autograd::Variable;
use axonml_serialize::StateDict;

pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    pool: MaxPool2d,
    dropout: Dropout,
}

impl Model {
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, 3),
            conv2: Conv2d::new(32, 64, 3),
            fc1: Linear::new(64 * 5 * 5, 128),
            fc2: Linear::new(128, 10),
            pool: MaxPool2d::new(2),
            dropout: Dropout::new(0.25),
        }
    }
}

impl Module for Model {
    fn forward(&self, input: &Variable) -> Variable {
        let x = self.conv1.forward(input);
        let x = x.relu();
        let x = self.pool.forward(&x);

        let x = self.conv2.forward(&x);
        let x = x.relu();
        let x = self.pool.forward(&x);

        // Flatten
        let shape = x.shape();
        let batch_size = shape[0];
        let flat_size: usize = shape[1..].iter().product();
        let flat_data = x.data().to_vec();
        let x = Variable::new(
            axonml_tensor::Tensor::from_vec(flat_data, &[batch_size, flat_size]).unwrap(),
            x.requires_grad()
        );

        let x = self.fc1.forward(&x);
        let x = x.relu();
        let x = self.dropout.forward(&x);
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

pub fn create_model() -> Sequential {
    Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Dropout::new(0.2))
        .add(Linear::new(256, 128))
        .add(ReLU)
        .add(Dropout::new(0.2))
        .add(Linear::new(128, 10))
}
"
        }
        _ => {
            r"//! Model definition

use axonml_nn::{Module, Sequential, Linear, ReLU, Dropout};
use axonml_autograd::Variable;
use axonml_serialize::StateDict;

pub fn create_model() -> Sequential {
    Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Dropout::new(0.2))
        .add(Linear::new(256, 128))
        .add(ReLU)
        .add(Dropout::new(0.2))
        .add(Linear::new(128, 10))
}
"
        }
    };

    fs::write(path.join("src").join("model.rs"), model_code)?;
    Ok(())
}

fn generate_config_toml(path: &PathBuf, analysis: Option<&DatasetAnalysis>) -> CliResult<()> {
    let (batch_size, epochs, lr, optimizer) = if let Some(a) = analysis {
        (
            a.recommendations.batch_size,
            a.recommendations.epochs,
            a.recommendations.learning_rate,
            a.recommendations.optimizer.clone(),
        )
    } else {
        (32, 10, 0.001, "adam".to_string())
    };

    let content = format!(
        r#"# Axonml Training Configuration

[project]
name = "training"
version = "0.1.0"

[model]
architecture = "mlp"
input_size = 784
hidden_sizes = [256, 128]
num_classes = 10
dropout = 0.2

[training]
epochs = {epochs}
batch_size = {batch_size}
learning_rate = {lr}

[training.optimizer]
name = "{optimizer}"
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.0

[data]
path = "./data"
format = "auto"
train_split = 0.8
val_split = 0.1
test_split = 0.1

[output]
dir = "./output"
checkpoint_frequency = 5
save_best_only = true
"#
    );

    fs::write(path.join("axonml.toml"), content)?;
    Ok(())
}

fn generate_readme(path: &PathBuf, name: &str, task: &str, architecture: &str) -> CliResult<()> {
    let content = format!(
        r"# {name}

A Axonml ML training project.

## Task
- **Type:** {task}
- **Architecture:** {architecture}

## Quick Start

```bash
# Build the project
cargo build --release

# Train the model
cargo run --release -- train

# Train with custom parameters
cargo run --release -- train --epochs 20 --batch-size 64 --lr 0.0001

# Evaluate the model
cargo run --release -- eval --model output/model.axonml
```

## Project Structure

```
{name}/
├── Cargo.toml          # Rust dependencies
├── axonml.toml        # Training configuration
├── src/
│   ├── main.rs         # Training/eval entry point
│   └── model.rs        # Model definition
├── data/               # Dataset directory
├── models/             # Pretrained models
└── output/             # Training outputs
```

## Configuration

Edit `axonml.toml` to customize:
- Model architecture
- Training hyperparameters
- Data loading settings
- Output options

## Generated by Axonml CLI

This project was scaffolded using:
```bash
axonml scaffold generate {name}
```
"
    );

    fs::write(path.join("README.md"), content)?;
    Ok(())
}

// =============================================================================
// Templates Subcommand
// =============================================================================

fn execute_templates(args: ScaffoldTemplatesArgs) -> CliResult<()> {
    print_header("Available Project Templates");
    println!();

    let templates = [
        (
            "training",
            "Complete training pipeline with model, optimizer, and data loading",
        ),
        ("minimal", "Minimal training setup for quick experiments"),
        ("distributed", "Multi-GPU distributed training setup"),
        ("transfer", "Transfer learning / fine-tuning project"),
    ];

    for (name, desc) in templates {
        print_kv("Template", name);
        if args.detailed {
            println!("  {desc}");
        }
        println!();
    }

    print_info("Use: axonml scaffold generate <name> --template <template>");

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_cargo_toml() {
        let temp_dir = std::env::temp_dir().join("axonml_test_scaffold");
        let _ = fs::create_dir_all(&temp_dir);

        generate_cargo_toml(&temp_dir, "test_project", false).unwrap();

        let content = fs::read_to_string(temp_dir.join("Cargo.toml")).unwrap();
        assert!(content.contains("test_project"));
        assert!(content.contains("axonml"));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
