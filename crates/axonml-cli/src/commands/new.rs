//! New - Create New Axonml Project
//!
//! Creates a new Axonml project with the standard directory structure.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::path::PathBuf;

use super::utils::{ensure_dir, print_info, print_step, print_success};
use crate::cli::NewArgs;
use crate::config::ProjectConfig;
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `new` command
pub fn execute(args: NewArgs) -> CliResult<()> {
    let project_name = &args.name;

    // Determine project directory
    let base_path = args.path.map_or_else(|| PathBuf::from("."), PathBuf::from);
    let project_path = base_path.join(project_name);

    // Check if project already exists
    if project_path.exists() {
        return Err(CliError::ProjectExists(project_path.display().to_string()));
    }

    println!();
    print_info(&format!("Creating new Axonml project: {project_name}"));
    println!();

    // Create project structure
    print_step(1, 5, "Creating directory structure...");
    create_directory_structure(&project_path)?;

    print_step(2, 5, "Generating configuration files...");
    create_config_files(&project_path, project_name)?;

    print_step(3, 5, "Creating source files...");
    create_source_files(&project_path, project_name, &args.template)?;

    print_step(4, 5, "Creating data directories...");
    create_data_directories(&project_path)?;

    if args.no_git {
        print_step(5, 5, "Skipping git initialization...");
    } else {
        print_step(5, 5, "Initializing git repository...");
        init_git_repo(&project_path)?;
    }

    println!();
    print_success(&format!(
        "Created project '{}' at {}",
        project_name,
        project_path.display()
    ));
    println!();
    print_info("Get started with:");
    println!("  cd {project_name}");
    println!("  axonml train");
    println!();

    Ok(())
}

// =============================================================================
// Directory Structure
// =============================================================================

fn create_directory_structure(project_path: &PathBuf) -> CliResult<()> {
    // Create main project directory
    ensure_dir(project_path)?;

    // Create subdirectories
    let dirs = [
        "src",
        "src/models",
        "src/data",
        "data/train",
        "data/val",
        "data/test",
        "configs",
        "checkpoints",
        "logs",
        "outputs",
    ];

    for dir in &dirs {
        ensure_dir(project_path.join(dir))?;
    }

    Ok(())
}

// =============================================================================
// Configuration Files
// =============================================================================

fn create_config_files(project_path: &PathBuf, project_name: &str) -> CliResult<()> {
    // Create axonml.toml
    let config = ProjectConfig::new(project_name);
    config.save(project_path.join("axonml.toml"))?;

    // Create .gitignore
    let gitignore = r"# Axonml project gitignore

# Output directories
/checkpoints/
/logs/
/outputs/

# Data (typically large)
/data/train/
/data/val/
/data/test/

# Python virtual environment (if using Python tools)
venv/
.venv/
__pycache__/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Model files (typically large)
*.axonml
*.onnx
*.pt
*.pth
*.safetensors

# Temporary files
*.tmp
*.log
";
    fs::write(project_path.join(".gitignore"), gitignore)?;

    // Create training config
    let train_config = r#"# Training Configuration
# This file can be used with: axonml train --config configs/train.toml

[training]
epochs = 10
batch_size = 32
learning_rate = 0.001

[training.optimizer]
name = "adam"
weight_decay = 0.0001
beta1 = 0.9
beta2 = 0.999

[training.scheduler]
name = "cosine"
t_max = 10
eta_min = 0.00001

[model]
architecture = "custom"
path = "src/models/model.rs"

[data]
train_path = "data/train"
val_path = "data/val"
shuffle = true
augmentation = true
"#;
    fs::write(project_path.join("configs/train.toml"), train_config)?;

    Ok(())
}

// =============================================================================
// Source Files
// =============================================================================

fn create_source_files(
    project_path: &PathBuf,
    project_name: &str,
    template: &str,
) -> CliResult<()> {
    // Create main.rs
    let main_rs = format!(
        r#"//! {project_name} - Main Entry Point
//!
//! Training script for the {project_name} model.

use axonml::prelude::*;

mod models;
mod data;

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    println!("Starting {project_name} training...");

    // Load configuration
    // let config = load_config("axonml.toml")?;

    // Create model
    let model = models::create_model();
    println!("Model created: {{}} parameters", model.num_parameters());

    // Create optimizer
    let optimizer = Adam::new(model.parameters(), 0.001);

    // Training loop would go here
    println!("Training complete!");

    Ok(())
}}
"#
    );
    fs::write(project_path.join("src/main.rs"), main_rs)?;

    // Create model file based on template
    let model_rs = match template {
        "cnn" => create_cnn_template(project_name),
        "transformer" => create_transformer_template(project_name),
        "mlp" => create_mlp_template(project_name),
        _ => create_default_template(project_name),
    };
    fs::write(project_path.join("src/models/mod.rs"), model_rs)?;

    // Create data module
    let data_rs = format!(
        r"//! Data Module for {project_name}
//!
//! Data loading and preprocessing utilities.

use axonml::prelude::*;

/// Custom dataset for this project
pub struct CustomDataset {{
    // Add your data fields here
}}

impl CustomDataset {{
    pub fn new() -> Self {{
        Self {{}}
    }}
}}

/// Create a data loader for training
pub fn create_train_loader(batch_size: usize) -> DataLoader<CustomDataset> {{
    let dataset = CustomDataset::new();
    DataLoader::new(dataset, batch_size)
}}

/// Create a data loader for validation
pub fn create_val_loader(batch_size: usize) -> DataLoader<CustomDataset> {{
    let dataset = CustomDataset::new();
    DataLoader::new(dataset, batch_size)
}}
"
    );
    fs::write(project_path.join("src/data/mod.rs"), data_rs)?;

    // Create README.md
    let readme = format!(
        r"# {project_name}

A machine learning project built with Axonml.

## Project Structure

```
{project_name}/
├── axonml.toml          # Project configuration
├── src/
│   ├── main.rs          # Training entry point
│   ├── models/          # Model definitions
│   └── data/            # Data loading utilities
├── configs/             # Training configurations
├── data/                # Dataset directories
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
└── outputs/             # Model outputs
```

## Getting Started

1. Add your training data to `data/train/`
2. Configure training in `axonml.toml` or `configs/train.toml`
3. Run training:

```bash
axonml train
```

## Training

```bash
# Train with default configuration
axonml train

# Train with custom configuration
axonml train --config configs/train.toml

# Resume from checkpoint
axonml resume checkpoints/latest.axonml
```

## Evaluation

```bash
# Evaluate on test set
axonml eval outputs/model.axonml data/test

# Make predictions
axonml predict outputs/model.axonml input.json
```

## License

MIT
"
    );
    fs::write(project_path.join("README.md"), readme)?;

    Ok(())
}

fn create_default_template(project_name: &str) -> String {
    format!(
        r"//! Model Module for {project_name}
//!
//! Neural network model definitions.

use axonml::prelude::*;

/// Main model for this project
pub struct Model {{
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}}

impl Model {{
    pub fn new(input_size: usize, hidden_size: usize, num_classes: usize) -> Self {{
        Self {{
            fc1: Linear::new(input_size, hidden_size),
            fc2: Linear::new(hidden_size, hidden_size),
            fc3: Linear::new(hidden_size, num_classes),
        }}
    }}

    pub fn num_parameters(&self) -> usize {{
        // Count total parameters from all layers
        self.parameters().iter().map(|p| p.data().len()).sum()
    }}
}}

impl Module for Model {{
    fn forward(&self, input: &Variable) -> Variable {{
        let x = self.fc1.forward(input);
        let x = x.relu();
        let x = self.fc2.forward(&x);
        let x = x.relu();
        self.fc3.forward(&x)
    }}

    fn parameters(&self) -> Vec<Variable> {{
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }}

    fn train_mode(&mut self, _mode: bool) {{}}
}}

/// Create the model with default configuration
pub fn create_model() -> Model {{
    Model::new(784, 256, 10)
}}
"
    )
}

fn create_cnn_template(project_name: &str) -> String {
    format!(
        r"//! CNN Model for {project_name}
//!
//! Convolutional neural network model.

use axonml::prelude::*;

/// CNN model for image classification
pub struct CnnModel {{
    conv1: Conv2d,
    conv2: Conv2d,
    pool: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
}}

impl CnnModel {{
    pub fn new(num_classes: usize) -> Self {{
        Self {{
            conv1: Conv2d::new(1, 32, (3, 3)),
            conv2: Conv2d::new(32, 64, (3, 3)),
            pool: MaxPool2d::new((2, 2)),
            fc1: Linear::new(64 * 5 * 5, 128),
            fc2: Linear::new(128, num_classes),
        }}
    }}

    pub fn num_parameters(&self) -> usize {{
        self.parameters().iter().map(|p| p.data().len()).sum()
    }}
}}

impl Module for CnnModel {{
    fn forward(&self, input: &Variable) -> Variable {{
        let x = self.conv1.forward(input);
        let x = x.relu();
        let x = self.pool.forward(&x);
        let x = self.conv2.forward(&x);
        let x = x.relu();
        let x = self.pool.forward(&x);
        // Flatten
        let x = self.fc1.forward(&x);
        let x = x.relu();
        self.fc2.forward(&x)
    }}

    fn parameters(&self) -> Vec<Variable> {{
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }}

    fn train_mode(&mut self, _mode: bool) {{}}
}}

pub fn create_model() -> CnnModel {{
    CnnModel::new(10)
}}
"
    )
}

fn create_mlp_template(project_name: &str) -> String {
    format!(
        r"//! MLP Model for {project_name}
//!
//! Multi-layer perceptron model.

use axonml::prelude::*;

/// MLP model
pub struct MlpModel {{
    layers: Vec<Linear>,
}}

impl MlpModel {{
    pub fn new(input_size: usize, hidden_sizes: &[usize], num_classes: usize) -> Self {{
        let mut layers = Vec::new();
        let mut prev_size = input_size;

        for &size in hidden_sizes {{
            layers.push(Linear::new(prev_size, size));
            prev_size = size;
        }}
        layers.push(Linear::new(prev_size, num_classes));

        Self {{ layers }}
    }}

    pub fn num_parameters(&self) -> usize {{
        self.parameters().iter().map(|p| p.data().len()).sum()
    }}
}}

impl Module for MlpModel {{
    fn forward(&self, input: &Variable) -> Variable {{
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {{
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {{
                x = x.relu();
            }}
        }}
        x
    }}

    fn parameters(&self) -> Vec<Variable> {{
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }}

    fn train_mode(&mut self, _mode: bool) {{}}
}}

pub fn create_model() -> MlpModel {{
    MlpModel::new(784, &[512, 256, 128], 10)
}}
"
    )
}

fn create_transformer_template(project_name: &str) -> String {
    format!(
        r"//! Transformer Model for {project_name}
//!
//! Transformer model for sequence tasks.

use axonml::prelude::*;

/// Transformer model
pub struct TransformerModel {{
    embedding: Embedding,
    encoder: TransformerEncoder,
    fc: Linear,
}}

impl TransformerModel {{
    pub fn new(vocab_size: usize, d_model: usize, nhead: usize, num_layers: usize, num_classes: usize) -> Self {{
        Self {{
            embedding: Embedding::new(vocab_size, d_model),
            encoder: TransformerEncoder::new(d_model, nhead, num_layers),
            fc: Linear::new(d_model, num_classes),
        }}
    }}

    pub fn num_parameters(&self) -> usize {{
        self.parameters().iter().map(|p| p.data().len()).sum()
    }}
}}

impl Module for TransformerModel {{
    fn forward(&self, input: &Variable) -> Variable {{
        let x = self.embedding.forward(input);
        let x = self.encoder.forward(&x);
        // Take [CLS] token or mean pool
        self.fc.forward(&x)
    }}

    fn parameters(&self) -> Vec<Variable> {{
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        params.extend(self.encoder.parameters());
        params.extend(self.fc.parameters());
        params
    }}

    fn train_mode(&mut self, _mode: bool) {{}}
}}

pub fn create_model() -> TransformerModel {{
    TransformerModel::new(30000, 512, 8, 6, 10)
}}
"
    )
}

// =============================================================================
// Data Directories
// =============================================================================

fn create_data_directories(project_path: &PathBuf) -> CliResult<()> {
    // Create placeholder files
    let placeholder = "# Place your data files here\n";

    fs::write(project_path.join("data/train/.gitkeep"), placeholder)?;
    fs::write(project_path.join("data/val/.gitkeep"), placeholder)?;
    fs::write(project_path.join("data/test/.gitkeep"), placeholder)?;
    fs::write(project_path.join("checkpoints/.gitkeep"), "")?;
    fs::write(project_path.join("logs/.gitkeep"), "")?;
    fs::write(project_path.join("outputs/.gitkeep"), "")?;

    Ok(())
}

// =============================================================================
// Git Repository
// =============================================================================

fn init_git_repo(project_path: &PathBuf) -> CliResult<()> {
    use std::process::Command;

    // Try to initialize git repository
    let output = Command::new("git")
        .arg("init")
        .current_dir(project_path)
        .output();

    match output {
        Ok(result) if result.status.success() => {
            // Initial commit
            let _ = Command::new("git")
                .args(["add", "."])
                .current_dir(project_path)
                .output();

            let _ = Command::new("git")
                .args(["commit", "-m", "Initial commit: Axonml project scaffold"])
                .current_dir(project_path)
                .output();

            Ok(())
        }
        _ => {
            // Git not available, skip silently
            Ok(())
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_directory_structure() {
        let temp = tempdir().unwrap();
        let project_path = temp.path().join("test-project");

        create_directory_structure(&project_path).unwrap();

        assert!(project_path.exists());
        assert!(project_path.join("src").exists());
        assert!(project_path.join("src/models").exists());
        assert!(project_path.join("data/train").exists());
    }

    #[test]
    fn test_create_config_files() {
        let temp = tempdir().unwrap();
        let project_path = temp.path().join("test-project");
        std::fs::create_dir_all(&project_path).unwrap();
        std::fs::create_dir_all(project_path.join("configs")).unwrap();

        create_config_files(&project_path, "test-project").unwrap();

        assert!(project_path.join("axonml.toml").exists());
        assert!(project_path.join(".gitignore").exists());
    }
}
