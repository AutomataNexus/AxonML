# axonml-cli

<!-- Logo placeholder -->
<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust 1.75+"></a>
  <a href="https://crates.io/crates/axonml-cli"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Crate Version"></a>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

`axonml-cli` is the official command-line interface for the AxonML machine learning framework. It provides a comprehensive toolkit for managing ML projects, training models, evaluating performance, and deploying to production.

The CLI supports the full ML workflow from project initialization to model serving, with integrations for Kaggle, Weights & Biases, and a built-in model hub.

---

## Features

- **Project Management** - Create and initialize AxonML projects with customizable templates and configurations.

- **Training & Evaluation** - Train models from configuration files, resume from checkpoints, and evaluate with comprehensive metrics.

- **Model Operations** - Convert between formats (ONNX, SafeTensors, Ferrite), quantize models (Q4, Q8, F16), and inspect architectures.

- **Data Management** - Upload, analyze, validate, and preview datasets with automatic type detection and statistics.

- **Deployment** - Export models for production, start inference servers, and manage dashboard services.

- **Integrations** - Kaggle dataset downloads, W&B experiment tracking, and pretrained model hub access.

- **Terminal UI** - Interactive TUI for exploring models and datasets with real-time visualization.

- **GPU Support** - Detect, benchmark, and manage GPU devices for accelerated training.

- **Server Sync** - Sync CLI with the AxonML webapp - training runs, models, and datasets sync automatically between CLI and web interface.

---

## Modules

| Module | Description |
|--------|-------------|
| `cli` | Command-line argument definitions using clap derive macros |
| `commands` | Implementation of all CLI subcommands |
| `config` | Project configuration file parsing (TOML/JSON) |
| `error` | CLI-specific error types and result definitions |

---

## Installation

Install from crates.io:

```bash
cargo install axonml-cli
```

Or build from source:

```bash
cargo build --release -p axonml-cli
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `new` | Create a new AxonML project |
| `init` | Initialize AxonML in existing directory |
| `train` | Train a model from configuration |
| `resume` | Resume training from checkpoint |
| `eval` | Evaluate model performance |
| `predict` | Make predictions with trained model |
| `convert` | Convert models between formats |
| `export` | Export models for deployment |
| `inspect` | Inspect model architecture |
| `report` | Generate evaluation reports |
| `serve` | Start inference server (feature: serve) |
| `wandb` | W&B integration (feature: wandb) |
| `upload` | Upload model files |
| `data` | Dataset management |
| `scaffold` | Generate Rust training projects |
| `zip` | Create/extract model bundles |
| `rename` | Rename models and datasets |
| `quant` | Quantize models (Q4, Q8, F16) |
| `load` | Load models/datasets into workspace |
| `analyze` | Comprehensive analysis and reports |
| `bench` | Benchmark models and hardware |
| `gpu` | GPU detection and management |
| `tui` | Launch terminal user interface |
| `kaggle` | Kaggle dataset integration |
| `hub` | Pretrained model hub |
| `dataset` | Dataset management (NexusConnectBridge) |
| `start` | Start dashboard and API server |
| `stop` | Stop running services |
| `status` | Check service status |
| `logs` | View service logs |
| `login` | Login to AxonML server (sync with webapp) |
| `logout` | Logout from AxonML server |
| `sync` | Check/perform sync with server |

---

## Usage

### Project Commands

```bash
# Create a new AxonML project
axonml new my-project --template default

# Initialize AxonML in an existing directory
axonml init --name my-project

# Generate a Rust training project scaffold
axonml scaffold generate my-trainer --template training --wandb
```

### Training Commands

```bash
# Train a model from configuration
axonml train --config config.toml --epochs 50 --device cuda:0

# Resume training from a checkpoint
axonml resume checkpoint.pt --epochs 20 --lr 0.0001

# Evaluate model performance
axonml eval model.pt data/ --metrics accuracy,f1,precision,recall

# Make predictions
axonml predict model.pt input.json --format json --top-k 5
```

### Model Commands

```bash
# Inspect model architecture
axonml inspect model.pt --detailed --show-params 5

# Convert model formats
axonml convert model.pt model.onnx --to onnx --optimize

# Export for deployment
axonml export model.pt ./deploy --format onnx --target cuda --quantize

# Quantize model
axonml quant convert model.pt --target Q8_0 --output model_q8.axon

# Generate evaluation report
axonml report model.pt --data test/ --format html --confusion-matrix
```

### Data Commands

```bash
# Analyze a dataset
axonml data analyze ./data --detailed --recommend

# Upload and configure dataset
axonml data upload ./images --data-type image --task classification

# Validate dataset structure
axonml data validate ./data --check-balance --check-missing

# Preview dataset samples
axonml data preview ./data --num-samples 10 --random
```

### Workspace Commands

```bash
# Load model into workspace
axonml load model model.pt --name my-model

# Load dataset into workspace
axonml load data ./dataset --data-type tabular

# Analyze loaded model
axonml analyze model --detailed --output report.json

# Generate comprehensive report
axonml analyze report --format html --visualize
```

### Benchmarking

```bash
# Benchmark model performance
axonml bench model model.pt --iterations 100 --device cuda:0

# Benchmark at different batch sizes
axonml bench inference model.pt --batch-sizes 1,2,4,8,16,32

# Compare multiple models
axonml bench compare "model1.pt,model2.pt,model3.pt" --iterations 50

# Benchmark hardware capabilities
axonml bench hardware --iterations 10
```

### GPU Management

```bash
# List available GPUs
axonml gpu list

# Show detailed GPU information
axonml gpu info

# Select GPU for training
axonml gpu select cuda:0 --persistent

# Benchmark GPU performance
axonml gpu bench --all --iterations 10
```

### Hub & Kaggle Integration

```bash
# List pretrained models
axonml hub list

# Download pretrained weights
axonml hub download resnet50 --force

# Configure Kaggle credentials
axonml kaggle login --username USER --key API_KEY

# Search Kaggle datasets
axonml kaggle search "image classification" --limit 20

# Download Kaggle dataset
axonml kaggle download username/dataset-name --output ./data
```

### Server Sync (CLI ↔ Webapp)

```bash
# Login to AxonML server
axonml login
axonml login --server http://myserver:3021

# Check sync status
axonml sync

# Full sync (training runs, models, datasets)
axonml sync --full

# Logout and clear stored credentials
axonml logout
```

### Dashboard & Server

```bash
# Start dashboard and API server
axonml start --port 3000 --dashboard-port 8080

# Start only the API server
axonml start --server --foreground

# Check service status
axonml status --detailed

# View logs
axonml logs --follow --lines 100

# Stop services
axonml stop
```

### Additional Commands

```bash
# Create model/dataset bundle
axonml zip create --output bundle.axonzip --model model.pt --data ./data

# Extract bundle
axonml zip extract bundle.axonzip --output ./extracted

# Rename model
axonml rename model model.pt new_model.pt

# Launch terminal UI
axonml tui --model model.pt --data ./dataset
```

---

## Configuration

The CLI uses `axonml.toml` for project configuration:

```toml
[project]
name = "my-ml-project"
version = "0.1.0"
description = "My machine learning project"

[training]
epochs = 50
batch_size = 32
learning_rate = 0.001
device = "cuda:0"
checkpoint_frequency = 1
output_dir = "./output"
num_workers = 4

[training.optimizer]
name = "adam"
weight_decay = 0.0001
beta1 = 0.9
beta2 = 0.999

[training.scheduler]
name = "cosine"
t_max = 50
eta_min = 0.00001
warmup_epochs = 5

[model]
architecture = "resnet18"
num_classes = 10
dropout = 0.1

[data]
train_path = "./data/train"
val_path = "./data/val"
val_split = 0.1
augmentation = true
shuffle = true
normalize = true
```

---

## Global Options

```bash
# Enable verbose output
axonml --verbose <command>

# Suppress all output except errors
axonml --quiet <command>
```

---

## Tests

Run the test suite:

```bash
cargo test -p axonml-cli
```

Run integration tests:

```bash
cargo test -p axonml-cli --test cli_integration_test
```

---

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
