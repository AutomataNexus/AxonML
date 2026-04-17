# axonml-cli

<!-- Logo placeholder -->
<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust 1.75+"></a>
  <a href="https://crates.io/crates/axonml-cli"><img src="https://img.shields.io/badge/crates.io-0.6.1-green.svg" alt="Crate Version"></a>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

`axonml-cli` is the official command-line interface for the AxonML machine learning framework. The binary is named `axonml` and ships a single `clap`-derived subcommand tree that covers the full ML workflow: project scaffolding, training, evaluation, model conversion, quantization, dataset tooling, GPU management, benchmarking, an embedded terminal UI, optional inference server, optional Weights & Biases integration, dashboard/API-server lifecycle control, and (under the `server-sync` feature) authenticated sync with a running `axonml-server`.

Last updated: 2026-04-16 — version 0.6.1.

---

## Features

- **Project Management** — Create new projects (`new`), initialize existing directories (`init`), and generate Rust training-project scaffolds (`scaffold generate` / `scaffold templates`).

- **Training & Evaluation** — Train from a config file (`train`), resume from a checkpoint (`resume`), evaluate a model (`eval`), run single-input prediction (`predict`), and produce HTML/JSON/text reports (`report`).

- **Model Operations** — Convert between formats (`convert`), export for deployment targets (`export`), inspect architecture (`inspect`), quantize to Q4/Q5/Q8/F16/F32 via `quant convert`, and introspect/benchmark quant levels (`quant info`, `quant benchmark`, `quant list`).

- **Data Management** — Upload, analyze, list, configure, preview, and validate datasets (`data {upload,analyze,list,config,preview,validate}`).

- **Deployment** — Optional REST inference server (`serve`, requires `--features serve`) plus dashboard/server lifecycle (`start`, `stop`, `status`, `logs`).

- **Integrations** — Kaggle dataset search/download (`kaggle`), optional Weights & Biases config (`wandb`, requires `--features wandb`), and pretrained-model hub (`hub`).

- **Terminal UI** — `axonml tui` launches the `axonml-tui` Ratatui interface with an optional model/dataset preloaded.

- **GPU Support** — `gpu list`, `gpu info`, `gpu select`, `gpu bench`, `gpu memory`, `gpu status` over `wgpu`-based device detection.

- **Server Sync** — With `--features server-sync`: `login`, `logout`, and `sync` talk to a live `axonml-server` over HTTPS to mirror training runs, models, and datasets between CLI and web interface.

- **Bundling** — `zip create/extract/list` for model+dataset bundles; `rename model|data` for safe renames; `upload` for pushing a model file into a workspace directory; `load {model,data,both,status,clear}` and `analyze {model,data,both,report}` for workspace-scoped inspection; `bench {model,inference,compare,hardware}` for performance sweeps.

---

## Binary

The crate produces a single binary:

```toml
[[bin]]
name = "axonml"
path = "src/main.rs"
```

There is no `axon` alias target; users who want a shorter name should add a shell alias (`alias axon='axonml'`).

---

## Modules

| Module | Description |
|--------|-------------|
| `cli` | Top-level `clap` derive definitions (the full subcommand tree lives here) |
| `commands` | Per-subcommand `execute` functions under `commands::{new,init,train,resume,eval,predict,convert,export,inspect,report,serve,wandb,upload,data,scaffold,zip,rename,quant,load,analyze,bench,gpu,tui,kaggle,hub,dataset,dashboard,sync}` |
| `config` | Project configuration file parsing (TOML/JSON via `serde`/`toml`) |
| `error` | `CliError` / `CliResult` types |
| `api_client` | HTTP client for server-sync (feature-gated on `server-sync`) |

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

The produced binary is `target/release/axonml`.

### Cargo features

Defined in `Cargo.toml`:

- `default = ["wandb", "kaggle", "dataset-api", "hub-download", "server-sync"]`
- `serve` — enables the `serve` subcommand (pulls in `tokio` + `axum`)
- `wandb` — enables the `wandb` subcommand group
- `kaggle` — enables Kaggle HTTP client used by `kaggle search`/`download`
- `dataset-api` — remote dataset listing/downloads
- `hub-download` — HTTP downloads of pretrained weights
- `server-sync` — enables `login`, `logout`, `sync` and the `api_client` module (pulls in `reqwest` + `tokio`)

---

## Command Reference

All commands below are defined in `src/cli.rs` and dispatched in `src/main.rs`.

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
| `report` | Generate evaluation reports (HTML/JSON/text/all) |
| `serve` | Start inference server (feature: `serve`) |
| `wandb` | W&B integration (feature: `wandb`) |
| `upload` | Upload a model file into a workspace |
| `data` | Dataset management subcommands |
| `scaffold` | Generate Rust training projects |
| `zip` | Create/extract/list model+dataset bundles |
| `rename` | Rename models and datasets |
| `quant` | Quantize, inspect, or benchmark quant levels |
| `load` | Load models/datasets into workspace |
| `analyze` | Comprehensive analysis and reports |
| `bench` | Benchmark models and hardware |
| `gpu` | GPU detection and management |
| `tui` | Launch the terminal user interface |
| `kaggle` | Kaggle dataset integration |
| `hub` | Pretrained model hub |
| `dataset` | Dataset management (NexusConnectBridge) |
| `start` | Start dashboard and API server |
| `stop` | Stop running services |
| `status` | Check service status |
| `logs` | View service logs |
| `login` | Login to AxonML server (feature: `server-sync`) |
| `logout` | Logout from AxonML server (feature: `server-sync`) |
| `sync` | Check/perform sync with server (feature: `server-sync`) |

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
axonml train --config config.toml --data ./data/train --epochs 50 --device cuda:0

# Resume training from a checkpoint (requires --data)
axonml resume checkpoint.pt --data ./data/train --epochs 20 --lr 0.0001

# Evaluate model performance
axonml eval model.pt ./data/val --metrics accuracy,loss --batch-size 32

# Make predictions
axonml predict model.pt input.json --format json --top-k 5
```

### Model Commands

```bash
# Inspect model architecture
axonml inspect model.pt --detailed --show-params 5

# Convert model formats (ferrite / onnx / safetensors)
axonml convert model.pt model.onnx --to onnx --optimize

# Export for deployment
axonml export model.pt ./deploy --format onnx --target cuda --quantize --precision fp16

# Quantize model (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32)
axonml quant convert model.pt --target Q8_0 --output model_q8.axon

# Inspect or benchmark quant levels
axonml quant info model_q8.axon --detailed
axonml quant benchmark model.pt --iterations 10 --types Q4_0,Q8_0
axonml quant list

# Generate evaluation report
axonml report model.pt --data ./test --format html --confusion-matrix --loss-curves
```

### Data Commands

```bash
# Analyze a dataset
axonml data analyze ./data --detailed --recommend

# Upload and configure dataset
axonml data upload ./images --data-type image --task classification --split 0.8,0.1,0.1

# Validate dataset structure
axonml data validate ./data --check-balance --check-missing

# Preview dataset samples
axonml data preview ./data --num-samples 10 --random

# Generate preprocessing config
axonml data config ./data --output data_config.toml
```

### Workspace Commands

```bash
# Load model into workspace
axonml load model model.pt --name my-model

# Load dataset into workspace
axonml load data ./dataset --data-type tabular

# Load both at once
axonml load both --model model.pt --data ./dataset

# Workspace status / clear
axonml load status
axonml load clear

# Analyze loaded model/dataset
axonml analyze model --detailed --output report.json --format json
axonml analyze data --detailed --max-samples 10000
axonml analyze both --output compat.json

# Generate comprehensive report
axonml analyze report --format html --visualize --output analysis_report.html
```

### Benchmarking

```bash
# Benchmark model performance
axonml bench model model.pt --iterations 100 --batch-size 1 --device cuda:0

# Benchmark at different batch sizes
axonml bench inference model.pt --batch-sizes 1,2,4,8,16,32 --iterations 50

# Compare multiple models
axonml bench compare "model1.pt,model2.pt,model3.pt" --iterations 50

# Benchmark hardware capabilities
axonml bench hardware --iterations 10
```

### GPU Management

```bash
axonml gpu list
axonml gpu info
axonml gpu select cuda:0 --persistent
axonml gpu bench --all --iterations 10
axonml gpu memory
axonml gpu status
```

### Hub & Kaggle Integration

```bash
# Pretrained model hub
axonml hub list
axonml hub info resnet50
axonml hub download resnet50 --force
axonml hub cached
axonml hub clear resnet50        # clear one
axonml hub clear                 # clear all

# Kaggle
axonml kaggle login --username USER --key API_KEY   # credentials saved to ~/.kaggle/kaggle.json with 0600 perms; key is NOT echoed back
axonml kaggle status
axonml kaggle search "image classification" --limit 20
axonml kaggle download username/dataset-name --output ./data
axonml kaggle list
```

Note: `kaggle login` writes credentials to `~/.kaggle/kaggle.json` with Unix mode `0600` and deliberately does not echo the API key (or username) back to stdout to keep it out of shell history / CI logs. Run `axonml kaggle status` after login to confirm.

### Dataset (NexusConnectBridge)

```bash
axonml dataset list --source kaggle
axonml dataset info mnist
axonml dataset search "language model" --source all --limit 20
axonml dataset download mnist --output ./data
axonml dataset sources
```

### Server Sync (CLI <-> Webapp, feature `server-sync`)

```bash
# Login to AxonML server (defaults to http://localhost:3021)
axonml login
axonml login --server http://myserver:3021 --username me

# Check sync status or force re-sync
axonml sync --status
axonml sync --force

# Logout and clear stored credentials
axonml logout
```

### Dashboard & Server

```bash
# Start dashboard and API server (default: API on 3000, dashboard on 8080)
axonml start --port 3000 --dashboard-port 8080

# Start only the API server, in foreground
axonml start --server --foreground

# Check service status
axonml status --detailed --format json

# View logs
axonml logs --follow --lines 100 --level info

# Stop services (SIGTERM; use --force for SIGKILL)
axonml stop
axonml stop --server --force
```

### Bundles, rename, upload, TUI

```bash
# Create / extract / list bundles
axonml zip create --output bundle.axonzip --model model.pt --data ./data --include-config
axonml zip extract bundle.axonzip --output ./extracted --verbose
axonml zip list bundle.axonzip --detailed

# Rename
axonml rename model model.pt new_model.pt --force
axonml rename data ./dataset new_name --force

# Upload a model file into a workspace directory
axonml upload ./model.pt --name my-model --version latest --output ./models --inspect

# Launch TUI
axonml tui --model model.pt --data ./dataset
```

### Weights & Biases (feature `wandb`)

```bash
axonml wandb login
axonml wandb status
axonml wandb config --api-key ... --entity my-team --project my-proj --log-frequency 50 --log-checkpoints true
axonml wandb enable
axonml wandb disable
axonml wandb logout
```

### Inference Server (feature `serve`)

```bash
axonml serve model.pt --host 127.0.0.1 --port 8080 --workers 4 --batch --max-batch-size 32 --timeout 30000
```

---

## Configuration

The CLI uses `axonml.toml` for project configuration. Example:

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
# Enable verbose output (global flag)
axonml --verbose <command>

# Suppress all output except errors (global flag)
axonml --quiet <command>
```

---

## Tests

```bash
# Unit tests
cargo test -p axonml-cli

# Integration tests (uses assert_cmd / predicates)
cargo test -p axonml-cli --test cli_integration_test
```

---

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
