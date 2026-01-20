<p align="center">
  <img src="../../assets/axonml-logo.png" alt="AxonML Logo" width="200"/>
</p>

<h1 align="center">axonml-tui</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-tui"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Crates.io: 0.1.0"></a>
  <a href="https://github.com/automatanexus/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-teal.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

**axonml-tui** is a comprehensive terminal user interface for the AxonML machine learning framework. It provides real-time visualization of neural network architectures, dataset statistics, training progress, loss curves, and a file browser for navigating models and datasets - all from the comfort of your terminal.

---

## Features

- **Model Architecture Visualization** - View layer structures, parameter counts, shapes, and trainable status with interactive navigation
- **Dataset Explorer** - Analyze class distributions, feature statistics, data splits, and preview dataset metadata
- **Real-Time Training Monitor** - Track epochs, batches, loss/accuracy metrics with sparkline trends and ETA estimation
- **Interactive Graphs** - Visualize loss curves, accuracy charts, and learning rate schedules with switchable chart types
- **File Browser** - Navigate directories, preview model and dataset files, with support for multiple formats
- **Help System** - Comprehensive keyboard shortcut reference organized by category
- **Themeable UI** - NexusForge-inspired color scheme with teal, terracotta, and cream accents

---

## Modules

| Module | Description |
|--------|-------------|
| `app` | Main application state, tab management, and navigation logic |
| `event` | Keyboard input handling and event routing to views |
| `theme` | NexusForge color palette and style presets (teal, terracotta, cream) |
| `ui` | Core rendering logic for header, content, footer, and help overlay |
| `views::model` | Neural network architecture display with layer details |
| `views::data` | Dataset statistics, class distribution, and feature tables |
| `views::training` | Real-time training progress with gauges and sparklines |
| `views::graphs` | Loss, accuracy, and learning rate chart rendering |
| `views::files` | Tree-based file browser with preview pane |
| `views::help` | Keyboard shortcut reference organized by category |

---

## Usage

Add `axonml-tui` to your `Cargo.toml`:

```toml
[dependencies]
axonml-tui = "0.1.0"
```

### Basic Example

```rust
use axonml_tui::run;

fn main() -> std::io::Result<()> {
    // Launch the TUI with no pre-loaded files
    run(None, None)
}
```

### Loading a Model on Startup

```rust
use axonml_tui::run;
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let model_path = PathBuf::from("models/mnist_classifier.axonml");
    run(Some(model_path), None)
}
```

### Loading Both Model and Dataset

```rust
use axonml_tui::run;
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let model_path = PathBuf::from("models/resnet.axonml");
    let data_path = PathBuf::from("datasets/cifar10");
    run(Some(model_path), Some(data_path))
}
```

---

## Keyboard Shortcuts

### Global

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Switch between views |
| `1` - `5` | Jump to specific view (Model, Data, Training, Graphs, Files) |
| `?` | Toggle help overlay |
| `q` | Quit application |

### Navigation

| Key | Action |
|-----|--------|
| `j` / `Down` | Move selection down |
| `k` / `Up` | Move selection up |
| `h` / `Left` | Collapse / Previous panel |
| `l` / `Right` | Expand / Next panel |
| `Enter` | Select / Open |

### Model View

| Key | Action |
|-----|--------|
| `d` | Toggle detailed layer view |
| `Enter` | View layer details |

### Training View

| Key | Action |
|-----|--------|
| `p` | Pause / Resume training |
| `r` | Refresh training data |

### Graphs View

| Key | Action |
|-----|--------|
| `<` / `>` | Switch chart type (Loss / Accuracy / LR) |
| `z` | Toggle zoom mode |

### Files View

| Key | Action |
|-----|--------|
| `Enter` | Open file / Toggle directory |
| `Backspace` / `u` | Go to parent directory |
| `~` | Go to home directory |

---

## Supported File Formats

| Category | Extensions |
|----------|------------|
| Models | `.axonml`, `.onnx`, `.pt`, `.pth`, `.safetensors`, `.h5`, `.keras` |
| Datasets | `.npz`, `.npy`, `.csv`, `.parquet`, `.arrow`, `.tfrecord` |
| Config | `.toml`, `.yaml`, `.yml`, `.json` |

---

## Tests

Run the test suite:

```bash
cargo test -p axonml-tui
```

Run with verbose output:

```bash
cargo test -p axonml-tui -- --nocapture
```

---

## Dependencies

- **ratatui** - Terminal UI framework
- **crossterm** - Cross-platform terminal manipulation
- **tokio** - Async runtime
- **serde** / **serde_json** - Serialization
- **chrono** - Time handling
- **dirs** - File system utilities

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
