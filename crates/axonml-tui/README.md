<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<h1 align="center">axonml-tui</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75%2B-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-tui"><img src="https://img.shields.io/badge/crates.io-0.6.5-green.svg" alt="Crates.io: 0.6.5"></a>
  <a href="https://github.com/automatanexus/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-teal.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

**axonml-tui** is the terminal user interface for the AxonML machine learning framework, built on Ratatui 0.30 and crossterm 0.28. It provides real-time visualization of neural network architectures, dataset statistics, training progress, loss curves, and a file browser for navigating models and datasets. The crate exposes a single `run(model_path, data_path)` entry point used by the `axonml tui` CLI subcommand.

Last updated: 2026-06-06 — version 0.6.5.

---

## Features

- **Six Tab Views** — `Model`, `Data`, `Training`, `Graphs`, `Files`, `Help`, each implemented as a dedicated module under `views::*`.
- **Model Architecture View** — Layer list, parameter counts, shapes, and trainable flags with interactive navigation (`d` toggles detail).
- **Dataset View** — Class distribution, feature statistics, data splits, and metadata preview.
- **Training View** — Real-time gauges and sparkline trends for loss/accuracy; `tick()` is called every frame while the tab is active.
- **Graphs View** — Loss, accuracy, and learning-rate charts with switchable chart types (`<` / `>`) and a zoom toggle (`z`).
- **Files View** — Directory tree with preview pane; navigates via `Enter`, `Backspace`/`u`, `~`.
- **Help View** — Keyboard-shortcut reference organized by category.
- **AxonmlTheme** — NexusForge-inspired palette with teal, terracotta, and cream accents (`theme::AxonmlTheme`).
- **Non-Blocking Event Loop** — `crossterm` event polling at 100 ms with graceful terminal restore on exit.

---

## Modules

| Module | Description |
|--------|-------------|
| `app` | `App` state machine, `Tab` enum, `active_tab`, `should_quit`, model/dataset loaders |
| `event` | Key-event polling, reading, and dispatch to the active view |
| `theme` | `AxonmlTheme` color palette and style presets |
| `ui` | Top-level `render()` that draws header, tab content, footer, and help overlay |
| `views::mod` | Re-exports all view types |
| `views::model` | `ModelView` — neural network architecture display |
| `views::data` | `DataView` — dataset statistics and class distribution |
| `views::training` | `TrainingView` — real-time progress with gauges + sparklines; `tick()` per-frame |
| `views::graphs` | `GraphsView` — loss / accuracy / LR charts |
| `views::files` | `FilesView` — directory browser with preview pane |
| `views::help` | `HelpView` — keyboard-shortcut reference |

Public re-exports from `lib.rs`: `App`, `Tab`, `AxonmlTheme`, `DataView`, `FilesView`, `GraphsView`, `HelpView`, `ModelView`, `TrainingView`, and the `run()` entry point.

---

## Usage

Add `axonml-tui` to your `Cargo.toml`:

```toml
[dependencies]
axonml-tui = "0.6.5"
```

Or launch it directly via the CLI:

```bash
axonml tui
axonml tui --model models/mnist.axonml
axonml tui --model models/resnet.axonml --data datasets/cifar10
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
| `1` - `6` | Jump to specific view (Model, Data, Training, Graphs, Files, Help) |
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

- **ratatui** 0.30 — Terminal UI framework
- **crossterm** 0.28 — Cross-platform terminal manipulation
- **tokio** — Async runtime (full features)
- **serde** / **serde_json** — Serialization
- **chrono** — Time handling
- **dirs** — Home-directory resolution for `~` navigation
- Internal: `axonml-core`, `axonml-tensor`, `axonml-nn`, `axonml-serialize`

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
