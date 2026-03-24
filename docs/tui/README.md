# axonml-tui

> Terminal User Interface for the Axonml ML Framework

## Overview

The `axonml-tui` crate provides an interactive terminal-based dashboard for ML development. Built with [Ratatui](https://ratatui.rs/) and [Crossterm](https://github.com/crossterm-rs/crossterm), it offers a comprehensive view of your models, datasets, training progress, and more.

## Features

- **Model Architecture Visualization** - View neural network layers, shapes, and parameters
- **Dataset Statistics** - Explore samples, class distributions, and features
- **Real-time Training Monitoring** - Track epoch/batch progress, loss, and accuracy
- **Training Graphs** - Visualize loss curves, accuracy curves, and learning rate
- **File Browser** - Navigate and load model/dataset files
- **Keyboard-driven Navigation** - Vim-style key bindings for efficient workflow

## Installation

The TUI is included with the Axonml CLI. Install it with:

```bash
cargo install --path crates/axonml-cli
```

## Usage

### Launch the TUI

```bash
# Basic launch
axonml tui

# Load a model on startup
axonml tui --model path/to/model.axonml

# Load a dataset on startup
axonml tui --data path/to/dataset/

# Load both
axonml tui --model model.axonml --data ./data/
```

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `Tab` | Next tab |
| `Shift+Tab` | Previous tab |
| `1` | Model view |
| `2` | Data view |
| `3` | Training view |
| `4` | Graphs view |
| `5` | Files view |
| `↑` / `k` | Move up |
| `↓` / `j` | Move down |
| `←` / `h` | Move left / Previous panel |
| `→` / `l` | Move right / Next panel |
| `Enter` | Select / Open |
| `d` | Toggle details (Model view) |
| `p` | Pause/Resume (Training view) |
| `r` | Refresh (Training view) |
| `z` | Toggle zoom (Graphs view) |
| `u` / `Backspace` | Go up directory (Files view) |
| `~` | Go to home (Files view) |
| `?` | Show help overlay |
| `q` | Quit |

## Views

### Model View

Displays the architecture of the loaded neural network:

- **Header** - Model name, format, file size, parameter counts
- **Layer List** - Scrollable list of layers with type, shape, and parameters
- **Details Panel** - Detailed information about the selected layer

### Data View

Shows statistics about the loaded dataset:

- **Summary** - Sample count, feature dimensions, data type
- **Class Distribution** - Bar chart of class frequencies
- **Sample Preview** - Preview of individual data samples

### Training View

Real-time monitoring of training progress:

- **Progress Bars** - Epoch and batch progress
- **Metrics** - Current loss, accuracy, learning rate
- **History** - Rolling window of recent metrics
- **Status** - Training state (Running, Paused, Complete)

### Graphs View

Visualization of training metrics over time:

- **Loss Curves** - Training and validation loss
- **Accuracy Curves** - Training and validation accuracy
- **Learning Rate** - LR schedule visualization

### Files View

File browser for models and datasets:

- **Directory Tree** - Expandable file tree
- **File Detection** - Automatic detection of model/dataset files
- **Preview Panel** - File information and preview

### Help View

Complete keyboard shortcuts reference organized by category.

## Theme

The TUI uses the NexusForge color scheme:

| Color | Hex | Usage |
|-------|-----|-------|
| Teal | `#14b8a6` | Primary accent |
| Teal Light | `#5eead4` | Highlights |
| Terracotta | `#c4a484` | Secondary accent |
| Cream | `#faf9f6` | Text |
| Dark Slate | `#1e293b` | Background |
| Success | `#10b981` | Success states |
| Warning | `#f59e0b` | Warnings |
| Error | `#ef4444` | Errors |
| Info | `#64b5f6` | Information |

## API

### Programmatic Usage

```rust
use axonml_tui::run;
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    // Launch with no pre-loaded files
    run(None, None)?;

    // Or with a model
    run(Some(PathBuf::from("model.axonml")), None)?;

    // Or with both
    run(
        Some(PathBuf::from("model.axonml")),
        Some(PathBuf::from("./data/")),
    )?;

    Ok(())
}
```

### Accessing Components

```rust
use axonml_tui::{App, Tab, AxonmlTheme};
use axonml_tui::views::{ModelView, DataView, TrainingView, GraphsView, FilesView, HelpView};

// Create app instance
let mut app = App::new();

// Navigate tabs
app.next_tab();
app.go_to_tab(Tab::Training);

// Load data
app.load_model(PathBuf::from("model.axonml"));
app.load_dataset(PathBuf::from("./data/"));

// Access views
app.model_view.select_next();
app.training_view.toggle_pause();
```

## Architecture

```
axonml-tui/
├── src/
│   ├── lib.rs          # Entry point and run() function
│   ├── app.rs          # Application state and Tab enum
│   ├── event.rs        # Keyboard event handling
│   ├── ui.rs           # Main rendering logic
│   ├── theme.rs        # NexusForge color theme
│   └── views/
│       ├── mod.rs      # View exports
│       ├── model.rs    # Model architecture view
│       ├── data.rs     # Dataset statistics view
│       ├── training.rs # Training progress view
│       ├── graphs.rs   # Training graphs view
│       ├── files.rs    # File browser view
│       └── help.rs     # Keyboard shortcuts view
└── Cargo.toml
```

## Dependencies

- `ratatui` - Terminal UI framework
- `crossterm` - Cross-platform terminal manipulation
- `tokio` - Async runtime (for future real-time updates)
- `chrono` - Time formatting
- `dirs` - Home directory detection
- `axonml-core`, `axonml-tensor`, `axonml-nn`, `axonml-serialize` - Axonml crates

## Testing

Run the tests:

```bash
cargo test -p axonml-tui
```

## Examples

See the [examples/](../../examples/) directory for working examples.

## License

Licensed under MIT or Apache 2.0, same as the Axonml framework.
