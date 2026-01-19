# axonml-tui

[![Crates.io](https://img.shields.io/crates/v/axonml-tui.svg)](https://crates.io/crates/axonml-tui)
[![Downloads](https://img.shields.io/crates/d/axonml-tui.svg)](https://crates.io/crates/axonml-tui)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Terminal user interface for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-tui` provides an interactive terminal interface for monitoring training progress, visualizing metrics, and managing experiments. Built with [ratatui](https://github.com/ratatui-org/ratatui) for a beautiful, responsive terminal experience.

## Features

### Training Monitor
- **Real-time metrics** - Loss, accuracy, learning rate
- **Progress bars** - Epoch and batch progress
- **Live plots** - ASCII charts of training curves
- **GPU monitoring** - Memory and utilization graphs

### Experiment Management
- **Run tracking** - Track multiple experiments
- **Compare runs** - Side-by-side comparison
- **Hyperparameter view** - See config for each run
- **Checkpoint browser** - Manage saved checkpoints

### Model Inspection
- **Architecture view** - Layer-by-layer breakdown
- **Parameter count** - Per-layer parameter stats
- **Memory estimate** - Activation memory requirements
- **Graph visualization** - ASCII computation graph

### Interactive Controls
- **Pause/resume** - Control training
- **Adjust LR** - Change learning rate on-the-fly
- **Save checkpoint** - Manual checkpoint save
- **Early stop** - Stop training early

## Installation

```bash
cargo install axonml-tui
```

## Usage

### Launch Training Monitor

```bash
# Start TUI and attach to training
axonml-tui monitor --run ./runs/experiment_1

# Or launch with training
axonml-tui train config.toml
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume training |
| `s` | Save checkpoint |
| `Tab` | Switch panels |
| `↑/↓` | Scroll |
| `+/-` | Adjust learning rate |
| `l` | Toggle log panel |
| `g` | Toggle GPU stats |
| `h` | Show help |

### Training Dashboard

```
┌─ Training Progress ─────────────────────────────────────────────┐
│ Epoch: 45/100  [████████████████████░░░░░░░░░░░░░░░] 45%       │
│ Batch: 128/500 [█████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 26%       │
│ ETA: 2h 34m                                                     │
├─ Metrics ───────────────────────────────────────────────────────┤
│ Loss: 0.2341 ↓  Accuracy: 92.4% ↑  LR: 0.0001                  │
├─ Loss Curve ────────────────────────────────────────────────────┤
│ 2.0 │*                                                          │
│     │ **                                                        │
│ 1.0 │   ***                                                     │
│     │      ****                                                 │
│ 0.0 │          **************************************           │
│     └──────────────────────────────────────────────────         │
│       0        25        50        75       100                 │
├─ GPU ───────────────────────────────────────────────────────────┤
│ GPU 0: [████████████████████░░░░░░░] 68%  Mem: 6.2/8.0 GB       │
└─────────────────────────────────────────────────────────────────┘
```

### Model Inspector

```bash
# View model architecture
axonml-tui inspect model.axonml
```

```
┌─ Model Architecture ────────────────────────────────────────────┐
│ Layer               │ Type      │ Params    │ Output Shape      │
├─────────────────────┼───────────┼───────────┼───────────────────┤
│ conv1               │ Conv2d    │ 9,408     │ [B, 64, 112, 112] │
│ bn1                 │ BatchNorm │ 128       │ [B, 64, 112, 112] │
│ relu                │ ReLU      │ 0         │ [B, 64, 112, 112] │
│ maxpool             │ MaxPool2d │ 0         │ [B, 64, 56, 56]   │
│ layer1.0.conv1      │ Conv2d    │ 36,864    │ [B, 64, 56, 56]   │
│ ...                 │ ...       │ ...       │ ...               │
├─────────────────────┴───────────┴───────────┴───────────────────┤
│ Total Parameters: 11,689,512 (44.6 MB)                          │
│ Trainable Parameters: 11,689,512                                │
│ Estimated Memory: 512 MB @ batch_size=32                        │
└─────────────────────────────────────────────────────────────────┘
```

### Experiment Comparison

```bash
# Compare multiple runs
axonml-tui compare run1/ run2/ run3/
```

```
┌─ Experiment Comparison ─────────────────────────────────────────┐
│                    │ run1        │ run2        │ run3          │
├────────────────────┼─────────────┼─────────────┼───────────────┤
│ Model              │ resnet18    │ resnet34    │ resnet50      │
│ LR                 │ 0.001       │ 0.001       │ 0.0001        │
│ Batch Size         │ 32          │ 32          │ 64            │
│ Best Val Acc       │ 91.2%       │ 93.4%       │ 94.1%         │
│ Final Loss         │ 0.312       │ 0.245       │ 0.198         │
│ Training Time      │ 2h 15m      │ 4h 32m      │ 8h 45m        │
│ Status             │ Completed   │ Completed   │ Running       │
└─────────────────────────────────────────────────────────────────┘
```

### Programmatic Integration

```rust
use axonml_tui::{TrainingMonitor, MonitorConfig};

let monitor = TrainingMonitor::new(MonitorConfig {
    update_interval: Duration::from_millis(100),
    show_gpu_stats: true,
    ..Default::default()
});

// In training loop
for epoch in 0..num_epochs {
    for (batch_idx, batch) in dataloader.iter().enumerate() {
        // ... training code ...

        monitor.update(|state| {
            state.epoch = epoch;
            state.batch = batch_idx;
            state.loss = loss.item();
            state.accuracy = accuracy;
            state.lr = optimizer.learning_rate();
        });
    }
}

monitor.finish();
```

## Themes

```bash
# Use dark theme (default)
axonml-tui --theme dark monitor

# Use light theme
axonml-tui --theme light monitor

# Use minimal theme (less decoration)
axonml-tui --theme minimal monitor
```

## Configuration

Create `~/.config/axonml/tui.toml`:

```toml
[display]
theme = "dark"
refresh_rate = 100  # ms
show_gpu = true

[charts]
history_length = 1000
smooth = true

[colors]
loss = "red"
accuracy = "green"
lr = "yellow"
```

## Part of Axonml

```toml
[dependencies]
axonml = { version = "0.1", features = ["tui"] }
```

## License

MIT OR Apache-2.0
