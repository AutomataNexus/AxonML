# axonml-hvac

<p align="center">
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.85%2B-orange.svg" alt="Rust: 1.85+"></a>
  <img src="https://img.shields.io/badge/version-0.6.1-green.svg" alt="Version 0.6.1"/>
  <a href="https://github.com/AutomataNexus/AxonML"><img src="https://img.shields.io/badge/part%20of-AxonML-teal.svg" alt="Part of AxonML"></a>
</p>

---

## Overview

**axonml-hvac** is a domain-specific sub-crate of the AxonML deep-learning framework providing nine specialized neural-network models for HVAC fault detection, predictive maintenance, and facility-wide diagnostic reasoning. It was extracted from the `axonml` umbrella crate in 0.6.1 (April 2026) to keep the framework itself domain-agnostic.

The models were designed for the **TMC HVAC controller** project and trained on physics-informed synthetic data derived from real building-automation-system (BAS) control logic.

Last updated: 2026-04-16 — version 0.6.1.

---

## Models

| Model | File | Purpose |
|-------|------|---------|
| **Apollo** | `apollo.rs` | Primary fault classifier (top-level diagnostic head) |
| **Aquilo** | `aquilo.rs` | Airflow anomaly detector (AHU/VAV blockage, dirty filter) |
| **Boreas** | `boreas.rs` | Cold-side (cooling) specialist — chilled-water / DX cooling faults |
| **Colossus** | `colossus.rs` | Large transformer diagnostician — multi-equipment reasoning |
| **Gaia** | `gaia.rs` | Environmental context encoder (OAT, humidity, enthalpy) |
| **Naiad** | `naiad.rs` | Water-side (hydronic) specialist — loop pressures, flows, pump behavior |
| **Panoptes** | `panoptes.rs` | Observability / multi-signal fusion across a full facility |
| **Vulcan** | `vulcan.rs` | Heat-side specialist — boilers, hot-water loops, reheat |
| **Zephyrus** | `zephyrus.rs` | Temporal predictor + autoencoder for drift detection |

## Supporting Modules

| Module | Purpose |
|--------|---------|
| `data` | `HvacSensorData`, `HvacLabels`, `PipelineOutput`, `SyntheticHvacGenerator` |
| `panoptes_datagen` | Warren HVAC simulator (`PanoptesTrainingData`, `WarrenSimulator`) — physics-informed training data |
| `pipeline` | End-to-end multi-model diagnostic pipeline (`HvacPipeline`) |

## Public Re-exports

`lib.rs` re-exports the full public surface: `Apollo`, `Aquilo`, `Boreas`, `Colossus`, `Gaia`, `Naiad`, `Panoptes`, `Vulcan`, `Zephyrus`, `HvacPipeline`, `HvacSensorData`, `HvacLabels`, `PipelineOutput`, `SyntheticHvacGenerator`, `PanoptesTrainingData`, and `WarrenSimulator`.

---

## Usage

```rust
use axonml_hvac::{Apollo, Boreas, HvacPipeline, Panoptes};

fn main() {
    // Full diagnostic pipeline
    let pipeline = HvacPipeline::new();

    // Or individual specialists
    let apollo = Apollo::new();
    let boreas = Boreas::new();
    let panoptes = Panoptes::new();
    let _ = (pipeline, apollo, boreas, panoptes);
}
```

## Examples

The crate ships four runnable examples under `examples/`:

```bash
# Train Panoptes (facility-wide anomaly detection)
cargo run --release -p axonml-hvac --example train_panoptes

# HVAC inference demo
cargo run --release -p axonml-hvac --example hvac_inference

# HVAC model scaffolding test
cargo run --release -p axonml-hvac --example hvac_model

# HVAC training loop reference
cargo run --release -p axonml-hvac --example hvac_training
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `axonml-core` | Device, DType, error types |
| `axonml-tensor` | Tensor operations |
| `axonml-autograd` | Automatic differentiation |
| `axonml-nn` | Neural network layers |
| `rand` | Random initialization / synthetic data generation |

Dev-dependencies add `axonml-optim`, `axonml-serialize`, and the `axonml` umbrella crate for the examples.

## Features

| Feature | Description |
|---------|-------------|
| `default` | (none) |
| `cuda` | Enables CUDA on `axonml-core`, `axonml-tensor`, `axonml-nn`, and `axonml-autograd` |

GPU acceleration:

```bash
cargo build --release -p axonml-hvac --features cuda
```

## License

Licensed under either of Apache License 2.0 or MIT at your option.

---

*Part of [AxonML](https://github.com/AutomataNexus/AxonML) — a complete ML/AI framework in pure Rust.*
