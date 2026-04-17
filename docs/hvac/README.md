---
layout: default
title: HVAC
nav_order: 10
has_children: true
description: "HVAC diagnostic and anomaly detection models"
---

# HVAC Models
{: .no_toc }

`axonml-hvac` is a domain-specific crate aggregating nine named neural-network
models for HVAC fault detection and diagnostic reasoning on top of AxonML. It
covers specialists for each physical subsystem, cross-subsystem aggregation,
safety validation, and a facility-wide anomaly detector. The full pipeline
wires these models together in `HvacPipeline`.

## Models

| Model        | Role                                                                                   |
|--------------|----------------------------------------------------------------------------------------|
| **Panoptes** | Facility-wide anomaly detection — per-type encoders + equipment embeddings + cross-equipment transformer + temporal LSTM |
| **Aquilo**   | Electrical systems specialist (voltage / current / power quality), with FFT1d input    |
| **Boreas**   | Cold-side / refrigeration specialist                                                   |
| **Naiad**    | Water-side / hydronic specialist                                                       |
| **Vulcan**   | Heat-side / mechanical specialist                                                      |
| **Zephyrus** | Airflow / temporal predictor + autoencoder                                             |
| **Colossus** | Cross-specialist transformer aggregator                                                |
| **Gaia**     | Environmental context encoder / safety validator                                       |
| **Apollo**   | Master coordinator — final fault classification, transformer + multi-head attention    |

Exact parameter counts vary by subsystem and are configured inside each
module; the lib documentation notes Apollo as approximately 1.8M params and
Aquilo as approximately 608K params.

## Architecture Overview

```
                    +-----------------------------+
                    |  PANOPTES (facility-wide)   |
                    |  All equipment at once      |
                    |  Cross-equip correlations   |
                    +-----------------------------+

 +--------+ +--------+ +--------+ +--------+ +---------+
 | Aquilo | | Boreas |  | Naiad |  | Vulcan|  |Zephyrus|
 |Electr. | | Refrig |  | Water |  |  Mech |  |Airflow |
 +---+----+ +---+----+  +---+---+  +---+---+  +----+---+
     |          |           |          |            |
     +----------+-----------+----------+------------+
                             |
                     +---------------+
                     |   Colossus    |
                     | (aggregator)  |
                     +-------+-------+
                             |
                     +---------------+
                     |     Gaia      |
                     | (safety check)|
                     +-------+-------+
                             |
                     +---------------+
                     |    Apollo     |
                     | (final diag)  |
                     +---------------+
```

## Crate Layout

```
crates/axonml-hvac/src/
├── lib.rs                — Crate root and re-exports
├── data.rs               — HvacSensorData, HvacLabels, PipelineOutput,
│                            SyntheticHvacGenerator
├── panoptes.rs           — Facility-wide Panoptes model
├── panoptes_datagen.rs   — PanoptesTrainingData + WarrenSimulator HVAC scenario engine
├── aquilo.rs             — Electrical systems specialist
├── boreas.rs             — Refrigeration specialist
├── naiad.rs              — Water systems specialist
├── vulcan.rs             — Mechanical specialist
├── zephyrus.rs           — Airflow specialist
├── colossus.rs           — Cross-specialist aggregator
├── gaia.rs               — Safety validator / environmental context
├── apollo.rs             — Master coordinator / final diagnosis
└── pipeline.rs           — Full HvacPipeline orchestration
```

## Public API

Top-level re-exports (`axonml_hvac::*`):

- Models: `Apollo`, `Aquilo`, `Boreas`, `Colossus`, `Gaia`, `Naiad`,
  `Panoptes`, `Vulcan`, `Zephyrus`
- Data: `HvacSensorData`, `HvacLabels`, `PipelineOutput`,
  `SyntheticHvacGenerator`
- Panoptes training data: `PanoptesTrainingData`, `WarrenSimulator`
- Pipeline: `HvacPipeline`

Each model struct implements `axonml_nn::Module` so it fits into the standard
training / inference flow.

## Quick Start

```rust
use axonml_hvac::{HvacPipeline, SyntheticHvacGenerator};

let mut gen = SyntheticHvacGenerator::new();
let (sensor_data, labels) = gen.sample_batch(32);

let pipeline = HvacPipeline::new();
let output = pipeline.forward(&sensor_data);
```

For facility-wide anomaly training, use `WarrenSimulator` (inside
`panoptes_datagen`) to produce `PanoptesTrainingData` batches drawn from
parameterised HVAC scenarios.

## Last updated

0.6.1 (2026-04-16)
