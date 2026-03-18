---
layout: default
title: HVAC
nav_order: 10
has_children: true
description: "HVAC diagnostic and anomaly detection models"
---

# HVAC Models
{: .no_toc }

AxonML's HVAC module provides specialized neural network models for building
automation, equipment diagnostics, and facility-wide anomaly detection.

## Models

| Model | Architecture | Purpose | Params |
|-------|-------------|---------|--------|
| **Panoptes** | Transformer + LSTM | Facility-wide anomaly detection | ~47K |
| Aquilo | GRU specialist | Electrical systems diagnostics | ~35K |
| Boreas | GRU specialist | Refrigeration systems diagnostics | ~35K |
| Naiad | GRU specialist | Water systems diagnostics | ~35K |
| Vulcan | GRU specialist | Mechanical systems diagnostics | ~35K |
| Zephyrus | GRU specialist | Airflow systems diagnostics | ~35K |
| Colossus | Transformer | Cross-specialist aggregation | ~50K |
| Gaia | Linear + attention | Safety validation | ~20K |
| Apollo | Transformer | Master coordinator / final diagnosis | ~45K |

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │  PANOPTES (facility-wide)    │
                    │  All 59 equipment at once    │
                    │  Cross-equip correlations    │
                    └─────────────────────────────┘

 ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
 │ Aquilo  │ │ Boreas  │ │  Naiad  │ │ Vulcan  │ │Zephyrus │
 │Electric │ │ Refrig  │ │  Water  │ │  Mech   │ │ Airflow │
 └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
      │           │           │           │           │
      └───────────┴───────────┴───────────┴───────────┘
                              │
                     ┌────────┴────────┐
                     │    Colossus     │
                     │  (aggregator)   │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │      Gaia      │
                     │ (safety check) │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │     Apollo     │
                     │ (final diag)   │
                     └────────────────┘
```

## Module Location

```
crates/axonml/src/hvac/
├── mod.rs          — Module exports
├── data.rs         — Sensor data structures, fault types, synthetic data
├── panoptes.rs     — Facility-wide anomaly detection (NEW)
├── aquilo.rs       — Electrical specialist
├── boreas.rs       — Refrigeration specialist
├── naiad.rs        — Water systems specialist
├── vulcan.rs       — Mechanical specialist
├── zephyrus.rs     — Airflow specialist
├── colossus.rs     — Cross-specialist aggregator
├── gaia.rs         — Safety validator
├── apollo.rs       — Master coordinator
└── pipeline.rs     — Full 8-model pipeline orchestration
```
