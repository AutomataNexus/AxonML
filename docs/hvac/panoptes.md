---
layout: default
title: "Panoptes — Facility-Wide Anomaly Detection"
parent: HVAC
nav_order: 9
description: "All-seeing facility monitor for multi-equipment anomaly detection"
---

# Panoptes — Facility-Wide Anomaly Detection
{: .no_toc }

*Named after Argus Panoptes — the hundred-eyed giant of Greek mythology.*

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Panoptes is a facility-level anomaly detection model that ingests **all equipment
sensor data simultaneously** and learns cross-equipment correlations to detect
abnormalities early. Unlike per-equipment models (Hermes, Tyche, etc.), Panoptes
understands that when a boiler's pressure drops, downstream steam bundles should
respond — and flags it as anomalous when they don't.

```
File:   crates/axonml-hvac/src/panoptes.rs
Author: Andrew Jewell Sr — AutomataNexus LLC (ORCID 0009-0005-2158-7060)
Params: ~47K (edge-deployable)
```

*The HVAC models were split out of the `axonml` umbrella into a dedicated `axonml-hvac` crate in April 2026 (v0.6.1) to reduce the umbrella's dep fan-out.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PANOPTES ARCHITECTURE                          │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  AHU     │ │ Boiler   │ │  Fan     │ │  Pump    │ │ Chiller  │ │
│  │ 12 sens  │ │  7 sens  │ │ Coil     │ │  7 sens  │ │  9 sens  │ │
│  │          │ │          │ │  9 sens  │ │          │ │          │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │             │            │             │            │       │
│       ▼             ▼            ▼             ▼            ▼       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │          PER-TYPE FEATURE ENCODERS (Linear + LayerNorm)      │   │
│  │     Each type has its own encoder: sensors → 32-dim embed    │   │
│  │                                                              │   │
│  │  AHU: 12→32   Boiler: 7→32   FanCoil: 9→32   Pump: 7→32   │   │
│  │  DOAS: 6→32   Bundle: 5→32   Chiller: 9→32                 │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              EMBEDDING ADDITION                              │   │
│  │                                                              │   │
│  │   encoded[i] += TypeEmbed(equip_type) + IDEmbed(slot_id)    │   │
│  │                                                              │   │
│  │   TypeEmbed: 7 types → 32-dim    (what kind of equipment)   │   │
│  │   IDEmbed:  59 IDs   → 32-dim    (which specific unit)      │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│                             ▼  [1, 59, 32]                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         CROSS-EQUIPMENT TRANSFORMER ENCODER                  │   │
│  │                                                              │   │
│  │   2 layers × 4 heads × d_model=32 × ff=64                  │   │
│  │                                                              │   │
│  │   Each equipment attends to ALL other equipment:             │   │
│  │   AHU-6 ←→ Boiler-1 ←→ SteamBundle-3 ←→ Pump-7 ←→ ...    │   │
│  │                                                              │   │
│  │   Learns correlations:                                       │   │
│  │   • Boiler pressure ↔ Steam bundle supply temps              │   │
│  │   • Chiller setpoint ↔ CW pump speed ↔ AHU CW valve        │   │
│  │   • Fan coil demand  ↔ HW pump amps                         │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│              ┌──────────────┼──────────────┐                       │
│              │              │              │                       │
│              ▼              │              ▼                       │
│  ┌─────────────────┐       │   ┌─────────────────────┐            │
│  │ SNAPSHOT MODE    │       │   │ TEMPORAL MODE        │            │
│  │                  │       │   │                      │            │
│  │ Per-equip head:  │       │   │ LSTM (32→64, 1 lyr)  │            │
│  │  Linear(32→1)   │       │   │ over window of       │            │
│  │                  │       │   │ snapshots             │            │
│  │ Facility head:   │       │   │                      │            │
│  │  MeanPool →     │       │   │ Per-equip head:      │            │
│  │  Linear(32→1)   │       │   │  Linear(64→1)        │            │
│  │                  │       │   │                      │            │
│  └────────┬────────┘       │   │ Facility head:       │            │
│           │                │   │  MeanPool →          │            │
│           │                │   │  Linear(64→1)        │            │
│           │                │   └──────────┬───────────┘            │
│           │                │              │                        │
│           ▼                │              ▼                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                       OUTPUT                                 │   │
│  │                                                              │   │
│  │   equip_scores: [1, 59]  — per-equipment anomaly score      │   │
│  │   facility_score: [1, 1] — overall facility health           │   │
│  │                                                              │   │
│  │   Score meaning: 0.0 = normal, higher = more anomalous      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Input: FacilitySnapshot

A single point-in-time reading of all equipment in a facility:

```
┌─────────────────────────────────────────────────────────────┐
│ FacilitySnapshot                                             │
│                                                              │
│ features: [num_equip × MAX_SENSORS] (zero-padded)           │
│ mask:     [num_equip × MAX_SENSORS] (1=real, 0=missing)     │
│ equip_types: [num_equip]  (0=AHU, 1=DOAS, 2=Boiler, ...)   │
│ equip_ids:   [num_equip]  ("warren-ahu-6", "warren-boiler-1")│
│                                                              │
│ Example row (AHU-6, 12 sensors):                            │
│ ┌──────┬───────┬───────┬───────┬──────┬──────┬────┬────┬────┐│
│ │  SP  │ Space │Supply │Mixed  │OA Dmp│Ret Dp│ HW │ CW │Amps││
│ │65.0  │  -    │65.6   │55.3   │100%  │ 13%  │ 0% │ 0% │5.83││
│ └──────┴───────┴───────┴───────┴──────┴──────┴────┴────┴────┘│
│ mask: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]                 │
│                                                              │
│ Example row (SteamBundle, 5 sensors, padded to 12):          │
│ ┌──────┬───────┬───────┬───────┬──────┬──┬──┬──┬──┬──┬──┬──┐│
│ │  SP  │Supply │Return │Valve1 │Valve2│ 0│ 0│ 0│ 0│ 0│ 0│ 0││
│ │135.5 │141.3  │  -    │ 21%   │  -   │  │  │  │  │  │  │  ││
│ └──────┴───────┴───────┴───────┴──────┴──┴──┴──┴──┴──┴──┴──┘│
│ mask: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]                 │
└─────────────────────────────────────────────────────────────┘
```

### Missing Value Handling

```
Sensor reading available?
        │
        ├── YES → raw value passed to type encoder
        │
        └── NO  → value = 0.0, mask = 0.0
                  If ALL sensors missing for an equipment:
                  → learnable missing_embed vector used instead
                  (model learns what "no data" looks like)
```

### Output: PanoptesOutput

```
┌─────────────────────────────────────────────────────────────┐
│ PanoptesOutput                                               │
│                                                              │
│ facility_score: 0.0312  (0=normal)                          │
│                                                              │
│ equipment_scores:                                            │
│   warren-ahu-6 ........... 0.0021  ✓ normal                 │
│   warren-ahu-1 ........... 0.0043  ✓ normal                 │
│   warren-ahu-5 ........... 0.8721  ✗ ALERT (high)           │
│   warren-boiler-1 ........ 0.0015  ✓ normal                 │
│   warren-fancoil-10 ...... 1.2340  ✗ ALERT (critical)       │
│   warren-cwpump-3 ........ 0.5102  ✗ ALERT (medium)         │
│   ...                                                        │
│                                                              │
│ alerts: [                                                    │
│   { id: "warren-fancoil-10", score: 1.234, sev: "critical" }│
│   { id: "warren-ahu-5",      score: 0.872, sev: "high"     }│
│   { id: "warren-cwpump-3",   score: 0.510, sev: "medium"   }│
│ ]                                                            │
│                                                              │
│ Severity thresholds (relative to configured threshold T):    │
│   low:      score > T                                        │
│   medium:   score > 1.5T                                     │
│   high:     score > 2.5T                                     │
│   critical: score > 4.0T                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Equipment Types & Sensor Channels

### Air Handler Unit (AHU) — 12 sensors

| # | Sensor | Unit | Normal Range | Example |
|---|--------|------|--------------|---------|
| 0 | Setpoint | °F | 65–78 | 65.0 |
| 1 | Space Temp | °F | 68–78 | — |
| 2 | Supply Temp | °F | 52–80 | 65.6 |
| 3 | Mixed Air Temp | °F | 55–72 | 55.3 |
| 4 | OA Damper | % | 0–100 | 100 |
| 5 | Return Damper | % | 0–100 | 13 |
| 6 | HW Valve | % | 0–100 | 0 |
| 7 | CW Valve | % | 0–100 | 0 |
| 8 | Amps | A | 0–50 | 5.83 |
| 9 | Status | 0/1 | 0=idle, 1=run | 1 |
| 10 | Fan Speed | % | 0–100 | — |
| 11 | Discharge Temp | °F | 50–90 | — |

### DOAS — 6 sensors

| # | Sensor | Unit | Example |
|---|--------|------|---------|
| 0 | Setpoint | °F | 55.0 |
| 1 | Supply Temp | °F | — |
| 2 | Space Temp | °F | — |
| 3 | Outdoor Temp | °F | — |
| 4 | Heat Status | 0/1 | 0 |
| 5 | Cool Status | 0/1 | 0 |

### Steam Boiler — 7 sensors

| # | Sensor | Unit | Example |
|---|--------|------|---------|
| 0 | Supply Temp | °F | 127.1 |
| 1 | Flash Tank Temp | °F | 111.3 |
| 2 | Header PSI | PSI | 88.52 |
| 3 | Status | 0/1 | 0 |
| 4 | Lead/Lag | 0/1 | 0=lead |
| 5 | Runtime | hrs | 17.0 |
| 6 | Safeties | 0/1 | 0=idle |

### Steam Bundle — 5 sensors

| # | Sensor | Unit | Example |
|---|--------|------|---------|
| 0 | Setpoint | °F | 135.5 |
| 1 | Supply Temp | °F | 141.3 |
| 2 | Return Temp | °F | — |
| 3 | Valve 1 | % | 21 |
| 4 | Valve 2 | % | — |

### Fan Coil — 9 sensors

| # | Sensor | Unit | Example |
|---|--------|------|---------|
| 0 | Setpoint | °F | 73.5 |
| 1 | Space Temp | °F | 74.5 |
| 2 | Supply Temp | °F | 71.4 |
| 3 | HW Valve | % | 0 |
| 4 | CW Valve | % | 0 |
| 5 | OA Damper | % | 100 |
| 6 | Amps | A | 2.20 |
| 7 | Status | 0/1 | 1 |
| 8 | Fan Speed | % | — |

### Pump — 7 sensors

| # | Sensor | Unit | Example |
|---|--------|------|---------|
| 0 | Speed | % | 45 |
| 1 | Amps | A | 10.78 |
| 2 | PSI Setpoint | PSI | 15.0 |
| 3 | Discharge PSI | PSI | — |
| 4 | Runtime | hrs | 58.9 |
| 5 | Status | 0/1 | 1 |
| 6 | Flow | GPM | — |

### Chiller — 9 sensors

| # | Sensor | Unit | Example |
|---|--------|------|---------|
| 0 | Setpoint | °F | 47.5 |
| 1 | Supply Temp | °F | 83.6 |
| 2 | Return Temp | °F | — |
| 3 | Pressure | PSI | 15.04 |
| 4 | Amps | A | — |
| 5 | Status | 0/1 | 1 |
| 6 | Runtime | hrs | 31.7 |
| 7 | Interlocks | 0/1 | 1=ok |
| 8 | Enabled | 0/1 | 1 |

---

## Warren Facility Layout

Heritage Pointe of Warren — 59 equipment total:

```
┌──────────────────────────────────────────────────────────────────┐
│                    HERITAGE POINTE OF WARREN                      │
│                                                                   │
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │   A WING          │    │   FAHL WING       │                    │
│  │                    │    │                    │                    │
│  │  AHU-6 (basement) │    │  AHU-1 (main)     │                    │
│  │  SteamBundle-1    │    │  AHU-4 (salon)    │                    │
│  │  SteamBundle-5    │    │  SteamBundle-fahl │                    │
│  │  CWBooster-1,2   │    │  HWPump-11,12     │                    │
│  │  CWPump-5,6      │    │  DOAS (fahl)      │                    │
│  │  HWPump-7,8      │    │                    │                    │
│  └──────────────────┘    └──────────────────┘                    │
│                                                                   │
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │   DINING/CHAPEL   │    │   INNIS WING      │                    │
│  │                    │    │                    │                    │
│  │  AHU-2 (dining)   │    │  Chiller-2        │                    │
│  │  FanCoil-7..10    │    │  CHWPump-3,4      │                    │
│  │  (chapel N/S)     │    │  CWPump-3,4       │                    │
│  │                    │    │  HWPump-5,6       │                    │
│  └──────────────────┘    │  SteamBundle-6    │                    │
│                           └──────────────────┘                    │
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │   MCALLISTER      │    │   SOUDER WING     │                    │
│  │                    │    │                    │                    │
│  │  AHU-5 (laundry)  │    │  Chiller-1        │                    │
│  │  SteamBundle-3    │    │  SteamBundle-2    │                    │
│  │  SteamBundle-7    │    │  HWPump-3,4       │                    │
│  │  SteamBundle-8    │    │                    │                    │
│  │  HWPump-1,2       │    └──────────────────┘                    │
│  └──────────────────┘                                            │
│                                                                   │
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │   COVE AREA       │    │   OTHER            │                    │
│  │                    │    │                    │                    │
│  │  FanCoil-1..6     │    │  AHU-7 (natator.) │                    │
│  │  (cove S/N/kitch) │    │  FanCoil-14..18   │                    │
│  │                    │    │  (activity, exec) │                    │
│  └──────────────────┘    │  FanCoil-11 (vest)│                    │
│                           └──────────────────┘                    │
│                                                                   │
│  ┌──────────────────────────────────────────┐                    │
│  │   BOILER ROOM (central)                   │                    │
│  │                                            │                    │
│  │  Boiler-1 (lead)   ──┐                    │                    │
│  │  Boiler-2 (lag)     ──┼── Header (88 PSI) │                    │
│  │  Boiler-3 (lag)     ──┘   Flash Tank      │                    │
│  │                            (111.3°F)       │                    │
│  │  Chompson SteamBundle-4                   │                    │
│  │  HWPump-9,10                              │                    │
│  └──────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Cross-Equipment Correlations

The transformer attention layer learns relationships like:

```
CORRELATION CHAINS (examples):

1. HEATING CHAIN
   Boiler PSI ──→ Steam Bundle Supply Temp ──→ HW Pump Speed ──→ AHU HW Valve ──→ Supply Temp

   Anomaly: Boiler PSI drops but steam bundle supply stays high
            → possible sensor drift or stuck valve

2. COOLING CHAIN
   Chiller Supply Temp ──→ CHW Pump Speed ──→ AHU CW Valve ──→ Supply Temp

   Anomaly: Chiller supply rises but CW pump doesn't speed up
            → possible pump fault or control failure

3. AIRFLOW CHAIN
   AHU OA Damper ──→ Mixed Air Temp ──→ Supply Temp ──→ Space Temp

   Anomaly: OA damper at 100% but mixed air temp doesn't match outdoor
            → possible damper stuck or sensor fault

4. LOAD BALANCE
   Fan Coil Amps (zone) ←→ AHU Amps (serving that zone)

   Anomaly: Fan coils drawing heavy amps but AHU idle
            → possible scheduling conflict or control error
```

---

## Usage

### Creating the Model

```rust
use axonml_hvac::panoptes::*;

// Create model for Warren (59 equipment)
let model = Panoptes::new(59);
println!("Parameters: {}", model.num_parameters()); // ~47K
```

### Building a Snapshot from Live Data

```rust
let config = FacilityConfig::warren();
let mut snap = FacilitySnapshot::for_warren(&config);

// AHU-6: setpoint=65, space=None, supply=65.6, mixed=55.3,
//         oa_damper=100, ret_damper=13, hw=0, cw=0, amps=5.83, status=1
let slot = config.id_to_slot["warren-ahu-6"];
snap.set_equipment(slot, "warren-ahu-6", EQUIP_AHU, &[
    Some(65.0),  // setpoint
    None,        // space temp (missing)
    Some(65.6),  // supply temp
    Some(55.3),  // mixed air temp
    Some(100.0), // OA damper %
    Some(13.0),  // return damper %
    Some(0.0),   // HW valve %
    Some(0.0),   // CW valve %
    Some(5.83),  // amps
    Some(1.0),   // status (running)
    None,        // fan speed (missing)
    None,        // discharge temp (missing)
]);

// Boiler-1: supply=127.1, flash=111.3, psi=88.52, idle, lead, 17hrs, ok
let slot = config.id_to_slot["warren-boiler-1"];
snap.set_equipment(slot, "warren-boiler-1", EQUIP_BOILER, &[
    Some(127.1), // supply temp
    Some(111.3), // flash tank temp
    Some(88.52), // header PSI
    Some(0.0),   // status (idle)
    Some(0.0),   // lead/lag (lead=0)
    Some(17.0),  // runtime hrs
    Some(0.0),   // safeties (idle=0)
]);

// ... repeat for all 59 equipment
```

### Single-Snapshot Inference

```rust
let (equip_scores, facility_score) = model.forward_snapshot(&snap);

// Convert to structured output
let scores_vec = equip_scores.data().to_vec();
let fac_score = facility_score.data().to_vec()[0];
let output = PanoptesOutput::from_scores(&scores_vec, fac_score, &config, 0.5);

println!("{}", output.summary());
// Panoptes Facility Health: 0.0312 (0=normal)
//   3 alert(s):
//     [FanCoil ] warren-fancoil-10 — score: 1.2340 (critical)
//     [AHU     ] warren-ahu-5 — score: 0.8721 (high)
//     [Pump    ] warren-cwpump-3 — score: 0.5102 (medium)
```

### Temporal Inference (Sliding Window)

```rust
// Collect snapshots over time (e.g., every 5 minutes)
let mut history: Vec<FacilitySnapshot> = Vec::new();

loop {
    let snap = read_live_sensors(&config);
    history.push(snap);

    // Keep a 1-hour window (12 snapshots at 5-min intervals)
    if history.len() > 12 {
        history.remove(0);
    }

    // Temporal inference uses LSTM to detect trends
    let (equip_scores, facility_score) = model.forward_temporal(&history);

    let scores_vec = equip_scores.data().to_vec();
    let fac_score = facility_score.data().to_vec()[0];
    let output = PanoptesOutput::from_scores(&scores_vec, fac_score, &config, 0.5);

    if !output.alerts.is_empty() {
        send_alerts(&output.alerts);
    }

    std::thread::sleep(std::time::Duration::from_secs(300));
}
```

### Training

```rust
use axonml_optim::{Adam, Optimizer};
use axonml_nn::MSELoss;

let model = Panoptes::new(59);
let params = model.parameters();
let mut optimizer = Adam::new(params, 1e-3);
let mse = MSELoss::new();

// Target: all zeros (normal operation = 0 anomaly score)
let target = Variable::new(
    Tensor::from_vec(vec![0.0; 59], &[1, 59]).unwrap(), false,
);

for epoch in 0..100 {
    for snapshot in &training_snapshots {
        optimizer.zero_grad();

        let (equip_scores, _) = model.forward_snapshot(snapshot);
        let loss = mse.compute(&equip_scores, &target);

        loss.backward();
        optimizer.step();
    }
}
```

---

## Training Strategy

### Phase 1: Normal Operation Baseline

Train on historical data where all equipment is operating normally:

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING PHASE 1: Learn "normal"                         │
│                                                          │
│ Input:  Snapshots from normal operation (weeks of data)  │
│ Target: All anomaly scores = 0.0                         │
│ Loss:   MSE(predicted_scores, zeros)                     │
│                                                          │
│ The model learns:                                        │
│  • Normal sensor ranges for each equipment type          │
│  • Expected cross-equipment correlations                 │
│  • Time-of-day and seasonal patterns (temporal mode)     │
└─────────────────────────────────────────────────────────┘
```

### Phase 2: Fault Injection (Optional)

Inject known fault patterns and train with positive anomaly scores:

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING PHASE 2: Learn fault signatures                 │
│                                                          │
│ Input:  Normal snapshots with injected faults            │
│ Target: Anomaly score > 0 for affected equipment         │
│                                                          │
│ Fault injection examples:                                │
│  • Stuck valve:     valve_pos constant despite demand    │
│  • Sensor drift:    gradual offset on temp sensor        │
│  • Pump failure:    amps→0, flow→0 while status=running  │
│  • Boiler lockout:  PSI drops, supply temp drops         │
│  • Short cycling:   rapid status toggling                │
└─────────────────────────────────────────────────────────┘
```

---

## Model Parameters

```
┌─────────────────────────────────────────────────┐
│ PARAMETER BREAKDOWN (59 equipment)               │
│                                                  │
│ Per-Type Encoders:                               │
│   AHU:     12×32 + 32 + 32×32 + 32 = 1,472     │
│   DOAS:     6×32 + 32 + 32×32 + 32 = 1,280     │
│   Boiler:   7×32 + 32 + 32×32 + 32 = 1,312     │
│   Bundle:   5×32 + 32 + 32×32 + 32 = 1,248     │
│   FanCoil:  9×32 + 32 + 32×32 + 32 = 1,376     │
│   Pump:     7×32 + 32 + 32×32 + 32 = 1,312     │
│   Chiller:  9×32 + 32 + 32×32 + 32 = 1,376     │
│                                                  │
│ Embeddings:                                      │
│   Type:  7 × 32  = 224                          │
│   ID:   59 × 32  = 1,888                        │
│   Missing:  1×32 = 32                           │
│                                                  │
│ Transformer (2 layers):    ~17,000              │
│ LSTM (32→64):              ~25,000              │
│ Output Heads:              ~260                  │
│                                                  │
│ TOTAL: ~47,000 parameters                       │
│ Memory: ~184 KB (f32)                           │
│ Edge-deployable: YES (Raspberry Pi class)       │
└─────────────────────────────────────────────────┘
```

---

## Comparison: Panoptes vs Per-Equipment Models

```
┌────────────────────────┬──────────────────┬──────────────────┐
│ Feature                │ Per-Equipment    │ Panoptes          │
│                        │ (Hermes/Tyche)   │ (facility-wide)   │
├────────────────────────┼──────────────────┼──────────────────┤
│ Scope                  │ 1 equipment      │ All 59 equipment  │
│ Cross-equip awareness  │ None             │ Full attention    │
│ Anomaly types          │ Local only       │ Local + systemic  │
│ Parameters             │ ~70K each        │ ~47K total        │
│ Detects stuck valve    │ ✓                │ ✓                 │
│ Detects sensor drift   │ ✓                │ ✓                 │
│ Detects cascade fault  │ ✗                │ ✓                 │
│ Detects control conflict│ ✗               │ ✓                 │
│ Detects load imbalance │ ✗                │ ✓                 │
│ Temporal trends        │ ✓ (per unit)     │ ✓ (facility-wide) │
│ Edge deployment        │ ✓                │ ✓                 │
└────────────────────────┴──────────────────┴──────────────────┘
```

---

## API Reference

### Structs

| Struct | Description |
|--------|-------------|
| `Panoptes` | Main model — cross-equipment transformer + temporal LSTM |
| `FacilitySnapshot` | Point-in-time sensor readings for all equipment |
| `FacilityConfig` | Equipment layout definition (IDs, types, slot mapping) |
| `PanoptesOutput` | Structured inference output with alerts |
| `PanoptesAlert` | Single equipment alert with severity |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `EQUIP_AHU` | 0 | Air Handler Unit type ID |
| `EQUIP_DOAS` | 1 | Dedicated Outdoor Air System type ID |
| `EQUIP_BOILER` | 2 | Steam Boiler type ID |
| `EQUIP_STEAM_BUNDLE` | 3 | Steam Bundle type ID |
| `EQUIP_FAN_COIL` | 4 | Fan Coil Unit type ID |
| `EQUIP_PUMP` | 5 | Pump type ID |
| `EQUIP_CHILLER` | 6 | Chiller type ID |
| `NUM_EQUIP_TYPES` | 7 | Total equipment types |
| `EMBED_DIM` | 32 | Common embedding dimension |
| `MAX_SENSORS` | 12 | Max sensor channels (for padding) |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `Panoptes::new` | `(num_equipment: usize) -> Panoptes` | Create model |
| `Panoptes::num_parameters` | `() -> usize` | Total trainable param count (~47K for 59 equip) |
| `Panoptes::encode_snapshot` | `(&FacilitySnapshot) -> Variable` | Encode raw sensors to `[1, N, 32]` |
| `Panoptes::forward_snapshot` | `(&FacilitySnapshot) -> (Variable, Variable)` | Single-point inference → (equip_scores, facility_score) |
| `Panoptes::forward_temporal` | `(&[FacilitySnapshot]) -> (Variable, Variable)` | Sequence inference via LSTM |
| `Panoptes::parameters` | `() -> Vec<Parameter>` | All trainable params |
| `FacilityConfig::warren` | `() -> FacilityConfig` | Pre-built Warren config (59 equipment) |
| `FacilityConfig::new` | `(Vec<(String, usize)>) -> FacilityConfig` | Custom facility from (id, equip_type) pairs |
| `FacilitySnapshot::new` | `(num_equipment: usize) -> FacilitySnapshot` | Empty snapshot |
| `FacilitySnapshot::for_warren` | `(&FacilityConfig) -> FacilitySnapshot` | Empty snapshot sized for Warren |
| `FacilitySnapshot::set_equipment` | `(slot, id, equip_type, &[Option<f32>])` | Set sensors for one equipment slot |
| `PanoptesOutput::from_scores` | `(&[f32], f32, &FacilityConfig, threshold: f32) -> PanoptesOutput` | Structure output into alerts |
| `PanoptesOutput::summary` | `() -> String` | Human-readable summary |

---

*Last updated: 2026-04-16 (v0.6.1)*

