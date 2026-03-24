# NightVision — Multi-Domain Infrared Object Detection

A thermal/infrared object detection model designed for deployment across wildly different imaging scenarios — wildlife monitoring, human detection, interstellar thermal signatures, and vehicle tracking. Built on a YOLOX-inspired architecture adapted for the unique characteristics of IR imagery.

**~2.6M parameters** (wildlife config) | Input: [B, 1, H, W] thermal or [B, 3, H, W] multi-band | Output: per-scale (cls, bbox, obj, domain)

---

## Table of Contents

- [Architecture](#architecture)
- [Thermal Domains](#thermal-domains)
- [Configurations](#configurations)
- [API Reference](#api-reference)
- [Thermal IR Challenges](#thermal-ir-challenges)
- [Loss Function](#loss-function)
- [File Structure](#file-structure)

---

## Architecture

```
IR Image [B, 1, H, W]  (single-channel thermal)
    or   [B, 3, H, W]  (multi-band / pseudo-color)
     |
 ┌─────────────────────────────────┐
 │ Thermal Stem                     │
 │  1-ch adapter: Conv1×1(1→3)     │  (skipped for 3-ch input)
 │  Stem: Conv3×3(3→32, s=2) + BN + SiLU
 │  Output: [B, 32, H/2, W/2]     │
 └─────────────────────────────────┘
     |
 ┌─────────────────────────────────┐
 │ CSP Backbone                     │
 │                                  │
 │ Stage 1: CSPBlock(32→64, 1 btn) │  P3: [B, 64, H/4, W/4]
 │     ↓                           │
 │ Stage 2: CSPBlock(64→128, 2 btn)│  P4: [B, 128, H/8, W/8]
 │     ↓                           │
 │ Stage 3: CSPBlock(128→256, 2 btn│  P5: [B, 256, H/16, W/16]
 │                                  │
 └─────────────────────────────────┘
     |  P3  |  P4  |  P5
     ↓      ↓      ↓
 ┌─────────────────────────────────┐
 │ Thermal FPN                      │
 │                                  │
 │ Top-down: P5 → upsample + P4   │  Fuses high-level semantics
 │           P4 → upsample + P3   │  with low-level detail
 │ Bottom-up: P3 → downsample + P4│
 │            P4 → downsample + P5│
 │                                  │
 │ All outputs: 128 channels       │
 └─────────────────────────────────┘
     |  FPN3  |  FPN4  |  FPN5
     ↓        ↓        ↓
 ┌─────────────────────────────────┐
 │ Decoupled Detection Heads (×3)   │
 │                                  │
 │ ┌──────────┐  ┌──────────────┐  │
 │ │ CLS branch│  │ REG branch   │  │
 │ │ Conv3×3   │  │ Conv3×3      │  │
 │ │ Conv1×1   │  │ Conv1×1 bbox │  │
 │ │ →[classes]│  │ Conv1×1 obj  │  │
 │ └──────────┘  └──────────────┘  │
 │                                  │
 │ Optional: Domain head            │
 │ Conv1×1 → [num_domains]         │
 └─────────────────────────────────┘
     |
  Per-scale: (cls_logits, bbox_pred, objectness, domain_logits)
```

### Key Components

**CSP (Cross-Stage Partial) Block:**
- Splits input channels into two branches
- Branch 1: passes through N bottleneck blocks (Conv1×1 → Conv3×3 + residual)
- Branch 2: skip connection
- Concatenate + Conv1×1 merge
- Reduces computation while maintaining gradient flow

**SiLU Activation (Swish):**
- `f(x) = x × sigmoid(x)`
- Smoother than ReLU, better for detection networks
- Used throughout backbone, neck, and heads

**Decoupled Head (YOLOX-style):**
- Separate conv branches for classification and regression
- Prevents task interference — cls and bbox objectives don't compete for the same features
- Objectness scored from the regression branch (shared spatial features)

### Parameter Breakdown

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Channel adapter | ~100 | 1→3 channel conv (thermal only) |
| Stem | ~1K | Conv3×3(3→32) + BN |
| CSP Stage 1 | ~25K | 32→64 channels, 1 bottleneck |
| CSP Stage 2 | ~155K | 64→128 channels, 2 bottlenecks |
| CSP Stage 3 | ~610K | 128→256 channels, 2 bottlenecks |
| Thermal FPN | ~460K | Lateral + top-down + bottom-up |
| Detection Heads ×3 | ~1.3M | Decoupled cls/reg/obj per scale |
| **Total (wildlife/10)** | **~2.6M** | |

## Thermal Domains

NightVision supports domain-specific detection via an optional domain classification head:

| Domain | Description | Thermal Characteristics |
|--------|-------------|------------------------|
| **Wildlife** | Animals, birds, insects | Warm-blooded: high contrast against cool backgrounds |
| **Human** | People, SAR, perimeter | Body heat 37°C, upright posture, limb articulation |
| **Interstellar** | Stars, nebulae, debris | Point sources or diffuse thermal emission against 2.7K background |
| **Vehicle** | Cars, drones, aircraft | Engine heat, tire friction, exhaust plumes |
| **General** | Domain-agnostic | No prior assumptions |

In multi-domain mode, each detection includes a domain tag alongside class/bbox/objectness.

## Configurations

| Preset | Input | Classes | Domains | FPN Ch | Image Size | Use Case |
|--------|-------|---------|---------|--------|------------|----------|
| `wildlife(N)` | 1-ch | N species | 0 | 128 | 320×320 | Trail cameras, nature reserves |
| `human()` | 1-ch | 1 (person) | 0 | 128 | 320×320 | SAR, perimeter security |
| `interstellar(N, bands)` | N-ch | N | 0 | 128 | 512×512 | Space telescopes, multi-band IR |
| `multi_domain(N)` | 1-ch | N | 5 | 128 | 320×320 | All targets with domain tags |
| `edge(N)` | 1-ch | N | 0 | 64 | 256×256 | Embedded / edge deployment |

```rust
use axonml_vision::models::nightvision::{NightVision, NightVisionConfig};

// Wildlife: detect 20 animal species from thermal camera
let model = NightVision::new(NightVisionConfig::wildlife(20));

// Search & rescue: find people in IR footage
let model = NightVision::new(NightVisionConfig::human());

// Astronomy: detect 3 object types in 3-band IR imagery
let model = NightVision::new(NightVisionConfig::interstellar(3, 3));

// Multi-domain: detect anything, tag the domain
let model = NightVision::new(NightVisionConfig::multi_domain(50));

// Edge: compact model for drones / embedded systems
let model = NightVision::new(NightVisionConfig::edge(5));
```

## API Reference

```rust
use axonml_vision::models::nightvision::{NightVision, NightVisionConfig, ThermalDomain};
use axonml_autograd::Variable;

let model = NightVision::new(NightVisionConfig::wildlife(10));

// Full multi-scale detection
let outputs = model.forward_detection(&ir_image);
// outputs: Vec<(cls, bbox, obj, domain)> — one per FPN level (P3, P4, P5)
// cls:    [B, num_classes, H, W]
// bbox:   [B, 4, H, W]
// obj:    [B, 1, H, W]
// domain: [B, num_domains, H, W] or None

// Flattened output (all scales concatenated)
let (cls, bbox, obj) = model.forward_flat(&ir_image);
// cls:  [B, total_anchors, num_classes]
// bbox: [B, total_anchors, 4]
// obj:  [B, total_anchors, 1]

// Module trait (returns class logits only)
let cls = model.forward(&ir_image);

// Domain enum
let domain = ThermalDomain::Wildlife;
println!("{}: index {}", domain.name(), domain.index());
let domain = ThermalDomain::from_index(2); // Interstellar
```

## Thermal IR Challenges

NightVision is designed to handle the unique challenges of infrared imagery:

| Challenge | How NightVision Handles It |
|-----------|--------------------------|
| **No color information** | Single-channel stem with learned 1→3 adapter |
| **Inverted contrast** | SiLU activation handles both polarities |
| **Thermal bloom** | CSP blocks with residual connections preserve edges through bloom |
| **Low spatial resolution** | Multi-scale FPN fuses coarse semantics with fine detail |
| **Varying emissivity** | BatchNorm normalizes intensity distributions |
| **Background clutter** | Decoupled objectness branch learns foreground/background |
| **Domain shift** | Multi-domain mode with domain classification head |
| **Edge deployment** | Compact `edge()` config with 64-ch FPN |

### Input Formats

- **LWIR (8-14 μm)**: Single-channel thermal (most common) — `[B, 1, H, W]`
- **MWIR (3-5 μm)**: Single-channel — `[B, 1, H, W]`
- **Multi-band**: Multiple IR bands stacked — `[B, N, H, W]`
- **Pseudo-color**: False-color thermal (iron, rainbow, etc.) — `[B, 3, H, W]`

## Loss Function

**NightVisionLoss** combines:

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification | 1.0 | BCE with logits per-class |
| Bounding box | 5.0 | CIoU loss (center + size regression) |
| Objectness | 1.0 | BCE with logits foreground/background |
| Domain | 0.5 | CrossEntropy (optional, multi-domain mode) |

## File Structure

```
nightvision/
├── README.md       # This file
├── mod.rs          # Module declaration + re-exports
├── backbone.rs     # ThermalBackbone: CSP blocks + adaptive thermal stem
├── neck.rs         # ThermalFPN: feature pyramid network
├── head.rs         # DecoupledHead: cls/bbox/obj/domain per scale
└── detector.rs     # NightVision: full detector + config + loss + tests
```

---

*Part of [AxonML](https://github.com/AutomataNexus/AxonML) — a Rust deep learning framework by AutomataNexus LLC.*
