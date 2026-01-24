# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AxonML is a PyTorch-equivalent machine learning framework written in pure Rust. It's a monorepo workspace with 22 crates organized in layered architecture.

## Build Commands

```bash
# Build
cargo build --workspace                    # Build all crates
cargo build --release                      # Release build (LTO enabled)

# Test
cargo test --workspace                     # Run all tests
cargo test -p axonml-nn                    # Test single crate
cargo test test_name                       # Run specific test

# Lint and format
cargo clippy --workspace                   # Lint (config in clippy.toml)
cargo fmt --all                            # Format code
cargo fmt --all -- --check                 # Check formatting

# Dashboard development (requires Trunk and wasm32-unknown-unknown target)
cd crates/axonml-dashboard
trunk serve                                # Dev server on :8081, proxies API to :3021
trunk build --release                      # Production WASM build

# Server development
cargo run -p axonml-server                 # API server on :3000 (dev: use --port 3021)
cargo run -p axonml-server -- --port 3021  # Match dashboard proxy for local dev

# CLI installation
cargo install --path crates/axonml-cli
```

## Architecture

**Foundation Layer:**
- `axonml-core` - Device abstraction (CPU, CUDA, Vulkan, Metal, WebGPU), data types, storage
- `axonml-tensor` - N-dimensional arrays with broadcasting, views, slicing, BLAS ops

**Computation Layer:**
- `axonml-autograd` - Reverse-mode automatic differentiation, computational graphs
- `axonml-nn` - Neural network modules (Linear, Conv, BatchNorm, Attention, RNN/LSTM/GRU)
- `axonml-optim` - Optimizers (SGD, Adam, AdamW, RMSprop) and LR schedulers

**Data & Domain Layer:**
- `axonml-data` - DataLoader, Dataset trait, batching, samplers
- `axonml-vision` - Image transforms, MNIST/CIFAR datasets, CNN architectures
- `axonml-audio` - MelSpectrogram, MFCC, audio transforms
- `axonml-text` - Tokenizers (Whitespace, Char, BPE), vocabulary

**Advanced Features:**
- `axonml-distributed` - DistributedDataParallel, collective ops
- `axonml-serialize` - Model save/load (SafeTensors, StateDict)
- `axonml-onnx` - ONNX import/export (40+ operators)
- `axonml-quant` - Quantization (INT8/INT4/INT5, F16)
- `axonml-fusion` - Kernel fusion optimization
- `axonml-jit` - JIT compilation, graph tracing
- `axonml-profile` - Profiling tools
- `axonml-llm` - LLM architectures (BERT, GPT-2)

**Application Layer:**
- `axonml` - Main umbrella crate with feature flags
- `axonml-cli` - CLI for training, evaluation, model management
- `axonml-tui` - Terminal UI dashboard
- `axonml-dashboard` - Leptos/WASM web frontend
- `axonml-server` - Axum REST API backend with JWT auth

## Code Conventions

**File structure pattern:**
```rust
//! Module Name - Description
//! @version X.Y.Z

// =============================================================================
// Imports
// =============================================================================

// =============================================================================
// Types and Traits
// =============================================================================

// =============================================================================
// Implementations
// =============================================================================

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests { ... }
```

**Clippy configuration (relaxed for ML code):**
- `too-many-arguments-threshold = 12`
- `single-char-binding-names-threshold = 8` (x, y, n, i, j, k allowed)
- `cognitive-complexity-threshold = 50`

**API design:** Follows PyTorch patterns for familiarity.

## Web Architecture

- **Dashboard (frontend):** Leptos/WASM (CSR) on port 8080 (dev: 8081 via Trunk)
- **Server (backend):** Axum REST + WebSocket on port 3000 (dev: 3021)
- **Dev proxy:** Trunk proxies `/api/*` from :8081 → :3021
- **Database:** Uses Aegis-DB (connection configured in `~/.axonml/config.toml`)

## Key Dependencies

- **Async:** tokio, axum, tower
- **Frontend:** leptos, gloo-net
- **Auth:** jsonwebtoken, totp-rs, argon2
- **ML:** matrixmultiply, rayon, half, bytemuck
- **Serialization:** serde, bincode, prost
- **CLI:** clap (derive), indicatif, colored
