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

## PM2 Process Management

AxonML server and dashboard run via PM2 for persistence across reboots.

```bash
# Start/Stop/Restart
pm2 start ecosystem.config.js             # Start all (server + dashboard dev)
pm2 start ecosystem.config.js --only axonml-server     # Server only
pm2 start ecosystem.config.js --only axonml-dashboard  # Dashboard dev server only
pm2 stop axonml-server                    # Stop server
pm2 stop axonml-dashboard                 # Stop dashboard dev server
pm2 restart axonml-server                 # Restart server
pm2 reload axonml-server                  # Zero-downtime reload

# Monitoring
pm2 status                                # View process status
pm2 logs axonml-server                    # View server logs
pm2 logs axonml-dashboard                 # View dashboard logs
pm2 logs axonml-server --lines 100        # View last 100 lines
pm2 monit                                 # Real-time monitoring dashboard

# Persistence (run once after first setup)
pm2 save                                  # Save current process list
pm2 startup                               # Generate startup script for reboot persistence

# Production deployment
cargo build --release -p axonml-server    # Build server binary
cd crates/axonml-dashboard && trunk build --release  # Build dashboard WASM
pm2 start ecosystem.config.js --only axonml-server --env production
pm2 save

# Ports
# - axonml-server:    3021 (API backend)
# - axonml-dashboard: 8082 (nginx production) or 8081 (trunk dev)
```

## Database Management

AxonML uses Aegis-DB as its document store. The database is also managed via PM2.

```bash
# Initialize/Re-initialize Database
./AxonML_DB_Init.sh                       # Create collections + default admin user
./AxonML_DB_Init.sh --with-user           # Also create DevOps admin user

# Default Users (created by init script)
# Admin:  admin@axonml.local / admin
# DevOps: DevOps@automatanexus.com / Invertedskynet2$ (with --with-user flag)

# Check Aegis-DB status
pm2 status aegis-db
curl http://127.0.0.1:7001/health

# Database location
# Data: configured in Aegis-DB startup (typically /var/lib/aegis or /tmp/aegis-data)
# Config: ~/.axonml/config.toml
```

## First-Time Setup

```bash
# 1. Build the release binary
cargo build --release -p axonml-server

# 2. Ensure Aegis-DB is running (should already be via PM2)
pm2 status aegis-db

# 3. Initialize database with DevOps user
./AxonML_DB_Init.sh --with-user

# 4. Create log directory
sudo mkdir -p /var/log/axonml
sudo chown $USER:$USER /var/log/axonml

# 5. Start server via PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # Follow the instructions to enable boot persistence
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

## Critical Rules - READ THIS

**NEVER use these flags without explicit user permission:**
- `--no-default-features` - This disables application functionality, not dev tools
- `--features=""` with empty value
- Any flag that removes or disables features

**Before running build commands:**
1. STOP and think: "What does this flag actually do?"
2. If unsure, ASK the user instead of guessing
3. Trunk's dev server (live-reload websocket) is controlled by `trunk serve` vs `trunk build`, NOT by Cargo features

**Dashboard builds:**
```bash
# Development (with live reload)
trunk serve

# Production (NO --no-default-features, just this)
trunk build --release
```

**Do not rush. Think first. Ask if unsure.**

## Web Architecture

- **Dashboard (frontend):** Leptos/WASM (CSR) on port 8082 (nginx) or 8081 (trunk dev)
- **Server (backend):** Axum REST + WebSocket on port 3021
- **Nginx proxy:** Serves static WASM from dist/, proxies `/api/*` to :3021
- **Database:** Aegis-DB on port 7001 (configured in `~/.axonml/config.toml`)

## Key Dependencies

- **Async:** tokio, axum, tower
- **Frontend:** leptos, gloo-net
- **Auth:** jsonwebtoken, totp-rs, argon2
- **ML:** matrixmultiply, rayon, half, bytemuck
- **Serialization:** serde, bincode, prost
- **CLI:** clap (derive), indicatif, colored
