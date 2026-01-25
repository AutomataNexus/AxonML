# axonml-dashboard

<!-- Logo placeholder -->
<p align="center">
  <img src="../../docs/assets/logo.svg" alt="AxonML Dashboard" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust 1.75+"/>
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0"/>
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"/>
</p>

---

## Overview

**axonml-dashboard** is a modern, reactive web dashboard for the AxonML Machine Learning Framework built with Leptos and WebAssembly. It provides a comprehensive interface for managing ML training runs, model registry, inference endpoints, and system settings with full authentication support including MFA via TOTP and WebAuthn.

---

## Features

- **Reactive UI** - Built with Leptos for fine-grained reactivity and optimal performance
- **WebAssembly** - Compiled to WASM for native-like speed in the browser
- **Authentication** - Complete auth flow with login, registration, and session management
- **Multi-Factor Authentication** - TOTP (authenticator apps) and WebAuthn (hardware keys) support
- **Training Management** - Monitor training runs with real-time metrics via WebSocket
- **Model Registry** - Browse, upload, and manage model versions
- **Inference Dashboard** - Deploy models and monitor inference endpoints
- **Dark Mode** - Toggle between light and dark themes
- **Toast Notifications** - Real-time feedback for user actions
- **In-App Terminal** - Slide-out terminal with WebSocket PTY for server-side shell access
- **Responsive Design** - Works seamlessly on desktop and tablet devices

---

## Modules

| Module | Description |
|--------|-------------|
| `api` | HTTP client for backend API communication with auth token management |
| `auth` | Authentication pages (login, register) and MFA setup components |
| `auth::session` | Session management, protected routes, and token refresh logic |
| `auth::mfa` | MFA verification UI for TOTP and WebAuthn challenges |
| `auth::mfa_setup` | MFA enrollment pages for TOTP, WebAuthn, and recovery codes |
| `components` | Reusable UI components (navbar, sidebar, charts, tables, etc.) |
| `components::charts` | Chart components for metrics visualization |
| `components::forms` | Form inputs with validation support |
| `components::modal` | Modal dialog component |
| `components::toast` | Toast notification system |
| `components::terminal` | Slide-out terminal with WebSocket PTY |
| `pages::dashboard` | Main dashboard overview and app shell layout |
| `pages::training` | Training run list, detail view, and new run creation |
| `pages::models` | Model registry with version management and upload |
| `pages::inference` | Inference endpoints, metrics, and deployment UI |
| `pages::settings` | User profile and security settings |
| `state` | Global reactive state management with Leptos signals |
| `types` | TypeScript-like type definitions for API models |

---

## Usage

### Prerequisites

- Rust 1.75+ with `wasm32-unknown-unknown` target
- [Trunk](https://trunkrs.dev/) for building and serving

### Install Dependencies

```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Install Trunk
cargo install trunk
```

### Development Server

```bash
cd crates/axonml-dashboard

# Start development server with hot reload
trunk serve

# The dashboard will be available at http://localhost:8080
```

### Production Build

```bash
# Build optimized WASM bundle
trunk build --release

# Output will be in the dist/ directory
```

### Configuration

The dashboard connects to the AxonML server API. Configure the backend URL in your environment:

```bash
# Set API base URL (defaults to same origin)
export AXONML_API_URL=http://localhost:3000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          axonml-dashboard                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Router    │───▶│   Pages     │───▶│ Components  │                 │
│  │  (Leptos)   │    │             │    │             │                 │
│  └─────────────┘    └──────┬──────┘    └─────────────┘                 │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Global State (Signals)                       │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │   │
│  │  │   User    │  │ Training  │  │  Models   │  │ Inference │    │   │
│  │  │   State   │  │   State   │  │   State   │  │   State   │    │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      API Client (gloo-net)                       │   │
│  │  ┌────────┐  ┌──────────┐  ┌────────┐  ┌───────────┐           │   │
│  │  │  Auth  │  │ Training │  │ Models │  │ Inference │           │   │
│  │  └────────┘  └──────────┘  └────────┘  └───────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
└────────────────────────────┼────────────────────────────────────────────┘
                             │ HTTP/WebSocket
                             ▼
                   ┌───────────────────┐
                   │  axonml-server    │
                   │   (REST API)      │
                   └───────────────────┘
```

---

## Page Routes

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing/Dashboard | Public landing or dashboard if authenticated |
| `/login` | Login | User authentication |
| `/register` | Register | New user registration |
| `/dashboard` | Dashboard | Main overview with stats and recent activity |
| `/training` | Training List | All training runs with status filters |
| `/training/new` | New Training | Create a new training run |
| `/training/:id` | Training Detail | Real-time metrics and logs for a run |
| `/models` | Models List | Browse registered models |
| `/models/upload` | Model Upload | Upload a new model version |
| `/models/:id` | Model Detail | Model versions and metadata |
| `/inference` | Inference Overview | Endpoint status and metrics |
| `/inference/endpoints` | Endpoints List | All inference endpoints |
| `/inference/endpoints/:id` | Endpoint Detail | Endpoint configuration and stats |
| `/inference/metrics` | Metrics | Inference performance metrics |
| `/settings` | Settings | User settings overview |
| `/settings/profile` | Profile | Edit user profile |
| `/settings/security` | Security | Password and MFA settings |
| `/settings/security/totp` | TOTP Setup | Configure authenticator app |
| `/settings/security/webauthn` | WebAuthn Setup | Register hardware security key |
| `/settings/security/recovery` | Recovery Codes | View/regenerate recovery codes |

---

## Tests

```bash
# Run all tests
cargo test -p axonml-dashboard

# Run with output
cargo test -p axonml-dashboard -- --nocapture

# Run WASM tests (requires wasm-pack)
wasm-pack test --headless --chrome
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
