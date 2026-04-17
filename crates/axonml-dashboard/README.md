# axonml-dashboard

<!-- Logo placeholder -->
<p align="center">
  <img src="../../docs/assets/logo.svg" alt="AxonML Dashboard" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust 1.75+"/>
  <img src="https://img.shields.io/badge/version-0.6.1-green.svg" alt="Version 0.6.1"/>
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"/>
</p>

---

## Overview

**axonml-dashboard** is the reactive web dashboard for the AxonML Machine Learning Framework. It is a client-side-rendered (CSR) Leptos 0.6 application compiled to WebAssembly, served via Trunk, and talking to `axonml-server` over HTTP + WebSocket. It covers authentication (password + TOTP + WebAuthn + recovery codes), training runs and JSON training notebooks, model registry, dataset management (upload, analyze, Kaggle, built-in library), inference endpoints and metrics, a pretrained-model hub, admin (user management, system stats), system overview, and an in-app PTY terminal.

Last updated: 2026-04-16 — version 0.6.1. Crate is `publish = false` (WASM frontend; distributed via GitHub releases, not crates.io).

---

## Features

- **Leptos 0.6 (CSR)** — Fine-grained reactivity with signals, router, and meta.
- **WebAssembly** — Compiled to `wasm32-unknown-unknown` for native-like speed in the browser.
- **Authentication** — Login, registration, session initialization, `ProtectedRoute` gating, token refresh.
- **Multi-Factor Authentication** — TOTP (authenticator apps), WebAuthn (hardware keys via `web-sys` `CredentialsContainer`), and recovery-code flows, each with a setup page and verification UI.
- **Training Runs** — List, detail (real-time metrics via WebSocket), and create-new pages.
- **Training Notebooks** — List, import, editor, and detail routes for JSON notebooks executed server-side.
- **Model Registry** — Browse models, detail view, and upload a new version (multipart).
- **Dataset Management** — List, upload, analyze, Kaggle search/download, and the built-in dataset library.
- **Inference** — Overview, endpoints list, endpoint detail, and aggregated metrics pages.
- **Pretrained Hub** — Browse pretrained models and manage the local cache.
- **Admin** — User management and system stats (admin-gated).
- **System Overview** — Live system metrics dashboard.
- **Settings** — Profile, security, and three MFA setup pages (TOTP / WebAuthn / Recovery codes).
- **In-App Terminal** — Slide-out terminal component backed by the server's WebSocket PTY.
- **UI Primitives** — Navbar, sidebar, modal, toast, spinner, progress, charts, forms, table, icons, and a page-level `ErrorBoundary`.
- **Landing Gate** — `/` shows a branded loader, then the marketing `LandingPage` or the authenticated dashboard.

---

## Modules

| Module | Description |
|--------|-------------|
| `api` | HTTP client (`gloo-net`) for backend API communication with auth token management |
| `auth::login` | `LoginPage` + `RegisterPage` |
| `auth::mfa` | MFA verification UI for TOTP and WebAuthn challenges |
| `auth::mfa_setup` | Enrollment: `TotpSetupPage`, `WebAuthnSetupPage`, `RecoveryCodesPage` |
| `auth::session` | `SessionInitializer`, `ProtectedRoute`, token refresh logic |
| `components::charts` | Chart components for metric visualization |
| `components::forms` | Form inputs with validation support |
| `components::modal` | Modal dialog |
| `components::toast` | Toast-notification system (`ToastContainer`) |
| `components::terminal` | Slide-out terminal backed by WebSocket PTY |
| `components::navbar` / `sidebar` | Navigation chrome |
| `components::table` / `progress` / `spinner` / `icons` | Reusable UI primitives |
| `components::error_boundary` | `PageErrorBoundary` that wraps the router |
| `pages::landing` | Marketing `LandingPage` |
| `pages::dashboard` | `DashboardPage` + `AppShell` layout |
| `pages::training` | `TrainingListPage`, `NewTrainingPage`, `TrainingDetailPage`, `NotebookListPage`, `NotebookEditorPage`, `NotebookImportPage` |
| `pages::models` | `ModelsListPage`, `ModelDetailPage`, `ModelUploadPage` |
| `pages::datasets` | `DatasetsListPage`, `DatasetUploadPage`, `DataAnalyzePage`, `KagglePage`, `BuiltinDatasetsPage` |
| `pages::inference` | `InferenceOverviewPage`, `EndpointsListPage`, `EndpointDetailPage`, `InferenceMetricsPage` |
| `pages::hub` | `HubBrowsePage`, `HubCachePage` |
| `pages::system` | `SystemOverviewPage` |
| `pages::admin` | `UserManagementPage`, `SystemStatsPage` |
| `pages::settings` | `SettingsPage`, `ProfileSettingsPage`, `SecuritySettingsPage` |
| `state` | Global reactive state via Leptos signals (`provide_app_state`, `use_app_state`) |
| `types` | Type definitions for API models |
| `constants` | Shared constants (routes, API paths, etc.) |
| `utils::js_helpers` | JS interop helpers |
| `utils::webauthn` | WebAuthn binding helpers (`CredentialsContainer`, attestation/assertion) |

---

## Usage

### Prerequisites

- Rust 1.75+ with the `wasm32-unknown-unknown` target
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

# Output is written to the dist/ directory
```

### Configuration

The dashboard connects to the AxonML server API. Configure the backend URL via environment / `Trunk.toml` (defaults to same origin):

```bash
export AXONML_API_URL=http://localhost:3000
```

---

## Architecture

```
+-------------------------------------------------------------------------+
|                          axonml-dashboard                               |
+-------------------------------------------------------------------------+
|                                                                         |
|  +-------------+    +-------------+    +-------------+                  |
|  |   Router    |--->|   Pages     |--->| Components  |                  |
|  |  (Leptos)   |    |             |    |             |                  |
|  +-------------+    +------+------+    +-------------+                  |
|                            |                                            |
|                            v                                            |
|  +-----------------------------------------------------------------+   |
|  |                     Global State (Signals)                       |   |
|  |    User / Session / Training / Models / Inference / Toasts       |   |
|  +-----------------------------------------------------------------+   |
|                            |                                            |
|                            v                                            |
|  +-----------------------------------------------------------------+   |
|  |                      API Client (gloo-net)                       |   |
|  |  Auth / Training / Models / Datasets / Inference / Hub / Admin   |   |
|  +-----------------------------------------------------------------+   |
|                            |                                            |
+----------------------------+--------------------------------------------+
                             | HTTP / WebSocket
                             v
                   +-------------------+
                   |  axonml-server    |
                   |   (REST + WS)     |
                   +-------------------+
```

---

## Page Routes

Assembled in `src/lib.rs::App`. `/`, `/login`, and `/register` are public; everything else is wrapped in `ProtectedRoute` + `AppShell`.

| Route | Page | Description |
|-------|------|-------------|
| `/` | `PublicOrDashboard` | Branded loader -> `LandingPage` if anonymous, else authenticated dashboard |
| `/login` | `LoginPage` | User authentication |
| `/register` | `RegisterPage` | New user registration |
| `/dashboard` | `DashboardPage` | Main overview with stats and recent activity |
| `/training` | `TrainingListPage` | All training runs |
| `/training/new` | `NewTrainingPage` | Create a new training run |
| `/training/:id` | `TrainingDetailPage` | Real-time metrics + logs for a run |
| `/training/notebooks` | `NotebookListPage` | JSON training notebooks |
| `/training/notebooks/new` | `NotebookEditorPage` | Create a new notebook |
| `/training/notebooks/import` | `NotebookImportPage` | Import a notebook |
| `/training/notebooks/:id` | `NotebookEditorPage` | Edit / run a notebook |
| `/models` | `ModelsListPage` | Browse registered models |
| `/models/upload` | `ModelUploadPage` | Upload a new model version |
| `/models/:id` | `ModelDetailPage` | Model versions and metadata |
| `/models/:id/upload` | `ModelUploadPage` | Upload a new version for a specific model |
| `/datasets` | `DatasetsListPage` | User-uploaded datasets |
| `/datasets/upload` | `DatasetUploadPage` | Upload a dataset |
| `/datasets/analyze` | `DataAnalyzePage` | Analyze / preview / validate a dataset |
| `/datasets/kaggle` | `KagglePage` | Kaggle credentials, search, download |
| `/datasets/library` | `BuiltinDatasetsPage` | Built-in dataset catalog |
| `/inference` | `InferenceOverviewPage` | Endpoint status overview |
| `/inference/endpoints` | `EndpointsListPage` | All inference endpoints |
| `/inference/endpoints/:id` | `EndpointDetailPage` | Endpoint configuration and stats |
| `/inference/metrics` | `InferenceMetricsPage` | Aggregated inference performance |
| `/system` | `SystemOverviewPage` | Live system metrics dashboard |
| `/hub` | `HubBrowsePage` | Pretrained-model hub |
| `/hub/cache` | `HubCachePage` | Manage local hub cache |
| `/admin/users` | `UserManagementPage` | User management (admin) |
| `/admin/system` | `SystemStatsPage` | System statistics (admin) |
| `/settings` | `SettingsPage` | User settings overview |
| `/settings/profile` | `ProfileSettingsPage` | Edit profile |
| `/settings/security` | `SecuritySettingsPage` | Password and MFA settings |
| `/settings/security/totp` | `TotpSetupPage` | Configure authenticator app |
| `/settings/security/webauthn` | `WebAuthnSetupPage` | Register hardware security key |
| `/settings/security/recovery` | `RecoveryCodesPage` | View / regenerate recovery codes |
| `/*any` | `NotFound` | 404 fallback |

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
