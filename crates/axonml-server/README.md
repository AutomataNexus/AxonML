# axonml-server

<!-- Logo placeholder -->
<p align="center">
  <img src="../../docs/assets/logo.svg" alt="AxonML Server" width="200"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/Rust-1.85+-orange.svg" alt="Rust 1.85+"/>
  <img src="https://img.shields.io/badge/version-0.6.5-green.svg" alt="Version 0.6.5"/>
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML"/>
</p>

---

## Overview

**axonml-server** is the REST API + WebSocket backend for the AxonML Machine Learning Framework, built with Axum 0.7 on Tokio. It provides endpoints for user authentication (JWT, TOTP, WebAuthn, recovery codes), training run management with real-time metric streaming, a versioned model registry, dataset management, inference serving with a pooled model cache, a JSON-notebook execution engine, Kaggle/Hub dataset integration, an Ollama-backed LLM assist endpoint, a PTY-based browser terminal, and comprehensive system metrics. It uses a document-store database backend for persistent storage (SQL + KV) and an optional HashiCorp Vault backend for secrets.

Last updated: 2026-06-06 — version 0.6.5.

---

## Features

- **Axum 0.7** — Async HTTP + WebSocket server on Tokio, with tower-http layers for CORS, tracing, compression, and static file serving.
- **Secrets Manager** — Pluggable `SecretsBackend` trait with Vault and environment-variable backends. Resolves JWT secret, database backend credentials, and Resend API key in priority order (Vault -> env -> config-file fallback). JWT secret is validated to be >=32 characters at startup.
- **JWT Authentication** — `jsonwebtoken` 10 with access + refresh tokens. JWT secret hot-loaded from Vault or env at boot.
- **Multi-Factor Authentication** — TOTP (RFC 6238 via `totp-rs`), WebAuthn / FIDO2, and one-time recovery codes.
- **Argon2id Password Hashing** — `argon2` crate with random per-password salts via `OsRng`.
- **Rate Limiting** — Sliding-window per-IP `RateLimiter` applied to auth endpoints (login / register / MFA).
- **Secure Default Admin** — On first boot, a 24-character cryptographically random password is generated for `admin@axonml.local` and written to `{tmp}/axonml-admin-password.txt`; there is no static default password.
- **DevOps Admin** — Optional `DevOps@AutomataNexus.com` user provisioned from the `AXONML_DEVOPS_PASSWORD` environment variable on boot.
- **Training Management** — Create, list, stop, complete, and delete runs; record metrics and logs; stream metrics live over WebSocket.
- **Training Executor** — Spawns and tracks training processes with a persistent `TrainingTracker` wired to the database backend.
- **Notebook Engine** — JSON training notebooks with cell add/update/delete/execute, AI-assist (Ollama), checkpoint save/list/best, and model-version export.
- **Model Registry** — Versioned model storage with multipart upload, download, inspect, convert, quantize, export, and deploy.
- **Datasets** — CRUD for user-uploaded datasets plus built-in dataset catalog (`builtin-datasets` list/search/sources/info/prepare).
- **Kaggle Integration** — Server-side Kaggle credentials, dataset search, download, and listing of downloaded datasets.
- **Pretrained Hub** — List, info, download, and cache management for pretrained-weight models.
- **Inference Serving** — `InferenceServer` + `ModelPool` (capacity 100, 5-minute idle timeout) + `InferenceMetrics` with per-endpoint latency histograms, p50/p95/p99, RPS, error rate.
- **WebSocket Streaming** — Real-time training-metric stream and PTY-backed browser terminal.
- **Ollama LLM Integration** — Client auto-probes the default Ollama URL and exposes AI-assist inside notebooks.
- **System Metrics** — `sysinfo`-based background collector maintains a 60-second rolling history of CPU and memory usage (disk/network/GPU slots reserved).
- **Audited Admin Query** — `/api/admin/query` is whitelisted to `SELECT / SHOW / DESCRIBE / COUNT` with a blocklist of destructive tokens. `/api/admin/execute` is intentionally disabled.
- **Structured Logging** — `tracing` + `tracing-subscriber` with JSON and env-filter support.

---

## Modules

| Module | Description |
|--------|-------------|
| `config` | `Config` struct with TOML loading, env overrides, validation, warnings, and directory helpers |
| `secrets` | `SecretsManager`, `SecretsBackend` trait, `SecretKey` constants, `SecretsError` |
| `secrets::vault` | HashiCorp Vault backend (`vaultrs`) with background token renewal |
| `secrets::env` | Environment-variable backend |
| `db` | Database backend `Database` wrapper (SQL + KV) plus health check |
| `db::schema` | Schema init + default admin / DevOps admin provisioning |
| `db::users` | User CRUD + auth queries |
| `db::runs` | Training-run persistence + metrics storage |
| `db::models` | Model registry persistence |
| `db::datasets` | Dataset catalog persistence |
| `db::notebooks` | Notebook persistence |
| `auth` | Module root; Argon2 `hash_password` / `verify_password`, `AuthError` |
| `auth::jwt` | `Claims`, `JwtAuth` (access + refresh) |
| `auth::totp` | `TotpAuth` (RFC 6238) |
| `auth::webauthn` | `WebAuthnAuth` registration + authentication ceremonies |
| `auth::recovery` | `RecoveryAuth` single-use recovery codes |
| `auth::middleware` | `AuthLayer`, `AuthUser`, `auth_middleware`, `optional_auth_middleware`, `require_admin_middleware`, `require_mfa_middleware` |
| `auth::rate_limit` | `RateLimiter` (sliding-window per IP) + `rate_limit_middleware` |
| `api` | `AppState`, `create_router`, top-level handlers (health, status, pool, cache, secure-info, admin query/execute/record-metrics) |
| `api::auth` | Login, register, verify-email, approve-user, logout, refresh, me, MFA endpoints, user admin |
| `api::training` | Training-run endpoints + metrics-stream WebSocket |
| `api::models` | Model registry endpoints + version upload/download/deploy |
| `api::datasets` | User dataset endpoints |
| `api::data` | Dataset analysis, preview, validation, generate-config |
| `api::builtin_datasets` | Built-in dataset catalog |
| `api::inference` | Endpoint CRUD, start/stop, metrics, predict |
| `api::hub` | Pretrained-model hub endpoints |
| `api::kaggle` | Server-side Kaggle credentials and dataset ops |
| `api::notebooks` | JSON notebook CRUD, cell execution, AI-assist, checkpoints |
| `api::metrics` | Aggregated metrics endpoint |
| `api::system` | `/api/system/*` — info, GPUs, benchmark, real-time metrics, history, correlation; `SystemMetricsHistory` struct |
| `api::terminal` | PTY-based WebSocket terminal (`portable-pty`) + info endpoint |
| `api::tools` | Inspect / convert / quantize / export per model version + list-quantization-types |
| `training::tracker` | `TrainingTracker` broadcasts metrics to subscribers |
| `training::executor` | `TrainingExecutor` spawns training processes |
| `training::notebook_executor` | `NotebookExecutor` runs notebook cells |
| `training::websocket` | WebSocket handler for metrics streaming |
| `inference::server` | `InferenceServer` (model loading + prediction) |
| `inference::pool` | `ModelPool` (connection/model pooling with idle cleanup) |
| `inference::metrics` | `InferenceMetrics` with latency histograms, p50/p95/p99, RPS |
| `llm::ollama` | `OllamaClient` for LLM assist; `DEFAULT_OLLAMA_URL` |
| `email` | `EmailService` (Resend-backed, optional) |

---

## Usage

### Prerequisites

- Rust 1.85+ (workspace edition)
- Running database backend instance (default: `localhost:9090`)
- Optional: HashiCorp Vault for production secrets
- Optional: Ollama server for notebook AI assist

### Development

```bash
# Start with defaults (0.0.0.0:3000)
cargo run -p axonml-server

# Or using the binary
axonml-server

# Custom host and port
axonml-server --host 127.0.0.1 --port 8000

# Development port (matches dashboard proxy)
cargo run -p axonml-server -- --port 3021

# With custom config file
axonml-server --config /path/to/config.toml
```

CLI flags are defined in `src/main.rs`:

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--host` | `-H` | `0.0.0.0` | Host to bind to |
| `--port` | `-p` | `3000` | Port to listen on |
| `--config` | `-c` | (none) | Path to TOML config file |

### Production Deployment (PM2)

```bash
# 1. Build release binary
cargo build --release -p axonml-server

# 2. Initialize database
./AxonML_DB_Init.sh --with-user    # Creates collections + DevOps user

# 3. Create log directory
sudo mkdir -p /var/log/axonml
sudo chown $USER:$USER /var/log/axonml

# 4. Start with PM2
pm2 start ecosystem.config.js
pm2 save                            # Save process list
pm2 startup                         # Enable boot persistence

# Management commands
pm2 status
pm2 logs axonml-server
pm2 restart axonml-server
pm2 stop axonml-server
pm2 reload axonml-server            # Zero-downtime reload
```

### Default Users

| User | Email | Password |
|------|-------|----------|
| Admin | `admin@axonml.local` | Cryptographic random (24 chars) generated on first boot and written to `{tmp}/axonml-admin-password.txt`. Read it, then delete the file. |
| DevOps | `DevOps@AutomataNexus.com` | From `AXONML_DEVOPS_PASSWORD` environment variable (provisioned on every start if set) |

There is no static default password in source.

### Configuration

Create `~/.axonml/config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 3000
data_dir = "~/.axonml"

[database]
host = "localhost"
port = 9090          # Must match the database backend's --port (default: 9090)
username = ""        # Prefer Vault or BACKEND_USER env var
password = ""        # Prefer Vault or BACKEND_PASS env var

[auth]
jwt_secret = ""      # Prefer Vault (jwt_secret) or AXONML_JWT_SECRET env var; MUST be >=32 chars
jwt_expiry_hours = 24
session_timeout_minutes = 30
require_mfa = false

[inference]
default_port_range_start = 8100
default_port_range_end = 8199
max_endpoints = 10

[dashboard]
port = 8080
```

---

## API Endpoints

Assembled in `src/api/mod.rs::create_router`. Grouped as `public_routes` / `protected_routes` / `admin_routes` / `mfa_protected_routes` / `optional_auth_routes` / `ws_routes` / `tower_auth_routes`.

### Public

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (DB + inference + pool) |
| `GET` | `/api/status/inference` | Inference server status |
| `GET` | `/api/status/cache` | KV cache status |
| `GET` | `/api/status/pool` | Model pool status (runs `cleanup_idle`) |
| `POST` | `/api/auth/register` | Register |
| `POST` | `/api/auth/login` | Login |
| `GET` | `/api/auth/verify-email` | Verify email link |
| `GET` | `/api/auth/approve-user` | Admin-approval link |
| `POST` | `/api/auth/mfa/totp/verify` | Verify TOTP code |
| `POST` | `/api/auth/mfa/webauthn/authenticate/{start,finish}` | WebAuthn login |
| `POST` | `/api/auth/mfa/recovery` | Use recovery code |

### Protected (JWT required)

Authentication and session:

- `POST /api/auth/logout`, `POST /api/auth/refresh`, `GET /api/auth/me`
- `POST /api/auth/mfa/totp/{setup,enable}`
- `POST /api/auth/mfa/webauthn/register/{start,finish}`
- `GET /api/auth/mfa/recovery/generate`, `POST /api/auth/mfa/disable`

Training runs:

- `GET|POST /api/training/runs`, `GET|DELETE /api/training/runs/:id`
- `POST /api/training/runs/:id/stop`, `POST /api/training/runs/:id/complete`
- `GET|POST /api/training/runs/:id/metrics`
- `GET|POST /api/training/runs/:id/logs`

Model registry:

- `GET|POST /api/models`, `GET|PUT|DELETE /api/models/:id`
- `GET|POST /api/models/:id/versions`
- `GET|DELETE /api/models/:id/versions/:version`
- `GET /api/models/:id/versions/:version/download`
- `POST /api/models/:id/versions/:version/deploy`

Datasets:

- `GET|POST /api/datasets`, `GET|DELETE /api/datasets/:id`
- `POST /api/data/:id/{analyze,preview,validate,generate-config}`

Inference endpoints:

- `GET|POST /api/inference/endpoints`
- `GET|PUT|DELETE /api/inference/endpoints/:id`
- `POST /api/inference/endpoints/:id/{start,stop}`
- `GET /api/inference/endpoints/:id/{metrics,info}`
- `POST /api/inference/predict/:name`

Metrics / system:

- `GET /api/metrics`
- `GET /api/system/{info,gpus,metrics,metrics/history,correlation}`
- `POST /api/system/benchmark`

Hub (pretrained):

- `GET /api/hub/models`, `GET /api/hub/models/:name`
- `POST /api/hub/models/:name/download`
- `GET|DELETE /api/hub/cache`, `DELETE /api/hub/cache/:name`

Model tools:

- `GET /api/models/:model_id/versions/:version_id/inspect`
- `POST /api/models/:model_id/versions/:version_id/{convert,quantize,export}`
- `GET /api/tools/quantization-types`

Kaggle:

- `POST|DELETE /api/kaggle/credentials`
- `GET /api/kaggle/status`
- `GET /api/kaggle/search`, `POST /api/kaggle/download`, `GET /api/kaggle/downloaded`

Built-in datasets:

- `GET /api/builtin-datasets`
- `GET /api/builtin-datasets/{search,sources}`
- `GET /api/builtin-datasets/:id`, `POST /api/builtin-datasets/:id/prepare`

Notebooks:

- `GET|POST /api/notebooks`, `POST /api/notebooks/import`
- `GET|PUT|DELETE /api/notebooks/:id`, `GET /api/notebooks/:id/export`
- `POST /api/notebooks/:id/{start,stop}`
- `POST /api/notebooks/:id/cells`, `PUT|DELETE /api/notebooks/:id/cells/:cell_id`
- `POST /api/notebooks/:id/cells/:cell_id/execute`
- `POST /api/notebooks/:id/ai-assist`
- `GET|POST /api/notebooks/:id/checkpoints`, `GET /api/notebooks/:id/checkpoints/best`
- `POST /api/notebooks/:id/upload-version`

### Admin (admin role required)

- `GET|POST /api/admin/users`, `GET|PUT|DELETE /api/admin/users/:id`
- `GET /api/admin/stats`
- `POST /api/admin/query` — read-only; whitelisted to `SELECT / SHOW / DESCRIBE / COUNT`; blocks `;`, SQL comments, `DROP`/`DELETE`/`TRUNCATE`/`ALTER`/`CREATE`/`INSERT`/`UPDATE`/`GRANT`/`REVOKE`/`EXEC`/`EXECUTE`/`xp_`/`sp_`
- `POST /api/admin/execute` — **disabled** (returns 403); use specific API endpoints for writes
- `POST /api/admin/metrics/record` — inject inference latency samples

### MFA-protected

- `DELETE /api/inference/endpoints/:id/delete-secure` — requires MFA if user has it enabled

### WebSocket

- `GET /api/training/runs/:id/stream` — live metrics stream
- `GET /api/terminal` — PTY-backed browser terminal (upgrade)
- `GET /api/terminal/info` — capability/info

### Tower-layered

- `GET /api/secure/info` — `AuthLayer`-protected aggregated system info

### Optional-auth

- `GET /api/public/models` — works signed in or anonymous

---

## Architecture

```
+-------------------------------------------------------------------------+
|                           axonml-server                                 |
+-------------------------------------------------------------------------+
|                                                                         |
|  +-----------------------------------------------------------------+   |
|  |                       Axum Router                                |   |
|  |  +----------+  +-----------+  +---------+  +---------------+   |   |
|  |  |  Public  |  | Protected |  |  Admin  |  |  WebSocket    |   |   |
|  |  |  Routes  |  |  Routes   |  | Routes  |  |   Routes      |   |   |
|  |  +----------+  +-----------+  +---------+  +---------------+   |   |
|  +-----------------------------------------------------------------+   |
|                            |                                            |
|              +-------------+-------------+                              |
|              v             v             v                              |
|  +---------------+  +-----------+  +---------------+                   |
|  |  Auth Layer   |  |  CORS     |  |  Tracing      |                   |
|  |  (JWT/MFA/    |  |  Layer    |  |  Layer        |                   |
|  |   RateLimit)  |  |           |  |               |                   |
|  +---------------+  +-----------+  +---------------+                   |
|              |                                                          |
|              v                                                          |
|  +-----------------------------------------------------------------+   |
|  |                       AppState (Arc, Clone)                      |   |
|  |  Database, JwtAuth, Config, EmailService, InferenceServer,       |   |
|  |  TrainingTracker, TrainingExecutor, NotebookExecutor,            |   |
|  |  ModelPool, InferenceMetrics, SystemMetricsHistory(Mutex),       |   |
|  |  OllamaClient, RateLimiter                                       |   |
|  +-----------------------------------------------------------------+   |
|            |                                                            |
+------------+------------------------------------------------------------+
             | HTTP
             v
   +---------------------+           +-------------------+
   |  Database backend   |           |  HashiCorp Vault  |
   |  (SQL + KV Store)   |           |   (optional)      |
   +---------------------+           +-------------------+
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AXONML_HOST` | Server bind address | `0.0.0.0` |
| `AXONML_PORT` | Server port | `3000` |
| `AXONML_DATA_DIR` | Data directory path | `~/.axonml` |
| `AXONML_JWT_SECRET` | JWT signing secret (>=32 chars) | (required unless set in Vault) |
| `AXONML_DEVOPS_PASSWORD` | Password for the DevOps admin seed account | (none — DevOps admin not created) |
| `AXONML_RESEND_API_KEY` / `RESEND_API_KEY` | Resend email API key | (optional) |
| `BACKEND_HOST` | Database backend host | `localhost` |
| `BACKEND_PORT` | Database backend port | `9090` |
| `BACKEND_USER` | Database backend username | (config / Vault) |
| `BACKEND_PASS` | Database backend password | (config / Vault) |
| `VAULT_ADDR` | HashiCorp Vault address; if set, Vault backend is enabled | (unset = disabled) |
| `RUST_LOG` | Log level filter | `axonml_server=info,tower_http=info` |

---

## Tests

```bash
# Run all tests
cargo test -p axonml-server

# Run with output
cargo test -p axonml-server -- --nocapture

# Run a specific module's tests
cargo test -p axonml-server auth::

# Integration tests (require a running database backend)
cargo test -p axonml-server --test '*'
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
