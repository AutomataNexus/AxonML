# axonml-server

<!-- Logo placeholder -->
<p align="center">
  <img src="../../docs/assets/logo.svg" alt="AxonML Server" width="200"/>
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

**axonml-server** is the REST API backend for the AxonML Machine Learning Framework, built with Axum. It provides comprehensive endpoints for user authentication (JWT + MFA), training run management, model registry, inference serving, and system metrics. The server integrates with Aegis-DB for persistent storage.

---

## Features

- **Axum Framework** - High-performance async HTTP server with tower middleware
- **JWT Authentication** - Secure token-based authentication with refresh tokens
- **Multi-Factor Authentication** - TOTP (RFC 6238) and WebAuthn (FIDO2) support
- **Argon2 Password Hashing** - Industry-standard password security
- **Training Management** - Create, track, and stream training run metrics
- **Model Registry** - Version-controlled model storage with file upload/download
- **Inference Endpoints** - Deploy and manage model serving instances
- **WebSocket Streaming** - Real-time training metrics via WebSocket
- **CORS Support** - Configurable cross-origin resource sharing
- **Structured Logging** - Tracing-based logging with configurable levels
- **Aegis-DB Integration** - SQL queries and key-value storage backend

---

## Modules

| Module | Description |
|--------|-------------|
| `config` | TOML configuration loading with environment variable support |
| `db` | Aegis-DB connection, queries, and schema management |
| `db::schema` | Database schema initialization and migrations |
| `db::users` | User CRUD operations and authentication queries |
| `db::runs` | Training run persistence and metrics storage |
| `db::models` | Model registry database operations |
| `auth` | Authentication core with password hashing utilities |
| `auth::jwt` | JWT token generation, validation, and refresh logic |
| `auth::totp` | TOTP secret generation and code verification |
| `auth::webauthn` | WebAuthn registration and authentication ceremonies |
| `auth::recovery` | Recovery code generation and validation |
| `auth::middleware` | Axum middleware for route protection |
| `api` | API router creation and route definitions |
| `api::auth` | Authentication endpoints (login, register, MFA) |
| `api::training` | Training run endpoints and WebSocket streaming |
| `api::models` | Model registry endpoints with file handling |
| `api::inference` | Inference endpoint management and prediction API |
| `api::metrics` | System and performance metrics endpoints |
| `training` | Training run tracking and metrics collection |
| `training::tracker` | Run lifecycle management and status tracking |
| `training::websocket` | Real-time metrics WebSocket handler |
| `inference` | Inference serving infrastructure |
| `inference::server` | Model loading and prediction execution |
| `inference::pool` | Connection pooling for inference workers |
| `inference::metrics` | Inference latency and throughput metrics |

---

## Usage

### Prerequisites

- Rust 1.75+
- Running Aegis-DB instance
- Node.js + PM2 (for production deployment)

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
pm2 status                          # Check status
pm2 logs axonml-server              # View logs
pm2 restart axonml-server           # Restart server
pm2 stop axonml-server              # Stop server
pm2 reload axonml-server            # Zero-downtime reload
```

### Default Users

| User | Email | Password |
|------|-------|----------|
| Admin | admin@axonml.local | admin |
| DevOps | DevOps@automatanexus.com | Invertedskynet2$ |

### Configuration

Create `~/.axonml/config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 3000
data_dir = "~/.axonml"

[aegis]
host = "localhost"
port = 9090
username = "demo"
password = "demo"

[auth]
jwt_secret = "your-secure-random-secret-here"
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

### Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/auth/register` | Register new user | No |
| `POST` | `/api/auth/login` | Login and get tokens | No |
| `POST` | `/api/auth/logout` | Invalidate session | Yes |
| `POST` | `/api/auth/refresh` | Refresh access token | Yes |
| `GET` | `/api/auth/me` | Get current user info | Yes |
| `POST` | `/api/auth/mfa/totp/setup` | Initialize TOTP setup | Yes |
| `POST` | `/api/auth/mfa/totp/enable` | Enable TOTP with code | Yes |
| `POST` | `/api/auth/mfa/totp/verify` | Verify TOTP code | No |
| `POST` | `/api/auth/mfa/webauthn/register/start` | Start WebAuthn registration | Yes |
| `POST` | `/api/auth/mfa/webauthn/register/finish` | Complete WebAuthn registration | Yes |
| `POST` | `/api/auth/mfa/webauthn/authenticate/start` | Start WebAuthn auth | No |
| `POST` | `/api/auth/mfa/webauthn/authenticate/finish` | Complete WebAuthn auth | No |
| `GET` | `/api/auth/mfa/recovery/generate` | Generate recovery codes | Yes |
| `POST` | `/api/auth/mfa/recovery` | Use recovery code | No |

### Training Runs

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/api/training/runs` | List training runs | Yes |
| `POST` | `/api/training/runs` | Create new run | Yes |
| `GET` | `/api/training/runs/:id` | Get run details | Yes |
| `DELETE` | `/api/training/runs/:id` | Delete a run | Yes |
| `POST` | `/api/training/runs/:id/stop` | Stop a running run | Yes |
| `GET` | `/api/training/runs/:id/metrics` | Get run metrics | Yes |
| `POST` | `/api/training/runs/:id/metrics` | Record metrics | Yes |
| `GET` | `/api/training/runs/:id/logs` | Get run logs | Yes |
| `POST` | `/api/training/runs/:id/logs` | Append log entry | Yes |
| `GET` | `/api/training/runs/:id/stream` | WebSocket metrics stream | Yes |

### Model Registry

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/api/models` | List all models | Yes |
| `POST` | `/api/models` | Create new model | Yes |
| `GET` | `/api/models/:id` | Get model details | Yes |
| `PUT` | `/api/models/:id` | Update model metadata | Yes |
| `DELETE` | `/api/models/:id` | Delete model | Yes |
| `GET` | `/api/models/:id/versions` | List model versions | Yes |
| `POST` | `/api/models/:id/versions` | Upload new version | Yes |
| `GET` | `/api/models/:id/versions/:v` | Get version details | Yes |
| `DELETE` | `/api/models/:id/versions/:v` | Delete version | Yes |
| `GET` | `/api/models/:id/versions/:v/download` | Download model file | Yes |
| `POST` | `/api/models/:id/versions/:v/deploy` | Deploy version | Yes |

### Inference

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/api/inference/endpoints` | List endpoints | Yes |
| `POST` | `/api/inference/endpoints` | Create endpoint | Yes |
| `GET` | `/api/inference/endpoints/:id` | Get endpoint details | Yes |
| `PUT` | `/api/inference/endpoints/:id` | Update endpoint | Yes |
| `DELETE` | `/api/inference/endpoints/:id` | Delete endpoint | Yes |
| `POST` | `/api/inference/endpoints/:id/start` | Start endpoint | Yes |
| `POST` | `/api/inference/endpoints/:id/stop` | Stop endpoint | Yes |
| `GET` | `/api/inference/endpoints/:id/metrics` | Get endpoint metrics | Yes |
| `POST` | `/api/inference/predict/:name` | Run prediction | Yes |

### Admin & System

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/health` | Health check | No |
| `GET` | `/api/metrics` | System metrics | Yes |
| `GET` | `/api/admin/users` | List users | Admin |
| `POST` | `/api/admin/users` | Create user | Admin |
| `PUT` | `/api/admin/users/:id` | Update user | Admin |
| `DELETE` | `/api/admin/users/:id` | Delete user | Admin |
| `GET` | `/api/admin/stats` | System statistics | Admin |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           axonml-server                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       Axum Router                                │   │
│  │  ┌──────────┐  ┌───────────┐  ┌─────────┐  ┌───────────────┐   │   │
│  │  │  Public  │  │ Protected │  │  Admin  │  │   WebSocket   │   │   │
│  │  │  Routes  │  │  Routes   │  │ Routes  │  │    Routes     │   │   │
│  │  └──────────┘  └───────────┘  └─────────┘  └───────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
│              ┌─────────────┼─────────────┐                             │
│              ▼             ▼             ▼                              │
│  ┌───────────────┐  ┌───────────┐  ┌───────────────┐                   │
│  │  Auth Layer   │  │  CORS     │  │  Tracing      │                   │
│  │  (JWT/MFA)    │  │  Layer    │  │  Layer        │                   │
│  └───────────────┘  └───────────┘  └───────────────┘                   │
│              │                                                          │
│              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       AppState (Arc)                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │  Database   │  │   JwtAuth   │  │   Config    │             │   │
│  │  │    (db)     │  │   (jwt)     │  │  (config)   │             │   │
│  │  └──────┬──────┘  └─────────────┘  └─────────────┘             │   │
│  └─────────┼───────────────────────────────────────────────────────┘   │
│            │                                                            │
└────────────┼────────────────────────────────────────────────────────────┘
             │ HTTP
             ▼
   ┌───────────────────┐
   │    Aegis-DB       │
   │  (SQL + KV Store) │
   └───────────────────┘
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AXONML_HOST` | Server bind address | `0.0.0.0` |
| `AXONML_PORT` | Server port | `3000` |
| `AXONML_DATA_DIR` | Data directory path | `~/.axonml` |
| `AEGIS_HOST` | Aegis-DB host | `localhost` |
| `AEGIS_PORT` | Aegis-DB port | `9090` |
| `AEGIS_USER` | Aegis-DB username | `demo` |
| `AEGIS_PASS` | Aegis-DB password | `demo` |
| `JWT_SECRET` | JWT signing secret | (required in prod) |
| `RUST_LOG` | Log level filter | `axonml_server=info` |

---

## Tests

```bash
# Run all tests
cargo test -p axonml-server

# Run with output
cargo test -p axonml-server -- --nocapture

# Run specific test module
cargo test -p axonml-server auth::

# Run integration tests (requires Aegis-DB)
cargo test -p axonml-server --features integration
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
