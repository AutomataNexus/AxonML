# AxonML Dashboard Architecture

## Overview

The AxonML Dashboard is a full-stack web application providing:
- **Training Monitor**: TensorBoard-like real-time training visualization
- **Model Registry**: MLflow-like model versioning and deployment
- **Inference Server**: Model serving with metrics and monitoring
- **User Management**: Authentication with MFA support

## Technology Stack

### Backend (axonml-server)
- **Framework**: Axum 0.7 (async web framework)
- **Runtime**: Tokio (async runtime)
- **Database**: Aegis-DB (document store + time series + key-value)
- **Auth**: JWT + TOTP + WebAuthn
- **WebSocket**: tokio-tungstenite (real-time updates)

### Frontend (axonml-dashboard)
- **Framework**: Leptos 0.6 (Rust WASM framework)
- **Mode**: CSR (Client-Side Rendering)
- **Bundler**: Trunk
- **HTTP Client**: gloo-net
- **State**: Leptos signals (reactive)

### Database (Aegis-DB)
- **Document Store**: Users, Models, Endpoints (JSONB)
- **Time Series**: Training metrics, Inference metrics
- **Key-Value**: Sessions, Cache

## Directory Structure

```
crates/
├── axonml-server/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 # Entry point, Axum server
│       ├── config.rs               # Configuration (TOML loading)
│       ├── db/
│       │   ├── mod.rs              # Database module
│       │   ├── schema.rs           # Table creation
│       │   ├── users.rs            # User CRUD
│       │   ├── runs.rs             # Training runs CRUD
│       │   └── models.rs           # Model registry CRUD
│       ├── auth/
│       │   ├── mod.rs              # Auth module
│       │   ├── jwt.rs              # JWT tokens
│       │   ├── totp.rs             # TOTP MFA
│       │   ├── webauthn.rs         # WebAuthn MFA
│       │   ├── recovery.rs         # Recovery codes
│       │   └── middleware.rs       # Auth middleware
│       ├── api/
│       │   ├── mod.rs              # API routes
│       │   ├── auth.rs             # /api/auth/*
│       │   ├── training.rs         # /api/training/*
│       │   ├── models.rs           # /api/models/*
│       │   ├── inference.rs        # /api/inference/*
│       │   └── metrics.rs          # /api/metrics/*
│       ├── training/
│       │   ├── mod.rs              # Training module
│       │   ├── tracker.rs          # Metrics collection
│       │   └── websocket.rs        # Real-time streaming
│       └── inference/
│           ├── mod.rs              # Inference module
│           ├── server.rs           # Model serving
│           ├── pool.rs             # Model pool
│           └── metrics.rs          # Latency tracking
│
└── axonml-dashboard/
    ├── Cargo.toml
    ├── Trunk.toml
    ├── index.html
    ├── assets/
    │   ├── styles.css              # NexusForge theme
    │   └── logo.svg
    └── src/
        ├── lib.rs                  # App entry, router
        ├── api.rs                  # HTTP client
        ├── state.rs                # Global state
        ├── types.rs                # Data types
        ├── components/
        │   ├── mod.rs
        │   ├── navbar.rs           # Top navigation
        │   ├── sidebar.rs          # Left sidebar
        │   ├── charts.rs           # SVG charts
        │   ├── progress.rs         # Progress bars
        │   ├── table.rs            # Data tables
        │   ├── modal.rs            # Modal dialogs
        │   ├── toast.rs            # Notifications
        │   ├── spinner.rs          # Loading spinners
        │   └── forms.rs            # Form inputs
        ├── auth/
        │   ├── mod.rs
        │   ├── login.rs            # Login page
        │   ├── mfa.rs              # MFA verification
        │   ├── mfa_setup.rs        # MFA setup
        │   └── session.rs          # Session management
        └── pages/
            ├── mod.rs
            ├── landing.rs          # Public landing
            ├── dashboard.rs        # Main dashboard
            ├── training/
            │   ├── mod.rs
            │   ├── list.rs         # Training runs list
            │   ├── detail.rs       # Run detail + charts
            │   └── new.rs          # New training form
            ├── models/
            │   ├── mod.rs
            │   ├── list.rs         # Model registry
            │   ├── detail.rs       # Model detail
            │   ├── upload.rs       # Upload model
            │   └── compare.rs      # Compare versions
            ├── inference/
            │   ├── mod.rs
            │   ├── overview.rs     # Inference overview
            │   ├── endpoints.rs    # Endpoint management
            │   └── metrics.rs      # Metrics charts
            └── settings/
                ├── mod.rs
                ├── profile.rs      # User profile
                ├── security.rs     # Security settings
                └── admin.rs        # User management
```

## Color Palette (NexusForge Theme)

```css
:root {
  /* Primary backgrounds */
  --background: #faf9f6;        /* Cream white */
  --card-bg: #f5ebe0;           /* Warm cream */
  --highlight: #fff7ed;         /* Light orange tint */
  --teal-bg: #f0fdfa;           /* Light teal */
  --slate-bg: #f3f4f6;          /* Light gray */

  /* Primary accents */
  --terracotta: #c4a484;        /* Warm brown */
  --terracotta-dark: #a08060;   /* Dark brown */
  --teal: #14b8a6;              /* Primary teal */
  --teal-dark: #0d9488;         /* Dark teal */

  /* Text colors */
  --text-primary: #111827;      /* Near black */
  --text-secondary: #6b7280;    /* Gray */
  --text-muted: #9ca3af;        /* Light gray */
  --text-light: #fffdf7;        /* Off white */

  /* Status colors */
  --success: #10b981;           /* Green */
  --warning: #f59e0b;           /* Amber */
  --error: #ef4444;             /* Red */
  --info: #3b82f6;              /* Blue */
}
```

## Database Schema (Aegis-DB)

### Document Collections

```sql
-- Users
CREATE TABLE axonml_users (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- data: {id, email, name, password_hash, role, mfa_enabled, totp_secret, webauthn_credentials, recovery_codes}

-- Training Runs
CREATE TABLE axonml_runs (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- data: {id, user_id, name, model_type, status, config, started_at, completed_at}

-- Models
CREATE TABLE axonml_models (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- data: {id, user_id, name, description, model_type}

-- Model Versions
CREATE TABLE axonml_model_versions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- data: {id, model_id, version, file_path, file_size, metrics, training_run_id}

-- Inference Endpoints
CREATE TABLE axonml_endpoints (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- data: {id, model_version_id, name, status, port, replicas, config}
```

### Time Series Tables

```sql
-- Training Metrics
CREATE TABLE axonml_metrics (
    run_id TEXT NOT NULL,
    epoch INT NOT NULL,
    step INT NOT NULL,
    loss FLOAT,
    accuracy FLOAT,
    lr FLOAT,
    gpu_util FLOAT,
    memory_mb FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, timestamp)
);

-- Inference Metrics
CREATE TABLE axonml_inference_metrics (
    endpoint_id TEXT NOT NULL,
    requests_total INT,
    requests_success INT,
    requests_error INT,
    latency_p50 FLOAT,
    latency_p95 FLOAT,
    latency_p99 FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (endpoint_id, timestamp)
);
```

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Create account |
| POST | /api/auth/login | Login, returns JWT |
| POST | /api/auth/logout | Invalidate session |
| POST | /api/auth/refresh | Refresh JWT token |
| GET | /api/auth/me | Get current user |
| POST | /api/auth/mfa/totp/setup | Get TOTP secret + QR |
| POST | /api/auth/mfa/totp/verify | Verify TOTP code |
| POST | /api/auth/mfa/totp/enable | Enable TOTP MFA |
| POST | /api/auth/mfa/webauthn/register/start | Start WebAuthn registration |
| POST | /api/auth/mfa/webauthn/register/finish | Finish WebAuthn registration |
| POST | /api/auth/mfa/webauthn/authenticate/start | Start WebAuthn auth |
| POST | /api/auth/mfa/webauthn/authenticate/finish | Finish WebAuthn auth |
| GET | /api/auth/mfa/recovery | Generate recovery codes |
| POST | /api/auth/mfa/recovery | Use recovery code |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/training/runs | List runs |
| POST | /api/training/runs | Create run |
| GET | /api/training/runs/:id | Get run details |
| DELETE | /api/training/runs/:id | Delete run |
| POST | /api/training/runs/:id/stop | Stop running |
| GET | /api/training/runs/:id/metrics | Get metrics history |
| WS | /api/training/runs/:id/stream | Real-time metrics |
| POST | /api/training/runs/:id/log | Append log entry |
| GET | /api/training/runs/:id/logs | Get logs |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/models | List models |
| POST | /api/models | Create model |
| GET | /api/models/:id | Get model |
| PUT | /api/models/:id | Update model |
| DELETE | /api/models/:id | Delete model |
| GET | /api/models/:id/versions | List versions |
| POST | /api/models/:id/versions | Upload version |
| GET | /api/models/:id/versions/:v | Get version |
| DELETE | /api/models/:id/versions/:v | Delete version |
| GET | /api/models/:id/versions/:v/download | Download model |
| POST | /api/models/:id/versions/:v/deploy | Deploy to inference |

### Inference
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/inference/endpoints | List endpoints |
| POST | /api/inference/endpoints | Create endpoint |
| GET | /api/inference/endpoints/:id | Get endpoint |
| PUT | /api/inference/endpoints/:id | Update endpoint |
| DELETE | /api/inference/endpoints/:id | Delete endpoint |
| POST | /api/inference/endpoints/:id/start | Start endpoint |
| POST | /api/inference/endpoints/:id/stop | Stop endpoint |
| GET | /api/inference/endpoints/:id/metrics | Get metrics |
| POST | /api/inference/predict/:endpoint_name | Run inference |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/admin/users | List users |
| POST | /api/admin/users | Create user |
| PUT | /api/admin/users/:id | Update user |
| DELETE | /api/admin/users/:id | Delete user |
| GET | /api/admin/stats | System stats |

## Frontend Routes

| Route | Page | Component |
|-------|------|-----------|
| / | Landing | pages/landing.rs |
| /login | Login | auth/login.rs |
| /mfa | MFA Verify | auth/mfa.rs |
| /dashboard | Overview | pages/dashboard.rs |
| /training | Training List | pages/training/list.rs |
| /training/:id | Training Detail | pages/training/detail.rs |
| /training/new | New Training | pages/training/new.rs |
| /models | Model Registry | pages/models/list.rs |
| /models/:id | Model Detail | pages/models/detail.rs |
| /models/upload | Upload Model | pages/models/upload.rs |
| /models/compare | Compare Versions | pages/models/compare.rs |
| /inference | Inference Overview | pages/inference/overview.rs |
| /inference/endpoints | Endpoints | pages/inference/endpoints.rs |
| /inference/metrics | Metrics | pages/inference/metrics.rs |
| /settings/profile | Profile | pages/settings/profile.rs |
| /settings/security | Security | pages/settings/security.rs |
| /settings/admin | Admin | pages/settings/admin.rs |

## CLI Commands

```bash
# Start both frontend and backend
axon start

# Start only backend (API on port 3000)
axon start --backend

# Start only frontend (UI on port 8080)
axon start --frontend

# Stop all services
axon stop

# Check service status
axon status

# View logs
axon logs
axon logs --backend
axon logs --frontend
```

## Data Flow

### Training Metrics Flow
```
Training Script
    │
    ├─── POST /api/training/runs (create run)
    │
    └─── POST /api/training/runs/:id/metrics (batch metrics)
              │
              ▼
         Aegis-DB (axonml_metrics table)
              │
              ▼
         WebSocket broadcast
              │
              ▼
         Dashboard (real-time charts)
```

### Model Deployment Flow
```
User uploads model
    │
    ├─── POST /api/models/:id/versions (multipart upload)
    │         │
    │         ▼
    │    Save to ~/.axonml/models/
    │         │
    │         ▼
    │    Store metadata in axonml_model_versions
    │
    └─── POST /api/models/:id/versions/:v/deploy
              │
              ▼
         Create inference endpoint
              │
              ▼
         Load model into memory
              │
              ▼
         POST /api/inference/predict/:name available
```

## Authentication Flow

### Login with MFA
```
1. POST /api/auth/login
   ├─── Valid credentials → 200 {requires_mfa: true, mfa_token: "..."}
   └─── Invalid → 401

2. POST /api/auth/mfa/totp/verify {mfa_token, code}
   ├─── Valid → 200 {token: "jwt...", user: {...}}
   └─── Invalid → 401

3. All subsequent requests include:
   Authorization: Bearer <jwt>
```

### MFA Setup Flow
```
1. GET /api/auth/mfa/totp/setup
   → 200 {secret: "...", qr_code: "data:image/png;base64,..."}

2. User scans QR code in authenticator app

3. POST /api/auth/mfa/totp/verify {code}
   → 200 {valid: true}

4. POST /api/auth/mfa/totp/enable
   → 200 {recovery_codes: ["...", "..."]}
```

## WebSocket Protocol

### Training Metrics Stream
```
Connect: WS /api/training/runs/:id/stream

Server sends (every ~1s during training):
{
  "type": "metrics",
  "data": {
    "epoch": 5,
    "step": 1250,
    "loss": 0.234,
    "accuracy": 0.891,
    "lr": 0.001,
    "gpu_util": 0.85,
    "memory_mb": 4096
  }
}

Server sends on completion:
{
  "type": "status",
  "data": {
    "status": "completed",
    "completed_at": "2024-01-15T10:30:00Z"
  }
}
```

## File Storage

```
~/.axonml/
├── config.toml          # Server configuration
├── models/              # Stored model files
│   └── {model_id}/
│       └── v{version}/
│           └── model.safetensors
├── runs/                # Training run logs
│   └── {run_id}/
│       └── logs.txt
├── logs/                # Server logs
│   ├── server.log
│   └── access.log
└── dashboard/           # Built frontend assets
    ├── index.html
    └── *.wasm
```

## Security Considerations

1. **Passwords**: Argon2id hashing
2. **Sessions**: JWT with short expiry + refresh tokens
3. **MFA**: TOTP (RFC 6238) + WebAuthn (FIDO2)
4. **CORS**: Configured for dashboard origin only
5. **Rate Limiting**: Per-IP and per-user limits
6. **Input Validation**: All inputs sanitized
7. **File Upload**: Size limits, type checking

---

## Implementation Status

**Last Updated**: 2026-01-20
**Overall Progress**: 90%

### Backend (axonml-server) - COMPLETE

| Module | Status | Files |
|--------|--------|-------|
| Core Setup | DONE | main.rs, config.rs |
| Database | DONE | db/mod.rs, schema.rs, users.rs, runs.rs, models.rs |
| Auth | DONE | auth/mod.rs, jwt.rs, totp.rs, webauthn.rs, recovery.rs, middleware.rs |
| API Routes | DONE | api/mod.rs, auth.rs, training.rs, models.rs, inference.rs, metrics.rs |
| Training | DONE | training/mod.rs, tracker.rs, websocket.rs |
| Inference | DONE | inference/mod.rs, server.rs, pool.rs, metrics.rs |

### Frontend (axonml-dashboard) - COMPLETE

| Module | Status | Files |
|--------|--------|-------|
| Core Setup | DONE | Cargo.toml, Trunk.toml, index.html, lib.rs |
| Assets | DONE | styles.css (NexusForge theme), logo.svg |
| Types/API | DONE | types.rs, api.rs, state.rs |
| Components | DONE | icons.rs, navbar.rs, sidebar.rs, charts.rs, progress.rs, table.rs, modal.rs, toast.rs, spinner.rs, forms.rs |
| Auth Pages | DONE | login.rs, mfa.rs, mfa_setup.rs, session.rs |
| Dashboard | DONE | landing.rs, dashboard.rs (AppShell) |
| Training | DONE | list.rs, detail.rs (WebSocket), new.rs |
| Models | DONE | list.rs, detail.rs, upload.rs |
| Inference | DONE | overview.rs, endpoints.rs, metrics.rs |
| Settings | DONE | mod.rs, profile.rs, security.rs |

### CLI Commands - IN PROGRESS

| Command | Status |
|---------|--------|
| `axon start` | TODO |
| `axon stop` | TODO |
| `axon status` | TODO |
| `axon logs` | TODO |

### Installation - PENDING

| Task | Status |
|------|--------|
| install.sh script | TODO |
| Aegis-DB auto-install | TODO |
| Schema initialization | TODO |

### Remaining Tasks

1. Add CLI serve commands to axonml-cli
2. Create install.sh script
3. Verify full build compiles
4. End-to-end integration testing
