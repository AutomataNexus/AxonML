# nexus-agent

Autonomous AI agent framework powered by local LLM inference via [nexus-serve](../nexus-serve/) (or the remote Anthropic API).

Eight specialized agents share a common ReAct execution loop and a 22-tool registry. Each agent has its own system prompt, default model, iteration budget, and temperature.

**Version:** 0.6.1 — updated 2026-04-16.

---

## Architecture

```
nexus-agent CLI (src/main.rs)
  |
  +--> ReAct loop (src/lib.rs)
  |      |
  |      +--> LlmBackend trait (src/backend/mod.rs)
  |      |      +--> AnthropicBackend (src/backend/anthropic.rs)
  |      |      |      POST /v1/messages (Anthropic Messages API shape;
  |      |      |      works against nexus-serve or the real Anthropic API)
  |      |      |      tool_use / tool_result content blocks,
  |      |      |      stop_reason = "tool_use"
  |      |      |
  |      |      +--> LocalBackend     (src/backend/local.rs)
  |      |             POST /v1/chat/completions (OpenAI shape)
  |      |             tool calls parsed from assistant text via regex
  |      |
  |      +--> ToolRegistry (22 tools across 8 modules, src/tools/)
  |
  +--> Desktop tickers (eframe 0.29 + glow / OpenGL, WSLg-compatible)
         +--> nexus-ticker   (CI monitor + ralph auto-fix)
         +--> tech-ticker    (field-technician fleet status)
         +--> ferrum-ticker  (Ferrum Mail stack probes)
```

The `Tool` trait, `ToolRegistry`, `Message` / `ToolCall` types, and `react_loop` driver all live in `src/lib.rs`. `FileMemory` (JSON-on-disk) implements the `Memory` trait there too.

## Agents

Defaults below are read directly from `src/agents/*.rs`.

| Agent | Model default | Temp | Max iter | Purpose |
|-------|--------------|------|----------|---------|
| `code` | `deepseek` | 0.1 | 12 | Agentic coder with shell / file / grep / git tools |
| `knowledge` | `qwen3` | 0.2 | 30 | Obsidian vault maintenance, codebase scanning, knowledge-task reads |
| `retrain` | `qwen3` | 0.2 | 25 | Model performance monitoring and retraining triggers |
| `fieldtech` | `gemma4` | 0.1 | 20 | HVAC fault detection, tech alerting via Tailscale |
| `research` | `qwen3` | 0.3 | 25 | Paper writing, lit review, citation management |
| `orchestrator` | `qwen3` | 0.1 | 15 | GPU scheduling, training-queue management |
| `ci-fixer` | `qwen3` | 0.1 | 25 | Fixes CI failures that `fmt` / `clippy` can't resolve |
| `shield` | `qwen3` | 0.1 | 15 | NexusShield drill-down: classify blocked/sql/ssrf/rate/threats/quarantine/audit events and emit proposed fixes (agent never applies them) |

## Tools (22)

Registered in `src/tools/mod.rs::register_all`:

| Module | Tools |
|--------|-------|
| `shell` | `shell` (run any command, output captured) |
| `file` | `read_file`, `write_file`, `search_files` (glob), `grep` (rg) |
| `git` | `git_status`, `git_log`, `git_diff`, `git_commit` |
| `obsidian` | `vault_read`, `vault_write`, `vault_search` |
| `email` | `send_email` (FerumMailSaaS API) |
| `tailscale` | `tailscale_status`, `tailscale_ping` |
| `training` | `start_training`, `check_training`, `list_checkpoints` |
| `github` | `gh_list_prs`, `gh_list_issues`, `gh_ci_status`, `gh_view_pr` |

Each tool exposes a JSON Schema via the `Tool` trait; the registry emits OpenAI / Anthropic-format tool definitions for the system prompt.

## Usage

```bash
# Build
cargo build --release

# Health check against nexus-serve
nexus-agent --url http://127.0.0.1:11436 health

# List loaded models
nexus-agent --url http://127.0.0.1:11436 models

# Run the code agent (Anthropic /v1/messages endpoint — recommended)
nexus-agent --url http://127.0.0.1:11436 --anthropic \
  code "Find all TODO comments in nexus-serve and list them"

# Override the default model
nexus-agent --url http://127.0.0.1:11436 --anthropic --model qwen3 \
  code "Refactor the error handling in weight.rs"

# Run the CI fixer (invoked by the nexus-ticker ralph loop)
nexus-agent ci-fixer "AxonML cargo test: assertion failed, final_loss > 0.01"
```

## Backends

### AnthropicBackend (`--anthropic`)

Talks to nexus-serve's `/v1/messages` endpoint (and works against the real Anthropic API unchanged). Tool calls round-trip as native `tool_use` / `tool_result` content blocks with `stop_reason = "tool_use"`. Supports reasoning-model `<think>` block stripping (R1-Distill, QwQ). This is the recommended backend.

### LocalBackend (default)

Talks to `/v1/chat/completions` (OpenAI format). Tool calls are extracted from the model's text output via regex. Kept for legacy agents written before the Anthropic endpoint existed.

## Desktop Tickers

Three always-on-top egui widgets that run via WSLg on Andrew's development laptop. Each renders as a frameless transparent pill with a custom titlebar, LED status indicators, and a scrolling event log. Built with `eframe 0.29` + `glow` (OpenGL) and `egui 0.29` for WSLg compatibility. Theme toggle (dark / light) persisted to `/tmp/.<name>-theme`.

| Binary | Source | Purpose |
|--------|--------|---------|
| `nexus-ticker` | `src/ui/ticker.rs` | CI status across 7 repos, ralph-loop auto-fix (fmt + clippy + `ci-fixer` agent fallback), uncommitted-changes warnings |
| `tech-ticker` | `src/ui/tech_ticker.rs` | NexusOracle daemon health, Tailscale online status for field technicians, rate limits, update-pending flags |
| `ferrum-ticker` | `src/ui/ferrum_ticker.rs` | Ferrum Mail stack probes (mailbox REST, public API, webmail, SMTP, Vultr relay) |

See `CHEATSHEET.md` and `TECH_TICKER_CHEATSHEET.md` in this crate for per-ticker operator notes.

## Configuration

- `--model` is optional. Each agent picks its own default (see the table above; source of truth is `src/agents/<name>.rs`). Pass `--model <alias>` only to override.
- `--url` defaults to `http://127.0.0.1:11435` (the standard nexus-serve port). DeepSeek-7B runs on `:11436` by convention in this workspace.
- `--anthropic` selects the Anthropic Messages API backend. Without it, the OpenAI-shape `LocalBackend` is used.

## Binaries

Four binaries are built from this crate (`Cargo.toml`):

| Bin | Path |
|-----|------|
| `nexus-agent` | `src/main.rs` |
| `nexus-ticker` | `src/ui/ticker.rs` |
| `tech-ticker` | `src/ui/tech_ticker.rs` |
| `ferrum-ticker` | `src/ui/ferrum_ticker.rs` |

## License

MIT / Apache-2.0
