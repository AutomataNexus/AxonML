# tech-ticker — Cheatsheet

Compact 260×380 always-on-top widget showing the five field techs' live stats.
Sibling of `nexus-ticker` — same crate, different binary.

---

## Layout

```
┌───────────────────────────────────┐
│ ● TECH MONITOR           14:42    │  ← header (sync time | ⚠ sync on error)
├───────────────────────────────────┤
│ ● Nick  ●                   1/3   │  ← LED · name · [update-LED] · usage
│   last used 2h ago       4h window│
│   ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  ← usage bar
├───────────────────────────────────┤
│ ● Leon                      0/3   │
│   never used             4h window│
│   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│ ... one row per tech              │
├───────────────────────────────────┤
│ ● online  ● offline  ● update     │  ← legend
└───────────────────────────────────┘
```

Window width is fixed; height resizes. Launch: `/opt/AxonML/nexus-agent/target/release/tech-ticker &`.

---

## Per-tech LED colors

| LED | Meaning |
|---|---|
| **Teal** (pulsing) | Tech is online on Tailscale right now |
| **Grey** | Tech is offline or never connected |
| **Amber** (pulsing, next to name) | A newer `oracle-chat-tauri.exe` exists on the build machine than what has been pushed to this tech — `tailscale-monitor.sh` will push it next time they come online |

---

## Usage counter `N/M`

- **N** = diagnostics this tech has run inside their rolling window
- **M** = configured max from `~/.nexusoracle/config.toml` on the daemon
- **Teal** = headroom (< 66%)
- **Amber** = close (≥ 66%)
- **Terracotta** = at/over limit — next request returns `429 Too Many Requests`

Defaults: 3 per 4 h. Apple Demo is 1 per 24 h. Andrew/DevOps bypass entirely.

---

## Second line

- **last used**: rolling-relative timestamp of this tech's most recent
  diagnostic timestamp from `~/.nexusoracle/rate_limits.json` on the daemon.
  `never used` means zero rate-limited requests are logged inside the window.
- **Nh window**: the configured rolling window length for this tech.

---

## Data sources

| Signal | Source | Poll |
|---|---|---|
| Online / offline | `tailscale status` (local) | 10 s |
| Update-pending LED | `stat` on `oracle-chat-tauri.exe` vs `/var/lib/tailscale-monitor/pushed-<name>` | 10 s |
| Usage counter + last-used | `scp devops@100.67.227.31:~/.nexusoracle/rate_limits.json` → `/tmp/.tech-ticker-rate-limits.json` | 30 s |
| Per-tech limits config | `ssh devops@100.67.227.31 cat ~/.nexusoracle/config.toml` (naive TOML parse) | 30 s |

---

## Header indicators

- **14:42** — timestamp of the last successful rate-limits fetch.
- **⚠ sync** — displayed only when scp actually failed (network / SSH). A
  missing `rate_limits.json` (daemon never wrote it yet because nobody hit
  the limiter) is NOT a warning — that's a clean zero state. Hover for the
  scp stderr first line.

---

## Tech roster

| Tech | Tailscale host | Daemon-side limit |
|---|---|---|
| Nick | `nick` | 3 / 4 h |
| Leon | `leon-wsl` | 3 / 4 h |
| John | *not on Tailscale yet* | 3 / 4 h |
| Keenan | *not on Tailscale yet* | 3 / 4 h |
| Denior | *not on Tailscale yet* | 3 / 4 h |

When a tech gets Tailscale, add them to `PEERS=()` in `/opt/tailscale-monitor.sh`
and update their host in `TECHS` (top of `src/ui/tech_ticker.rs`). Andrew/DevOps
are intentionally not listed — they bypass the limiter.

---

## Autostart (Windows)

If you want this launching at boot like `nexus-ticker`:

1. Create `TechTicker.vbs` in `C:\Users\Autom\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\` that invokes the binary through WSL:
   ```vbs
   Set WshShell = CreateObject("WScript.Shell")
   WshShell.Run "wsl.exe -d Ubuntu -e bash -lc ""DISPLAY=:0 /opt/AxonML/nexus-agent/target/release/tech-ticker""", 0, False
   ```
2. Double-click once to verify, then it's persistent.

---

## Rebuild

```bash
cd /opt/AxonML/nexus-agent && cargo build --release --bin tech-ticker
pkill -f 'target/release/tech-ticker$'; setsid nohup /opt/AxonML/nexus-agent/target/release/tech-ticker > /tmp/tech-ticker.log 2>&1 < /dev/null &
```
