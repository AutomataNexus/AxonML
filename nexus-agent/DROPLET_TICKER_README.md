# nexus-droplet-ticker

Desktop widget for monitoring + controlling DigitalOcean GPU droplets
(H100 / H200 class, ~$3-4/hr). Always-on-top eframe ticker that shows
live GPU% / VRAM / session cost + one-click **Start / Stop / Shutdown /
Destroy** lifecycle buttons, so a forgotten droplet can't silently bleed
credit.

Part of the nexus-agent ticker family:

| Ticker | Purpose |
| --- | --- |
| `nexus-ticker` | CI + ralph auto-fix |
| `tech-ticker` | field technician SSH launcher |
| `ferrum-ticker` | Ferrum mail probe monitor |
| `nexus-training-ticker` | live training-run metrics |
| **`nexus-droplet-ticker`** | **DigitalOcean GPU droplet lifecycle + cost** |

## Build

```bash
cd /opt/AxonML/nexus-agent
cargo build --release --bin nexus-droplet-ticker
```

Binary ends up at `./target/release/nexus-droplet-ticker`.

## First launch

Minimum env to get something useful on screen:

```bash
export DO_API_TOKEN="dop_v1_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export NEXUS_DROPLET_HOST="root@192.241.187.102"   # ssh target for live GPU probe
./target/release/nexus-droplet-ticker &
```

With just `DO_API_TOKEN` the ticker:

1. Calls `GET /v2/droplets` and filters to `gpu-*` sizes.
2. If exactly one GPU droplet exists, auto-pins to it and writes the id to
   `~/.config/nexus-droplet-ticker/state.toml` so next launch re-attaches.
3. If >1 GPU droplets exist, renders an in-widget picker.
4. Polls status every 10 s, billing every 60 s.

Without `DO_API_TOKEN` the ticker still renders — billing panel shows
zero, write buttons are disabled, header says `NO TOKEN`.

## Environment variables

| Var | Required | Default | Purpose |
| --- | --- | --- | --- |
| `DO_API_TOKEN` | yes for write ops | — | DO personal access token |
| `NEXUS_DROPLET_HOST` | recommended | — | `user@host` for SSH probe (nvidia-smi + /proc) |
| `NEXUS_DROPLET_ID` | optional | — | Pin ticker to a specific droplet id (bypasses picker) |
| `NEXUS_DROPLET_SNAPSHOT_ID` | required for Start-from-snapshot | — | Image id to clone from when provisioning |
| `NEXUS_DROPLET_SIZE` | optional | `gpu-h200x1-141gb` | Size slug for new droplets |
| `NEXUS_DROPLET_REGION` | optional | `nyc2` | Region slug for new droplets |
| `NEXUS_DROPLET_NAME` | optional | `nexus-gpu-<YYYYMMDDThhmmss>` | Name for newly provisioned droplet |

### How to find your DO API token

DigitalOcean control panel → **API** → **Tokens / Keys** → **Generate New Token**.
Grant both `read` and `write` scopes (write required for the lifecycle
buttons). Tokens look like `dop_v1_...` — paste into the env var.

### How to find your snapshot id

Either the DigitalOcean control panel (**Backups & Snapshots → Snapshots**,
the URL contains the id) or:

```bash
curl -s -H "Authorization: Bearer $DO_API_TOKEN" \
    "https://api.digitalocean.com/v2/snapshots?resource_type=droplet" | jq
```

Pick the snapshot id you want to clone future droplets from (typically the
one where your training environment is already baked in).

## systemd user service (always-on)

The user requirement is that the ticker is up whenever a GPU droplet is
live. Install this unit:

```ini
# ~/.config/systemd/user/nexus-droplet-ticker.service
[Unit]
Description=nexus-droplet-ticker — DigitalOcean GPU droplet monitor
After=graphical-session.target

[Service]
Environment=DO_API_TOKEN=dop_v1_xxx
Environment=NEXUS_DROPLET_HOST=root@1.2.3.4
Environment=NEXUS_DROPLET_SNAPSHOT_ID=123456789
ExecStart=/opt/AxonML/nexus-agent/target/release/nexus-droplet-ticker
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

Then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now nexus-droplet-ticker.service
systemctl --user status nexus-droplet-ticker.service
journalctl --user -u nexus-droplet-ticker.service -f
```

The binary takes a PID-based lock at `/tmp/.nexus-droplet-ticker.lock`,
so manual `nexus-droplet-ticker &` while the systemd copy is alive is a
no-op (second copy exits immediately).

## UI cheatsheet

| Element | Meaning |
| --- | --- |
| LED teal | droplet is ACTIVE (billing clock ticking) |
| LED amber (pulsing) | PROVISIONING / transitioning |
| LED terracotta | OFF / NO DROPLET / ERROR |
| LED slate | NO TOKEN / UNKNOWN |
| `session $X.XXXX` | live session cost = hourly rate × runtime |
| `MTD $X.XX` | month-to-date usage from `/v2/customers/my/balance` |
| `GPU% · last Nm` plot | dotted history over the last 5 minutes |
| `ip 1.2.3.4` | click to copy public IPv4 |

### Actions

- **Start** — if a droplet is pinned and `off`: `POST /actions {"type":"power_on"}`. If no droplet exists: `POST /droplets` from `$NEXUS_DROPLET_SNAPSHOT_ID`.
- **Stop** — hard power-off.
- **Shutdown** — graceful OS shutdown, then droplet stops.
- **Destroy** — opens a confirm modal with a "Snapshot first?" checkbox (default ON). Flow: `shutdown → snapshot → DELETE`, or just `DELETE` if unchecked.

## State file

`~/.config/nexus-droplet-ticker/state.toml` — tiny TOML file that stores
the pinned droplet id so the ticker re-attaches across restarts:

```toml
# nexus-droplet-ticker state
droplet_id = 123456789
updated = "14:22:10"
```

Safe to delete — the ticker will repopulate it on next droplet refresh.

## Theme

Dark / light theme toggle lives on the titlebar (☀ / ☾ glyph). Persists
to `/tmp/.nexus-droplet-ticker-theme` (separate from the other tickers so
you can mix themes across widgets).

## Files

- `/opt/AxonML/nexus-agent/src/ui/droplet_ticker.rs` — ticker binary
- `/opt/AxonML/nexus-agent/Cargo.toml` — `[[bin]] name = "nexus-droplet-ticker"`
- `/opt/AxonML/nexus-agent/DROPLET_TICKER_README.md` — this file

## Zero new deps

Uses only what's already in the workspace (eframe, egui, chrono).
DigitalOcean REST is called via `/usr/bin/curl` with
`--write-out '%{http_code}'`; live metrics via `/usr/bin/ssh`. No reqwest
/ ureq / ssh2 crates were added.
