//! Tailscale Tools — Network Status and Ping
//!
//! Two tools for the fieldtech agent's HVAC controller / tech laptop
//! monitoring on the Tailscale mesh:
//!
//! * `TailscaleStatusTool` (tailscale_status) runs `tailscale status --json`
//!   and formats Self + Peer entries into aligned
//!   `HOSTNAME / IP / STATUS / OS` rows, falling back to plain
//!   `tailscale status` text if the JSON path fails. Supports an optional
//!   `filter` substring that matches against hostnames case-insensitively.
//! * `TailscalePingTool` (tailscale_ping) runs `tailscale ping --c=3 <host>`
//!   with a 10-second tokio timeout for reachability and latency.
//!
//! # File
//! `nexus-agent/src/tools/tailscale.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::Tool;

// =============================================================================
// TailscaleStatusTool
// =============================================================================

pub struct TailscaleStatusTool;

#[derive(Deserialize)]
struct TailscaleArgs {
    #[serde(default)]
    filter: Option<String>,
}

#[async_trait]
impl Tool for TailscaleStatusTool {
    fn name(&self) -> &str { "tailscale_status" }

    fn description(&self) -> &str {
        "Query Tailscale network status. Returns all devices with their IPs, online/offline state, and OS. Optionally filter by hostname substring."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "filter": { "type": "string", "description": "Filter devices by hostname substring (optional)" }
            }
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: TailscaleArgs = serde_json::from_str(arguments)?;

        let output = tokio::process::Command::new("tailscale")
            .args(["status", "--json"])
            .output()
            .await?;

        if !output.status.success() {
            // Fall back to non-JSON output
            let plain = tokio::process::Command::new("tailscale")
                .arg("status")
                .output()
                .await?;
            let stdout = String::from_utf8_lossy(&plain.stdout);
            return match &args.filter {
                Some(f) => {
                    let filtered: String = stdout
                        .lines()
                        .filter(|l| l.to_lowercase().contains(&f.to_lowercase()))
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(if filtered.is_empty() {
                        format!("No devices matching '{f}'")
                    } else {
                        filtered
                    })
                }
                None => Ok(stdout.to_string()),
            };
        }

        let json_out: serde_json::Value =
            serde_json::from_slice(&output.stdout).unwrap_or(json!({}));

        // Extract peer info
        let peers = json_out.get("Peer").and_then(|p| p.as_object());
        let self_node = json_out.get("Self");

        let mut lines = Vec::new();

        // Add self node
        if let Some(self_n) = self_node {
            let name = self_n.get("HostName").and_then(|v| v.as_str()).unwrap_or("self");
            let ip = self_n.get("TailscaleIPs").and_then(|v| v.as_array())
                .and_then(|a| a.first()).and_then(|v| v.as_str()).unwrap_or("?");
            let os = self_n.get("OS").and_then(|v| v.as_str()).unwrap_or("?");
            lines.push(format!("{name:<25} {ip:<18} online    {os}"));
        }

        // Add peers
        if let Some(peers) = peers {
            for (_key, peer) in peers {
                let name = peer.get("HostName").and_then(|v| v.as_str()).unwrap_or("?");
                let ip = peer.get("TailscaleIPs").and_then(|v| v.as_array())
                    .and_then(|a| a.first()).and_then(|v| v.as_str()).unwrap_or("?");
                let online = if peer.get("Online").and_then(|v| v.as_bool()).unwrap_or(false) {
                    "online"
                } else {
                    "offline"
                };
                let os = peer.get("OS").and_then(|v| v.as_str()).unwrap_or("?");
                lines.push(format!("{name:<25} {ip:<18} {online:<9} {os}"));
            }
        }

        // Apply filter
        if let Some(f) = &args.filter {
            let f_lower = f.to_lowercase();
            lines.retain(|l| l.to_lowercase().contains(&f_lower));
        }

        if lines.is_empty() {
            Ok("No devices found".to_string())
        } else {
            let header = format!("{:<25} {:<18} {:<9} {}", "HOSTNAME", "IP", "STATUS", "OS");
            Ok(format!("{header}\n{}", lines.join("\n")))
        }
    }
}

// =============================================================================
// TailscalePingTool
// =============================================================================

pub struct TailscalePingTool;

#[derive(Deserialize)]
struct PingArgs {
    host: String,
}

#[async_trait]
impl Tool for TailscalePingTool {
    fn name(&self) -> &str { "tailscale_ping" }

    fn description(&self) -> &str {
        "Ping a Tailscale device by hostname or IP. Returns latency and reachability."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "host": { "type": "string", "description": "Tailscale hostname or IP to ping" }
            },
            "required": ["host"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: PingArgs = serde_json::from_str(arguments)?;

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio::process::Command::new("tailscale")
                .args(["ping", "--c=3", &args.host])
                .output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Ping timed out after 10s"))??;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if stdout.is_empty() && !stderr.is_empty() {
            Ok(format!("Ping failed: {stderr}"))
        } else {
            Ok(stdout.to_string())
        }
    }
}
