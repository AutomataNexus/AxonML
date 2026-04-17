//! Email Tool — Transactional Email via FerumMailSaaS
//!
//! Single-tool module providing `EmailTool` (name `send_email`). Accepts
//! `to`, `subject`, `body`, and an optional `from` (defaults to
//! `agent@automatanexus.com`), then POSTs a JSON payload with
//! `Authorization: Bearer <FERRUM_API_KEY>` to the FerumMailSaaS send
//! endpoint (default `http://127.0.0.1:3030/api/v1/send`, overridable via
//! `FERRUM_API_URL`).
//!
//! FerumMailSaaS is the AutomataNexus transactional email platform at
//! /opt/FerumMailSaaS. API key comes from the `FERRUM_API_KEY` environment
//! variable or the Vault path `secret/email/ferrum`. Used for alerting on
//! HVAC faults, CI failures, and training completions.
//!
//! # File
//! `nexus-agent/src/tools/email.rs`
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
// Ferrum Endpoint
// =============================================================================

const DEFAULT_FERRUM_URL: &str = "http://127.0.0.1:3030/api/v1/send";

// =============================================================================
// EmailTool
// =============================================================================

pub struct EmailTool;

#[derive(Deserialize)]
struct EmailArgs {
    to: String,
    subject: String,
    body: String,
    #[serde(default = "default_from")]
    from: String,
}

fn default_from() -> String {
    "agent@automatanexus.com".to_string()
}

#[async_trait]
impl Tool for EmailTool {
    fn name(&self) -> &str { "send_email" }

    fn description(&self) -> &str {
        "Send an email via FerumMailSaaS (self-hosted transactional email). Use for alerting on HVAC faults, CI failures, training completions. Requires FERRUM_API_KEY env var."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "to": { "type": "string", "description": "Recipient email address" },
                "subject": { "type": "string", "description": "Email subject line" },
                "body": { "type": "string", "description": "Email body (plain text or HTML)" },
                "from": { "type": "string", "description": "Sender address (default: agent@automatanexus.com)" }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: EmailArgs = serde_json::from_str(arguments)?;

        let api_url = std::env::var("FERRUM_API_URL")
            .unwrap_or_else(|_| DEFAULT_FERRUM_URL.to_string());

        let api_key = std::env::var("FERRUM_API_KEY")
            .map_err(|_| anyhow::anyhow!("FERRUM_API_KEY not set — cannot send email"))?;

        let client = reqwest::Client::new();
        let response = client
            .post(&api_url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&json!({
                "from": args.from,
                "to": args.to,
                "subject": args.subject,
                "html": args.body
            }))
            .send()
            .await?;

        if response.status().is_success() {
            let body: serde_json::Value = response.json().await.unwrap_or(json!({}));
            let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("ok");
            Ok(format!("Email sent to {} via Ferrum (id: {id})", args.to))
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("FerumMailSaaS returned {status}: {body}")
        }
    }
}
