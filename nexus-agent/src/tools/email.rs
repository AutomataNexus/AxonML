//! Email tool — send transactional email via FerumMailSaaS (self-hosted).
//!
//! FerumMailSaaS is the AutomataNexus transactional email platform at /opt/FerumMailSaaS.
//! Default endpoint: http://127.0.0.1:3030/api/v1/send (configurable via FERRUM_API_URL).
//! API key from FERRUM_API_KEY env var or vault at secret/email/ferrum.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::Tool;

const DEFAULT_FERRUM_URL: &str = "http://127.0.0.1:3030/api/v1/send";

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
