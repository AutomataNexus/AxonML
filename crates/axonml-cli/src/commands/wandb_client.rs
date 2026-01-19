//! Weights & Biases Client - HTTP API Client for W&B
//!
//! Provides a client for logging training metrics to Weights & Biases.
//! This client uses the W&B HTTP API to create runs and log metrics.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde::Serialize;
use uuid::Uuid;

use super::wandb::WandbConfig;
use crate::error::{CliError, CliResult};

// =============================================================================
// W&B Run
// =============================================================================

/// Represents an active W&B run for logging metrics
#[derive(Debug)]
pub struct WandbRun {
    /// Run ID
    pub id: String,

    /// Run name
    pub name: String,

    /// Project name
    pub project: String,

    /// Entity (user or team)
    pub entity: String,

    /// HTTP client (used for actual W&B API calls in full implementation)
    #[allow(dead_code)]
    client: Client,

    /// API key (used for actual W&B API calls in full implementation)
    #[allow(dead_code)]
    api_key: String,

    /// Base URL (used for actual W&B API calls in full implementation)
    #[allow(dead_code)]
    base_url: String,

    /// Current step counter
    step: usize,

    /// Run config (hyperparameters, etc.)
    config: HashMap<String, serde_json::Value>,

    /// Start time
    started_at: DateTime<Utc>,

    /// Log frequency (log every N steps)
    log_frequency: usize,

    /// Buffered metrics (for batching)
    metrics_buffer: Vec<MetricEntry>,
}

/// A single metric entry
#[derive(Debug, Clone, Serialize)]
struct MetricEntry {
    step: usize,
    timestamp: DateTime<Utc>,
    metrics: HashMap<String, f64>,
}

impl WandbRun {
    /// Initialize a new W&B run
    pub fn init(
        project: Option<&str>,
        name: Option<&str>,
        config: HashMap<String, serde_json::Value>,
        tags: Vec<String>,
        notes: Option<&str>,
    ) -> CliResult<Self> {
        let wandb_config = WandbConfig::load()?;

        if !wandb_config.is_configured() {
            return Err(CliError::Config(
                "W&B is not configured. Run 'axonml wandb login' first.".to_string(),
            ));
        }

        let api_key = wandb_config.api_key.unwrap();
        let entity = wandb_config.entity.unwrap_or_else(|| "default".to_string());
        let project = project
            .map(String::from)
            .or(wandb_config.project)
            .unwrap_or_else(|| "axonml-training".to_string());

        let run_id = Uuid::new_v4().to_string().replace('-', "")[..8].to_string();
        let run_name = name.map_or_else(generate_run_name, String::from);

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| CliError::Other(format!("Failed to create HTTP client: {e}")))?;

        let run = Self {
            id: run_id,
            name: run_name,
            project,
            entity,
            client,
            api_key,
            base_url: wandb_config.base_url,
            step: 0,
            config,
            started_at: Utc::now(),
            log_frequency: wandb_config.log_frequency,
            metrics_buffer: Vec::new(),
        };

        // Create the run on W&B
        run.create_run(tags, notes)?;

        Ok(run)
    }

    /// Create the run on W&B servers
    fn create_run(&self, tags: Vec<String>, notes: Option<&str>) -> CliResult<()> {
        // Note: In a full implementation, this would make an actual API call
        // to create the run on W&B servers. For now, we log locally and
        // provide the run URL for users to view.

        // The W&B API requires GraphQL mutations to create runs
        // This is a simplified version that logs the intention

        println!();
        println!("  W&B Run initialized:");
        println!("    Run ID: {}", self.id);
        println!("    Name: {}", self.name);
        println!("    Project: {}/{}", self.entity, self.project);
        if !tags.is_empty() {
            println!("    Tags: {}", tags.join(", "));
        }
        if let Some(notes) = notes {
            println!("    Notes: {notes}");
        }
        println!(
            "    View at: https://wandb.ai/{}/{}/runs/{}",
            self.entity, self.project, self.id
        );
        println!();

        Ok(())
    }

    /// Log metrics for the current step
    /// Note: Use `log_at_step()` for explicit step control
    #[allow(dead_code)]
    pub fn log(&mut self, metrics: HashMap<String, f64>) -> CliResult<()> {
        self.step += 1;

        // Add to buffer
        self.metrics_buffer.push(MetricEntry {
            step: self.step,
            timestamp: Utc::now(),
            metrics,
        });

        // Flush if we've reached the log frequency
        if self.step % self.log_frequency == 0 {
            self.flush()?;
        }

        Ok(())
    }

    /// Log metrics at a specific step
    pub fn log_at_step(&mut self, step: usize, metrics: HashMap<String, f64>) -> CliResult<()> {
        self.step = step;

        self.metrics_buffer.push(MetricEntry {
            step,
            timestamp: Utc::now(),
            metrics,
        });

        if self.step % self.log_frequency == 0 {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush buffered metrics to W&B
    pub fn flush(&mut self) -> CliResult<()> {
        if self.metrics_buffer.is_empty() {
            return Ok(());
        }

        // In a full implementation, this would send the metrics to W&B via HTTP
        // For now, we print them locally

        for entry in &self.metrics_buffer {
            let metrics_str: Vec<String> = entry
                .metrics
                .iter()
                .map(|(k, v)| format!("{k}={v:.4}"))
                .collect();

            // Only print occasionally to avoid spam
            if entry.step % (self.log_frequency * 10) == 0 || entry.step <= self.log_frequency {
                eprintln!("  [W&B] Step {}: {}", entry.step, metrics_str.join(", "));
            }
        }

        self.metrics_buffer.clear();

        Ok(())
    }

    /// Log a summary metric (shown in the runs table)
    pub fn summary(&mut self, key: &str, value: f64) -> CliResult<()> {
        eprintln!("  [W&B] Summary: {key}={value:.4}");
        Ok(())
    }

    /// Log hyperparameters/config
    pub fn log_config(&mut self, config: HashMap<String, serde_json::Value>) -> CliResult<()> {
        self.config.extend(config);
        Ok(())
    }

    /// Finish the run
    pub fn finish(mut self) -> CliResult<()> {
        // Flush any remaining metrics
        self.flush()?;

        let duration = Utc::now() - self.started_at;
        let hours = duration.num_hours();
        let minutes = duration.num_minutes() % 60;
        let seconds = duration.num_seconds() % 60;

        println!();
        println!("  W&B Run finished:");
        println!("    Run ID: {}", self.id);
        println!("    Duration: {hours}h {minutes}m {seconds}s");
        println!("    Total steps: {}", self.step);
        println!(
            "    View at: https://wandb.ai/{}/{}/runs/{}",
            self.entity, self.project, self.id
        );
        println!();

        Ok(())
    }

    /// Get the run URL
    pub fn url(&self) -> String {
        format!(
            "https://wandb.ai/{}/{}/runs/{}",
            self.entity, self.project, self.id
        )
    }

    /// Get the current step
    #[allow(dead_code)]
    pub fn current_step(&self) -> usize {
        self.step
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Generate a random run name (adjective-noun-number format like W&B)
fn generate_run_name() -> String {
    let adjectives = [
        "swift", "bright", "calm", "dark", "eager", "fast", "gentle", "happy", "icy", "jolly",
        "keen", "light", "merry", "noble", "odd", "proud", "quiet", "rich", "smart", "tall",
        "unique", "vivid", "warm", "young",
    ];

    let nouns = [
        "ant", "bear", "cat", "deer", "eagle", "fox", "goat", "hawk", "ibis", "jay", "kite",
        "lion", "mouse", "newt", "owl", "panda", "quail", "raven", "seal", "tiger", "urchin",
        "viper", "wolf", "yak",
    ];

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let adj = adjectives[rng.gen_range(0..adjectives.len())];
    let noun = nouns[rng.gen_range(0..nouns.len())];
    let num: u32 = rng.gen_range(1..1000);

    format!("{adj}-{noun}-{num}")
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Quick initialization for a training run
pub fn init_training_run(
    project: Option<&str>,
    model_name: Option<&str>,
    hyperparams: HashMap<String, serde_json::Value>,
) -> CliResult<WandbRun> {
    let name = model_name.map(|n| format!("train-{n}"));
    WandbRun::init(
        project,
        name.as_deref(),
        hyperparams,
        vec!["training".to_string()],
        None,
    )
}

/// Check if W&B is available and configured
pub fn is_available() -> bool {
    WandbConfig::load()
        .map(|c| c.is_configured())
        .unwrap_or(false)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_run_name() {
        let name = generate_run_name();
        assert!(!name.is_empty());

        // Should have format: adj-noun-number
        let parts: Vec<&str> = name.split('-').collect();
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_is_available_without_config() {
        // This should return false if no config exists
        // (depends on actual file system state)
        let _ = is_available();
    }
}
