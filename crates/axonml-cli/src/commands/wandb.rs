//! Weights & Biases Integration - W&B Credential Management
//!
//! Provides commands for configuring W&B integration, allowing users to
//! link their training runs to their W&B account for experiment tracking.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::io::{self, Write};
use std::path::PathBuf;

use super::utils::{print_header, print_info, print_kv, print_success, print_warning};
use crate::cli::{WandbArgs, WandbConfigArgs, WandbSubcommand};
use crate::error::{CliError, CliResult};

use serde::{Deserialize, Serialize};

// =============================================================================
// W&B Configuration
// =============================================================================

/// Weights & Biases configuration stored in ~/.axonml/wandb.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandbConfig {
    /// W&B API key
    pub api_key: Option<String>,

    /// Default entity (username or team name)
    pub entity: Option<String>,

    /// Default project name
    pub project: Option<String>,

    /// W&B base URL (for self-hosted instances)
    #[serde(default = "default_base_url")]
    pub base_url: String,

    /// Whether W&B logging is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Log frequency (every N steps)
    #[serde(default = "default_log_frequency")]
    pub log_frequency: usize,

    /// Whether to log model checkpoints to W&B
    #[serde(default)]
    pub log_checkpoints: bool,

    /// Whether to log system metrics (GPU, CPU, memory)
    #[serde(default = "default_log_system")]
    pub log_system_metrics: bool,
}

fn default_base_url() -> String {
    "https://api.wandb.ai".to_string()
}

fn default_enabled() -> bool {
    true
}

fn default_log_frequency() -> usize {
    10
}

fn default_log_system() -> bool {
    true
}

impl Default for WandbConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            entity: None,
            project: None,
            base_url: default_base_url(),
            enabled: default_enabled(),
            log_frequency: default_log_frequency(),
            log_checkpoints: false,
            log_system_metrics: default_log_system(),
        }
    }
}

impl WandbConfig {
    /// Get the path to the W&B config file
    pub fn config_path() -> CliResult<PathBuf> {
        let config_dir = dirs::home_dir()
            .ok_or_else(|| CliError::Config("Could not find home directory".to_string()))?
            .join(".axonml");

        Ok(config_dir.join("wandb.toml"))
    }

    /// Get the path to the axonml config directory
    pub fn config_dir() -> CliResult<PathBuf> {
        let config_dir = dirs::home_dir()
            .ok_or_else(|| CliError::Config("Could not find home directory".to_string()))?
            .join(".axonml");

        Ok(config_dir)
    }

    /// Load W&B configuration from disk
    pub fn load() -> CliResult<Self> {
        let path = Self::config_path()?;

        if !path.exists() {
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(&path)?;
        let config: WandbConfig = toml::from_str(&content)
            .map_err(|e| CliError::Config(format!("Failed to parse wandb.toml: {e}")))?;

        Ok(config)
    }

    /// Save W&B configuration to disk
    pub fn save(&self) -> CliResult<()> {
        let config_dir = Self::config_dir()?;

        // Create config directory if it doesn't exist
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir)?;
        }

        let path = Self::config_path()?;
        let content =
            toml::to_string_pretty(self).map_err(|e| CliError::Serialization(e.to_string()))?;

        std::fs::write(&path, content)?;

        Ok(())
    }

    /// Check if W&B is configured with valid credentials
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some() && self.enabled
    }

    /// Mask the API key for display (show first 4 and last 4 chars)
    pub fn masked_api_key(&self) -> String {
        match &self.api_key {
            Some(key) if key.len() > 8 => {
                format!("{}...{}", &key[..4], &key[key.len() - 4..])
            }
            Some(_) => "****".to_string(),
            None => "(not set)".to_string(),
        }
    }
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `wandb` command
pub fn execute(args: WandbArgs) -> CliResult<()> {
    match args.action {
        WandbSubcommand::Login => login_interactive(),
        WandbSubcommand::Logout => logout(),
        WandbSubcommand::Status => show_status(),
        WandbSubcommand::Config(config_args) => configure(config_args),
        WandbSubcommand::Enable => enable(),
        WandbSubcommand::Disable => disable(),
    }
}

// =============================================================================
// Command Implementations
// =============================================================================

/// Interactive login to W&B
fn login_interactive() -> CliResult<()> {
    print_header("Weights & Biases Login");

    println!();
    print_info("Get your API key from: https://wandb.ai/authorize");
    println!();

    // Read API key
    print!("Enter your W&B API key: ");
    io::stdout().flush()?;

    let mut api_key = String::new();
    io::stdin().read_line(&mut api_key)?;
    let api_key = api_key.trim().to_string();

    if api_key.is_empty() {
        return Err(CliError::Config("API key cannot be empty".to_string()));
    }

    // Read entity (optional)
    print!("Enter your W&B entity (username or team, press Enter to skip): ");
    io::stdout().flush()?;

    let mut entity = String::new();
    io::stdin().read_line(&mut entity)?;
    let entity = entity.trim();
    let entity = if entity.is_empty() {
        None
    } else {
        Some(entity.to_string())
    };

    // Read default project (optional)
    print!("Enter default project name (press Enter to skip): ");
    io::stdout().flush()?;

    let mut project = String::new();
    io::stdin().read_line(&mut project)?;
    let project = project.trim();
    let project = if project.is_empty() {
        None
    } else {
        Some(project.to_string())
    };

    // Load existing config and update
    let mut config = WandbConfig::load().unwrap_or_default();
    config.api_key = Some(api_key);
    config.entity = entity;
    config.project = project;
    config.enabled = true;

    // Save configuration
    config.save()?;

    println!();
    print_success("Successfully logged in to Weights & Biases!");

    if let Some(ref entity) = config.entity {
        print_kv("Entity", entity);
    }
    if let Some(ref project) = config.project {
        print_kv("Default Project", project);
    }

    let config_path = WandbConfig::config_path()?;
    print_kv("Config saved to", &config_path.display().to_string());

    println!();
    print_info("Your training runs will now be logged to W&B automatically.");
    print_info("Use 'axonml wandb status' to check your configuration.");

    Ok(())
}

/// Remove W&B credentials
fn logout() -> CliResult<()> {
    print_header("Weights & Biases Logout");

    let mut config = WandbConfig::load()?;

    if config.api_key.is_none() {
        print_warning("No W&B credentials found");
        return Ok(());
    }

    config.api_key = None;
    config.enabled = false;
    config.save()?;

    println!();
    print_success("Successfully logged out from Weights & Biases");
    print_info("Your other settings have been preserved");

    Ok(())
}

/// Show current W&B configuration status
fn show_status() -> CliResult<()> {
    print_header("Weights & Biases Status");

    let config = WandbConfig::load()?;

    println!();

    if config.is_configured() {
        print_success("W&B is configured and enabled");
    } else if config.api_key.is_some() {
        print_warning("W&B has credentials but is disabled");
    } else {
        print_warning("W&B is not configured");
        println!();
        print_info("Run 'axonml wandb login' to set up W&B integration");
        return Ok(());
    }

    println!();
    print_kv("API Key", &config.masked_api_key());
    print_kv("Entity", config.entity.as_deref().unwrap_or("(not set)"));
    print_kv("Project", config.project.as_deref().unwrap_or("(not set)"));
    print_kv("Base URL", &config.base_url);
    print_kv("Enabled", &config.enabled.to_string());
    print_kv(
        "Log Frequency",
        &format!("every {} steps", config.log_frequency),
    );
    print_kv("Log Checkpoints", &config.log_checkpoints.to_string());
    print_kv("Log System Metrics", &config.log_system_metrics.to_string());

    println!();
    let config_path = WandbConfig::config_path()?;
    print_kv("Config file", &config_path.display().to_string());

    Ok(())
}

/// Configure W&B settings
fn configure(args: WandbConfigArgs) -> CliResult<()> {
    print_header("Configure Weights & Biases");

    let mut config = WandbConfig::load()?;
    let mut updated = false;

    println!();

    if let Some(api_key) = args.api_key {
        config.api_key = Some(api_key);
        updated = true;
        print_success("API key updated");
    }

    if let Some(entity) = args.entity {
        config.entity = Some(entity.clone());
        updated = true;
        print_kv("Entity set to", &entity);
    }

    if let Some(project) = args.project {
        config.project = Some(project.clone());
        updated = true;
        print_kv("Project set to", &project);
    }

    if let Some(base_url) = args.base_url {
        config.base_url = base_url.clone();
        updated = true;
        print_kv("Base URL set to", &base_url);
    }

    if let Some(log_frequency) = args.log_frequency {
        config.log_frequency = log_frequency;
        updated = true;
        print_kv(
            "Log frequency set to",
            &format!("every {log_frequency} steps"),
        );
    }

    if let Some(log_checkpoints) = args.log_checkpoints {
        config.log_checkpoints = log_checkpoints;
        updated = true;
        print_kv("Log checkpoints", &log_checkpoints.to_string());
    }

    if let Some(log_system_metrics) = args.log_system_metrics {
        config.log_system_metrics = log_system_metrics;
        updated = true;
        print_kv("Log system metrics", &log_system_metrics.to_string());
    }

    if updated {
        config.save()?;
        println!();
        print_success("Configuration saved");
    } else {
        print_warning("No changes specified");
        println!();
        print_info("Available options:");
        println!("  --api-key <KEY>         Set W&B API key");
        println!("  --entity <NAME>         Set default entity");
        println!("  --project <NAME>        Set default project");
        println!("  --base-url <URL>        Set W&B API URL");
        println!("  --log-frequency <N>     Log every N steps");
        println!("  --log-checkpoints       Log model checkpoints");
        println!("  --log-system-metrics    Log system metrics");
    }

    Ok(())
}

/// Enable W&B logging
fn enable() -> CliResult<()> {
    let mut config = WandbConfig::load()?;

    if config.api_key.is_none() {
        print_warning("No W&B credentials found");
        print_info("Run 'axonml wandb login' first to set up credentials");
        return Ok(());
    }

    config.enabled = true;
    config.save()?;

    print_success("W&B logging enabled");
    print_info("Training runs will now be logged to Weights & Biases");

    Ok(())
}

/// Disable W&B logging
fn disable() -> CliResult<()> {
    let mut config = WandbConfig::load()?;

    config.enabled = false;
    config.save()?;

    print_success("W&B logging disabled");
    print_info("Your credentials have been preserved");
    print_info("Use 'axonml wandb enable' to re-enable logging");

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wandb_config_default() {
        let config = WandbConfig::default();
        assert!(!config.is_configured());
        assert_eq!(config.base_url, "https://api.wandb.ai");
    }

    #[test]
    fn test_masked_api_key() {
        let mut config = WandbConfig::default();

        // No key
        assert_eq!(config.masked_api_key(), "(not set)");

        // Short key
        config.api_key = Some("abcd".to_string());
        assert_eq!(config.masked_api_key(), "****");

        // Long key
        config.api_key = Some("abcdefghijklmnop".to_string());
        assert_eq!(config.masked_api_key(), "abcd...mnop");
    }

    #[test]
    fn test_is_configured() {
        let mut config = WandbConfig::default();
        assert!(!config.is_configured());

        config.api_key = Some("test-key".to_string());
        config.enabled = true;
        assert!(config.is_configured());

        config.enabled = false;
        assert!(!config.is_configured());
    }
}
