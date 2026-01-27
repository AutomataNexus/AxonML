//! Configuration module for AxonML Server
//!
//! Handles loading configuration from TOML files and environment variables.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    ReadError(#[from] std::io::Error),
    #[error("Failed to parse config: {0}")]
    ParseError(#[from] toml::de::Error),
    #[error("Missing required configuration: {0}")]
    MissingConfig(String),
}

/// Main server configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub aegis: AegisConfig,
    pub auth: AuthConfig,
    pub inference: InferenceConfig,
    pub dashboard: DashboardConfig,
    #[serde(default)]
    pub hub: HubConfig,
}

/// HTTP server configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
}

/// Aegis-DB connection configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AegisConfig {
    #[serde(default = "default_aegis_host")]
    pub host: String,
    #[serde(default = "default_aegis_port")]
    pub port: u16,
    #[serde(default = "default_aegis_user")]
    pub username: String,
    #[serde(default = "default_aegis_pass")]
    pub password: String,
}

/// Authentication configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AuthConfig {
    #[serde(default = "default_jwt_secret")]
    pub jwt_secret: String,
    #[serde(default = "default_jwt_expiry")]
    pub jwt_expiry_hours: u64,
    #[serde(default = "default_session_timeout")]
    pub session_timeout_minutes: u64,
    #[serde(default)]
    pub require_mfa: bool,
    #[serde(default = "default_allow_registration")]
    pub allow_public_registration: bool,
}

/// Inference server configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    #[serde(default = "default_port_start")]
    pub default_port_range_start: u16,
    #[serde(default = "default_port_end")]
    pub default_port_range_end: u16,
    #[serde(default = "default_max_endpoints")]
    pub max_endpoints: u32,
}

/// Dashboard configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DashboardConfig {
    #[serde(default = "default_dashboard_port")]
    pub port: u16,
}

/// Model Hub configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HubConfig {
    #[serde(default = "default_hub_url")]
    pub hub_url: String,
    #[serde(default = "default_hub_cache_dir")]
    pub cache_dir: String,
}

// Default value functions
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    3000
}
fn default_data_dir() -> String {
    "~/.axonml".to_string()
}
fn default_aegis_host() -> String {
    "localhost".to_string()
}
fn default_aegis_port() -> u16 {
    3020
}
// SECURITY: No default database credentials - must be explicitly configured
fn default_aegis_user() -> String {
    String::new()
}
fn default_aegis_pass() -> String {
    String::new()
}
// SECURITY: No default JWT secret - must be explicitly configured
fn default_jwt_secret() -> String {
    String::new()
}
fn default_jwt_expiry() -> u64 {
    24
}
fn default_session_timeout() -> u64 {
    30
}
fn default_allow_registration() -> bool {
    true
}
fn default_port_start() -> u16 {
    8100
}
fn default_port_end() -> u16 {
    8199
}
fn default_max_endpoints() -> u32 {
    10
}
fn default_dashboard_port() -> u16 {
    8080
}
fn default_hub_url() -> String {
    "https://hub.axonml.dev/v1".to_string()
}
fn default_hub_cache_dir() -> String {
    "~/.axonml/hub_cache".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            aegis: AegisConfig::default(),
            auth: AuthConfig::default(),
            inference: InferenceConfig::default(),
            dashboard: DashboardConfig::default(),
            hub: HubConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            data_dir: default_data_dir(),
        }
    }
}

impl Default for AegisConfig {
    fn default() -> Self {
        Self {
            host: default_aegis_host(),
            port: default_aegis_port(),
            username: default_aegis_user(),
            password: default_aegis_pass(),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: default_jwt_secret(),
            jwt_expiry_hours: default_jwt_expiry(),
            session_timeout_minutes: default_session_timeout(),
            require_mfa: false,
            allow_public_registration: default_allow_registration(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            default_port_range_start: default_port_start(),
            default_port_range_end: default_port_end(),
            max_endpoints: default_max_endpoints(),
        }
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            port: default_dashboard_port(),
        }
    }
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            hub_url: default_hub_url(),
            cache_dir: default_hub_cache_dir(),
        }
    }
}

impl Config {
    /// Load configuration from the default location (~/.axonml/config.toml)
    pub fn load() -> Result<Self, ConfigError> {
        let config_path = Self::config_path();
        if config_path.exists() {
            Self::load_from_path(&config_path)
        } else {
            Ok(Self::default())
        }
    }

    /// Load configuration from a specific path
    pub fn load_from_path(path: &PathBuf) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Get the default configuration file path
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".axonml")
            .join("config.toml")
    }

    /// Get the data directory path (expanded)
    pub fn data_dir(&self) -> PathBuf {
        let path = self.server.data_dir.replace(
            "~",
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .to_str()
                .unwrap_or("."),
        );
        PathBuf::from(path)
    }

    /// Get the models directory
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir().join("models")
    }

    /// Get the runs directory
    pub fn runs_dir(&self) -> PathBuf {
        self.data_dir().join("runs")
    }

    /// Get the logs directory
    pub fn logs_dir(&self) -> PathBuf {
        self.data_dir().join("logs")
    }

    /// Get the checkpoints directory (for training notebook checkpoints)
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.data_dir().join("checkpoints")
    }

    /// Get the hub cache directory
    pub fn hub_cache_dir(&self) -> PathBuf {
        let path = self.hub.cache_dir.replace(
            "~",
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .to_str()
                .unwrap_or("."),
        );
        PathBuf::from(path)
    }

    /// Ensure all required directories exist
    pub fn ensure_directories(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.data_dir())?;
        std::fs::create_dir_all(self.models_dir())?;
        std::fs::create_dir_all(self.runs_dir())?;
        std::fs::create_dir_all(self.logs_dir())?;
        std::fs::create_dir_all(self.checkpoints_dir())?;
        std::fs::create_dir_all(self.hub_cache_dir())?;
        Ok(())
    }

    /// Get the Aegis-DB connection URL
    pub fn aegis_url(&self) -> String {
        format!("http://{}:{}", self.aegis.host, self.aegis.port)
    }

    /// Validate configuration - always called on startup
    pub fn validate(&self) -> Result<(), ConfigError> {
        // SECURITY: JWT secret must be explicitly configured
        if self.auth.jwt_secret.is_empty() {
            return Err(ConfigError::MissingConfig(
                "jwt_secret is required. Set auth.jwt_secret in config.toml or AXONML_JWT_SECRET environment variable.".to_string()
            ));
        }

        // Check if JWT secret is long enough (at least 32 bytes for HS256)
        if self.auth.jwt_secret.len() < 32 {
            return Err(ConfigError::MissingConfig(
                "jwt_secret must be at least 32 characters long for security.".to_string(),
            ));
        }

        // SECURITY: Database credentials must be explicitly configured
        if self.aegis.username.is_empty() || self.aegis.password.is_empty() {
            return Err(ConfigError::MissingConfig(
                "Database credentials are required. Set aegis.username and aegis.password in config.toml.".to_string()
            ));
        }

        Ok(())
    }

    /// Validate configuration for production (returns warnings for non-critical issues)
    pub fn validate_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.auth.allow_public_registration {
            warnings.push("INFO: Public registration is enabled.".to_string());
        }

        if !self.auth.require_mfa {
            warnings.push("INFO: MFA is not required for users.".to_string());
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.aegis.port, 3020);
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
[server]
host = "127.0.0.1"
port = 8000

[aegis]
host = "db.example.com"
port = 5432

[auth]
jwt_secret = "test_secret"
require_mfa = true

[inference]
max_endpoints = 5

[dashboard]
port = 3000
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.server.port, 8000);
        assert_eq!(config.aegis.host, "db.example.com");
        assert!(config.auth.require_mfa);
    }
}
