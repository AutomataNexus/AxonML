//! Secrets management with HashiCorp Vault integration.
//!
//! Supports both Vault (production) and environment variables (development).
//!
//! # Priority Order
//!
//! 1. **Vault** - If `VAULT_ADDR` environment variable is set
//! 2. **Environment variables** - `AXONML_*` prefixed variables
//! 3. **Config file** - Fallback to `~/.axonml/config.toml` values

pub mod env;
pub mod vault;

use std::sync::Arc;
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Error, Debug)]
pub enum SecretsError {
    #[error("Secret not found: {0}")]
    NotFound(String),

    #[error("Vault error: {0}")]
    Vault(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Authentication failed: {0}")]
    AuthFailed(String),
}

// =============================================================================
// Secret Keys
// =============================================================================

/// Well-known secret key constants
pub struct SecretKey;

impl SecretKey {
    /// JWT signing secret (min 32 characters)
    pub const JWT_SECRET: &'static str = "jwt_secret";

    /// Database username for Aegis-DB
    pub const DB_USERNAME: &'static str = "db_username";

    /// Database password for Aegis-DB
    pub const DB_PASSWORD: &'static str = "db_password";

    /// Resend email API key
    pub const RESEND_API_KEY: &'static str = "resend_api_key";
}

// =============================================================================
// Secrets Backend Trait
// =============================================================================

/// Trait for secrets storage backends
#[async_trait::async_trait]
pub trait SecretsBackend: Send + Sync {
    /// Get a secret by key, returning None if not found in this backend
    async fn get_secret(&self, key: &str) -> Result<Option<String>, SecretsError>;

    /// Get the name of this backend for logging
    fn name(&self) -> &'static str;
}

// =============================================================================
// Secrets Manager
// =============================================================================

/// Manager for retrieving secrets from multiple backends
///
/// Backends are tried in order until one returns a value.
pub struct SecretsManager {
    backends: Vec<Arc<dyn SecretsBackend>>,
}

impl SecretsManager {
    /// Create a new empty secrets manager
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Add a backend to the manager (backends are tried in order added)
    pub fn with_backend(mut self, backend: Arc<dyn SecretsBackend>) -> Self {
        self.backends.push(backend);
        self
    }

    /// Get a secret, trying backends in order until one returns a value
    ///
    /// Returns an error if no backend has the secret.
    pub async fn get_secret(&self, key: &str) -> Result<String, SecretsError> {
        for backend in &self.backends {
            match backend.get_secret(key).await {
                Ok(Some(value)) => {
                    tracing::debug!(
                        secret_key = key,
                        backend = backend.name(),
                        "Secret loaded"
                    );
                    return Ok(value);
                }
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(
                        secret_key = key,
                        backend = backend.name(),
                        error = %e,
                        "Backend failed to retrieve secret"
                    );
                    continue;
                }
            }
        }
        Err(SecretsError::NotFound(key.to_string()))
    }

    /// Get a secret, returning None if not found in any backend
    ///
    /// Only returns an error for actual failures, not missing secrets.
    pub async fn get_secret_optional(&self, key: &str) -> Result<Option<String>, SecretsError> {
        match self.get_secret(key).await {
            Ok(v) => Ok(Some(v)),
            Err(SecretsError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Check if any backends are configured
    pub fn has_backends(&self) -> bool {
        !self.backends.is_empty()
    }

    /// Get the names of configured backends
    pub fn backend_names(&self) -> Vec<&'static str> {
        self.backends.iter().map(|b| b.name()).collect()
    }
}

impl Default for SecretsManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend {
        secrets: std::collections::HashMap<String, String>,
    }

    impl MockBackend {
        fn new(secrets: Vec<(&str, &str)>) -> Self {
            Self {
                secrets: secrets
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            }
        }
    }

    #[async_trait::async_trait]
    impl SecretsBackend for MockBackend {
        async fn get_secret(&self, key: &str) -> Result<Option<String>, SecretsError> {
            Ok(self.secrets.get(key).cloned())
        }

        fn name(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_secrets_manager_priority() {
        let backend1 = Arc::new(MockBackend::new(vec![("key1", "value1")]));
        let backend2 = Arc::new(MockBackend::new(vec![("key1", "value2"), ("key2", "value2")]));

        let manager = SecretsManager::new()
            .with_backend(backend1)
            .with_backend(backend2);

        // First backend has priority
        assert_eq!(manager.get_secret("key1").await.unwrap(), "value1");

        // Falls back to second backend
        assert_eq!(manager.get_secret("key2").await.unwrap(), "value2");

        // Not found in any backend
        assert!(manager.get_secret("key3").await.is_err());
    }

    #[tokio::test]
    async fn test_get_secret_optional() {
        let backend = Arc::new(MockBackend::new(vec![("exists", "value")]));
        let manager = SecretsManager::new().with_backend(backend);

        assert_eq!(
            manager.get_secret_optional("exists").await.unwrap(),
            Some("value".to_string())
        );
        assert_eq!(manager.get_secret_optional("missing").await.unwrap(), None);
    }
}
