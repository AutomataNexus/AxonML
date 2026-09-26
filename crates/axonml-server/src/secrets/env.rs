//! Environment Backend — Secrets from Environment Variables
//!
//! Implements the `SecretsBackend` trait via `EnvBackend`, which reads secrets
//! from environment variables with a configurable prefix (default `AXONML_`).
//! Maps well-known `SecretKey` constants (JWT_SECRET, DB_USERNAME, DB_PASSWORD,
//! RESEND_API_KEY) to their corresponding env var names, and falls back to
//! uppercased key conversion for unknown keys. Empty values are treated as
//! missing; invalid UTF-8 is logged and ignored.
//!
//! # File
//! `crates/axonml-server/src/secrets/env.rs`
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

use super::{SecretKey, SecretsBackend, SecretsError};

// =============================================================================
// Environment Backend
// =============================================================================

/// Environment variable secrets backend
///
/// Reads secrets from environment variables with a configurable prefix.
/// By default, uses the `AXONML_` prefix.
pub struct EnvBackend {
    prefix: String,
    lookup: Lookup,
}

/// How a variable is read. Production reads the process environment; tests
/// supply their own table so they never mutate global process state.
type Lookup = Box<dyn Fn(&str) -> Result<String, std::env::VarError> + Send + Sync>;

impl EnvBackend {
    /// Create a new environment backend with a custom prefix
    ///
    /// # Arguments
    /// * `prefix` - Prefix for environment variables (e.g., "AXONML" for "AXONML_JWT_SECRET")
    pub fn new(prefix: &str) -> Self {
        Self::with_lookup(prefix, Box::new(|key: &str| std::env::var(key)))
    }

    /// Create a backend that resolves variables through `lookup` instead of
    /// the process environment.
    pub fn with_lookup(prefix: &str, lookup: Lookup) -> Self {
        Self {
            prefix: prefix.to_string(),
            lookup,
        }
    }

    /// Map a secret key to its environment variable name
    fn env_key(&self, key: &str) -> String {
        // Map well-known secret keys to environment variable names
        let env_suffix = match key {
            SecretKey::JWT_SECRET => "JWT_SECRET",
            SecretKey::DB_USERNAME => "BACKEND_USERNAME",
            SecretKey::DB_PASSWORD => "BACKEND_PASSWORD",
            SecretKey::RESEND_API_KEY => "RESEND_API_KEY",
            // For unknown keys, convert to uppercase with underscores
            other => {
                return format!("{}_{}", self.prefix, other.to_uppercase().replace('-', "_"));
            }
        };

        format!("{}_{}", self.prefix, env_suffix)
    }
}

impl Default for EnvBackend {
    fn default() -> Self {
        Self::new("AXONML")
    }
}

// =============================================================================
// SecretsBackend Implementation
// =============================================================================

#[async_trait::async_trait]
impl SecretsBackend for EnvBackend {
    async fn get_secret(&self, key: &str) -> Result<Option<String>, SecretsError> {
        let env_key = self.env_key(key);

        match (self.lookup)(&env_key) {
            Ok(value) if !value.is_empty() => {
                tracing::trace!(
                    env_var = %env_key,
                    "Secret loaded from environment"
                );
                Ok(Some(value))
            }
            Ok(_) => {
                // Empty value is treated as not set
                Ok(None)
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                tracing::warn!(
                    env_var = %env_key,
                    "Environment variable contains invalid UTF-8"
                );
                Ok(None)
            }
        }
    }

    fn name(&self) -> &'static str {
        "environment"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_key_mapping() {
        let backend = EnvBackend::default();

        assert_eq!(backend.env_key(SecretKey::JWT_SECRET), "AXONML_JWT_SECRET");
        assert_eq!(
            backend.env_key(SecretKey::DB_USERNAME),
            "AXONML_BACKEND_USERNAME"
        );
        assert_eq!(
            backend.env_key(SecretKey::DB_PASSWORD),
            "AXONML_BACKEND_PASSWORD"
        );
        assert_eq!(
            backend.env_key(SecretKey::RESEND_API_KEY),
            "AXONML_RESEND_API_KEY"
        );
    }

    #[test]
    fn test_custom_prefix() {
        let backend = EnvBackend::new("MYAPP");

        assert_eq!(backend.env_key(SecretKey::JWT_SECRET), "MYAPP_JWT_SECRET");
    }

    #[test]
    fn test_unknown_key_mapping() {
        let backend = EnvBackend::default();

        assert_eq!(backend.env_key("custom-key"), "AXONML_CUSTOM_KEY");
        assert_eq!(backend.env_key("another_key"), "AXONML_ANOTHER_KEY");
    }

    fn table(vars: &[(&str, &str)]) -> Lookup {
        let map: std::collections::HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Box::new(move |key: &str| map.get(key).cloned().ok_or(std::env::VarError::NotPresent))
    }

    #[tokio::test]
    async fn test_get_secret_from_env() {
        let backend = EnvBackend::with_lookup(
            "TEST_SECRETS",
            table(&[("TEST_SECRETS_JWT_SECRET", "test_value")]),
        );

        let result = backend.get_secret(SecretKey::JWT_SECRET).await.unwrap();
        assert_eq!(result, Some("test_value".to_string()));
    }

    #[tokio::test]
    async fn test_missing_secret() {
        let backend = EnvBackend::new("NONEXISTENT_PREFIX");

        let result = backend.get_secret(SecretKey::JWT_SECRET).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_empty_value_treated_as_missing() {
        let backend =
            EnvBackend::with_lookup("TEST_EMPTY", table(&[("TEST_EMPTY_JWT_SECRET", "")]));

        let result = backend.get_secret(SecretKey::JWT_SECRET).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_invalid_utf8_treated_as_missing() {
        let backend = EnvBackend::with_lookup(
            "TEST_BAD",
            Box::new(|_: &str| Err(std::env::VarError::NotUnicode("\u{fffd}".into()))),
        );

        let result = backend.get_secret(SecretKey::JWT_SECRET).await.unwrap();
        assert_eq!(result, None);
    }
}
