//! Environment variable backend for secrets management.
//!
//! This backend reads secrets from environment variables, making it suitable
//! for development and simple deployments.
//!
//! # Environment Variables
//!
//! | Secret Key | Environment Variable |
//! |------------|---------------------|
//! | `jwt_secret` | `AXONML_JWT_SECRET` |
//! | `db_username` | `AXONML_AEGIS_USERNAME` |
//! | `db_password` | `AXONML_AEGIS_PASSWORD` |
//! | `resend_api_key` | `AXONML_RESEND_API_KEY` |

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
}

impl EnvBackend {
    /// Create a new environment backend with a custom prefix
    ///
    /// # Arguments
    /// * `prefix` - Prefix for environment variables (e.g., "AXONML" for "AXONML_JWT_SECRET")
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    /// Map a secret key to its environment variable name
    fn env_key(&self, key: &str) -> String {
        // Map well-known secret keys to environment variable names
        let env_suffix = match key {
            SecretKey::JWT_SECRET => "JWT_SECRET",
            SecretKey::DB_USERNAME => "AEGIS_USERNAME",
            SecretKey::DB_PASSWORD => "AEGIS_PASSWORD",
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

#[async_trait::async_trait]
impl SecretsBackend for EnvBackend {
    async fn get_secret(&self, key: &str) -> Result<Option<String>, SecretsError> {
        let env_key = self.env_key(key);

        match std::env::var(&env_key) {
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
            "AXONML_AEGIS_USERNAME"
        );
        assert_eq!(
            backend.env_key(SecretKey::DB_PASSWORD),
            "AXONML_AEGIS_PASSWORD"
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

    #[tokio::test]
    async fn test_get_secret_from_env() {
        let backend = EnvBackend::new("TEST_SECRETS");

        // Set a test environment variable
        std::env::set_var("TEST_SECRETS_JWT_SECRET", "test_value");

        let result = backend.get_secret(SecretKey::JWT_SECRET).await.unwrap();
        assert_eq!(result, Some("test_value".to_string()));

        // Clean up
        std::env::remove_var("TEST_SECRETS_JWT_SECRET");
    }

    #[tokio::test]
    async fn test_missing_secret() {
        let backend = EnvBackend::new("NONEXISTENT_PREFIX");

        let result = backend.get_secret(SecretKey::JWT_SECRET).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_empty_value_treated_as_missing() {
        let backend = EnvBackend::new("TEST_EMPTY");

        std::env::set_var("TEST_EMPTY_JWT_SECRET", "");

        let result = backend.get_secret(SecretKey::JWT_SECRET).await.unwrap();
        assert_eq!(result, None);

        std::env::remove_var("TEST_EMPTY_JWT_SECRET");
    }
}
