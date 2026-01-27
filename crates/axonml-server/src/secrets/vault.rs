//! HashiCorp Vault backend for secrets management.
//!
//! Supports Token and AppRole authentication with automatic token renewal.
//!
//! # Environment Variables
//!
//! ## Required
//! - `VAULT_ADDR` - Vault server address (e.g., `https://vault.example.com`)
//!
//! ## Authentication (one of)
//! - `VAULT_TOKEN` - Direct token authentication
//! - `VAULT_ROLE_ID` + `VAULT_SECRET_ID` - AppRole authentication
//!
//! ## Optional
//! - `VAULT_APPROLE_MOUNT` - AppRole auth mount path (default: `approle`)
//! - `VAULT_MOUNT` - KV secrets engine mount (default: `secret`)
//! - `VAULT_PATH` - Path within mount (default: `axonml`)

use super::{SecretsBackend, SecretsError};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use vaultrs::client::{VaultClient, VaultClientSettingsBuilder};
use vaultrs::kv2;

// =============================================================================
// Types
// =============================================================================

/// Vault authentication method
#[derive(Debug, Clone)]
pub enum VaultAuth {
    /// Direct token authentication
    Token(String),

    /// AppRole authentication (for automated deployments)
    AppRole {
        role_id: String,
        secret_id: String,
        mount: String,
    },
}

/// Internal settings for the Vault backend
#[derive(Clone)]
struct VaultSettings {
    address: String,
    mount: String,
    path: String,
}

// =============================================================================
// Vault Backend
// =============================================================================

/// HashiCorp Vault secrets backend
///
/// Supports both token and AppRole authentication, with automatic
/// token renewal for long-running servers.
pub struct VaultBackend {
    client: Arc<RwLock<VaultClient>>,
    settings: VaultSettings,
    auth: VaultAuth,
}

impl VaultBackend {
    /// Create a new Vault backend
    ///
    /// # Arguments
    /// * `address` - Vault server address
    /// * `auth` - Authentication method (Token or AppRole)
    /// * `mount` - KV secrets engine mount path
    /// * `path` - Path within the mount to read secrets from
    pub async fn new(
        address: &str,
        auth: VaultAuth,
        mount: &str,
        path: &str,
    ) -> Result<Self, SecretsError> {
        let initial_token = match &auth {
            VaultAuth::Token(t) => t.clone(),
            VaultAuth::AppRole { .. } => String::new(), // Will authenticate below
        };

        let client = VaultClient::new(
            VaultClientSettingsBuilder::default()
                .address(address)
                .token(&initial_token)
                .build()
                .map_err(|e| SecretsError::Config(format!("Invalid Vault settings: {}", e)))?,
        )
        .map_err(|e| SecretsError::Vault(format!("Failed to create Vault client: {}", e)))?;

        let backend = Self {
            client: Arc::new(RwLock::new(client)),
            settings: VaultSettings {
                address: address.to_string(),
                mount: mount.to_string(),
                path: path.to_string(),
            },
            auth,
        };

        // Authenticate with AppRole if configured
        if matches!(&backend.auth, VaultAuth::AppRole { .. }) {
            backend.authenticate_approle().await?;
        }

        // Verify we can connect by reading secrets
        backend.verify_connection().await?;

        Ok(backend)
    }

    /// Create a Vault backend from environment variables
    ///
    /// Returns `Ok(None)` if `VAULT_ADDR` is not set (Vault not configured).
    /// Returns an error if `VAULT_ADDR` is set but authentication is not configured.
    pub async fn from_env() -> Result<Option<Self>, SecretsError> {
        let address = match std::env::var("VAULT_ADDR") {
            Ok(v) if !v.is_empty() => v,
            _ => return Ok(None), // Vault not configured
        };

        // Determine authentication method
        let auth = if let (Ok(role_id), Ok(secret_id)) = (
            std::env::var("VAULT_ROLE_ID"),
            std::env::var("VAULT_SECRET_ID"),
        ) {
            // AppRole authentication
            let mount = std::env::var("VAULT_APPROLE_MOUNT").unwrap_or_else(|_| "approle".into());
            tracing::info!(
                mount = %mount,
                "Vault AppRole authentication configured"
            );
            VaultAuth::AppRole {
                role_id,
                secret_id,
                mount,
            }
        } else if let Ok(token) = std::env::var("VAULT_TOKEN") {
            // Token authentication
            tracing::info!("Vault token authentication configured");
            VaultAuth::Token(token)
        } else {
            return Err(SecretsError::Config(
                "VAULT_ADDR is set but no authentication method configured. \
                 Set VAULT_TOKEN for token auth, or VAULT_ROLE_ID + VAULT_SECRET_ID for AppRole."
                    .into(),
            ));
        };

        let mount = std::env::var("VAULT_MOUNT").unwrap_or_else(|_| "secret".into());
        let path = std::env::var("VAULT_PATH").unwrap_or_else(|_| "axonml".into());

        tracing::info!(
            address = %address,
            mount = %mount,
            path = %path,
            "Connecting to Vault"
        );

        Ok(Some(Self::new(&address, auth, &mount, &path).await?))
    }

    /// Authenticate using AppRole and get a token
    async fn authenticate_approle(&self) -> Result<(), SecretsError> {
        if let VaultAuth::AppRole {
            role_id,
            secret_id,
            mount,
        } = &self.auth
        {
            let client = self.client.read().await;

            // Login with AppRole
            let auth_info = vaultrs::auth::approle::login(&*client, mount, role_id, secret_id)
                .await
                .map_err(|e| SecretsError::AuthFailed(format!("AppRole login failed: {}", e)))?;

            drop(client);

            // Update client with new token
            let mut client = self.client.write().await;
            *client = VaultClient::new(
                VaultClientSettingsBuilder::default()
                    .address(&self.settings.address)
                    .token(&auth_info.client_token)
                    .build()
                    .map_err(|e| SecretsError::Config(e.to_string()))?,
            )
            .map_err(|e| SecretsError::Vault(e.to_string()))?;

            tracing::info!(
                lease_duration = auth_info.lease_duration,
                renewable = auth_info.renewable,
                "Vault AppRole authentication successful"
            );
        }
        Ok(())
    }

    /// Verify connection by attempting to read secrets
    async fn verify_connection(&self) -> Result<(), SecretsError> {
        let client = self.client.read().await;

        // Try to read the secrets path (will fail if path doesn't exist or no permission)
        let _: HashMap<String, String> =
            kv2::read(&*client, &self.settings.mount, &self.settings.path)
                .await
                .map_err(|e| {
                    SecretsError::Vault(format!(
                        "Failed to verify Vault connection (path: {}/{}): {}",
                        self.settings.mount, self.settings.path, e
                    ))
                })?;

        tracing::info!(
            mount = %self.settings.mount,
            path = %self.settings.path,
            "Vault connection verified"
        );
        Ok(())
    }

    /// Start background token renewal task
    ///
    /// This spawns a tokio task that periodically renews the Vault token.
    /// For AppRole auth, it will re-authenticate if renewal fails.
    pub fn start_token_renewal(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                if let Err(e) = self.maybe_renew_token().await {
                    tracing::error!(error = %e, "Vault token renewal failed");

                    // Try to re-authenticate with AppRole
                    if matches!(self.auth, VaultAuth::AppRole { .. }) {
                        tracing::info!("Attempting AppRole re-authentication");
                        if let Err(e) = self.authenticate_approle().await {
                            tracing::error!(error = %e, "AppRole re-authentication failed");
                        } else {
                            tracing::info!("AppRole re-authentication successful");
                        }
                    }
                }
            }
        })
    }

    /// Renew the current token if it's renewable
    async fn maybe_renew_token(&self) -> Result<(), SecretsError> {
        let client = self.client.read().await;

        match vaultrs::token::renew_self(&*client, None).await {
            Ok(auth_info) => {
                tracing::debug!(
                    ttl_seconds = auth_info.lease_duration,
                    renewable = auth_info.renewable,
                    "Vault token renewed"
                );
                Ok(())
            }
            Err(e) => {
                // Token might not be renewable - this is not always an error
                tracing::debug!(
                    error = %e,
                    "Token renewal failed (token may not be renewable)"
                );
                Err(SecretsError::Vault(format!("Token renewal failed: {}", e)))
            }
        }
    }
}

#[async_trait::async_trait]
impl SecretsBackend for VaultBackend {
    async fn get_secret(&self, key: &str) -> Result<Option<String>, SecretsError> {
        let client = self.client.read().await;

        let secrets: HashMap<String, String> =
            kv2::read(&*client, &self.settings.mount, &self.settings.path)
                .await
                .map_err(|e| SecretsError::Vault(format!("Failed to read secrets: {}", e)))?;

        Ok(secrets.get(key).cloned())
    }

    fn name(&self) -> &'static str {
        "vault"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vault_auth_variants() {
        let token_auth = VaultAuth::Token("hvs.test".to_string());
        assert!(matches!(token_auth, VaultAuth::Token(_)));

        let approle_auth = VaultAuth::AppRole {
            role_id: "role".to_string(),
            secret_id: "secret".to_string(),
            mount: "approle".to_string(),
        };
        assert!(matches!(approle_auth, VaultAuth::AppRole { .. }));
    }
}
