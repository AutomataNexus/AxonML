//! Database schema initialization for AxonML
//!
//! Creates all required collections in Aegis-DB Document Store.

use super::{Database, DbError};
use tracing::info;

/// Collection names used by AxonML
pub const USERS_COLLECTION: &str = "axonml_users";
pub const RUNS_COLLECTION: &str = "axonml_runs";
pub const MODELS_COLLECTION: &str = "axonml_models";
pub const VERSIONS_COLLECTION: &str = "axonml_model_versions";
pub const ENDPOINTS_COLLECTION: &str = "axonml_endpoints";
pub const DATASETS_COLLECTION: &str = "axonml_datasets";

/// Schema definitions for all AxonML collections
pub struct Schema;

impl Schema {
    /// Initialize all database collections
    pub async fn init(db: &Database) -> Result<(), DbError> {
        info!("Initializing AxonML database schema...");

        // Create document store collections
        Self::create_users_collection(db).await?;
        Self::create_runs_collection(db).await?;
        Self::create_models_collection(db).await?;
        Self::create_model_versions_collection(db).await?;
        Self::create_endpoints_collection(db).await?;
        Self::create_datasets_collection(db).await?;

        info!("Database schema initialized successfully");
        Ok(())
    }

    /// Create users collection
    async fn create_users_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(USERS_COLLECTION).await?;
        info!("Created {} collection", USERS_COLLECTION);
        Ok(())
    }

    /// Create training runs collection
    async fn create_runs_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(RUNS_COLLECTION).await?;
        info!("Created {} collection", RUNS_COLLECTION);
        Ok(())
    }

    /// Create models collection
    async fn create_models_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(MODELS_COLLECTION).await?;
        info!("Created {} collection", MODELS_COLLECTION);
        Ok(())
    }

    /// Create model versions collection
    async fn create_model_versions_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(VERSIONS_COLLECTION).await?;
        info!("Created {} collection", VERSIONS_COLLECTION);
        Ok(())
    }

    /// Create inference endpoints collection
    async fn create_endpoints_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(ENDPOINTS_COLLECTION).await?;
        info!("Created {} collection", ENDPOINTS_COLLECTION);
        Ok(())
    }

    /// Create datasets collection
    async fn create_datasets_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(DATASETS_COLLECTION).await?;
        info!("Created {} collection", DATASETS_COLLECTION);
        Ok(())
    }

    /// Create default admin user if not exists
    pub async fn create_default_admin(db: &Database, password_hash: &str) -> Result<(), DbError> {
        // Check if admin exists
        let admin = db.doc_get(USERS_COLLECTION, "admin").await?;

        if admin.is_none() {
            let admin_data = serde_json::json!({
                "id": "admin",
                "email": "admin@axonml.local",
                "name": "Administrator",
                "password_hash": password_hash,
                "role": "admin",
                "mfa_enabled": false,
                "totp_secret": null,
                "webauthn_credentials": [],
                "recovery_codes": [],
                "email_pending": false,
                "email_verified": true,
                "verification_token": null,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "updated_at": chrono::Utc::now().to_rfc3339()
            });

            db.doc_insert(USERS_COLLECTION, Some("admin"), admin_data).await?;

            info!("Created default admin user");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_names() {
        assert_eq!(USERS_COLLECTION, "axonml_users");
        assert_eq!(RUNS_COLLECTION, "axonml_runs");
        assert_eq!(MODELS_COLLECTION, "axonml_models");
        assert_eq!(VERSIONS_COLLECTION, "axonml_model_versions");
        assert_eq!(ENDPOINTS_COLLECTION, "axonml_endpoints");
    }
}
