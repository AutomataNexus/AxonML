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
pub const NOTEBOOKS_COLLECTION: &str = "axonml_notebooks";
pub const CHECKPOINTS_COLLECTION: &str = "axonml_checkpoints";

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
        Self::create_notebooks_collection(db).await?;
        Self::create_checkpoints_collection(db).await?;

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

    /// Create training notebooks collection
    async fn create_notebooks_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(NOTEBOOKS_COLLECTION).await?;
        info!("Created {} collection", NOTEBOOKS_COLLECTION);
        Ok(())
    }

    /// Create checkpoints collection
    async fn create_checkpoints_collection(db: &Database) -> Result<(), DbError> {
        db.create_collection(CHECKPOINTS_COLLECTION).await?;
        info!("Created {} collection", CHECKPOINTS_COLLECTION);
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

    /// Create DevOps admin user if not exists
    /// Username: Devops
    /// Email: DevOps@automatanexus.com
    /// Full Name: Andrew Jewell
    pub async fn create_devops_user(db: &Database) -> Result<(), DbError> {
        // Pre-computed Argon2 hash for "Invertedskynet2$"
        const DEVOPS_PASSWORD_HASH: &str = "$argon2id$v=19$m=19456,t=2,p=1$acr9WUuS7lg2yoi8AHZAOQ$JsbYql+uEabmalV21GLetVjDZ3Q4MImyqXEx77nOlfM";

        // Check if DevOps user exists by email using filter query
        let filter = serde_json::json!({
            "email": { "$eq": "DevOps@automatanexus.com" }
        });
        let existing = db.doc_find_one(USERS_COLLECTION, filter).await?;

        if existing.is_none() {
            let user_id = uuid::Uuid::new_v4().to_string();
            let devops_data = serde_json::json!({
                "id": user_id,
                "email": "DevOps@automatanexus.com",
                "name": "Andrew Jewell",
                "password_hash": DEVOPS_PASSWORD_HASH,
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

            db.doc_insert(USERS_COLLECTION, Some(&user_id), devops_data).await?;

            info!("Created DevOps admin user (DevOps@automatanexus.com)");
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
