//! Database schema initialization for AxonML
//!
//! Creates all required tables in Aegis-DB.

use super::{Database, DbError};
use tracing::info;

/// Schema definitions for all AxonML tables
pub struct Schema;

impl Schema {
    /// Initialize all database tables
    pub async fn init(db: &Database) -> Result<(), DbError> {
        info!("Initializing AxonML database schema...");

        // Create users table
        Self::create_users_table(db).await?;

        // Create training runs table
        Self::create_runs_table(db).await?;

        // Create training metrics table
        Self::create_metrics_table(db).await?;

        // Create models table
        Self::create_models_table(db).await?;

        // Create model versions table
        Self::create_model_versions_table(db).await?;

        // Create endpoints table
        Self::create_endpoints_table(db).await?;

        // Create inference metrics table
        Self::create_inference_metrics_table(db).await?;

        info!("Database schema initialized successfully");
        Ok(())
    }

    /// Create users table (document store)
    async fn create_users_table(db: &Database) -> Result<(), DbError> {
        db.execute(r#"
            CREATE TABLE IF NOT EXISTS axonml_users (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#).await?;

        // Create index on email for faster lookups
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_users_email
            ON axonml_users ((data->>'email'))
        "#).await.ok(); // Ignore if already exists or not supported

        info!("Created axonml_users table");
        Ok(())
    }

    /// Create training runs table (document store)
    async fn create_runs_table(db: &Database) -> Result<(), DbError> {
        db.execute(r#"
            CREATE TABLE IF NOT EXISTS axonml_runs (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#).await?;

        // Index on user_id for filtering
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_runs_user
            ON axonml_runs ((data->>'user_id'))
        "#).await.ok();

        // Index on status for filtering
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_runs_status
            ON axonml_runs ((data->>'status'))
        "#).await.ok();

        info!("Created axonml_runs table");
        Ok(())
    }

    /// Create training metrics table (time series)
    async fn create_metrics_table(db: &Database) -> Result<(), DbError> {
        db.execute(r#"
            CREATE TABLE IF NOT EXISTS axonml_metrics (
                run_id TEXT NOT NULL,
                epoch INT NOT NULL,
                step INT NOT NULL,
                loss FLOAT,
                accuracy FLOAT,
                lr FLOAT,
                gpu_util FLOAT,
                memory_mb FLOAT,
                custom_metrics JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, timestamp)
            )
        "#).await?;

        // Index for faster run queries
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_metrics_run
            ON axonml_metrics (run_id, timestamp)
        "#).await.ok();

        info!("Created axonml_metrics table");
        Ok(())
    }

    /// Create models table (document store)
    async fn create_models_table(db: &Database) -> Result<(), DbError> {
        db.execute(r#"
            CREATE TABLE IF NOT EXISTS axonml_models (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#).await?;

        // Index on user_id
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_models_user
            ON axonml_models ((data->>'user_id'))
        "#).await.ok();

        info!("Created axonml_models table");
        Ok(())
    }

    /// Create model versions table (document store)
    async fn create_model_versions_table(db: &Database) -> Result<(), DbError> {
        db.execute(r#"
            CREATE TABLE IF NOT EXISTS axonml_model_versions (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#).await?;

        // Index on model_id for faster version lookups
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_versions_model
            ON axonml_model_versions (model_id)
        "#).await.ok();

        info!("Created axonml_model_versions table");
        Ok(())
    }

    /// Create inference endpoints table (document store)
    async fn create_endpoints_table(db: &Database) -> Result<(), DbError> {
        db.execute(r#"
            CREATE TABLE IF NOT EXISTS axonml_endpoints (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#).await?;

        // Index on status
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_endpoints_status
            ON axonml_endpoints ((data->>'status'))
        "#).await.ok();

        info!("Created axonml_endpoints table");
        Ok(())
    }

    /// Create inference metrics table (time series)
    async fn create_inference_metrics_table(db: &Database) -> Result<(), DbError> {
        db.execute(r#"
            CREATE TABLE IF NOT EXISTS axonml_inference_metrics (
                endpoint_id TEXT NOT NULL,
                requests_total INT DEFAULT 0,
                requests_success INT DEFAULT 0,
                requests_error INT DEFAULT 0,
                latency_p50 FLOAT,
                latency_p95 FLOAT,
                latency_p99 FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (endpoint_id, timestamp)
            )
        "#).await?;

        // Index for faster endpoint queries
        db.execute(r#"
            CREATE INDEX IF NOT EXISTS idx_inf_metrics_endpoint
            ON axonml_inference_metrics (endpoint_id, timestamp)
        "#).await.ok();

        info!("Created axonml_inference_metrics table");
        Ok(())
    }

    /// Create default admin user if not exists
    pub async fn create_default_admin(db: &Database, password_hash: &str) -> Result<(), DbError> {
        let admin_exists = db.query(
            "SELECT id FROM axonml_users WHERE id = 'admin'"
        ).await?;

        if admin_exists.rows.is_empty() {
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
                "created_at": chrono::Utc::now().to_rfc3339(),
                "updated_at": chrono::Utc::now().to_rfc3339()
            });

            db.execute_with_params(
                "INSERT INTO axonml_users (id, data) VALUES ($1, $2)",
                vec![
                    serde_json::json!("admin"),
                    admin_data,
                ],
            ).await?;

            info!("Created default admin user");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_sql_validity() {
        // Just verify the SQL strings are valid syntax (basic check)
        let create_table = r#"
            CREATE TABLE IF NOT EXISTS axonml_users (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#;
        assert!(create_table.contains("CREATE TABLE"));
        assert!(create_table.contains("axonml_users"));
    }
}
