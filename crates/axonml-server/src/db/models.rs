//! Model registry database operations for AxonML
//!
//! Provides CRUD operations for models, versions, and endpoints.

use super::{Database, DbError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Model data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub user_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub model_type: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// New model creation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewModel {
    pub user_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub model_type: String,
}

/// Model version data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: String,
    pub model_id: String,
    pub version: u32,
    pub file_path: String,
    pub file_size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_run_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// New model version data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewModelVersion {
    pub model_id: String,
    pub file_path: String,
    pub file_size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_run_id: Option<String>,
}

/// Inference endpoint status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EndpointStatus {
    Starting,
    Running,
    Stopped,
    Error,
}

impl Default for EndpointStatus {
    fn default() -> Self {
        EndpointStatus::Stopped
    }
}

/// Inference endpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub id: String,
    pub model_version_id: String,
    pub name: String,
    #[serde(default)]
    pub status: EndpointStatus,
    pub port: u16,
    #[serde(default = "default_replicas")]
    pub replicas: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

fn default_replicas() -> u32 {
    1
}

/// New endpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewEndpoint {
    pub model_version_id: String,
    pub name: String,
    pub port: u16,
    #[serde(default = "default_replicas")]
    pub replicas: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
}

/// Model repository
pub struct ModelRepository<'a> {
    db: &'a Database,
}

impl<'a> ModelRepository<'a> {
    /// Create a new model repository
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Create a new model
    pub async fn create(&self, new_model: NewModel) -> Result<Model, DbError> {
        let now = Utc::now();
        let model = Model {
            id: Uuid::new_v4().to_string(),
            user_id: new_model.user_id,
            name: new_model.name,
            description: new_model.description,
            model_type: new_model.model_type,
            created_at: now,
            updated_at: now,
        };

        let model_json = serde_json::to_value(&model)?;

        self.db.execute_with_params(
            "INSERT INTO axonml_models (id, data) VALUES ($1, $2)",
            vec![
                serde_json::json!(&model.id),
                model_json,
            ],
        ).await?;

        Ok(model)
    }

    /// Find model by ID
    pub async fn find_by_id(&self, id: &str) -> Result<Option<Model>, DbError> {
        let result = self.db.query_with_params(
            "SELECT data FROM axonml_models WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let data = result.rows[0].get("data")
            .ok_or_else(|| DbError::InvalidData("Missing data field".to_string()))?;

        let model: Model = serde_json::from_value(data.clone())?;
        Ok(Some(model))
    }

    /// List models for a user
    pub async fn list_by_user(
        &self,
        user_id: &str,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Model>, DbError> {
        let limit = limit.unwrap_or(100);
        let offset = offset.unwrap_or(0);

        let result = self.db.query_with_params(
            "SELECT data FROM axonml_models WHERE data->>'user_id' = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            vec![
                serde_json::json!(user_id),
                serde_json::json!(limit),
                serde_json::json!(offset),
            ],
        ).await?;

        let mut models = Vec::new();
        for row in result.rows {
            if let Some(data) = row.get("data") {
                let model: Model = serde_json::from_value(data.clone())?;
                models.push(model);
            }
        }

        Ok(models)
    }

    /// List all models
    pub async fn list_all(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Model>, DbError> {
        let limit = limit.unwrap_or(100);
        let offset = offset.unwrap_or(0);

        let result = self.db.query_with_params(
            "SELECT data FROM axonml_models ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            vec![
                serde_json::json!(limit),
                serde_json::json!(offset),
            ],
        ).await?;

        let mut models = Vec::new();
        for row in result.rows {
            if let Some(data) = row.get("data") {
                let model: Model = serde_json::from_value(data.clone())?;
                models.push(model);
            }
        }

        Ok(models)
    }

    /// Update model
    pub async fn update(&self, id: &str, name: Option<String>, description: Option<String>) -> Result<Model, DbError> {
        let mut model = self.find_by_id(id).await?
            .ok_or_else(|| DbError::NotFound(format!("Model {} not found", id)))?;

        if let Some(n) = name {
            model.name = n;
        }
        if let Some(d) = description {
            model.description = Some(d);
        }
        model.updated_at = Utc::now();

        let model_json = serde_json::to_value(&model)?;

        self.db.execute_with_params(
            "UPDATE axonml_models SET data = $2 WHERE id = $1",
            vec![
                serde_json::json!(id),
                model_json,
            ],
        ).await?;

        Ok(model)
    }

    /// Delete model and all versions
    pub async fn delete(&self, id: &str) -> Result<(), DbError> {
        // Delete all versions first
        self.db.execute_with_params(
            "DELETE FROM axonml_model_versions WHERE model_id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        // Delete the model
        let affected = self.db.execute_with_params(
            "DELETE FROM axonml_models WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if affected == 0 {
            return Err(DbError::NotFound(format!("Model {} not found", id)));
        }

        Ok(())
    }

    /// Create a new model version
    pub async fn create_version(&self, new_version: NewModelVersion) -> Result<ModelVersion, DbError> {
        // Get the next version number
        let result = self.db.query_with_params(
            "SELECT MAX((data->>'version')::int) as max_version FROM axonml_model_versions WHERE model_id = $1",
            vec![serde_json::json!(&new_version.model_id)],
        ).await?;

        let next_version = result.rows.first()
            .and_then(|r| r.get("max_version"))
            .and_then(|v| v.as_i64())
            .map(|v| v as u32 + 1)
            .unwrap_or(1);

        let version = ModelVersion {
            id: Uuid::new_v4().to_string(),
            model_id: new_version.model_id,
            version: next_version,
            file_path: new_version.file_path,
            file_size: new_version.file_size,
            metrics: new_version.metrics,
            training_run_id: new_version.training_run_id,
            created_at: Utc::now(),
        };

        let version_json = serde_json::to_value(&version)?;

        self.db.execute_with_params(
            "INSERT INTO axonml_model_versions (id, model_id, data) VALUES ($1, $2, $3)",
            vec![
                serde_json::json!(&version.id),
                serde_json::json!(&version.model_id),
                version_json,
            ],
        ).await?;

        Ok(version)
    }

    /// Get model version by ID
    pub async fn get_version(&self, id: &str) -> Result<Option<ModelVersion>, DbError> {
        let result = self.db.query_with_params(
            "SELECT data FROM axonml_model_versions WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let data = result.rows[0].get("data")
            .ok_or_else(|| DbError::InvalidData("Missing data field".to_string()))?;

        let version: ModelVersion = serde_json::from_value(data.clone())?;
        Ok(Some(version))
    }

    /// Get model version by model ID and version number
    pub async fn get_version_by_number(&self, model_id: &str, version: u32) -> Result<Option<ModelVersion>, DbError> {
        let result = self.db.query_with_params(
            "SELECT data FROM axonml_model_versions WHERE model_id = $1 AND (data->>'version')::int = $2",
            vec![
                serde_json::json!(model_id),
                serde_json::json!(version),
            ],
        ).await?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let data = result.rows[0].get("data")
            .ok_or_else(|| DbError::InvalidData("Missing data field".to_string()))?;

        let ver: ModelVersion = serde_json::from_value(data.clone())?;
        Ok(Some(ver))
    }

    /// List versions for a model
    pub async fn list_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>, DbError> {
        let result = self.db.query_with_params(
            "SELECT data FROM axonml_model_versions WHERE model_id = $1 ORDER BY (data->>'version')::int DESC",
            vec![serde_json::json!(model_id)],
        ).await?;

        let mut versions = Vec::new();
        for row in result.rows {
            if let Some(data) = row.get("data") {
                let version: ModelVersion = serde_json::from_value(data.clone())?;
                versions.push(version);
            }
        }

        Ok(versions)
    }

    /// Delete a model version
    pub async fn delete_version(&self, id: &str) -> Result<(), DbError> {
        let affected = self.db.execute_with_params(
            "DELETE FROM axonml_model_versions WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if affected == 0 {
            return Err(DbError::NotFound(format!("Version {} not found", id)));
        }

        Ok(())
    }

    /// Create an inference endpoint
    pub async fn create_endpoint(&self, new_endpoint: NewEndpoint) -> Result<Endpoint, DbError> {
        // Check if name already exists
        let existing = self.db.query_with_params(
            "SELECT id FROM axonml_endpoints WHERE data->>'name' = $1",
            vec![serde_json::json!(&new_endpoint.name)],
        ).await?;

        if !existing.rows.is_empty() {
            return Err(DbError::AlreadyExists(format!(
                "Endpoint with name {} already exists",
                new_endpoint.name
            )));
        }

        let now = Utc::now();
        let endpoint = Endpoint {
            id: Uuid::new_v4().to_string(),
            model_version_id: new_endpoint.model_version_id,
            name: new_endpoint.name,
            status: EndpointStatus::Stopped,
            port: new_endpoint.port,
            replicas: new_endpoint.replicas,
            config: new_endpoint.config,
            error_message: None,
            created_at: now,
            updated_at: now,
        };

        let endpoint_json = serde_json::to_value(&endpoint)?;

        self.db.execute_with_params(
            "INSERT INTO axonml_endpoints (id, data) VALUES ($1, $2)",
            vec![
                serde_json::json!(&endpoint.id),
                endpoint_json,
            ],
        ).await?;

        Ok(endpoint)
    }

    /// Get endpoint by ID
    pub async fn get_endpoint(&self, id: &str) -> Result<Option<Endpoint>, DbError> {
        let result = self.db.query_with_params(
            "SELECT data FROM axonml_endpoints WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let data = result.rows[0].get("data")
            .ok_or_else(|| DbError::InvalidData("Missing data field".to_string()))?;

        let endpoint: Endpoint = serde_json::from_value(data.clone())?;
        Ok(Some(endpoint))
    }

    /// Get endpoint by name
    pub async fn get_endpoint_by_name(&self, name: &str) -> Result<Option<Endpoint>, DbError> {
        let result = self.db.query_with_params(
            "SELECT data FROM axonml_endpoints WHERE data->>'name' = $1",
            vec![serde_json::json!(name)],
        ).await?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let data = result.rows[0].get("data")
            .ok_or_else(|| DbError::InvalidData("Missing data field".to_string()))?;

        let endpoint: Endpoint = serde_json::from_value(data.clone())?;
        Ok(Some(endpoint))
    }

    /// List all endpoints
    pub async fn list_endpoints(&self) -> Result<Vec<Endpoint>, DbError> {
        let result = self.db.query(
            "SELECT data FROM axonml_endpoints ORDER BY created_at DESC"
        ).await?;

        let mut endpoints = Vec::new();
        for row in result.rows {
            if let Some(data) = row.get("data") {
                let endpoint: Endpoint = serde_json::from_value(data.clone())?;
                endpoints.push(endpoint);
            }
        }

        Ok(endpoints)
    }

    /// Update endpoint status
    pub async fn update_endpoint_status(
        &self,
        id: &str,
        status: EndpointStatus,
        error_message: Option<String>,
    ) -> Result<Endpoint, DbError> {
        let mut endpoint = self.get_endpoint(id).await?
            .ok_or_else(|| DbError::NotFound(format!("Endpoint {} not found", id)))?;

        endpoint.status = status;
        endpoint.error_message = error_message;
        endpoint.updated_at = Utc::now();

        let endpoint_json = serde_json::to_value(&endpoint)?;

        self.db.execute_with_params(
            "UPDATE axonml_endpoints SET data = $2 WHERE id = $1",
            vec![
                serde_json::json!(id),
                endpoint_json,
            ],
        ).await?;

        Ok(endpoint)
    }

    /// Update endpoint configuration
    pub async fn update_endpoint(
        &self,
        id: &str,
        replicas: Option<u32>,
        config: Option<serde_json::Value>,
    ) -> Result<Endpoint, DbError> {
        let mut endpoint = self.get_endpoint(id).await?
            .ok_or_else(|| DbError::NotFound(format!("Endpoint {} not found", id)))?;

        if let Some(r) = replicas {
            endpoint.replicas = r;
        }
        if let Some(c) = config {
            endpoint.config = Some(c);
        }
        endpoint.updated_at = Utc::now();

        let endpoint_json = serde_json::to_value(&endpoint)?;

        self.db.execute_with_params(
            "UPDATE axonml_endpoints SET data = $2 WHERE id = $1",
            vec![
                serde_json::json!(id),
                endpoint_json,
            ],
        ).await?;

        Ok(endpoint)
    }

    /// Delete endpoint
    pub async fn delete_endpoint(&self, id: &str) -> Result<(), DbError> {
        // Delete associated metrics first
        self.db.execute_with_params(
            "DELETE FROM axonml_inference_metrics WHERE endpoint_id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        let affected = self.db.execute_with_params(
            "DELETE FROM axonml_endpoints WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if affected == 0 {
            return Err(DbError::NotFound(format!("Endpoint {} not found", id)));
        }

        Ok(())
    }

    /// Record inference metrics
    pub async fn record_inference_metrics(
        &self,
        endpoint_id: &str,
        requests_total: u64,
        requests_success: u64,
        requests_error: u64,
        latency_p50: f64,
        latency_p95: f64,
        latency_p99: f64,
    ) -> Result<(), DbError> {
        self.db.execute_with_params(
            r#"INSERT INTO axonml_inference_metrics
               (endpoint_id, requests_total, requests_success, requests_error, latency_p50, latency_p95, latency_p99, timestamp)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"#,
            vec![
                serde_json::json!(endpoint_id),
                serde_json::json!(requests_total),
                serde_json::json!(requests_success),
                serde_json::json!(requests_error),
                serde_json::json!(latency_p50),
                serde_json::json!(latency_p95),
                serde_json::json!(latency_p99),
                serde_json::json!(Utc::now().to_rfc3339()),
            ],
        ).await?;

        Ok(())
    }

    /// Get inference metrics history
    pub async fn get_inference_metrics(
        &self,
        endpoint_id: &str,
        limit: Option<u32>,
    ) -> Result<Vec<serde_json::Value>, DbError> {
        let limit = limit.unwrap_or(1000);

        let result = self.db.query_with_params(
            r#"SELECT requests_total, requests_success, requests_error, latency_p50, latency_p95, latency_p99, timestamp
               FROM axonml_inference_metrics
               WHERE endpoint_id = $1
               ORDER BY timestamp DESC
               LIMIT $2"#,
            vec![
                serde_json::json!(endpoint_id),
                serde_json::json!(limit),
            ],
        ).await?;

        Ok(result.rows)
    }

    /// Count running endpoints
    pub async fn count_running_endpoints(&self) -> Result<u64, DbError> {
        let result = self.db.query(
            "SELECT COUNT(*) as count FROM axonml_endpoints WHERE data->>'status' = 'running'"
        ).await?;

        if let Some(row) = result.rows.first() {
            if let Some(count) = row.get("count") {
                return Ok(count.as_u64().unwrap_or(0));
            }
        }

        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_serialization() {
        let model = Model {
            id: "model-123".to_string(),
            user_id: "user-456".to_string(),
            name: "Test Model".to_string(),
            description: Some("A test model".to_string()),
            model_type: "transformer".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("model-123"));
        assert!(json.contains("transformer"));
    }

    #[test]
    fn test_endpoint_status() {
        let status = EndpointStatus::Running;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"running\"");
    }
}
