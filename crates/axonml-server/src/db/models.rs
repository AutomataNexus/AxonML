//! Model registry database operations for AxonML
//!
//! Uses Aegis-DB Document Store for models, versions, and endpoints.
//! Uses Aegis-DB Time Series for inference metrics.

use super::{Database, DbError, DocumentQuery, TimeSeriesQuery};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Collection names
const MODELS_COLLECTION: &str = "axonml_models";
const VERSIONS_COLLECTION: &str = "axonml_model_versions";
const ENDPOINTS_COLLECTION: &str = "axonml_endpoints";

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

    // ========================================================================
    // Model Operations
    // ========================================================================

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

        self.db.doc_insert(MODELS_COLLECTION, Some(&model.id), model_json).await?;

        Ok(model)
    }

    /// Find model by ID
    pub async fn find_by_id(&self, id: &str) -> Result<Option<Model>, DbError> {
        let doc = self.db.doc_get(MODELS_COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let model: Model = serde_json::from_value(data)?;
                Ok(Some(model))
            }
            None => Ok(None),
        }
    }

    /// List models for a user
    pub async fn list_by_user(
        &self,
        user_id: &str,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Model>, DbError> {
        let filter = serde_json::json!({
            "user_id": { "$eq": user_id }
        });

        let query = DocumentQuery {
            filter: Some(filter),
            sort: Some(serde_json::json!({ "field": "created_at", "ascending": false })),
            limit,
            skip: offset,
        };

        let docs = self.db.doc_query(MODELS_COLLECTION, query).await?;

        let mut models = Vec::new();
        for doc in docs {
            let model: Model = serde_json::from_value(doc)?;
            models.push(model);
        }

        Ok(models)
    }

    /// List all models
    pub async fn list_all(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Model>, DbError> {
        let query = DocumentQuery {
            filter: None,
            sort: Some(serde_json::json!({ "field": "created_at", "ascending": false })),
            limit,
            skip: offset,
        };

        let docs = self.db.doc_query(MODELS_COLLECTION, query).await?;

        let mut models = Vec::new();
        for doc in docs {
            let model: Model = serde_json::from_value(doc)?;
            models.push(model);
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

        self.db.doc_update(MODELS_COLLECTION, id, model_json).await?;

        Ok(model)
    }

    /// Delete model and all versions
    pub async fn delete(&self, id: &str) -> Result<(), DbError> {
        // Delete all versions first
        let versions = self.list_versions(id).await?;
        for version in versions {
            self.db.doc_delete(VERSIONS_COLLECTION, &version.id).await?;
        }

        // Delete the model
        self.db.doc_delete(MODELS_COLLECTION, id).await?;

        Ok(())
    }

    // ========================================================================
    // Model Version Operations
    // ========================================================================

    /// Create a new model version
    pub async fn create_version(&self, new_version: NewModelVersion) -> Result<ModelVersion, DbError> {
        // Get the next version number
        let versions = self.list_versions(&new_version.model_id).await?;
        let next_version = versions.iter()
            .map(|v| v.version)
            .max()
            .map(|v| v + 1)
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

        self.db.doc_insert(VERSIONS_COLLECTION, Some(&version.id), version_json).await?;

        Ok(version)
    }

    /// Get model version by ID
    pub async fn get_version(&self, id: &str) -> Result<Option<ModelVersion>, DbError> {
        let doc = self.db.doc_get(VERSIONS_COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let version: ModelVersion = serde_json::from_value(data)?;
                Ok(Some(version))
            }
            None => Ok(None),
        }
    }

    /// Get model version by model ID and version number
    pub async fn get_version_by_number(&self, model_id: &str, version: u32) -> Result<Option<ModelVersion>, DbError> {
        let filter = serde_json::json!({
            "model_id": { "$eq": model_id },
            "version": { "$eq": version }
        });

        let doc = self.db.doc_find_one(VERSIONS_COLLECTION, filter).await?;

        match doc {
            Some(data) => {
                let ver: ModelVersion = serde_json::from_value(data)?;
                Ok(Some(ver))
            }
            None => Ok(None),
        }
    }

    /// List versions for a model
    pub async fn list_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>, DbError> {
        let filter = serde_json::json!({
            "model_id": { "$eq": model_id }
        });

        let query = DocumentQuery {
            filter: Some(filter),
            sort: Some(serde_json::json!({ "field": "version", "ascending": false })),
            limit: None,
            skip: None,
        };

        let docs = self.db.doc_query(VERSIONS_COLLECTION, query).await?;

        let mut versions = Vec::new();
        for doc in docs {
            let version: ModelVersion = serde_json::from_value(doc)?;
            versions.push(version);
        }

        Ok(versions)
    }

    /// Delete a model version
    pub async fn delete_version(&self, id: &str) -> Result<(), DbError> {
        self.db.doc_delete(VERSIONS_COLLECTION, id).await
    }

    // ========================================================================
    // Endpoint Operations
    // ========================================================================

    /// Create an inference endpoint
    pub async fn create_endpoint(&self, new_endpoint: NewEndpoint) -> Result<Endpoint, DbError> {
        // Check if name already exists
        let existing = self.get_endpoint_by_name(&new_endpoint.name).await?;
        if existing.is_some() {
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

        self.db.doc_insert(ENDPOINTS_COLLECTION, Some(&endpoint.id), endpoint_json).await?;

        Ok(endpoint)
    }

    /// Get endpoint by ID
    pub async fn get_endpoint(&self, id: &str) -> Result<Option<Endpoint>, DbError> {
        let doc = self.db.doc_get(ENDPOINTS_COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let endpoint: Endpoint = serde_json::from_value(data)?;
                Ok(Some(endpoint))
            }
            None => Ok(None),
        }
    }

    /// Get endpoint by name
    pub async fn get_endpoint_by_name(&self, name: &str) -> Result<Option<Endpoint>, DbError> {
        let filter = serde_json::json!({
            "name": { "$eq": name }
        });

        let doc = self.db.doc_find_one(ENDPOINTS_COLLECTION, filter).await?;

        match doc {
            Some(data) => {
                let endpoint: Endpoint = serde_json::from_value(data)?;
                Ok(Some(endpoint))
            }
            None => Ok(None),
        }
    }

    /// List all endpoints
    pub async fn list_endpoints(&self) -> Result<Vec<Endpoint>, DbError> {
        let query = DocumentQuery {
            filter: None,
            sort: Some(serde_json::json!({ "field": "created_at", "ascending": false })),
            limit: None,
            skip: None,
        };

        let docs = self.db.doc_query(ENDPOINTS_COLLECTION, query).await?;

        let mut endpoints = Vec::new();
        for doc in docs {
            let endpoint: Endpoint = serde_json::from_value(doc)?;
            endpoints.push(endpoint);
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

        self.db.doc_update(ENDPOINTS_COLLECTION, id, endpoint_json).await?;

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

        self.db.doc_update(ENDPOINTS_COLLECTION, id, endpoint_json).await?;

        Ok(endpoint)
    }

    /// Delete endpoint
    pub async fn delete_endpoint(&self, id: &str) -> Result<(), DbError> {
        // Note: Time series inference metrics are retained for historical analysis
        self.db.doc_delete(ENDPOINTS_COLLECTION, id).await
    }

    // ========================================================================
    // Inference Metrics (Time Series)
    // ========================================================================

    /// Record inference metrics to time series
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
        let mut tags: HashMap<String, String> = HashMap::new();
        tags.insert("endpoint_id".to_string(), endpoint_id.to_string());

        // Record request metrics
        let mut req_tags = tags.clone();
        req_tags.insert("metric".to_string(), "requests_total".to_string());
        self.db.ts_write_one(
            &format!("axonml.inference.{}.requests_total", endpoint_id),
            requests_total as f64,
            req_tags
        ).await?;

        let mut success_tags = tags.clone();
        success_tags.insert("metric".to_string(), "requests_success".to_string());
        self.db.ts_write_one(
            &format!("axonml.inference.{}.requests_success", endpoint_id),
            requests_success as f64,
            success_tags
        ).await?;

        let mut error_tags = tags.clone();
        error_tags.insert("metric".to_string(), "requests_error".to_string());
        self.db.ts_write_one(
            &format!("axonml.inference.{}.requests_error", endpoint_id),
            requests_error as f64,
            error_tags
        ).await?;

        // Record latency metrics
        let mut p50_tags = tags.clone();
        p50_tags.insert("metric".to_string(), "latency_p50".to_string());
        self.db.ts_write_one(
            &format!("axonml.inference.{}.latency_p50", endpoint_id),
            latency_p50,
            p50_tags
        ).await?;

        let mut p95_tags = tags.clone();
        p95_tags.insert("metric".to_string(), "latency_p95".to_string());
        self.db.ts_write_one(
            &format!("axonml.inference.{}.latency_p95", endpoint_id),
            latency_p95,
            p95_tags
        ).await?;

        let mut p99_tags = tags.clone();
        p99_tags.insert("metric".to_string(), "latency_p99".to_string());
        self.db.ts_write_one(
            &format!("axonml.inference.{}.latency_p99", endpoint_id),
            latency_p99,
            p99_tags
        ).await?;

        Ok(())
    }

    /// Get inference metrics history from time series
    pub async fn get_inference_metrics(
        &self,
        endpoint_id: &str,
        limit: Option<u32>,
    ) -> Result<Vec<serde_json::Value>, DbError> {
        // Query latency_p50 as the primary metric
        let query = TimeSeriesQuery {
            metric: format!("axonml.inference.{}.latency_p50", endpoint_id),
            start: None,
            end: None,
            tags: None,
            aggregation: None,
            limit,
        };

        let points = self.db.ts_query(query).await?;

        // Convert to JSON for API compatibility
        let metrics: Vec<serde_json::Value> = points.into_iter().map(|p| {
            serde_json::json!({
                "latency_p50": p.value,
                "timestamp": p.timestamp.to_rfc3339()
            })
        }).collect();

        Ok(metrics)
    }

    /// Count running endpoints
    pub async fn count_running_endpoints(&self) -> Result<u64, DbError> {
        let filter = serde_json::json!({
            "status": { "$eq": "running" }
        });

        let query = DocumentQuery {
            filter: Some(filter),
            sort: None,
            limit: None,
            skip: None,
        };

        let docs = self.db.doc_query(ENDPOINTS_COLLECTION, query).await?;
        Ok(docs.len() as u64)
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
