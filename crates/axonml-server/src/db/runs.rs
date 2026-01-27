//! Training runs database operations for AxonML
//!
//! Uses Aegis-DB Document Store for run metadata and Time Series for metrics.

use super::{Database, DbError, DocumentQuery, TimeSeriesQuery};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Collection name for runs
const COLLECTION: &str = "axonml_runs";

/// Training run status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Stopped,
}

impl Default for RunStatus {
    fn default() -> Self {
        RunStatus::Pending
    }
}

/// Training run configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    #[serde(default = "default_steps_per_epoch")]
    pub steps_per_epoch: u32,
    #[serde(default)]
    pub optimizer: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

fn default_steps_per_epoch() -> u32 {
    100
}

/// Training run data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub model_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    #[serde(default)]
    pub status: RunStatus,
    pub config: RunConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_metrics: Option<TrainingMetrics>,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Training metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: u32,
    pub step: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_util: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<f64>,
    #[serde(default)]
    pub custom: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

/// New training run data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewTrainingRun {
    pub user_id: String,
    pub name: String,
    pub model_type: String,
    pub model_version_id: Option<String>,
    pub dataset_id: Option<String>,
    pub config: RunConfig,
}

/// Training run repository
pub struct RunRepository<'a> {
    db: &'a Database,
}

impl<'a> RunRepository<'a> {
    /// Create a new run repository
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Create a new training run
    pub async fn create(&self, new_run: NewTrainingRun) -> Result<TrainingRun, DbError> {
        let now = Utc::now();
        let run = TrainingRun {
            id: Uuid::new_v4().to_string(),
            user_id: new_run.user_id,
            name: new_run.name,
            model_type: new_run.model_type,
            model_version_id: new_run.model_version_id,
            dataset_id: new_run.dataset_id,
            status: RunStatus::Pending,
            config: new_run.config,
            latest_metrics: None,
            started_at: now,
            completed_at: None,
            created_at: now,
        };

        let run_json = serde_json::to_value(&run)?;

        // Insert using document store
        self.db
            .doc_insert(COLLECTION, Some(&run.id), run_json)
            .await?;

        Ok(run)
    }

    /// Find run by ID
    pub async fn find_by_id(&self, id: &str) -> Result<Option<TrainingRun>, DbError> {
        let doc = self.db.doc_get(COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let run: TrainingRun = serde_json::from_value(data)?;
                Ok(Some(run))
            }
            None => Ok(None),
        }
    }

    /// List runs for a user
    pub async fn list_by_user(
        &self,
        user_id: &str,
        status: Option<RunStatus>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<TrainingRun>, DbError> {
        let mut filter = serde_json::json!({
            "user_id": { "$eq": user_id }
        });

        if let Some(s) = status {
            let status_str = serde_json::to_string(&s)?;
            let status_str = status_str.trim_matches('"');
            // filter is guaranteed to be an object since we created it with json!({...})
            if let Some(obj) = filter.as_object_mut() {
                obj.insert(
                    "status".to_string(),
                    serde_json::json!({ "$eq": status_str }),
                );
            }
        }

        let query = DocumentQuery {
            filter: Some(filter),
            sort: Some(serde_json::json!({ "field": "created_at", "ascending": false })),
            limit,
            skip: offset,
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;

        let mut runs = Vec::new();
        for doc in docs {
            let run: TrainingRun = serde_json::from_value(doc)?;
            runs.push(run);
        }

        Ok(runs)
    }

    /// List all runs
    pub async fn list_all(
        &self,
        status: Option<RunStatus>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<TrainingRun>, DbError> {
        let filter = if let Some(s) = status {
            let status_str = serde_json::to_string(&s)?;
            let status_str = status_str.trim_matches('"');
            Some(serde_json::json!({
                "status": { "$eq": status_str }
            }))
        } else {
            None
        };

        let query = DocumentQuery {
            filter,
            sort: Some(serde_json::json!({ "field": "created_at", "ascending": false })),
            limit,
            skip: offset,
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;

        let mut runs = Vec::new();
        for doc in docs {
            let run: TrainingRun = serde_json::from_value(doc)?;
            runs.push(run);
        }

        Ok(runs)
    }

    /// Update run status
    pub async fn update_status(&self, id: &str, status: RunStatus) -> Result<TrainingRun, DbError> {
        let mut run = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Run {} not found", id)))?;

        run.status = status.clone();
        if status == RunStatus::Completed
            || status == RunStatus::Failed
            || status == RunStatus::Stopped
        {
            run.completed_at = Some(Utc::now());
        }

        let run_json = serde_json::to_value(&run)?;

        self.db.doc_update(COLLECTION, id, run_json).await?;

        Ok(run)
    }

    /// Update latest metrics (stored in document)
    pub async fn update_metrics(
        &self,
        id: &str,
        metrics: TrainingMetrics,
    ) -> Result<TrainingRun, DbError> {
        let mut run = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Run {} not found", id)))?;

        run.latest_metrics = Some(metrics);

        let run_json = serde_json::to_value(&run)?;

        self.db.doc_update(COLLECTION, id, run_json).await?;

        Ok(run)
    }

    /// Delete run
    pub async fn delete(&self, id: &str) -> Result<(), DbError> {
        // Check if run exists
        let _ = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Run {} not found", id)))?;

        // Note: Time series data is retained (metrics history remains for analysis)
        // Delete the run document
        self.db.doc_delete(COLLECTION, id).await?;

        Ok(())
    }

    /// Record training metrics to time series
    pub async fn record_metrics(
        &self,
        run_id: &str,
        metrics: &TrainingMetrics,
    ) -> Result<(), DbError> {
        // Create tags for this run
        let mut tags: HashMap<String, String> = HashMap::new();
        tags.insert("run_id".to_string(), run_id.to_string());
        tags.insert("epoch".to_string(), metrics.epoch.to_string());
        tags.insert("step".to_string(), metrics.step.to_string());

        // Record loss metric
        if let Some(loss) = metrics.loss {
            let mut loss_tags = tags.clone();
            loss_tags.insert("metric".to_string(), "loss".to_string());
            self.db
                .ts_write_one(&format!("axonml.training.{}.loss", run_id), loss, loss_tags)
                .await?;
        }

        // Record accuracy metric
        if let Some(accuracy) = metrics.accuracy {
            let mut acc_tags = tags.clone();
            acc_tags.insert("metric".to_string(), "accuracy".to_string());
            self.db
                .ts_write_one(
                    &format!("axonml.training.{}.accuracy", run_id),
                    accuracy,
                    acc_tags,
                )
                .await?;
        }

        // Record learning rate
        if let Some(lr) = metrics.lr {
            let mut lr_tags = tags.clone();
            lr_tags.insert("metric".to_string(), "learning_rate".to_string());
            self.db
                .ts_write_one(&format!("axonml.training.{}.lr", run_id), lr, lr_tags)
                .await?;
        }

        // Record GPU utilization
        if let Some(gpu_util) = metrics.gpu_util {
            let mut gpu_tags = tags.clone();
            gpu_tags.insert("metric".to_string(), "gpu_util".to_string());
            self.db
                .ts_write_one(
                    &format!("axonml.training.{}.gpu_util", run_id),
                    gpu_util,
                    gpu_tags,
                )
                .await?;
        }

        // Record memory usage
        if let Some(memory_mb) = metrics.memory_mb {
            let mut mem_tags = tags.clone();
            mem_tags.insert("metric".to_string(), "memory_mb".to_string());
            self.db
                .ts_write_one(
                    &format!("axonml.training.{}.memory_mb", run_id),
                    memory_mb,
                    mem_tags,
                )
                .await?;
        }

        Ok(())
    }

    /// Get metrics history for a run from time series
    pub async fn get_metrics_history(
        &self,
        run_id: &str,
        limit: Option<u32>,
    ) -> Result<Vec<TrainingMetrics>, DbError> {
        // Query loss metric time series
        let query = TimeSeriesQuery {
            metric: format!("axonml.training.{}.loss", run_id),
            start: None,
            end: None,
            tags: None,
            aggregation: None,
            limit,
        };

        let loss_points = self.db.ts_query(query).await?;

        // Convert time series points back to TrainingMetrics
        // This is a simplified version - in production you might want to join multiple metrics
        let mut metrics = Vec::new();
        for point in loss_points {
            let epoch = point
                .tags
                .get("epoch")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let step = point
                .tags
                .get("step")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            metrics.push(TrainingMetrics {
                epoch,
                step,
                loss: Some(point.value),
                accuracy: None, // Would need separate query to get this
                lr: None,
                gpu_util: None,
                memory_mb: None,
                custom: serde_json::json!({}),
                timestamp: point.timestamp,
            });
        }

        Ok(metrics)
    }

    /// Get running runs count
    pub async fn count_running(&self) -> Result<u64, DbError> {
        let filter = serde_json::json!({
            "status": { "$eq": "running" }
        });

        let query = DocumentQuery {
            filter: Some(filter),
            sort: None,
            limit: None,
            skip: None,
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;
        Ok(docs.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_serialization() {
        let run = TrainingRun {
            id: "run-123".to_string(),
            user_id: "user-456".to_string(),
            name: "Test Run".to_string(),
            model_type: "resnet".to_string(),
            model_version_id: None,
            dataset_id: None,
            status: RunStatus::Running,
            config: RunConfig {
                epochs: 10,
                batch_size: 32,
                learning_rate: 0.001,
                steps_per_epoch: 100,
                optimizer: "adam".to_string(),
                extra: serde_json::json!({}),
            },
            latest_metrics: None,
            started_at: Utc::now(),
            completed_at: None,
            created_at: Utc::now(),
        };

        let json = serde_json::to_string(&run).unwrap();
        assert!(json.contains("run-123"));
        assert!(json.contains("\"status\":\"running\""));
    }

    #[test]
    fn test_metrics_serialization() {
        let metrics = TrainingMetrics {
            epoch: 5,
            step: 1000,
            loss: Some(0.234),
            accuracy: Some(0.891),
            lr: Some(0.001),
            gpu_util: Some(0.85),
            memory_mb: Some(4096.0),
            custom: serde_json::json!({"custom_metric": 1.5}),
            timestamp: Utc::now(),
        };

        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("0.234"));
        assert!(json.contains("custom_metric"));
    }
}
