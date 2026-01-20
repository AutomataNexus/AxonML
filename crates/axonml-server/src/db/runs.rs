//! Training runs database operations for AxonML
//!
//! Provides CRUD operations for training runs and metrics.

use super::{Database, DbError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    #[serde(default)]
    pub optimizer: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Training run data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub model_type: String,
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
            status: RunStatus::Pending,
            config: new_run.config,
            latest_metrics: None,
            started_at: now,
            completed_at: None,
            created_at: now,
        };

        let run_json = serde_json::to_value(&run)?;

        self.db.execute_with_params(
            "INSERT INTO axonml_runs (id, data) VALUES ($1, $2)",
            vec![
                serde_json::json!(&run.id),
                run_json,
            ],
        ).await?;

        Ok(run)
    }

    /// Find run by ID
    pub async fn find_by_id(&self, id: &str) -> Result<Option<TrainingRun>, DbError> {
        let result = self.db.query_with_params(
            "SELECT data FROM axonml_runs WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let data = result.rows[0].get("data")
            .ok_or_else(|| DbError::InvalidData("Missing data field".to_string()))?;

        let run: TrainingRun = serde_json::from_value(data.clone())?;
        Ok(Some(run))
    }

    /// List runs for a user
    pub async fn list_by_user(
        &self,
        user_id: &str,
        status: Option<RunStatus>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<TrainingRun>, DbError> {
        let limit = limit.unwrap_or(100);
        let offset = offset.unwrap_or(0);

        let result = if let Some(status) = status {
            let status_str = serde_json::to_string(&status)?;
            let status_str = status_str.trim_matches('"');
            self.db.query_with_params(
                "SELECT data FROM axonml_runs WHERE data->>'user_id' = $1 AND data->>'status' = $2 ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                vec![
                    serde_json::json!(user_id),
                    serde_json::json!(status_str),
                    serde_json::json!(limit),
                    serde_json::json!(offset),
                ],
            ).await?
        } else {
            self.db.query_with_params(
                "SELECT data FROM axonml_runs WHERE data->>'user_id' = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                vec![
                    serde_json::json!(user_id),
                    serde_json::json!(limit),
                    serde_json::json!(offset),
                ],
            ).await?
        };

        let mut runs = Vec::new();
        for row in result.rows {
            if let Some(data) = row.get("data") {
                let run: TrainingRun = serde_json::from_value(data.clone())?;
                runs.push(run);
            }
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
        let limit = limit.unwrap_or(100);
        let offset = offset.unwrap_or(0);

        let result = if let Some(status) = status {
            let status_str = serde_json::to_string(&status)?;
            let status_str = status_str.trim_matches('"');
            self.db.query_with_params(
                "SELECT data FROM axonml_runs WHERE data->>'status' = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                vec![
                    serde_json::json!(status_str),
                    serde_json::json!(limit),
                    serde_json::json!(offset),
                ],
            ).await?
        } else {
            self.db.query_with_params(
                "SELECT data FROM axonml_runs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                vec![
                    serde_json::json!(limit),
                    serde_json::json!(offset),
                ],
            ).await?
        };

        let mut runs = Vec::new();
        for row in result.rows {
            if let Some(data) = row.get("data") {
                let run: TrainingRun = serde_json::from_value(data.clone())?;
                runs.push(run);
            }
        }

        Ok(runs)
    }

    /// Update run status
    pub async fn update_status(&self, id: &str, status: RunStatus) -> Result<TrainingRun, DbError> {
        let mut run = self.find_by_id(id).await?
            .ok_or_else(|| DbError::NotFound(format!("Run {} not found", id)))?;

        run.status = status.clone();
        if status == RunStatus::Completed || status == RunStatus::Failed || status == RunStatus::Stopped {
            run.completed_at = Some(Utc::now());
        }

        let run_json = serde_json::to_value(&run)?;

        self.db.execute_with_params(
            "UPDATE axonml_runs SET data = $2 WHERE id = $1",
            vec![
                serde_json::json!(id),
                run_json,
            ],
        ).await?;

        Ok(run)
    }

    /// Update latest metrics
    pub async fn update_metrics(&self, id: &str, metrics: TrainingMetrics) -> Result<TrainingRun, DbError> {
        let mut run = self.find_by_id(id).await?
            .ok_or_else(|| DbError::NotFound(format!("Run {} not found", id)))?;

        run.latest_metrics = Some(metrics);

        let run_json = serde_json::to_value(&run)?;

        self.db.execute_with_params(
            "UPDATE axonml_runs SET data = $2 WHERE id = $1",
            vec![
                serde_json::json!(id),
                run_json,
            ],
        ).await?;

        Ok(run)
    }

    /// Delete run
    pub async fn delete(&self, id: &str) -> Result<(), DbError> {
        // Delete associated metrics first
        self.db.execute_with_params(
            "DELETE FROM axonml_metrics WHERE run_id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        // Delete the run
        let affected = self.db.execute_with_params(
            "DELETE FROM axonml_runs WHERE id = $1",
            vec![serde_json::json!(id)],
        ).await?;

        if affected == 0 {
            return Err(DbError::NotFound(format!("Run {} not found", id)));
        }

        Ok(())
    }

    /// Record training metrics
    pub async fn record_metrics(&self, run_id: &str, metrics: &TrainingMetrics) -> Result<(), DbError> {
        let custom_json = serde_json::to_value(&metrics.custom)?;

        self.db.execute_with_params(
            r#"INSERT INTO axonml_metrics
               (run_id, epoch, step, loss, accuracy, lr, gpu_util, memory_mb, custom_metrics, timestamp)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"#,
            vec![
                serde_json::json!(run_id),
                serde_json::json!(metrics.epoch),
                serde_json::json!(metrics.step),
                serde_json::json!(metrics.loss),
                serde_json::json!(metrics.accuracy),
                serde_json::json!(metrics.lr),
                serde_json::json!(metrics.gpu_util),
                serde_json::json!(metrics.memory_mb),
                custom_json,
                serde_json::json!(metrics.timestamp.to_rfc3339()),
            ],
        ).await?;

        Ok(())
    }

    /// Get metrics history for a run
    pub async fn get_metrics_history(
        &self,
        run_id: &str,
        limit: Option<u32>,
    ) -> Result<Vec<TrainingMetrics>, DbError> {
        let limit = limit.unwrap_or(1000);

        let result = self.db.query_with_params(
            r#"SELECT epoch, step, loss, accuracy, lr, gpu_util, memory_mb, custom_metrics, timestamp
               FROM axonml_metrics
               WHERE run_id = $1
               ORDER BY timestamp ASC
               LIMIT $2"#,
            vec![
                serde_json::json!(run_id),
                serde_json::json!(limit),
            ],
        ).await?;

        let mut metrics = Vec::new();
        for row in result.rows {
            let m = TrainingMetrics {
                epoch: row.get("epoch").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                step: row.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                loss: row.get("loss").and_then(|v| v.as_f64()),
                accuracy: row.get("accuracy").and_then(|v| v.as_f64()),
                lr: row.get("lr").and_then(|v| v.as_f64()),
                gpu_util: row.get("gpu_util").and_then(|v| v.as_f64()),
                memory_mb: row.get("memory_mb").and_then(|v| v.as_f64()),
                custom: row.get("custom_metrics").cloned().unwrap_or(serde_json::json!({})),
                timestamp: row.get("timestamp")
                    .and_then(|v| v.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(Utc::now),
            };
            metrics.push(m);
        }

        Ok(metrics)
    }

    /// Get running runs count
    pub async fn count_running(&self) -> Result<u64, DbError> {
        let result = self.db.query(
            "SELECT COUNT(*) as count FROM axonml_runs WHERE data->>'status' = 'running'"
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
    fn test_run_serialization() {
        let run = TrainingRun {
            id: "run-123".to_string(),
            user_id: "user-456".to_string(),
            name: "Test Run".to_string(),
            model_type: "resnet".to_string(),
            status: RunStatus::Running,
            config: RunConfig {
                epochs: 10,
                batch_size: 32,
                learning_rate: 0.001,
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
