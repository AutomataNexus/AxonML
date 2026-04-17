//! Training Tracker — Metrics Persistence and Live Broadcast
//!
//! `TrainingTracker` is the fan-out layer between `TrainingExecutor` and the
//! WebSocket subscribers. It holds a `Database` handle plus a map of
//! `run_id -> tokio::sync::broadcast::Sender<TrainingMetrics>` (capacity 100
//! per run). `start_run` transitions the run to `RunStatus::Running` via
//! `RunRepository` and allocates a broadcaster. `record_metrics` stamps the
//! metrics with a UTC timestamp, writes both the time-series row and the
//! latest snapshot, and broadcasts the value to any subscribers.
//! `complete_run` / `stop_run` update the final status
//! (`Completed`/`Failed`/`Stopped`) and drop the broadcaster. `subscribe`
//! returns a `broadcast::Receiver`, and `is_tracking` reports whether a run
//! currently has a broadcaster registered.
//!
//! # File
//! `crates/axonml-server/src/training/tracker.rs`
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

// =============================================================================
// Imports
// =============================================================================

use crate::db::Database;
use crate::db::runs::{RunRepository, RunStatus, TrainingMetrics};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

// =============================================================================
// Tracker Struct
// =============================================================================

/// Training run tracker
pub struct TrainingTracker {
    db: Arc<Database>,
    broadcasters: Arc<RwLock<HashMap<String, broadcast::Sender<TrainingMetrics>>>>,
}

// =============================================================================
// Tracker Implementation
// =============================================================================

impl TrainingTracker {
    /// Create a new training tracker
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            broadcasters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // -------------------------------------------------------------------------
    // Run Lifecycle
    // -------------------------------------------------------------------------

    /// Start tracking a run
    pub async fn start_run(&self, run_id: &str) -> Result<(), String> {
        let repo = RunRepository::new(&self.db);

        // Update status to running
        repo.update_status(run_id, RunStatus::Running)
            .await
            .map_err(|e| e.to_string())?;

        // Create broadcaster for this run
        let (tx, _) = broadcast::channel(100);
        let mut broadcasters = self.broadcasters.write().await;
        broadcasters.insert(run_id.to_string(), tx);

        Ok(())
    }

    /// Complete a run
    pub async fn complete_run(&self, run_id: &str, success: bool) -> Result<(), String> {
        let repo = RunRepository::new(&self.db);

        let status = if success {
            RunStatus::Completed
        } else {
            RunStatus::Failed
        };

        repo.update_status(run_id, status)
            .await
            .map_err(|e| e.to_string())?;

        // Remove broadcaster
        let mut broadcasters = self.broadcasters.write().await;
        broadcasters.remove(run_id);

        Ok(())
    }

    /// Stop a run
    pub async fn stop_run(&self, run_id: &str) -> Result<(), String> {
        let repo = RunRepository::new(&self.db);

        repo.update_status(run_id, RunStatus::Stopped)
            .await
            .map_err(|e| e.to_string())?;

        // Remove broadcaster
        let mut broadcasters = self.broadcasters.write().await;
        broadcasters.remove(run_id);

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Metrics Recording and Broadcast
    // -------------------------------------------------------------------------

    /// Record metrics for a run
    pub async fn record_metrics(
        &self,
        run_id: &str,
        epoch: u32,
        step: u32,
        loss: Option<f64>,
        accuracy: Option<f64>,
        lr: Option<f64>,
        gpu_util: Option<f64>,
        memory_mb: Option<f64>,
        custom: serde_json::Value,
    ) -> Result<(), String> {
        let metrics = TrainingMetrics {
            epoch,
            step,
            loss,
            accuracy,
            lr,
            gpu_util,
            memory_mb,
            custom: custom.clone(),
            timestamp: Utc::now(),
        };

        let repo = RunRepository::new(&self.db);

        // Record to time series
        repo.record_metrics(run_id, &metrics)
            .await
            .map_err(|e| e.to_string())?;

        // Update latest metrics
        repo.update_metrics(run_id, metrics.clone())
            .await
            .map_err(|e| e.to_string())?;

        // Broadcast to subscribers
        let broadcasters = self.broadcasters.read().await;
        if let Some(tx) = broadcasters.get(run_id) {
            let _ = tx.send(metrics);
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Subscription and Status
    // -------------------------------------------------------------------------

    /// Subscribe to metrics for a run
    pub async fn subscribe(&self, run_id: &str) -> Option<broadcast::Receiver<TrainingMetrics>> {
        let broadcasters = self.broadcasters.read().await;
        broadcasters.get(run_id).map(|tx| tx.subscribe())
    }

    /// Check if a run is being tracked
    pub async fn is_tracking(&self, run_id: &str) -> bool {
        let broadcasters = self.broadcasters.read().await;
        broadcasters.contains_key(run_id)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_structure() {
        let metrics = TrainingMetrics {
            epoch: 5,
            step: 1000,
            loss: Some(0.5),
            accuracy: Some(0.9),
            lr: Some(0.001),
            gpu_util: Some(0.8),
            memory_mb: Some(4096.0),
            custom: serde_json::json!({}),
            timestamp: Utc::now(),
        };

        assert_eq!(metrics.epoch, 5);
        assert_eq!(metrics.step, 1000);
    }
}
