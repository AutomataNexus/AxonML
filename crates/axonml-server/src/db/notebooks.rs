//! Training notebooks database operations for AxonML
//!
//! Uses Aegis-DB Document Store for notebook data and checkpoints.

use super::{Database, DbError, DocumentQuery};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Collection name for notebooks
const COLLECTION: &str = "axonml_notebooks";

/// Collection name for checkpoints
const CHECKPOINTS_COLLECTION: &str = "axonml_checkpoints";

/// Cell type enum
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CellType {
    Code,
    Markdown,
}

impl Default for CellType {
    fn default() -> Self {
        Self::Code
    }
}

/// Cell execution status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CellStatus {
    Idle,
    Running,
    Completed,
    Error,
}

impl Default for CellStatus {
    fn default() -> Self {
        Self::Idle
    }
}

/// Notebook status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum NotebookStatus {
    Draft,
    Running,
    Completed,
    Failed,
    Stopped,
}

impl Default for NotebookStatus {
    fn default() -> Self {
        Self::Draft
    }
}

/// Cell output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellOutput {
    pub output_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceback: Option<Vec<String>>,
}

/// Notebook cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookCell {
    pub id: String,
    #[serde(default)]
    pub cell_type: CellType,
    pub source: String,
    #[serde(default)]
    pub outputs: Vec<CellOutput>,
    #[serde(default)]
    pub status: CellStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_count: Option<u32>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for NotebookCell {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cell_type: CellType::Code,
            source: String::new(),
            outputs: vec![],
            status: CellStatus::Idle,
            execution_count: None,
            metadata: HashMap::new(),
        }
    }
}

/// Notebook metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotebookMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Training notebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingNotebook {
    pub id: String,
    pub user_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub cells: Vec<NotebookCell>,
    #[serde(default)]
    pub metadata: NotebookMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    #[serde(default)]
    pub status: NotebookStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookCheckpoint {
    pub id: String,
    pub notebook_id: String,
    pub epoch: u32,
    pub step: u32,
    pub metrics: serde_json::Value,
    pub model_state_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimizer_state_path: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// New notebook data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewNotebook {
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub cells: Vec<NotebookCell>,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
}

/// New checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewCheckpoint {
    pub notebook_id: String,
    pub epoch: u32,
    pub step: u32,
    pub metrics: serde_json::Value,
    pub model_state_path: String,
    pub optimizer_state_path: Option<String>,
}

/// Notebook repository
pub struct NotebookRepository<'a> {
    db: &'a Database,
}

impl<'a> NotebookRepository<'a> {
    /// Create a new notebook repository
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Create a new training notebook
    pub async fn create(&self, new_notebook: NewNotebook) -> Result<TrainingNotebook, DbError> {
        let now = Utc::now();
        let notebook = TrainingNotebook {
            id: Uuid::new_v4().to_string(),
            user_id: new_notebook.user_id,
            name: new_notebook.name,
            description: new_notebook.description,
            cells: if new_notebook.cells.is_empty() {
                // Create default starter cells
                vec![
                    NotebookCell {
                        id: Uuid::new_v4().to_string(),
                        cell_type: CellType::Markdown,
                        source: "# Training Notebook\n\nDescribe your training experiment here.".to_string(),
                        ..Default::default()
                    },
                    NotebookCell {
                        id: Uuid::new_v4().to_string(),
                        cell_type: CellType::Code,
                        source: "# Import AxonML\nimport axonml\nfrom axonml import nn, optim, data\n\nprint(f\"AxonML version: {axonml.__version__}\")".to_string(),
                        ..Default::default()
                    },
                ]
            } else {
                new_notebook.cells
            },
            metadata: NotebookMetadata {
                kernel: Some("axonml".to_string()),
                language: Some("rust".to_string()),
                framework: Some("axonml".to_string()),
                tags: vec![],
                extra: HashMap::new(),
            },
            model_id: new_notebook.model_id,
            dataset_id: new_notebook.dataset_id,
            status: NotebookStatus::Draft,
            created_at: now,
            updated_at: now,
        };

        let notebook_json = serde_json::to_value(&notebook)?;
        self.db
            .doc_insert(COLLECTION, Some(&notebook.id), notebook_json)
            .await?;

        Ok(notebook)
    }

    /// Find notebook by ID
    pub async fn find_by_id(&self, id: &str) -> Result<Option<TrainingNotebook>, DbError> {
        let doc = self.db.doc_get(COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let notebook: TrainingNotebook = serde_json::from_value(data)?;
                Ok(Some(notebook))
            }
            None => Ok(None),
        }
    }

    /// List notebooks for a user
    pub async fn list_by_user(
        &self,
        user_id: &str,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<TrainingNotebook>, DbError> {
        let filter = serde_json::json!({
            "user_id": { "$eq": user_id }
        });

        let query = DocumentQuery {
            filter: Some(filter),
            sort: Some(serde_json::json!({ "field": "updated_at", "ascending": false })),
            limit,
            skip: offset,
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;

        let mut notebooks = Vec::new();
        for doc in docs {
            let notebook: TrainingNotebook = serde_json::from_value(doc)?;
            notebooks.push(notebook);
        }

        Ok(notebooks)
    }

    /// List all notebooks (admin)
    pub async fn list_all(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<TrainingNotebook>, DbError> {
        let query = DocumentQuery {
            filter: None,
            sort: Some(serde_json::json!({ "field": "updated_at", "ascending": false })),
            limit,
            skip: offset,
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;

        let mut notebooks = Vec::new();
        for doc in docs {
            let notebook: TrainingNotebook = serde_json::from_value(doc)?;
            notebooks.push(notebook);
        }

        Ok(notebooks)
    }

    /// Update notebook
    pub async fn update(
        &self,
        id: &str,
        updates: UpdateNotebook,
    ) -> Result<TrainingNotebook, DbError> {
        let mut notebook = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Notebook {} not found", id)))?;

        if let Some(name) = updates.name {
            notebook.name = name;
        }
        if let Some(description) = updates.description {
            notebook.description = Some(description);
        }
        if let Some(cells) = updates.cells {
            notebook.cells = cells;
        }
        if let Some(model_id) = updates.model_id {
            notebook.model_id = Some(model_id);
        }
        if let Some(dataset_id) = updates.dataset_id {
            notebook.dataset_id = Some(dataset_id);
        }
        if let Some(status) = updates.status {
            notebook.status = status;
        }

        notebook.updated_at = Utc::now();

        let notebook_json = serde_json::to_value(&notebook)?;
        self.db.doc_update(COLLECTION, id, notebook_json).await?;

        Ok(notebook)
    }

    /// Update a single cell in a notebook
    pub async fn update_cell(
        &self,
        notebook_id: &str,
        cell: NotebookCell,
    ) -> Result<TrainingNotebook, DbError> {
        let mut notebook = self
            .find_by_id(notebook_id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Notebook {} not found", notebook_id)))?;

        // Find and update the cell
        let mut found = false;
        for c in &mut notebook.cells {
            if c.id == cell.id {
                *c = cell.clone();
                found = true;
                break;
            }
        }

        if !found {
            return Err(DbError::NotFound(format!(
                "Cell {} not found in notebook",
                cell.id
            )));
        }

        notebook.updated_at = Utc::now();

        let notebook_json = serde_json::to_value(&notebook)?;
        self.db
            .doc_update(COLLECTION, notebook_id, notebook_json)
            .await?;

        Ok(notebook)
    }

    /// Add a cell to a notebook
    pub async fn add_cell(
        &self,
        notebook_id: &str,
        cell: NotebookCell,
        position: Option<usize>,
    ) -> Result<TrainingNotebook, DbError> {
        let mut notebook = self
            .find_by_id(notebook_id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Notebook {} not found", notebook_id)))?;

        match position {
            Some(pos) if pos < notebook.cells.len() => {
                notebook.cells.insert(pos, cell);
            }
            _ => {
                notebook.cells.push(cell);
            }
        }

        notebook.updated_at = Utc::now();

        let notebook_json = serde_json::to_value(&notebook)?;
        self.db
            .doc_update(COLLECTION, notebook_id, notebook_json)
            .await?;

        Ok(notebook)
    }

    /// Delete a cell from a notebook
    pub async fn delete_cell(
        &self,
        notebook_id: &str,
        cell_id: &str,
    ) -> Result<TrainingNotebook, DbError> {
        let mut notebook = self
            .find_by_id(notebook_id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Notebook {} not found", notebook_id)))?;

        let original_len = notebook.cells.len();
        notebook.cells.retain(|c| c.id != cell_id);

        if notebook.cells.len() == original_len {
            return Err(DbError::NotFound(format!(
                "Cell {} not found in notebook",
                cell_id
            )));
        }

        notebook.updated_at = Utc::now();

        let notebook_json = serde_json::to_value(&notebook)?;
        self.db
            .doc_update(COLLECTION, notebook_id, notebook_json)
            .await?;

        Ok(notebook)
    }

    /// Update notebook status
    pub async fn update_status(
        &self,
        id: &str,
        status: NotebookStatus,
    ) -> Result<TrainingNotebook, DbError> {
        self.update(
            id,
            UpdateNotebook {
                status: Some(status),
                ..Default::default()
            },
        )
        .await
    }

    /// Delete notebook
    pub async fn delete(&self, id: &str) -> Result<(), DbError> {
        // Check if notebook exists
        let _ = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("Notebook {} not found", id)))?;

        // Delete all associated checkpoints
        self.delete_checkpoints_for_notebook(id).await?;

        // Delete the notebook
        self.db.doc_delete(COLLECTION, id).await?;

        Ok(())
    }

    // ========================================================================
    // Checkpoint Operations
    // ========================================================================

    /// Create a checkpoint
    pub async fn create_checkpoint(
        &self,
        new_checkpoint: NewCheckpoint,
    ) -> Result<NotebookCheckpoint, DbError> {
        let checkpoint = NotebookCheckpoint {
            id: Uuid::new_v4().to_string(),
            notebook_id: new_checkpoint.notebook_id,
            epoch: new_checkpoint.epoch,
            step: new_checkpoint.step,
            metrics: new_checkpoint.metrics,
            model_state_path: new_checkpoint.model_state_path,
            optimizer_state_path: new_checkpoint.optimizer_state_path,
            created_at: Utc::now(),
        };

        let checkpoint_json = serde_json::to_value(&checkpoint)?;
        self.db
            .doc_insert(
                CHECKPOINTS_COLLECTION,
                Some(&checkpoint.id),
                checkpoint_json,
            )
            .await?;

        Ok(checkpoint)
    }

    /// Get checkpoint by ID
    pub async fn get_checkpoint(&self, id: &str) -> Result<Option<NotebookCheckpoint>, DbError> {
        let doc = self.db.doc_get(CHECKPOINTS_COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let checkpoint: NotebookCheckpoint = serde_json::from_value(data)?;
                Ok(Some(checkpoint))
            }
            None => Ok(None),
        }
    }

    /// List checkpoints for a notebook
    pub async fn list_checkpoints(
        &self,
        notebook_id: &str,
    ) -> Result<Vec<NotebookCheckpoint>, DbError> {
        let filter = serde_json::json!({
            "notebook_id": { "$eq": notebook_id }
        });

        let query = DocumentQuery {
            filter: Some(filter),
            sort: Some(serde_json::json!({ "field": "created_at", "ascending": false })),
            limit: None,
            skip: None,
        };

        let docs = self.db.doc_query(CHECKPOINTS_COLLECTION, query).await?;

        let mut checkpoints = Vec::new();
        for doc in docs {
            let checkpoint: NotebookCheckpoint = serde_json::from_value(doc)?;
            checkpoints.push(checkpoint);
        }

        Ok(checkpoints)
    }

    /// Get best checkpoint by metric
    pub async fn get_best_checkpoint(
        &self,
        notebook_id: &str,
        metric_key: &str,
        minimize: bool,
    ) -> Result<Option<NotebookCheckpoint>, DbError> {
        let checkpoints = self.list_checkpoints(notebook_id).await?;

        let best = checkpoints
            .into_iter()
            .filter_map(|cp| {
                let value = cp.metrics.get(metric_key)?.as_f64()?;
                Some((cp, value))
            })
            .reduce(|a, b| {
                if minimize {
                    if a.1 < b.1 {
                        a
                    } else {
                        b
                    }
                } else {
                    if a.1 > b.1 {
                        a
                    } else {
                        b
                    }
                }
            })
            .map(|(cp, _)| cp);

        Ok(best)
    }

    /// Delete checkpoint
    pub async fn delete_checkpoint(&self, id: &str) -> Result<(), DbError> {
        self.db.doc_delete(CHECKPOINTS_COLLECTION, id).await
    }

    /// Delete all checkpoints for a notebook
    async fn delete_checkpoints_for_notebook(&self, notebook_id: &str) -> Result<(), DbError> {
        let checkpoints = self.list_checkpoints(notebook_id).await?;
        for checkpoint in checkpoints {
            self.delete_checkpoint(&checkpoint.id).await?;
        }
        Ok(())
    }
}

/// Update notebook data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateNotebook {
    pub name: Option<String>,
    pub description: Option<String>,
    pub cells: Option<Vec<NotebookCell>>,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
    pub status: Option<NotebookStatus>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notebook_serialization() {
        let notebook = TrainingNotebook {
            id: "nb-123".to_string(),
            user_id: "user-456".to_string(),
            name: "Test Notebook".to_string(),
            description: Some("A test notebook".to_string()),
            cells: vec![NotebookCell {
                id: "cell-1".to_string(),
                cell_type: CellType::Markdown,
                source: "# Hello".to_string(),
                ..Default::default()
            }],
            metadata: NotebookMetadata::default(),
            model_id: None,
            dataset_id: None,
            status: NotebookStatus::Draft,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let json = serde_json::to_string(&notebook).unwrap();
        assert!(json.contains("nb-123"));
        assert!(json.contains("\"status\":\"draft\""));
    }

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = NotebookCheckpoint {
            id: "cp-123".to_string(),
            notebook_id: "nb-456".to_string(),
            epoch: 10,
            step: 1000,
            metrics: serde_json::json!({"loss": 0.234, "accuracy": 0.89}),
            model_state_path: "/checkpoints/cp-123/model.bin".to_string(),
            optimizer_state_path: Some("/checkpoints/cp-123/optimizer.bin".to_string()),
            created_at: Utc::now(),
        };

        let json = serde_json::to_string(&checkpoint).unwrap();
        assert!(json.contains("cp-123"));
        assert!(json.contains("0.234"));
    }
}
