//! Dataset database operations for AxonML
//!
//! Manages dataset metadata and file storage.

use super::{Database, DbError, DocumentQuery};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Collection name for datasets
const COLLECTION: &str = "axonml_datasets";

/// Dataset type enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DatasetType {
    Image,
    Tabular,
    Text,
    Audio,
    Custom,
}

impl Default for DatasetType {
    fn default() -> Self {
        DatasetType::Tabular
    }
}

/// Dataset data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub dataset_type: DatasetType,
    pub file_path: String,
    pub file_size: u64,
    pub num_samples: Option<u64>,
    pub num_features: Option<u64>,
    pub num_classes: Option<u64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// New dataset creation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewDataset {
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub dataset_type: DatasetType,
    pub file_path: String,
    pub file_size: u64,
    pub num_samples: Option<u64>,
    pub num_features: Option<u64>,
    pub num_classes: Option<u64>,
}

/// Dataset repository for database operations
pub struct DatasetRepository<'a> {
    db: &'a Database,
}

impl<'a> DatasetRepository<'a> {
    /// Create a new dataset repository
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Create a new dataset
    pub async fn create(&self, new_dataset: NewDataset) -> Result<Dataset, DbError> {
        let now = Utc::now();
        let dataset = Dataset {
            id: Uuid::new_v4().to_string(),
            user_id: new_dataset.user_id,
            name: new_dataset.name,
            description: new_dataset.description,
            dataset_type: new_dataset.dataset_type,
            file_path: new_dataset.file_path,
            file_size: new_dataset.file_size,
            num_samples: new_dataset.num_samples,
            num_features: new_dataset.num_features,
            num_classes: new_dataset.num_classes,
            created_at: now,
            updated_at: now,
        };

        let dataset_json = serde_json::to_value(&dataset)?;
        self.db.doc_insert(COLLECTION, Some(&dataset.id), dataset_json).await?;

        Ok(dataset)
    }

    /// Find dataset by ID
    pub async fn find_by_id(&self, id: &str) -> Result<Option<Dataset>, DbError> {
        let doc = self.db.doc_get(COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let dataset: Dataset = serde_json::from_value(data)?;
                Ok(Some(dataset))
            }
            None => Ok(None),
        }
    }

    /// Find all datasets for a user
    pub async fn find_by_user(&self, user_id: &str) -> Result<Vec<Dataset>, DbError> {
        let query = DocumentQuery {
            filter: Some(serde_json::json!({ "user_id": user_id })),
            ..Default::default()
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;

        let datasets: Vec<Dataset> = docs
            .into_iter()
            .filter_map(|d| serde_json::from_value(d).ok())
            .collect();

        Ok(datasets)
    }

    /// Delete a dataset
    pub async fn delete(&self, id: &str) -> Result<(), DbError> {
        self.db.doc_delete(COLLECTION, id).await
    }
}
