//! Built-in Datasets API endpoints for AxonML
//!
//! Provides access to built-in datasets and NexusConnectBridge integration.

use axum::{
    extract::{Path, Query, State},
    Json,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};

// ============================================================================
// Constants
// ============================================================================

// NexusConnectBridge URLs (reserved for future integration)
#[allow(dead_code)]
const NEXUS_API_URL: &str = "https://nexusconnectbridge.automatanexus.com/api/v1/bridge/datasets";
#[allow(dead_code)]
const NEXUS_FALLBACK_URL: &str = "http://100.85.154.94:8000/api/v1/bridge/datasets";

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinDataset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub num_samples: u64,
    pub num_features: u64,
    pub num_classes: u64,
    pub size_mb: f64,
    pub data_type: String,
    pub task_type: String,
    pub source: String,
    pub download_url: Option<String>,
    pub loading_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSource {
    pub id: String,
    pub name: String,
    pub description: String,
    pub dataset_count: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub name: String,
    pub source: String,
    pub size: String,
    pub download_count: u64,
    pub description: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListQuery {
    pub source: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    #[allow(dead_code)]
    pub source: Option<String>,
    pub limit: Option<usize>,
}

// ============================================================================
// Built-in Dataset Registry
// ============================================================================

fn get_builtin_datasets() -> Vec<BuiltinDataset> {
    vec![
        BuiltinDataset {
            id: "mnist".to_string(),
            name: "MNIST Handwritten Digits".to_string(),
            description: "Classic handwritten digit recognition dataset with 70,000 grayscale images".to_string(),
            num_samples: 70000,
            num_features: 784,
            num_classes: 10,
            size_mb: 50.0,
            data_type: "image".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_vision::datasets::MNIST;
let dataset = MNIST::new("./data", true)?;"#.to_string()),
        },
        BuiltinDataset {
            id: "fashion-mnist".to_string(),
            name: "Fashion MNIST".to_string(),
            description: "Fashion product images dataset - a more challenging drop-in replacement for MNIST".to_string(),
            num_samples: 70000,
            num_features: 784,
            num_classes: 10,
            size_mb: 50.0,
            data_type: "image".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_vision::datasets::FashionMNIST;
let dataset = FashionMNIST::new("./data", true)?;"#.to_string()),
        },
        BuiltinDataset {
            id: "cifar-10".to_string(),
            name: "CIFAR-10".to_string(),
            description: "60,000 32x32 color images in 10 classes".to_string(),
            num_samples: 60000,
            num_features: 3072,
            num_classes: 10,
            size_mb: 170.0,
            data_type: "image".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_vision::datasets::CIFAR10;
let dataset = CIFAR10::new("./data", true)?;"#.to_string()),
        },
        BuiltinDataset {
            id: "cifar-100".to_string(),
            name: "CIFAR-100".to_string(),
            description: "60,000 32x32 color images in 100 fine-grained classes".to_string(),
            num_samples: 60000,
            num_features: 3072,
            num_classes: 100,
            size_mb: 170.0,
            data_type: "image".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_vision::datasets::CIFAR100;
let dataset = CIFAR100::new("./data", true)?;"#.to_string()),
        },
        BuiltinDataset {
            id: "iris".to_string(),
            name: "Iris Flower Dataset".to_string(),
            description: "Classic dataset for classification - 150 samples with 4 features".to_string(),
            num_samples: 150,
            num_features: 4,
            num_classes: 3,
            size_mb: 0.005,
            data_type: "tabular".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_data::datasets::Iris;
let dataset = Iris::load()?;"#.to_string()),
        },
        BuiltinDataset {
            id: "wine-quality".to_string(),
            name: "Wine Quality Dataset".to_string(),
            description: "Wine quality ratings based on physicochemical properties".to_string(),
            num_samples: 6497,
            num_features: 11,
            num_classes: 10,
            size_mb: 0.4,
            data_type: "tabular".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_data::datasets::WineQuality;
let dataset = WineQuality::load()?;"#.to_string()),
        },
        BuiltinDataset {
            id: "breast-cancer".to_string(),
            name: "Breast Cancer Wisconsin".to_string(),
            description: "Diagnostic dataset for breast cancer classification".to_string(),
            num_samples: 569,
            num_features: 30,
            num_classes: 2,
            size_mb: 0.1,
            data_type: "tabular".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_data::datasets::BreastCancer;
let dataset = BreastCancer::load()?;"#.to_string()),
        },
        BuiltinDataset {
            id: "boston-housing".to_string(),
            name: "Boston Housing".to_string(),
            description: "Housing prices regression dataset".to_string(),
            num_samples: 506,
            num_features: 13,
            num_classes: 0,
            size_mb: 0.05,
            data_type: "tabular".to_string(),
            task_type: "regression".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_data::datasets::BostonHousing;
let dataset = BostonHousing::load()?;"#.to_string()),
        },
        BuiltinDataset {
            id: "imdb".to_string(),
            name: "IMDB Movie Reviews".to_string(),
            description: "Sentiment analysis dataset with 50,000 movie reviews".to_string(),
            num_samples: 50000,
            num_features: 0,
            num_classes: 2,
            size_mb: 80.0,
            data_type: "text".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_text::datasets::IMDB;
let dataset = IMDB::new("./data", true)?;"#.to_string()),
        },
        BuiltinDataset {
            id: "ag-news".to_string(),
            name: "AG News".to_string(),
            description: "News article classification dataset with 4 categories".to_string(),
            num_samples: 120000,
            num_features: 0,
            num_classes: 4,
            size_mb: 30.0,
            data_type: "text".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_text::datasets::AGNews;
let dataset = AGNews::new("./data", true)?;"#.to_string()),
        },
        BuiltinDataset {
            id: "speech-commands".to_string(),
            name: "Speech Commands".to_string(),
            description: "Audio dataset for keyword spotting with 35 commands".to_string(),
            num_samples: 105000,
            num_features: 16000,
            num_classes: 35,
            size_mb: 2400.0,
            data_type: "audio".to_string(),
            task_type: "classification".to_string(),
            source: "builtin".to_string(),
            download_url: None,
            loading_code: Some(r#"use axonml_audio::datasets::SpeechCommands;
let dataset = SpeechCommands::new("./data", true)?;"#.to_string()),
        },
    ]
}

fn get_dataset_sources() -> Vec<DatasetSource> {
    vec![
        DatasetSource {
            id: "builtin".to_string(),
            name: "Built-in Datasets".to_string(),
            description: "Pre-configured datasets included with AxonML".to_string(),
            dataset_count: "11".to_string(),
        },
        DatasetSource {
            id: "kaggle".to_string(),
            name: "Kaggle".to_string(),
            description: "Community-driven dataset repository".to_string(),
            dataset_count: "65,000+".to_string(),
        },
        DatasetSource {
            id: "uci".to_string(),
            name: "UCI ML Repository".to_string(),
            description: "Classic machine learning datasets".to_string(),
            dataset_count: "600+".to_string(),
        },
        DatasetSource {
            id: "data.gov".to_string(),
            name: "data.gov".to_string(),
            description: "US Government Open Data".to_string(),
            dataset_count: "250,000+".to_string(),
        },
        DatasetSource {
            id: "huggingface".to_string(),
            name: "Hugging Face".to_string(),
            description: "NLP and ML datasets hub".to_string(),
            dataset_count: "50,000+".to_string(),
        },
    ]
}

// ============================================================================
// Handlers
// ============================================================================

/// List available datasets
pub async fn list_datasets(
    State(_state): State<AppState>,
    _user: AuthUser,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<BuiltinDataset>>, AuthError> {
    // Return local datasets immediately - no external API calls
    let mut datasets = get_builtin_datasets();
    if let Some(source) = &query.source {
        datasets.retain(|d| d.source == *source);
    }
    Ok(Json(datasets))
}

/// Get dataset info
pub async fn get_dataset_info(
    State(_state): State<AppState>,
    _user: AuthUser,
    Path(dataset_id): Path<String>,
) -> Result<Json<BuiltinDataset>, AuthError> {
    // Return from local registry immediately
    let datasets = get_builtin_datasets();
    let dataset = datasets.into_iter()
        .find(|d| d.id == dataset_id)
        .ok_or_else(|| AuthError::NotFound(format!("Dataset '{}' not found", dataset_id)))?;

    Ok(Json(dataset))
}

/// Search datasets across sources
pub async fn search_datasets(
    State(_state): State<AppState>,
    _user: AuthUser,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<SearchResult>>, AuthError> {
    let limit = query.limit.unwrap_or(20);

    // Search local registry immediately
    let datasets = get_builtin_datasets();
    let query_lower = query.query.to_lowercase();

    let results: Vec<SearchResult> = datasets.into_iter()
        .filter(|d| {
            d.name.to_lowercase().contains(&query_lower) ||
            d.description.to_lowercase().contains(&query_lower) ||
            d.id.to_lowercase().contains(&query_lower)
        })
        .take(limit)
        .map(|d| SearchResult {
            id: d.id,
            name: d.name,
            source: d.source,
            size: format!("{:.1} MB", d.size_mb),
            download_count: 0,
            description: d.description,
        })
        .collect();

    Ok(Json(results))
}

/// List available dataset sources
pub async fn list_sources(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<Vec<DatasetSource>>, AuthError> {
    Ok(Json(get_dataset_sources()))
}

/// Download/prepare a built-in dataset
pub async fn prepare_dataset(
    State(_state): State<AppState>,
    _user: AuthUser,
    Path(dataset_id): Path<String>,
) -> Result<Json<BuiltinDataset>, AuthError> {
    let datasets = get_builtin_datasets();
    let dataset = datasets.into_iter()
        .find(|d| d.id == dataset_id)
        .ok_or_else(|| AuthError::NotFound(format!("Dataset '{}' not found", dataset_id)))?;

    // For built-in datasets, we return the loading code
    // The actual download happens client-side when the code is run

    Ok(Json(dataset))
}

// ============================================================================
// NexusConnectBridge Integration (reserved for future use)
// ============================================================================

#[allow(dead_code)]
async fn fetch_from_nexus(source: &Option<String>) -> Result<Vec<BuiltinDataset>, String> {
    let client = Client::new();

    let url = match source {
        Some(s) => format!("{}?source={}", NEXUS_API_URL, s),
        None => NEXUS_API_URL.to_string(),
    };

    let response = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        // Try fallback URL
        let fallback_url = match source {
            Some(s) => format!("{}?source={}", NEXUS_FALLBACK_URL, s),
            None => NEXUS_FALLBACK_URL.to_string(),
        };

        let response = client
            .get(&fallback_url)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            return Err("Both primary and fallback APIs failed".to_string());
        }

        return response.json().await.map_err(|e| e.to_string());
    }

    response.json().await.map_err(|e| e.to_string())
}

#[allow(dead_code)]
async fn fetch_dataset_info_from_nexus(dataset_id: &str) -> Result<BuiltinDataset, String> {
    let client = Client::new();

    let url = format!("{}/{}", NEXUS_API_URL, dataset_id);

    let response = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        return Err("API request failed".to_string());
    }

    response.json().await.map_err(|e| e.to_string())
}

#[allow(dead_code)]
async fn search_nexus(
    query: &str,
    source: &Option<String>,
    limit: usize,
) -> Result<Vec<SearchResult>, String> {
    let client = Client::new();

    let mut url = format!("{}/search?q={}&limit={}", NEXUS_API_URL, urlencoding::encode(query), limit);
    if let Some(s) = source {
        url.push_str(&format!("&source={}", s));
    }

    let response = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        return Err("Search request failed".to_string());
    }

    response.json().await.map_err(|e| e.to_string())
}
