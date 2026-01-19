# Axonml ML Framework - NexusConnectBridge Dataset Integration

## Quick Start

### REST API Base URL
```
https://nexusconnectbridge.automatanexus.com/api/v1/bridge/datasets
```

Or via Tailscale:
```
http://100.85.154.94:8000/api/v1/bridge/datasets
```

---

## Built-in Datasets

### Endpoints

```rust
// List all built-in datasets
GET /builtin

// Get specific dataset metadata + loading code
GET /builtin/{dataset_id}

// Get PyTorch loading code (adapt for Rust)
GET /code/{dataset_id}
```

### Dataset IDs
- `mnist` - 70k samples, 784 features, 10 classes
- `fashion-mnist` - 70k samples, 784 features, 10 classes  
- `cifar-10` - 60k samples, 3072 features, 10 classes
- `cifar-100` - 60k samples, 3072 features, 100 classes
- `iris` - 150 samples, 4 features, 3 classes
- `wine-quality` - 6497 samples, 11 features, 10 classes
- `breast-cancer` - 569 samples, 30 features, 2 classes

### Rust Example (reqwest)

```rust
use reqwest;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct DatasetInfo {
    id: String,
    name: String,
    samples: u32,
    features: u32,
    classes: u32,
    format: String,
    input_shape: Vec<u32>,
}

async fn get_dataset(dataset_id: &str) -> Result<DatasetInfo, reqwest::Error> {
    let url = format!(
        "https://nexusconnectbridge.automatanexus.com/api/v1/bridge/datasets/builtin/{}",
        dataset_id
    );
    reqwest::get(&url).await?.json().await
}
```

---

## External Dataset Search

### Kaggle (65,000+ datasets)
```
GET /search?source=kaggle&query=energy&maxResults=50
```

### UCI ML Repository
```
GET /search?source=uci&query=classification
```

### data.gov
```
GET /search?source=data.gov&query=building
```

### All Sources
```
GET /search?source=all&query=timeseries&maxResults=100
```

---

## Upload Custom Datasets

```
POST /upload
Content-Type: multipart/form-data

file: <your_file.csv|json|parquet|zip>
user_id: 1
tier: pro
```

### Supported Formats
- CSV - Tabular data
- JSON - Array of objects or `{"data": [...]}`
- Parquet - Columnar format
- ZIP - Image datasets (folder per class)

### Upload Limits by Tier
| Tier | Max Size |
|------|----------|
| free | 10 MB |
| basic | 50 MB |
| pro | 500 MB |
| enterprise | 2 GB |

---

## Direct File Access (Server-side)

If running on the same network or server:

```
/opt/NexusConnectBridge/uploads/datasets/     # User uploads
/opt/NexusConnectBridge/models/compressors/   # Compression dicts (zstd)
/opt/NexusConnectBridge/data/                 # Analytics data
```

### Compression Dictionaries (zstd format)
```
models/compressors/conversation_dict.zdict
models/compressors/model_config_dict.zdict
models/compressors/metrics_dict.zdict
models/compressors/visualization_dict.zdict
```

---

## Rust Integration Patterns

### Pattern 1: HTTP Client

```rust
// Cargo.toml
[dependencies]
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

// src/datasets.rs
pub struct NexusDatasetClient {
    base_url: String,
    client: reqwest::Client,
}

impl NexusDatasetClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://nexusconnectbridge.automatanexus.com/api/v1/bridge/datasets".into(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn list_builtin(&self) -> Result<Vec<DatasetInfo>, Error> {
        let resp = self.client
            .get(format!("{}/builtin", self.base_url))
            .send()
            .await?;
        resp.json().await
    }

    pub async fn search(&self, query: &str, source: Option<&str>) -> Result<SearchResults, Error> {
        let mut url = format!("{}/search?query={}", self.base_url, query);
        if let Some(src) = source {
            url.push_str(&format!("&source={}", src));
        }
        self.client.get(&url).send().await?.json().await
    }
}
```

### Pattern 2: Direct MNIST Loading (Pure Rust)

```rust
// For local/offline use - download IDX files directly
use std::fs::File;
use std::io::{BufReader, Read};
use byteorder::{BigEndian, ReadBytesExt};

pub fn load_mnist_images(path: &str) -> Vec<Vec<f32>> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    
    let magic = reader.read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 2051); // Images magic number
    
    let num_images = reader.read_u32::<BigEndian>().unwrap() as usize;
    let rows = reader.read_u32::<BigEndian>().unwrap() as usize;
    let cols = reader.read_u32::<BigEndian>().unwrap() as usize;
    
    let mut images = Vec::with_capacity(num_images);
    for _ in 0..num_images {
        let mut image = vec![0u8; rows * cols];
        reader.read_exact(&mut image).unwrap();
        images.push(image.iter().map(|&x| x as f32 / 255.0).collect());
    }
    images
}
```

### Pattern 3: CSV Dataset Loading

```rust
use csv::Reader;
use ndarray::{Array2, Array1};

pub fn load_csv_dataset(path: &str) -> (Array2<f32>, Array1<i32>) {
    let mut reader = Reader::from_path(path).unwrap();
    let headers = reader.headers().unwrap().clone();
    
    let mut features: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    
    for result in reader.records() {
        let record = result.unwrap();
        let row: Vec<f32> = record.iter()
            .take(record.len() - 1)
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        let label: i32 = record.get(record.len() - 1)
            .unwrap()
            .parse()
            .unwrap_or(0);
        
        features.push(row);
        labels.push(label);
    }
    
    let n_samples = features.len();
    let n_features = features[0].len();
    
    let flat: Vec<f32> = features.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((n_samples, n_features), flat).unwrap();
    let y = Array1::from_vec(labels);
    
    (x, y)
}
```

---

## Authentication (if required)

```rust
// For protected endpoints
let client = reqwest::Client::new();
let resp = client
    .get(url)
    .header("Authorization", format!("Bearer {}", jwt_token))
    .send()
    .await?;
```

---

## Response Formats

### Dataset Info Response
```json
{
  "id": "mnist",
  "name": "MNIST",
  "description": "Handwritten digit classification",
  "samples": 70000,
  "features": 784,
  "classes": 10,
  "format": "images",
  "input_shape": [1, 28, 28],
  "size_mb": 50
}
```

### Search Response
```json
{
  "results": [
    {
      "id": "kaggle-dataset-id",
      "name": "Dataset Name",
      "source": "kaggle",
      "description": "...",
      "size": "50 MB",
      "downloads": 12345
    }
  ],
  "total": 100
}
```

---

## Contact / Server Info

- **API Server:** nexusconnectbridge.automatanexus.com
- **Tailscale IP:** 100.85.154.94
- **Port:** 8000
- **Health Check:** GET /health

