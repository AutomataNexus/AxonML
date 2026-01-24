//! Dataset CLI Command
//!
//! Commands for dataset management with NexusConnectBridge integration.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use colored::Colorize;

// =============================================================================
// NexusConnectBridge Configuration
// =============================================================================

/// NexusConnectBridge API base URL.
pub const NEXUS_API_URL: &str = "https://nexusconnectbridge.automatanexus.com/api/v1/bridge/datasets";

/// Tailscale fallback URL.
pub const NEXUS_TAILSCALE_URL: &str = "http://100.85.154.94:8000/api/v1/bridge/datasets";

/// Built-in dataset information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BuiltinDataset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub samples: usize,
    pub features: usize,
    pub classes: usize,
    pub format: String,
    pub size_mb: f64,
}

/// Get list of built-in datasets.
pub fn builtin_datasets() -> Vec<BuiltinDataset> {
    vec![
        BuiltinDataset {
            id: "mnist".to_string(),
            name: "MNIST Handwritten Digits".to_string(),
            description: "70k grayscale images of handwritten digits (0-9), 28x28 pixels".to_string(),
            samples: 70000,
            features: 784,
            classes: 10,
            format: "images".to_string(),
            size_mb: 50.0,
        },
        BuiltinDataset {
            id: "fashion-mnist".to_string(),
            name: "Fashion MNIST".to_string(),
            description: "70k grayscale images of clothing items (10 categories), 28x28 pixels".to_string(),
            samples: 70000,
            features: 784,
            classes: 10,
            format: "images".to_string(),
            size_mb: 50.0,
        },
        BuiltinDataset {
            id: "cifar-10".to_string(),
            name: "CIFAR-10".to_string(),
            description: "60k color images in 10 classes (airplane, car, bird, etc.), 32x32 RGB".to_string(),
            samples: 60000,
            features: 3072,
            classes: 10,
            format: "images".to_string(),
            size_mb: 170.0,
        },
        BuiltinDataset {
            id: "cifar-100".to_string(),
            name: "CIFAR-100".to_string(),
            description: "60k color images in 100 fine-grained classes, 32x32 RGB".to_string(),
            samples: 60000,
            features: 3072,
            classes: 100,
            format: "images".to_string(),
            size_mb: 170.0,
        },
        BuiltinDataset {
            id: "iris".to_string(),
            name: "Iris Flower Dataset".to_string(),
            description: "150 samples of iris flowers with 4 features, 3 classes".to_string(),
            samples: 150,
            features: 4,
            classes: 3,
            format: "csv".to_string(),
            size_mb: 0.005,
        },
        BuiltinDataset {
            id: "wine-quality".to_string(),
            name: "Wine Quality Dataset".to_string(),
            description: "6,497 samples of wine with chemical properties, predict quality (0-10)".to_string(),
            samples: 6497,
            features: 11,
            classes: 10,
            format: "csv".to_string(),
            size_mb: 0.4,
        },
        BuiltinDataset {
            id: "breast-cancer".to_string(),
            name: "Breast Cancer Wisconsin".to_string(),
            description: "569 samples for breast cancer classification, 30 features, binary".to_string(),
            samples: 569,
            features: 30,
            classes: 2,
            format: "csv".to_string(),
            size_mb: 0.1,
        },
    ]
}

/// Dataset search result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub name: String,
    pub source: String,
    pub description: String,
    pub size: String,
    pub downloads: u64,
}

// =============================================================================
// NexusConnectBridge Client
// =============================================================================

/// NexusConnectBridge API client.
pub struct NexusClient {
    base_url: String,
}

impl NexusClient {
    /// Create a new client.
    pub fn new() -> Self {
        Self {
            base_url: NEXUS_API_URL.to_string(),
        }
    }

    /// Create client with Tailscale URL.
    pub fn with_tailscale() -> Self {
        Self {
            base_url: NEXUS_TAILSCALE_URL.to_string(),
        }
    }

    /// List built-in datasets from API, fallback to local registry.
    pub fn list_builtin(&self) -> Result<Vec<BuiltinDataset>, String> {
        let url = format!("{}/builtin", self.base_url);

        let client = reqwest::blocking::Client::new();
        match client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
        {
            Ok(response) if response.status().is_success() => {
                match response.json::<Vec<BuiltinDataset>>() {
                    Ok(datasets) => Ok(datasets),
                    Err(_) => Ok(builtin_datasets()) // Fallback on parse error
                }
            }
            _ => {
                // Fallback to local registry if API unavailable
                Ok(builtin_datasets())
            }
        }
    }

    /// Search datasets across sources via NexusConnectBridge API.
    pub fn search(&self, query: &str, source: Option<&str>, max_results: usize) -> Result<Vec<SearchResult>, String> {
        let mut url = format!("{}/search?query={}&maxResults={}", self.base_url, query, max_results);
        if let Some(src) = source {
            url.push_str(&format!("&source={}", src));
        }

        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .map_err(|e| format!("Network error: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("API error: {} - check NexusConnectBridge service", response.status()));
        }

        let results: Vec<SearchResult> = response.json().map_err(|e| e.to_string())?;
        Ok(results)
    }

    /// Get dataset info from API, fallback to local registry.
    pub fn get_info(&self, dataset_id: &str) -> Result<BuiltinDataset, String> {
        let url = format!("{}/builtin/{}", self.base_url, dataset_id);

        let client = reqwest::blocking::Client::new();
        match client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
        {
            Ok(response) if response.status().is_success() => {
                if let Ok(dataset) = response.json::<BuiltinDataset>() {
                    return Ok(dataset);
                }
            }
            _ => {}
        }

        // Fallback to local registry
        builtin_datasets()
            .into_iter()
            .find(|d| d.id == dataset_id)
            .ok_or_else(|| format!("Dataset not found: {}", dataset_id))
    }
}

impl Default for NexusClient {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CLI Commands
// =============================================================================

/// Execute dataset list command.
pub fn execute_list(source: Option<&str>) -> Result<(), String> {
    let client = NexusClient::new();

    match source {
        Some("builtin") | None => {
            let datasets = client.list_builtin()?;

            println!("{}", "Built-in Datasets".bold());
            println!("{}", "═".repeat(90));
            println!(
                "{:<15} {:<30} {:>10} {:>8} {:>6} {:>8}",
                "ID".bold(),
                "NAME".bold(),
                "SAMPLES".bold(),
                "FEATURES".bold(),
                "CLASSES".bold(),
                "SIZE".bold()
            );
            println!("{}", "─".repeat(90));

            for ds in &datasets {
                println!(
                    "{:<15} {:<30} {:>10} {:>8} {:>6} {:>6.1} MB",
                    ds.id.cyan(),
                    truncate(&ds.name, 28),
                    ds.samples,
                    ds.features,
                    ds.classes,
                    ds.size_mb
                );
            }

            println!("{}", "─".repeat(90));
            println!(
                "Get info: {} <dataset-id>",
                "axonml dataset info".cyan()
            );
        }
        Some(src) => {
            println!("Listing datasets from source: {}", src);
            println!("Use 'axonml dataset search' to search external sources.");
        }
    }

    Ok(())
}

/// Execute dataset info command.
pub fn execute_info(dataset_id: &str) -> Result<(), String> {
    let client = NexusClient::new();
    let ds = client.get_info(dataset_id)?;

    println!("{}", format!("Dataset: {}", ds.name).bold());
    println!("{}", "─".repeat(60));
    println!("ID:          {}", ds.id.cyan());
    println!("Description: {}", ds.description);
    println!("Samples:     {}", ds.samples);
    println!("Features:    {}", ds.features);
    println!("Classes:     {}", ds.classes);
    println!("Format:      {}", ds.format);
    println!("Size:        {:.1} MB", ds.size_mb);
    println!("{}", "─".repeat(60));

    // Show loading code
    println!("\n{}", "Rust Loading Code:".bold());
    println!(
        r#"
use axonml_vision::datasets::{{{}}};

// Load the dataset
let train = {}::train();
let test = {}::test();

// Access samples
let (image, label) = train.get(0).unwrap();
println!("Image shape: {{:?}}", image.shape());
println!("Label shape: {{:?}}", label.shape());
"#,
        ds.id.replace("-", "_").to_uppercase(),
        ds.id.replace("-", "_").to_uppercase(),
        ds.id.replace("-", "_").to_uppercase()
    );

    Ok(())
}

/// Execute dataset search command.
pub fn execute_search(query: &str, source: Option<&str>, limit: usize) -> Result<(), String> {
    let client = NexusClient::new();

    println!("{} Searching for '{}'...", "🔍".cyan(), query);
    if let Some(src) = source {
        println!("Source filter: {}", src);
    }

    // Try primary API, fall back to Tailscale if needed
    let results = match client.search(query, source, limit) {
        Ok(r) => r,
        Err(_) => {
            // Try Tailscale fallback
            let tailscale_client = NexusClient::with_tailscale();
            tailscale_client.search(query, source, limit)?
        }
    };

    if results.is_empty() {
        println!("No datasets found for '{}'", query);
        return Ok(());
    }

    println!("\n{}", "Search Results".bold());
    println!("{}", "═".repeat(80));

    for (i, result) in results.iter().take(limit).enumerate() {
        println!(
            "{:2}. {} {} [{}]",
            i + 1,
            result.name.green(),
            format!("({})", result.id).dimmed(),
            result.source.cyan()
        );
        println!(
            "    {} | Downloads: {}",
            result.size, result.downloads
        );
        println!("    {}", result.description.dimmed());
    }

    println!("{}", "─".repeat(80));
    println!("Total results: {}", results.len());

    Ok(())
}

/// Execute dataset download command.
pub fn execute_download(dataset_id: &str, output: Option<&str>) -> Result<(), String> {
    let output_dir = output
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./data"));

    fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

    println!("{} Downloading {}...", "⬇".cyan(), dataset_id);

    // Check if it's a built-in dataset
    let builtin = builtin_datasets();
    if let Some(ds) = builtin.iter().find(|d| d.id == dataset_id) {
        println!("Dataset: {} ({:.1} MB)", ds.name, ds.size_mb);

        // For built-in datasets, generate loading code instead of downloading
        let code_path = output_dir.join(format!("{}_loader.rs", dataset_id.replace("-", "_")));

        let code = generate_loader_code(ds);
        let mut file = File::create(&code_path).map_err(|e| e.to_string())?;
        file.write_all(code.as_bytes()).map_err(|e| e.to_string())?;

        println!("{} Generated loader code at {:?}", "✓".green(), code_path);
        println!("\nBuilt-in datasets are downloaded automatically on first use.");
        println!("The generated code shows how to load the dataset in your project.");
    } else {
        // External dataset - would need API download
        println!("Note: External dataset download requires API access.");
        println!("Use Kaggle CLI for Kaggle datasets: axonml kaggle download");
    }

    Ok(())
}

/// Generate loader code for a dataset.
fn generate_loader_code(ds: &BuiltinDataset) -> String {
    let struct_name = ds.id.replace("-", "_").to_uppercase();

    format!(
        r#"//! {} Dataset Loader
//!
//! Auto-generated by Axonml CLI

use axonml_vision::datasets::{{{}}};
use axonml_data::{{DataLoader, Dataset}};

fn main() {{
    // Load training data
    let train_dataset = {}::train();
    println!("Training samples: {{}}", train_dataset.len());

    // Load test data
    let test_dataset = {}::test();
    println!("Test samples: {{}}", test_dataset.len());

    // Create data loaders
    let train_loader = DataLoader::new(train_dataset, 32);
    let test_loader = DataLoader::new(test_dataset, 32);

    // Iterate over batches
    for batch in train_loader.iter().take(1) {{
        println!("Batch data shape: {{:?}}", batch.data.shape());
        println!("Batch labels shape: {{:?}}", batch.labels.shape());
    }}
}}

// Dataset Info:
// - Samples: {}
// - Features: {}
// - Classes: {}
// - Format: {}
"#,
        ds.name,
        struct_name,
        struct_name,
        struct_name,
        ds.samples,
        ds.features,
        ds.classes,
        ds.format
    )
}

/// Execute dataset sources command.
pub fn execute_sources() -> Result<(), String> {
    println!("{}", "Available Dataset Sources".bold());
    println!("{}", "═".repeat(60));

    let sources = [
        ("builtin", "Built-in datasets (MNIST, CIFAR, Iris, etc.)", "axonml dataset list"),
        ("kaggle", "Kaggle (65,000+ datasets)", "axonml dataset search --source kaggle"),
        ("uci", "UCI ML Repository", "axonml dataset search --source uci"),
        ("data.gov", "US Government Open Data", "axonml dataset search --source data.gov"),
    ];

    for (name, desc, cmd) in &sources {
        println!("\n{}", name.cyan().bold());
        println!("  {}", desc);
        println!("  Command: {}", cmd.dimmed());
    }

    println!("\n{}", "─".repeat(60));
    println!("API endpoint: {}", NEXUS_API_URL);
    println!("Tailscale fallback: {}", NEXUS_TAILSCALE_URL);

    Ok(())
}

// =============================================================================
// Utilities
// =============================================================================

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_datasets() {
        let datasets = builtin_datasets();
        assert!(!datasets.is_empty());

        let mnist = datasets.iter().find(|d| d.id == "mnist");
        assert!(mnist.is_some());

        let mnist = mnist.unwrap();
        assert_eq!(mnist.samples, 70000);
        assert_eq!(mnist.classes, 10);
    }

    #[test]
    fn test_nexus_client_builtin() {
        let client = NexusClient::new();

        // List built-in datasets (uses local fallback if API unreachable)
        let datasets = client.list_builtin().unwrap();
        assert!(!datasets.is_empty());
    }

    #[test]
    fn test_nexus_client_search() {
        let client = NexusClient::new();
        // Search calls real NexusConnectBridge API
        // Result depends on API availability
        let _ = client.search("image", None, 10);
    }

    #[test]
    fn test_get_info() {
        let client = NexusClient::new();

        // Get info uses API with local fallback
        let info = client.get_info("mnist").unwrap();
        assert_eq!(info.id, "mnist");
        assert_eq!(info.classes, 10);
    }
}
