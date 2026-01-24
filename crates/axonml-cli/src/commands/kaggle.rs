//! Kaggle CLI Command
//!
//! Commands for Kaggle dataset integration.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use colored::Colorize;

// =============================================================================
// Kaggle Configuration
// =============================================================================

/// Kaggle credentials.
#[derive(Debug, Clone)]
pub struct KaggleCredentials {
    pub username: String,
    pub key: String,
}

/// Get the Kaggle configuration directory.
pub fn kaggle_config_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".kaggle")
}

/// Get the Kaggle credentials file path.
pub fn kaggle_credentials_path() -> PathBuf {
    kaggle_config_dir().join("kaggle.json")
}

/// Check if Kaggle is configured.
pub fn is_configured() -> bool {
    kaggle_credentials_path().exists()
}

/// Load Kaggle credentials.
pub fn load_credentials() -> Option<KaggleCredentials> {
    let path = kaggle_credentials_path();
    if !path.exists() {
        return None;
    }

    let mut file = File::open(&path).ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).ok()?;

    let json: serde_json::Value = serde_json::from_str(&contents).ok()?;
    let username = json.get("username")?.as_str()?.to_string();
    let key = json.get("key")?.as_str()?.to_string();

    Some(KaggleCredentials { username, key })
}

/// Save Kaggle credentials.
pub fn save_credentials(creds: &KaggleCredentials) -> std::io::Result<()> {
    let dir = kaggle_config_dir();
    fs::create_dir_all(&dir)?;

    let path = kaggle_credentials_path();
    let json = serde_json::json!({
        "username": creds.username,
        "key": creds.key
    });

    let mut file = File::create(&path)?;
    file.write_all(serde_json::to_string_pretty(&json)?.as_bytes())?;

    // Set file permissions (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path)?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&path, perms)?;
    }

    Ok(())
}

// =============================================================================
// Kaggle API Client
// =============================================================================

/// Kaggle API client.
pub struct KaggleClient {
    credentials: KaggleCredentials,
    base_url: String,
}

impl KaggleClient {
    /// Create a new Kaggle client.
    pub fn new(credentials: KaggleCredentials) -> Self {
        Self {
            credentials,
            base_url: "https://www.kaggle.com/api/v1".to_string(),
        }
    }

    /// Create client from stored credentials.
    pub fn from_stored() -> Option<Self> {
        load_credentials().map(Self::new)
    }

    /// Search for datasets via Kaggle API.
    pub fn search_datasets(&self, query: &str, page: u32) -> Result<Vec<KaggleDataset>, String> {
        let url = format!(
            "{}/datasets/list?search={}&page={}",
            self.base_url, query, page
        );

        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&url)
            .basic_auth(&self.credentials.username, Some(&self.credentials.key))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .map_err(|e| format!("Network error: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Kaggle API error: {} - verify credentials", response.status()));
        }

        let datasets: Vec<KaggleDataset> = response.json().map_err(|e| e.to_string())?;
        Ok(datasets)
    }

    /// Download a dataset from Kaggle.
    pub fn download_dataset(&self, dataset_ref: &str, output_dir: &PathBuf) -> Result<PathBuf, String> {
        let url = format!(
            "{}/datasets/download/{}",
            self.base_url, dataset_ref
        );

        fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;

        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&url)
            .basic_auth(&self.credentials.username, Some(&self.credentials.key))
            .timeout(std::time::Duration::from_secs(300))
            .send()
            .map_err(|e| format!("Network error: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Download failed: {} - check dataset reference", response.status()));
        }

        let filename = dataset_ref.replace('/', "_") + ".zip";
        let output_path = output_dir.join(&filename);

        let bytes = response.bytes().map_err(|e| e.to_string())?;
        let mut file = File::create(&output_path).map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;

        Ok(output_path)
    }
}

/// Kaggle dataset information.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct KaggleDataset {
    #[serde(rename = "ref")]
    pub ref_name: String,
    pub title: String,
    #[serde(rename = "totalBytes", default)]
    pub size: String,
    #[serde(rename = "downloadCount", default)]
    pub download_count: u64,
    #[serde(rename = "voteCount", default)]
    pub vote_count: u64,
    #[serde(rename = "lastUpdated", default)]
    pub last_updated: String,
}

// =============================================================================
// CLI Commands
// =============================================================================

/// Execute kaggle login command.
pub fn execute_login(username: &str, key: &str) -> Result<(), String> {
    println!("{}", "Configuring Kaggle credentials...".cyan());

    let creds = KaggleCredentials {
        username: username.to_string(),
        key: key.to_string(),
    };

    save_credentials(&creds).map_err(|e| e.to_string())?;

    println!("{} Kaggle credentials saved to {:?}", "✓".green(), kaggle_credentials_path());
    println!("  Username: {}", username);
    println!("  Key: {}...", &key[..8.min(key.len())]);

    Ok(())
}

/// Execute kaggle status command.
pub fn execute_status() -> Result<(), String> {
    if is_configured() {
        if let Some(creds) = load_credentials() {
            println!("{} Kaggle is configured", "✓".green());
            println!("  Username: {}", creds.username);
            println!("  Config: {:?}", kaggle_credentials_path());
        }
    } else {
        println!("{} Kaggle is not configured", "✗".red());
        println!("  Run: axonml kaggle login --username <USER> --key <KEY>");
    }
    Ok(())
}

/// Execute kaggle search command.
pub fn execute_search(query: &str, limit: usize) -> Result<(), String> {
    let client = KaggleClient::from_stored()
        .ok_or_else(|| "Kaggle not configured. Run: axonml kaggle login".to_string())?;

    println!("{} Searching Kaggle for '{}'...", "🔍".cyan(), query);

    let datasets = client.search_datasets(query, 1)?;

    if datasets.is_empty() {
        println!("No datasets found for '{}'", query);
        return Ok(());
    }

    println!("\n{}", "Found datasets:".bold());
    println!("{}", "─".repeat(80));

    for (i, ds) in datasets.iter().take(limit).enumerate() {
        println!(
            "{:2}. {} {}",
            i + 1,
            ds.title.green(),
            format!("({})", ds.ref_name).dimmed()
        );
        println!(
            "    Size: {} | Downloads: {} | Votes: {} | Updated: {}",
            ds.size, ds.download_count, ds.vote_count, ds.last_updated
        );
    }

    println!("{}", "─".repeat(80));
    println!(
        "Download with: {} <dataset-ref>",
        "axonml kaggle download".cyan()
    );

    Ok(())
}

/// Execute kaggle download command.
pub fn execute_download(dataset_ref: &str, output: Option<&str>) -> Result<(), String> {
    let client = KaggleClient::from_stored()
        .ok_or_else(|| "Kaggle not configured. Run: axonml kaggle login".to_string())?;

    let output_dir = output
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./data"));

    println!("{} Downloading {}...", "⬇".cyan(), dataset_ref);

    let path = client.download_dataset(dataset_ref, &output_dir)?;

    println!("{} Downloaded to {:?}", "✓".green(), path);

    Ok(())
}

/// Execute kaggle list command (list downloaded datasets).
pub fn execute_list() -> Result<(), String> {
    let data_dir = PathBuf::from("./data");

    if !data_dir.exists() {
        println!("No data directory found. Download datasets first.");
        return Ok(());
    }

    println!("{}", "Downloaded datasets:".bold());
    println!("{}", "─".repeat(60));

    let mut count = 0;
    for entry in fs::read_dir(&data_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.is_file() {
            let filename = path.file_name().unwrap().to_string_lossy();
            let metadata = fs::metadata(&path).map_err(|e| e.to_string())?;
            let size_mb = metadata.len() as f64 / 1_000_000.0;

            println!("  {} ({:.1} MB)", filename.green(), size_mb);
            count += 1;
        }
    }

    if count == 0 {
        println!("  No datasets downloaded yet.");
    }

    println!("{}", "─".repeat(60));

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_paths() {
        let dir = kaggle_config_dir();
        assert!(dir.to_string_lossy().contains(".kaggle"));

        let creds_path = kaggle_credentials_path();
        assert!(creds_path.to_string_lossy().contains("kaggle.json"));
    }

    #[test]
    fn test_kaggle_client_creation() {
        let creds = KaggleCredentials {
            username: "test_user".to_string(),
            key: "test_api_key".to_string(),
        };
        // Client creation succeeds - actual API calls require valid Kaggle credentials
        let _client = KaggleClient::new(creds);
    }
}
