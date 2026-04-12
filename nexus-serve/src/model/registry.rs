//! Model registry — tracks loaded models, supports hot-swap.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    pub architecture: String,
    pub parameters: u64,
    pub quantization: String,
    pub context_length: usize,
    pub vocab_size: usize,
}

/// The model registry tracks all loaded models and their state.
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    /// Alias → canonical model ID. Allows user-friendly names like "sage"
    /// to route to "Qwen2.5 Coder 1.5B Instruct".
    aliases: Arc<RwLock<HashMap<String, String>>>,
    default_model: Arc<RwLock<Option<String>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            aliases: Arc::new(RwLock::new(HashMap::new())),
            default_model: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn register(&self, info: ModelInfo) {
        let id = info.id.clone();
        let mut models = self.models.write().await;
        let is_first = models.is_empty();
        models.insert(id.clone(), info);

        if is_first {
            *self.default_model.write().await = Some(id);
        }
    }

    /// Register an alias (e.g., "sage" → "Qwen2.5 Coder 1.5B Instruct").
    pub async fn register_alias(&self, alias: &str, canonical_id: &str) {
        let mut aliases = self.aliases.write().await;
        aliases.insert(alias.to_string(), canonical_id.to_string());
    }

    /// Resolve an id or alias to the canonical model id.
    pub async fn resolve(&self, id_or_alias: &str) -> Option<String> {
        // Direct match first
        if self.models.read().await.contains_key(id_or_alias) {
            return Some(id_or_alias.to_string());
        }
        // Try alias (case-insensitive)
        let aliases = self.aliases.read().await;
        if let Some(canonical) = aliases.get(id_or_alias) {
            return Some(canonical.clone());
        }
        // Case-insensitive alias fallback
        let lower = id_or_alias.to_lowercase();
        for (alias, canonical) in aliases.iter() {
            if alias.to_lowercase() == lower {
                return Some(canonical.clone());
            }
        }
        None
    }

    pub async fn list(&self) -> Vec<ModelInfo> {
        self.models.read().await.values().cloned().collect()
    }

    /// Return (alias, canonical_id) pairs.
    pub async fn list_aliases(&self) -> Vec<(String, String)> {
        self.aliases
            .read()
            .await
            .iter()
            .map(|(a, c)| (a.clone(), c.clone()))
            .collect()
    }

    pub async fn get(&self, id: &str) -> Option<ModelInfo> {
        // Resolve alias first
        let canonical = self.resolve(id).await?;
        self.models.read().await.get(&canonical).cloned()
    }

    pub async fn default_model(&self) -> Option<String> {
        self.default_model.read().await.clone()
    }

    pub async fn set_default(&self, id: &str) {
        *self.default_model.write().await = Some(id.to_string());
    }
}
