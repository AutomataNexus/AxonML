//! registry — Loaded-Model Tracker + Alias Resolution
//!
//! Async registry that nexus-serve uses to list, resolve, and hot-swap the
//! set of currently-loaded models. All state lives behind `tokio::sync::RwLock`
//! so handlers can read concurrently while the startup loop writes.
//!
//! Types:
//! - [`ModelInfo`]: metadata for one registered model — canonical `id`,
//!   filesystem `path`, `architecture`, parameter count, quantization label,
//!   `context_length`, `vocab_size`. Returned from `list` / `get`.
//! - [`ModelRegistry`]: three `Arc<RwLock<..>>` maps — canonical models,
//!   alias → canonical mappings, and an optional default model id.
//!
//! Methods: [`ModelRegistry::new`], [`ModelRegistry::register`] (first model
//! registered becomes the default), [`ModelRegistry::register_alias`],
//! [`ModelRegistry::resolve`] (direct match → exact alias → case-insensitive
//! alias), [`ModelRegistry::list`], [`ModelRegistry::list_aliases`],
//! [`ModelRegistry::get`], [`ModelRegistry::default_model`],
//! [`ModelRegistry::set_default`].
//!
//! Aliases let users route friendly short names like "sage" or "oracle" to
//! the full canonical model id ("Qwen2.5 Coder 1.5B Instruct", etc.).
//!
//! # File
//! `nexus-serve/src/model/registry.rs`
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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// =============================================================================
// Types
// =============================================================================

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

// =============================================================================
// Registry
// =============================================================================

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
