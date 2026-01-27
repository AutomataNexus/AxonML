//! Model pool for AxonML
//!
//! Manages a pool of loaded models for inference.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Model pool entry
#[derive(Debug)]
pub struct PoolEntry {
    pub endpoint_id: String,
    pub model_id: String,
    pub version_id: String,
    pub replicas: u32,
    pub current_load: u32,
    pub last_used: std::time::Instant,
}

/// Model pool for managing model instances
pub struct ModelPool {
    entries: Arc<RwLock<HashMap<String, PoolEntry>>>,
    max_entries: usize,
    idle_timeout: std::time::Duration,
}

impl ModelPool {
    /// Create a new model pool
    pub fn new(max_entries: usize, idle_timeout_secs: u64) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
            idle_timeout: std::time::Duration::from_secs(idle_timeout_secs),
        }
    }

    /// Add a model to the pool
    pub async fn add(
        &self,
        endpoint_id: &str,
        model_id: &str,
        version_id: &str,
        replicas: u32,
    ) -> Result<(), String> {
        let mut entries = self.entries.write().await;

        if entries.len() >= self.max_entries && !entries.contains_key(endpoint_id) {
            // Evict least recently used entry
            let lru_key = entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());

            if let Some(key) = lru_key {
                entries.remove(&key);
            }
        }

        entries.insert(
            endpoint_id.to_string(),
            PoolEntry {
                endpoint_id: endpoint_id.to_string(),
                model_id: model_id.to_string(),
                version_id: version_id.to_string(),
                replicas,
                current_load: 0,
                last_used: std::time::Instant::now(),
            },
        );

        Ok(())
    }

    /// Remove a model from the pool
    pub async fn remove(&self, endpoint_id: &str) -> Result<(), String> {
        let mut entries = self.entries.write().await;
        entries.remove(endpoint_id);
        Ok(())
    }

    /// Acquire a model for inference
    pub async fn acquire(&self, endpoint_id: &str) -> Result<(), String> {
        let mut entries = self.entries.write().await;
        let entry = entries
            .get_mut(endpoint_id)
            .ok_or_else(|| format!("Model not in pool: {}", endpoint_id))?;

        if entry.current_load >= entry.replicas {
            return Err("Model at capacity".to_string());
        }

        entry.current_load += 1;
        entry.last_used = std::time::Instant::now();

        Ok(())
    }

    /// Release a model after inference
    pub async fn release(&self, endpoint_id: &str) -> Result<(), String> {
        let mut entries = self.entries.write().await;
        let entry = entries
            .get_mut(endpoint_id)
            .ok_or_else(|| format!("Model not in pool: {}", endpoint_id))?;

        if entry.current_load > 0 {
            entry.current_load -= 1;
        }

        Ok(())
    }

    /// Get current load for a model
    pub async fn get_load(&self, endpoint_id: &str) -> Option<u32> {
        let entries = self.entries.read().await;
        entries.get(endpoint_id).map(|e| e.current_load)
    }

    /// Get pool size
    pub async fn size(&self) -> usize {
        let entries = self.entries.read().await;
        entries.len()
    }

    /// Cleanup idle entries
    pub async fn cleanup_idle(&self) {
        let mut entries = self.entries.write().await;
        let now = std::time::Instant::now();

        entries.retain(|_, entry| {
            entry.current_load > 0 || now.duration_since(entry.last_used) < self.idle_timeout
        });
    }

    /// Get entry info for an endpoint
    pub async fn get_entry(&self, endpoint_id: &str) -> Option<PoolEntryInfo> {
        let entries = self.entries.read().await;
        entries.get(endpoint_id).map(|e| PoolEntryInfo {
            endpoint_id: e.endpoint_id.clone(),
            model_id: e.model_id.clone(),
            version_id: e.version_id.clone(),
            replicas: e.replicas,
            current_load: e.current_load,
            idle_time_secs: e.last_used.elapsed().as_secs(),
        })
    }

    /// Get all entry infos
    pub async fn list_entries(&self) -> Vec<PoolEntryInfo> {
        let entries = self.entries.read().await;
        entries
            .values()
            .map(|e| PoolEntryInfo {
                endpoint_id: e.endpoint_id.clone(),
                model_id: e.model_id.clone(),
                version_id: e.version_id.clone(),
                replicas: e.replicas,
                current_load: e.current_load,
                idle_time_secs: e.last_used.elapsed().as_secs(),
            })
            .collect()
    }

    /// Check if pool has capacity for an endpoint
    pub async fn has_capacity(&self, endpoint_id: &str) -> bool {
        let entries = self.entries.read().await;
        entries
            .get(endpoint_id)
            .map(|e| e.current_load < e.replicas)
            .unwrap_or(false)
    }

    /// Get idle timeout in seconds
    pub fn idle_timeout_secs(&self) -> u64 {
        self.idle_timeout.as_secs()
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        let entries = self.entries.read().await;
        let total_entries = entries.len();
        let total_load: u32 = entries.values().map(|e| e.current_load).sum();
        let total_capacity: u32 = entries.values().map(|e| e.replicas).sum();

        PoolStats {
            total_entries,
            total_load,
            total_capacity,
            utilization: if total_capacity > 0 {
                total_load as f64 / total_capacity as f64
            } else {
                0.0
            },
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_entries: usize,
    pub total_load: u32,
    pub total_capacity: u32,
    pub utilization: f64,
}

/// Pool entry info (safe to expose)
#[derive(Debug, Clone)]
pub struct PoolEntryInfo {
    pub endpoint_id: String,
    pub model_id: String,
    pub version_id: String,
    pub replicas: u32,
    pub current_load: u32,
    pub idle_time_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pool_operations() {
        let pool = ModelPool::new(10, 300);

        pool.add("ep-1", "model-1", "ver-1", 2).await.unwrap();

        assert_eq!(pool.size().await, 1);
        assert_eq!(pool.get_load("ep-1").await, Some(0));

        pool.acquire("ep-1").await.unwrap();
        assert_eq!(pool.get_load("ep-1").await, Some(1));

        pool.acquire("ep-1").await.unwrap();
        assert_eq!(pool.get_load("ep-1").await, Some(2));

        // Should fail - at capacity
        assert!(pool.acquire("ep-1").await.is_err());

        pool.release("ep-1").await.unwrap();
        assert_eq!(pool.get_load("ep-1").await, Some(1));
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let pool = ModelPool::new(10, 300);

        pool.add("ep-1", "model-1", "ver-1", 4).await.unwrap();
        pool.add("ep-2", "model-2", "ver-1", 2).await.unwrap();

        pool.acquire("ep-1").await.unwrap();
        pool.acquire("ep-1").await.unwrap();
        pool.acquire("ep-2").await.unwrap();

        let stats = pool.stats().await;
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.total_load, 3);
        assert_eq!(stats.total_capacity, 6);
        assert!((stats.utilization - 0.5).abs() < 0.01);
    }
}
