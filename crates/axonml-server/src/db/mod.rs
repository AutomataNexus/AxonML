//! Database module for AxonML Server
//!
//! Provides connection to Aegis-DB and CRUD operations for all entities.

pub mod schema;
pub mod users;
pub mod runs;
pub mod models;

use crate::config::AegisConfig;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Error, Debug)]
pub enum DbError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Query failed: {0}")]
    QueryFailed(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Already exists: {0}")]
    AlreadyExists(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Database connection wrapper for Aegis-DB
#[derive(Clone)]
pub struct Database {
    client: Client,
    base_url: String,
    auth: Option<(String, String)>,
    token: Arc<RwLock<Option<String>>>,
}

/// Query request body
#[derive(Debug, Serialize)]
struct QueryRequest {
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Vec<Value>>,
}

/// Query response from Aegis-DB
#[derive(Debug, Deserialize)]
pub struct QueryResponse {
    pub rows: Vec<Value>,
    #[serde(default)]
    pub affected_rows: u64,
}

/// KV get response
#[derive(Debug, Deserialize)]
struct KvGetResponse {
    value: Option<Value>,
}

impl Database {
    /// Create a new database connection
    pub async fn new(config: &AegisConfig) -> Result<Self, DbError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| DbError::ConnectionFailed(e.to_string()))?;

        let base_url = format!("http://{}:{}", config.host, config.port);
        let auth = Some((config.username.clone(), config.password.clone()));

        let db = Self {
            client,
            base_url,
            auth,
            token: Arc::new(RwLock::new(None)),
        };

        // Test connection and authenticate
        db.authenticate().await?;

        Ok(db)
    }

    /// Authenticate with Aegis-DB
    async fn authenticate(&self) -> Result<(), DbError> {
        if let Some((username, password)) = &self.auth {
            let resp = self.client
                .post(format!("{}/api/v1/auth/login", self.base_url))
                .json(&serde_json::json!({
                    "username": username,
                    "password": password
                }))
                .send()
                .await?;

            if resp.status().is_success() {
                let body: Value = resp.json().await?;
                if let Some(token) = body.get("token").and_then(|t| t.as_str()) {
                    let mut lock = self.token.write().await;
                    *lock = Some(token.to_string());
                }
            }
        }
        Ok(())
    }

    /// Get authorization header
    async fn auth_header(&self) -> Option<String> {
        let lock = self.token.read().await;
        lock.as_ref().map(|t| format!("Bearer {}", t))
    }

    /// Execute a query and return results
    pub async fn query(&self, sql: &str) -> Result<QueryResponse, DbError> {
        self.query_with_params(sql, vec![]).await
    }

    /// Execute a query with parameters
    pub async fn query_with_params(&self, sql: &str, params: Vec<Value>) -> Result<QueryResponse, DbError> {
        let mut request = self.client
            .post(format!("{}/api/v1/query", self.base_url))
            .json(&QueryRequest {
                query: sql.to_string(),
                params: if params.is_empty() { None } else { Some(params) },
            });

        if let Some(auth) = self.auth_header().await {
            request = request.header("Authorization", auth);
        }

        let resp = request.send().await?;

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DbError::QueryFailed(error));
        }

        let result: QueryResponse = resp.json().await?;
        Ok(result)
    }

    /// Execute a statement (INSERT, UPDATE, DELETE)
    pub async fn execute(&self, sql: &str) -> Result<u64, DbError> {
        self.execute_with_params(sql, vec![]).await
    }

    /// Execute a statement with parameters
    pub async fn execute_with_params(&self, sql: &str, params: Vec<Value>) -> Result<u64, DbError> {
        let result = self.query_with_params(sql, params).await?;
        Ok(result.affected_rows)
    }

    /// Set a key-value pair
    pub async fn kv_set(&self, key: &str, value: Value) -> Result<(), DbError> {
        let mut request = self.client
            .post(format!("{}/api/v1/kv/keys", self.base_url))
            .json(&serde_json::json!({
                "key": key,
                "value": value
            }));

        if let Some(auth) = self.auth_header().await {
            request = request.header("Authorization", auth);
        }

        let resp = request.send().await?;

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DbError::QueryFailed(error));
        }

        Ok(())
    }

    /// Get a key-value pair
    pub async fn kv_get(&self, key: &str) -> Result<Option<Value>, DbError> {
        let mut request = self.client
            .get(format!("{}/api/v1/kv/keys/{}", self.base_url, key));

        if let Some(auth) = self.auth_header().await {
            request = request.header("Authorization", auth);
        }

        let resp = request.send().await?;

        if resp.status().as_u16() == 404 {
            return Ok(None);
        }

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DbError::QueryFailed(error));
        }

        let result: KvGetResponse = resp.json().await?;
        Ok(result.value)
    }

    /// Delete a key-value pair
    pub async fn kv_delete(&self, key: &str) -> Result<(), DbError> {
        let mut request = self.client
            .delete(format!("{}/api/v1/kv/keys/{}", self.base_url, key));

        if let Some(auth) = self.auth_header().await {
            request = request.header("Authorization", auth);
        }

        let resp = request.send().await?;

        if !resp.status().is_success() && resp.status().as_u16() != 404 {
            let error = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DbError::QueryFailed(error));
        }

        Ok(())
    }

    /// Check if database is healthy
    pub async fn health_check(&self) -> bool {
        match self.client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_request_serialization() {
        let req = QueryRequest {
            query: "SELECT * FROM users".to_string(),
            params: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("SELECT * FROM users"));
        assert!(!json.contains("params"));
    }

    #[test]
    fn test_query_request_with_params() {
        let req = QueryRequest {
            query: "SELECT * FROM users WHERE id = $1".to_string(),
            params: Some(vec![serde_json::json!("user-123")]),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("params"));
    }
}
