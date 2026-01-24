//! Common test utilities for integration tests

use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

pub const TEST_API_URL: &str = "http://localhost:3021";
pub const ADMIN_EMAIL: &str = "admin@axonml.local";
pub const ADMIN_PASSWORD: &str = "admin";

/// Test HTTP client with common configuration
pub fn test_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create test client")
}

/// Login and get auth token
pub async fn login(client: &Client, email: &str, password: &str) -> Result<String, String> {
    let response = client
        .post(format!("{}/api/auth/login", TEST_API_URL))
        .json(&serde_json::json!({
            "email": email,
            "password": password
        }))
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Login failed: {}", response.status()));
    }

    let body: Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    body.get("access_token")
        .or_else(|| body.get("token"))
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "No token in response".to_string())
}

/// Login as admin and get token
pub async fn login_as_admin(client: &Client) -> Result<String, String> {
    login(client, ADMIN_EMAIL, ADMIN_PASSWORD).await
}

/// Make authenticated GET request
pub async fn auth_get(client: &Client, path: &str, token: &str) -> Result<reqwest::Response, String> {
    client
        .get(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))
}

/// Make authenticated POST request
pub async fn auth_post(client: &Client, path: &str, token: &str, body: Value) -> Result<reqwest::Response, String> {
    client
        .post(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))
}

/// Make authenticated PUT request
pub async fn auth_put(client: &Client, path: &str, token: &str, body: Value) -> Result<reqwest::Response, String> {
    client
        .put(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))
}

/// Make authenticated DELETE request
pub async fn auth_delete(client: &Client, path: &str, token: &str) -> Result<reqwest::Response, String> {
    client
        .delete(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))
}

/// Check if test server is running
pub async fn is_server_running() -> bool {
    let client = test_client();
    client
        .get(format!("{}/health", TEST_API_URL))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}
