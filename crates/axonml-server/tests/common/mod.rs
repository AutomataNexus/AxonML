//! Common test utilities for integration tests
//!
//! # File
//! `crates/axonml-server/tests/common/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

pub const TEST_API_URL: &str = "http://localhost:3021";

pub fn admin_email() -> String {
    std::env::var("AXONML_TEST_ADMIN_EMAIL").unwrap_or_else(|_| "admin@axonml.local".to_string())
}

pub fn admin_password() -> String {
    std::env::var("AXONML_TEST_ADMIN_PASSWORD").unwrap_or_else(|_| "admin".to_string())
}

/// Check if a specific route exists by making an authenticated request
/// and checking if it returns 404 (route not found) vs other responses
pub async fn route_exists(client: &Client, path: &str, token: &str) -> bool {
    let response = client
        .get(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await;

    match response {
        Ok(resp) => resp.status().as_u16() != 404,
        Err(_) => false,
    }
}

/// Check if a POST route exists
pub async fn post_route_exists(client: &Client, path: &str, token: &str) -> bool {
    let response = client
        .post(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .body("{}")
        .send()
        .await;

    match response {
        Ok(resp) => resp.status().as_u16() != 404,
        Err(_) => false,
    }
}

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
    login(client, &admin_email(), &admin_password()).await
}

/// Make authenticated GET request
pub async fn auth_get(
    client: &Client,
    path: &str,
    token: &str,
) -> Result<reqwest::Response, String> {
    client
        .get(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))
}

/// Make authenticated POST request
pub async fn auth_post(
    client: &Client,
    path: &str,
    token: &str,
    body: Value,
) -> Result<reqwest::Response, String> {
    client
        .post(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))
}

/// Make authenticated PUT request
pub async fn auth_put(
    client: &Client,
    path: &str,
    token: &str,
    body: Value,
) -> Result<reqwest::Response, String> {
    client
        .put(format!("{}{}", TEST_API_URL, path))
        .header("Authorization", format!("Bearer {}", token))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))
}

/// Make authenticated DELETE request
pub async fn auth_delete(
    client: &Client,
    path: &str,
    token: &str,
) -> Result<reqwest::Response, String> {
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

/// Macro to skip integration tests when the server is not running
/// or the test admin user is not configured.
/// Place at the top of any test that requires a live server.
#[macro_export]
macro_rules! require_server {
    () => {
        if !common::is_server_running().await {
            eprintln!("SKIP: axonml-server not running on {}", common::TEST_API_URL);
            return;
        }
        // Also verify admin login works (DB must be initialized)
        let _check_client = common::test_client();
        if common::login_as_admin(&_check_client).await.is_err() {
            eprintln!("SKIP: admin login failed (run AxonML_DB_Init.sh to set up test user)");
            return;
        }
    };
}
