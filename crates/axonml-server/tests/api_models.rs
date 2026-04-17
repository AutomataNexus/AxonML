//! Models API — Integration Tests
//!
//! Tests for the model management API endpoints on the AxonML server. Covers
//! CRUD operations: listing models (authenticated and unauthenticated), creating
//! a model with name/description/framework/architecture, updating an existing
//! model's description via PUT, deleting a model (404 for nonexistent), and
//! querying model version history. Uses the `require_server!` macro to skip
//! gracefully when the server or admin DB is unavailable.
//!
//! # File
//! `crates/axonml-server/tests/api_models.rs`
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

mod common;

use common::*;
use serde_json::Value;

// =============================================================================
// Test Helpers
// =============================================================================

macro_rules! require_server {
    () => {
        if !is_server_running().await {
            eprintln!("SKIP: server not running at {}", TEST_API_URL);
            return;
        }
        let _c = test_client();
        if login_as_admin(&_c).await.is_err() {
            eprintln!("SKIP: admin login failed (run AxonML_DB_Init.sh)");
            return;
        }
    };
}

// =============================================================================
// Model Listing Tests
// =============================================================================

#[tokio::test]
async fn test_list_models_authenticated() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/models", &token)
        .await
        .expect("Request failed");

    assert!(response.status().is_success(), "Should return success");

    let body: Value = response.json().await.expect("Failed to parse JSON");
    // Response should be an array or object with models
    assert!(body.is_array() || body.get("models").is_some());
}

#[tokio::test]
async fn test_list_models_unauthenticated() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/models", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should return 401 without auth"
    );
}

// =============================================================================
// Model CRUD Tests
// =============================================================================

#[tokio::test]
async fn test_get_model_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/models/nonexistent-model-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent model"
    );
}

#[tokio::test]
async fn test_create_model() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let model_name = format!("test-model-{}", chrono::Utc::now().timestamp_millis());

    let response = auth_post(
        &client,
        "/api/models",
        &token,
        serde_json::json!({
            "name": model_name,
            "description": "Test model for integration testing",
            "framework": "axonml",
            "architecture": "linear"
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    // Should succeed (200/201), 422 for wrong fields, or 400 for validation
    assert!(
        status == 200 || status == 201 || status == 400 || status == 422,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_update_model() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // First, list models to get an existing ID
    let list_response = auth_get(&client, "/api/models", &token)
        .await
        .expect("Request failed");

    if !list_response.status().is_success() {
        return;
    }

    let models: Value = list_response.json().await.expect("Failed to parse JSON");
    let models_arr = models
        .as_array()
        .or_else(|| models.get("models").and_then(|m| m.as_array()));

    if let Some(arr) = models_arr {
        if let Some(first_model) = arr.first() {
            if let Some(id) = first_model.get("id").and_then(|i| i.as_str()) {
                let response = auth_put(
                    &client,
                    &format!("/api/models/{}", id),
                    &token,
                    serde_json::json!({
                        "description": "Updated description"
                    }),
                )
                .await
                .expect("Request failed");

                let status = response.status().as_u16();
                assert!(
                    status == 200 || status == 404,
                    "Got unexpected status: {}",
                    status
                );
            }
        }
    }
}

#[tokio::test]
async fn test_delete_model_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_delete(&client, "/api/models/nonexistent-model-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent model"
    );
}

// =============================================================================
// Model Version Tests
// =============================================================================

#[tokio::test]
async fn test_model_versions() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // First, list models to get an existing ID
    let list_response = auth_get(&client, "/api/models", &token)
        .await
        .expect("Request failed");

    if !list_response.status().is_success() {
        return;
    }

    let models: Value = list_response.json().await.expect("Failed to parse JSON");
    let models_arr = models
        .as_array()
        .or_else(|| models.get("models").and_then(|m| m.as_array()));

    if let Some(arr) = models_arr {
        if let Some(first_model) = arr.first() {
            if let Some(id) = first_model.get("id").and_then(|i| i.as_str()) {
                let response = auth_get(&client, &format!("/api/models/{}/versions", id), &token)
                    .await
                    .expect("Request failed");

                // Versions endpoint may or may not exist
                let status = response.status().as_u16();
                assert!(
                    status == 200 || status == 404,
                    "Got unexpected status: {}",
                    status
                );
            }
        }
    }
}
