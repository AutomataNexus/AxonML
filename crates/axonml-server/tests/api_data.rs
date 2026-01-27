//! Integration tests for data analysis API endpoints

mod common;

use common::*;
use serde_json::Value;

/// Skip test if server not running
macro_rules! require_server {
    () => {
        if !is_server_running().await {
            eprintln!("Skipping: server not running at {}", TEST_API_URL);
            return;
        }
    };
}

// ============================================================================
// Data Analysis Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_analyze_dataset_requires_auth() {
    require_server!();

    let client = test_client();
    let response = client
        .post(format!("{}/api/data/test-id/analyze", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should require authentication"
    );
}

#[tokio::test]
async fn test_analyze_dataset_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/data/nonexistent-dataset-id/analyze",
        &token,
        serde_json::json!({}),
    )
    .await
    .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent dataset"
    );
}

#[tokio::test]
async fn test_preview_dataset_requires_auth() {
    require_server!();

    let client = test_client();
    let response = client
        .post(format!("{}/api/data/test-id/preview", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should require authentication"
    );
}

#[tokio::test]
async fn test_validate_dataset_requires_auth() {
    require_server!();

    let client = test_client();
    let response = client
        .post(format!("{}/api/data/test-id/validate", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should require authentication"
    );
}

#[tokio::test]
async fn test_generate_config_requires_auth() {
    require_server!();

    let client = test_client();
    let response = client
        .post(format!("{}/api/data/test-id/generate-config", TEST_API_URL))
        .json(&serde_json::json!({}))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should require authentication"
    );
}

// ============================================================================
// Kaggle Integration Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_kaggle_status_endpoint() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/kaggle/status", &token)
        .await
        .expect("Request failed");

    // Should return status (configured or not)
    assert!(
        response.status().is_success(),
        "Kaggle status should return success, got {}",
        response.status()
    );

    let body: Value = response.json().await.expect("Failed to parse JSON");
    assert!(
        body.get("configured").is_some(),
        "Response should have 'configured' field"
    );
}

#[tokio::test]
async fn test_kaggle_status_requires_auth() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/kaggle/status", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should require authentication"
    );
}

#[tokio::test]
async fn test_kaggle_search_requires_auth() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/kaggle/search?query=mnist", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should require authentication"
    );
}

#[tokio::test]
async fn test_kaggle_search_without_credentials() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // First check if Kaggle is configured
    let status_resp = auth_get(&client, "/api/kaggle/status", &token)
        .await
        .expect("Request failed");

    let status: Value = status_resp.json().await.expect("Failed to parse JSON");
    let configured = status
        .get("configured")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let response = auth_get(&client, "/api/kaggle/search?query=mnist&limit=5", &token)
        .await
        .expect("Request failed");

    if configured {
        // If configured, search should work
        assert!(
            response.status().is_success() || response.status().as_u16() == 500,
            "Search should succeed or return server error if API fails"
        );
    } else {
        // If not configured, should return error
        assert!(
            response.status().is_client_error() || response.status().is_server_error(),
            "Should return error when not configured"
        );
    }
}

#[tokio::test]
async fn test_kaggle_downloaded_list() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/kaggle/downloaded", &token)
        .await
        .expect("Request failed");

    // Should return list (possibly empty)
    assert!(
        response.status().is_success(),
        "Downloaded list should return success, got {}",
        response.status()
    );
}

#[tokio::test]
async fn test_kaggle_save_credentials_validation() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // Test with empty credentials (should fail validation)
    let response = auth_post(
        &client,
        "/api/kaggle/credentials",
        &token,
        serde_json::json!({
            "username": "",
            "key": ""
        }),
    )
    .await
    .expect("Request failed");

    // Should reject empty credentials
    assert!(
        response.status().is_client_error() || response.status().is_server_error(),
        "Should reject empty credentials"
    );
}

// ============================================================================
// Built-in Datasets Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_list_builtin_datasets() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = match auth_get(&client, "/api/builtin-datasets", &token).await {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("Note: Request failed (network issue): {} - skipping", e);
            return;
        }
    };

    assert!(
        response.status().is_success(),
        "List builtin datasets should succeed, got {}",
        response.status()
    );

    let body: Value = response.json().await.expect("Failed to parse JSON");
    assert!(body.is_array(), "Should return array of datasets");
}

#[tokio::test]
async fn test_list_builtin_datasets_requires_auth() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/builtin-datasets", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        401,
        "Should require authentication"
    );
}

#[tokio::test]
async fn test_search_builtin_datasets() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/builtin-datasets/search?query=mnist", &token)
        .await
        .expect("Request failed");

    assert!(
        response.status().is_success(),
        "Search builtin datasets should succeed, got {}",
        response.status()
    );
}

#[tokio::test]
async fn test_list_dataset_sources() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/builtin-datasets/sources", &token)
        .await
        .expect("Request failed");

    assert!(
        response.status().is_success(),
        "List sources should succeed, got {}",
        response.status()
    );

    let body: Value = response.json().await.expect("Failed to parse JSON");
    assert!(body.is_array(), "Should return array of sources");
}

#[tokio::test]
async fn test_get_builtin_dataset_info_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/builtin-datasets/nonexistent-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent dataset"
    );
}

#[tokio::test]
async fn test_prepare_builtin_dataset_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/builtin-datasets/nonexistent-id/prepare",
        &token,
        serde_json::json!({}),
    )
    .await
    .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent dataset"
    );
}
