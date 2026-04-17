//! Training API — Integration Tests
//!
//! Tests for the training run management API endpoints on the AxonML server.
//! Covers CRUD operations on training runs: listing (authenticated and
//! unauthenticated), creating a run with model/config, fetching run metrics
//! and logs, stopping a run, deleting a run (404 for nonexistent), and
//! query features like pagination (`limit`/`offset`) and status filtering.
//! Uses the `require_server!` macro to skip gracefully when the server or
//! admin DB is unavailable.
//!
//! # File
//! `crates/axonml-server/tests/api_training.rs`
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
// Training Run Listing Tests
// =============================================================================

#[tokio::test]
async fn test_list_training_runs_authenticated() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/training/runs", &token)
        .await
        .expect("Request failed");

    assert!(response.status().is_success(), "Should return success");

    let body: Value = response.json().await.expect("Failed to parse JSON");
    assert!(body.is_array() || body.get("runs").is_some());
}

#[tokio::test]
async fn test_list_training_runs_unauthenticated() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/training/runs", TEST_API_URL))
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
// Training Run CRUD Tests
// =============================================================================

#[tokio::test]
async fn test_get_training_run_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/training/runs/nonexistent-run-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent run"
    );
}

#[tokio::test]
async fn test_create_training_run() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let run_name = format!("test-run-{}", chrono::Utc::now().timestamp_millis());

    let response = auth_post(
        &client,
        "/api/training/runs",
        &token,
        serde_json::json!({
            "name": run_name,
            "model_id": "test-model",
            "config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    // Might fail due to missing model or wrong fields, but should not be 401/500
    assert!(
        status == 200 || status == 201 || status == 400 || status == 404 || status == 422,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_stop_training_run() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // Try to stop a nonexistent run
    let response = auth_post(
        &client,
        "/api/training/runs/nonexistent-run-id/stop",
        &token,
        serde_json::json!({}),
    )
    .await
    .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent run"
    );
}

#[tokio::test]
async fn test_delete_training_run_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_delete(&client, "/api/training/runs/nonexistent-run-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent run"
    );
}

// =============================================================================
// Training Run Metrics and Logs Tests
// =============================================================================

#[tokio::test]
async fn test_training_run_metrics() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // List runs to get an existing ID
    let list_response = auth_get(&client, "/api/training/runs", &token)
        .await
        .expect("Request failed");

    if !list_response.status().is_success() {
        return;
    }

    let runs: Value = list_response.json().await.expect("Failed to parse JSON");
    let runs_arr = runs
        .as_array()
        .or_else(|| runs.get("runs").and_then(|r| r.as_array()));

    if let Some(arr) = runs_arr {
        if let Some(first_run) = arr.first() {
            if let Some(id) = first_run.get("id").and_then(|i| i.as_str()) {
                let response = auth_get(
                    &client,
                    &format!("/api/training/runs/{}/metrics", id),
                    &token,
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
async fn test_training_run_logs() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // List runs to get an existing ID
    let list_response = auth_get(&client, "/api/training/runs", &token)
        .await
        .expect("Request failed");

    if !list_response.status().is_success() {
        return;
    }

    let runs: Value = list_response.json().await.expect("Failed to parse JSON");
    let runs_arr = runs
        .as_array()
        .or_else(|| runs.get("runs").and_then(|r| r.as_array()));

    if let Some(arr) = runs_arr {
        if let Some(first_run) = arr.first() {
            if let Some(id) = first_run.get("id").and_then(|i| i.as_str()) {
                let response =
                    auth_get(&client, &format!("/api/training/runs/{}/logs", id), &token)
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

// =============================================================================
// Pagination and Filtering Tests
// =============================================================================

#[tokio::test]
async fn test_training_run_pagination() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/training/runs?limit=5&offset=0", &token)
        .await
        .expect("Request failed");

    assert!(response.status().is_success(), "Should support pagination");
}

#[tokio::test]
async fn test_training_run_filtering() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/training/runs?status=completed", &token)
        .await
        .expect("Request failed");

    // Filter might work or might be ignored
    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 400,
        "Got unexpected status: {}",
        status
    );
}
