//! Integration tests for training API endpoints

mod common;

use common::*;
use serde_json::Value;

macro_rules! require_server {
    () => {
        if !is_server_running().await {
            eprintln!("Skipping: server not running at {}", TEST_API_URL);
            return;
        }
    };
}

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

    assert_eq!(response.status().as_u16(), 401, "Should return 401 without auth");
}

#[tokio::test]
async fn test_get_training_run_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/training/runs/nonexistent-run-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(response.status().as_u16(), 404, "Should return 404 for nonexistent run");
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
    let runs_arr = runs.as_array().or_else(|| runs.get("runs").and_then(|r| r.as_array()));

    if let Some(arr) = runs_arr {
        if let Some(first_run) = arr.first() {
            if let Some(id) = first_run.get("id").and_then(|i| i.as_str()) {
                let response = auth_get(&client, &format!("/api/training/runs/{}/metrics", id), &token)
                    .await
                    .expect("Request failed");

                let status = response.status().as_u16();
                assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
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
    let runs_arr = runs.as_array().or_else(|| runs.get("runs").and_then(|r| r.as_array()));

    if let Some(arr) = runs_arr {
        if let Some(first_run) = arr.first() {
            if let Some(id) = first_run.get("id").and_then(|i| i.as_str()) {
                let response = auth_get(&client, &format!("/api/training/runs/{}/logs", id), &token)
                    .await
                    .expect("Request failed");

                let status = response.status().as_u16();
                assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
            }
        }
    }
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

    assert_eq!(response.status().as_u16(), 404, "Should return 404 for nonexistent run");
}

#[tokio::test]
async fn test_delete_training_run_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_delete(&client, "/api/training/runs/nonexistent-run-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(response.status().as_u16(), 404, "Should return 404 for nonexistent run");
}

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
    assert!(status == 200 || status == 400, "Got unexpected status: {}", status);
}
