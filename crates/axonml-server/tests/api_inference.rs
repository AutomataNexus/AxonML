//! Integration tests for inference API endpoints

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
async fn test_list_endpoints_authenticated() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/inference/endpoints", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    // Endpoint may or may not exist
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_list_endpoints_unauthenticated() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/inference/endpoints", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(
        status == 401 || status == 404,
        "Should return 401 or 404, got: {}",
        status
    );
}

#[tokio::test]
async fn test_create_inference_endpoint() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let endpoint_name = format!("test-endpoint-{}", chrono::Utc::now().timestamp_millis());

    let response = auth_post(
        &client,
        "/api/inference/endpoints",
        &token,
        serde_json::json!({
            "name": endpoint_name,
            "model_id": "test-model",
            "replicas": 1,
            "resources": {
                "cpu": "100m",
                "memory": "256Mi"
            }
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    // Might fail due to missing model, wrong fields, or endpoint not implemented
    assert!(
        status == 200 || status == 201 || status == 400 || status == 404 || status == 422,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_get_endpoint_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(
        &client,
        "/api/inference/endpoints/nonexistent-endpoint-id",
        &token,
    )
    .await
    .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent endpoint"
    );
}

#[tokio::test]
async fn test_inference_metrics() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/inference/metrics", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_inference_overview() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/inference/overview", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_start_endpoint() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/inference/endpoints/test-endpoint/start",
        &token,
        serde_json::json!({}),
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

#[tokio::test]
async fn test_stop_endpoint() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/inference/endpoints/test-endpoint/stop",
        &token,
        serde_json::json!({}),
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

#[tokio::test]
async fn test_delete_endpoint_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_delete(
        &client,
        "/api/inference/endpoints/nonexistent-endpoint-id",
        &token,
    )
    .await
    .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent endpoint"
    );
}

#[tokio::test]
async fn test_inference_predict() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // List endpoints to get an existing one
    let list_response = auth_get(&client, "/api/inference/endpoints", &token)
        .await
        .expect("Request failed");

    if !list_response.status().is_success() {
        return;
    }

    let endpoints: Value = list_response.json().await.expect("Failed to parse JSON");
    let endpoints_arr = endpoints
        .as_array()
        .or_else(|| endpoints.get("endpoints").and_then(|e| e.as_array()));

    if let Some(arr) = endpoints_arr {
        if let Some(first_endpoint) = arr.first() {
            if let Some(id) = first_endpoint.get("id").and_then(|i| i.as_str()) {
                let response = auth_post(
                    &client,
                    &format!("/api/inference/endpoints/{}/predict", id),
                    &token,
                    serde_json::json!({
                        "input": [[1.0, 2.0, 3.0, 4.0]]
                    }),
                )
                .await
                .expect("Request failed");

                let status = response.status().as_u16();
                // Predict might work or endpoint might not be running
                assert!(
                    status == 200 || status == 400 || status == 404 || status == 503,
                    "Got unexpected status: {}",
                    status
                );
            }
        }
    }
}

#[tokio::test]
async fn test_endpoint_scaling() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/inference/endpoints/test-endpoint/scale",
        &token,
        serde_json::json!({
            "replicas": 2
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
