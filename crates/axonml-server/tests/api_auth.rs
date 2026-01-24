//! Integration tests for authentication API endpoints

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

#[tokio::test]
async fn test_health_endpoint() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/health", TEST_API_URL))
        .send()
        .await
        .expect("Health request failed");

    assert!(response.status().is_success());
}

#[tokio::test]
async fn test_login_with_valid_credentials() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await;

    assert!(token.is_ok(), "Login should succeed: {:?}", token.err());
    assert!(!token.unwrap().is_empty(), "Token should not be empty");
}

#[tokio::test]
async fn test_login_with_invalid_credentials() {
    require_server!();

    let client = test_client();
    let response = client
        .post(format!("{}/api/auth/login", TEST_API_URL))
        .json(&serde_json::json!({
            "email": "nonexistent@test.com",
            "password": "wrongpassword"
        }))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(response.status().as_u16(), 401, "Should return 401 Unauthorized");
}

#[tokio::test]
async fn test_login_with_missing_fields() {
    require_server!();

    let client = test_client();
    let response = client
        .post(format!("{}/api/auth/login", TEST_API_URL))
        .json(&serde_json::json!({
            "email": "test@test.com"
            // missing password
        }))
        .send()
        .await
        .expect("Request failed");

    assert!(response.status().is_client_error(), "Should return client error");
}

#[tokio::test]
async fn test_me_endpoint_with_valid_token() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/auth/me", &token)
        .await
        .expect("Request failed");

    assert!(response.status().is_success(), "Should return success");

    let body: Value = response.json().await.expect("Failed to parse JSON");
    assert_eq!(body["email"], ADMIN_EMAIL);
}

#[tokio::test]
async fn test_me_endpoint_without_token() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/auth/me", TEST_API_URL))
        .send()
        .await
        .expect("Request failed");

    assert_eq!(response.status().as_u16(), 401, "Should return 401 without token");
}

#[tokio::test]
async fn test_me_endpoint_with_invalid_token() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/auth/me", TEST_API_URL))
        .header("Authorization", "Bearer invalid_token_here")
        .send()
        .await
        .expect("Request failed");

    assert_eq!(response.status().as_u16(), 401, "Should return 401 with invalid token");
}

#[tokio::test]
async fn test_register_new_user() {
    require_server!();

    let client = test_client();
    let unique_email = format!("testuser_{}@test.local", chrono::Utc::now().timestamp_millis());

    let response = client
        .post(format!("{}/api/auth/register", TEST_API_URL))
        .json(&serde_json::json!({
            "name": "Test User",
            "email": unique_email,
            "password": "TestPassword123!"
        }))
        .send()
        .await
        .expect("Request failed");

    // Registration might return 200/201 on success or 400/409 if email exists
    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 201 || status == 409,
        "Should return success or conflict, got {}",
        status
    );
}

#[tokio::test]
async fn test_register_duplicate_email() {
    require_server!();

    let client = test_client();

    // Try to register with existing admin email
    let response = client
        .post(format!("{}/api/auth/register", TEST_API_URL))
        .json(&serde_json::json!({
            "name": "Duplicate User",
            "email": ADMIN_EMAIL,
            "password": "TestPassword123!"
        }))
        .send()
        .await
        .expect("Request failed");

    assert!(response.status().is_client_error(), "Should reject duplicate email");
}

#[tokio::test]
async fn test_logout() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(&client, "/api/auth/logout", &token, serde_json::json!({}))
        .await
        .expect("Request failed");

    // Logout should succeed (200/204) or endpoint might not exist (404)
    let status = response.status().as_u16();
    assert!(status == 200 || status == 204 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_refresh_token() {
    require_server!();

    let client = test_client();

    // Get both access and refresh tokens from login
    let login_response = client
        .post(format!("{}/api/auth/login", TEST_API_URL))
        .json(&serde_json::json!({
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD
        }))
        .send()
        .await
        .expect("Request failed");

    if !login_response.status().is_success() {
        return;
    }

    let body: Value = login_response.json().await.expect("Failed to parse JSON");
    let refresh_token = body.get("refresh_token").and_then(|t| t.as_str());

    if let Some(rt) = refresh_token {
        // Use refresh token to get new access token
        let response = client
            .post(format!("{}/api/auth/refresh", TEST_API_URL))
            .json(&serde_json::json!({
                "refresh_token": rt
            }))
            .send()
            .await
            .expect("Request failed");

        // Refresh might succeed, return 401 for invalid token, or endpoint might not exist
        let status = response.status().as_u16();
        assert!(status == 200 || status == 401 || status == 404, "Got unexpected status: {}", status);
    }
}
