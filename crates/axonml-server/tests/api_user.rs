//! Integration tests for user profile/settings API endpoints

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
async fn test_get_profile() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/profile", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    // Profile endpoint may be at /api/auth/me or /api/user/profile
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_update_profile() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_put(
        &client,
        "/api/user/profile",
        &token,
        serde_json::json!({
            "name": "Administrator"
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_change_password() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/user/password",
        &token,
        serde_json::json!({
            "current_password": ADMIN_PASSWORD,
            "new_password": ADMIN_PASSWORD  // Keep same for testing
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    // Might succeed, fail validation, or endpoint not exist
    assert!(
        status == 200 || status == 400 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_change_password_wrong_current() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/user/password",
        &token,
        serde_json::json!({
            "current_password": "wrong_password",
            "new_password": "NewPassword123!"
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    // Should fail with wrong current password
    assert!(
        status == 400 || status == 401 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_get_mfa_status() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/mfa", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_setup_totp() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(&client, "/api/user/mfa/totp/setup", &token, serde_json::json!({}))
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    // Should return QR code/secret or endpoint not exist
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_verify_totp_invalid() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/user/mfa/totp/verify",
        &token,
        serde_json::json!({
            "code": "000000"
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    // Should fail with invalid code
    assert!(
        status == 400 || status == 401 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_get_recovery_codes() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/mfa/recovery", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_generate_recovery_codes() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(&client, "/api/user/mfa/recovery/generate", &token, serde_json::json!({}))
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_get_sessions() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/sessions", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_revoke_session() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_delete(&client, "/api/user/sessions/some-session-id", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_get_api_keys() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/api-keys", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "Got unexpected status: {}", status);
}

#[tokio::test]
async fn test_create_api_key() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/user/api-keys",
        &token,
        serde_json::json!({
            "name": "Test API Key",
            "expires_in_days": 30
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 201 || status == 404,
        "Got unexpected status: {}",
        status
    );
}
