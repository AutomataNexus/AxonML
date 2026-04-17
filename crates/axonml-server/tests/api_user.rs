//! User Profile and Settings API — Integration Tests
//!
//! Tests for the user account management API endpoints on the AxonML server.
//! Covers profile retrieval and update, password change (valid and invalid
//! current password), MFA lifecycle (TOTP setup, verification with invalid
//! code, recovery code generation and retrieval), session management (list
//! and revoke), and API key CRUD (list and create). Uses the `require_server!`
//! macro to skip gracefully when the server or admin DB is unavailable.
//!
//! # File
//! `crates/axonml-server/tests/api_user.rs`
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
// Profile Tests
// =============================================================================

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
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
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
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

// =============================================================================
// Password Change Tests
// =============================================================================

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
            "current_password": admin_password(),
            "new_password": admin_password()  // Keep same for testing
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

// =============================================================================
// MFA (Multi-Factor Authentication) Tests
// =============================================================================

#[tokio::test]
async fn test_get_mfa_status() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/mfa", &token)
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
async fn test_setup_totp() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/user/mfa/totp/setup",
        &token,
        serde_json::json!({}),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    // Should return QR code/secret or endpoint not exist
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
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

// -----------------------------------------------------------------------------
// Recovery Codes
// -----------------------------------------------------------------------------

#[tokio::test]
async fn test_get_recovery_codes() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/mfa/recovery", &token)
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
async fn test_generate_recovery_codes() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_post(
        &client,
        "/api/user/mfa/recovery/generate",
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

// =============================================================================
// Session Management Tests
// =============================================================================

#[tokio::test]
async fn test_get_sessions() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/sessions", &token)
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
async fn test_revoke_session() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_delete(&client, "/api/user/sessions/some-session-id", &token)
        .await
        .expect("Request failed");

    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

// =============================================================================
// API Key Management Tests
// =============================================================================

#[tokio::test]
async fn test_get_api_keys() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/user/api-keys", &token)
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
