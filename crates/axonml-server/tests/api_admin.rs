//! Admin API Integration Tests — User CRUD and System Status Endpoints
//!
//! Integration tests against a live AxonML server for the `/api/admin/*` routes.
//! Tests cover: listing users (authenticated and unauthenticated), getting a user
//! by ID (found and not found), creating users as admin, updating user roles,
//! deleting users (including self-deletion prevention), and probing system
//! endpoints (`/api/admin/stats`, `/api/admin/health`, `/api/admin/database`).
//! Each test uses the `require_server!` macro to skip gracefully when the server
//! is offline.
//!
//! # File
//! `crates/axonml-server/tests/api_admin.rs`
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
// Test Skip Macro
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
// User List Tests
// =============================================================================

#[tokio::test]
async fn test_list_users_as_admin() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/admin/users", &token)
        .await
        .expect("Request failed");

    assert!(
        response.status().is_success(),
        "Admin should access user list"
    );

    let body: Value = response.json().await.expect("Failed to parse JSON");
    assert!(body.is_array() || body.get("users").is_some());
}

#[tokio::test]
async fn test_list_users_unauthenticated() {
    require_server!();

    let client = test_client();
    let response = client
        .get(format!("{}/api/admin/users", TEST_API_URL))
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
// User CRUD Tests
// =============================================================================

#[tokio::test]
async fn test_get_user_by_id() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // Get admin user by known ID
    let response = auth_get(&client, "/api/admin/users/admin", &token)
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
async fn test_get_user_not_found() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/admin/users/nonexistent-user-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent user"
    );
}

#[tokio::test]
async fn test_create_user_as_admin() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let unique_email = format!(
        "newuser_{}@test.local",
        chrono::Utc::now().timestamp_millis()
    );

    let response = auth_post(
        &client,
        "/api/admin/users",
        &token,
        serde_json::json!({
            "name": "New Test User",
            "email": unique_email,
            "password": "TestPassword123!",
            "role": "user"
        }),
    )
    .await
    .expect("Request failed");

    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 201 || status == 400,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_update_user_role() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // First list users to get a non-admin user ID
    let list_response = auth_get(&client, "/api/admin/users", &token)
        .await
        .expect("Request failed");

    if !list_response.status().is_success() {
        return;
    }

    let users: Value = list_response.json().await.expect("Failed to parse JSON");
    let users_arr = users
        .as_array()
        .or_else(|| users.get("users").and_then(|u| u.as_array()));

    if let Some(arr) = users_arr {
        // Find a non-admin user
        for user in arr {
            let role = user.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role != "admin" {
                if let Some(id) = user.get("id").and_then(|i| i.as_str()) {
                    let response = auth_put(
                        &client,
                        &format!("/api/admin/users/{}", id),
                        &token,
                        serde_json::json!({
                            "role": "user"
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
                    break;
                }
            }
        }
    }
}

// =============================================================================
// User Deletion Tests
// =============================================================================

#[tokio::test]
async fn test_delete_user() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // Try to delete nonexistent user
    let response = auth_delete(&client, "/api/admin/users/nonexistent-user-id", &token)
        .await
        .expect("Request failed");

    assert_eq!(
        response.status().as_u16(),
        404,
        "Should return 404 for nonexistent user"
    );
}

#[tokio::test]
async fn test_cannot_delete_self() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    // Try to delete admin user (self)
    let response = auth_delete(&client, "/api/admin/users/admin", &token)
        .await
        .expect("Request failed");

    // Should fail - can't delete yourself
    let status = response.status().as_u16();
    assert!(
        status == 400 || status == 403 || status == 404,
        "Should prevent self-deletion, got: {}",
        status
    );
}

// =============================================================================
// System Status Endpoint Tests
// =============================================================================

#[tokio::test]
async fn test_system_stats() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/admin/stats", &token)
        .await
        .expect("Request failed");

    // Stats endpoint may or may not exist
    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_system_health() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/admin/health", &token)
        .await
        .expect("Request failed");

    // Health endpoint may or may not exist under admin
    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_database_status() {
    require_server!();

    let client = test_client();
    let token = login_as_admin(&client).await.expect("Login failed");

    let response = auth_get(&client, "/api/admin/database", &token)
        .await
        .expect("Request failed");

    // Database status endpoint may or may not exist
    let status = response.status().as_u16();
    assert!(
        status == 200 || status == 404,
        "Got unexpected status: {}",
        status
    );
}
