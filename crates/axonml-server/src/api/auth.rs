//! Authentication API endpoints for AxonML
//!
//! Handles user registration, login, MFA, and session management.

use crate::api::AppState;
use crate::auth::{
    hash_password, verify_password, AuthError, AuthUser, RecoveryAuth, TotpAuth,
    WebAuthnAuth,
};
use crate::db::users::{NewUser, UpdateUser, UserRepository, UserRole};
use axum::{
    extract::{ConnectInfo, Path, State},
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Extract client IP address from headers or connection info.
/// Checks X-Forwarded-For, X-Real-IP headers (for proxy scenarios), then falls back to connection IP.
fn extract_client_ip(headers: &HeaderMap, conn_info: Option<&SocketAddr>) -> Option<String> {
    // Check X-Forwarded-For header (may contain multiple IPs, take the first)
    if let Some(xff) = headers.get("x-forwarded-for") {
        if let Ok(xff_str) = xff.to_str() {
            if let Some(first_ip) = xff_str.split(',').next() {
                return Some(first_ip.trim().to_string());
            }
        }
    }

    // Check X-Real-IP header
    if let Some(xri) = headers.get("x-real-ip") {
        if let Ok(ip) = xri.to_str() {
            return Some(ip.to_string());
        }
    }

    // Fall back to connection info
    conn_info.map(|addr| addr.ip().to_string())
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub name: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<i64>,
    pub requires_mfa: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mfa_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mfa_methods: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<UserResponse>,
}

#[derive(Debug, Deserialize)]
pub struct VerifyTotpRequest {
    pub mfa_token: String,
    pub code: String,
}

#[derive(Debug, Deserialize)]
pub struct RefreshRequest {
    pub refresh_token: String,
}

#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub token: String,
    pub refresh_token: String,
    pub expires_in: i64,
    pub token_type: String,
}

#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub id: String,
    pub email: String,
    pub name: String,
    pub role: String,
    pub mfa_enabled: bool,
    pub mfa_verified: bool,
    pub totp_enabled: bool,
    pub webauthn_enabled: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct TotpSetupResponse {
    pub secret: String,
    pub qr_code: String,
    pub otpauth_url: String,
}

#[derive(Debug, Deserialize)]
pub struct EnableTotpRequest {
    pub code: String,
    pub secret: String,
}

#[derive(Debug, Serialize)]
pub struct RecoveryCodesResponse {
    pub codes: Vec<String>,
    pub formatted: String,
}

#[derive(Debug, Deserialize)]
pub struct UseRecoveryCodeRequest {
    pub mfa_token: String,
    pub code: String,
}

#[derive(Debug, Deserialize)]
pub struct WebAuthnStartRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mfa_token: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WebAuthnChallengeResponse {
    pub challenge: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct WebAuthnFinishRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mfa_token: Option<String>,
    pub response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub email: String,
    pub name: String,
    pub password: String,
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

// ============================================================================
// Handlers
// ============================================================================

/// Register a new user
pub async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<(StatusCode, Json<RegisterResponse>), AuthError> {
    // Check if public registration is allowed
    if !state.config.auth.allow_public_registration {
        return Err(AuthError::Forbidden("Public registration is disabled".to_string()));
    }

    let repo = UserRepository::new(&state.db);

    // Hash password
    let password_hash = hash_password(&req.password)?;

    // Check if email already exists
    if let Some(_) = repo.find_by_email(&req.email).await.map_err(|e| AuthError::Internal(e.to_string()))? {
        return Err(AuthError::Forbidden("Email address is already registered".to_string()));
    }

    // Create user (with email_pending=true, email_verified=false, verification_token set)
    let user = repo
        .create(NewUser {
            email: req.email.clone(),
            name: req.name.clone(),
            password_hash,
            role: UserRole::User,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Send verification email to user
    let base_url = format!("http://{}:{}", state.config.server.host, state.config.server.port);
    let verification_token = user.verification_token.as_ref().ok_or_else(||
        AuthError::Internal("Verification token not generated".to_string())
    )?;

    if let Err(e) = state.email.send_verification_email(
        &user.email,
        &user.name,
        verification_token,
        &base_url,
    ).await {
        tracing::error!("Failed to send verification email: {}", e);
        // Don't fail registration if email fails, user can request new verification email
    }

    // Send notification email to admin
    if let Err(e) = state.email.send_admin_signup_notification(
        &user.email,
        &user.name,
    ).await {
        tracing::error!("Failed to send admin notification: {}", e);
    }

    Ok((
        StatusCode::CREATED,
        Json(RegisterResponse {
            success: true,
            message: "Registration successful! Please check your email to verify your account.".to_string(),
        }),
    ))
}

/// Login user
pub async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AuthError> {
    let repo = UserRepository::new(&state.db);

    // Find user by email or username (name field)
    let user = repo
        .find_by_email(&req.email)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let user = if user.is_none() {
        // Try to find by name (username) if email lookup failed
        repo.find_by_name(&req.email)
            .await
            .map_err(|e| AuthError::Internal(e.to_string()))?
    } else {
        user
    };

    let user = user.ok_or(AuthError::InvalidCredentials)?;

    // Verify password
    if !verify_password(&req.password, &user.password_hash)? {
        return Err(AuthError::InvalidCredentials);
    }

    // Check email verification status (skip for admins)
    if user.role != UserRole::Admin && (user.email_pending || !user.email_verified) {
        if user.email_pending && !user.email_verified {
            return Err(AuthError::Forbidden("Please verify your email address before logging in. Check your inbox for the verification link.".to_string()));
        } else if user.email_verified && user.email_pending {
            return Err(AuthError::Forbidden("Your account is pending admin approval. You will receive an email once approved.".to_string()));
        }
    }

    // Check if MFA is required
    if user.mfa_enabled {
        let mfa_token = state
            .jwt
            .create_mfa_token(&user.id, &user.email)
            .map_err(|e| AuthError::Internal(e.to_string()))?;

        let mut mfa_methods = Vec::new();
        if user.totp_secret.is_some() {
            mfa_methods.push("totp".to_string());
        }
        if !user.webauthn_credentials.is_empty() {
            mfa_methods.push("webauthn".to_string());
        }
        if !user.recovery_codes.is_empty() {
            mfa_methods.push("recovery".to_string());
        }

        return Ok(Json(LoginResponse {
            access_token: None,
            refresh_token: None,
            expires_in: None,
            requires_mfa: true,
            mfa_token: Some(mfa_token.mfa_token),
            mfa_methods: Some(mfa_methods),
            user: None,
        }));
    }

    // Generate tokens
    let role = format!("{:?}", user.role).to_lowercase();
    let token_pair = state
        .jwt
        .create_token_pair(&user.id, &user.email, &role, true)?;

    Ok(Json(LoginResponse {
        access_token: Some(token_pair.access_token),
        refresh_token: Some(token_pair.refresh_token),
        expires_in: Some(token_pair.expires_in),
        requires_mfa: false,
        mfa_token: None,
        mfa_methods: None,
        user: Some(UserResponse {
            id: user.id.clone(),
            email: user.email.clone(),
            name: user.name.clone(),
            role: role.clone(),
            mfa_enabled: user.mfa_enabled,
            mfa_verified: true, // MFA verified since no MFA was required or already passed
            totp_enabled: user.totp_secret.is_some(),
            webauthn_enabled: !user.webauthn_credentials.is_empty(),
            created_at: user.created_at.to_rfc3339(),
            updated_at: user.updated_at.to_rfc3339(),
        }),
    }))
}

/// Verify TOTP code
pub async fn verify_totp(
    State(state): State<AppState>,
    Json(req): Json<VerifyTotpRequest>,
) -> Result<Json<TokenResponse>, AuthError> {
    // Validate MFA token
    let claims = state.jwt.validate_mfa_token(&req.mfa_token)?;

    let repo = UserRepository::new(&state.db);
    let user = repo
        .find_by_id(&claims.sub)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::UserNotFound)?;

    // Get TOTP secret
    let secret = user.totp_secret.ok_or(AuthError::InvalidMfaCode)?;

    // Verify TOTP code
    let totp = TotpAuth::new("AxonML");
    if !totp.verify(&secret, &req.code, &user.email)? {
        return Err(AuthError::InvalidMfaCode);
    }

    // Generate tokens
    let role = format!("{:?}", user.role).to_lowercase();
    let token_pair = state
        .jwt
        .create_token_pair(&user.id, &user.email, &role, true)?;

    Ok(Json(TokenResponse {
        token: token_pair.access_token,
        refresh_token: token_pair.refresh_token,
        expires_in: token_pair.expires_in,
        token_type: token_pair.token_type,
    }))
}

/// Logout user
pub async fn logout(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<StatusCode, AuthError> {
    // In a production system, we'd invalidate the token/session
    // For now, just return success (client should discard token)
    let _ = state.db.kv_delete(&format!("session:{}", user.id)).await;
    Ok(StatusCode::NO_CONTENT)
}

/// Refresh access token
pub async fn refresh(
    State(state): State<AppState>,
    Json(req): Json<RefreshRequest>,
) -> Result<Json<TokenResponse>, AuthError> {
    let token_pair = state.jwt.refresh_access_token(&req.refresh_token)?;

    Ok(Json(TokenResponse {
        token: token_pair.access_token,
        refresh_token: token_pair.refresh_token,
        expires_in: token_pair.expires_in,
        token_type: token_pair.token_type,
    }))
}

/// Get current user
pub async fn me(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<UserResponse>, AuthError> {
    let repo = UserRepository::new(&state.db);
    let user_data = repo
        .find_by_id(&user.id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::UserNotFound)?;

    Ok(Json(UserResponse {
        id: user_data.id,
        email: user_data.email,
        name: user_data.name,
        role: format!("{:?}", user_data.role).to_lowercase(),
        mfa_enabled: user_data.mfa_enabled,
        mfa_verified: user.mfa_verified,
        totp_enabled: user_data.totp_secret.is_some(),
        webauthn_enabled: !user_data.webauthn_credentials.is_empty(),
        created_at: user_data.created_at.to_rfc3339(),
        updated_at: user_data.updated_at.to_rfc3339(),
    }))
}

/// Setup TOTP
pub async fn setup_totp(
    State(_state): State<AppState>,
    user: AuthUser,
) -> Result<Json<TotpSetupResponse>, AuthError> {
    let totp = TotpAuth::new("AxonML");
    let setup = totp.setup(&user.email)?;

    Ok(Json(TotpSetupResponse {
        secret: setup.secret,
        qr_code: setup.qr_code_data_url,
        otpauth_url: setup.otpauth_url,
    }))
}

/// Enable TOTP
pub async fn enable_totp(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<EnableTotpRequest>,
) -> Result<Json<RecoveryCodesResponse>, AuthError> {
    // Verify the code first
    let totp = TotpAuth::new("AxonML");
    if !totp.verify(&req.secret, &req.code, &user.email)? {
        return Err(AuthError::InvalidMfaCode);
    }

    // Generate recovery codes
    let recovery = RecoveryAuth::generate_codes(8)?;

    // Enable TOTP using dedicated repository method
    let repo = UserRepository::new(&state.db);
    repo.enable_totp(&user.id, &req.secret)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Set recovery codes
    repo.set_recovery_codes(&user.id, recovery.hashed_codes)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(RecoveryCodesResponse {
        formatted: RecoveryAuth::format_for_display(&recovery.codes),
        codes: recovery.codes,
    }))
}

/// Disable MFA for current user
pub async fn disable_mfa(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<StatusCode, AuthError> {
    let repo = UserRepository::new(&state.db);
    repo.disable_mfa(&user.id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Generate new recovery codes
pub async fn generate_recovery_codes(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<RecoveryCodesResponse>, AuthError> {
    let recovery = RecoveryAuth::generate_codes(8)?;

    let repo = UserRepository::new(&state.db);
    repo.set_recovery_codes(&user.id, recovery.hashed_codes)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(RecoveryCodesResponse {
        formatted: RecoveryAuth::format_for_display(&recovery.codes),
        codes: recovery.codes,
    }))
}

/// Use recovery code
pub async fn use_recovery_code(
    State(state): State<AppState>,
    Json(req): Json<UseRecoveryCodeRequest>,
) -> Result<Json<TokenResponse>, AuthError> {
    // Validate MFA token
    let claims = state.jwt.validate_mfa_token(&req.mfa_token)?;

    let repo = UserRepository::new(&state.db);
    let user = repo
        .find_by_id(&claims.sub)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::UserNotFound)?;

    // Verify recovery code and get the matching hash index
    let index = RecoveryAuth::verify_code(&req.code, &user.recovery_codes)?
        .ok_or(AuthError::InvalidMfaCode)?;

    // Get the hash of the used code
    let code_hash = user.recovery_codes.get(index)
        .ok_or(AuthError::InvalidMfaCode)?
        .clone();

    // Remove the used code using dedicated repository method
    let removed = repo.use_recovery_code(&user.id, &code_hash)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    if !removed {
        return Err(AuthError::InvalidMfaCode);
    }

    // Generate tokens
    let role = format!("{:?}", user.role).to_lowercase();
    let token_pair = state
        .jwt
        .create_token_pair(&user.id, &user.email, &role, true)?;

    Ok(Json(TokenResponse {
        token: token_pair.access_token,
        refresh_token: token_pair.refresh_token,
        expires_in: token_pair.expires_in,
        token_type: token_pair.token_type,
    }))
}

/// Start WebAuthn registration
pub async fn webauthn_register_start(
    State(_state): State<AppState>,
    user: AuthUser,
) -> Result<Json<WebAuthnChallengeResponse>, AuthError> {
    let webauthn = WebAuthnAuth::new("localhost", "AxonML", "http://localhost:8080");
    let challenge = webauthn.start_registration(&user.id, &user.email, &user.email)?;

    Ok(Json(WebAuthnChallengeResponse {
        challenge: serde_json::to_value(challenge)
            .map_err(|e| AuthError::Internal(e.to_string()))?,
    }))
}

/// Finish WebAuthn registration
pub async fn webauthn_register_finish(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<WebAuthnFinishRequest>,
) -> Result<Json<RecoveryCodesResponse>, AuthError> {
    let webauthn = WebAuthnAuth::new("localhost", "AxonML", "http://localhost:8080");

    // Parse response
    let response: crate::auth::webauthn::RegistrationResponse =
        serde_json::from_value(req.response)
            .map_err(|e| AuthError::Internal(e.to_string()))?;

    let name = req.name.unwrap_or_else(|| "Security Key".to_string());
    let credential = webauthn.finish_registration("", &response, &name)?;

    // Generate recovery codes if first MFA method
    let recovery = RecoveryAuth::generate_codes(8)?;

    // Update user
    let repo = UserRepository::new(&state.db);
    let mut user_data = repo
        .find_by_id(&user.id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::UserNotFound)?;

    user_data
        .webauthn_credentials
        .push(serde_json::to_value(&credential).unwrap());

    repo.update(
        &user.id,
        UpdateUser {
            mfa_enabled: Some(true),
            webauthn_credentials: Some(user_data.webauthn_credentials),
            recovery_codes: Some(recovery.hashed_codes),
            ..Default::default()
        },
    )
    .await
    .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(RecoveryCodesResponse {
        formatted: RecoveryAuth::format_for_display(&recovery.codes),
        codes: recovery.codes,
    }))
}

/// Start WebAuthn authentication
pub async fn webauthn_auth_start(
    State(state): State<AppState>,
    Json(req): Json<WebAuthnStartRequest>,
) -> Result<Json<WebAuthnChallengeResponse>, AuthError> {
    let mfa_token = req.mfa_token.ok_or(AuthError::InvalidToken)?;
    let claims = state.jwt.validate_mfa_token(&mfa_token)?;

    let repo = UserRepository::new(&state.db);
    let user = repo
        .find_by_id(&claims.sub)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::UserNotFound)?;

    // Parse stored credentials
    let credentials: Vec<crate::auth::webauthn::WebAuthnCredential> = user
        .webauthn_credentials
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect();

    if credentials.is_empty() {
        return Err(AuthError::InvalidMfaCode);
    }

    let webauthn = WebAuthnAuth::new("localhost", "AxonML", "http://localhost:8080");
    let challenge = webauthn.start_authentication(&credentials)?;

    Ok(Json(WebAuthnChallengeResponse {
        challenge: serde_json::to_value(challenge)
            .map_err(|e| AuthError::Internal(e.to_string()))?,
    }))
}

/// Finish WebAuthn authentication
pub async fn webauthn_auth_finish(
    State(state): State<AppState>,
    Json(req): Json<WebAuthnFinishRequest>,
) -> Result<Json<TokenResponse>, AuthError> {
    let mfa_token = req.mfa_token.ok_or(AuthError::InvalidToken)?;
    let claims = state.jwt.validate_mfa_token(&mfa_token)?;

    let repo = UserRepository::new(&state.db);
    let user = repo
        .find_by_id(&claims.sub)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::UserNotFound)?;

    // Parse response and credentials
    let response: crate::auth::webauthn::AuthenticationResponse =
        serde_json::from_value(req.response)
            .map_err(|e| AuthError::Internal(e.to_string()))?;

    let credentials: Vec<crate::auth::webauthn::WebAuthnCredential> = user
        .webauthn_credentials
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect();

    let webauthn = WebAuthnAuth::new("localhost", "AxonML", "http://localhost:8080");
    let _updated_cred = webauthn.finish_authentication("", &response, &credentials)?;

    // Generate tokens
    let role = format!("{:?}", user.role).to_lowercase();
    let token_pair = state
        .jwt
        .create_token_pair(&user.id, &user.email, &role, true)?;

    Ok(Json(TokenResponse {
        token: token_pair.access_token,
        refresh_token: token_pair.refresh_token,
        expires_in: token_pair.expires_in,
        token_type: token_pair.token_type,
    }))
}

// ============================================================================
// Admin Handlers
// ============================================================================

/// List all users (admin only)
pub async fn list_users(
    State(state): State<AppState>,
) -> Result<Json<Vec<UserResponse>>, AuthError> {
    let repo = UserRepository::new(&state.db);
    let users = repo
        .list(Some(100), Some(0))
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let response: Vec<UserResponse> = users
        .into_iter()
        .map(|u| UserResponse {
            id: u.id,
            email: u.email,
            name: u.name,
            role: format!("{:?}", u.role).to_lowercase(),
            mfa_enabled: u.mfa_enabled,
            mfa_verified: false, // N/A for admin user listing
            totp_enabled: u.totp_secret.is_some(),
            webauthn_enabled: !u.webauthn_credentials.is_empty(),
            created_at: u.created_at.to_rfc3339(),
            updated_at: u.updated_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(response))
}

/// Create user (admin only)
pub async fn create_user(
    State(state): State<AppState>,
    Json(req): Json<CreateUserRequest>,
) -> Result<(StatusCode, Json<UserResponse>), AuthError> {
    let repo = UserRepository::new(&state.db);
    let password_hash = hash_password(&req.password)?;

    let role = match req.role.as_str() {
        "admin" => UserRole::Admin,
        "viewer" => UserRole::Viewer,
        _ => UserRole::User,
    };

    let user = repo
        .create(NewUser {
            email: req.email,
            name: req.name,
            password_hash,
            role,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(UserResponse {
            id: user.id.clone(),
            email: user.email.clone(),
            name: user.name.clone(),
            role: format!("{:?}", user.role).to_lowercase(),
            mfa_enabled: user.mfa_enabled,
            mfa_verified: false, // New user, not yet verified MFA
            totp_enabled: user.totp_secret.is_some(),
            webauthn_enabled: !user.webauthn_credentials.is_empty(),
            created_at: user.created_at.to_rfc3339(),
            updated_at: user.updated_at.to_rfc3339(),
        }),
    ))
}

/// Get user by ID (admin only)
pub async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<UserResponse>, AuthError> {
    let repo = UserRepository::new(&state.db);

    let user = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::UserNotFound)?;

    Ok(Json(UserResponse {
        id: user.id,
        email: user.email,
        name: user.name,
        role: format!("{:?}", user.role).to_lowercase(),
        mfa_enabled: user.mfa_enabled,
        mfa_verified: false, // N/A for admin lookup
        totp_enabled: user.totp_secret.is_some(),
        webauthn_enabled: !user.webauthn_credentials.is_empty(),
        created_at: user.created_at.to_rfc3339(),
        updated_at: user.updated_at.to_rfc3339(),
    }))
}

/// Update user (admin only)
pub async fn update_user(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateUserRequest>,
) -> Result<Json<UserResponse>, AuthError> {
    let repo = UserRepository::new(&state.db);

    let mut update = UpdateUser::default();

    if let Some(name) = req.name {
        update.name = Some(name);
    }
    if let Some(email) = req.email {
        update.email = Some(email);
    }
    if let Some(password) = req.password {
        update.password_hash = Some(hash_password(&password)?);
    }
    if let Some(role) = req.role {
        update.role = Some(match role.as_str() {
            "admin" => UserRole::Admin,
            "viewer" => UserRole::Viewer,
            _ => UserRole::User,
        });
    }

    let user = repo
        .update(&id, update)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(UserResponse {
        id: user.id,
        email: user.email,
        name: user.name,
        role: format!("{:?}", user.role).to_lowercase(),
        mfa_enabled: user.mfa_enabled,
        mfa_verified: false, // N/A for admin update
        totp_enabled: user.totp_secret.is_some(),
        webauthn_enabled: !user.webauthn_credentials.is_empty(),
        created_at: user.created_at.to_rfc3339(),
        updated_at: user.updated_at.to_rfc3339(),
    }))
}

/// Delete user (admin only)
pub async fn delete_user(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<StatusCode, AuthError> {
    // Prevent self-deletion
    if user.id == id {
        return Err(AuthError::Forbidden("Cannot delete your own account".into()));
    }

    let repo = UserRepository::new(&state.db);

    // Check if user exists first
    let existing = repo.find_by_id(&id).await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    if existing.is_none() {
        return Err(AuthError::UserNotFound);
    }

    repo.delete(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Verify email endpoint - User clicks link in verification email
pub async fn verify_email(
    State(state): State<AppState>,
    headers: HeaderMap,
    conn_info: Option<ConnectInfo<SocketAddr>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<axum::response::Redirect, AuthError> {
    // Extract client IP for admin notification
    let client_ip = extract_client_ip(&headers, conn_info.as_ref().map(|c| &c.0));
    let token = params.get("token")
        .ok_or_else(|| AuthError::InvalidToken)?;

    let repo = UserRepository::new(&state.db);

    // Find user by verification token
    let filter = serde_json::json!({
        "verification_token": { "$eq": token }
    });

    let user_doc = state.db.doc_find_one("axonml_users", filter).await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::InvalidToken)?;

    let user: crate::db::users::User = serde_json::from_value(user_doc)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Check if already verified
    if user.email_verified {
        // Redirect to login with message
        return Ok(axum::response::Redirect::to("/login?already_verified=true"));
    }

    // Generate approval token for admin
    let approval_token = uuid::Uuid::new_v4().to_string();

    // Update user - email verified but still pending admin approval
    repo.update(&user.id, UpdateUser {
        verification_token: Some(approval_token.clone()),
        ..Default::default()
    }).await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Send approval request to admin
    let base_url = format!("http://{}:{}", state.config.server.host, state.config.server.port);

    // Location lookup could be done via IP geolocation service if needed
    // For now, we pass the IP which admin can look up manually
    if let Err(e) = state.email.send_admin_approval_request(
        &user.id,
        &user.email,
        &user.name,
        None, // location - would require external geolocation API
        client_ip.as_deref(),
        &approval_token,
        &base_url,
    ).await {
        tracing::error!("Failed to send admin approval request: {}", e);
    }

    // Redirect to login page with success message
    Ok(axum::response::Redirect::to("/login?email_verified=true"))
}

/// Approve user endpoint - Admin clicks link in approval email
pub async fn approve_user(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> axum::response::Html<String> {
    let token = match params.get("token") {
        Some(t) => t,
        None => {
            return axum::response::Html(
                "<html><body><h1>Invalid approval link</h1><p>The approval token is missing.</p></body></html>".to_string()
            );
        }
    };

    let repo = UserRepository::new(&state.db);

    // Find user by approval token (stored in verification_token field after email verification)
    let filter = serde_json::json!({
        "verification_token": { "$eq": token }
    });

    let user_doc = match state.db.doc_find_one("axonml_users", filter).await {
        Ok(Some(doc)) => doc,
        Ok(None) => {
            return axum::response::Html(
                "<html><body><h1>Invalid approval link</h1><p>No user found with this token.</p></body></html>".to_string()
            );
        }
        Err(e) => {
            tracing::error!("Failed to find user: {}", e);
            return axum::response::Html(
                "<html><body><h1>Error</h1><p>Failed to process approval request.</p></body></html>".to_string()
            );
        }
    };

    let user: crate::db::users::User = match serde_json::from_value(user_doc) {
        Ok(u) => u,
        Err(e) => {
            tracing::error!("Failed to parse user: {}", e);
            return axum::response::Html(
                "<html><body><h1>Error</h1><p>Failed to process user data.</p></body></html>".to_string()
            );
        }
    };

    // Check if already approved
    if user.email_verified && !user.email_pending {
        return axum::response::Html(format!(
            r#"<html>
            <head><title>Already Approved - AxonML</title></head>
            <body style="font-family: 'Inter', sans-serif; max-width: 600px; margin: 50px auto; padding: 20px;">
                <h1 style="color: #14b8a6;">User Already Approved</h1>
                <p>The user <strong>{}</strong> ({}) has already been approved.</p>
            </body>
            </html>"#,
            user.name, user.email
        ));
    }

    // Approve user - set email_verified=true, email_pending=false
    if let Err(e) = repo.update(&user.id, UpdateUser {
        email_verified: Some(true),
        email_pending: Some(false),
        verification_token: None, // Clear token after use
        ..Default::default()
    }).await {
        tracing::error!("Failed to approve user: {}", e);
        return axum::response::Html(
            "<html><body><h1>Error</h1><p>Failed to approve user.</p></body></html>".to_string()
        );
    }

    // Send welcome email to user
    let dashboard_url = format!("http://{}:{}", state.config.server.host, state.config.dashboard.port);
    if let Err(e) = state.email.send_welcome_email(
        &user.email,
        &user.name,
        &dashboard_url,
    ).await {
        tracing::error!("Failed to send welcome email: {}", e);
    }

    // Return success page
    axum::response::Html(format!(
        r#"<html>
        <head>
            <title>User Approved - AxonML</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; background-color: #faf9f6; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 50px auto; background: white; border-radius: 12px; padding: 40px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #14b8a6; font-size: 32px; margin: 0;">AxonML</h1>
                </div>
                <h2 style="color: #111827; font-size: 24px;">✓ User Approved Successfully</h2>
                <div style="background-color: #f0fdfa; border-left: 4px solid #14b8a6; padding: 16px; margin: 24px 0; border-radius: 4px;">
                    <p style="margin: 8px 0;"><strong>User:</strong> {}</p>
                    <p style="margin: 8px 0;"><strong>Email:</strong> {}</p>
                </div>
                <p style="color: #6b7280;">The user has been granted access to AxonML and will receive a welcome email shortly.</p>
                <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                    <p style="color: #9ca3af; font-size: 12px;">Secured by AutomataNexus</p>
                </div>
            </div>
        </body>
        </html>"#,
        user.name, user.email
    ))
}
