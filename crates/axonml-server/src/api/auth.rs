//! Authentication API endpoints for AxonML
//!
//! Handles user registration, login, MFA, and session management.

use crate::api::AppState;
use crate::auth::{
    hash_password, verify_password, AuthError, AuthUser, JwtAuth, RecoveryAuth, TotpAuth,
    WebAuthnAuth,
};
use crate::db::users::{NewUser, UpdateUser, UserRepository, UserRole};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub name: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<i64>,
    pub requires_mfa: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mfa_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mfa_methods: Option<Vec<String>>,
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
    pub created_at: String,
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
) -> Result<(StatusCode, Json<UserResponse>), AuthError> {
    let repo = UserRepository::new(&state.db);

    // Hash password
    let password_hash = hash_password(&req.password)?;

    // Create user
    let user = repo
        .create(NewUser {
            email: req.email,
            name: req.name,
            password_hash,
            role: UserRole::User,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(UserResponse {
            id: user.id,
            email: user.email,
            name: user.name,
            role: format!("{:?}", user.role).to_lowercase(),
            mfa_enabled: user.mfa_enabled,
            created_at: user.created_at.to_rfc3339(),
        }),
    ))
}

/// Login user
pub async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AuthError> {
    let repo = UserRepository::new(&state.db);

    // Find user by email
    let user = repo
        .find_by_email(&req.email)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::InvalidCredentials)?;

    // Verify password
    if !verify_password(&req.password, &user.password_hash)? {
        return Err(AuthError::InvalidCredentials);
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
            token: None,
            refresh_token: None,
            expires_in: None,
            requires_mfa: true,
            mfa_token: Some(mfa_token.mfa_token),
            mfa_methods: Some(mfa_methods),
        }));
    }

    // Generate tokens
    let role = format!("{:?}", user.role).to_lowercase();
    let token_pair = state
        .jwt
        .create_token_pair(&user.id, &user.email, &role, true)?;

    Ok(Json(LoginResponse {
        token: Some(token_pair.access_token),
        refresh_token: Some(token_pair.refresh_token),
        expires_in: Some(token_pair.expires_in),
        requires_mfa: false,
        mfa_token: None,
        mfa_methods: None,
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
        created_at: user_data.created_at.to_rfc3339(),
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
        secret: setup.secret_base32,
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

    // Update user
    let repo = UserRepository::new(&state.db);
    repo.update(
        &user.id,
        UpdateUser {
            mfa_enabled: Some(true),
            totp_secret: Some(req.secret),
            recovery_codes: Some(recovery.hashed_codes),
            ..Default::default()
        },
    )
    .await
    .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(RecoveryCodesResponse {
        codes: recovery.codes,
    }))
}

/// Generate new recovery codes
pub async fn generate_recovery_codes(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<RecoveryCodesResponse>, AuthError> {
    let recovery = RecoveryAuth::generate_codes(8)?;

    let repo = UserRepository::new(&state.db);
    repo.update(
        &user.id,
        UpdateUser {
            recovery_codes: Some(recovery.hashed_codes),
            ..Default::default()
        },
    )
    .await
    .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(RecoveryCodesResponse {
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

    // Verify recovery code
    let index = RecoveryAuth::verify_code(&req.code, &user.recovery_codes)?
        .ok_or(AuthError::InvalidMfaCode)?;

    // Remove used code
    let mut remaining_codes = user.recovery_codes.clone();
    remaining_codes.remove(index);

    repo.update(
        &user.id,
        UpdateUser {
            recovery_codes: Some(remaining_codes),
            ..Default::default()
        },
    )
    .await
    .map_err(|e| AuthError::Internal(e.to_string()))?;

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
            created_at: u.created_at.to_rfc3339(),
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
            id: user.id,
            email: user.email,
            name: user.name,
            role: format!("{:?}", user.role).to_lowercase(),
            mfa_enabled: user.mfa_enabled,
            created_at: user.created_at.to_rfc3339(),
        }),
    ))
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
        created_at: user.created_at.to_rfc3339(),
    }))
}

/// Delete user (admin only)
pub async fn delete_user(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AuthError> {
    let repo = UserRepository::new(&state.db);
    repo.delete(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}
