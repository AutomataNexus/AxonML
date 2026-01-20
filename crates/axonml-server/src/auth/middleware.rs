//! Authentication middleware for AxonML
//!
//! Provides Axum middleware for JWT authentication.

use super::{jwt::JwtAuth, AuthError, Claims};
use axum::{
    body::Body,
    extract::{FromRequestParts, State},
    http::{header::AUTHORIZATION, request::Parts, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Authenticated user extracted from JWT
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub id: String,
    pub email: String,
    pub role: String,
    pub mfa_verified: bool,
}

impl From<Claims> for AuthUser {
    fn from(claims: Claims) -> Self {
        Self {
            id: claims.sub,
            email: claims.email,
            role: claims.role,
            mfa_verified: claims.mfa_verified,
        }
    }
}

/// Auth layer for Axum
pub struct AuthLayer {
    jwt: Arc<JwtAuth>,
}

impl AuthLayer {
    pub fn new(jwt: Arc<JwtAuth>) -> Self {
        Self { jwt }
    }
}

/// Error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, error, message) = match &self {
            AuthError::InvalidCredentials => {
                (StatusCode::UNAUTHORIZED, "invalid_credentials", "Invalid credentials")
            }
            AuthError::TokenExpired => {
                (StatusCode::UNAUTHORIZED, "token_expired", "Token has expired")
            }
            AuthError::InvalidToken => {
                (StatusCode::UNAUTHORIZED, "invalid_token", "Invalid token")
            }
            AuthError::MfaRequired => {
                (StatusCode::FORBIDDEN, "mfa_required", "MFA verification required")
            }
            AuthError::InvalidMfaCode => {
                (StatusCode::UNAUTHORIZED, "invalid_mfa_code", "Invalid MFA code")
            }
            AuthError::UserNotFound => {
                (StatusCode::NOT_FOUND, "user_not_found", "User not found")
            }
            AuthError::Unauthorized => {
                (StatusCode::FORBIDDEN, "unauthorized", "Unauthorized access")
            }
            AuthError::Internal(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg.as_str())
            }
        };

        let body = Json(ErrorResponse {
            error: error.to_string(),
            message: message.to_string(),
        });

        (status, body).into_response()
    }
}

/// Extract user from request parts (for use in handlers)
#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Get user from extensions (set by middleware)
        parts
            .extensions
            .get::<AuthUser>()
            .cloned()
            .ok_or(AuthError::Unauthorized)
    }
}

/// Authentication middleware function
pub async fn auth_middleware(
    State(jwt): State<Arc<JwtAuth>>,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, AuthError> {
    // Get authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or(AuthError::InvalidToken)?;

    // Extract token
    let token = JwtAuth::extract_from_header(auth_header)
        .ok_or(AuthError::InvalidToken)?;

    // Validate token
    let claims = jwt.validate_access_token(token)?;

    // Create auth user
    let user = AuthUser::from(claims);

    // Insert user into request extensions
    request.extensions_mut().insert(user);

    // Continue to next handler
    Ok(next.run(request).await)
}

/// Optional authentication middleware (doesn't fail if no token)
pub async fn optional_auth_middleware(
    State(jwt): State<Arc<JwtAuth>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // Try to get authorization header
    if let Some(auth_header) = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
    {
        // Try to extract and validate token
        if let Some(token) = JwtAuth::extract_from_header(auth_header) {
            if let Ok(claims) = jwt.validate_access_token(token) {
                let user = AuthUser::from(claims);
                request.extensions_mut().insert(user);
            }
        }
    }

    // Continue regardless of auth status
    next.run(request).await
}

/// Require MFA verification middleware
pub async fn require_mfa_middleware(
    State(jwt): State<Arc<JwtAuth>>,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, AuthError> {
    // Get authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or(AuthError::InvalidToken)?;

    // Extract token
    let token = JwtAuth::extract_from_header(auth_header)
        .ok_or(AuthError::InvalidToken)?;

    // Validate token
    let claims = jwt.validate_access_token(token)?;

    // Check MFA verification
    if !claims.mfa_verified {
        return Err(AuthError::MfaRequired);
    }

    // Create auth user
    let user = AuthUser::from(claims);

    // Insert user into request extensions
    request.extensions_mut().insert(user);

    // Continue to next handler
    Ok(next.run(request).await)
}

/// Require admin role middleware
pub async fn require_admin_middleware(
    State(jwt): State<Arc<JwtAuth>>,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, AuthError> {
    // Get authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or(AuthError::InvalidToken)?;

    // Extract token
    let token = JwtAuth::extract_from_header(auth_header)
        .ok_or(AuthError::InvalidToken)?;

    // Validate token
    let claims = jwt.validate_access_token(token)?;

    // Check admin role
    if claims.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Create auth user
    let user = AuthUser::from(claims);

    // Insert user into request extensions
    request.extensions_mut().insert(user);

    // Continue to next handler
    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_user_from_claims() {
        let claims = Claims {
            sub: "user-123".to_string(),
            email: "test@example.com".to_string(),
            role: "admin".to_string(),
            exp: 0,
            iat: 0,
            token_type: "access".to_string(),
            mfa_verified: true,
        };

        let user = AuthUser::from(claims);

        assert_eq!(user.id, "user-123");
        assert_eq!(user.email, "test@example.com");
        assert_eq!(user.role, "admin");
        assert!(user.mfa_verified);
    }

    #[test]
    fn test_error_response() {
        let error = AuthError::InvalidCredentials;
        let response = error.into_response();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
