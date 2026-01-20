//! JWT authentication for AxonML
//!
//! Provides JWT token creation and validation.

use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};
use serde::{Deserialize, Serialize};
use super::AuthError;

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// User email
    pub email: String,
    /// User role
    pub role: String,
    /// Expiration time (Unix timestamp)
    pub exp: i64,
    /// Issued at time (Unix timestamp)
    pub iat: i64,
    /// Token type (access or refresh)
    #[serde(default)]
    pub token_type: String,
    /// MFA verified flag
    #[serde(default)]
    pub mfa_verified: bool,
}

/// Token pair (access + refresh)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: i64,
    pub token_type: String,
}

/// MFA token (intermediate token before MFA verification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaToken {
    pub mfa_token: String,
    pub expires_in: i64,
}

/// JWT authentication handler
pub struct JwtAuth {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    access_expiry_hours: i64,
    refresh_expiry_days: i64,
}

impl JwtAuth {
    /// Create a new JWT auth handler
    pub fn new(secret: &str, access_expiry_hours: u64) -> Self {
        Self {
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            access_expiry_hours: access_expiry_hours as i64,
            refresh_expiry_days: 7, // Refresh tokens valid for 7 days
        }
    }

    /// Create an access token
    pub fn create_access_token(
        &self,
        user_id: &str,
        email: &str,
        role: &str,
        mfa_verified: bool,
    ) -> Result<String, AuthError> {
        let now = Utc::now();
        let expiry = now + Duration::hours(self.access_expiry_hours);

        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            role: role.to_string(),
            exp: expiry.timestamp(),
            iat: now.timestamp(),
            token_type: "access".to_string(),
            mfa_verified,
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| AuthError::Internal(format!("Token encoding failed: {}", e)))
    }

    /// Create a refresh token
    pub fn create_refresh_token(
        &self,
        user_id: &str,
        email: &str,
        role: &str,
    ) -> Result<String, AuthError> {
        let now = Utc::now();
        let expiry = now + Duration::days(self.refresh_expiry_days);

        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            role: role.to_string(),
            exp: expiry.timestamp(),
            iat: now.timestamp(),
            token_type: "refresh".to_string(),
            mfa_verified: true, // Refresh tokens are only issued after full auth
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| AuthError::Internal(format!("Token encoding failed: {}", e)))
    }

    /// Create a token pair
    pub fn create_token_pair(
        &self,
        user_id: &str,
        email: &str,
        role: &str,
        mfa_verified: bool,
    ) -> Result<TokenPair, AuthError> {
        let access_token = self.create_access_token(user_id, email, role, mfa_verified)?;
        let refresh_token = self.create_refresh_token(user_id, email, role)?;

        Ok(TokenPair {
            access_token,
            refresh_token,
            expires_in: self.access_expiry_hours * 3600,
            token_type: "Bearer".to_string(),
        })
    }

    /// Create an MFA token (short-lived, for MFA verification step)
    pub fn create_mfa_token(&self, user_id: &str, email: &str) -> Result<MfaToken, AuthError> {
        let now = Utc::now();
        let expiry = now + Duration::minutes(5); // MFA tokens valid for 5 minutes

        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            role: "pending_mfa".to_string(),
            exp: expiry.timestamp(),
            iat: now.timestamp(),
            token_type: "mfa".to_string(),
            mfa_verified: false,
        };

        let token = encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| AuthError::Internal(format!("Token encoding failed: {}", e)))?;

        Ok(MfaToken {
            mfa_token: token,
            expires_in: 300, // 5 minutes
        })
    }

    /// Validate a token and return claims
    pub fn validate_token(&self, token: &str) -> Result<Claims, AuthError> {
        let token_data: TokenData<Claims> = decode(token, &self.decoding_key, &Validation::default())
            .map_err(|e| match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                _ => AuthError::InvalidToken,
            })?;

        Ok(token_data.claims)
    }

    /// Validate an access token
    pub fn validate_access_token(&self, token: &str) -> Result<Claims, AuthError> {
        let claims = self.validate_token(token)?;

        if claims.token_type != "access" {
            return Err(AuthError::InvalidToken);
        }

        Ok(claims)
    }

    /// Validate a refresh token
    pub fn validate_refresh_token(&self, token: &str) -> Result<Claims, AuthError> {
        let claims = self.validate_token(token)?;

        if claims.token_type != "refresh" {
            return Err(AuthError::InvalidToken);
        }

        Ok(claims)
    }

    /// Validate an MFA token
    pub fn validate_mfa_token(&self, token: &str) -> Result<Claims, AuthError> {
        let claims = self.validate_token(token)?;

        if claims.token_type != "mfa" {
            return Err(AuthError::InvalidToken);
        }

        Ok(claims)
    }

    /// Refresh an access token using a refresh token
    pub fn refresh_access_token(&self, refresh_token: &str) -> Result<TokenPair, AuthError> {
        let claims = self.validate_refresh_token(refresh_token)?;

        self.create_token_pair(&claims.sub, &claims.email, &claims.role, true)
    }

    /// Extract token from Authorization header
    pub fn extract_from_header(header: &str) -> Option<&str> {
        if header.starts_with("Bearer ") {
            Some(&header[7..])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_validate_access_token() {
        let jwt = JwtAuth::new("test_secret_key_32_bytes_long!", 24);

        let token = jwt.create_access_token("user-123", "test@example.com", "user", true).unwrap();
        let claims = jwt.validate_access_token(&token).unwrap();

        assert_eq!(claims.sub, "user-123");
        assert_eq!(claims.email, "test@example.com");
        assert_eq!(claims.role, "user");
        assert!(claims.mfa_verified);
    }

    #[test]
    fn test_token_pair() {
        let jwt = JwtAuth::new("test_secret_key_32_bytes_long!", 24);

        let pair = jwt.create_token_pair("user-123", "test@example.com", "admin", true).unwrap();

        assert!(!pair.access_token.is_empty());
        assert!(!pair.refresh_token.is_empty());
        assert_eq!(pair.token_type, "Bearer");
    }

    #[test]
    fn test_mfa_token() {
        let jwt = JwtAuth::new("test_secret_key_32_bytes_long!", 24);

        let mfa_token = jwt.create_mfa_token("user-123", "test@example.com").unwrap();
        let claims = jwt.validate_mfa_token(&mfa_token.mfa_token).unwrap();

        assert_eq!(claims.sub, "user-123");
        assert_eq!(claims.token_type, "mfa");
        assert!(!claims.mfa_verified);
    }

    #[test]
    fn test_invalid_token() {
        let jwt = JwtAuth::new("test_secret_key_32_bytes_long!", 24);

        let result = jwt.validate_token("invalid.token.here");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_from_header() {
        assert_eq!(
            JwtAuth::extract_from_header("Bearer abc123"),
            Some("abc123")
        );
        assert_eq!(JwtAuth::extract_from_header("abc123"), None);
        assert_eq!(JwtAuth::extract_from_header("bearer abc123"), None);
    }
}
