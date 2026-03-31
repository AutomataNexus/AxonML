//! Authentication module for AxonML Server
//!
//! # File
//! `crates/axonml-server/src/auth/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

pub mod jwt;
pub mod middleware;
pub mod rate_limit;
pub mod recovery;
pub mod totp;
pub mod webauthn;

pub use jwt::{Claims, JwtAuth};
pub use middleware::{
    AuthLayer, AuthUser, auth_middleware, optional_auth_middleware, require_admin_middleware,
    require_mfa_middleware,
};
pub use rate_limit::RateLimiter;
pub use recovery::RecoveryAuth;
pub use totp::TotpAuth;
pub use webauthn::WebAuthnAuth;

use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Token expired")]
    TokenExpired,
    #[error("Invalid token")]
    InvalidToken,
    #[error("MFA required")]
    MfaRequired,
    #[error("Invalid MFA code")]
    InvalidMfaCode,
    #[error("User not found")]
    UserNotFound,
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Forbidden: {0}")]
    Forbidden(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Hash a password using Argon2
pub fn hash_password(password: &str) -> Result<String, AuthError> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();

    argon2
        .hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|e| AuthError::Internal(format!("Password hashing failed: {}", e)))
}

/// Verify a password against a hash
pub fn verify_password(password: &str, hash: &str) -> Result<bool, AuthError> {
    let parsed_hash = PasswordHash::new(hash)
        .map_err(|e| AuthError::Internal(format!("Invalid password hash: {}", e)))?;

    Ok(Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_password() -> String {
        format!("test_pw_{}", std::process::id())
    }

    #[test]
    fn test_password_hashing() {
        let password = test_password();
        let hash = hash_password(&password).unwrap();

        assert!(hash.starts_with("$argon2"));
        assert!(verify_password(&password, &hash).unwrap());
        assert!(!verify_password(&format!("{}_wrong", password), &hash).unwrap());
    }

    #[test]
    fn test_different_passwords_different_hashes() {
        let password = test_password();
        let hash1 = hash_password(&password).unwrap();
        let hash2 = hash_password(&password).unwrap();

        // Same password should produce different hashes (different salts)
        assert_ne!(hash1, hash2);
    }
}
