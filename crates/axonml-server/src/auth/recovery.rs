//! Recovery codes for AxonML
//!
//! Provides generation and verification of backup recovery codes.

use super::AuthError;
use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use rand::Rng;

/// Recovery code authentication handler
pub struct RecoveryAuth;

/// Recovery codes response
#[derive(Debug, Clone)]
pub struct RecoveryCodes {
    /// The plain text codes to show to user (once only!)
    pub codes: Vec<String>,
    /// The hashed codes to store in database
    pub hashed_codes: Vec<String>,
}

impl RecoveryAuth {
    /// Generate a new set of recovery codes
    pub fn generate_codes(count: usize) -> Result<RecoveryCodes, AuthError> {
        let mut codes = Vec::with_capacity(count);
        let mut hashed_codes = Vec::with_capacity(count);

        for _ in 0..count {
            let code = Self::generate_code();
            let hash = Self::hash_code(&code)?;
            codes.push(code);
            hashed_codes.push(hash);
        }

        Ok(RecoveryCodes { codes, hashed_codes })
    }

    /// Generate a single recovery code
    fn generate_code() -> String {
        let mut rng = rand::thread_rng();
        let parts: Vec<String> = (0..2)
            .map(|_| {
                let num: u32 = rng.gen_range(0..100000);
                format!("{:05}", num)
            })
            .collect();
        parts.join("-")
    }

    /// Hash a recovery code
    fn hash_code(code: &str) -> Result<String, AuthError> {
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();

        argon2
            .hash_password(code.replace("-", "").as_bytes(), &salt)
            .map(|hash| hash.to_string())
            .map_err(|e| AuthError::Internal(format!("Code hashing failed: {}", e)))
    }

    /// Verify a recovery code against stored hashes
    pub fn verify_code(code: &str, stored_hashes: &[String]) -> Result<Option<usize>, AuthError> {
        let normalized_code = code.replace("-", "").replace(" ", "");

        for (index, hash) in stored_hashes.iter().enumerate() {
            let parsed_hash = PasswordHash::new(hash)
                .map_err(|e| AuthError::Internal(format!("Invalid hash: {}", e)))?;

            if Argon2::default()
                .verify_password(normalized_code.as_bytes(), &parsed_hash)
                .is_ok()
            {
                return Ok(Some(index));
            }
        }

        Ok(None)
    }

    /// Format codes for display
    pub fn format_for_display(codes: &[String]) -> String {
        codes
            .iter()
            .enumerate()
            .map(|(i, code)| format!("{}. {}", i + 1, code))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_codes() {
        let recovery = RecoveryAuth::generate_codes(8).unwrap();

        assert_eq!(recovery.codes.len(), 8);
        assert_eq!(recovery.hashed_codes.len(), 8);

        // Codes should be in format XXXXX-XXXXX
        for code in &recovery.codes {
            assert_eq!(code.len(), 11);
            assert!(code.contains('-'));
        }
    }

    #[test]
    fn test_verify_code() {
        let recovery = RecoveryAuth::generate_codes(3).unwrap();

        // Verify first code
        let result = RecoveryAuth::verify_code(&recovery.codes[0], &recovery.hashed_codes).unwrap();
        assert_eq!(result, Some(0));

        // Verify second code
        let result = RecoveryAuth::verify_code(&recovery.codes[1], &recovery.hashed_codes).unwrap();
        assert_eq!(result, Some(1));

        // Invalid code should fail
        let result = RecoveryAuth::verify_code("00000-00000", &recovery.hashed_codes).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_verify_code_without_dash() {
        let recovery = RecoveryAuth::generate_codes(1).unwrap();
        let code_without_dash = recovery.codes[0].replace("-", "");

        let result = RecoveryAuth::verify_code(&code_without_dash, &recovery.hashed_codes).unwrap();
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_format_for_display() {
        let codes = vec![
            "12345-67890".to_string(),
            "11111-22222".to_string(),
        ];

        let formatted = RecoveryAuth::format_for_display(&codes);
        assert!(formatted.contains("1. 12345-67890"));
        assert!(formatted.contains("2. 11111-22222"));
    }
}
