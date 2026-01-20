//! TOTP (Time-based One-Time Password) authentication for AxonML
//!
//! Provides TOTP generation, verification, and QR code generation.

use super::AuthError;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::Luma;
use qrcode::QrCode;
use totp_rs::{Algorithm, Secret, TOTP};

/// TOTP authentication handler
pub struct TotpAuth {
    issuer: String,
}

/// TOTP setup response
#[derive(Debug, Clone)]
pub struct TotpSetup {
    pub secret: String,
    pub secret_base32: String,
    pub qr_code_data_url: String,
    pub otpauth_url: String,
}

impl TotpAuth {
    /// Create a new TOTP auth handler
    pub fn new(issuer: &str) -> Self {
        Self {
            issuer: issuer.to_string(),
        }
    }

    /// Generate a new TOTP secret
    pub fn generate_secret(&self) -> String {
        let secret = Secret::generate_secret();
        secret.to_encoded().to_string()
    }

    /// Create TOTP setup data including QR code
    pub fn setup(&self, user_email: &str) -> Result<TotpSetup, AuthError> {
        let secret = Secret::generate_secret();
        let secret_base32 = secret.to_encoded().to_string();

        let totp = self.create_totp(&secret_base32, user_email)?;
        let otpauth_url = totp.get_url();

        // Generate QR code
        let qr_code_data_url = self.generate_qr_code(&otpauth_url)?;

        Ok(TotpSetup {
            secret: secret_base32.clone(),
            secret_base32,
            qr_code_data_url,
            otpauth_url,
        })
    }

    /// Verify a TOTP code
    pub fn verify(&self, secret: &str, code: &str, user_email: &str) -> Result<bool, AuthError> {
        let totp = self.create_totp(secret, user_email)?;

        // Check with some tolerance for time drift
        Ok(totp.check_current(code).unwrap_or(false))
    }

    /// Generate a QR code as a data URL
    fn generate_qr_code(&self, data: &str) -> Result<String, AuthError> {
        let code = QrCode::new(data.as_bytes())
            .map_err(|e| AuthError::Internal(format!("QR code generation failed: {}", e)))?;

        let image = code.render::<Luma<u8>>().build();

        let mut png_bytes: Vec<u8> = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);

        encoder
            .encode(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::L8,
            )
            .map_err(|e| AuthError::Internal(format!("PNG encoding failed: {}", e)))?;

        let base64_image = BASE64.encode(&png_bytes);
        Ok(format!("data:image/png;base64,{}", base64_image))
    }

    /// Create a TOTP instance for a user
    fn create_totp(&self, secret: &str, user_email: &str) -> Result<TOTP, AuthError> {
        let secret = Secret::Encoded(secret.to_string());

        TOTP::new(
            Algorithm::SHA1,
            6,  // 6-digit codes
            1,  // 1 step tolerance
            30, // 30-second period
            secret.to_bytes().map_err(|e| AuthError::Internal(format!("Invalid secret: {}", e)))?,
            Some(self.issuer.clone()),
            user_email.to_string(),
        )
        .map_err(|e| AuthError::Internal(format!("TOTP creation failed: {}", e)))
    }

    /// Get the current TOTP code (for testing/debugging)
    #[allow(dead_code)]
    pub fn get_current_code(&self, secret: &str, user_email: &str) -> Result<String, AuthError> {
        let totp = self.create_totp(secret, user_email)?;
        totp.generate_current()
            .map_err(|e| AuthError::Internal(format!("Code generation failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_secret() {
        let totp = TotpAuth::new("AxonML");
        let secret = totp.generate_secret();

        assert!(!secret.is_empty());
        // Base32 encoded secrets are typically 32 chars
        assert!(secret.len() >= 16);
    }

    #[test]
    fn test_setup() {
        let totp = TotpAuth::new("AxonML");
        let setup = totp.setup("test@example.com").unwrap();

        assert!(!setup.secret.is_empty());
        assert!(!setup.secret_base32.is_empty());
        assert!(setup.qr_code_data_url.starts_with("data:image/png;base64,"));
        assert!(setup.otpauth_url.contains("otpauth://totp/"));
        assert!(setup.otpauth_url.contains("AxonML"));
    }

    #[test]
    fn test_verify() {
        let totp = TotpAuth::new("AxonML");
        let setup = totp.setup("test@example.com").unwrap();

        // Get the current valid code
        let current_code = totp.get_current_code(&setup.secret, "test@example.com").unwrap();

        // Verify it works
        let result = totp.verify(&setup.secret, &current_code, "test@example.com").unwrap();
        assert!(result);

        // Wrong code should fail
        let result = totp.verify(&setup.secret, "000000", "test@example.com").unwrap();
        assert!(!result);
    }
}
