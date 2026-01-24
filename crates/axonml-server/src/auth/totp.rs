//! TOTP (Time-based One-Time Password) authentication for AxonML
//!
//! Provides TOTP generation, verification, and QR code generation.

use super::AuthError;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use qrcode::QrCode;
use qrcode::render::svg;
use totp_rs::{Algorithm, Secret, TOTP};

/// TOTP authentication handler
pub struct TotpAuth {
    issuer: String,
}

/// TOTP setup response
#[derive(Debug, Clone)]
pub struct TotpSetup {
    /// The secret in base32 encoding (used for verification)
    pub secret: String,
    /// QR code as a data URL (SVG format)
    pub qr_code_data_url: String,
    /// Full OTPAuth URL for manual entry
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
        let secret = self.generate_secret();

        let totp = self.create_totp(&secret, user_email)?;
        let otpauth_url = totp.get_url();

        // Generate QR code
        let qr_code_data_url = self.generate_qr_code(&otpauth_url)?;

        Ok(TotpSetup {
            secret,
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

    /// Generate a QR code as a data URL (SVG format)
    fn generate_qr_code(&self, data: &str) -> Result<String, AuthError> {
        let code = QrCode::new(data.as_bytes())
            .map_err(|e| AuthError::Internal(format!("QR code generation failed: {}", e)))?;

        let svg_string = code
            .render()
            .min_dimensions(200, 200)
            .dark_color(svg::Color("#000000"))
            .light_color(svg::Color("#ffffff"))
            .build();

        let base64_svg = BASE64.encode(svg_string.as_bytes());
        Ok(format!("data:image/svg+xml;base64,{}", base64_svg))
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
        // Base32 encoded secrets are typically 32 chars
        assert!(setup.secret.len() >= 16);
        assert!(setup.qr_code_data_url.starts_with("data:image/svg+xml;base64,"));
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
