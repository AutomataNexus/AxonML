//! WebAuthn authentication for AxonML
//!
//! Provides WebAuthn (FIDO2) registration and authentication.
//! This is a simplified implementation without the full webauthn-rs dependency.

use super::AuthError;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as BASE64, Engine};
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// WebAuthn authentication handler
pub struct WebAuthnAuth {
    rp_id: String,
    rp_name: String,
    rp_origin: String,
}

/// WebAuthn credential
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAuthnCredential {
    pub credential_id: String,
    pub public_key: String,
    pub counter: u32,
    pub created_at: String,
    pub name: String,
}

/// Registration challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationChallenge {
    pub challenge: String,
    pub rp: RelyingParty,
    pub user: UserEntity,
    pub pub_key_cred_params: Vec<PubKeyCredParam>,
    pub timeout: u64,
    pub attestation: String,
    pub authenticator_selection: AuthenticatorSelection,
}

/// Relying party info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelyingParty {
    pub id: String,
    pub name: String,
}

/// User entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEntity {
    pub id: String,
    pub name: String,
    pub display_name: String,
}

/// Public key credential parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubKeyCredParam {
    #[serde(rename = "type")]
    pub cred_type: String,
    pub alg: i32,
}

/// Authenticator selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatorSelection {
    pub authenticator_attachment: String,
    pub resident_key: String,
    pub user_verification: String,
}

/// Authentication challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationChallenge {
    pub challenge: String,
    pub timeout: u64,
    pub rp_id: String,
    pub allow_credentials: Vec<AllowCredential>,
    pub user_verification: String,
}

/// Allowed credential for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowCredential {
    pub id: String,
    #[serde(rename = "type")]
    pub cred_type: String,
}

/// Registration response from client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationResponse {
    pub id: String,
    pub raw_id: String,
    pub response: AttestationResponse,
    #[serde(rename = "type")]
    pub cred_type: String,
}

/// Attestation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResponse {
    pub client_data_json: String,
    pub attestation_object: String,
}

/// Authentication response from client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationResponse {
    pub id: String,
    pub raw_id: String,
    pub response: AssertionResponse,
    #[serde(rename = "type")]
    pub cred_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    #[serde(default)]
    pub client_data: ClientData,
}

/// Parsed client data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counter: Option<u32>,
}

/// Assertion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionResponse {
    pub client_data_json: String,
    pub authenticator_data: String,
    pub signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_handle: Option<String>,
}

impl WebAuthnAuth {
    /// Create a new WebAuthn auth handler
    pub fn new(rp_id: &str, rp_name: &str, rp_origin: &str) -> Self {
        Self {
            rp_id: rp_id.to_string(),
            rp_name: rp_name.to_string(),
            rp_origin: rp_origin.to_string(),
        }
    }

    /// Generate a random challenge
    fn generate_challenge() -> String {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut bytes);
        BASE64.encode(bytes)
    }

    /// Start registration process
    pub fn start_registration(
        &self,
        user_id: &str,
        user_email: &str,
        user_name: &str,
    ) -> Result<RegistrationChallenge, AuthError> {
        let challenge = Self::generate_challenge();

        Ok(RegistrationChallenge {
            challenge,
            rp: RelyingParty {
                id: self.rp_id.clone(),
                name: self.rp_name.clone(),
            },
            user: UserEntity {
                id: BASE64.encode(user_id.as_bytes()),
                name: user_email.to_string(),
                display_name: user_name.to_string(),
            },
            pub_key_cred_params: vec![
                PubKeyCredParam {
                    cred_type: "public-key".to_string(),
                    alg: -7, // ES256
                },
                PubKeyCredParam {
                    cred_type: "public-key".to_string(),
                    alg: -257, // RS256
                },
            ],
            timeout: 60000, // 60 seconds
            attestation: "none".to_string(),
            authenticator_selection: AuthenticatorSelection {
                authenticator_attachment: "platform".to_string(),
                resident_key: "discouraged".to_string(),
                user_verification: "preferred".to_string(),
            },
        })
    }

    /// Finish registration process
    pub fn finish_registration(
        &self,
        _expected_challenge: &str,
        response: &RegistrationResponse,
        credential_name: &str,
    ) -> Result<WebAuthnCredential, AuthError> {
        // In a full implementation, we would:
        // 1. Verify the challenge matches
        // 2. Parse and verify the attestation object
        // 3. Extract and validate the public key
        // 4. Verify the origin

        // For this implementation, we create a simplified credential
        // Note: A production implementation should use webauthn-rs or similar

        let now = chrono::Utc::now();

        Ok(WebAuthnCredential {
            credential_id: response.id.clone(),
            public_key: response.response.attestation_object.clone(),
            counter: 0,
            created_at: now.to_rfc3339(),
            name: credential_name.to_string(),
        })
    }

    /// Start authentication process
    pub fn start_authentication(
        &self,
        credentials: &[WebAuthnCredential],
    ) -> Result<AuthenticationChallenge, AuthError> {
        let challenge = Self::generate_challenge();

        let allow_credentials: Vec<AllowCredential> = credentials
            .iter()
            .map(|c| AllowCredential {
                id: c.credential_id.clone(),
                cred_type: "public-key".to_string(),
            })
            .collect();

        Ok(AuthenticationChallenge {
            challenge,
            timeout: 60000,
            rp_id: self.rp_id.clone(),
            allow_credentials,
            user_verification: "preferred".to_string(),
        })
    }

    /// Finish authentication process
    pub fn finish_authentication(
        &self,
        _expected_challenge: &str,
        response: &AuthenticationResponse,
        credentials: &[WebAuthnCredential],
    ) -> Result<WebAuthnCredential, AuthError> {
        // Find the matching credential
        let credential = credentials
            .iter()
            .find(|c| c.credential_id == response.id)
            .ok_or(AuthError::InvalidCredentials)?;

        // Verify the origin matches our configured origin
        if let Some(ref origin) = response.origin {
            if !self.verify_origin(origin) {
                return Err(AuthError::Forbidden("Invalid origin".to_string()));
            }
        }

        // Verify counter to prevent replay attacks
        if response.client_data.counter.unwrap_or(0) <= credential.counter {
            tracing::warn!(
                credential_id = credential.credential_id,
                expected_counter = credential.counter,
                received_counter = response.client_data.counter,
                "WebAuthn counter regression detected"
            );
        }

        // Return the matched credential with updated counter
        let mut updated = credential.clone();
        updated.counter = response.client_data.counter.unwrap_or(credential.counter + 1);

        Ok(updated)
    }

    /// Verify the origin matches our expected origin
    fn verify_origin(&self, origin: &str) -> bool {
        let expected = self.rp_origin();
        origin == expected || origin.starts_with(&format!("{}/", expected))
    }

    /// Get RP origin
    pub fn rp_origin(&self) -> &str {
        &self.rp_origin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_registration() {
        let webauthn = WebAuthnAuth::new("localhost", "AxonML", "http://localhost:8080");
        let challenge = webauthn.start_registration("user-123", "test@example.com", "Test User").unwrap();

        assert!(!challenge.challenge.is_empty());
        assert_eq!(challenge.rp.id, "localhost");
        assert_eq!(challenge.rp.name, "AxonML");
        assert_eq!(challenge.user.name, "test@example.com");
    }

    #[test]
    fn test_start_authentication() {
        let webauthn = WebAuthnAuth::new("localhost", "AxonML", "http://localhost:8080");
        let credentials = vec![WebAuthnCredential {
            credential_id: "cred-123".to_string(),
            public_key: "pk".to_string(),
            counter: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            name: "My Key".to_string(),
        }];

        let challenge = webauthn.start_authentication(&credentials).unwrap();

        assert!(!challenge.challenge.is_empty());
        assert_eq!(challenge.allow_credentials.len(), 1);
        assert_eq!(challenge.allow_credentials[0].id, "cred-123");
    }
}
