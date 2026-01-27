//! WebAuthn utility functions for WASM
//!
//! Implements WebAuthn registration and authentication using web-sys bindings.

use js_sys::{Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;

/// WebAuthn error types
#[derive(Debug, Clone)]
pub enum WebAuthnError {
    NotSupported,
    UserCancelled,
    SecurityError(String),
    NetworkError(String),
    InvalidResponse(String),
}

impl std::fmt::Display for WebAuthnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WebAuthnError::NotSupported => write!(f, "WebAuthn is not supported in this browser"),
            WebAuthnError::UserCancelled => write!(f, "User cancelled the WebAuthn operation"),
            WebAuthnError::SecurityError(e) => write!(f, "Security error: {}", e),
            WebAuthnError::NetworkError(e) => write!(f, "Network error: {}", e),
            WebAuthnError::InvalidResponse(e) => write!(f, "Invalid response: {}", e),
        }
    }
}

/// Check if WebAuthn is available in the current browser
pub fn is_webauthn_available() -> bool {
    if let Some(window) = web_sys::window() {
        if let Ok(navigator) = Reflect::get(&window, &JsValue::from_str("navigator")) {
            return Reflect::has(&navigator, &JsValue::from_str("credentials")).unwrap_or(false);
        }
    }
    false
}

/// WebAuthn credential response from registration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebAuthnRegistrationResponse {
    pub id: String,
    pub raw_id: String,
    pub attestation_object: String,
    pub client_data_json: String,
}

/// WebAuthn credential response from authentication
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebAuthnAuthenticationResponse {
    pub id: String,
    pub raw_id: String,
    pub authenticator_data: String,
    pub client_data_json: String,
    pub signature: String,
    pub user_handle: Option<String>,
}

/// Start WebAuthn registration (credential creation)
///
/// # Arguments
/// * `challenge` - Server-provided challenge (base64url encoded)
/// * `rp_name` - Relying party name (your app name)
/// * `rp_id` - Relying party ID (domain)
/// * `user_id` - User ID (base64url encoded)
/// * `user_name` - User's display name
/// * `user_display_name` - User's display name for the credential
pub async fn create_credential(
    challenge: &str,
    rp_name: &str,
    rp_id: &str,
    user_id: &str,
    user_name: &str,
    user_display_name: &str,
) -> Result<WebAuthnRegistrationResponse, WebAuthnError> {
    if !is_webauthn_available() {
        return Err(WebAuthnError::NotSupported);
    }

    let window = web_sys::window().ok_or(WebAuthnError::NotSupported)?;
    let navigator = window.navigator();
    let credentials = navigator.credentials();

    // Decode challenge from base64url
    let challenge_bytes = base64_url_decode(challenge)
        .map_err(|e| WebAuthnError::InvalidResponse(format!("Invalid challenge: {}", e)))?;
    let user_id_bytes = base64_url_decode(user_id)
        .map_err(|e| WebAuthnError::InvalidResponse(format!("Invalid user ID: {}", e)))?;

    // Build the publicKey options object
    let public_key = Object::new();

    // Challenge
    let challenge_array = Uint8Array::from(challenge_bytes.as_slice());
    Reflect::set(&public_key, &"challenge".into(), &challenge_array)
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set challenge".into()))?;

    // Relying Party
    let rp = Object::new();
    Reflect::set(&rp, &"name".into(), &JsValue::from_str(rp_name))
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set rp name".into()))?;
    Reflect::set(&rp, &"id".into(), &JsValue::from_str(rp_id))
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set rp id".into()))?;
    Reflect::set(&public_key, &"rp".into(), &rp)
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set rp".into()))?;

    // User
    let user = Object::new();
    let user_id_array = Uint8Array::from(user_id_bytes.as_slice());
    Reflect::set(&user, &"id".into(), &user_id_array)
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set user id".into()))?;
    Reflect::set(&user, &"name".into(), &JsValue::from_str(user_name))
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set user name".into()))?;
    Reflect::set(
        &user,
        &"displayName".into(),
        &JsValue::from_str(user_display_name),
    )
    .map_err(|_| WebAuthnError::InvalidResponse("Failed to set display name".into()))?;
    Reflect::set(&public_key, &"user".into(), &user)
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set user".into()))?;

    // Supported algorithms (ES256, RS256)
    let pub_key_cred_params = Array::new();

    let es256 = Object::new();
    Reflect::set(&es256, &"type".into(), &JsValue::from_str("public-key")).unwrap();
    Reflect::set(&es256, &"alg".into(), &JsValue::from_f64(-7.0)).unwrap(); // ES256
    pub_key_cred_params.push(&es256);

    let rs256 = Object::new();
    Reflect::set(&rs256, &"type".into(), &JsValue::from_str("public-key")).unwrap();
    Reflect::set(&rs256, &"alg".into(), &JsValue::from_f64(-257.0)).unwrap(); // RS256
    pub_key_cred_params.push(&rs256);

    Reflect::set(
        &public_key,
        &"pubKeyCredParams".into(),
        &pub_key_cred_params,
    )
    .map_err(|_| WebAuthnError::InvalidResponse("Failed to set algorithms".into()))?;

    // Timeout (60 seconds)
    Reflect::set(&public_key, &"timeout".into(), &JsValue::from_f64(60000.0))
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set timeout".into()))?;

    // Create options
    let options = Object::new();
    Reflect::set(&options, &"publicKey".into(), &public_key)
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set publicKey".into()))?;

    // Call navigator.credentials.create()
    let credential_promise = credentials
        .create_with_options(&options.unchecked_into())
        .map_err(|e| WebAuthnError::SecurityError(format!("{:?}", e)))?;

    let credential = JsFuture::from(credential_promise).await.map_err(|e| {
        let error_str = format!("{:?}", e);
        if error_str.contains("NotAllowedError") {
            WebAuthnError::UserCancelled
        } else {
            WebAuthnError::SecurityError(error_str)
        }
    })?;

    // Extract response data
    let credential: PublicKeyCredential = credential.unchecked_into();
    let response: AuthenticatorAttestationResponse = credential.response().unchecked_into();

    let id = credential.id();
    let raw_id = base64_url_encode(&js_array_buffer_to_vec(&credential.raw_id()));
    let attestation_object =
        base64_url_encode(&js_array_buffer_to_vec(&response.attestation_object()));
    let client_data_json = base64_url_encode(&js_array_buffer_to_vec(&response.client_data_json()));

    Ok(WebAuthnRegistrationResponse {
        id,
        raw_id,
        attestation_object,
        client_data_json,
    })
}

/// Authenticate with WebAuthn (credential assertion)
///
/// # Arguments
/// * `challenge` - Server-provided challenge (base64url encoded)
/// * `rp_id` - Relying party ID (domain)
/// * `allow_credentials` - List of allowed credential IDs (base64url encoded)
pub async fn get_assertion(
    challenge: &str,
    rp_id: &str,
    allow_credentials: &[String],
) -> Result<WebAuthnAuthenticationResponse, WebAuthnError> {
    if !is_webauthn_available() {
        return Err(WebAuthnError::NotSupported);
    }

    let window = web_sys::window().ok_or(WebAuthnError::NotSupported)?;
    let navigator = window.navigator();
    let credentials = navigator.credentials();

    // Decode challenge
    let challenge_bytes = base64_url_decode(challenge)
        .map_err(|e| WebAuthnError::InvalidResponse(format!("Invalid challenge: {}", e)))?;

    // Build publicKey options
    let public_key = Object::new();

    // Challenge
    let challenge_array = Uint8Array::from(challenge_bytes.as_slice());
    Reflect::set(&public_key, &"challenge".into(), &challenge_array)
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set challenge".into()))?;

    // Relying Party ID
    Reflect::set(&public_key, &"rpId".into(), &JsValue::from_str(rp_id))
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set rpId".into()))?;

    // Timeout
    Reflect::set(&public_key, &"timeout".into(), &JsValue::from_f64(60000.0))
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set timeout".into()))?;

    // Allow credentials
    if !allow_credentials.is_empty() {
        let allowed = Array::new();
        for cred_id in allow_credentials {
            let cred_id_bytes = base64_url_decode(cred_id).map_err(|e| {
                WebAuthnError::InvalidResponse(format!("Invalid credential ID: {}", e))
            })?;

            let cred = Object::new();
            Reflect::set(&cred, &"type".into(), &JsValue::from_str("public-key")).unwrap();
            let id_array = Uint8Array::from(cred_id_bytes.as_slice());
            Reflect::set(&cred, &"id".into(), &id_array).unwrap();
            allowed.push(&cred);
        }
        Reflect::set(&public_key, &"allowCredentials".into(), &allowed)
            .map_err(|_| WebAuthnError::InvalidResponse("Failed to set allowCredentials".into()))?;
    }

    // Create options
    let options = Object::new();
    Reflect::set(&options, &"publicKey".into(), &public_key)
        .map_err(|_| WebAuthnError::InvalidResponse("Failed to set publicKey".into()))?;

    // Call navigator.credentials.get()
    let credential_promise = credentials
        .get_with_options(&options.unchecked_into())
        .map_err(|e| WebAuthnError::SecurityError(format!("{:?}", e)))?;

    let credential = JsFuture::from(credential_promise).await.map_err(|e| {
        let error_str = format!("{:?}", e);
        if error_str.contains("NotAllowedError") {
            WebAuthnError::UserCancelled
        } else {
            WebAuthnError::SecurityError(error_str)
        }
    })?;

    // Extract response data
    let credential: PublicKeyCredential = credential.unchecked_into();
    let response: AuthenticatorAssertionResponse = credential.response().unchecked_into();

    let id = credential.id();
    let raw_id = base64_url_encode(&js_array_buffer_to_vec(&credential.raw_id()));
    let authenticator_data =
        base64_url_encode(&js_array_buffer_to_vec(&response.authenticator_data()));
    let client_data_json = base64_url_encode(&js_array_buffer_to_vec(&response.client_data_json()));
    let signature = base64_url_encode(&js_array_buffer_to_vec(&response.signature()));

    let user_handle = response
        .user_handle()
        .map(|uh| base64_url_encode(&js_array_buffer_to_vec(&uh)));

    Ok(WebAuthnAuthenticationResponse {
        id,
        raw_id,
        authenticator_data,
        client_data_json,
        signature,
        user_handle,
    })
}

// Helper functions

fn js_array_buffer_to_vec(buffer: &js_sys::ArrayBuffer) -> Vec<u8> {
    let array = Uint8Array::new(buffer);
    array.to_vec()
}

fn base64_url_decode(input: &str) -> Result<Vec<u8>, String> {
    // Convert base64url to standard base64
    let standard = input.replace('-', "+").replace('_', "/");

    // Add padding if necessary
    let padded = match standard.len() % 4 {
        2 => format!("{}==", standard),
        3 => format!("{}=", standard),
        _ => standard,
    };

    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &padded)
        .map_err(|e| format!("Base64 decode error: {}", e))
}

fn base64_url_encode(input: &[u8]) -> String {
    let standard = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, input);
    // Convert to base64url (no padding)
    standard
        .replace('+', "-")
        .replace('/', "_")
        .trim_end_matches('=')
        .to_string()
}
