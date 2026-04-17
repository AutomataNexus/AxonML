//! JavaScript Interop Helpers — Non-Panicking Reflect Property Setters
//!
//! Safe wrappers around `js_sys::Reflect::set` for assigning properties on
//! JavaScript objects from Rust/WASM code. The generic `js_set` helper returns
//! a boolean indicating success or failure and logs a warning to the browser
//! console via `web_sys::console::warn_1` rather than panicking the WASM
//! module. The `js_set_str`, `js_set_bool`, and `js_set_f64` convenience
//! wrappers forward to `js_set` after converting the Rust value into a
//! `JsValue` via `Into` or `JsValue::from_f64`.
//!
//! # File
//! `crates/axonml-dashboard/src/utils/js_helpers.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use wasm_bindgen::JsValue;

// =============================================================================
// Generic Property Setter
// =============================================================================

/// Sets a property on a JS object without panicking.
///
/// Returns `true` on success, `false` on failure (logs warning to console).
pub fn js_set(obj: &JsValue, key: &str, value: &JsValue) -> bool {
    match js_sys::Reflect::set(obj, &key.into(), value) {
        Ok(_) => true,
        Err(e) => {
            web_sys::console::warn_1(
                &format!("JS property set failed for '{}': {:?}", key, e).into(),
            );
            false
        }
    }
}

// =============================================================================
// Typed Convenience Wrappers
// =============================================================================

/// Convenience: set a string property.
pub fn js_set_str(obj: &JsValue, key: &str, value: &str) -> bool {
    js_set(obj, key, &value.into())
}

/// Convenience: set a bool property.
pub fn js_set_bool(obj: &JsValue, key: &str, value: bool) -> bool {
    js_set(obj, key, &value.into())
}

/// Convenience: set an f64 property.
pub fn js_set_f64(obj: &JsValue, key: &str, value: f64) -> bool {
    js_set(obj, key, &JsValue::from_f64(value))
}
