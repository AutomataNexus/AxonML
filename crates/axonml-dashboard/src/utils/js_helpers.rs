//! JavaScript Interop Helpers
//!
//! Safe wrappers around `js_sys::Reflect::set` and other JS operations
//! that avoid panicking on failure in WASM.

use wasm_bindgen::JsValue;

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
