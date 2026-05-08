//! hailo_ffi — Raw FFI bindings to hailo_genai_shim.cpp.
//!
//! These are the unsafe `extern "C"` declarations matching the C++ shim.
//! Safe Rust wrappers live in `hailo10h.rs` which consumes this module.
//!
//! Gated behind `--features hailo10h`. On non-Pi builds the module doesn't
//! compile and the linker never looks for `libhailo_genai_shim.a` / `libhailort.so`.

#![cfg(feature = "hailo10h")]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void};
use std::os::raw::c_float;

// Opaque handle types from the C++ shim
pub type HailoLlmHandle = c_void;
pub type HailoGeneratorHandle = c_void;
pub type HailoCompletionHandle = c_void;

unsafe extern "C" {
    // Error reporting
    pub fn hailo_llm_last_error() -> *const c_char;

    // LLM lifecycle
    pub fn hailo_llm_create(hef_path: *const c_char) -> *mut HailoLlmHandle;
    pub fn hailo_llm_destroy(h: *mut HailoLlmHandle);

    // Context management
    pub fn hailo_llm_max_context(h: *mut HailoLlmHandle) -> i64;
    pub fn hailo_llm_context_usage(h: *mut HailoLlmHandle) -> i64;
    pub fn hailo_llm_clear_context(h: *mut HailoLlmHandle) -> c_int;
    pub fn hailo_llm_prompt_template(h: *mut HailoLlmHandle) -> *mut c_char;
    pub fn hailo_llm_free_string(s: *mut c_char);
    pub fn hailo_llm_set_stop_tokens(
        h: *mut HailoLlmHandle,
        tokens: *const *const c_char,
        count: c_int,
    ) -> c_int;

    // Context save/load
    pub fn hailo_llm_save_context(
        h: *mut HailoLlmHandle,
        buf: *mut u8,
        buf_size: i64,
    ) -> i64;
    pub fn hailo_llm_load_context(
        h: *mut HailoLlmHandle,
        buf: *const u8,
        size: i64,
    ) -> c_int;

    // Tokenizer
    pub fn hailo_llm_tokenize(
        h: *mut HailoLlmHandle,
        text: *const c_char,
        token_ids: *mut c_int,
        max_tokens: c_int,
    ) -> c_int;

    // Generator
    pub fn hailo_llm_create_generator(
        h: *mut HailoLlmHandle,
        temperature: c_float,
        top_p: c_float,
        top_k: u32,
        frequency_penalty: c_float,
        max_generated_tokens: u32,
        do_sample: c_int,
        seed: u32,
    ) -> *mut HailoGeneratorHandle;
    pub fn hailo_generator_destroy(g: *mut HailoGeneratorHandle);
    pub fn hailo_generator_write_raw(
        g: *mut HailoGeneratorHandle,
        text: *const c_char,
    ) -> c_int;
    pub fn hailo_generator_write_chat(
        g: *mut HailoGeneratorHandle,
        messages: *const *const c_char,
        message_count: c_int,
        tools: *const *const c_char,
        tool_count: c_int,
    ) -> c_int;

    // Completion (token streaming)
    pub fn hailo_generator_generate(
        g: *mut HailoGeneratorHandle,
    ) -> *mut HailoCompletionHandle;
    pub fn hailo_completion_read(
        c: *mut HailoCompletionHandle,
        buf: *mut c_char,
        buf_size: c_int,
        timeout_ms: c_int,
    ) -> c_int;
    pub fn hailo_completion_status(c: *mut HailoCompletionHandle) -> c_int;
    pub fn hailo_completion_abort(c: *mut HailoCompletionHandle) -> c_int;
    pub fn hailo_completion_destroy(c: *mut HailoCompletionHandle);
}

/// Generation status codes matching the C++ enum
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationStatus {
    Generating = 0,
    MaxTokensReached = 1,
    LogicalEnd = 2,
    Aborted = 3,
}

impl From<c_int> for GenerationStatus {
    fn from(v: c_int) -> Self {
        match v {
            0 => Self::Generating,
            1 => Self::MaxTokensReached,
            2 => Self::LogicalEnd,
            3 => Self::Aborted,
            _ => Self::Aborted,
        }
    }
}
