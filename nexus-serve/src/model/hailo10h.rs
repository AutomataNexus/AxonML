//! hailo10h — Hailo-10H NPU LLM backend for nexus-serve.
//!
//! Replaces hailo-ollama (Python) with a pure-Rust API layer backed by
//! HailoRT's GenAI C++ API via a thin C++ shim (`hailo_genai_shim.cpp`).
//!
//! The entire LLM inference pipeline — tokenization, KV-cache management,
//! prefill/TBT network-group switching, and sampling — runs on the Hailo-10H
//! NPU via `libhailort.so`. nexus-serve handles HTTP API + token streaming.
//!
//! # Build
//!
//! On the Pi (natively):
//! ```bash
//! # Compile the C++ shim
//! g++ -c -std=c++17 -fPIC -I/usr/include src/hailo_genai_shim.cpp -o hailo_genai_shim.o
//! ar rcs libhailo_genai_shim.a hailo_genai_shim.o
//!
//! # Build nexus-serve with Hailo backend
//! RUSTFLAGS="-L /path/to/libhailo_genai_shim.a/dir" \
//!   cargo build --release --features hailo10h
//! ```
//!
//! # Usage
//!
//! ```bash
//! nexus-serve --hailo /path/to/qwen3-1.7b.hef --port 11435
//! ```
//!
//! Then hit the same OpenAI-compatible API as the CPU/CUDA backend:
//! ```bash
//! curl http://pi:11435/v1/chat/completions \
//!   -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Hello"}]}'
//! ```

#![cfg(feature = "hailo10h")]

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow};

use super::hailo_ffi::{self, GenerationStatus};

/// Safe wrapper around the Hailo GenAI LLM C++ handle.
pub struct Hailo10hEngine {
    handle: Mutex<*mut hailo_ffi::HailoLlmHandle>,
    hef_path: String,
    max_context: usize,
}

// The C++ handle is thread-safe internally (HailoRT manages device locking).
// We add a Mutex on the Rust side to serialize generator creation (only one
// generator can be active at a time per the Hailo API contract).
unsafe impl Send for Hailo10hEngine {}
unsafe impl Sync for Hailo10hEngine {}

impl Hailo10hEngine {
    /// Load an LLM from a HEF file. Claims the Hailo-10H VDevice.
    pub fn load(hef_path: impl AsRef<Path>) -> Result<Self> {
        let path_str = hef_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow!("HEF path is not valid UTF-8"))?;
        let c_path = CString::new(path_str)?;

        let handle = unsafe { hailo_ffi::hailo_llm_create(c_path.as_ptr()) };
        if handle.is_null() {
            let err = unsafe {
                CStr::from_ptr(hailo_ffi::hailo_llm_last_error())
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(anyhow!("hailo_llm_create failed: {err}"));
        }

        let max_ctx = unsafe { hailo_ffi::hailo_llm_max_context(handle) };

        tracing::info!(
            hef = path_str,
            max_context = max_ctx,
            "Hailo-10H LLM loaded",
        );

        Ok(Self {
            handle: Mutex::new(handle),
            hef_path: path_str.to_string(),
            max_context: max_ctx.max(0) as usize,
        })
    }

    /// Maximum context capacity (tokens) as compiled into the HEF.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Current context usage (tokens consumed so far in the conversation).
    pub fn context_usage(&self) -> usize {
        let h = self.handle.lock().unwrap();
        let n = unsafe { hailo_ffi::hailo_llm_context_usage(*h) };
        n.max(0) as usize
    }

    /// Clear the conversation context (reset KV-cache for a new conversation).
    pub fn clear_context(&self) -> Result<()> {
        let h = self.handle.lock().unwrap();
        if unsafe { hailo_ffi::hailo_llm_clear_context(*h) } != 0 {
            return Err(anyhow!("clear_context failed"));
        }
        Ok(())
    }

    /// Get the prompt template baked into the HEF.
    pub fn prompt_template(&self) -> Option<String> {
        let h = self.handle.lock().unwrap();
        let ptr = unsafe { hailo_ffi::hailo_llm_prompt_template(*h) };
        if ptr.is_null() {
            return None;
        }
        let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
        unsafe { hailo_ffi::hailo_llm_free_string(ptr) };
        Some(s)
    }

    /// Set custom stop tokens.
    pub fn set_stop_tokens(&self, tokens: &[&str]) -> Result<()> {
        let h = self.handle.lock().unwrap();
        let c_tokens: Vec<CString> = tokens.iter().map(|t| CString::new(*t).unwrap()).collect();
        let c_ptrs: Vec<*const std::ffi::c_char> = c_tokens.iter().map(|t| t.as_ptr() as *const std::ffi::c_char).collect();
        if unsafe {
            hailo_ffi::hailo_llm_set_stop_tokens(*h, c_ptrs.as_ptr(), c_ptrs.len() as i32)
        } != 0
        {
            return Err(anyhow!("set_stop_tokens failed"));
        }
        Ok(())
    }

    /// Generate a response from a raw text prompt. Streams tokens via callback.
    ///
    /// The callback receives UTF-8 token fragments as they're produced by the NPU.
    /// Returns the full concatenated response.
    pub fn generate_raw(
        &self,
        prompt: &str,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        max_tokens: u32,
        callback: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_inner(prompt, None, temperature, top_p, top_k, max_tokens, callback)
    }

    /// Generate a response from structured chat messages (JSON format).
    ///
    /// Each message is a JSON string like `{"role":"user","content":"Hello"}`.
    /// Streams tokens via callback. Returns the full concatenated response.
    pub fn generate_chat(
        &self,
        messages: &[String],
        tools: &[String],
        temperature: f32,
        top_p: f32,
        top_k: u32,
        max_tokens: u32,
        callback: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_inner("", Some((messages, tools)), temperature, top_p, top_k, max_tokens, callback)
    }

    fn generate_inner(
        &self,
        raw_prompt: &str,
        chat: Option<(&[String], &[String])>,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        max_tokens: u32,
        mut callback: impl FnMut(&str),
    ) -> Result<String> {
        let h = self.handle.lock().unwrap();

        // Create generator with sampling params
        let do_sample = if temperature > 0.01 { 1 } else { 0 };
        let genr = unsafe {
            hailo_ffi::hailo_llm_create_generator(
                *h,
                temperature,
                top_p,
                top_k,
                1.0, // frequency_penalty (HailoRT rejects 0.0; 1.0 = standard OpenAI default)
                max_tokens,
                do_sample,
                0, // seed (0 = random)
            )
        };
        if genr.is_null() {
            return Err(anyhow!("create_generator failed"));
        }

        // Write prompt
        let write_ok = if let Some((messages, tools)) = chat {
            let c_msgs: Vec<CString> = messages.iter().map(|m| CString::new(m.as_str()).unwrap()).collect();
            let c_msg_ptrs: Vec<*const std::ffi::c_char> = c_msgs.iter().map(|m| m.as_ptr() as *const std::ffi::c_char).collect();
            let c_tools: Vec<CString> = tools.iter().map(|t| CString::new(t.as_str()).unwrap()).collect();
            let c_tool_ptrs: Vec<*const std::ffi::c_char> = c_tools.iter().map(|t| t.as_ptr() as *const std::ffi::c_char).collect();
            unsafe {
                hailo_ffi::hailo_generator_write_chat(
                    genr,
                    c_msg_ptrs.as_ptr(),
                    c_msg_ptrs.len() as i32,
                    c_tool_ptrs.as_ptr(),
                    c_tool_ptrs.len() as i32,
                )
            }
        } else {
            let c_prompt = CString::new(raw_prompt)?;
            unsafe { hailo_ffi::hailo_generator_write_raw(genr, c_prompt.as_ptr()) }
        };

        if write_ok != 0 {
            unsafe { hailo_ffi::hailo_generator_destroy(genr) };
            return Err(anyhow!("generator write failed"));
        }

        // Start generation
        let comp = unsafe { hailo_ffi::hailo_generator_generate(genr) };
        if comp.is_null() {
            unsafe { hailo_ffi::hailo_generator_destroy(genr) };
            return Err(anyhow!("generate failed"));
        }

        // Stream tokens
        let mut full_response = String::new();
        let mut buf = vec![0u8; 4096]; // UTF-8 token buffer
        loop {
            let status: GenerationStatus =
                unsafe { hailo_ffi::hailo_completion_status(comp) }.into();
            if status != GenerationStatus::Generating {
                break;
            }

            let n = unsafe {
                hailo_ffi::hailo_completion_read(
                    comp,
                    buf.as_mut_ptr() as *mut std::ffi::c_char,
                    buf.len() as i32,
                    10_000, // 10s timeout per token
                )
            };

            if n > 0 {
                let text = std::str::from_utf8(&buf[..n as usize]).unwrap_or("\u{FFFD}");
                callback(text);
                full_response.push_str(text);
            } else if n == 0 {
                break; // end of generation
            } else {
                break; // error
            }
        }

        // Cleanup
        unsafe {
            hailo_ffi::hailo_completion_destroy(comp);
            hailo_ffi::hailo_generator_destroy(genr);
        }

        // Strip any leaked stop tokens from the response
        let cleaned = full_response
            .replace("<|eot_id|>", "")
            .replace("<|end_of_text|>", "")
            .replace("<|im_end|>", "")
            .trim()
            .to_string();

        Ok(cleaned)
    }

    pub fn hef_path(&self) -> &str {
        &self.hef_path
    }
}

impl Drop for Hailo10hEngine {
    fn drop(&mut self) {
        let h = self.handle.lock().unwrap();
        if !(*h).is_null() {
            unsafe { hailo_ffi::hailo_llm_destroy(*h) };
        }
    }
}
