//! hailo_custom — Standard HailoRT inference for custom HEFs.
//!
//! Unlike hailo10h.rs (GenAI LLM API), this module uses the standard
//! VDevice → InferModel → Bindings flow for custom-compiled HEFs from
//! AxonML/NexusFoundry. Supports any model topology, not just LLMs.
//!
//! The inference loop for LLMs:
//!   CPU: tokenize → embed → reshape to NCHW
//!   NPU: transformer forward (this module)
//!   CPU: reshape → lm_head → sample → stream

#![cfg(feature = "hailo10h")]

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow};

use super::hailo_infer_ffi;

pub struct HailoCustomEngine {
    ctx: Mutex<*mut hailo_infer_ffi::HailoInferCtx>,
    hef_path: String,
    input_frame_size: usize,
    output_frame_size: usize,
}

unsafe impl Send for HailoCustomEngine {}
unsafe impl Sync for HailoCustomEngine {}

impl HailoCustomEngine {
    pub fn load(hef_path: impl AsRef<Path>) -> Result<Self> {
        let path_str = hef_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow!("HEF path not valid UTF-8"))?;
        let c_path = CString::new(path_str)?;

        let ctx = unsafe { hailo_infer_ffi::hailo_infer_create(c_path.as_ptr()) };
        if ctx.is_null() {
            let err = unsafe {
                CStr::from_ptr(hailo_infer_ffi::hailo_infer_last_error())
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(anyhow!("hailo_infer_create failed: {err}"));
        }

        let num_inputs = unsafe { hailo_infer_ffi::hailo_infer_num_inputs(ctx) };
        let num_outputs = unsafe { hailo_infer_ffi::hailo_infer_num_outputs(ctx) };
        let in_size = unsafe { hailo_infer_ffi::hailo_infer_input_frame_size(ctx) };
        let out_size = unsafe { hailo_infer_ffi::hailo_infer_output_frame_size(ctx) };

        tracing::info!(
            hef = path_str,
            inputs = num_inputs,
            outputs = num_outputs,
            input_frame_size = in_size,
            output_frame_size = out_size,
            "Hailo custom HEF loaded",
        );

        Ok(Self {
            ctx: Mutex::new(ctx),
            hef_path: path_str.to_string(),
            input_frame_size: in_size.max(0) as usize,
            output_frame_size: out_size.max(0) as usize,
        })
    }

    /// Run synchronous inference on the NPU.
    /// Input/output are raw byte buffers (quantized UINT8 in NHWC format).
    pub fn input_frame_size(&self) -> usize { self.input_frame_size }
    pub fn output_frame_size(&self) -> usize { self.output_frame_size }

    pub fn infer(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let h = self.ctx.lock().unwrap();
        let rc = unsafe {
            hailo_infer_ffi::hailo_infer_run(
                *h,
                input.as_ptr(),
                input.len() as i64,
                output.as_mut_ptr(),
                output.len() as i64,
            )
        };
        if rc != 0 {
            let err = unsafe {
                CStr::from_ptr(hailo_infer_ffi::hailo_infer_last_error())
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(anyhow!("hailo_infer_run failed: {err}"));
        }
        Ok(())
    }

    pub fn hef_path(&self) -> &str {
        &self.hef_path
    }
}

impl Drop for HailoCustomEngine {
    fn drop(&mut self) {
        let h = self.ctx.lock().unwrap();
        if !(*h).is_null() {
            unsafe { hailo_infer_ffi::hailo_infer_destroy(*h) };
        }
    }
}
