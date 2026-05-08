//! Raw FFI bindings for the standard HailoRT inference shim.
//! For custom HEFs compiled from AxonML/NexusFoundry models.

#![cfg(feature = "hailo10h")]

use std::ffi::c_char;
use std::ffi::c_void;

pub type HailoInferCtx = c_void;

unsafe extern "C" {
    pub fn hailo_infer_last_error() -> *const c_char;
    pub fn hailo_infer_create(hef_path: *const c_char) -> *mut HailoInferCtx;
    pub fn hailo_infer_destroy(ctx: *mut HailoInferCtx);
    pub fn hailo_infer_num_inputs(ctx: *mut HailoInferCtx) -> i32;
    pub fn hailo_infer_num_outputs(ctx: *mut HailoInferCtx) -> i32;
    pub fn hailo_infer_input_frame_size(ctx: *mut HailoInferCtx) -> i64;
    pub fn hailo_infer_output_frame_size(ctx: *mut HailoInferCtx) -> i64;
    pub fn hailo_infer_run(
        ctx: *mut HailoInferCtx,
        input_data: *const u8,
        input_size: i64,
        output_data: *mut u8,
        output_size: i64,
    ) -> i32;
}
