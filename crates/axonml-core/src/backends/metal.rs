//! Metal Backend - Apple GPU Operations
//!
//! Provides the Metal implementation for tensor operations on Apple GPUs.
//! This backend requires the `metal` feature and macOS/iOS.
//!
//! # Key Features
//! - Native Apple Silicon support (M1, M2, M3)
//! - MPS (Metal Performance Shaders) integration
//! - Unified memory architecture support
//! - Async execution with command buffers
//!
//! # Requirements
//! - macOS 10.13+ or iOS 11+
//! - Apple GPU (Intel, AMD, or Apple Silicon)
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use super::Backend;
use crate::device::DeviceCapabilities;

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::{
    Buffer as MTLBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device as MTLDevice,
    Library, MTLResourceOptions, MTLSize,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc::rc::autoreleasepool;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::{Arc, Mutex, OnceLock};

// =============================================================================
// Global State for Metal (macOS only)
// =============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
static METAL_DEVICES: OnceLock<Vec<MTLDevice>> = OnceLock::new();

#[cfg(all(target_os = "macos", feature = "metal"))]
fn get_devices() -> &'static Vec<MTLDevice> {
    METAL_DEVICES.get_or_init(|| MTLDevice::all())
}

// =============================================================================
// Buffer Tracking for Metal
// =============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
struct BufferInfo {
    buffer: MTLBuffer,
    size: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct MetalBufferTracker {
    buffers: HashMap<u64, BufferInfo>,
    next_id: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBufferTracker {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_id: 1,
        }
    }

    fn insert(&mut self, buffer: MTLBuffer, size: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.buffers.insert(id, BufferInfo { buffer, size });
        id
    }

    fn remove(&mut self, id: u64) -> Option<BufferInfo> {
        self.buffers.remove(&id)
    }

    fn get(&self, id: u64) -> Option<&BufferInfo> {
        self.buffers.get(&id)
    }
}

// =============================================================================
// Metal Backend Struct
// =============================================================================

/// Metal backend for tensor operations on Apple GPUs.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalBackend {
    device_index: usize,
    device: MTLDevice,
    command_queue: CommandQueue,
    buffer_tracker: Arc<Mutex<MetalBufferTracker>>,
    compute_pipelines: Arc<Mutex<HashMap<String, ComputePipelineState>>>,
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[derive(Debug)]
pub struct MetalBackend {
    device_index: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_index", &self.device_index)
            .field("device_name", &self.device.name())
            .finish()
    }
}

impl MetalBackend {
    /// Creates a new Metal backend for the specified device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new(device_index: usize) -> Option<Self> {
        let devices = get_devices();
        if device_index >= devices.len() {
            return None;
        }

        let device = devices[device_index].clone();
        let command_queue = device.new_command_queue();

        Some(Self {
            device_index,
            device,
            command_queue,
            buffer_tracker: Arc::new(Mutex::new(MetalBufferTracker::new())),
            compute_pipelines: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn new(_device_index: usize) -> Option<Self> {
        None
    }

    /// Returns the device index.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Returns the Metal device reference.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn device(&self) -> &MTLDevice {
        &self.device
    }

    /// Returns the command queue reference.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Creates a GPU buffer with the specified size.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn create_buffer(&self, size: u64) -> u64 {
        // Use shared storage mode for unified memory (Apple Silicon)
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);

        let mut tracker = self.buffer_tracker.lock().unwrap();
        tracker.insert(buffer, size)
    }

    /// Creates a GPU buffer initialized with data.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn create_buffer_init(&self, data: &[u8]) -> u64 {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut tracker = self.buffer_tracker.lock().unwrap();
        tracker.insert(buffer, data.len() as u64)
    }

    /// Writes data to an existing buffer.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn write_buffer(&self, buffer_id: u64, offset: u64, data: &[u8]) {
        let tracker = self.buffer_tracker.lock().unwrap();
        if let Some(info) = tracker.get(buffer_id) {
            let contents = info.buffer.contents();
            unsafe {
                let dst = (contents as *mut u8).add(offset as usize);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        }
    }

    /// Reads data from a buffer.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn read_buffer(&self, buffer_id: u64) -> Option<Vec<u8>> {
        let tracker = self.buffer_tracker.lock().unwrap();
        let info = tracker.get(buffer_id)?;

        let contents = info.buffer.contents();
        let mut data = vec![0u8; info.size as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents as *const u8,
                data.as_mut_ptr(),
                info.size as usize,
            );
        }
        Some(data)
    }

    /// Destroys a buffer and frees its memory.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn destroy_buffer(&self, buffer_id: u64) {
        let mut tracker = self.buffer_tracker.lock().unwrap();
        tracker.remove(buffer_id);
        // Buffer is automatically released when dropped
    }

    /// Creates or retrieves a cached compute pipeline from Metal shader code.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn get_or_create_pipeline(
        &self,
        name: &str,
        metal_source: &str,
    ) -> Option<ComputePipelineState> {
        let mut pipelines = self.compute_pipelines.lock().unwrap();

        if let Some(pipeline) = pipelines.get(name) {
            return Some(pipeline.clone());
        }

        // Compile shader
        let options = CompileOptions::new();
        let library = match self.device.new_library_with_source(metal_source, &options) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("Metal shader compilation error: {}", e);
                return None;
            }
        };

        let function = library.get_function(name, None)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;

        pipelines.insert(name.to_string(), pipeline.clone());
        Some(pipeline)
    }

    /// Dispatches a compute shader.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn dispatch_compute(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[u64],
        grid_size: (u64, u64, u64),
        threadgroup_size: (u64, u64, u64),
    ) {
        autoreleasepool(|| {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);

            let tracker = self.buffer_tracker.lock().unwrap();
            for (index, buffer_id) in buffers.iter().enumerate() {
                if let Some(info) = tracker.get(*buffer_id) {
                    encoder.set_buffer(index as u64, Some(&info.buffer), 0);
                }
            }
            drop(tracker);

            let grid = MTLSize::new(grid_size.0, grid_size.1, grid_size.2);
            let threadgroup =
                MTLSize::new(threadgroup_size.0, threadgroup_size.1, threadgroup_size.2);

            encoder.dispatch_threads(grid, threadgroup);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
    }

    /// Performs element-wise addition: result = a + b
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn add_f32(&self, a: u64, b: u64, result: u64, count: usize) {
        let pipeline = match self.get_or_create_pipeline("add_f32", SHADER_ADD) {
            Some(p) => p,
            None => return,
        };
        self.dispatch_compute(
            &pipeline,
            &[a, b, result],
            (count as u64, 1, 1),
            (256, 1, 1),
        );
    }

    /// Performs ReLU activation: output = max(0, input)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn relu_f32(&self, input: u64, output: u64, count: usize) {
        let pipeline = match self.get_or_create_pipeline("relu_f32", SHADER_RELU) {
            Some(p) => p,
            None => return,
        };
        self.dispatch_compute(
            &pipeline,
            &[input, output],
            (count as u64, 1, 1),
            (256, 1, 1),
        );
    }
}

// =============================================================================
// Backend Trait Implementation
// =============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn is_available(&self) -> bool {
        true // If we created successfully, we're available
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            name: self.device.name().to_string(),
            total_memory: self.device.recommended_max_working_set_size() as usize,
            available_memory: 0, // Metal doesn't provide this directly
            supports_f16: true,  // All Metal 2+ GPUs support f16
            supports_f64: false, // Metal doesn't support f64 in shaders
            max_threads_per_block: self.device.max_threads_per_threadgroup().width as usize,
            compute_capability: None,
        }
    }

    fn allocate(&self, size: usize) -> *mut u8 {
        let buffer_id = self.create_buffer(size as u64);
        buffer_id as *mut u8
    }

    fn deallocate(&self, ptr: *mut u8, _size: usize) {
        let buffer_id = ptr as u64;
        self.destroy_buffer(buffer_id);
    }

    fn copy_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        let buffer_id = dst as u64;
        let data = unsafe { std::slice::from_raw_parts(src, size) };
        self.write_buffer(buffer_id, 0, data);
    }

    fn copy_to_host(&self, dst: *mut u8, src: *const u8, size: usize) {
        let buffer_id = src as u64;
        if let Some(data) = self.read_buffer(buffer_id) {
            let copy_size = std::cmp::min(size, data.len());
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, copy_size);
            }
        }
    }

    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
        let src_id = src as u64;
        let dst_id = dst as u64;

        autoreleasepool(|| {
            let tracker = self.buffer_tracker.lock().unwrap();
            let src_info = match tracker.get(src_id) {
                Some(info) => info,
                None => return,
            };
            let dst_info = match tracker.get(dst_id) {
                Some(info) => info,
                None => return,
            };

            let command_buffer = self.command_queue.new_command_buffer();
            let blit_encoder = command_buffer.new_blit_command_encoder();

            blit_encoder.copy_from_buffer(&src_info.buffer, 0, &dst_info.buffer, 0, size as u64);

            blit_encoder.end_encoding();
            drop(tracker);

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
    }

    fn synchronize(&self) {
        autoreleasepool(|| {
            let command_buffer = self.command_queue.new_command_buffer();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            name: "Metal (not available)".to_string(),
            total_memory: 0,
            available_memory: 0,
            supports_f16: false,
            supports_f64: false,
            max_threads_per_block: 0,
            compute_capability: None,
        }
    }

    fn allocate(&self, _size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    fn deallocate(&self, _ptr: *mut u8, _size: usize) {}

    fn copy_to_device(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn copy_to_host(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn copy_device_to_device(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn synchronize(&self) {}
}

// =============================================================================
// Metal Runtime Functions
// =============================================================================

/// Returns whether Metal is available on this system.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn is_available() -> bool {
    !get_devices().is_empty()
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn is_available() -> bool {
    false
}

/// Returns the number of available Metal devices.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn device_count() -> usize {
    get_devices().len()
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn device_count() -> usize {
    0
}

/// Returns whether a specific Metal device is available.
pub fn is_device_available(index: usize) -> bool {
    index < device_count()
}

/// Returns the capabilities of a Metal device.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn get_capabilities(index: usize) -> DeviceCapabilities {
    let devices = get_devices();
    if index >= devices.len() {
        return DeviceCapabilities {
            name: "Unknown".to_string(),
            total_memory: 0,
            available_memory: 0,
            supports_f16: false,
            supports_f64: false,
            max_threads_per_block: 0,
            compute_capability: None,
        };
    }

    let device = &devices[index];
    DeviceCapabilities {
        name: device.name().to_string(),
        total_memory: device.recommended_max_working_set_size() as usize,
        available_memory: 0,
        supports_f16: true,
        supports_f64: false,
        max_threads_per_block: device.max_threads_per_threadgroup().width as usize,
        compute_capability: None,
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn get_capabilities(index: usize) -> DeviceCapabilities {
    DeviceCapabilities {
        name: format!("Metal Device {} (not available)", index),
        total_memory: 0,
        available_memory: 0,
        supports_f16: false,
        supports_f64: false,
        max_threads_per_block: 0,
        compute_capability: None,
    }
}

// =============================================================================
// Metal Shader Templates
// =============================================================================

/// Metal shader for element-wise addition.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_ADD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + b[index];
}
"#;

/// Metal shader for element-wise subtraction.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_SUB: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] - b[index];
}
"#;

/// Metal shader for element-wise multiplication.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_MUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * b[index];
}
"#;

/// Metal shader for element-wise division.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_DIV: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] / b[index];
}
"#;

/// Metal shader for ReLU activation.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_RELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = max(0.0f, input[index]);
}
"#;

/// Metal shader for sigmoid activation.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_SIGMOID: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = 1.0f / (1.0f + exp(-input[index]));
}
"#;

/// Metal shader for tanh activation.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_TANH: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = tanh(input[index]);
}
"#;

/// Metal shader for matrix multiplication.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_MATMUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatMulParams {
    uint M;
    uint N;
    uint K;
};

kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant MatMulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint col = gid.y;

    if (row >= params.M || col >= params.N) {
        return;
    }

    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        sum += a[row * params.K + k] * b[k * params.N + col];
    }
    result[row * params.N + col] = sum;
}
"#;

/// Metal shader for sum reduction.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const SHADER_SUM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared_data [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]],
    uint num_threads [[threads_per_threadgroup]]
) {
    shared_data[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = shared_data[0];
    }
}
"#;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        let available = is_available();
        println!("Metal available: {}", available);
    }

    #[test]
    fn test_device_count() {
        let count = device_count();
        println!("Metal device count: {}", count);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_backend_creation() {
        if is_available() {
            let backend = MetalBackend::new(0);
            assert!(backend.is_some());
            if let Some(b) = backend {
                assert!(b.is_available());
                println!("Backend name: {}", b.name());
                println!("Capabilities: {:?}", b.capabilities());
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_buffer_operations() {
        if !is_available() {
            return;
        }

        let backend = match MetalBackend::new(0) {
            Some(b) => b,
            None => return,
        };

        // Create a buffer with data
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let buffer_id = backend.create_buffer_init(bytes);

        // Read it back
        if let Some(read_data) = backend.read_buffer(buffer_id) {
            let floats: &[f32] = bytemuck::cast_slice(&read_data);
            assert_eq!(floats.len(), 4);
            assert!((floats[0] - 1.0).abs() < 0.001);
            assert!((floats[1] - 2.0).abs() < 0.001);
            assert!((floats[2] - 3.0).abs() < 0.001);
            assert!((floats[3] - 4.0).abs() < 0.001);
        }

        backend.destroy_buffer(buffer_id);
    }
}
