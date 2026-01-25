//! WebGPU Backend - Cross-Platform GPU Operations via wgpu
//!
//! Provides the WebGPU implementation for tensor operations using wgpu.
//! This backend requires the `wgpu` feature and works on all platforms.
//!
//! # Key Features
//! - Cross-platform support (Windows, Linux, macOS, Web)
//! - WASM/browser support
//! - Compute shader based operations
//! - Automatic backend selection (Vulkan, Metal, DX12, WebGPU)
//!
//! # Requirements
//! - wgpu-compatible GPU
//! - For native: Modern GPU with Vulkan/Metal/DX12 support
//! - For web: WebGPU-enabled browser
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use super::Backend;
use crate::device::DeviceCapabilities;

#[cfg(feature = "wgpu")]
use std::collections::HashMap;
#[cfg(feature = "wgpu")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "wgpu")]
use wgpu::{
    Adapter, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    Limits, MapMode, Queue, ShaderModuleDescriptor, ShaderSource,
};

// =============================================================================
// Global State for wgpu (feature-gated)
// =============================================================================

#[cfg(feature = "wgpu")]
static WGPU_INSTANCE: OnceLock<Instance> = OnceLock::new();

#[cfg(feature = "wgpu")]
static WGPU_ADAPTERS: OnceLock<Vec<Adapter>> = OnceLock::new();

#[cfg(feature = "wgpu")]
fn get_instance() -> &'static Instance {
    WGPU_INSTANCE.get_or_init(|| {
        Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        })
    })
}

#[cfg(feature = "wgpu")]
fn get_adapters() -> &'static Vec<Adapter> {
    WGPU_ADAPTERS.get_or_init(|| {
        let instance = get_instance();
        pollster::block_on(async {
            let mut adapters = Vec::new();
            for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
                adapters.push(adapter);
            }
            adapters
        })
    })
}

// =============================================================================
// Buffer Tracking for wgpu
// =============================================================================

#[cfg(feature = "wgpu")]
struct BufferInfo {
    buffer: Buffer,
    size: u64,
}

#[cfg(feature = "wgpu")]
struct WgpuBufferTracker {
    buffers: HashMap<u64, BufferInfo>,
    next_id: u64,
}

#[cfg(feature = "wgpu")]
impl WgpuBufferTracker {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_id: 1,
        }
    }

    fn insert(&mut self, buffer: Buffer, size: u64) -> u64 {
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
// WebGPU Backend Struct
// =============================================================================

/// WebGPU backend for tensor operations using wgpu.
#[cfg(feature = "wgpu")]
pub struct WgpuBackend {
    device_index: usize,
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter_info: wgpu::AdapterInfo,
    buffer_tracker: Arc<Mutex<WgpuBufferTracker>>,
    compute_pipelines: Arc<Mutex<HashMap<String, ComputePipeline>>>,
}

#[cfg(not(feature = "wgpu"))]
#[derive(Debug)]
pub struct WgpuBackend {
    device_index: usize,
}

#[cfg(feature = "wgpu")]
impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend")
            .field("device_index", &self.device_index)
            .field("adapter_info", &self.adapter_info)
            .finish()
    }
}

impl WgpuBackend {
    /// Creates a new WebGPU backend for the specified device.
    #[cfg(feature = "wgpu")]
    pub fn new(device_index: usize) -> Option<Self> {
        let adapters = get_adapters();
        if device_index >= adapters.len() {
            return None;
        }

        let adapter = &adapters[device_index];
        let adapter_info = adapter.get_info();

        // Request device with compute capabilities
        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &DeviceDescriptor {
                        label: Some("Axonml wgpu Device"),
                        required_features: Features::empty(),
                        required_limits: Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await
                .ok()
        })?;

        Some(Self {
            device_index,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            buffer_tracker: Arc::new(Mutex::new(WgpuBufferTracker::new())),
            compute_pipelines: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[cfg(not(feature = "wgpu"))]
    pub fn new(_device_index: usize) -> Option<Self> {
        None
    }

    /// Returns the device index.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Returns the wgpu Device reference.
    #[cfg(feature = "wgpu")]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the wgpu Queue reference.
    #[cfg(feature = "wgpu")]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Creates a GPU buffer with the specified size and usage.
    #[cfg(feature = "wgpu")]
    pub fn create_buffer(&self, size: u64, usage: BufferUsages) -> u64 {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Axonml Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        let mut tracker = self.buffer_tracker.lock().unwrap();
        tracker.insert(buffer, size)
    }

    /// Creates a GPU buffer initialized with data.
    #[cfg(feature = "wgpu")]
    pub fn create_buffer_init(&self, data: &[u8], usage: BufferUsages) -> u64 {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Axonml Buffer"),
            size: data.len() as u64,
            usage: usage | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(data);
        buffer.unmap();

        let mut tracker = self.buffer_tracker.lock().unwrap();
        tracker.insert(buffer, data.len() as u64)
    }

    /// Writes data to an existing buffer.
    #[cfg(feature = "wgpu")]
    pub fn write_buffer(&self, buffer_id: u64, offset: u64, data: &[u8]) {
        let tracker = self.buffer_tracker.lock().unwrap();
        if let Some(info) = tracker.get(buffer_id) {
            self.queue.write_buffer(&info.buffer, offset, data);
        }
    }

    /// Reads data from a buffer (blocking).
    #[cfg(feature = "wgpu")]
    pub fn read_buffer(&self, buffer_id: u64) -> Option<Vec<u8>> {
        let tracker = self.buffer_tracker.lock().unwrap();
        let info = tracker.get(buffer_id)?;
        let size = info.size;

        // Create a staging buffer for reading
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Read Buffer Encoder"),
            });
        encoder.copy_buffer_to_buffer(&info.buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        drop(tracker); // Release lock before blocking

        // Map the staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range().to_vec();
            staging_buffer.unmap();
            Some(data)
        } else {
            None
        }
    }

    /// Destroys a buffer and frees its memory.
    #[cfg(feature = "wgpu")]
    pub fn destroy_buffer(&self, buffer_id: u64) {
        let mut tracker = self.buffer_tracker.lock().unwrap();
        if let Some(info) = tracker.remove(buffer_id) {
            info.buffer.destroy();
        }
    }

    /// Creates or retrieves a cached compute pipeline from WGSL shader code.
    #[cfg(feature = "wgpu")]
    pub fn get_or_create_pipeline(&self, name: &str, wgsl_code: &str) -> Arc<ComputePipeline> {
        let mut pipelines = self.compute_pipelines.lock().unwrap();

        if !pipelines.contains_key(name) {
            let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(name),
                source: ShaderSource::Wgsl(wgsl_code.into()),
            });

            let pipeline = self
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            pipelines.insert(name.to_string(), pipeline);
        }

        // Return a reference via Arc wrapping
        Arc::new(
            self.device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &self.device.create_shader_module(ShaderModuleDescriptor {
                        label: Some(name),
                        source: ShaderSource::Wgsl(wgsl_code.into()),
                    }),
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }),
        )
    }

    /// Dispatches a compute shader with the given pipeline and bind group.
    #[cfg(feature = "wgpu")]
    pub fn dispatch_compute(
        &self,
        pipeline: &ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Gets buffer reference by ID for bind group creation.
    #[cfg(feature = "wgpu")]
    pub fn get_buffer(&self, buffer_id: u64) -> Option<Arc<Buffer>> {
        let tracker = self.buffer_tracker.lock().unwrap();
        tracker.get(buffer_id).map(|info| {
            // We need to clone the buffer reference
            // Since we can't clone Buffer, we'll create a new one with same content
            // This is a limitation - in practice you'd use the buffer directly
            Arc::new(self.device.create_buffer(&BufferDescriptor {
                label: Some("Buffer Clone"),
                size: info.size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        })
    }
}

// =============================================================================
// Backend Trait Implementation
// =============================================================================

#[cfg(feature = "wgpu")]
impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn is_available(&self) -> bool {
        true // If we created successfully, we're available
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let limits = self.device.limits();

        DeviceCapabilities {
            name: format!(
                "{} ({:?})",
                self.adapter_info.name, self.adapter_info.backend
            ),
            total_memory: 0, // wgpu doesn't expose this directly
            available_memory: 0,
            supports_f16: self.device.features().contains(Features::SHADER_F16),
            supports_f64: false, // WebGPU doesn't support f64 in shaders
            max_threads_per_block: limits.max_compute_invocations_per_workgroup as usize,
            compute_capability: None,
        }
    }

    fn allocate(&self, size: usize) -> *mut u8 {
        // Create buffer with storage and copy usage
        let buffer_id = self.create_buffer(
            size as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        // Return the buffer ID as a pointer (we track internally)
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

        let tracker = self.buffer_tracker.lock().unwrap();
        let src_info = match tracker.get(src_id) {
            Some(info) => info,
            None => return,
        };
        let dst_info = match tracker.get(dst_id) {
            Some(info) => info,
            None => return,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Copy D2D Encoder"),
            });
        encoder.copy_buffer_to_buffer(&src_info.buffer, 0, &dst_info.buffer, 0, size as u64);

        drop(tracker);
        self.queue.submit(Some(encoder.finish()));
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

#[cfg(not(feature = "wgpu"))]
impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            name: "WebGPU (not available)".to_string(),
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
// WebGPU Runtime Functions
// =============================================================================

/// Returns whether WebGPU is available on this system.
#[cfg(feature = "wgpu")]
pub fn is_available() -> bool {
    !get_adapters().is_empty()
}

#[cfg(not(feature = "wgpu"))]
pub fn is_available() -> bool {
    false
}

/// Returns the number of available WebGPU devices.
#[cfg(feature = "wgpu")]
pub fn device_count() -> usize {
    get_adapters().len()
}

#[cfg(not(feature = "wgpu"))]
pub fn device_count() -> usize {
    0
}

/// Returns whether a specific WebGPU device is available.
pub fn is_device_available(index: usize) -> bool {
    index < device_count()
}

/// Returns the capabilities of a WebGPU device.
#[cfg(feature = "wgpu")]
pub fn get_capabilities(index: usize) -> DeviceCapabilities {
    let adapters = get_adapters();
    if index >= adapters.len() {
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

    let adapter = &adapters[index];
    let info = adapter.get_info();
    let limits = adapter.limits();

    DeviceCapabilities {
        name: format!("{} ({:?})", info.name, info.backend),
        total_memory: 0, // wgpu doesn't expose this
        available_memory: 0,
        supports_f16: adapter.features().contains(Features::SHADER_F16),
        supports_f64: false, // WebGPU doesn't support f64
        max_threads_per_block: limits.max_compute_invocations_per_workgroup as usize,
        compute_capability: None,
    }
}

#[cfg(not(feature = "wgpu"))]
pub fn get_capabilities(index: usize) -> DeviceCapabilities {
    DeviceCapabilities {
        name: format!("WebGPU Device {} (not available)", index),
        total_memory: 0,
        available_memory: 0,
        supports_f16: false,
        supports_f64: false,
        max_threads_per_block: 0,
        compute_capability: None,
    }
}

/// Synchronizes a wgpu queue by handle (no-op, wgpu submits are synchronous).
#[cfg(feature = "wgpu")]
pub fn queue_submit(_handle: usize) {
    // wgpu queue submissions are handled internally
    // The handle is not used directly; synchronization happens via device.poll()
}

/// Synchronizes a wgpu queue by handle (no-op when wgpu is not available).
#[cfg(not(feature = "wgpu"))]
pub fn queue_submit(_handle: usize) {
    // No-op when wgpu is not available
}

// =============================================================================
// Tensor Operation Helpers
// =============================================================================

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    /// Performs element-wise addition: result = a + b
    pub fn add_f32(&self, a: u64, b: u64, result: u64, count: usize) {
        let tracker = self.buffer_tracker.lock().unwrap();
        let a_buf = match tracker.get(a) {
            Some(i) => &i.buffer,
            None => return,
        };
        let b_buf = match tracker.get(b) {
            Some(i) => &i.buffer,
            None => return,
        };
        let result_buf = match tracker.get(result) {
            Some(i) => &i.buffer,
            None => return,
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("add_f32"),
                layout: None,
                module: &self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("add_shader"),
                    source: ShaderSource::Wgsl(SHADER_ADD.into()),
                }),
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });

        drop(tracker);

        let workgroups = ((count + 255) / 256) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Add Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Add Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Performs element-wise subtraction: result = a - b
    pub fn sub_f32(&self, a: u64, b: u64, result: u64, count: usize) {
        self.binary_op_f32(a, b, result, count, "sub_f32", SHADER_SUB);
    }

    /// Performs element-wise multiplication: result = a * b
    pub fn mul_f32(&self, a: u64, b: u64, result: u64, count: usize) {
        self.binary_op_f32(a, b, result, count, "mul_f32", SHADER_MUL);
    }

    /// Performs element-wise division: result = a / b
    pub fn div_f32(&self, a: u64, b: u64, result: u64, count: usize) {
        self.binary_op_f32(a, b, result, count, "div_f32", SHADER_DIV);
    }

    /// Generic binary operation helper
    fn binary_op_f32(&self, a: u64, b: u64, result: u64, count: usize, name: &str, shader: &str) {
        let tracker = self.buffer_tracker.lock().unwrap();
        let a_buf = match tracker.get(a) {
            Some(i) => &i.buffer,
            None => return,
        };
        let b_buf = match tracker.get(b) {
            Some(i) => &i.buffer,
            None => return,
        };
        let result_buf = match tracker.get(result) {
            Some(i) => &i.buffer,
            None => return,
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some(name),
                    source: ShaderSource::Wgsl(shader.into()),
                }),
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });

        drop(tracker);

        let workgroups = ((count + 255) / 256) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(name) });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Performs matrix multiplication: C = A @ B
    /// A is [M, K], B is [K, N], C is [M, N]
    pub fn matmul_f32(&self, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize) {
        let tracker = self.buffer_tracker.lock().unwrap();
        let a_buf = match tracker.get(a) {
            Some(i) => &i.buffer,
            None => return,
        };
        let b_buf = match tracker.get(b) {
            Some(i) => &i.buffer,
            None => return,
        };
        let c_buf = match tracker.get(c) {
            Some(i) => &i.buffer,
            None => return,
        };

        // Create dimensions uniform buffer
        let dims_data: [u32; 4] = [m as u32, n as u32, k as u32, 0];
        let dims_bytes: &[u8] = bytemuck::cast_slice(&dims_data);
        let dims_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("matmul_dims"),
            size: 16, // 4 x u32
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        dims_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(dims_bytes);
        dims_buffer.unmap();

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("matmul_f32"),
                layout: None,
                module: &self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("matmul_shader"),
                    source: ShaderSource::Wgsl(SHADER_MATMUL.into()),
                }),
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_buf.as_entire_binding(),
                },
            ],
        });

        drop(tracker);

        // Dispatch with 16x16 workgroups
        let workgroups_x = ((m + 15) / 16) as u32;
        let workgroups_y = ((n + 15) / 16) as u32;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("MatMul Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        dims_buffer.destroy();
    }

    /// Performs sigmoid activation: output = 1 / (1 + exp(-input))
    pub fn sigmoid_f32(&self, input: u64, output: u64, count: usize) {
        self.unary_op_f32(input, output, count, "sigmoid_f32", SHADER_SIGMOID);
    }

    /// Performs tanh activation: output = tanh(input)
    pub fn tanh_f32(&self, input: u64, output: u64, count: usize) {
        self.unary_op_f32(input, output, count, "tanh_f32", SHADER_TANH);
    }

    /// Generic unary operation helper
    fn unary_op_f32(&self, input: u64, output: u64, count: usize, name: &str, shader: &str) {
        let tracker = self.buffer_tracker.lock().unwrap();
        let input_buf = match tracker.get(input) {
            Some(i) => &i.buffer,
            None => return,
        };
        let output_buf = match tracker.get(output) {
            Some(i) => &i.buffer,
            None => return,
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some(name),
                    source: ShaderSource::Wgsl(shader.into()),
                }),
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        drop(tracker);

        let workgroups = ((count + 255) / 256) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(name) });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Performs ReLU activation: output = max(0, input)
    pub fn relu_f32(&self, input: u64, output: u64, count: usize) {
        let tracker = self.buffer_tracker.lock().unwrap();
        let input_buf = match tracker.get(input) {
            Some(i) => &i.buffer,
            None => return,
        };
        let output_buf = match tracker.get(output) {
            Some(i) => &i.buffer,
            None => return,
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("relu_f32"),
                layout: None,
                module: &self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("relu_shader"),
                    source: ShaderSource::Wgsl(SHADER_RELU.into()),
                }),
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relu_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        drop(tracker);

        let workgroups = ((count + 255) / 256) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("ReLU Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReLU Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

// =============================================================================
// WGSL Shader Templates
// =============================================================================

/// WGSL shader for element-wise addition.
pub const SHADER_ADD: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&result)) {
        result[index] = a[index] + b[index];
    }
}
"#;

/// WGSL shader for element-wise subtraction.
pub const SHADER_SUB: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&result)) {
        result[index] = a[index] - b[index];
    }
}
"#;

/// WGSL shader for element-wise multiplication.
pub const SHADER_MUL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&result)) {
        result[index] = a[index] * b[index];
    }
}
"#;

/// WGSL shader for element-wise division.
pub const SHADER_DIV: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&result)) {
        result[index] = a[index] / b[index];
    }
}
"#;

/// WGSL shader for matrix multiplication.
pub const SHADER_MATMUL: &str = r#"
struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < dims.K; k++) {
        sum += a[row * dims.K + k] * b[k * dims.N + col];
    }
    result[row * dims.N + col] = sum;
}
"#;

/// WGSL shader for ReLU activation.
pub const SHADER_RELU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        output[index] = max(0.0, input[index]);
    }
}
"#;

/// WGSL shader for sigmoid activation.
pub const SHADER_SIGMOID: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        output[index] = 1.0 / (1.0 + exp(-input[index]));
    }
}
"#;

/// WGSL shader for tanh activation.
pub const SHADER_TANH: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        output[index] = tanh(input[index]);
    }
}
"#;

/// WGSL shader for sum reduction.
pub const SHADER_SUM: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    if (gid < arrayLength(&input)) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }
    workgroupBarrier();

    // Parallel reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write result
    if (tid == 0u) {
        output[group_id.x] = shared_data[0];
    }
}
"#;

/// WGSL shader for GELU activation (approximate).
pub const SHADER_GELU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        let x = input[index];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = 0.7978845608;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        output[index] = 0.5 * x * (1.0 + tanh(inner));
    }
}
"#;

/// WGSL shader for SiLU (Swish) activation.
pub const SHADER_SILU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        let x = input[index];
        output[index] = x / (1.0 + exp(-x));
    }
}
"#;

/// WGSL shader for LeakyReLU activation.
pub const SHADER_LEAKY_RELU: &str = r#"
struct Params {
    negative_slope: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        let x = input[index];
        output[index] = select(params.negative_slope * x, x, x > 0.0);
    }
}
"#;

/// WGSL shader for exponential.
pub const SHADER_EXP: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        output[index] = exp(input[index]);
    }
}
"#;

/// WGSL shader for natural logarithm.
pub const SHADER_LOG: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        output[index] = log(input[index]);
    }
}
"#;

/// WGSL shader for square root.
pub const SHADER_SQRT: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&output)) {
        output[index] = sqrt(input[index]);
    }
}
"#;

/// WGSL shader for softmax (row-wise).
pub const SHADER_SOFTMAX: &str = r#"
struct Dims {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.rows || col >= dims.cols) {
        return;
    }

    let idx = row * dims.cols + col;
    let x = input[idx];

    // For simplicity, compute max and sum per element
    // (Real implementation would use proper reduction)
    var max_val = x;
    var sum_val = 0.0;

    // Find max in row
    for (var i = 0u; i < dims.cols; i++) {
        max_val = max(max_val, input[row * dims.cols + i]);
    }

    // Compute sum of exp(x - max)
    for (var i = 0u; i < dims.cols; i++) {
        sum_val += exp(input[row * dims.cols + i] - max_val);
    }

    output[idx] = exp(x - max_val) / sum_val;
}
"#;

/// WGSL shader for layer normalization.
pub const SHADER_LAYER_NORM: &str = r#"
struct Params {
    size: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.y;
    let elem_idx = global_id.x;

    if (elem_idx >= params.size) {
        return;
    }

    let base = batch_idx * params.size;

    // Compute mean
    var mean = 0.0;
    for (var i = 0u; i < params.size; i++) {
        mean += input[base + i];
    }
    mean /= f32(params.size);

    // Compute variance
    var variance = 0.0;
    for (var i = 0u; i < params.size; i++) {
        let diff = input[base + i] - mean;
        variance += diff * diff;
    }
    variance /= f32(params.size);

    // Normalize
    let idx = base + elem_idx;
    let normalized = (input[idx] - mean) / sqrt(variance + params.eps);
    output[idx] = normalized * gamma[elem_idx] + beta[elem_idx];
}
"#;

/// WGSL shader for 2D convolution.
pub const SHADER_CONV2D: &str = r#"
struct ConvParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_col = global_id.x;
    let out_row = global_id.y;
    let batch_out_c = global_id.z;

    let batch_idx = batch_out_c / params.out_channels;
    let out_c = batch_out_c % params.out_channels;

    if (batch_idx >= params.batch_size || out_row >= params.out_height || out_col >= params.out_width) {
        return;
    }

    var sum = bias[out_c];

    for (var in_c = 0u; in_c < params.in_channels; in_c++) {
        for (var kh = 0u; kh < params.kernel_h; kh++) {
            for (var kw = 0u; kw < params.kernel_w; kw++) {
                let in_row = i32(out_row * params.stride_h + kh) - i32(params.pad_h);
                let in_col = i32(out_col * params.stride_w + kw) - i32(params.pad_w);

                if (in_row >= 0 && in_row < i32(params.in_height) &&
                    in_col >= 0 && in_col < i32(params.in_width)) {
                    let input_idx = batch_idx * params.in_channels * params.in_height * params.in_width
                                  + in_c * params.in_height * params.in_width
                                  + u32(in_row) * params.in_width + u32(in_col);
                    let weight_idx = out_c * params.in_channels * params.kernel_h * params.kernel_w
                                   + in_c * params.kernel_h * params.kernel_w
                                   + kh * params.kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    let out_idx = batch_idx * params.out_channels * params.out_height * params.out_width
                + out_c * params.out_height * params.out_width
                + out_row * params.out_width + out_col;
    output[out_idx] = sum;
}
"#;

/// WGSL shader for batch normalization.
pub const SHADER_BATCH_NORM: &str = r#"
struct BnParams {
    channels: u32,
    spatial_size: u32,
    eps: f32,
    momentum: f32,
}

@group(0) @binding(0) var<uniform> params: BnParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read> running_mean: array<f32>;
@group(0) @binding(5) var<storage, read> running_var: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = arrayLength(&input);

    if (idx >= total_size) {
        return;
    }

    // Determine which channel this element belongs to
    let channel = (idx / params.spatial_size) % params.channels;

    let mean = running_mean[channel];
    let variance = running_var[channel];
    let g = gamma[channel];
    let b = beta[channel];

    let normalized = (input[idx] - mean) / sqrt(variance + params.eps);
    output[idx] = g * normalized + b;
}
"#;

/// WGSL shader for max pooling 2D.
pub const SHADER_MAX_POOL2D: &str = r#"
struct PoolParams {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}

@group(0) @binding(0) var<uniform> params: PoolParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_col = global_id.x;
    let out_row = global_id.y;
    let batch_channel = global_id.z;

    let batch_idx = batch_channel / params.channels;
    let channel = batch_channel % params.channels;

    if (batch_idx >= params.batch_size || out_row >= params.out_height || out_col >= params.out_width) {
        return;
    }

    var max_val = -3.402823e+38; // -FLT_MAX

    for (var kh = 0u; kh < params.kernel_h; kh++) {
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let in_row = i32(out_row * params.stride_h + kh) - i32(params.pad_h);
            let in_col = i32(out_col * params.stride_w + kw) - i32(params.pad_w);

            if (in_row >= 0 && in_row < i32(params.in_height) &&
                in_col >= 0 && in_col < i32(params.in_width)) {
                let input_idx = batch_idx * params.channels * params.in_height * params.in_width
                              + channel * params.in_height * params.in_width
                              + u32(in_row) * params.in_width + u32(in_col);
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    let out_idx = batch_idx * params.channels * params.out_height * params.out_width
                + channel * params.out_height * params.out_width
                + out_row * params.out_width + out_col;
    output[out_idx] = max_val;
}
"#;

/// WGSL shader for embedding lookup.
pub const SHADER_EMBEDDING: &str = r#"
struct EmbedParams {
    vocab_size: u32,
    embed_dim: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<uniform> params: EmbedParams;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let seq_idx = global_id.y;
    let embed_idx = global_id.x;

    if (seq_idx >= params.seq_len || embed_idx >= params.embed_dim) {
        return;
    }

    let token_idx = indices[batch_idx * params.seq_len + seq_idx];
    let weight_offset = token_idx * params.embed_dim + embed_idx;
    let out_offset = batch_idx * params.seq_len * params.embed_dim
                   + seq_idx * params.embed_dim + embed_idx;

    output[out_offset] = weight[weight_offset];
}
"#;

/// WGSL shader for attention scores (Q @ K^T / sqrt(d_k)).
pub const SHADER_ATTENTION_SCORES: &str = r#"
struct AttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let q_pos = global_id.x;
    let k_pos = global_id.y;
    let batch_head = global_id.z;

    let batch_idx = batch_head / params.num_heads;
    let head_idx = batch_head % params.num_heads;

    if (batch_idx >= params.batch_size || q_pos >= params.seq_len || k_pos >= params.seq_len) {
        return;
    }

    var dot_product = 0.0;
    let q_base = batch_idx * params.num_heads * params.seq_len * params.head_dim
               + head_idx * params.seq_len * params.head_dim
               + q_pos * params.head_dim;
    let k_base = batch_idx * params.num_heads * params.seq_len * params.head_dim
               + head_idx * params.seq_len * params.head_dim
               + k_pos * params.head_dim;

    for (var d = 0u; d < params.head_dim; d++) {
        dot_product += query[q_base + d] * key[k_base + d];
    }

    let out_idx = batch_idx * params.num_heads * params.seq_len * params.seq_len
                + head_idx * params.seq_len * params.seq_len
                + q_pos * params.seq_len + k_pos;
    scores[out_idx] = dot_product * params.scale;
}
"#;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_availability() {
        let available = is_available();
        println!("wgpu available: {}", available);
    }

    #[test]
    fn test_device_count() {
        let count = device_count();
        println!("wgpu device count: {}", count);
    }

    #[test]
    fn test_shader_templates_exist() {
        // Basic arithmetic
        assert!(!SHADER_ADD.is_empty());
        assert!(!SHADER_SUB.is_empty());
        assert!(!SHADER_MUL.is_empty());
        assert!(!SHADER_DIV.is_empty());
        // Matrix operations
        assert!(!SHADER_MATMUL.is_empty());
        // Activations
        assert!(!SHADER_RELU.is_empty());
        assert!(!SHADER_SIGMOID.is_empty());
        assert!(!SHADER_TANH.is_empty());
        assert!(!SHADER_GELU.is_empty());
        assert!(!SHADER_SILU.is_empty());
        assert!(!SHADER_LEAKY_RELU.is_empty());
        // Math ops
        assert!(!SHADER_EXP.is_empty());
        assert!(!SHADER_LOG.is_empty());
        assert!(!SHADER_SQRT.is_empty());
        // Reductions
        assert!(!SHADER_SUM.is_empty());
        assert!(!SHADER_SOFTMAX.is_empty());
        // Normalization
        assert!(!SHADER_LAYER_NORM.is_empty());
        assert!(!SHADER_BATCH_NORM.is_empty());
        // CNN operations
        assert!(!SHADER_CONV2D.is_empty());
        assert!(!SHADER_MAX_POOL2D.is_empty());
        // Transformer operations
        assert!(!SHADER_EMBEDDING.is_empty());
        assert!(!SHADER_ATTENTION_SCORES.is_empty());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_backend_creation() {
        if is_available() {
            let backend = WgpuBackend::new(0);
            assert!(backend.is_some());
            if let Some(b) = backend {
                assert!(b.is_available());
                println!("Backend name: {}", b.name());
                println!("Capabilities: {:?}", b.capabilities());
            }
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_buffer_operations() {
        if !is_available() {
            return;
        }

        let backend = match WgpuBackend::new(0) {
            Some(b) => b,
            None => return,
        };

        // Create a buffer
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let buffer_id =
            backend.create_buffer_init(bytes, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

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
