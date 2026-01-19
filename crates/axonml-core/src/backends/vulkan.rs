//! Vulkan Backend - Cross-Platform GPU Operations
//!
//! Provides the Vulkan implementation for tensor operations on GPUs.
//! This backend requires the `vulkan` feature and Vulkan SDK.
//!
//! # Key Features
//! - Cross-platform GPU support (Windows, Linux, macOS via MoltenVK)
//! - Compute shader based operations
//! - Async execution with command queues
//! - Multi-GPU support
//! - Memory management via gpu-allocator
//!
//! # Requirements
//! - Vulkan 1.1+ capable GPU
//! - Vulkan SDK/runtime
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use super::Backend;
use crate::device::DeviceCapabilities;

#[cfg(feature = "vulkan")]
use std::collections::HashMap;
#[cfg(feature = "vulkan")]
use std::ffi::CStr;
#[cfg(feature = "vulkan")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "vulkan")]
use ash::{ext::debug_utils, vk, Device, Entry, Instance};

#[cfg(feature = "vulkan")]
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
    MemoryLocation,
};

// =============================================================================
// Global State for Vulkan
// =============================================================================

#[cfg(feature = "vulkan")]
struct VulkanGlobalState {
    entry: Entry,
    instance: Instance,
    physical_devices: Vec<vk::PhysicalDevice>,
    device_properties: Vec<vk::PhysicalDeviceProperties>,
    device_memory_properties: Vec<vk::PhysicalDeviceMemoryProperties>,
}

#[cfg(feature = "vulkan")]
unsafe impl Send for VulkanGlobalState {}
#[cfg(feature = "vulkan")]
unsafe impl Sync for VulkanGlobalState {}

#[cfg(feature = "vulkan")]
static VULKAN_STATE: OnceLock<Option<VulkanGlobalState>> = OnceLock::new();

#[cfg(feature = "vulkan")]
fn get_vulkan_state() -> Option<&'static VulkanGlobalState> {
    VULKAN_STATE
        .get_or_init(|| unsafe { init_vulkan().ok() })
        .as_ref()
}

#[cfg(feature = "vulkan")]
unsafe fn init_vulkan() -> Result<VulkanGlobalState, vk::Result> {
    let entry = Entry::linked();

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Axonml")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"Axonml Engine")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_1);

    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = entry.create_instance(&create_info, None)?;

    let physical_devices = instance.enumerate_physical_devices()?;

    let device_properties: Vec<_> = physical_devices
        .iter()
        .map(|&pd| instance.get_physical_device_properties(pd))
        .collect();

    let device_memory_properties: Vec<_> = physical_devices
        .iter()
        .map(|&pd| instance.get_physical_device_memory_properties(pd))
        .collect();

    Ok(VulkanGlobalState {
        entry,
        instance,
        physical_devices,
        device_properties,
        device_memory_properties,
    })
}

// =============================================================================
// Buffer Tracking for Vulkan
// =============================================================================

#[cfg(feature = "vulkan")]
struct BufferInfo {
    buffer: vk::Buffer,
    allocation: Allocation,
    size: u64,
}

#[cfg(feature = "vulkan")]
struct VulkanBufferTracker {
    buffers: HashMap<u64, BufferInfo>,
    next_id: u64,
}

#[cfg(feature = "vulkan")]
impl VulkanBufferTracker {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_id: 1,
        }
    }

    fn insert(&mut self, buffer: vk::Buffer, allocation: Allocation, size: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.buffers.insert(
            id,
            BufferInfo {
                buffer,
                allocation,
                size,
            },
        );
        id
    }

    fn remove(&mut self, id: u64) -> Option<BufferInfo> {
        self.buffers.remove(&id)
    }

    fn get(&self, id: u64) -> Option<&BufferInfo> {
        self.buffers.get(&id)
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut BufferInfo> {
        self.buffers.get_mut(&id)
    }
}

// =============================================================================
// Vulkan Backend Struct
// =============================================================================

/// Vulkan backend for tensor operations on GPUs.
#[cfg(feature = "vulkan")]
pub struct VulkanBackend {
    device_index: usize,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    allocator: Arc<Mutex<Allocator>>,
    buffer_tracker: Arc<Mutex<VulkanBufferTracker>>,
    compute_pipelines: Arc<Mutex<HashMap<String, vk::Pipeline>>>,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
}

#[cfg(not(feature = "vulkan"))]
#[derive(Debug)]
pub struct VulkanBackend {
    device_index: usize,
}

#[cfg(feature = "vulkan")]
impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("device_index", &self.device_index)
            .finish()
    }
}

impl VulkanBackend {
    /// Creates a new Vulkan backend for the specified device.
    #[cfg(feature = "vulkan")]
    pub fn new(device_index: usize) -> Option<Self> {
        let state = get_vulkan_state()?;

        if device_index >= state.physical_devices.len() {
            return None;
        }

        let physical_device = state.physical_devices[device_index];

        unsafe {
            // Find compute queue family
            let queue_families = state
                .instance
                .get_physical_device_queue_family_properties(physical_device);
            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(idx, _)| idx as u32)?;

            // Create logical device
            let queue_priorities = [1.0f32];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info));

            let device = state
                .instance
                .create_device(physical_device, &device_create_info, None)
                .ok()?;
            let queue = device.get_device_queue(queue_family_index, 0);

            // Create command pool
            let pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let command_pool = device.create_command_pool(&pool_create_info, None).ok()?;

            // Create allocator
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: state.instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .ok()?;

            // Create descriptor set layout for compute shaders
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];

            let layout_create_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            let descriptor_set_layout = device
                .create_descriptor_set_layout(&layout_create_info, None)
                .ok()?;

            // Create pipeline layout
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .ok()?;

            // Create descriptor pool
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1000,
            }];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(100);

            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .ok()?;

            Some(Self {
                device_index,
                physical_device,
                device,
                queue,
                queue_family_index,
                command_pool,
                allocator: Arc::new(Mutex::new(allocator)),
                buffer_tracker: Arc::new(Mutex::new(VulkanBufferTracker::new())),
                compute_pipelines: Arc::new(Mutex::new(HashMap::new())),
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
            })
        }
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn new(_device_index: usize) -> Option<Self> {
        None
    }

    /// Returns the device index.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Creates a GPU buffer with the specified size.
    #[cfg(feature = "vulkan")]
    pub fn create_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Option<u64> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = self.device.create_buffer(&buffer_info, None).ok()?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let allocation = {
                let mut allocator = self.allocator.lock().unwrap();
                allocator
                    .allocate(&AllocationCreateDesc {
                        name: "buffer",
                        requirements,
                        location: MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .ok()?
            };

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .ok()?;

            let mut tracker = self.buffer_tracker.lock().unwrap();
            Some(tracker.insert(buffer, allocation, size))
        }
    }

    /// Creates a GPU buffer initialized with data.
    #[cfg(feature = "vulkan")]
    pub fn create_buffer_init(&self, data: &[u8]) -> Option<u64> {
        let buffer_id = self.create_buffer(
            data.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        self.write_buffer(buffer_id, 0, data);
        Some(buffer_id)
    }

    /// Writes data to an existing buffer.
    #[cfg(feature = "vulkan")]
    pub fn write_buffer(&self, buffer_id: u64, offset: u64, data: &[u8]) {
        let tracker = self.buffer_tracker.lock().unwrap();
        if let Some(info) = tracker.get(buffer_id) {
            if let Some(mapped) = info.allocation.mapped_ptr() {
                unsafe {
                    let dst = (mapped.as_ptr() as *mut u8).add(offset as usize);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                }
            }
        }
    }

    /// Reads data from a buffer.
    #[cfg(feature = "vulkan")]
    pub fn read_buffer(&self, buffer_id: u64) -> Option<Vec<u8>> {
        let tracker = self.buffer_tracker.lock().unwrap();
        let info = tracker.get(buffer_id)?;

        let mapped = info.allocation.mapped_ptr()?;
        let mut data = vec![0u8; info.size as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr() as *const u8,
                data.as_mut_ptr(),
                info.size as usize,
            );
        }
        Some(data)
    }

    /// Destroys a buffer and frees its memory.
    #[cfg(feature = "vulkan")]
    pub fn destroy_buffer(&self, buffer_id: u64) {
        let mut tracker = self.buffer_tracker.lock().unwrap();
        if let Some(info) = tracker.remove(buffer_id) {
            unsafe {
                self.device.destroy_buffer(info.buffer, None);
            }
            let mut allocator = self.allocator.lock().unwrap();
            let _ = allocator.free(info.allocation);
        }
    }

    /// Executes a command buffer.
    #[cfg(feature = "vulkan")]
    pub fn execute_commands<F>(&self, f: F)
    where
        F: FnOnce(vk::CommandBuffer),
    {
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffers = self.device.allocate_command_buffers(&alloc_info).unwrap();
            let cmd = command_buffers[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device.begin_command_buffer(cmd, &begin_info).unwrap();
            f(cmd);
            self.device.end_command_buffer(cmd).unwrap();

            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())
                .unwrap();
            self.device.queue_wait_idle(self.queue).unwrap();

            self.device
                .free_command_buffers(self.command_pool, &command_buffers);
        }
    }

    /// Copies data between buffers.
    #[cfg(feature = "vulkan")]
    pub fn copy_buffer(&self, src_id: u64, dst_id: u64, size: u64) {
        let tracker = self.buffer_tracker.lock().unwrap();
        let src_info = match tracker.get(src_id) {
            Some(info) => info,
            None => return,
        };
        let dst_info = match tracker.get(dst_id) {
            Some(info) => info,
            None => return,
        };

        let src_buffer = src_info.buffer;
        let dst_buffer = dst_info.buffer;
        drop(tracker);

        self.execute_commands(|cmd| {
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            unsafe {
                self.device
                    .cmd_copy_buffer(cmd, src_buffer, dst_buffer, &[copy_region]);
            }
        });
    }

    /// Creates a compute pipeline from SPIR-V bytecode.
    #[cfg(feature = "vulkan")]
    pub fn create_compute_pipeline(&self, name: &str, spirv: &[u32]) -> Option<vk::Pipeline> {
        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv);

            let shader_module = self.device.create_shader_module(&shader_info, None).ok()?;

            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(self.pipeline_layout);

            let pipelines = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .ok()?;

            self.device.destroy_shader_module(shader_module, None);

            let pipeline = pipelines[0];
            self.compute_pipelines
                .lock()
                .unwrap()
                .insert(name.to_string(), pipeline);

            Some(pipeline)
        }
    }

    /// Dispatches a compute shader.
    #[cfg(feature = "vulkan")]
    pub fn dispatch_compute(
        &self,
        pipeline: vk::Pipeline,
        buffers: &[u64],
        group_count: (u32, u32, u32),
    ) {
        unsafe {
            // Allocate descriptor set
            let layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);

            let descriptor_sets = match self.device.allocate_descriptor_sets(&alloc_info) {
                Ok(sets) => sets,
                Err(_) => return,
            };

            let descriptor_set = descriptor_sets[0];

            // Update descriptor sets with buffer bindings
            let tracker = self.buffer_tracker.lock().unwrap();
            let mut buffer_infos = Vec::new();
            let mut writes = Vec::new();

            for (i, buffer_id) in buffers.iter().enumerate() {
                if let Some(info) = tracker.get(*buffer_id) {
                    buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: info.buffer,
                        offset: 0,
                        range: info.size,
                    });
                }
            }

            for (i, buffer_info) in buffer_infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(buffer_info)),
                );
            }

            self.device.update_descriptor_sets(&writes, &[]);
            drop(tracker);

            // Execute compute
            self.execute_commands(|cmd| {
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                self.device
                    .cmd_dispatch(cmd, group_count.0, group_count.1, group_count.2);
            });
        }
    }
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Clean up buffers
            let mut tracker = self.buffer_tracker.lock().unwrap();
            let buffer_ids: Vec<u64> = tracker.buffers.keys().cloned().collect();
            for id in buffer_ids {
                if let Some(info) = tracker.remove(id) {
                    self.device.destroy_buffer(info.buffer, None);
                    let mut allocator = self.allocator.lock().unwrap();
                    let _ = allocator.free(info.allocation);
                }
            }
            drop(tracker);

            // Clean up pipelines
            let pipelines = self.compute_pipelines.lock().unwrap();
            for (_, pipeline) in pipelines.iter() {
                self.device.destroy_pipeline(*pipeline, None);
            }
            drop(pipelines);

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

// =============================================================================
// Backend Trait Implementation
// =============================================================================

#[cfg(feature = "vulkan")]
impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn is_available(&self) -> bool {
        true // If we created successfully, we're available
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let state = get_vulkan_state().unwrap();
        let props = &state.device_properties[self.device_index];
        let mem_props = &state.device_memory_properties[self.device_index];

        let total_memory: usize = (0..mem_props.memory_heap_count as usize)
            .map(|i| mem_props.memory_heaps[i].size as usize)
            .sum();

        let device_name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };

        DeviceCapabilities {
            name: device_name,
            total_memory,
            available_memory: 0, // Vulkan doesn't provide this directly
            supports_f16: true,  // Most modern GPUs support f16
            supports_f64: props.limits.shader_float64 != 0,
            max_threads_per_block: props.limits.max_compute_work_group_invocations as usize,
            compute_capability: None,
        }
    }

    fn allocate(&self, size: usize) -> *mut u8 {
        match self.create_buffer(
            size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        ) {
            Some(buffer_id) => buffer_id as *mut u8,
            None => std::ptr::null_mut(),
        }
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
        self.copy_buffer(src_id, dst_id, size as u64);
    }

    fn synchronize(&self) {
        unsafe {
            self.device.device_wait_idle().ok();
        }
    }
}

#[cfg(not(feature = "vulkan"))]
impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            name: "Vulkan (not available)".to_string(),
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
// Vulkan Runtime Functions
// =============================================================================

/// Returns whether Vulkan is available on this system.
#[cfg(feature = "vulkan")]
pub fn is_available() -> bool {
    get_vulkan_state()
        .map(|s| !s.physical_devices.is_empty())
        .unwrap_or(false)
}

#[cfg(not(feature = "vulkan"))]
pub fn is_available() -> bool {
    false
}

/// Returns the number of available Vulkan devices.
#[cfg(feature = "vulkan")]
pub fn device_count() -> usize {
    get_vulkan_state()
        .map(|s| s.physical_devices.len())
        .unwrap_or(0)
}

#[cfg(not(feature = "vulkan"))]
pub fn device_count() -> usize {
    0
}

/// Returns whether a specific Vulkan device is available.
pub fn is_device_available(index: usize) -> bool {
    index < device_count()
}

/// Returns the capabilities of a Vulkan device.
#[cfg(feature = "vulkan")]
pub fn get_capabilities(index: usize) -> DeviceCapabilities {
    let state = match get_vulkan_state() {
        Some(s) => s,
        None => {
            return DeviceCapabilities {
                name: "Unknown".to_string(),
                total_memory: 0,
                available_memory: 0,
                supports_f16: false,
                supports_f64: false,
                max_threads_per_block: 0,
                compute_capability: None,
            }
        }
    };

    if index >= state.physical_devices.len() {
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

    let props = &state.device_properties[index];
    let mem_props = &state.device_memory_properties[index];

    let total_memory: usize = (0..mem_props.memory_heap_count as usize)
        .map(|i| mem_props.memory_heaps[i].size as usize)
        .sum();

    let device_name = unsafe {
        CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .to_string()
    };

    DeviceCapabilities {
        name: device_name,
        total_memory,
        available_memory: 0,
        supports_f16: true,
        supports_f64: props.limits.shader_float64 != 0,
        max_threads_per_block: props.limits.max_compute_work_group_invocations as usize,
        compute_capability: None,
    }
}

#[cfg(not(feature = "vulkan"))]
pub fn get_capabilities(index: usize) -> DeviceCapabilities {
    DeviceCapabilities {
        name: format!("Vulkan Device {} (not available)", index),
        total_memory: 0,
        available_memory: 0,
        supports_f16: false,
        supports_f64: false,
        max_threads_per_block: 0,
        compute_capability: None,
    }
}

// =============================================================================
// SPIR-V Shader Templates (Pre-compiled bytecode would go here)
// =============================================================================

/// Note: SPIR-V shaders need to be pre-compiled from GLSL.
/// Use glslangValidator or shaderc to compile GLSL to SPIR-V.
///
/// Example GLSL compute shader for addition:
/// ```glsl
/// #version 450
/// layout(local_size_x = 256) in;
///
/// layout(set = 0, binding = 0) readonly buffer A { float a[]; };
/// layout(set = 0, binding = 1) readonly buffer B { float b[]; };
/// layout(set = 0, binding = 2) buffer Result { float result[]; };
///
/// void main() {
///     uint index = gl_GlobalInvocationID.x;
///     result[index] = a[index] + b[index];
/// }
/// ```

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_availability() {
        let available = is_available();
        println!("Vulkan available: {}", available);
    }

    #[test]
    fn test_device_count() {
        let count = device_count();
        println!("Vulkan device count: {}", count);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn test_backend_creation() {
        if is_available() {
            let backend = VulkanBackend::new(0);
            assert!(backend.is_some());
            if let Some(b) = backend {
                assert!(b.is_available());
                println!("Backend name: {}", b.name());
                println!("Capabilities: {:?}", b.capabilities());
            }
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn test_buffer_operations() {
        if !is_available() {
            return;
        }

        let backend = match VulkanBackend::new(0) {
            Some(b) => b,
            None => return,
        };

        // Create a buffer with data
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let buffer_id = match backend.create_buffer_init(bytes) {
            Some(id) => id,
            None => return,
        };

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
