//! System API endpoints for AxonML
//!
//! Provides system information including GPU detection, memory, and benchmarking.

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use wgpu::{Backends, DeviceType, Instance, InstanceDescriptor, MemoryHints};

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};

// ============================================================================
// Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub id: usize,
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub memory_total: u64,
    pub is_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub platform: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub axonml_version: String,
    pub rust_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuListResponse {
    pub gpus: Vec<GpuInfo>,
    pub cuda_available: bool,
    pub total_gpu_memory: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub gpu_id: usize,
    pub gpu_name: String,
    pub buffer_copy_1mb_ms: f64,
    pub buffer_copy_16mb_ms: f64,
    pub buffer_copy_64mb_ms: f64,
    pub compute_dispatch_ms: f64,
    pub effective_bandwidth_1mb: String,
    pub effective_bandwidth_16mb: String,
    pub effective_bandwidth_64mb: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    pub results: Vec<BenchmarkResult>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMetrics {
    pub timestamp: String,
    pub cpu_usage_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub memory_percent: f64,
    pub disk_used_bytes: u64,
    pub disk_total_bytes: u64,
    pub disk_percent: f64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub process_count: usize,
    pub uptime_seconds: u64,
    pub load_avg_1m: f64,
    pub load_avg_5m: f64,
    pub load_avg_15m: f64,
    pub cpu_per_core: Vec<f64>,
    pub gpu_metrics: Vec<GpuMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub id: usize,
    pub name: String,
    pub utilization_percent: f64,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub temperature_c: f64,
    pub power_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsHistory {
    pub timestamps: Vec<String>,
    pub cpu_history: Vec<f64>,
    pub memory_history: Vec<f64>,
    pub disk_io_read: Vec<f64>,
    pub disk_io_write: Vec<f64>,
    pub network_rx: Vec<f64>,
    pub network_tx: Vec<f64>,
    pub gpu_utilization: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationData {
    pub points: Vec<CorrelationPoint>,
    pub x_label: String,
    pub y_label: String,
    pub z_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub label: String,
    pub category: String,
}

// ============================================================================
// Handlers
// ============================================================================

/// Get system information
pub async fn get_system_info(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<SystemInfo>, AuthError> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let info = SystemInfo {
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_count: sys.cpus().len(),
        total_memory_bytes: sys.total_memory(),
        available_memory_bytes: sys.available_memory(),
        axonml_version: env!("CARGO_PKG_VERSION").to_string(),
        rust_version: rustc_version_runtime::version().to_string(),
    };

    Ok(Json(info))
}

/// List available GPUs
pub async fn list_gpus(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<GpuListResponse>, AuthError> {
    let gpus = detect_gpus();
    let cuda_available = !gpus.is_empty();
    let total_gpu_memory: u64 = gpus.iter().map(|g| g.memory_total).sum();

    Ok(Json(GpuListResponse {
        gpus,
        cuda_available,
        total_gpu_memory,
    }))
}

/// Run GPU benchmark
pub async fn run_benchmark(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<BenchmarkResponse>, AuthError> {
    let gpus = detect_gpus();
    let mut results = Vec::new();

    for gpu in &gpus {
        match run_gpu_benchmark(gpu.id) {
            Ok(result) => results.push(result),
            Err(e) => {
                tracing::warn!(gpu_id = gpu.id, error = %e, "Failed to benchmark GPU");
            }
        }
    }

    Ok(Json(BenchmarkResponse {
        results,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Get real-time system metrics
pub async fn get_realtime_metrics(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<RealtimeMetrics>, AuthError> {
    use sysinfo::{Disks, Networks, System};

    let mut sys = System::new_all();
    sys.refresh_all();

    // CPU usage per core
    let cpu_per_core: Vec<f64> = sys
        .cpus()
        .iter()
        .map(|cpu| cpu.cpu_usage() as f64)
        .collect();
    let cpu_usage = if cpu_per_core.is_empty() {
        0.0
    } else {
        cpu_per_core.iter().sum::<f64>() / cpu_per_core.len() as f64
    };

    // Memory
    let memory_used = sys.used_memory();
    let memory_total = sys.total_memory();
    let memory_percent = if memory_total > 0 {
        (memory_used as f64 / memory_total as f64) * 100.0
    } else {
        0.0
    };

    // Disk
    let disks = Disks::new_with_refreshed_list();
    let (disk_used, disk_total) = disks.iter().fold((0u64, 0u64), |(used, total), disk| {
        (
            used + disk.total_space() - disk.available_space(),
            total + disk.total_space(),
        )
    });
    let disk_percent = if disk_total > 0 {
        (disk_used as f64 / disk_total as f64) * 100.0
    } else {
        0.0
    };

    // Network
    let networks = Networks::new_with_refreshed_list();
    let (rx, tx) = networks.iter().fold((0u64, 0u64), |(rx, tx), (_, data)| {
        (rx + data.received(), tx + data.transmitted())
    });

    // Process count
    let process_count = sys.processes().len();

    // Load average (Linux/macOS)
    let load_avg = System::load_average();

    // GPU metrics via nvidia-smi
    let gpu_metrics = get_gpu_metrics();

    // Uptime
    let uptime = System::uptime();

    Ok(Json(RealtimeMetrics {
        timestamp: chrono::Utc::now().to_rfc3339(),
        cpu_usage_percent: cpu_usage,
        memory_used_bytes: memory_used,
        memory_total_bytes: memory_total,
        memory_percent,
        disk_used_bytes: disk_used,
        disk_total_bytes: disk_total,
        disk_percent,
        network_rx_bytes: rx,
        network_tx_bytes: tx,
        process_count,
        uptime_seconds: uptime,
        load_avg_1m: load_avg.one,
        load_avg_5m: load_avg.five,
        load_avg_15m: load_avg.fifteen,
        cpu_per_core,
        gpu_metrics,
    }))
}

/// Get metrics history for charts
pub async fn get_metrics_history(
    State(state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<SystemMetricsHistory>, AuthError> {
    // Get from in-memory history collected by background task
    let history = state.metrics_history.lock().await;
    Ok(Json(history.clone()))
}

/// Get correlation data for 3D scatter plot
pub async fn get_correlation_data(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<CorrelationData>, AuthError> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    // Generate correlation data from processes
    let mut points = Vec::new();

    for (_pid, process) in sys.processes().iter().take(100) {
        let cpu = process.cpu_usage() as f64;
        let memory = process.memory() as f64 / (1024.0 * 1024.0); // MB
        let runtime = process.run_time() as f64;

        if cpu > 0.0 || memory > 10.0 {
            let category = if cpu > 50.0 {
                "high-cpu"
            } else if memory > 500.0 {
                "high-memory"
            } else {
                "normal"
            };

            points.push(CorrelationPoint {
                x: cpu.min(100.0),
                y: memory.min(2000.0),
                z: (runtime / 60.0).min(1000.0), // minutes, capped
                label: process.name().to_string_lossy().to_string(),
                category: category.to_string(),
            });
        }
    }

    // If no interesting processes, add some demo data
    if points.len() < 10 {
        for i in 0..50 {
            let angle = i as f64 * 0.3;
            points.push(CorrelationPoint {
                x: 20.0 + 30.0 * angle.cos() + (i as f64 * 1.5),
                y: 100.0 + 200.0 * angle.sin() + (i as f64 * 5.0),
                z: i as f64 * 2.0,
                label: format!("Process {}", i),
                category: if i % 3 == 0 {
                    "high-cpu"
                } else if i % 3 == 1 {
                    "high-memory"
                } else {
                    "normal"
                }
                .to_string(),
            });
        }
    }

    Ok(Json(CorrelationData {
        points,
        x_label: "CPU Usage (%)".to_string(),
        y_label: "Memory (MB)".to_string(),
        z_label: "Runtime (min)".to_string(),
    }))
}

fn get_gpu_metrics() -> Vec<GpuMetrics> {
    use std::process::Command;

    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ])
        .output()
    {
        Ok(output) => output,
        Err(_) => return Vec::new(),
    };

    if !output.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut metrics = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(", ").collect();
        if parts.len() >= 7 {
            metrics.push(GpuMetrics {
                id: parts[0].trim().parse().unwrap_or(0),
                name: parts[1].trim().to_string(),
                utilization_percent: parts[2].trim().parse().unwrap_or(0.0),
                memory_used_mb: parts[3].trim().parse().unwrap_or(0),
                memory_total_mb: parts[4].trim().parse().unwrap_or(0),
                temperature_c: parts[5].trim().parse().unwrap_or(0.0),
                power_watts: parts[6].trim().parse().unwrap_or(0.0),
            });
        }
    }

    metrics
}

// ============================================================================
// GPU Detection
// ============================================================================

fn detect_gpus() -> Vec<GpuInfo> {
    // First try wgpu detection
    let mut gpus = detect_gpus_wgpu();

    // If no GPUs found via wgpu, try nvidia-smi (for WSL2/CUDA)
    if gpus.is_empty() {
        tracing::info!("No GPUs via wgpu, trying nvidia-smi fallback...");
        gpus = detect_gpus_nvidia_smi();
    }

    if gpus.is_empty() {
        tracing::warn!("No GPUs detected! Check that GPU drivers are installed and accessible.");
    } else {
        tracing::info!("Detected {} GPU(s)", gpus.len());
    }

    gpus
}

fn detect_gpus_wgpu() -> Vec<GpuInfo> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(Backends::all());
    let mut gpus = Vec::new();
    let mut gpu_id = 0;

    tracing::info!("Enumerating GPU adapters via wgpu...");

    for adapter in &adapters {
        let info = adapter.get_info();

        tracing::info!(
            "Found adapter: name={}, vendor={:#x}, device_type={:?}, backend={:?}",
            info.name,
            info.vendor,
            info.device_type,
            info.backend
        );

        // Skip CPU-based software renderers
        if info.device_type == DeviceType::Cpu {
            tracing::info!("  -> Skipping CPU adapter");
            continue;
        }

        let device_type = match info.device_type {
            DeviceType::DiscreteGpu => "Discrete",
            DeviceType::IntegratedGpu => "Integrated",
            DeviceType::VirtualGpu => "Virtual",
            DeviceType::Cpu => "CPU",
            DeviceType::Other => "Other",
        };

        // Determine vendor name from vendor ID
        let vendor_name = match info.vendor {
            0x10DE => "NVIDIA",
            0x1002 | 0x1022 => "AMD",
            0x8086 => "Intel",
            0x1414 => "Microsoft",
            _ => "Unknown",
        };

        let backend = match info.backend {
            wgpu::Backend::Vulkan => {
                if vendor_name == "NVIDIA" {
                    "Vulkan/CUDA"
                } else {
                    "Vulkan"
                }
            }
            wgpu::Backend::Dx12 => {
                if vendor_name == "NVIDIA" {
                    "DX12/CUDA"
                } else {
                    "DX12"
                }
            }
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            wgpu::Backend::Empty => "None",
        };

        let limits = adapter.limits();

        tracing::info!(
            "  -> Adding GPU {}: {} ({}, {}, max_buffer={})",
            gpu_id,
            info.name,
            vendor_name,
            backend,
            limits.max_buffer_size
        );

        gpus.push(GpuInfo {
            id: gpu_id,
            name: info.name.clone(),
            vendor: vendor_name.to_string(),
            device_type: device_type.to_string(),
            backend: backend.to_string(),
            driver: info.driver.clone(),
            memory_total: limits.max_buffer_size,
            is_available: true,
        });

        gpu_id += 1;
    }

    gpus
}

/// Detect NVIDIA GPUs via nvidia-smi (useful for WSL2/CUDA environments)
fn detect_gpus_nvidia_smi() -> Vec<GpuInfo> {
    use std::process::Command;

    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            tracing::debug!("nvidia-smi not available: {}", e);
            return Vec::new();
        }
    };

    if !output.status.success() {
        tracing::debug!("nvidia-smi failed");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(", ").collect();
        if parts.len() >= 4 {
            let id: usize = parts[0].trim().parse().unwrap_or(0);
            let name = parts[1].trim().to_string();
            let memory_mb: u64 = parts[2].trim().parse().unwrap_or(0);
            let driver = parts[3].trim().to_string();

            tracing::info!(
                "nvidia-smi found GPU {}: {} ({} MB, driver {})",
                id,
                name,
                memory_mb,
                driver
            );

            gpus.push(GpuInfo {
                id,
                name,
                vendor: "NVIDIA".to_string(),
                device_type: "Discrete".to_string(),
                backend: "CUDA".to_string(),
                driver,
                memory_total: memory_mb * 1024 * 1024, // Convert MB to bytes
                is_available: true,
            });
        }
    }

    gpus
}

// ============================================================================
// GPU Benchmarking
// ============================================================================

fn run_gpu_benchmark(gpu_id: usize) -> Result<BenchmarkResult, String> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    // Filter to only real GPUs (not CPU software renderers)
    let adapters: Vec<_> = instance
        .enumerate_adapters(Backends::all())
        .into_iter()
        .filter(|a| {
            let info = a.get_info();
            info.device_type != DeviceType::Cpu
        })
        .collect();

    if gpu_id >= adapters.len() {
        return Err(format!("GPU {} not found", gpu_id));
    }

    let adapter = &adapters[gpu_id];
    let gpu_name = adapter.get_info().name.clone();

    let (device, queue) = pollster::block_on(async {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Benchmark Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
    })
    .map_err(|e| format!("Failed to create device: {}", e))?;

    const ITERATIONS: usize = 10;

    let buffer_copy_1mb_ms = benchmark_buffer_copy(&device, &queue, 1024 * 1024, ITERATIONS);
    let buffer_copy_16mb_ms = benchmark_buffer_copy(&device, &queue, 16 * 1024 * 1024, ITERATIONS);
    let buffer_copy_64mb_ms = benchmark_buffer_copy(&device, &queue, 64 * 1024 * 1024, ITERATIONS);
    let compute_dispatch_ms = benchmark_compute_dispatch(&device, &queue, ITERATIONS);

    Ok(BenchmarkResult {
        gpu_id,
        gpu_name,
        buffer_copy_1mb_ms,
        buffer_copy_16mb_ms,
        buffer_copy_64mb_ms,
        compute_dispatch_ms,
        effective_bandwidth_1mb: format_bandwidth(1024.0 * 1024.0, buffer_copy_1mb_ms),
        effective_bandwidth_16mb: format_bandwidth(16.0 * 1024.0 * 1024.0, buffer_copy_16mb_ms),
        effective_bandwidth_64mb: format_bandwidth(64.0 * 1024.0 * 1024.0, buffer_copy_64mb_ms),
    })
}

fn benchmark_buffer_copy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u64,
    iterations: usize,
) -> f64 {
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let src_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Source Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Destination Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    queue.write_buffer(&src_buffer, 0, &data);

    // Warmup
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_compute_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    iterations: usize,
) -> f64 {
    let shader_source = r"
        @group(0) @binding(0) var<storage, read> input_a: array<f32>;
        @group(0) @binding(1) var<storage, read> input_b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx < arrayLength(&output)) {
                output[idx] = input_a[idx] + input_b[idx];
            }
        }
    ";

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let buffer_size: u64 = 1024 * 1024 * 4;
    let num_elements = buffer_size / 4;

    let input_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Input A"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let input_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Input B"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let data_a: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..num_elements).map(|i| (i * 2) as f32).collect();
    queue.write_buffer(&input_a, 0, bytemuck::cast_slice(&data_a));
    queue.write_buffer(&input_b, 0, bytemuck::cast_slice(&data_b));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let workgroup_count = num_elements.div_ceil(256) as u32;

    // Warmup
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn format_bandwidth(bytes: f64, time_ms: f64) -> String {
    let bytes_per_sec = bytes / (time_ms / 1000.0);
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;

    if bytes_per_sec >= GB {
        format!("{:.2} GB/s", bytes_per_sec / GB)
    } else {
        format!("{:.2} MB/s", bytes_per_sec / MB)
    }
}
